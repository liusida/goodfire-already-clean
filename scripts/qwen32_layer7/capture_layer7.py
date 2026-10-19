from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from gb10_load_llm import load_model_to_cuda
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "Qwen/Qwen3-32B"
MODEL_REVISION = "9216db5781bf21249d130ec9da846c4624c16137"
LAYER_INDEX = 7
TARGET_NEURON = 2026
PROMPT = """Human: A 32-year-old female at 30 weeks gestation presents with severe right upper quadrant pain, mild headache, and nausea. BP is 162/98 mmHg, and labs show mildly elevated liver enzymes.

If we can only ask about one other symptom, we should ask whether she's experiencing...

Assistant: We should ask whether she's experiencing **"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture every Layer 7 MLP-neuron activation for the blog prompt."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/layer7/layer7_mlp_activations.pt"),
        help="Destination for the captured tensor and prompt metadata.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run Qwen3-32B")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    model, touched = load_model_to_cuda(
        AutoModelForCausalLM,
        MODEL_ID,
        revision=MODEL_REVISION,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        touch="auto",
        return_touched=True,
    )
    model.eval()

    encoded = tokenizer(PROMPT, add_special_tokens=False, return_tensors="pt")
    input_ids = encoded.input_ids.to(model.device)
    attention_mask = encoded.attention_mask.to(model.device)
    captured: list[torch.Tensor] = []

    def capture_down_proj_input(
        _module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]
    ) -> None:
        # Qwen3MLP computes down_proj(silu(gate_proj(x)) * up_proj(x)).
        # Therefore down_proj's input is the native post-SwiGLU neuron vector.
        captured.append(inputs[0].detach().to(device="cpu", dtype=torch.bfloat16))

    down_proj = model.model.layers[LAYER_INDEX].mlp.down_proj
    handle = down_proj.register_forward_pre_hook(capture_down_proj_input)
    try:
        with torch.inference_mode():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
    finally:
        handle.remove()

    if len(captured) != 1:
        raise RuntimeError(f"Expected one Layer 7 capture, got {len(captured)}")
    activations = captured[0]
    expected_shape = (1, input_ids.shape[1], model.config.intermediate_size)
    if tuple(activations.shape) != expected_shape:
        raise RuntimeError(
            f"Unexpected activation shape {tuple(activations.shape)}; "
            f"expected {expected_shape}"
        )

    ids_cpu = input_ids[0].detach().cpu()
    token_ids = ids_cpu.tolist()
    token_text = [tokenizer.decode([token_id]) for token_id in token_ids]
    target_trace = activations[0, :, TARGET_NEURON].float()
    peak_position = int(target_trace.abs().argmax().item())
    next_token_id = int(outputs.logits[0, -1].argmax().item())

    artifact = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "prompt": PROMPT,
        "layer_index": LAYER_INDEX,
        "target_neuron": TARGET_NEURON,
        "input_ids": ids_cpu,
        "tokens": token_text,
        "activations": activations,
        "target_trace": target_trace,
        "next_token_id": next_token_id,
        "next_token": tokenizer.decode([next_token_id]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, args.output)

    metadata_path = args.output.with_suffix(".json")
    metadata = {
        key: value
        for key, value in artifact.items()
        if key not in {"input_ids", "activations", "target_trace"}
    }
    metadata["input_ids"] = token_ids
    metadata["target_trace"] = target_trace.tolist()
    metadata["activation_shape"] = list(activations.shape)
    metadata["activation_dtype"] = str(activations.dtype)
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"Touched CPU pages: {touched / 1024**3:.2f} GiB")
    print(f"Prompt tokens: {input_ids.shape[1]}")
    print(f"Layer {LAYER_INDEX} activation shape: {tuple(activations.shape)}")
    print(
        f"N{TARGET_NEURON} largest |activation|: position {peak_position}, "
        f"token {token_text[peak_position]!r}, value {target_trace[peak_position]:.4f}"
    )
    print(f"Greedy next token: {artifact['next_token']!r}")
    print(f"Saved tensor artifact: {args.output.resolve()}")
    print(f"Saved readable metadata: {metadata_path.resolve()}")


if __name__ == "__main__":
    main()
