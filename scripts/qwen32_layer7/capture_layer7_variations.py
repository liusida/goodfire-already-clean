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

PROMPTS = [
    {
        "id": "original_preeclampsia",
        "variation": "original reference",
        "prompt": """Human: A 32-year-old female at 30 weeks gestation presents with severe right upper quadrant pain, mild headache, and nausea. BP is 162/98 mmHg, and labs show mildly elevated liver enzymes.

If we can only ask about one other symptom, we should ask whether she's experiencing...

Assistant: We should ask whether she's experiencing **""",
    },
    {
        "id": "changed_symptoms",
        "variation": "same pregnancy framing, unrelated respiratory symptoms",
        "prompt": """Human: A 32-year-old female at 30 weeks gestation presents with a persistent cough, sore throat, and fever. BP is 118/76 mmHg, and routine labs are normal.

If we can only ask about one other symptom, we should ask whether she's experiencing...

Assistant: We should ask whether she's experiencing **""",
    },
    {
        "id": "changed_question",
        "variation": "same clinical facts, asks for a laboratory finding",
        "prompt": """Human: A 32-year-old female at 30 weeks gestation presents with severe right upper quadrant pain, mild headache, and nausea. BP is 162/98 mmHg, and labs show mildly elevated liver enzymes.

If we can only order one other laboratory test, we should check whether she has...

Assistant: We should check whether she has **""",
    },
    {
        "id": "different_medical_topic",
        "variation": "asthma-like presentation with the original question structure",
        "prompt": """Human: A 45-year-old male presents with wheezing, chest tightness, and shortness of breath after visiting a home with cats. His oxygen saturation is 93%, and his temperature is normal.

If we can only ask about one other symptom, we should ask whether he's experiencing...

Assistant: We should ask whether he's experiencing **""",
    },
    {
        "id": "software_debugging",
        "variation": "non-medical software diagnosis",
        "prompt": """Human: A Python web service has high CPU usage, slow requests, and frequent database timeout errors. Memory usage is stable, and the network health checks pass.

If we can only inspect one other signal, we should check whether the service is experiencing...

Assistant: We should check whether the service is experiencing **""",
    },
    {
        "id": "weather_topic",
        "variation": "non-medical weather diagnosis",
        "prompt": """Human: A coastal town is reporting rapidly falling air pressure, strong onshore winds, and increasingly rough seas. Temperatures remain warm, and heavy clouds are approaching.

If we can only ask about one other condition, we should ask whether the area is experiencing...

Assistant: We should ask whether the area is experiencing **""",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture Layer 7 MLP neurons for the original prompt and five variations."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/layer7/layer7_prompt_variations.pt"),
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

    hook_outputs: list[torch.Tensor] = []

    def capture_down_proj_input(
        _module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]
    ) -> None:
        hook_outputs.append(
            inputs[0].detach().to(device="cpu", dtype=torch.bfloat16)[0]
        )

    records = []
    down_proj = model.model.layers[LAYER_INDEX].mlp.down_proj
    handle = down_proj.register_forward_pre_hook(capture_down_proj_input)
    try:
        for prompt_spec in PROMPTS:
            encoded = tokenizer(
                prompt_spec["prompt"],
                add_special_tokens=False,
                return_tensors="pt",
            )
            input_ids = encoded.input_ids.to(model.device)
            attention_mask = encoded.attention_mask.to(model.device)
            with torch.inference_mode():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                )
            if len(hook_outputs) != 1:
                raise RuntimeError(
                    f"Expected one Layer 7 capture for {prompt_spec['id']}, "
                    f"got {len(hook_outputs)}"
                )
            activations = hook_outputs.pop()
            expected_shape = (input_ids.shape[1], model.config.intermediate_size)
            if tuple(activations.shape) != expected_shape:
                raise RuntimeError(
                    f"Unexpected shape for {prompt_spec['id']}: "
                    f"{tuple(activations.shape)} != {expected_shape}"
                )

            ids_cpu = input_ids[0].cpu()
            token_ids = ids_cpu.tolist()
            tokens = [tokenizer.decode([token_id]) for token_id in token_ids]
            target_trace = activations[:, TARGET_NEURON].float()
            mean_activations = activations.float().mean(dim=0)
            top_values, top_indices = torch.topk(mean_activations, k=10)
            next_token_id = int(outputs.logits[0, -1].argmax().item())
            peak_position = int(target_trace.abs().argmax().item())

            record = {
                **prompt_spec,
                "input_ids": ids_cpu,
                "tokens": tokens,
                "activations": activations,
                "target_trace": target_trace,
                "next_token_id": next_token_id,
                "next_token": tokenizer.decode([next_token_id]),
                "top_mean_neurons": [
                    {"neuron": int(neuron), "mean_activation": float(value)}
                    for neuron, value in zip(top_indices, top_values, strict=True)
                ],
            }
            records.append(record)
            print(
                f"{prompt_spec['id']}: {len(tokens)} tokens, "
                f"next={record['next_token']!r}, N{TARGET_NEURON} peak="
                f"{float(target_trace[peak_position]):.4f} at "
                f"{tokens[peak_position]!r}"
            )
            del outputs
    finally:
        handle.remove()

    artifact = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "layer_index": LAYER_INDEX,
        "target_neuron": TARGET_NEURON,
        "activation_definition": "input to mlp.down_proj (post-SwiGLU)",
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, args.output)

    metadata = {
        **{key: value for key, value in artifact.items() if key != "records"},
        "records": [],
    }
    for record in records:
        metadata["records"].append(
            {
                key: value
                for key, value in record.items()
                if key not in {"input_ids", "activations", "target_trace"}
            }
            | {
                "input_ids": record["input_ids"].tolist(),
                "target_trace": record["target_trace"].tolist(),
                "activation_shape": list(record["activations"].shape),
                "activation_dtype": str(record["activations"].dtype),
            }
        )
    metadata_path = args.output.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Touched CPU pages: {touched / 1024**3:.2f} GiB")
    print(f"Saved tensor artifact: {args.output.resolve()}")
    print(f"Saved readable metadata: {metadata_path.resolve()}")


if __name__ == "__main__":
    main()
