from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

import torch
from gb10_load_llm import load_model_to_cuda
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from capture_michael_jordan_neurons import MODEL_ID, build_prompts  # noqa: E402


OUTPUT_DIR = Path(
    "outputs/mlp_measurement/Qwen3.5-9B-Base/michael_jordan_name_probe"
)
CANDIDATES = [(5, 11725), (6, 813), (8, 680), (4, 463)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render token-level prompt heatmaps for Jordan candidate neurons."
    )
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def token_text(tokenizer: AutoTokenizer, token_id: int) -> str:
    return tokenizer.decode(
        [token_id],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


def background(value: float, scale: float) -> str:
    strength = min(abs(value) / scale, 1.0) if scale > 0 else 0.0
    alpha = 0.04 + 0.76 * strength
    rgb = "225, 88, 54" if value >= 0 else "60, 110, 190"
    return f"rgba({rgb}, {alpha:.3f})"


def render_token(token: str, value: float, scale: float, target: bool) -> str:
    visible = html.escape(token).replace("\n", "↵<br>")
    if not visible:
        visible = "∅"
    outline = "outline:2px solid #111;outline-offset:1px;" if target else ""
    return (
        f'<span title="activation {value:+.6f}" '
        f'style="background:{background(value, scale)};{outline}'
        'border-radius:3px;padding:2px 1px;">'
        f"{visible}</span>"
    )


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    prompts = build_prompts()
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    encoded = tokenizer(
        [str(prompt["text"]) for prompt in prompts],
        add_special_tokens=False,
        padding=True,
        return_tensors="pt",
    )

    metadata_path = args.output_dir / "metadata.json"
    saved_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    saved_by_id = {prompt["id"]: prompt for prompt in saved_metadata["prompts"]}
    target_positions = [int(saved_by_id[prompt["id"]]["target_position"]) for prompt in prompts]

    model, _touched = load_model_to_cuda(
        AutoModelForCausalLM,
        args.model,
        revision=args.revision,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        touch="auto",
        return_touched=True,
    )
    model.eval()

    captured: dict[str, torch.Tensor] = {}
    handles: list[torch.utils.hooks.RemovableHandle] = []

    def make_hook(layer: int, neuron: int):
        key = f"L{layer}:N{neuron}"

        def hook(_module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
            captured[key] = (
                inputs[0][..., neuron].detach().to(dtype=torch.float32, device="cpu")
            )

        return hook

    for layer, neuron in CANDIDATES:
        handles.append(
            model.model.layers[layer].mlp.down_proj.register_forward_pre_hook(
                make_hook(layer, neuron)
            )
        )
    try:
        with torch.inference_mode():
            model.model(
                input_ids=encoded.input_ids.to(model.device),
                attention_mask=encoded.attention_mask.to(model.device),
                use_cache=False,
            )
    finally:
        for handle in handles:
            handle.remove()

    expected = {f"L{layer}:N{neuron}" for layer, neuron in CANDIDATES}
    if set(captured) != expected:
        raise RuntimeError(f"Missing captures: {sorted(expected - set(captured))}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(captured, args.output_dir / "candidate_token_activations.pt")

    sections: list[str] = []
    for layer, neuron in CANDIDATES:
        key = f"L{layer}:N{neuron}"
        values = captured[key]
        valid_values = torch.cat(
            [values[index, : int(encoded.attention_mask[index].sum())] for index in range(len(prompts))]
        )
        scale = float(valid_values.abs().max())
        prompt_blocks: list[str] = []
        for index, prompt in enumerate(prompts):
            length = int(encoded.attention_mask[index].sum())
            ids = encoded.input_ids[index, :length].tolist()
            token_values = values[index, :length].tolist()
            target_position = target_positions[index]
            colored = "".join(
                render_token(
                    token_text(tokenizer, token_id),
                    float(value),
                    scale,
                    position == target_position,
                )
                for position, (token_id, value) in enumerate(zip(ids, token_values))
            )
            target_value = float(token_values[target_position])
            prompt_blocks.append(
                f"### {prompt['id']} · {prompt['name']} · target {target_value:+.5f}\n\n"
                '<div style="font-family:ui-monospace,monospace;line-height:1.9;'
                'padding:10px 12px;border:1px solid #d8dee6;border-radius:8px;">'
                f"{colored}</div>"
            )
        sections.append(
            f"## {key}\n\n"
            f"Color scale: **±{scale:.5f}** across all valid prompt tokens for this neuron.\n\n"
            + "\n\n".join(prompt_blocks)
        )

    report = f"""# Jordan candidate neuron token heatmaps

Model: **{args.model}**  
Activation: **input to `mlp.down_proj` (post-SwiGLU)**  
Prompts: **{len(prompts)}**

Red indicates positive activation and blue indicates negative activation. Color
intensity is linear in the absolute raw activation, using one symmetric scale
per neuron across all prompts. The outlined token is the captured target token.
Hover over a token in an HTML-capable Markdown viewer to see its exact raw
activation.

{"\n\n".join(sections)}
"""
    report_path = args.output_dir / "candidate_neuron_prompt_heatmaps.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"Saved token activations: {(args.output_dir / 'candidate_token_activations.pt').resolve()}")
    print(f"Saved Markdown report: {report_path.resolve()}")


if __name__ == "__main__":
    main()
