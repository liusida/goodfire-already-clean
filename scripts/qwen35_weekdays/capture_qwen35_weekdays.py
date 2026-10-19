from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from gb10_load_llm import load_model_to_cuda
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "Qwen/Qwen3.5-0.8B-Base"
MODEL_REVISION = "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"
WEEKDAYS = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]
TEMPLATES = [
    "The team reviewed the project plan and scheduled the next meeting for {weekday}.",
    "After checking the forecast, Maya decided to begin the hiking trip on {weekday}.",
    "The neighborhood bakery will introduce its new seasonal menu this {weekday}.",
    "Jordan finished preparing the lecture and will present it to the class on {weekday}.",
    "The mechanic ordered the replacement part and expects it to arrive by {weekday}.",
    "Our family compared everyone's calendar and chose to have dinner together on {weekday}.",
    "The city council completed its review and will announce the final decision on {weekday}.",
    "Elena has trained for several months, and the championship race takes place on {weekday}.",
    "The research group finished collecting the measurements and will analyze them on {weekday}.",
    "The museum is renovating its main gallery and plans to reopen it next {weekday}.",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture every Qwen3.5-0.8B Layer/MLP activation for 70 weekday prompts."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/weekdays/qwen35_08b_weekday_activations.pt"),
    )
    return parser.parse_args()


def build_samples() -> list[dict[str, str | int]]:
    return [
        {
            "sample_id": f"template_{template_index:02d}_{weekday.lower()}",
            "template_index": template_index,
            "template": template,
            "weekday": weekday,
            "prompt": template.format(weekday=weekday),
        }
        for template_index, template in enumerate(TEMPLATES)
        for weekday in WEEKDAYS
    ]


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run Qwen3.5-0.8B-Base")

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

    layers = model.model.layers
    num_layers = len(layers)
    intermediate_size = int(model.config.intermediate_size)
    captured: dict[int, torch.Tensor] = {}
    handles = []

    def make_hook(layer_index: int):
        def capture_down_proj_input(
            _module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]
        ) -> None:
            if layer_index in captured:
                raise RuntimeError(f"Layer {layer_index} fired more than once")
            # Input to down_proj is silu(gate_proj(x)) * up_proj(x).
            captured[layer_index] = inputs[0].detach().to(
                device="cpu", dtype=torch.bfloat16
            )

        return capture_down_proj_input

    for layer_index, layer in enumerate(layers):
        handles.append(
            layer.mlp.down_proj.register_forward_pre_hook(make_hook(layer_index))
        )

    samples = build_samples()
    records = []
    weekday_token_ids = {
        weekday: tokenizer(
            f" {weekday}", add_special_tokens=False, return_tensors="pt"
        ).input_ids[0].tolist()
        for weekday in WEEKDAYS
    }
    if any(len(ids) != 1 for ids in weekday_token_ids.values()):
        raise RuntimeError(f"Expected single-token weekdays, got {weekday_token_ids}")

    try:
        for template_index, template in enumerate(TEMPLATES):
            batch_samples = [
                sample for sample in samples if sample["template_index"] == template_index
            ]
            prompts = [str(sample["prompt"]) for sample in batch_samples]
            encoded = tokenizer(
                prompts,
                add_special_tokens=False,
                padding=True,
                return_tensors="pt",
            )
            if not torch.all(encoded.attention_mask == 1):
                raise RuntimeError(
                    f"Template {template_index} produced unequal token lengths across weekdays"
                )
            captured.clear()
            with torch.inference_mode():
                model.model(
                    input_ids=encoded.input_ids.to(model.device),
                    attention_mask=encoded.attention_mask.to(model.device),
                    use_cache=False,
                )
            if set(captured) != set(range(num_layers)):
                raise RuntimeError(
                    f"Expected captures for {num_layers} layers, got {sorted(captured)}"
                )

            for batch_index, sample in enumerate(batch_samples):
                input_ids = encoded.input_ids[batch_index].cpu()
                token_ids = input_ids.tolist()
                tokens = [tokenizer.decode([token_id]) for token_id in token_ids]
                weekday_id = weekday_token_ids[str(sample["weekday"])][0]
                positions = (input_ids == weekday_id).nonzero(as_tuple=False).flatten()
                if len(positions) != 1:
                    raise RuntimeError(
                        f"Expected one weekday token in {sample['sample_id']}, got {positions.tolist()}"
                    )
                weekday_position = int(positions.item())
                activations = torch.stack(
                    [captured[layer][batch_index] for layer in range(num_layers)],
                    dim=0,
                )
                expected_shape = (num_layers, len(token_ids), intermediate_size)
                if tuple(activations.shape) != expected_shape:
                    raise RuntimeError(
                        f"Unexpected shape for {sample['sample_id']}: "
                        f"{tuple(activations.shape)} != {expected_shape}"
                    )
                records.append(
                    {
                        **sample,
                        "input_ids": input_ids,
                        "tokens": tokens,
                        "weekday_token_id": weekday_id,
                        "weekday_position": weekday_position,
                        "activations": activations,
                    }
                )
            print(
                f"template {template_index:02d}: captured 7 prompts × "
                f"{encoded.input_ids.shape[1]} tokens × {num_layers} layers"
            )
    finally:
        for handle in handles:
            handle.remove()

    artifact = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "activation_definition": "input to mlp.down_proj (post-SwiGLU)",
        "num_layers": num_layers,
        "intermediate_size": intermediate_size,
        "weekdays": WEEKDAYS,
        "templates": TEMPLATES,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, args.output)

    metadata = {
        **{key: value for key, value in artifact.items() if key != "records"},
        "records": [
            {
                key: value
                for key, value in record.items()
                if key not in {"input_ids", "activations"}
            }
            | {
                "input_ids": record["input_ids"].tolist(),
                "activation_shape": list(record["activations"].shape),
                "activation_dtype": str(record["activations"].dtype),
            }
            for record in records
        ],
    }
    metadata_path = args.output.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Touched CPU pages: {touched / 1024**3:.2f} GiB")
    print(f"Saved {len(records)} prompt captures: {args.output.resolve()}")
    print(f"Saved readable index: {metadata_path.resolve()}")


if __name__ == "__main__":
    main()
