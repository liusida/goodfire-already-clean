from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from gb10_load_llm import load_model_to_cuda
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "Qwen/Qwen3.5-9B-Base"
OUTPUT_DIR = Path(
    "outputs/mlp_measurement/Qwen3.5-9B-Base/michael_jordan_name_probe"
)

NATURAL_TEXTS = [
    "Michael Jordan led the Chicago Bulls to six NBA championships during the 1990s.",
    "Fans still debate whether Michael Jordan is the greatest basketball player of all time.",
    "The documentary follows Michael Jordan through the Bulls' final championship season.",
    "As a rookie, Michael Jordan quickly became one of the league's most exciting scorers.",
    "Nike's partnership with Michael Jordan transformed the market for signature basketball shoes.",
    "After retiring from basketball, Michael Jordan became the principal owner of the Charlotte Hornets.",
    "Collectors paid a record price for a jersey worn by Michael Jordan in the 1998 NBA Finals.",
    "The coach described how Michael Jordan demanded intense effort from everyone at practice.",
    "A generation of young athletes tried to imitate Michael Jordan on neighborhood basketball courts.",
    "The documentary traces how Michael Jordan became one of the most recognizable public figures in the world.",
]

COUNTERFACTUAL_NAMES = [
    "Michael Jackson",
    "Brad Pitt",
    "Elon Musk",
    "Taylor Swift",
    "Barack Obama",
    "Serena Williams",
    "Cristiano Ronaldo",
    "Oprah Winfrey",
    "Steve Jobs",
    "Muhammad Ali",
    "Scottie Pippen",
    "Dennis Rodman",
    "Michael B. Jordan",
]

COUNTERFACTUAL_TEMPLATE = (
    "The documentary traces how {name} became one of the most recognizable "
    "public figures in the world."
)

DECISIVE_CONTROLS = [
    ("D01", "Jordan Peele", "The director Jordan Peele wrote and directed the film Get Out.", "Jordan"),
    ("D02", "Jordan Peterson", "The psychologist Jordan Peterson became known for his lectures and books.", "Jordan"),
    ("D03", "Air Jordan", "Nike introduced the Air Jordan sneaker line in the 1980s.", "Jordan"),
    ("D04", "Jordan River", "The Jordan River flows through the Jordan Rift Valley toward the Dead Sea.", "Jordan"),
    ("D05", "Jordan (country)", "The country Jordan is located at the crossroads of Asia, Africa, and Europe.", "Jordan"),
    ("D06", "Jordan (surname)", "The surname is Jordan.", "Jordan"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Capture every post-SwiGLU MLP neuron at the final name token in "
            "Michael Jordan and matched famous-name prompts."
        )
    )
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def prompt_record(
    prompt_id: str,
    group: str,
    text: str,
    name: str,
    target_text: str | None = None,
) -> dict[str, object]:
    target_text = name if target_text is None else target_text
    target_start = text.index(target_text)
    start = text.find(name)
    if start < 0:
        start = target_start
    return {
        "id": prompt_id,
        "group": group,
        "name": name,
        "text": text,
        "name_start": start,
        "name_end": start + len(name),
        "target_text": target_text,
        "target_start": target_start,
        "target_end": target_start + len(target_text),
    }


def build_prompts() -> list[dict[str, str]]:
    prompts = [
        prompt_record(f"J{index:02d}", "jordan-natural", text, "Michael Jordan")
        for index, text in enumerate(NATURAL_TEXTS, start=1)
    ]
    prompts.extend(
        prompt_record(
            f"C{index:02d}",
            "matched-counterfactual",
            COUNTERFACTUAL_TEMPLATE.format(name=name),
            name,
        )
        for index, name in enumerate(COUNTERFACTUAL_NAMES, start=1)
    )
    prompts.extend(
        prompt_record(test_id, "jordan-meaning-control", text, label, target_text)
        for test_id, label, text, target_text in DECISIVE_CONTROLS
    )
    return prompts


def locate_final_name_token(
    tokenizer: AutoTokenizer,
    text: str,
    name_end: int,
) -> tuple[int, int, str]:
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    candidates = [
        index
        for index, (start, end) in enumerate(encoded.offset_mapping)
        if start < name_end <= end
    ]
    if not candidates:
        candidates = [
            index
            for index, (_start, end) in enumerate(encoded.offset_mapping)
            if end <= name_end
        ]
    if not candidates:
        raise ValueError(f"Could not locate final name token in {text!r}")
    position = candidates[-1]
    token_id = int(encoded.input_ids[position])
    token = tokenizer.decode(
        [token_id],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    return position, token_id, token


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    prompts = build_prompts()
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    for record in prompts:
        position, token_id, token = locate_final_name_token(
            tokenizer,
            str(record["text"]),
            int(record["target_end"]),
        )
        record["target_position"] = position
        record["target_token_id"] = token_id
        record["target_token"] = token

    encoded = tokenizer(
        [str(record["text"]) for record in prompts],
        add_special_tokens=False,
        padding=True,
        return_tensors="pt",
    )
    target_positions = torch.tensor(
        [int(record["target_position"]) for record in prompts],
        dtype=torch.long,
        device="cuda",
    )
    batch_indices = torch.arange(len(prompts), device="cuda")

    model, touched = load_model_to_cuda(
        AutoModelForCausalLM,
        args.model,
        revision=args.revision,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        touch="auto",
        return_touched=True,
    )
    model.eval()

    captured: list[torch.Tensor | None] = [None] * len(model.model.layers)
    handles: list[torch.utils.hooks.RemovableHandle] = []

    def make_hook(layer_index: int):
        def hook(_module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
            captured[layer_index] = (
                inputs[0][batch_indices, target_positions]
                .detach()
                .to(dtype=torch.float32, device="cpu")
            )

        return hook

    for layer_index, layer in enumerate(model.model.layers):
        handles.append(
            layer.mlp.down_proj.register_forward_pre_hook(make_hook(layer_index))
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

    if any(value is None for value in captured):
        missing = [index for index, value in enumerate(captured) if value is None]
        raise RuntimeError(f"MLP hooks did not fire for layers: {missing}")
    activations = torch.stack([value for value in captured if value is not None], dim=1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(activations, args.output_dir / "activations.pt")
    metadata = {
        "model_id": args.model,
        "model_revision": args.revision,
        "activation_definition": "input to mlp.down_proj (post-SwiGLU)",
        "capture_position": (
            "final token overlapping target_text; final name token for matched "
            "people and the Jordan token for meaning controls"
        ),
        "shape": list(activations.shape),
        "dtype": str(activations.dtype),
        "cpu_pages_touched_gib": touched / 1024**3,
        "matched_jordan_prompt_id": "J10",
        "prompts": prompts,
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"Captured tensor shape: {tuple(activations.shape)}")
    for record in prompts:
        print(
            f"{record['id']}  {record['name']:<18}  "
            f"position {record['target_position']:<2}  {record['target_token']!r}"
        )
    print(f"Saved: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
