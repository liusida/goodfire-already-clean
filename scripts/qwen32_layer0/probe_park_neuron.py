from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from gb10_load_llm import load_model_to_cuda
from transformers import AutoConfig, AutoTokenizer, Qwen3Model


MODEL_ID = "Qwen/Qwen3-32B"
MODEL_REVISION = "9216db5781bf21249d130ec9da846c4624c16137"
LAYER_INDEX = 0
TARGET_NEURON = 15647


@dataclass(frozen=True)
class Case:
    case_id: str
    group: str
    text: str
    target: str
    expected_min: float | None
    expected_max: float | None


CASES = [
    # Same token, unrelated senses: these test the paper's context-blind claim.
    Case("park_national", "exact Park: different senses", "The trail enters Yellowstone National Park before sunset.", "Park", 8.0, None),
    Case("park_surname", "exact Park: different senses", "The report was signed by Dr. Park after lunch.", "Park", 8.0, None),
    Case("park_band", "exact Park: different senses", "My brother listened to Linkin Park during the drive.", "Park", 8.0, None),
    Case("park_street", "exact Park: different senses", "Their office moved to Park Avenue last spring.", "Park", 8.0, None),
    Case("park_tv", "exact Park: different senses", "They watched an old South Park episode together.", "Park", 8.0, None),

    # A fixed carrier sentence isolates token form.
    Case("exact_Park", "case and inflection", "The sign on the old building simply said Park in faded paint.", "Park", 8.0, None),
    Case("upper_PARK", "case and inflection", "The sign on the old building simply said PARK in faded paint.", "PARK", 7.0, None),
    Case("lower_park", "case and inflection", "The sign on the old building simply said park in faded paint.", "park", 4.0, 9.0),
    Case("plural_parks", "case and inflection", "The sign on the old building simply said parks in faded paint.", "parks", 3.0, 8.0),
    Case("parking", "case and inflection", "The sign on the old building simply said parking in faded paint.", "parking", 3.0, 8.0),

    # These distinguish embedding-neighbourhood behavior from substring matching.
    Case("Bark", "embedding halo", "The sign on the old building simply said Bark in faded paint.", "Bark", 1.0, 5.0),
    Case("Clark", "embedding halo", "The sign on the old building simply said Clark in faded paint.", "Clark", 0.5, 5.0),
    Case("Stark", "embedding halo", "The sign on the old building simply said Stark in faded paint.", "Stark", 0.5, 5.0),
    Case("shark", "embedding halo", "The sign on the old building simply said shark in faded paint.", "shark", 0.5, 5.0),
    Case("Parkway", "merged spelling confounds", "The sign on the old building simply said Parkway in faded paint.", "Parkway", None, 4.0),
    Case("Parkinson", "merged spelling confounds", "The sign on the old building simply said Parkinson in faded paint.", "Parkinson", None, 4.0),
    Case("Parker", "merged spelling confounds", "The sign on the old building simply said Parker in faded paint.", "Parker", None, 4.5),
    Case("Parkland", "tokenization check", "The sign on the old building simply said Parkland in faded paint.", "Parkland", None, None),

    # Meaning-matched and unrelated controls should remain near zero.
    Case("Garden", "semantic controls", "The sign on the old building simply said Garden in faded paint.", "Garden", None, 1.5),
    Case("Reserve", "semantic controls", "The sign on the old building simply said Reserve in faded paint.", "Reserve", None, 1.5),
    Case("Forest", "semantic controls", "The sign on the old building simply said Forest in faded paint.", "Forest", None, 1.5),
    Case("parque", "translation controls", "The sign on the old building simply said parque in faded paint.", "parque", None, 1.5),
    Case("cjk_park", "translation controls", "The sign on the old building simply said 公园 in faded paint.", "公园", 1.0, 6.0),
    Case("Station", "unrelated controls", "The sign on the old building simply said Station in faded paint.", "Station", None, 1.0),
    Case("Museum", "unrelated controls", "The sign on the old building simply said Museum in faded paint.", "Museum", None, 1.0),
    Case("Library", "unrelated controls", "The sign on the old building simply said Library in faded paint.", "Library", None, 1.0),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe Qwen3-32B L0/N15647 (' Park').")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/layer0/park_neuron_probe"),
    )
    return parser.parse_args()


def case_passes(value: float, case: Case) -> bool | None:
    if case.expected_min is None and case.expected_max is None:
        return None
    if case.expected_min is not None and value < case.expected_min:
        return False
    if case.expected_max is not None and value > case.expected_max:
        return False
    return True


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run Qwen3-32B")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Build the Qwen3 base architecture with only layer 0. Transformers uses the
    # checkpoint index to match only this model's expected keys; the other 63
    # layers are never materialized as parameters. On this checkpoint, the token
    # embeddings and every layer-0 tensor live in shard 1 of 17.
    config = AutoConfig.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    checkpoint_layer_count = int(config.num_hidden_layers)
    config.num_hidden_layers = 1
    model, touched = load_model_to_cuda(
        Qwen3Model,
        MODEL_ID,
        config=config,
        revision=MODEL_REVISION,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        touch="auto",
        return_touched=True,
    )
    model.eval()

    texts = [case.text for case in CASES]
    encoded = tokenizer(
        texts,
        add_special_tokens=False,
        padding=True,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    offsets = encoded.pop("offset_mapping")
    encoded = {name: tensor.to(model.device) for name, tensor in encoded.items()}
    captured: list[torch.Tensor] = []

    def capture_target(
        _module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]
    ) -> None:
        captured.append(inputs[0][..., TARGET_NEURON].detach().float().cpu())

    down_proj = model.layers[LAYER_INDEX].mlp.down_proj
    handle = down_proj.register_forward_pre_hook(capture_target)
    try:
        with torch.inference_mode():
            model(**encoded, use_cache=False)
    finally:
        handle.remove()

    if len(captured) != 1:
        raise RuntimeError(f"Expected one layer-0 capture, got {len(captured)}")
    traces = captured[0]

    records = []
    for row, case in enumerate(CASES):
        start = case.text.index(case.target)
        end = start + len(case.target)
        positions = [
            pos
            for pos, (token_start, token_end) in enumerate(offsets[row].tolist())
            if token_end > start and token_start < end
        ]
        if not positions:
            raise RuntimeError(f"No tokens overlap target {case.target!r} in {case.text!r}")
        token_ids = encoded["input_ids"][row, positions].detach().cpu().tolist()
        tokens = [tokenizer.decode([token_id]) for token_id in token_ids]
        activations = [float(traces[row, pos]) for pos in positions]
        peak_local = max(range(len(positions)), key=lambda i: activations[i])
        value = activations[peak_local]
        record = {
            "case_id": case.case_id,
            "group": case.group,
            "text": case.text,
            "target": case.target,
            "target_tokens": tokens,
            "target_positions": positions,
            "target_activations": activations,
            "peak_token": tokens[peak_local],
            "peak_activation": value,
            "expected_min": case.expected_min,
            "expected_max": case.expected_max,
            "passes_expected_range": case_passes(value, case),
        }
        records.append(record)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "results.json"
    csv_path = args.output_dir / "results.csv"
    plot_path = args.output_dir / "activations.png"
    json_path.write_text(
        json.dumps(
            {
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "checkpoint_layer_count": checkpoint_layer_count,
                "loaded_layer_count": int(model.config.num_hidden_layers),
                "layer": LAYER_INDEX,
                "neuron": TARGET_NEURON,
                "activation_definition": "input to mlp.down_proj (post-SwiGLU)",
                "selection_rule": "maximum activation among tokens overlapping the marked target text",
                "records": records,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "case_id",
                "group",
                "target",
                "target_tokens",
                "target_activations",
                "peak_token",
                "peak_activation",
                "expected_min",
                "expected_max",
                "passes_expected_range",
                "text",
            ],
        )
        writer.writeheader()
        for record in records:
            flat = dict(record)
            flat.pop("target_positions")
            flat["target_tokens"] = " | ".join(record["target_tokens"])
            flat["target_activations"] = " | ".join(
                f"{value:.4f}" for value in record["target_activations"]
            )
            writer.writerow(flat)

    colors = {
        "exact Park: different senses": "#d9480f",
        "case and inflection": "#e98242",
        "embedding halo": "#7e57a2",
        "merged spelling confounds": "#cc79a7",
        "tokenization check": "#b0a236",
        "semantic controls": "#4c956c",
        "translation controls": "#58a6a6",
        "unrelated controls": "#8a9499",
    }
    fig, ax = plt.subplots(figsize=(14, 7), dpi=170)
    labels = [record["case_id"] for record in records]
    values = [record["peak_activation"] for record in records]
    bars = ax.bar(
        range(len(records)),
        values,
        color=[colors[record["group"]] for record in records],
        edgecolor="white",
        linewidth=0.5,
    )
    ax.axhline(0, color="#38444a", linewidth=0.8)
    ax.set_xticks(range(len(records)), labels, rotation=55, ha="right")
    ax.set_ylabel("L0/N15647 post-SwiGLU activation")
    ax.set_title("Qwen3-32B ' Park' neuron: controlled prompt tests")
    ax.grid(axis="y", alpha=0.25)
    ax.grid(axis="x", visible=False)
    ax.spines[["top", "right"]].set_visible(False)
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.18,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=7,
            rotation=90,
        )
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=color, label=group)
        for group, color in colors.items()
    ]
    ax.legend(handles=handles, fontsize=8, frameon=False, ncol=2, loc="upper right")
    fig.tight_layout()
    fig.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)

    tested = [r for r in records if r["passes_expected_range"] is not None]
    passed = sum(bool(r["passes_expected_range"]) for r in tested)
    print(f"Touched CPU pages: {touched / 1024**3:.2f} GiB")
    print(
        f"Loaded embeddings + {model.config.num_hidden_layers} transformer layer "
        f"from a {checkpoint_layer_count}-layer checkpoint"
    )
    print(f"Measured {len(records)} cases; expected-range checks: {passed}/{len(tested)} passed")
    for record in records:
        status = record["passes_expected_range"]
        mark = "OBS" if status is None else ("PASS" if status else "FAIL")
        token_summary = "/".join(repr(token) for token in record["target_tokens"])
        print(
            f"{mark:4} {record['case_id']:<18} {record['peak_activation']:>7.3f}  "
            f"tokens={token_summary}"
        )
    print(f"Saved {json_path.resolve()}")
    print(f"Saved {csv_path.resolve()}")
    print(f"Saved {plot_path.resolve()}")


if __name__ == "__main__":
    main()
