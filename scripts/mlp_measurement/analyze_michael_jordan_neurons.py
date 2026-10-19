from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch


INPUT_DIR = Path(
    "outputs/mlp_measurement/Qwen3.5-9B-Base/michael_jordan_name_probe"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank strong and name-selective neurons from the saved name probe."
    )
    parser.add_argument("--input-dir", type=Path, default=INPUT_DIR)
    parser.add_argument("--top-k", type=int, default=100)
    return parser.parse_args()


def flat_coordinates(indices: torch.Tensor, neurons_per_layer: int) -> list[tuple[int, int]]:
    return [
        (int(index) // neurons_per_layer, int(index) % neurons_per_layer)
        for index in indices.tolist()
    ]


def write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    def cell(value: object) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(cell(header) for header in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend(
        "| " + " | ".join(cell(value) for value in row) + " |" for row in rows
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")
    activations = torch.load(
        args.input_dir / "activations.pt",
        map_location="cpu",
        weights_only=True,
    ).float()
    metadata = json.loads((args.input_dir / "metadata.json").read_text())
    prompts = metadata["prompts"]
    prompt_count, layer_count, neuron_count = activations.shape
    if prompt_count != len(prompts):
        raise ValueError("Activation tensor and metadata prompt counts disagree")

    # This is a within-layer scale correction, not corpus calibration. It keeps
    # layers with large raw activation scales from dominating global rankings.
    layer_rms = activations.square().mean(dim=(0, 2)).sqrt().clamp_min(1e-8)
    normalized = activations / layer_rms[None, :, None]
    torch.save(layer_rms, args.input_dir / "layer_rms.pt")

    per_prompt_rows: list[dict[str, object]] = []
    flat_width = layer_count * neuron_count
    k = min(args.top_k, flat_width)
    for prompt_index, prompt in enumerate(prompts):
        flat = normalized[prompt_index].reshape(-1)
        for tail, indices in (
            ("positive", torch.topk(flat, k).indices),
            ("negative", torch.topk(-flat, k).indices),
        ):
            for rank, (layer, neuron) in enumerate(
                flat_coordinates(indices, neuron_count), start=1
            ):
                per_prompt_rows.append(
                    {
                        "prompt_id": prompt["id"],
                        "group": prompt["group"],
                        "name": prompt["name"],
                        "target_token": prompt["target_token"],
                        "tail": tail,
                        "rank": rank,
                        "layer": layer,
                        "neuron": neuron,
                        "activation": float(activations[prompt_index, layer, neuron]),
                        "layer_rms_units": float(normalized[prompt_index, layer, neuron]),
                    }
                )
    write_rows(
        args.input_dir / "per_prompt_strong_neurons.csv",
        [
            "prompt_id", "group", "name", "target_token", "tail", "rank",
            "layer", "neuron", "activation", "layer_rms_units",
        ],
        per_prompt_rows,
    )

    jordan_indices = [
        index for index, prompt in enumerate(prompts) if prompt["group"] == "jordan-natural"
    ]
    jordan_values = activations[jordan_indices]
    jordan_mean = jordan_values.mean(dim=0)
    jordan_normalized = jordan_mean / layer_rms[:, None]
    jordan_flat = jordan_normalized.reshape(-1)
    jordan_rows: list[dict[str, object]] = []
    for tail, indices in (
        ("positive", torch.topk(jordan_flat, k).indices),
        ("negative", torch.topk(-jordan_flat, k).indices),
    ):
        for rank, (layer, neuron) in enumerate(
            flat_coordinates(indices, neuron_count), start=1
        ):
            values = jordan_values[:, layer, neuron]
            mean = float(values.mean())
            jordan_rows.append(
                {
                    "tail": tail,
                    "rank": rank,
                    "layer": layer,
                    "neuron": neuron,
                    "mean_activation": mean,
                    "min_activation": float(values.min()),
                    "max_activation": float(values.max()),
                    "layer_rms_units": float(jordan_normalized[layer, neuron]),
                    "sign_consistency": float((values.sign() == (1 if mean >= 0 else -1)).float().mean()),
                }
            )
    write_rows(
        args.input_dir / "jordan_context_consistent_neurons.csv",
        [
            "tail", "rank", "layer", "neuron", "mean_activation",
            "min_activation", "max_activation", "layer_rms_units", "sign_consistency",
        ],
        jordan_rows,
    )

    matched_id = metadata["matched_jordan_prompt_id"]
    jordan_reference_index = next(
        index for index, prompt in enumerate(prompts) if prompt["id"] == matched_id
    )
    other_indices = [
        index
        for index, prompt in enumerate(prompts)
        if prompt["group"] == "matched-counterfactual"
    ]
    decisive_indices = [
        index
        for index, prompt in enumerate(prompts)
        if prompt["group"] == "jordan-meaning-control"
    ]
    reference = activations[jordan_reference_index]
    others = activations[other_indices]
    other_mean = others.mean(dim=0)
    other_std = others.std(dim=0, unbiased=False)
    delta = reference - other_mean
    delta_normalized = delta / layer_rms[:, None]
    selectivity_order = torch.argsort(delta_normalized.abs().reshape(-1), descending=True)
    selectivity_rows: list[dict[str, object]] = []
    other_names = [prompts[index]["name"] for index in other_indices]
    for rank, (layer, neuron) in enumerate(
        flat_coordinates(selectivity_order[: min(args.top_k * 5, flat_width)], neuron_count),
        start=1,
    ):
        row: dict[str, object] = {
            "rank": rank,
            "layer": layer,
            "neuron": neuron,
            "jordan_activation": float(reference[layer, neuron]),
            "other_mean": float(other_mean[layer, neuron]),
            "other_std": float(other_std[layer, neuron]),
            "jordan_minus_other_mean": float(delta[layer, neuron]),
            "delta_layer_rms_units": float(delta_normalized[layer, neuron]),
        }
        for name, prompt_index in zip(other_names, other_indices):
            row[name] = float(activations[prompt_index, layer, neuron])
        selectivity_rows.append(row)
    selectivity_fields = [
        "rank", "layer", "neuron", "jordan_activation", "other_mean",
        "other_std", "jordan_minus_other_mean", "delta_layer_rms_units",
        *other_names,
    ]
    write_rows(
        args.input_dir / "jordan_vs_names_selective_neurons.csv",
        selectivity_fields,
        selectivity_rows,
    )

    michael_b_jordan_index = next(
        index for index in other_indices if prompts[index]["name"] == "Michael B. Jordan"
    )
    decisive_comparison_indices = [
        jordan_reference_index,
        michael_b_jordan_index,
        *decisive_indices,
    ]
    decisive_rows: list[dict[str, object]] = []
    for selectivity_row in selectivity_rows[:30]:
        layer = int(selectivity_row["layer"])
        neuron = int(selectivity_row["neuron"])
        row: dict[str, object] = {
            "selectivity_rank": selectivity_row["rank"],
            "layer": layer,
            "neuron": neuron,
        }
        for prompt_index in decisive_comparison_indices:
            row[prompts[prompt_index]["id"]] = float(
                activations[prompt_index, layer, neuron]
            )
        decisive_rows.append(row)
    decisive_fields = [
        "selectivity_rank",
        "layer",
        "neuron",
        *[prompts[index]["id"] for index in decisive_comparison_indices],
    ]
    write_rows(
        args.input_dir / "decisive_jordan_controls.csv",
        decisive_fields,
        decisive_rows,
    )

    summary = {
        "activation_shape": list(activations.shape),
        "normalization": (
            "raw activation divided by RMS over all prompts and neurons within each layer"
        ),
        "top_k_per_tail_per_prompt": k,
        "jordan_natural_prompt_count": len(jordan_indices),
        "counterfactual_prompt_count": len(other_indices),
        "decisive_control_count": len(decisive_indices),
        "matched_jordan_prompt_id": matched_id,
        "files": {
            "per_prompt": "per_prompt_strong_neurons.csv",
            "jordan_contexts": "jordan_context_consistent_neurons.csv",
            "jordan_selectivity": "jordan_vs_names_selective_neurons.csv",
            "decisive_controls": "decisive_jordan_controls.csv",
            "report": "report.md",
        },
    }
    (args.input_dir / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    prompt_rows = [
        [
            prompt["id"],
            prompt["group"],
            prompt["name"],
            f"`{prompt['target_token']}`",
            prompt["text"],
        ]
        for prompt in prompts
    ]
    positive_jordan = [row for row in jordan_rows if row["tail"] == "positive"][:20]
    negative_jordan = [row for row in jordan_rows if row["tail"] == "negative"][:20]

    def jordan_table(rows: list[dict[str, object]]) -> str:
        return markdown_table(
            [
                "Rank", "Neuron", "Mean", "Min", "Max", "Layer RMS units",
                "Sign consistency",
            ],
            [
                [
                    row["rank"],
                    f"L{row['layer']}:N{row['neuron']}",
                    f"{row['mean_activation']:+.5f}",
                    f"{row['min_activation']:+.5f}",
                    f"{row['max_activation']:+.5f}",
                    f"{row['layer_rms_units']:+.2f}",
                    f"{float(row['sign_consistency']):.0%}",
                ]
                for row in rows
            ],
        )

    selectivity_table = markdown_table(
        ["Rank", "Neuron", "Jordan", "Other-name mean", "Raw delta", "Delta RMS units"],
        [
            [
                row["rank"],
                f"L{row['layer']}:N{row['neuron']}",
                f"{row['jordan_activation']:+.5f}",
                f"{row['other_mean']:+.5f}",
                f"{row['jordan_minus_other_mean']:+.5f}",
                f"{row['delta_layer_rms_units']:+.2f}",
            ]
            for row in selectivity_rows[:30]
        ],
    )

    decisive_headers = [
        f"{prompts[index]['name']} ({prompts[index]['id']})"
        for index in decisive_comparison_indices
    ]
    decisive_table = markdown_table(
        ["Rank", "Neuron", *decisive_headers],
        [
            [
                row["selectivity_rank"],
                f"L{row['layer']}:N{row['neuron']}",
                *[
                    f"{float(row[prompts[index]['id']]):+.4f}"
                    for index in decisive_comparison_indices
                ],
            ]
            for row in decisive_rows
        ],
    )

    compared_prompt_ids = [
        matched_id,
        *[prompts[index]["id"] for index in other_indices],
        *[prompts[index]["id"] for index in decisive_indices],
    ]
    per_name_sections: list[str] = []
    for prompt_id in compared_prompt_ids:
        prompt = next(prompt for prompt in prompts if prompt["id"] == prompt_id)
        rows = [
            row
            for row in per_prompt_rows
            if row["prompt_id"] == prompt_id and int(row["rank"]) <= 5
        ]
        rows.sort(key=lambda row: (str(row["tail"]), int(row["rank"])))
        per_name_sections.append(
            f"### {prompt['name']} (`{prompt_id}`, target `{prompt['target_token']}`)\n\n"
            + markdown_table(
                ["Tail", "Rank", "Neuron", "Raw activation", "Layer RMS units"],
                [
                    [
                        row["tail"],
                        row["rank"],
                        f"L{row['layer']}:N{row['neuron']}",
                        f"{row['activation']:+.5f}",
                        f"{row['layer_rms_units']:+.2f}",
                    ]
                    for row in rows
                ],
            )
        )

    report = f"""# Michael Jordan MLP-neuron probe

Model: **{metadata['model_id']}**  
Activation: **{metadata['activation_definition']}**  
Captured position: **{metadata['capture_position']}**  
Tensor shape: **{prompt_count} prompts × {layer_count} layers × {neuron_count} neurons**

## Method

The capture contains the raw post-SwiGLU activation of every MLP neuron at the
specified target token. Ten natural prompts use `Michael Jordan`. The
matched comparison uses the same sentence as `J10`, replacing only the name
with {len(other_indices)} other famous people. Six decisive controls capture
the `Jordan` token in other people, a product, a river, a country, and a
standalone surname use.

Global rankings use **within-layer RMS units**: each raw activation is divided
by the RMS over all {prompt_count} captured prompts and all neurons in that layer. This
corrects gross scale differences between layers, but it is not a corpus-level
z-score or proof of semantic selectivity. Raw activations are retained in every
table and in `activations.pt`.

## Prompt set

{markdown_table(['ID', 'Group', 'Name', 'Target token', 'Prompt'], prompt_rows)}

## Strongest positive neurons averaged across 10 Jordan contexts

{jordan_table(positive_jordan)}

## Strongest negative neurons averaged across 10 Jordan contexts

{jordan_table(negative_jordan)}

## Largest Jordan-versus-other-name differences

This comparison uses `J10` and the {len(other_indices)} counterfactual prompts, whose wording is
identical except for the person name. Ranking is by the absolute difference in
within-layer RMS units, so both Jordan-positive and Jordan-negative candidates
appear.

{selectivity_table}

## Decisive Jordan-meaning controls

These are the same leading matched-template selectivity candidates evaluated
on `Michael B. Jordan` and on the `Jordan` token in the six meaning controls.
This distinguishes Michael Jordan identity from the surname token, other
people named Jordan, the product name, and geographic meanings.

{decisive_table}

## Strongest neurons for each matched name and meaning control

{chr(10).join(per_name_sections)}

## Data files

- `activations.pt`: complete float32 tensor, shape `{tuple(activations.shape)}`
- `metadata.json`: prompts, names, target positions, token IDs, and target tokens
- `per_prompt_strong_neurons.csv`: top positive and negative neurons for every prompt
- `jordan_context_consistent_neurons.csv`: neurons ranked across ten Jordan contexts
- `jordan_vs_names_selective_neurons.csv`: matched-template Jordan/name comparison
- `decisive_jordan_controls.csv`: leading candidates across same-surname and Jordan-meaning controls
- `layer_rms.pt`: per-layer scale used for normalized rankings
"""
    (args.input_dir / "report.md").write_text(report, encoding="utf-8")

    print(f"Activation tensor: {tuple(activations.shape)}")
    print("\nTop Jordan-context positive neurons:")
    for row in [row for row in jordan_rows if row["tail"] == "positive"][:10]:
        print(
            f"L{row['layer']}:N{row['neuron']}  mean {row['mean_activation']:+.5f}  "
            f"{row['layer_rms_units']:+.2f} RMS  consistency {row['sign_consistency']:.0%}"
        )
    print("\nTop Jordan-context negative neurons:")
    for row in [row for row in jordan_rows if row["tail"] == "negative"][:10]:
        print(
            f"L{row['layer']}:N{row['neuron']}  mean {row['mean_activation']:+.5f}  "
            f"{row['layer_rms_units']:+.2f} RMS  consistency {row['sign_consistency']:.0%}"
        )
    print("\nLargest matched-template Jordan/name differences:")
    for row in selectivity_rows[:10]:
        print(
            f"L{row['layer']}:N{row['neuron']}  Jordan {row['jordan_activation']:+.5f}  "
            f"others {row['other_mean']:+.5f}  delta {row['delta_layer_rms_units']:+.2f} RMS"
        )
    print(f"\nSaved analysis: {args.input_dir.resolve()}")


if __name__ == "__main__":
    main()
