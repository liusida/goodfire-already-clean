from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract the top activation fraction independently within each layer."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("outputs/weekdays/qwen35_08b_weekday_activations.pt"),
    )
    parser.add_argument("--fraction", type=float, default=0.0001)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "outputs/weekdays/qwen35_08b_weekday_layerwise_top_0.01pct_activations.csv"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 < args.fraction <= 1:
        raise ValueError("--fraction must be in (0, 1]")

    artifact = torch.load(args.input, map_location="cpu", weights_only=False)
    records = artifact["records"]
    num_layers = artifact["num_layers"]
    width = artifact["intermediate_size"]
    values_per_layer = sum(len(record["input_ids"]) * width for record in records)
    keep_per_layer = math.ceil(values_per_layer * args.fraction)

    record_ends: list[int] = []
    running = 0
    for record in records:
        running += len(record["input_ids"]) * width
        record_ends.append(running)

    rows: list[dict[str, object]] = []
    layer_stats: list[dict[str, object]] = []
    for layer in range(num_layers):
        layer_values = torch.cat(
            [record["activations"][layer].reshape(-1).float() for record in records]
        )
        mean = layer_values.double().mean().item()
        std = layer_values.double().std(correction=0).item()
        top_values, _ = torch.topk(
            layer_values, keep_per_layer, sorted=True
        )
        cutoff = top_values[-1].item()
        strictly_above = torch.count_nonzero(layer_values > cutoff).item()
        equal_to_cutoff = torch.count_nonzero(layer_values == cutoff).item()
        selected_offsets = torch.nonzero(layer_values >= cutoff, as_tuple=False).squeeze(1)
        selected_values = layer_values[selected_offsets]
        selected_values, order = torch.sort(selected_values, descending=True)
        selected_offsets = selected_offsets[order]

        layer_stats.append(
            {
                "layer": layer,
                "count": layer_values.numel(),
                "mean": mean,
                "std": std,
                "minimum": layer_values.min().item(),
                "maximum": layer_values.max().item(),
                "top_fraction_cutoff": cutoff,
                "cutoff_z_score": (cutoff - mean) / std,
                "target_selected_count": keep_per_layer,
                "selected_count_including_cutoff_ties": selected_values.numel(),
                "strictly_above_cutoff": strictly_above,
                "equal_to_cutoff": equal_to_cutoff,
            }
        )

        for layer_rank, (value, layer_offset) in enumerate(
            zip(selected_values.tolist(), selected_offsets.tolist(), strict=True),
            start=1,
        ):
            record_index = bisect.bisect_right(record_ends, layer_offset)
            record_start = 0 if record_index == 0 else record_ends[record_index - 1]
            record_offset = layer_offset - record_start
            token_position = record_offset // width
            neuron = record_offset % width
            record = records[record_index]
            rows.append(
                {
                    "layer": layer,
                    "layer_rank": layer_rank,
                    "layer_percentile_lower_bound": 1.0
                    - (layer_rank / values_per_layer),
                    "value": value,
                    "layer_z_score": (value - mean) / std,
                    "neuron": neuron,
                    "sample_id": record["sample_id"],
                    "template_index": record["template_index"],
                    "weekday": record["weekday"],
                    "token_position": token_position,
                    "token": record["tokens"][token_position],
                    "is_weekday_token": token_position == record["weekday_position"],
                    "prompt": record["prompt"],
                }
            )

    rows.sort(key=lambda row: (int(row["layer"]), int(row["layer_rank"])))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "input": str(args.input),
        "output": str(args.output),
        "normalization": "within-layer upper-tail rank/percentile",
        "fraction_per_layer": args.fraction,
        "percentage_per_layer": args.fraction * 100,
        "values_per_layer": values_per_layer,
        "target_selected_per_layer": keep_per_layer,
        "tie_policy": "include every activation equal to the per-layer cutoff",
        "selected_total": len(rows),
        "unique_layer_neurons": len(
            {(row["layer"], row["neuron"]) for row in rows}
        ),
        "unique_prompts": len({row["sample_id"] for row in rows}),
        "weekday_token_count": sum(bool(row["is_weekday_token"]) for row in rows),
        "layer_stats": layer_stats,
    }
    summary_path = args.output.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print(
        f"Targeted {keep_per_layer:,} per layer × {num_layers} layers; "
        f"selected {len(rows):,} activations after including cutoff ties"
    )
    print("layer  cutoff      z(cutoff)  max")
    for stats in layer_stats:
        print(
            f"L{stats['layer']:02d}  {stats['top_fraction_cutoff']:>10.6g}  "
            f"{stats['cutoff_z_score']:>9.3f}  {stats['maximum']:>10.6g}"
        )
    print(f"Saved entries: {args.output.resolve()}")
    print(f"Saved summary: {summary_path.resolve()}")


if __name__ == "__main__":
    main()
