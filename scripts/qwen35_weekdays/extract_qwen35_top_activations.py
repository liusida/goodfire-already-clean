from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract the exact global top fraction of saved MLP activations."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("outputs/weekdays/qwen35_08b_weekday_activations.pt"),
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=0.0001,
        help="Fraction to retain; 0.0001 is the top 0.01%%.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/weekdays/qwen35_08b_weekday_top_0.01pct_activations.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 < args.fraction <= 1:
        raise ValueError("--fraction must be in (0, 1]")

    artifact = torch.load(args.input, map_location="cpu", weights_only=False)
    records = artifact["records"]
    total_count = sum(record["activations"].numel() for record in records)
    keep_count = math.ceil(total_count * args.fraction)

    candidate_values: list[torch.Tensor] = []
    candidate_records: list[torch.Tensor] = []
    candidate_offsets: list[torch.Tensor] = []
    for record_index, record in enumerate(records):
        flat = record["activations"].reshape(-1)
        local_count = min(keep_count, flat.numel())
        values, offsets = torch.topk(flat.float(), local_count, sorted=False)
        candidate_values.append(values)
        candidate_records.append(
            torch.full((local_count,), record_index, dtype=torch.int16)
        )
        candidate_offsets.append(offsets)

    all_values = torch.cat(candidate_values)
    all_records = torch.cat(candidate_records)
    all_offsets = torch.cat(candidate_offsets)
    values, selected = torch.topk(all_values, keep_count, sorted=True)
    selected_records = all_records[selected]
    selected_offsets = all_offsets[selected]

    mlp_width = artifact["intermediate_size"]
    rows: list[dict[str, object]] = []
    for rank, (value, record_index, flat_offset) in enumerate(
        zip(values.tolist(), selected_records.tolist(), selected_offsets.tolist(), strict=True),
        start=1,
    ):
        record = records[record_index]
        seq_len = len(record["input_ids"])
        layer_stride = seq_len * mlp_width
        layer = flat_offset // layer_stride
        within_layer = flat_offset % layer_stride
        token_position = within_layer // mlp_width
        neuron = within_layer % mlp_width
        rows.append(
            {
                "rank": rank,
                "value": value,
                "sample_id": record["sample_id"],
                "template_index": record["template_index"],
                "weekday": record["weekday"],
                "layer": layer,
                "neuron": neuron,
                "token_position": token_position,
                "token": record["tokens"][token_position],
                "is_weekday_token": token_position == record["weekday_position"],
                "prompt": record["prompt"],
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "input": str(args.input),
        "output": str(args.output),
        "fraction": args.fraction,
        "percentage": args.fraction * 100,
        "total_activation_values": total_count,
        "selected_count": keep_count,
        "minimum_selected_value": rows[-1]["value"],
        "maximum_selected_value": rows[0]["value"],
        "unique_layers": len({row["layer"] for row in rows}),
        "unique_neurons_with_layer": len(
            {(row["layer"], row["neuron"]) for row in rows}
        ),
        "unique_prompts": len({row["sample_id"] for row in rows}),
        "weekday_token_count": sum(bool(row["is_weekday_token"]) for row in rows),
    }
    summary_path = args.output.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print(json.dumps(summary, indent=2))
    print("\nTop 20:")
    for row in rows[:20]:
        print(
            f"#{row['rank']:>2} {row['value']:>8.4f}  "
            f"L{row['layer']:02d}/N{row['neuron']:04d}  "
            f"{row['sample_id']}  token={row['token']!r}"
        )


if __name__ == "__main__":
    main()
