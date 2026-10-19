from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from gb10_load_llm import load_model_to_cuda
from transformers import AutoModelForCausalLM

from find_downstream_consumers import projection_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Average nearest downstream-row cosine over every attention output direction."
    )
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B-Base")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--source-layer", type=int, default=0)
    parser.add_argument(
        "--output-root", type=Path, default=Path("outputs/mlp_measurement")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    model, _ = load_model_to_cuda(
        AutoModelForCausalLM,
        args.model,
        revision=args.revision,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        touch="auto",
        return_touched=True,
    )
    model.eval()
    source = model.model.layers[args.source_layer]
    output_projection = (
        source.linear_attn.out_proj if hasattr(source, "linear_attn") else source.self_attn.o_proj
    )
    directions = output_projection.weight.detach().float()
    directions = directions / directions.norm(dim=0, keepdim=True).clamp_min(1e-12)

    rows = []
    all_nearest = []
    with torch.inference_mode():
        for layer, family, name, weight in projection_rows(model, args.source_layer):
            matrix = weight.detach().float()
            matrix = matrix / matrix.norm(dim=1, keepdim=True).clamp_min(1e-12)
            cosine = matrix @ directions
            nearest = cosine.abs().amax(dim=0).cpu()
            all_nearest.append(nearest)
            rows.append(
                {
                    "order": len(rows),
                    "layer": layer,
                    "family": family,
                    "weight": f"L{layer}.{name}",
                    "rows": matrix.shape[0],
                    "direction_count": directions.shape[1],
                    "mean_nearest_absolute_cosine": float(nearest.mean()),
                    "std_nearest_absolute_cosine": float(nearest.std()),
                    "median_nearest_absolute_cosine": float(nearest.median()),
                    "p05_nearest_absolute_cosine": float(torch.quantile(nearest, 0.05)),
                    "p95_nearest_absolute_cosine": float(torch.quantile(nearest, 0.95)),
                }
            )

    model_name = args.model.rsplit("/", 1)[-1]
    output_dir = (
        args.output_root
        / model_name
        / f"layer_{args.source_layer:02d}"
        / "all_attention_output_directions_downstream_consumers"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "average_nearest_consumers.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    torch.save(
        {
            "weights": [row["weight"] for row in rows],
            "nearest_absolute_cosine": torch.stack(all_nearest),
        },
        output_dir / "per_direction_nearest_cosines.pt",
    )

    families = list(dict.fromkeys(row["family"] for row in rows))
    palette = plt.get_cmap("tab10")
    colors = {family: palette(index % 10) for index, family in enumerate(families)}
    x = np.arange(len(rows))
    mean = np.array([row["mean_nearest_absolute_cosine"] for row in rows])
    p05 = np.array([row["p05_nearest_absolute_cosine"] for row in rows])
    p95 = np.array([row["p95_nearest_absolute_cosine"] for row in rows])
    fig, ax = plt.subplots(figsize=(max(18, len(rows) * 0.23), 7))
    ax.vlines(x, p05, p95, color="#cbd5e1", linewidth=1, zorder=1)
    for family in families:
        selected = np.array(
            [index for index, row in enumerate(rows) if row["family"] == family]
        )
        ax.scatter(
            x[selected], mean[selected], s=30, color=colors[family], label=family, zorder=3
        )
    for layer in range(args.source_layer + 1, len(model.model.layers)):
        positions = [row["order"] for row in rows if row["layer"] == layer]
        if positions:
            ax.axvline(min(positions) - 0.5, color="#d0d5dd", linewidth=0.45)
    ax.set_xticks(x)
    ax.set_xticklabels([row["weight"] for row in rows], rotation=90, fontsize=6.5)
    ax.set_ylabel("Mean nearest |cosine| over all $W_O$ directions")
    ax.set_xlabel("Downstream residual-reading weight matrix")
    ax.set_title(
        f"Average downstream alignment of all L{args.source_layer} attention-output directions"
    )
    ax.grid(axis="y", alpha=0.22)
    ax.legend(ncol=min(5, len(families)), fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(output_dir / "average_nearest_consumer_cosines.png", dpi=190)
    plt.close(fig)

    ranked = sorted(
        rows, key=lambda row: row["mean_nearest_absolute_cosine"], reverse=True
    )
    summary = {
        "model": args.model,
        "source_layer": args.source_layer,
        "direction_count": directions.shape[1],
        "definition": "mean_j max_i abs(cos(W[i,:], W_O[:,j]))",
        "matrix_count": len(rows),
        "top_20": ranked[:20],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"Measured {len(rows)} matrices across {directions.shape[1]} directions.\n")
    for row in ranked[:20]:
        print(
            f"{row['weight']:34s} mean={row['mean_nearest_absolute_cosine']:.6f}  "
            f"median={row['median_nearest_absolute_cosine']:.6f}  "
            f"p05–p95={row['p05_nearest_absolute_cosine']:.6f}–{row['p95_nearest_absolute_cosine']:.6f}"
        )
    print(f"\nSaved to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
