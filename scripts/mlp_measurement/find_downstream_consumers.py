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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find downstream residual-reading weight rows aligned with an attention output direction."
    )
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B-Base")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--source-layer", type=int, default=0)
    parser.add_argument("--direction", type=int, default=1825)
    parser.add_argument(
        "--output-root", type=Path, default=Path("outputs/mlp_measurement")
    )
    return parser.parse_args()


def projection_rows(model, source_layer: int):
    for layer_index in range(source_layer, len(model.model.layers)):
        layer = model.model.layers[layer_index]
        # The source direction is created after source-layer attention, so only
        # that layer's MLP and later residual readers can consume it.
        if layer_index > source_layer:
            if hasattr(layer, "linear_attn"):
                attention = layer.linear_attn
                yield layer_index, "attn_qkv", "linear_attn.in_proj_qkv", attention.in_proj_qkv.weight
                yield layer_index, "attn_gate", "linear_attn.in_proj_z", attention.in_proj_z.weight
                yield layer_index, "attn_beta", "linear_attn.in_proj_b", attention.in_proj_b.weight
                yield layer_index, "attn_decay", "linear_attn.in_proj_a", attention.in_proj_a.weight
            elif hasattr(layer, "self_attn"):
                attention = layer.self_attn
                yield layer_index, "attn_q_gate", "self_attn.q_proj", attention.q_proj.weight
                yield layer_index, "attn_k", "self_attn.k_proj", attention.k_proj.weight
                yield layer_index, "attn_v", "self_attn.v_proj", attention.v_proj.weight
            else:
                raise RuntimeError(f"Layer {layer_index} has no recognized attention module")
        yield layer_index, "mlp_gate", "mlp.gate_proj", layer.mlp.gate_proj.weight
        yield layer_index, "mlp_up", "mlp.up_proj", layer.mlp.up_proj.weight


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
    direction = output_projection.weight[:, args.direction].detach().float()
    direction_unit = direction / direction.norm().clamp_min(1e-12)

    rows = []
    with torch.inference_mode():
        for layer, family, name, weight in projection_rows(model, args.source_layer):
            matrix = weight.detach().float()
            cosine = (matrix @ direction_unit) / matrix.norm(dim=1).clamp_min(1e-12)
            row_index = int(cosine.abs().argmax())
            signed = float(cosine[row_index])
            rows.append(
                {
                    "order": len(rows),
                    "layer": layer,
                    "family": family,
                    "weight": f"L{layer}.{name}",
                    "rows": matrix.shape[0],
                    "nearest_row": row_index,
                    "signed_cosine": signed,
                    "absolute_cosine": abs(signed),
                }
            )

    model_name = args.model.rsplit("/", 1)[-1]
    output_dir = (
        args.output_root
        / model_name
        / f"layer_{args.source_layer:02d}"
        / f"attention_output_O{args.direction}_downstream_consumers"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "nearest_consumers.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    families = list(dict.fromkeys(row["family"] for row in rows))
    palette = plt.get_cmap("tab10")
    colors = {family: palette(index % 10) for index, family in enumerate(families)}
    x = np.arange(len(rows))
    y = np.array([row["signed_cosine"] for row in rows])
    fig, ax = plt.subplots(figsize=(max(18, len(rows) * 0.23), 7))
    ax.axhline(0, color="#667085", linewidth=0.8)
    for family in families:
        selected = [index for index, row in enumerate(rows) if row["family"] == family]
        ax.scatter(
            x[selected], y[selected], s=28, color=colors[family], label=family, zorder=3
        )
    for layer in range(args.source_layer + 1, len(model.model.layers)):
        positions = [row["order"] for row in rows if row["layer"] == layer]
        if positions:
            ax.axvline(min(positions) - 0.5, color="#d0d5dd", linewidth=0.45)
    ax.set_xticks(x)
    ax.set_xticklabels([row["weight"] for row in rows], rotation=90, fontsize=6.5)
    ax.set_ylabel("Signed cosine of max-|cosine| row")
    ax.set_xlabel("Downstream residual-reading weight matrix")
    ax.set_title(
        f"Nearest downstream consumers of L{args.source_layer}:O{args.direction}"
    )
    ax.grid(axis="y", alpha=0.22)
    ax.legend(ncol=min(5, len(families)), fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(output_dir / "nearest_consumer_cosines.png", dpi=190)
    plt.close(fig)

    ranked = sorted(rows, key=lambda row: row["absolute_cosine"], reverse=True)
    summary = {
        "model": args.model,
        "source_layer": args.source_layer,
        "source_direction": args.direction,
        "definition": "max_i abs(cos(W[i,:], W_O[:,direction]))",
        "matrix_count": len(rows),
        "top_20": ranked[:20],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Measured {len(rows)} downstream matrices.\n")
    for row in ranked[:20]:
        print(
            f"{row['weight']:34s} row {row['nearest_row']:5d}  "
            f"cos={row['signed_cosine']:+.6f}  |cos|={row['absolute_cosine']:.6f}"
        )
    print(f"\nSaved to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
