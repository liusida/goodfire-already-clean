from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from gb10_load_llm import load_model_to_cuda
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "For every layer's attention W_o and MLP W_down directions, plot "
            "mean nearest-row cosine against every residual-reading matrix."
        )
    )
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B-Base")
    parser.add_argument("--revision", default=None)
    parser.add_argument(
        "--output-root", type=Path, default=Path("outputs/direction_consumptions")
    )
    return parser.parse_args()


def model_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise RuntimeError(f"Unsupported model architecture: {type(model).__name__}")


def model_hidden_size(model) -> int:
    if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        return model.model.embed_tokens.embedding_dim
    if hasattr(model, "transformer") and hasattr(model.transformer, "wte"):
        return model.transformer.wte.embedding_dim
    raise RuntimeError("Could not determine residual-stream width")


def is_gpt2(model) -> bool:
    return hasattr(model, "transformer") and hasattr(model.transformer, "h")


def all_residual_readers(model):
    for layer_index, layer in enumerate(model_layers(model)):
        if is_gpt2(model):
            # GPT-2 Conv1D stores [input, output]. Transpose into
            # [consumer direction, residual input].
            q_weight, k_weight, v_weight = layer.attn.c_attn.weight.T.chunk(3, dim=0)
            yield layer_index, "attn_q", "attn.c_attn.q", q_weight
            yield layer_index, "attn_k", "attn.c_attn.k", k_weight
            yield layer_index, "attn_v", "attn.c_attn.v", v_weight
            yield layer_index, "mlp_expand", "mlp.c_fc", layer.mlp.c_fc.weight.T
            continue
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


def direction_sources(model):
    for layer_index, layer in enumerate(model_layers(model)):
        if is_gpt2(model):
            # Each internal Conv1D input contributes one row of its weight to
            # the residual output. Return [residual, directions].
            yield "attention_wo", layer_index, layer.attn.c_proj.weight.T
            yield "mlp_down", layer_index, layer.mlp.c_proj.weight.T
            continue
        if hasattr(layer, "linear_attn"):
            output_weight = layer.linear_attn.out_proj.weight
        else:
            output_weight = layer.self_attn.o_proj.weight
        yield "attention_wo", layer_index, output_weight
        yield "mlp_down", layer_index, layer.mlp.down_proj.weight


def summarize_consumers(directions: torch.Tensor, consumers: list[dict]) -> list[dict]:
    directions = directions.detach().float()
    directions = directions / directions.norm(dim=0, keepdim=True).clamp_min(1e-12)
    rows = []
    with torch.inference_mode():
        for consumer in consumers:
            cosine = consumer["normalized_weight"] @ directions
            nearest = cosine.abs().amax(dim=0).cpu()
            rows.append(
                {
                    "order": consumer["order"],
                    "layer": consumer["layer"],
                    "family": consumer["family"],
                    "weight": consumer["weight"],
                    "rows": consumer["rows"],
                    "direction_count": directions.shape[1],
                    "mean": float(nearest.mean()),
                    "std": float(nearest.std()),
                    "median": float(nearest.median()),
                    "p05": float(torch.quantile(nearest, 0.05)),
                    "p95": float(torch.quantile(nearest, 0.95)),
                }
            )
    return rows


def save_plot(rows: list[dict], output: Path, title: str) -> None:
    families = list(dict.fromkeys(row["family"] for row in rows))
    palette = plt.get_cmap("tab10")
    colors = {family: palette(index % 10) for index, family in enumerate(families)}
    x = np.arange(len(rows))
    mean = np.array([row["mean"] for row in rows])
    p05 = np.array([row["p05"] for row in rows])
    p95 = np.array([row["p95"] for row in rows])
    fig, ax = plt.subplots(figsize=(max(20, len(rows) * 0.23), 7))
    ax.vlines(x, p05, p95, color="#cbd5e1", linewidth=1, zorder=1)
    for family in families:
        selected = np.array(
            [index for index, row in enumerate(rows) if row["family"] == family]
        )
        ax.scatter(
            x[selected], mean[selected], s=30, color=colors[family], label=family, zorder=3
        )
    layers = sorted({row["layer"] for row in rows})
    for layer in layers[1:]:
        positions = [row["order"] for row in rows if row["layer"] == layer]
        ax.axvline(min(positions) - 0.5, color="#d0d5dd", linewidth=0.45)
    ax.set_xticks(x)
    ax.set_xticklabels([row["weight"] for row in rows], rotation=90, fontsize=6.5)
    ax.set_ylabel("Mean nearest |cosine| over source directions")
    layer_max = max(row["layer"] for row in rows)
    ax.set_xlabel(f"All residual-reading weight matrices (Layers 0–{layer_max})")
    ax.set_title(title)
    # Cosine similarity has a common natural range. Keeping it fixed makes
    # every source layer and both source families directly comparable.
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", alpha=0.22)
    ax.legend(ncol=min(5, len(families)), fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(output, dpi=190)
    plt.close(fig)


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
    hidden_size = model_hidden_size(model)
    consumers = []
    for order, (layer, family, name, weight) in enumerate(all_residual_readers(model)):
        if weight.shape[1] != hidden_size:
            raise RuntimeError(f"{name} does not read residual space: {tuple(weight.shape)}")
        normalized = weight.detach().float()
        normalized = normalized / normalized.norm(dim=1, keepdim=True).clamp_min(1e-12)
        consumers.append(
            {
                "order": order,
                "layer": layer,
                "family": family,
                "weight": f"L{layer}.{name}",
                "rows": weight.shape[0],
                "normalized_weight": normalized,
            }
        )

    model_name = args.model.rsplit("/", 1)[-1]
    output_dir = args.output_root / model_name
    source_list = list(direction_sources(model))
    master_rows = []
    progress = tqdm(source_list, desc="direction sources", unit="matrix")
    for source_type, source_layer, weight in progress:
        progress.set_postfix_str(f"{source_type} L{source_layer}")
        rows = summarize_consumers(weight, consumers)
        source_dir = output_dir / source_type
        source_dir.mkdir(parents=True, exist_ok=True)
        stem = f"layer_{source_layer:02d}"
        title_name = "attention $W_O$" if source_type == "attention_wo" else "MLP $W_{down}$"
        save_plot(
            rows,
            source_dir / f"{stem}.png",
            f"{model_name} · Layer {source_layer} {title_name} directions",
        )
        with (source_dir / f"{stem}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        for row in rows:
            master_rows.append(
                {
                    "source_type": source_type,
                    "source_layer": source_layer,
                    **row,
                }
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "all_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=master_rows[0].keys())
        writer.writeheader()
        writer.writerows(master_rows)
    layer_count = len(model_layers(model))
    summary = {
        "model": args.model,
        "layers": layer_count,
        "consumer_matrices": len(consumers),
        "source_families": {
            "attention_wo": layer_count,
            "mlp_down": layer_count,
        },
        "plot_count": len(source_list),
        "definition": "mean_j max_i abs(cos(consumer_W[i,:], source_D[:,j]))",
        "note": "All residual-reading matrices from all layers are included, including matrices earlier than the source.",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print(f"Saved to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
