from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot mean Layer 7 MLP activation per neuron for the prompt."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("outputs/layer7/layer7_mlp_activations.pt"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/layer7/sampled_neurons/mean_activation_distribution.png"),
    )
    parser.add_argument("--bins", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact = torch.load(args.input, map_location="cpu", weights_only=True)
    activations = artifact["activations"][0].float()
    layer = int(artifact["layer_index"])
    target = int(artifact["target_neuron"])
    token_count, neuron_count = activations.shape

    mean_activations = activations.mean(dim=0).numpy()
    target_mean = float(mean_activations[target])
    descending_rank = int(np.sum(mean_activations > target_mean) + 1)
    percentile = float(np.mean(mean_activations <= target_mean) * 100)

    fig, ax = plt.subplots(figsize=(11, 7), constrained_layout=True)
    ax.hist(
        mean_activations,
        bins=args.bins,
        color="#168f8b",
        edgecolor="none",
        alpha=0.9,
    )
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.axvline(target_mean, color="#b33a3a", linewidth=2, linestyle="--")
    ax.annotate(
        f"N{target}: mean {target_mean:.4f}\n"
        f"rank {descending_rank:,} / {neuron_count:,} ({percentile:.3f} percentile)",
        xy=(target_mean, 1),
        xycoords=("data", "axes fraction"),
        xytext=(-8, -10),
        textcoords="offset points",
        ha="right" if target_mean > 0 else "left",
        va="top",
        color="#8f2727",
        fontsize=10,
    )
    ax.set_yscale("log")
    ax.set_xlabel(f"Mean post-SwiGLU activation across {token_count} prompt tokens")
    ax.set_ylabel("Number of neurons (log scale)")
    ax.set_title(
        f"Qwen3-32B Layer {layer}: mean prompt activation of {neuron_count:,} MLP neurons"
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#dddddd", linewidth=0.7, alpha=0.8)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)

    summary = {
        "source": str(args.input),
        "layer": layer,
        "token_count": token_count,
        "neuron_count": neuron_count,
        "statistic": "signed arithmetic mean across token positions",
        "distribution": {
            "minimum": float(mean_activations.min()),
            "maximum": float(mean_activations.max()),
            "mean": float(mean_activations.mean()),
            "median": float(np.median(mean_activations)),
            "standard_deviation": float(mean_activations.std()),
        },
        "target": {
            "neuron": target,
            "mean_activation": target_mean,
            "descending_rank": descending_rank,
            "percentile": percentile,
        },
    }
    summary_path = args.output.with_suffix(".json")
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Saved plot: {args.output.resolve()}")
    print(f"Saved summary: {summary_path.resolve()}")
    print(
        f"N{target}: mean={target_mean:.6f}, rank={descending_rank}/{neuron_count}, "
        f"percentile={percentile:.3f}"
    )


if __name__ == "__main__":
    main()
