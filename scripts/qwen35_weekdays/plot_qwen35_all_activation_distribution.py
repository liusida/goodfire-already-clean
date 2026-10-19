from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the distribution of every saved Qwen3.5 MLP activation value."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("outputs/weekdays/qwen35_08b_weekday_activations.pt"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/weekdays/qwen35_08b_weekday_all_activation_distribution.png"),
    )
    parser.add_argument("--bins", type=int, default=400)
    parser.add_argument(
        "--quantile-sample-size",
        type=int,
        default=2_000_000,
        help="Maximum deterministic sample used only to estimate display quantiles.",
    )
    return parser.parse_args()


def deterministic_sample(values: torch.Tensor, target: int) -> torch.Tensor:
    flat = values.reshape(-1)
    if flat.numel() <= target:
        return flat.float()
    step = max(flat.numel() // target, 1)
    return flat[::step][:target].float()


def main() -> None:
    args = parse_args()
    artifact = torch.load(args.input, map_location="cpu", weights_only=False)
    records = artifact["records"]

    count = 0
    value_sum = 0.0
    square_sum = 0.0
    negative_count = 0
    zero_count = 0
    positive_count = 0
    global_min = float("inf")
    global_max = float("-inf")
    samples: list[torch.Tensor] = []
    per_record_target = max(args.quantile_sample_size // len(records), 1)

    for record in records:
        values = record["activations"].float().reshape(-1)
        count += values.numel()
        value_sum += values.double().sum().item()
        square_sum += values.double().square().sum().item()
        negative_count += torch.count_nonzero(values < 0).item()
        zero_count += torch.count_nonzero(values == 0).item()
        positive_count += torch.count_nonzero(values > 0).item()
        global_min = min(global_min, values.min().item())
        global_max = max(global_max, values.max().item())
        samples.append(deterministic_sample(values, per_record_target))

    sampled = torch.cat(samples)
    quantile_levels = torch.tensor(
        [0.0001, 0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999, 0.9999]
    )
    quantile_values = torch.quantile(sampled, quantile_levels).tolist()
    quantiles = {
        f"{level.item():.4f}": value
        for level, value in zip(quantile_levels, quantile_values, strict=True)
    }
    central_min = quantiles["0.0010"]
    central_max = quantiles["0.9990"]

    full_edges = np.linspace(global_min, global_max, args.bins + 1)
    central_edges = np.linspace(central_min, central_max, args.bins + 1)
    full_counts = np.zeros(args.bins, dtype=np.int64)
    central_counts = np.zeros(args.bins, dtype=np.int64)
    below_central = 0
    above_central = 0

    for record in records:
        values = record["activations"].float().reshape(-1).numpy()
        full_counts += np.histogram(values, bins=full_edges)[0]
        central_counts += np.histogram(values, bins=central_edges)[0]
        below_central += int(np.count_nonzero(values < central_min))
        above_central += int(np.count_nonzero(values > central_max))

    mean = value_sum / count
    variance = max(square_sum / count - mean * mean, 0.0)
    std = variance**0.5

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    for ax, edges, counts, title in (
        (axes[0], full_edges, full_counts, "Full observed range"),
        (axes[1], central_edges, central_counts, "Central 99.8% (sample-estimated bounds)"),
    ):
        centers = (edges[:-1] + edges[1:]) / 2
        widths = np.diff(edges)
        ax.bar(centers, counts, width=widths, color="#4C78A8", edgecolor="none")
        ax.set_yscale("log")
        ax.set_xlabel("Post-SwiGLU MLP activation value")
        ax.set_ylabel("Number of activation values (log scale)")
        ax.set_title(title)
        ax.axvline(0, color="#333333", linewidth=0.8, alpha=0.7)
        ax.grid(axis="y", alpha=0.2)

    axes[0].text(
        0.98,
        0.96,
        f"n = {count:,}\nmean = {mean:.5g}\nstd = {std:.5g}",
        transform=axes[0].transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc"},
    )
    axes[1].text(
        0.98,
        0.96,
        f"below = {below_central:,}\nabove = {above_central:,}",
        transform=axes[1].transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc"},
    )
    fig.suptitle(
        "Qwen3.5-0.8B-Base: all MLP activations across 70 weekday prompts",
        fontsize=13,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)

    stats = {
        "input": str(args.input),
        "plot": str(args.output),
        "model_id": artifact["model_id"],
        "activation_definition": artifact["activation_definition"],
        "num_prompts": len(records),
        "num_layers": artifact["num_layers"],
        "intermediate_size": artifact["intermediate_size"],
        "count": count,
        "min": global_min,
        "max": global_max,
        "mean": mean,
        "std": std,
        "negative_count": negative_count,
        "zero_count": zero_count,
        "positive_count": positive_count,
        "quantile_method": f"deterministic sample of {sampled.numel():,} values",
        "quantiles": quantiles,
        "central_histogram_bounds": [central_min, central_max],
        "below_central_bounds": below_central,
        "above_central_bounds": above_central,
        "bins": args.bins,
    }
    stats_path = args.output.with_suffix(".json")
    stats_path.write_text(json.dumps(stats, indent=2) + "\n")

    print(f"Values: {count:,}")
    print(f"Range: [{global_min:.7g}, {global_max:.7g}]")
    print(f"Mean ± std: {mean:.7g} ± {std:.7g}")
    print(f"Negative / zero / positive: {negative_count:,} / {zero_count:,} / {positive_count:,}")
    print(f"Quantiles ({stats['quantile_method']}): {quantiles}")
    print(f"Saved plot: {args.output.resolve()}")
    print(f"Saved stats: {stats_path.resolve()}")


if __name__ == "__main__":
    main()
