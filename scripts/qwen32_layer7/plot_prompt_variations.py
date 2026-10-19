from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from plot_neuron_activity import plot_trace


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot N2026 traces and per-neuron mean distributions for prompt variations."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("outputs/layer7/layer7_prompt_variations.pt"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/layer7/prompt_variation_plots"),
    )
    parser.add_argument("--bins", type=int, default=200)
    return parser.parse_args()


def padded_limits(minimum: float, maximum: float) -> tuple[float, float]:
    span = maximum - minimum
    padding = max(span * 0.05, 1e-6)
    return minimum - padding, maximum + padding


def main() -> None:
    args = parse_args()
    artifact = torch.load(args.input, map_location="cpu", weights_only=True)
    records = artifact["records"]
    layer = int(artifact["layer_index"])
    target = int(artifact["target_neuron"])

    target_traces = [record["target_trace"].float().numpy() for record in records]
    trace_min = min(0.0, min(float(trace.min()) for trace in target_traces))
    trace_max = max(0.0, max(float(trace.max()) for trace in target_traces))
    trace_limits = padded_limits(trace_min, trace_max)

    mean_arrays = [
        record["activations"].float().mean(dim=0).numpy() for record in records
    ]
    distribution_min = min(float(values.min()) for values in mean_arrays)
    distribution_max = max(float(values.max()) for values in mean_arrays)
    distribution_limits = padded_limits(distribution_min, distribution_max)
    bin_edges = np.linspace(distribution_min, distribution_max, args.bins + 1)
    histogram_counts = [np.histogram(values, bins=bin_edges)[0] for values in mean_arrays]
    shared_y_max = max(int(counts.max()) for counts in histogram_counts)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_records = []
    for record, target_trace, mean_values, counts in zip(
        records, target_traces, mean_arrays, histogram_counts, strict=True
    ):
        prompt_id = record["id"]
        trace_path = args.output_dir / f"{prompt_id}_n{target}_activity.png"
        trace_result = plot_trace(
            tokens=list(record["tokens"]),
            activity=target_trace,
            layer=layer,
            neuron=target,
            output=trace_path,
            x_limits=trace_limits,
        )

        target_mean = float(mean_values[target])
        rank = int(np.sum(mean_values > target_mean) + 1)
        percentile = float(np.mean(mean_values <= target_mean) * 100)
        distribution_path = args.output_dir / f"{prompt_id}_mean_distribution.png"
        fig, ax = plt.subplots(figsize=(11, 7), constrained_layout=True)
        widths = np.diff(bin_edges)
        ax.bar(
            bin_edges[:-1],
            counts,
            width=widths,
            align="edge",
            color="#168f8b",
            edgecolor="none",
            alpha=0.9,
        )
        ax.axvline(0, color="#333333", linewidth=0.8)
        ax.axvline(target_mean, color="#b33a3a", linewidth=2, linestyle="--")
        ax.annotate(
            f"N{target}: mean {target_mean:.4f}\n"
            f"rank {rank:,} / {len(mean_values):,} ({percentile:.3f} percentile)",
            xy=(target_mean, 1),
            xycoords=("data", "axes fraction"),
            xytext=(-8 if target_mean > 0 else 8, -10),
            textcoords="offset points",
            ha="right" if target_mean > 0 else "left",
            va="top",
            color="#8f2727",
            fontsize=10,
        )
        ax.set_xlim(*distribution_limits)
        ax.set_ylim(0.7, shared_y_max * 1.35)
        ax.set_yscale("log")
        ax.set_xlabel(
            f"Mean post-SwiGLU activation across {len(record['tokens'])} prompt tokens"
        )
        ax.set_ylabel("Number of neurons (log scale)")
        ax.set_title(
            f"Layer {layer} mean MLP-neuron activation — {prompt_id.replace('_', ' ')}"
        )
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#dddddd", linewidth=0.7, alpha=0.8)
        fig.savefig(distribution_path, dpi=180)
        plt.close(fig)

        manifest_records.append(
            {
                "id": prompt_id,
                "variation": record["variation"],
                "token_count": len(record["tokens"]),
                "next_token": record["next_token"],
                "target_trace_plot": trace_path.name,
                "mean_distribution_plot": distribution_path.name,
                "target_peak": trace_result,
                "target_mean_activation": target_mean,
                "target_descending_rank": rank,
                "target_percentile": percentile,
            }
        )
        print(
            f"{prompt_id}: N{target} mean={target_mean:.4f}, "
            f"rank={rank}/{len(mean_values)}"
        )

    manifest = {
        "source": str(args.input),
        "layer": layer,
        "target_neuron": target,
        "shared_target_trace_x_limits": list(trace_limits),
        "shared_distribution_x_limits": list(distribution_limits),
        "shared_distribution_y_limits": [0.7, shared_y_max * 1.35],
        "records": manifest_records,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Saved 12 plots and manifest to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
