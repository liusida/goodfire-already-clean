from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute distribution statistics from a dense neuron score stream."
    )
    parser.add_argument(
        "scan_dir",
        nargs="?",
        type=Path,
        default=Path("outputs/layer0/park_neuron_pile10k"),
    )
    parser.add_argument("--bins", type=int, default=500)
    parser.add_argument("--chunk-values", type=int, default=1_000_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_path = args.scan_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    score_meta = summary["scores"]
    scores_path = args.scan_dir / score_meta["path"]
    expected_count = int(score_meta["shape"][0])
    actual_count = scores_path.stat().st_size // np.dtype("<f4").itemsize
    if actual_count != expected_count:
        raise RuntimeError(
            f"Score size mismatch: metadata={expected_count}, file={actual_count}"
        )
    scores = np.memmap(scores_path, mode="r", dtype="<f4", shape=(actual_count,))

    total = 0.0
    minimum = float("inf")
    maximum = float("-inf")
    for start in range(0, actual_count, args.chunk_values):
        chunk = np.asarray(scores[start : start + args.chunk_values], dtype=np.float64)
        total += float(chunk.sum(dtype=np.float64))
        minimum = min(minimum, float(chunk.min()))
        maximum = max(maximum, float(chunk.max()))
    mean = total / actual_count

    centered_2 = centered_3 = centered_4 = 0.0
    for start in range(0, actual_count, args.chunk_values):
        chunk = np.asarray(scores[start : start + args.chunk_values], dtype=np.float64)
        delta = chunk - mean
        delta2 = delta * delta
        centered_2 += float(delta2.sum(dtype=np.float64))
        centered_3 += float((delta2 * delta).sum(dtype=np.float64))
        centered_4 += float((delta2 * delta2).sum(dtype=np.float64))
    moment_2 = centered_2 / actual_count
    moment_3 = centered_3 / actual_count
    moment_4 = centered_4 / actual_count
    std = moment_2**0.5
    skewness = moment_3 / std**3
    pearson_kurtosis = moment_4 / moment_2**2
    excess_kurtosis = pearson_kurtosis - 3.0

    quantile_levels = np.array(
        [0, 0.0001, 0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999, 0.9999, 1]
    )
    quantile_values = np.quantile(scores, quantile_levels)
    counts, edges = np.histogram(scores, bins=args.bins, range=(minimum, maximum))

    layer = int(summary["layer"])
    neuron = int(summary["neuron"])
    stats = {
        "layer": layer,
        "neuron": neuron,
        "count": actual_count,
        "dtype": "float32",
        "minimum": minimum,
        "maximum": maximum,
        "mean": mean,
        "population_standard_deviation": std,
        "population_skewness": skewness,
        "pearson_kurtosis": pearson_kurtosis,
        "excess_kurtosis": excess_kurtosis,
        "moment_convention": "population central moments; excess kurtosis = Pearson kurtosis - 3",
        "quantiles": {
            f"{level:.4%}": float(value)
            for level, value in zip(quantile_levels, quantile_values, strict=True)
        },
        "histogram": {
            "bin_edges": edges.tolist(),
            "counts": counts.tolist(),
        },
    }
    stats_path = args.scan_dir / "distribution.json"
    stats_path.write_text(
        json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    centers = (edges[:-1] + edges[1:]) / 2
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), dpi=170)
    axes[0].bar(centers, counts, width=np.diff(edges), color="#28627b")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Post-SwiGLU activation")
    axes[0].set_ylabel("Token positions (log scale)")
    axes[0].set_title(
        f"Qwen3-32B L{layer}/N{neuron}: all {actual_count:,} Pile-10k scores"
    )
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].spines[["top", "right"]].set_visible(False)

    tail_mask = centers >= max(0.0, float(np.quantile(scores, 0.99)))
    axes[1].bar(
        centers[tail_mask],
        counts[tail_mask],
        width=np.diff(edges)[tail_mask],
        color="#d9480f",
    )
    axes[1].set_yscale("log")
    axes[1].axvline(8.0, color="#263238", linestyle="--", linewidth=1, label="threshold 8")
    axes[1].set_xlabel("Post-SwiGLU activation")
    axes[1].set_ylabel("Token positions (log scale)")
    axes[1].set_title("Upper tail (from the 99th percentile)")
    axes[1].legend(frameon=False)
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    plot_path = args.scan_dir / "score_distribution.png"
    fig.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Count: {actual_count:,}")
    print(f"Range: [{minimum:.9g}, {maximum:.9g}]")
    print(f"Mean: {mean:.9g}")
    print(f"Population standard deviation: {std:.9g}")
    print(f"Population skewness: {skewness:.9g}")
    print(f"Pearson kurtosis: {pearson_kurtosis:.9g}")
    print(f"Excess kurtosis: {excess_kurtosis:.9g}")
    print(f"Saved {stats_path.resolve()}")
    print(f"Saved {plot_path.resolve()}")


if __name__ == "__main__":
    main()
