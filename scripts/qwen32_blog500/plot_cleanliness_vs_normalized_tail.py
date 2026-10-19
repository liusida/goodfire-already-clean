#!/usr/bin/env python3
"""Plot blog cleanliness metadata against three activation-shape metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "metrics",
        nargs="?",
        type=Path,
        default=Path("outputs/blog500_pile1m/neuron_metrics.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "outputs/blog500_pile1m/cleanliness_vs_normalized_zero_tail.png"
        ),
    )
    parser.add_argument(
        "--kurtosis-output",
        type=Path,
        default=Path("outputs/blog500_pile1m/cleanliness_vs_excess_kurtosis.png"),
    )
    parser.add_argument(
        "--absolute-skewness-output",
        type=Path,
        default=Path("outputs/blog500_pile1m/cleanliness_vs_absolute_skewness.png"),
    )
    return parser.parse_args()


def plot_metric(
    cleanliness: np.ndarray,
    values: np.ndarray,
    *,
    title: str,
    ylabel: str,
    median_label: str,
    output: Path,
) -> None:
    if not np.all(np.isfinite(cleanliness)) or not np.all(np.isfinite(values)):
        raise ValueError("Plot inputs contain non-finite values")
    if np.any(values <= 0):
        raise ValueError("A logarithmic y-axis requires positive metric values")

    # Jitter affects display only; the underlying cleanliness scores remain unchanged.
    rng = np.random.default_rng(0)
    display_x = cleanliness + rng.uniform(-0.16, 0.16, size=len(cleanliness))

    fig, ax = plt.subplots(figsize=(9.2, 6.2), constrained_layout=True)
    ax.scatter(
        display_x,
        values,
        s=27,
        alpha=0.58,
        color="#2b718e",
        edgecolors="white",
        linewidths=0.35,
        rasterized=True,
    )

    scores = np.arange(int(cleanliness.min()), int(cleanliness.max()) + 1)
    medians = np.asarray(
        [np.median(values[cleanliness == score]) for score in scores]
    )
    ax.plot(
        scores,
        medians,
        color="#b54720",
        marker="o",
        markersize=4.5,
        linewidth=1.7,
        label=median_label,
        zorder=4,
    )

    ax.set_yscale("log")
    ax.set_xticks(scores)
    ax.set_xlim(scores[0] - 0.45, scores[-1] + 0.45)
    ax.set_xlabel("Goodfire metadata cleanliness score")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.text(
        0.01,
        0.985,
        "500 blog-selected neurons · horizontal jitter for visibility only",
        transform=ax.transAxes,
        ha="left",
        va="top",
        color="#66727d",
        fontsize=9.5,
    )
    ax.grid(axis="y", which="both", alpha=0.22)
    ax.grid(axis="x", alpha=0.12)
    ax.legend(loc="lower right", frameon=False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)
    print(f"Wrote {output.resolve()}")


def main() -> None:
    args = parse_args()
    rows = json.loads(args.metrics.read_text())
    if len(rows) != 500:
        raise ValueError(f"Expected 500 neurons, found {len(rows)}")

    cleanliness = np.asarray([float(row["cleanliness_score"]) for row in rows])
    plots = [
        (
            np.asarray(
                [float(row["normalized_zero_anchored_tail_score"]) for row in rows]
            ),
            "Cleanliness vs. normalized zero-anchored tail score",
            "Normalized zero-anchored tail score (log scale)",
            "Median tail score",
            args.output,
        ),
        (
            np.asarray([float(row["excess_kurtosis"]) for row in rows]),
            "Cleanliness vs. excess kurtosis",
            "Excess kurtosis (log scale)",
            "Median excess kurtosis",
            args.kurtosis_output,
        ),
        (
            np.asarray([abs(float(row["population_skewness"])) for row in rows]),
            "Cleanliness vs. |skewness|",
            "Absolute population skewness (log scale)",
            "Median |skewness|",
            args.absolute_skewness_output,
        ),
    ]
    for values, title, ylabel, median_label, output in plots:
        plot_metric(
            cleanliness,
            values,
            title=title,
            ylabel=ylabel,
            median_label=median_label,
            output=output,
        )
    print(f"Neurons: {len(rows)}")


if __name__ == "__main__":
    main()
