from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import NormalDist

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot tail diagnostics for a saved MLP-direction projection smoke run."
    )
    parser.add_argument("scan_dir", type=Path)
    parser.add_argument("--score-key", default="direction_coefficient")
    parser.add_argument("--bins", type=int, default=240)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = json.loads((args.scan_dir / "summary.json").read_text(encoding="utf-8"))
    with np.load(args.scan_dir / "scores.npz") as archive:
        values = archive[args.score_key].astype(np.float64)
    mean = float(values.mean())
    std = float(values.std())
    z = (values - mean) / std
    minimum, maximum = float(z.min()), float(z.max())

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), dpi=170)

    edges = np.linspace(minimum, maximum, args.bins + 1)
    axes[0].hist(z, bins=edges, density=True, color="#397792", alpha=0.85)
    grid = np.linspace(minimum, maximum, 1600)
    gaussian_pdf = np.exp(-(grid**2) / 2) / math.sqrt(2 * math.pi)
    axes[0].plot(grid, gaussian_pdf, color="#b54720", linewidth=2, label="N(0,1)")
    axes[0].set_yscale("log")
    axes[0].set_ylim(bottom=1e-5)
    axes[0].set_title("Full standardized distribution")
    axes[0].set_xlabel("z-score")
    axes[0].set_ylabel("density (log scale)")
    axes[0].legend(frameon=False)

    absolute = np.sort(np.abs(z))
    survival = (len(absolute) - np.arange(len(absolute))) / len(absolute)
    # Keep every point in the sparse tail and thin only the dense center.
    selected = np.unique(
        np.concatenate(
            [np.linspace(0, len(absolute) - 1, 2500, dtype=int), np.arange(max(0, len(absolute) - 1000), len(absolute))]
        )
    )
    tail_grid = np.linspace(0, float(absolute[-1]), 1200)
    gaussian_survival = np.array(
        [math.erfc(value / math.sqrt(2)) for value in tail_grid]
    )
    axes[1].plot(
        absolute[selected], survival[selected], color="#397792", linewidth=1.7, label="observed"
    )
    axes[1].plot(
        tail_grid, gaussian_survival, color="#b54720", linewidth=2, label="Gaussian"
    )
    axes[1].set_yscale("log")
    axes[1].set_ylim(0.5 / len(z), 1)
    axes[1].set_title("Two-sided tail probability")
    axes[1].set_xlabel("absolute z-score")
    axes[1].set_ylabel("P(|Z| ≥ z) · log scale")
    axes[1].legend(frameon=False)

    probabilities = np.linspace(0.0001, 0.9999, 1200)
    observed_quantiles = np.quantile(z, probabilities)
    normal = NormalDist()
    normal_quantiles = np.array([normal.inv_cdf(float(p)) for p in probabilities])
    axes[2].plot(normal_quantiles, observed_quantiles, color="#397792", linewidth=1.8)
    limit = max(abs(normal_quantiles).max(), abs(observed_quantiles).max())
    axes[2].plot([-limit, limit], [-limit, limit], color="#b54720", linestyle="--", label="Gaussian reference")
    axes[2].set_title("Normal Q–Q plot")
    axes[2].set_xlabel("Gaussian theoretical quantile")
    axes[2].set_ylabel("observed quantile")
    axes[2].legend(frameon=False)

    for ax in axes:
        ax.grid(alpha=0.22)
        ax.spines[["top", "right"]].set_visible(False)
    stats = summary["statistics"][args.score_key]
    fig.suptitle(
        f"Qwen3-32B L{summary['layer']}/N{summary['neuron']} projection tails · "
        f"{len(z):,} tokens · skew {stats['population_skewness']:.3f} · "
        f"excess kurtosis {stats['excess_kurtosis']:.3f}"
    )
    fig.tight_layout()
    output_path = args.scan_dir / "projection_tail_diagnostics.png"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Standardized range: [{minimum:.6g}, {maximum:.6g}]")
    for threshold in (3, 4, 5, 6, 8, 10):
        observed = float(np.mean(np.abs(z) >= threshold))
        expected = math.erfc(threshold / math.sqrt(2))
        ratio = observed / expected if expected else float("inf")
        print(
            f"|Z| >= {threshold:>2}: observed={observed:.8g}, Gaussian={expected:.8g}, ratio={ratio:.6g}"
        )
    print(f"Saved {output_path.resolve()}")


if __name__ == "__main__":
    main()
