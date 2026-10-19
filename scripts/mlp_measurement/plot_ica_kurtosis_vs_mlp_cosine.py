from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.corrcoef(left, right)[0, 1])


def ordinal_ranks(values: np.ndarray) -> np.ndarray:
    # Values here are continuous, so exact ties are not expected.
    order = np.argsort(values, kind="stable")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scatter ICA vocabulary-projection kurtosis against the absolute cosine "
            "to each component's nearest MLP neuron direction."
        )
    )
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument(
        "--projection-dir",
        type=Path,
        default=Path(
            "outputs/icalens_direction_projection/layer_20/"
            "all_direction_kurtosis_comparison"
        ),
    )
    parser.add_argument(
        "--matching-dir",
        type=Path,
        default=Path(
            "outputs/icalens_direction_projection/layer_20/"
            "ica_to_mlp_nearest_neighbors"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    kurtosis_groups = torch.load(
        args.projection_dir / "kurtosis_values.pt",
        map_location="cpu",
        weights_only=True,
    )
    matches = torch.load(
        args.matching_dir / "nearest_mlp_neurons.pt",
        map_location="cpu",
        weights_only=True,
    )
    kurtosis = kurtosis_groups["ICA components"].double().numpy()
    cosine = matches["absolute_cosines"].double().numpy()
    neurons = matches["nearest_mlp_neurons"].long().numpy()
    signed_cosine = matches["signed_cosines"].double().numpy()
    if kurtosis.shape != cosine.shape:
        raise RuntimeError(
            f"ICA component mismatch: kurtosis {kurtosis.shape}, cosine {cosine.shape}"
        )

    pearson = correlation(cosine, kurtosis)
    spearman = correlation(ordinal_ranks(cosine), ordinal_ranks(kurtosis))
    log_pearson = correlation(cosine, np.log(kurtosis))
    cutoff = float(np.quantile(kurtosis, 0.99))
    clipped_count = int(np.sum(kurtosis > cutoff))

    figure = plt.figure(figsize=(12.5, 5.6))
    grid = figure.add_gridspec(1, 2, width_ratios=[2.15, 1.0], wspace=0.24)
    main = figure.add_subplot(grid[0, 0])
    full = figure.add_subplot(grid[0, 1])

    central = kurtosis <= cutoff
    main.scatter(
        cosine[central],
        kurtosis[central],
        s=13,
        alpha=0.38,
        color="#d1495b",
        linewidths=0,
        rasterized=True,
    )
    main.scatter(
        cosine[~central],
        np.full(clipped_count, cutoff),
        marker="^",
        s=28,
        alpha=0.7,
        color="#8e2635",
        linewidths=0,
        label=f"top 1% shown at boundary (n={clipped_count})",
    )
    main.set_ylim(float(kurtosis.min()) - 0.05, cutoff * 1.025)
    main.set_xlabel("Nearest MLP neuron |cosine similarity|")
    main.set_ylabel("ICA vocabulary-projection Pearson kurtosis")
    main.set_title("Main body (99% of components)", loc="left")
    main.grid(alpha=0.18)
    main.legend(frameon=False, loc="upper left")
    main.text(
        0.985,
        0.04,
        f"Pearson r = {pearson:+.3f}\n"
        f"Spearman ρ = {spearman:+.3f}",
        transform=main.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        color="#344250",
        bbox={"facecolor": "white", "edgecolor": "#d9e0e7", "alpha": 0.9},
    )

    full.scatter(
        cosine,
        kurtosis,
        s=11,
        alpha=0.34,
        color="#d1495b",
        linewidths=0,
        rasterized=True,
    )
    full.set_yscale("log")
    full.set_xlabel("Nearest |cosine|")
    full.set_ylabel("Pearson kurtosis (log scale)")
    full.set_title("Full range", loc="left")
    full.grid(alpha=0.18, which="both")

    figure.suptitle(
        f"Layer {args.layer}: ICA component kurtosis vs nearest MLP-neuron alignment",
        fontsize=15.5,
    )
    figure.text(
        0.5,
        0.925,
        "Each point is one ICA writing direction; cosine sign ignored because ICA orientation is arbitrary",
        ha="center",
        fontsize=10,
        color="#596877",
    )
    figure.subplots_adjust(left=0.08, right=0.98, bottom=0.12, top=0.84)

    output_path = args.matching_dir / "component_kurtosis_vs_nearest_mlp_cosine.png"
    figure.savefig(output_path, dpi=190, facecolor="white")
    plt.close(figure)

    with (args.matching_dir / "component_kurtosis_vs_nearest_mlp_cosine.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "ica_component",
                "vocabulary_projection_kurtosis",
                "nearest_mlp_neuron",
                "signed_cosine",
                "absolute_cosine",
            ]
        )
        for component in range(len(kurtosis)):
            writer.writerow(
                [
                    component,
                    kurtosis[component],
                    neurons[component],
                    signed_cosine[component],
                    cosine[component],
                ]
            )

    statistics = {
        "count": len(kurtosis),
        "pearson_r": pearson,
        "spearman_rho": spearman,
        "pearson_r_with_log_kurtosis": log_pearson,
        "main_body_kurtosis_cutoff_p99": cutoff,
    }
    (args.matching_dir / "component_kurtosis_vs_nearest_mlp_cosine.json").write_text(
        json.dumps(statistics, indent=2), encoding="utf-8"
    )
    print(json.dumps(statistics, indent=2))
    print(f"Saved plot to {output_path.resolve()}")


if __name__ == "__main__":
    main()
