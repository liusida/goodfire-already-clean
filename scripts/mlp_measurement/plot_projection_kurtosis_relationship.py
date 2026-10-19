from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scatter input-activation kurtosis against direct vocabulary-"
            "projection kurtosis and fit an ordinary linear regression."
        )
    )
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B-Base")
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--dataset", choices=("en", "cn", "both"), default="en")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/mlp_measurement"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_name = args.model.rstrip("/").rsplit("/", 1)[-1]
    output_dir = (
        args.output_root
        / model_name
        / f"layer_{args.layer:02d}"
        / "down_projection_kurtosis_extremes"
        / args.dataset
    )
    tensor_path = output_dir / "projections.pt"
    results_path = output_dir / "results.json"
    if not tensor_path.exists() or not results_path.exists():
        raise FileNotFoundError(
            f"Missing projection results beneath {output_dir}; run "
            "project_down_directions_to_vocab.py first"
        )

    payload = torch.load(tensor_path, map_location="cpu", weights_only=True)
    projections = payload["vocabulary_projections"].double()
    metadata = json.loads(results_path.read_text(encoding="utf-8"))
    rows = metadata["neurons"]
    if len(rows) != len(projections):
        raise RuntimeError("Projection tensor and metadata neuron counts differ")

    centered = projections - projections.mean(dim=1, keepdim=True)
    variance = centered.square().mean(dim=1)
    projection_kurtosis = centered.pow(4).mean(dim=1) / variance.square()
    input_kurtosis = np.asarray(
        [float(row["input_activation_kurtosis"]) for row in rows],
        dtype=np.float64,
    )
    output_kurtosis = projection_kurtosis.numpy()
    neurons = [int(row["neuron"]) for row in rows]

    slope, intercept = np.polyfit(input_kurtosis, output_kurtosis, 1)
    correlation = float(np.corrcoef(input_kurtosis, output_kurtosis)[0, 1])
    x_line = np.linspace(input_kurtosis.min(), input_kurtosis.max(), 300)

    colors = ["#4c78a8" if row["tail"] == "lowest" else "#e07b39" for row in rows]
    fig, ax = plt.subplots(figsize=(8.5, 6.2), constrained_layout=True)
    ax.scatter(
        input_kurtosis,
        output_kurtosis,
        c=colors,
        s=54,
        edgecolor="white",
        linewidth=0.7,
        zorder=3,
    )
    ax.plot(
        x_line,
        slope * x_line + intercept,
        color="#343a40",
        linewidth=1.5,
        label=f"linear fit: y = {slope:.3g}x + {intercept:.3g}",
    )
    for neuron, x_value, y_value in zip(neurons, input_kurtosis, output_kurtosis):
        ax.annotate(
            f"N{neuron}",
            (x_value, y_value),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
            color="#46515e",
        )
    ax.set_xlabel("Input activation kurtosis (OpenWebText)")
    ax.set_ylabel("Direct vocabulary-projection kurtosis")
    ax.set_title(
        f"{model_name} · Layer {args.layer}\n"
        "MLP activation kurtosis vs. down-direction logit-lens kurtosis"
    )
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, loc="best")
    ax.text(
        0.99,
        0.02,
        f"Pearson r = {correlation:.3f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
    )
    plot_path = output_dir / "input_vs_projection_kurtosis.png"
    fig.savefig(plot_path, dpi=180, facecolor="white")
    plt.close(fig)

    with (output_dir / "input_vs_projection_kurtosis.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["neuron", "input_kurtosis", "projection_kurtosis"])
        writer.writerows(zip(neurons, input_kurtosis, output_kurtosis))
    (output_dir / "input_vs_projection_kurtosis_fit.json").write_text(
        json.dumps(
            {
                "slope": float(slope),
                "intercept": float(intercept),
                "pearson_r": correlation,
                "point_count": len(neurons),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Linear fit: y = {slope:.8g}x + {intercept:.8g}")
    print(f"Pearson r: {correlation:.6f}")
    print(f"Saved: {plot_path.resolve()}")


if __name__ == "__main__":
    main()
