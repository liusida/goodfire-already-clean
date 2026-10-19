from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


DATA = Path("outputs/weekdays/qwen35_08b_weekday_feature_matrix_values.csv")
SUMMARY = Path("outputs/weekdays/qwen35_08b_weekday_layerwise_top_0.002pct_activations.json")
OUTPUT = Path("outputs/weekdays/qwen35_08b_weekday_L1N2_vs_L4N279.png")
X_FEATURE = (1, 2)
Y_FEATURE = (4, 279)
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
COLORS = {
    "Monday": "#4C78A8",
    "Tuesday": "#F58518",
    "Wednesday": "#54A24B",
    "Thursday": "#E45756",
    "Friday": "#B279A2",
    "Saturday": "#FF9DA6",
    "Sunday": "#9D755D",
}


def main() -> None:
    values: dict[tuple[int, str], dict[tuple[int, int], float]] = defaultdict(dict)
    with DATA.open(newline="") as handle:
        for row in csv.DictReader(handle):
            key = (int(row["template_index"]), row["weekday"])
            feature = (int(row["layer"]), int(row["neuron"]))
            values[key][feature] = float(row["value"])

    summary = json.loads(SUMMARY.read_text())
    cutoffs = {
        int(item["layer"]): float(item["top_fraction_cutoff"])
        for item in summary["layer_stats"]
    }

    fig, ax = plt.subplots(figsize=(16, 12), constrained_layout=True)
    for template_index in range(10):
        for weekday in WEEKDAYS:
            point = values[(template_index, weekday)]
            x = point[X_FEATURE]
            y = point[Y_FEATURE]
            ax.scatter(
                x,
                y,
                s=74,
                marker="o",
                color=COLORS[weekday],
                edgecolor="white",
                linewidth=0.7,
                zorder=3,
            )

    ax.axvline(
        cutoffs[X_FEATURE[0]],
        color="#4C78A8",
        linestyle="--",
        linewidth=1.1,
        alpha=0.7,
        label=f"L1 cutoff = {cutoffs[X_FEATURE[0]]:.5g}",
    )
    ax.axhline(
        cutoffs[Y_FEATURE[0]],
        color="#E45756",
        linestyle="--",
        linewidth=1.1,
        alpha=0.7,
        label=f"L4 cutoff = {cutoffs[Y_FEATURE[0]]:.5g}",
    )
    ax.set_xlabel("L1:N2 activation at weekday token", fontsize=12)
    ax.set_ylabel("L4:N279 activation at weekday token", fontsize=12)
    ax.set_title(
        "Weekday-token feature space: L1:N2 vs L4:N279\n"
        "70 points = 10 templates × 7 weekdays",
        fontsize=15,
    )
    ax.grid(True, color="#d9dee4", linewidth=0.7, alpha=0.7)
    ax.set_axisbelow(True)

    weekday_legend = [
        Line2D(
            [0], [0], marker="o", color="none", markerfacecolor=COLORS[day],
            markeredgecolor="white", markersize=8, label=day
        )
        for day in WEEKDAYS
    ]
    ax.legend(
        handles=weekday_legend,
        title="Weekday",
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
        frameon=True,
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Points: {len(values)}")
    print(f"Saved plot: {OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
