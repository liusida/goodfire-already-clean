from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot one captured MLP neuron's activity across prompt tokens."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("outputs/layer7/layer7_mlp_activations.json"),
        help="JSON metadata written by capture_layer7.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/layer7/sampled_neurons/layer7_n2026_activity.png"),
        help="Output image path.",
    )
    return parser.parse_args()


def display_token(token: str) -> str:
    if token == " ":
        return "[space]"
    return token.replace(" ", "·").replace("\n", "↵")


def plot_trace(
    *,
    tokens: list[str],
    activity: np.ndarray,
    layer: int,
    neuron: int,
    output: Path,
    x_limits: tuple[float, float] | None = None,
) -> dict[str, int | float | str]:
    activity = np.asarray(activity, dtype=np.float32)
    if activity.shape != (len(tokens),):
        raise ValueError(
            f"Trace shape {activity.shape} does not match {len(tokens)} tokens"
        )

    positions = np.arange(len(tokens))
    colors = np.where(activity >= 0, "#168f8b", "#b65f42")
    peak_position = int(np.argmax(np.abs(activity)))
    fig, ax = plt.subplots(figsize=(12, 12), constrained_layout=True)
    ax.barh(positions, activity, color=colors, height=0.42)
    ax.axvline(0, color="#252525", linewidth=0.8)
    ax.axhline(peak_position, color="#555555", linewidth=1, linestyle="--", alpha=0.7)
    ax.scatter(
        [activity[peak_position]],
        [peak_position],
        color="#111111",
        s=28,
        zorder=3,
    )
    ax.annotate(
        f"peak: position {peak_position}, {tokens[peak_position]!r}\n"
        f"activation {activity[peak_position]:.3f}",
        xy=(activity[peak_position], peak_position),
        xytext=(activity[peak_position] * 0.67, peak_position + 3),
        arrowprops={"arrowstyle": "->", "color": "#333333"},
        fontsize=10,
    )

    ax.set_yticks(positions)
    ax.set_yticklabels([display_token(token) for token in tokens], fontsize=8)
    ax.set_ylim(-1, len(tokens))
    ax.invert_yaxis()
    ax.set_xlabel("Post-SwiGLU activation")
    ax.set_ylabel("Decoded token")
    if x_limits is not None:
        ax.set_xlim(*x_limits)
    ax.set_title(
        f"Qwen3-32B Layer {layer}, MLP neuron N{neuron} across the medical prompt"
    )
    ax.text(
        0.995,
        0.97,
        "teal: positive   rust: negative",
        transform=ax.transAxes,
        ha="right",
        va="top",
        color="#555555",
        fontsize=9,
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color="#dddddd", linewidth=0.7, alpha=0.8)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return {
        "neuron": neuron,
        "peak_position": peak_position,
        "peak_token": tokens[peak_position],
        "peak_activation": float(activity[peak_position]),
    }


def main() -> None:
    args = parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    result = plot_trace(
        tokens=data["tokens"],
        activity=np.asarray(data["target_trace"], dtype=np.float32),
        layer=int(data["layer_index"]),
        neuron=int(data["target_neuron"]),
        output=args.output,
    )
    print(f"Saved plot: {args.output.resolve()}")
    print(
        f"Peak |activation|: position {result['peak_position']}, "
        f"token {result['peak_token']!r}, value {result['peak_activation']:.4f}"
    )


if __name__ == "__main__":
    main()
