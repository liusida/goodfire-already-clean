from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from plot_neuron_activity import plot_trace


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot prompt-token activity for sampled Layer 7 MLP neurons."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("outputs/layer7/layer7_mlp_activations.pt"),
        help="Tensor artifact written by capture_layer7.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/layer7/sampled_neurons"),
        help="Directory for plots and the sample manifest.",
    )
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact = torch.load(args.input, map_location="cpu", weights_only=True)
    activations = artifact["activations"][0].float().numpy()
    tokens = list(artifact["tokens"])
    layer = int(artifact["layer_index"])
    excluded_neuron = int(artifact["target_neuron"])

    neuron_count = activations.shape[1]
    if not 0 < args.count < neuron_count:
        raise ValueError(f"--count must be between 1 and {neuron_count - 1}")
    candidates = np.delete(np.arange(neuron_count), excluded_neuron)
    rng = np.random.default_rng(args.seed)
    sampled = rng.choice(candidates, size=args.count, replace=False).tolist()

    mean_activations = activations.mean(axis=0)
    top_mean_neurons = np.argsort(mean_activations)[-5:][::-1].tolist()
    stronger_than_target = [
        neuron for neuron in top_mean_neurons if neuron != excluded_neuron
    ]
    compared_neurons = list(
        dict.fromkeys([excluded_neuron, *sampled, *stronger_than_target])
    )
    compared_activity = activations[:, compared_neurons]
    shared_min = min(0.0, float(compared_activity.min()))
    shared_max = max(0.0, float(compared_activity.max()))
    span = shared_max - shared_min
    padding = max(span * 0.05, 1e-6)
    x_limits = (shared_min - padding, shared_max + padding)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    target_output = args.output_dir / f"layer{layer}_n{excluded_neuron}_activity.png"
    target_result = plot_trace(
        tokens=tokens,
        activity=activations[:, excluded_neuron],
        layer=layer,
        neuron=excluded_neuron,
        output=target_output,
        x_limits=x_limits,
    )
    target_result["plot"] = target_output.name
    print(
        f"N{excluded_neuron} (target): peak {target_result['peak_activation']:.4f} "
        f"at {target_result['peak_token']!r} -> {target_output}"
    )
    for neuron in sampled:
        output = args.output_dir / f"layer{layer}_n{neuron}_activity.png"
        result = plot_trace(
            tokens=tokens,
            activity=activations[:, neuron],
            layer=layer,
            neuron=neuron,
            output=output,
            x_limits=x_limits,
        )
        result["plot"] = output.name
        results.append(result)
        print(
            f"N{neuron}: peak {result['peak_activation']:.4f} at "
            f"{result['peak_token']!r} -> {output}"
        )

    top_mean_results = []
    for neuron in stronger_than_target:
        output = args.output_dir / f"layer{layer}_n{neuron}_activity.png"
        result = plot_trace(
            tokens=tokens,
            activity=activations[:, neuron],
            layer=layer,
            neuron=neuron,
            output=output,
            x_limits=x_limits,
        )
        result["mean_activation"] = float(mean_activations[neuron])
        result["plot"] = output.name
        top_mean_results.append(result)
        print(
            f"N{neuron} (higher mean): mean {result['mean_activation']:.4f}, "
            f"peak {result['peak_activation']:.4f} at {result['peak_token']!r} "
            f"-> {output}"
        )

    manifest = {
        "source": str(args.input),
        "layer": layer,
        "excluded_target_neuron": excluded_neuron,
        "seed": args.seed,
        "sampled_neurons": sampled,
        "target_result": target_result,
        "higher_mean_neurons": stronger_than_target,
        "higher_mean_results": top_mean_results,
        "shared_x_limits": list(x_limits),
        "results": results,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Saved manifest: {manifest_path.resolve()}")


if __name__ == "__main__":
    main()
