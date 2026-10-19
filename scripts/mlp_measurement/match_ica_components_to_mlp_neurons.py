from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from gb10_load_llm import load_model_to_cuda
from icalens import ICALens
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Match every ICA writing direction to the MLP down-projection column "
            "with maximum absolute cosine similarity."
        )
    )
    parser.add_argument(
        "--lens", default="sida/icalens-qwen3.5-9b-base-pile10k"
    )
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument(
        "--mlp-layers",
        default=None,
        help=(
            "MLP layers to search, e.g. '20', '0-20', or '0,3,7-10'. "
            "Defaults to the ICA layer only."
        ),
    )
    parser.add_argument("--component-batch-size", type=int, default=256)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/icalens_direction_projection/layer_20/"
            "ica_to_mlp_nearest_neighbors"
        ),
    )
    return parser.parse_args()


def parse_layers(specification: str | None, *, default: int) -> list[int]:
    if specification is None:
        return [default]
    layers: set[int] = set()
    for piece in specification.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if "-" in piece:
            left, right = piece.split("-", 1)
            start, stop = int(left), int(right)
            if stop < start:
                raise ValueError(f"Invalid descending layer range: {piece!r}")
            layers.update(range(start, stop + 1))
        else:
            layers.add(int(piece))
    if not layers or min(layers) < 0:
        raise ValueError("MLP layers must be a nonempty set of nonnegative indices")
    return sorted(layers)


def summarize(values: torch.Tensor) -> dict[str, float | int]:
    return {
        "count": int(values.numel()),
        "mean": float(values.mean()),
        "std": float(values.std(unbiased=False)),
        "minimum": float(values.min()),
        "p05": float(torch.quantile(values, 0.05)),
        "p25": float(torch.quantile(values, 0.25)),
        "median": float(torch.quantile(values, 0.50)),
        "p75": float(torch.quantile(values, 0.75)),
        "p95": float(torch.quantile(values, 0.95)),
        "p99": float(torch.quantile(values, 0.99)),
        "maximum": float(values.max()),
    }


def render_plot(
    path: Path,
    absolute_cosines: torch.Tensor,
    *,
    ica_layer: int,
    mlp_layers: list[int],
) -> None:
    values = absolute_cosines.numpy()
    ordered = np.sort(values)
    cumulative = np.arange(1, len(ordered) + 1, dtype=np.float64) / len(ordered)
    median = float(np.median(values))
    mean = float(np.mean(values))

    fig, (histogram, cdf) = plt.subplots(1, 2, figsize=(11, 4.5))
    histogram.hist(values, bins=70, color="#d1495b", alpha=0.88)
    histogram.axvline(median, color="#202830", linestyle="--", linewidth=1.4)
    histogram.set_xlabel("Nearest |cosine similarity|")
    histogram.set_ylabel("ICA components")
    histogram.set_title(f"Distribution · mean {mean:.4f} · median {median:.4f}")
    histogram.grid(axis="y", alpha=0.18)

    cdf.step(ordered, cumulative, where="post", color="#d1495b", linewidth=2.0)
    cdf.axvline(median, color="#202830", linestyle="--", linewidth=1.4)
    cdf.set_xlabel("Nearest |cosine similarity|")
    cdf.set_ylabel("Cumulative fraction of ICA components")
    cdf.set_title("Cumulative distribution")
    cdf.grid(alpha=0.18)

    fig.suptitle(
        f"Layer {ica_layer} ICA writing directions matched to "
        f"MLP layers {mlp_layers[0]}–{mlp_layers[-1]}",
        fontsize=15,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)


def render_layer_counts(
    path: Path,
    nearest_layers: torch.Tensor,
    *,
    ica_layer: int,
    mlp_layers: list[int],
) -> dict[int, int]:
    counts = {
        layer: int((nearest_layers == layer).sum()) for layer in mlp_layers
    }
    total = int(nearest_layers.numel())
    figure, axis = plt.subplots(figsize=(10.5, 4.4))
    axis.bar(
        list(counts),
        list(counts.values()),
        color=["#d1495b" if layer == ica_layer else "#5b84b1" for layer in counts],
        width=0.82,
    )
    axis.set_xticks(list(counts))
    axis.set_xlabel("Source MLP layer of nearest neuron")
    axis.set_ylabel("L20 ICA components")
    axis.set_title(
        "Which layer supplies each ICA component's nearest MLP direction?"
    )
    axis.grid(axis="y", alpha=0.18)
    for layer, count in counts.items():
        if count:
            axis.text(
                layer,
                count + max(counts.values()) * 0.015,
                f"{count}\n{count / total:.1%}",
                ha="center",
                va="bottom",
                fontsize=7.5,
            )
    figure.tight_layout()
    figure.savefig(path, dpi=180, facecolor="white")
    plt.close(figure)
    return counts


def main() -> None:
    args = parse_args()
    mlp_layers = parse_layers(args.mlp_layers, default=args.layer)
    if args.layer < 0 or args.component_batch_size <= 0:
        raise ValueError("layer must be nonnegative and batch size positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    lens = ICALens.from_pretrained(args.lens)
    if args.layer not in lens.available_layers:
        raise ValueError(f"Layer {args.layer} unavailable in lens")
    artifact = lens._get_layer(args.layer)
    ica = torch.as_tensor(artifact.writing_matrix, dtype=torch.float32)

    model, _ = load_model_to_cuda(
        AutoModelForCausalLM,
        lens.model_id,
        revision=None if lens.model_revision == "unknown" else lens.model_revision,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        touch="auto",
        return_touched=True,
    )
    model.eval()
    if max(mlp_layers) >= len(model.model.layers):
        raise ValueError(
            f"MLP layer {max(mlp_layers)} outside model range "
            f"0..{len(model.model.layers) - 1}"
        )
    first_mlp = model.model.layers[mlp_layers[0]].mlp.down_proj.weight
    if ica.shape[0] != first_mlp.shape[0]:
        raise RuntimeError(f"Direction mismatch: ICA {ica.shape}, MLP {first_mlp.shape}")

    device = first_mlp.device
    ica_unit = ica.to(device)
    ica_unit /= torch.linalg.vector_norm(ica_unit, dim=0, keepdim=True).clamp_min(1e-12)
    nearest_layers = torch.full((ica.shape[1],), -1, dtype=torch.long)
    nearest_neurons = torch.full((ica.shape[1],), -1, dtype=torch.long)
    signed_cosines = torch.zeros(ica.shape[1], dtype=torch.float32)
    component_starts = list(range(0, ica.shape[1], args.component_batch_size))
    progress = tqdm(
        total=len(mlp_layers) * len(component_starts),
        desc=f"L{args.layer} ICA→MLP layers {mlp_layers[0]}-{mlp_layers[-1]}",
        unit="batch",
    )
    for mlp_layer in mlp_layers:
        mlp = model.model.layers[mlp_layer].mlp.down_proj.weight.detach().float()
        mlp_unit = mlp / torch.linalg.vector_norm(
            mlp, dim=0, keepdim=True
        ).clamp_min(1e-12)
        for start in component_starts:
            stop = min(start + args.component_batch_size, ica.shape[1])
            cosines = torch.matmul(ica_unit[:, start:stop].T, mlp_unit)
            candidate_abs, candidate_neurons = cosines.abs().max(dim=1)
            candidate_signed = cosines.gather(
                1, candidate_neurons[:, None]
            ).squeeze(1)
            current_abs = signed_cosines[start:stop].abs().to(device)
            improved = candidate_abs > current_abs
            if improved.any():
                improved_cpu = improved.cpu()
                layer_slice = nearest_layers[start:stop]
                neuron_slice = nearest_neurons[start:stop]
                cosine_slice = signed_cosines[start:stop]
                layer_slice[improved_cpu] = mlp_layer
                neuron_slice[improved_cpu] = candidate_neurons[improved].cpu()
                cosine_slice[improved_cpu] = candidate_signed[improved].cpu()
            del cosines, candidate_abs, candidate_neurons, candidate_signed
            progress.update(1)
        del mlp, mlp_unit
    progress.close()

    absolute_cosines = signed_cosines.abs().double()
    stats = summarize(absolute_cosines)
    neuron_count_per_layer = int(first_mlp.shape[1])
    combined_ids = nearest_layers * neuron_count_per_layer + nearest_neurons
    unique_pairs, match_counts = torch.unique(
        combined_ids, return_counts=True
    )
    most_reused_index = int(torch.argmax(match_counts))
    most_reused_id = int(unique_pairs[most_reused_index])
    reuse = {
        "unique_nearest_layer_neuron_pairs": int(unique_pairs.numel()),
        "most_reused_layer": most_reused_id // neuron_count_per_layer,
        "most_reused_neuron": most_reused_id % neuron_count_per_layer,
        "most_reused_pair_match_count": int(match_counts[most_reused_index]),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "nearest_mlp_neurons.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "ica_component",
                "nearest_mlp_layer",
                "nearest_mlp_neuron",
                "signed_cosine",
                "absolute_cosine",
            ]
        )
        for component, (layer, neuron, signed) in enumerate(
            zip(
                nearest_layers.tolist(),
                nearest_neurons.tolist(),
                signed_cosines.tolist(),
            )
        ):
            writer.writerow([component, layer, neuron, signed, abs(signed)])

    torch.save(
        {
            "nearest_mlp_layers": nearest_layers,
            "nearest_mlp_neurons": nearest_neurons,
            "signed_cosines": signed_cosines,
            "absolute_cosines": absolute_cosines,
        },
        args.output_dir / "nearest_mlp_neurons.pt",
    )
    metadata = {
        "lens": args.lens,
        "model": lens.model_id,
        "layer": args.layer,
        "mlp_layers": mlp_layers,
        "matching_definition": (
            "argmax over specified MLP layers and neurons of absolute cosine similarity"
        ),
        "ica_direction_definition": "ICALens writing_matrix columns",
        "mlp_direction_definition": "mlp.down_proj.weight columns",
        "ica_components": int(ica.shape[1]),
        "mlp_neurons_per_layer": neuron_count_per_layer,
        "candidate_mlp_directions": neuron_count_per_layer * len(mlp_layers),
        "absolute_cosine_summary": stats,
        **reuse,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    render_plot(
        args.output_dir / "nearest_cosine_distribution.png",
        absolute_cosines,
        ica_layer=args.layer,
        mlp_layers=mlp_layers,
    )
    layer_counts = render_layer_counts(
        args.output_dir / "nearest_source_layer_counts.png",
        nearest_layers,
        ica_layer=args.layer,
        mlp_layers=mlp_layers,
    )
    with (args.output_dir / "nearest_source_layer_counts.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["mlp_layer", "winning_components", "fraction"])
        for layer, count in layer_counts.items():
            writer.writerow([layer, count, count / int(nearest_layers.numel())])

    top_values, top_components = torch.topk(absolute_cosines, 10)
    print(json.dumps(metadata, indent=2), flush=True)
    print("\nTen closest ICA–MLP pairs:")
    for rank, (component, value) in enumerate(
        zip(top_components.tolist(), top_values.tolist()), start=1
    ):
        print(
            f"{rank:2d}. C{component} → "
            f"L{nearest_layers[component].item()}:N{nearest_neurons[component].item()}  "
            f"cos={signed_cosines[component].item():+.6f}  |cos|={value:.6f}"
        )
    print(f"Saved results to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
