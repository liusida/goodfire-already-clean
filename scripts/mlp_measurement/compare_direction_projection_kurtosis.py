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
            "Compare full-vocabulary kurtosis after projecting ICA writing directions, "
            "matched random directions, and MLP down-projection columns through W_unemb."
        )
    )
    parser.add_argument(
        "--lens", default="sida/icalens-qwen3.5-9b-base-pile10k"
    )
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--direction-batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=352)
    parser.add_argument("--bins", type=int, default=80)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/icalens_direction_projection/layer_20/"
            "all_direction_kurtosis_comparison"
        ),
    )
    return parser.parse_args()


def projection_kurtoses(
    unembedding: torch.Tensor,
    directions: torch.Tensor,
    *,
    batch_size: int,
    label: str,
) -> torch.Tensor:
    """Return Pearson kurtosis over vocabulary for every direction.

    Each projection block is converted to float64 before moments are measured. The
    complete vocabulary is used, but vocabulary projections are discarded block by
    block so the output matrix is never retained in memory.
    """
    count = int(directions.shape[1])
    output = torch.empty(count, dtype=torch.float64)
    progress = tqdm(range(0, count, batch_size), desc=label, unit="batch")
    with torch.inference_mode():
        for start in progress:
            stop = min(start + batch_size, count)
            block = directions[:, start:stop].to(
                device=unembedding.device, dtype=unembedding.dtype
            )
            projected = torch.matmul(unembedding, block).double()
            centered = projected - projected.mean(dim=0, keepdim=True)
            variance = centered.square().mean(dim=0)
            fourth = centered.square().square().mean(dim=0)
            kurtosis = fourth / variance.square()
            output[start:stop] = kurtosis.cpu()
            del block, projected, centered, variance, fourth, kurtosis
    return output


def summarize(values: torch.Tensor) -> dict[str, float | int]:
    finite = values[torch.isfinite(values)]
    if finite.numel() != values.numel():
        raise RuntimeError(
            f"Encountered {values.numel() - finite.numel()} non-finite kurtosis values"
        )
    return {
        "count": int(finite.numel()),
        "mean": float(finite.mean()),
        "median": float(finite.median()),
        "std": float(finite.std(unbiased=False)),
        "minimum": float(finite.min()),
        "maximum": float(finite.max()),
        "p05": float(torch.quantile(finite, 0.05)),
        "p95": float(torch.quantile(finite, 0.95)),
    }


def save_csv(path: Path, groups: dict[str, torch.Tensor]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["group", "direction_index", "pearson_kurtosis", "excess_kurtosis"])
        for group, values in groups.items():
            for index, value in enumerate(values.tolist()):
                writer.writerow([group, index, value, value - 3.0])


def plot_histograms(
    path: Path,
    groups: dict[str, torch.Tensor],
    summaries: dict[str, dict[str, float | int]],
    *,
    bins: int,
    layer: int,
) -> None:
    colors = {
        "ICA components": "#d1495b",
        "Random directions": "#4c78a8",
        "MLP neurons": "#59a14f",
        "Attention output": "#b279a2",
    }
    arrays = {name: values.numpy() for name, values in groups.items()}
    lower = min(float(array.min()) for array in arrays.values())
    # The shared main-body window includes at least 90% of every group.
    upper = max(float(np.quantile(array, 0.90)) for array in arrays.values())
    linear_bins = np.linspace(lower, upper, bins + 1)

    fig = plt.figure(figsize=(13.5, 8.8))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.25, 1.0], hspace=0.32, wspace=0.24)

    cdf = fig.add_subplot(grid[0, :])
    for name, array in arrays.items():
        ordered = np.sort(array)
        probability = np.arange(1, len(ordered) + 1, dtype=np.float64) / len(ordered)
        cdf.step(
            ordered,
            probability,
            where="post",
            color=colors[name],
            linewidth=2.4,
            label=(
                f"{name}  (median {summaries[name]['median']:.3f}; "
                f"{np.mean(array <= 4.0) * 100:.1f}% at κ≤4)"
            ),
        )
    cdf.axvline(3.0, color="#78838f", linestyle=":", linewidth=1.0)
    cdf.axvline(4.0, color="#9aa3ad", linestyle="--", linewidth=1.0)
    cdf.set_xlim(lower, upper)
    cdf.set_ylim(0.0, 0.92)
    cdf.set_xlabel("Pearson kurtosis κ (linear scale)")
    cdf.set_ylabel("Cumulative fraction of directions")
    cdf.set_title("Main body: where does a typical direction fall?", loc="left", fontsize=13)
    cdf.grid(alpha=0.18)
    cdf.legend(frameon=False, loc="lower right")

    ridges = fig.add_subplot(grid[1, 0])
    offsets = np.arange(len(arrays) - 1, -1, -1, dtype=np.float64)
    for offset, (name, array) in zip(offsets, arrays.items()):
        counts, edges = np.histogram(array, bins=linear_bins)
        normalized = counts / max(float(counts.max()), 1.0) * 0.72
        centers = (edges[:-1] + edges[1:]) / 2.0
        ridges.fill_between(
            centers,
            offset,
            offset + normalized,
            step="mid",
            color=colors[name],
            alpha=0.82,
        )
        ridges.plot(centers, offset + normalized, color=colors[name], linewidth=1.0)
        ridges.axhline(offset, color="#cad1d8", linewidth=0.8)
    ridges.axvline(3.0, color="#78838f", linestyle=":", linewidth=1.0)
    ridges.set_xlim(lower, upper)
    ridges.set_ylim(-0.08, float(offsets.max()) + 0.82)
    ridges.set_yticks(offsets + 0.30, list(arrays))
    ridges.set_xlabel("Pearson kurtosis κ (shared main-body range)")
    ridges.set_title("Histogram shapes (each peak normalized to 1)", loc="left", fontsize=12)
    ridges.grid(axis="x", alpha=0.16)

    quantiles = fig.add_subplot(grid[1, 1])
    group_names = list(arrays)
    y_positions = np.arange(len(group_names))[::-1]
    for y, name in zip(y_positions, group_names):
        array = arrays[name]
        p10, p25, median, p75, p90 = np.quantile(array, [0.10, 0.25, 0.5, 0.75, 0.90])
        quantiles.hlines(y, p10, p90, color=colors[name], linewidth=2.5, alpha=0.55)
        quantiles.hlines(y, p25, p75, color=colors[name], linewidth=9, alpha=0.88)
        quantiles.scatter(median, y, s=62, color="#18212a", zorder=3)
        quantiles.text(
            p90 + 0.04,
            y,
            f"median {median:.3f}",
            va="center",
            fontsize=9,
            color="#33404d",
        )
    quantiles.axvline(3.0, color="#78838f", linestyle=":", linewidth=1.0)
    quantiles.set_yticks(y_positions, group_names)
    quantiles.set_xlim(lower, upper + 1.20)
    quantiles.set_xlabel("Pearson kurtosis κ (linear scale)")
    quantiles.set_title("Central 80% (thin), IQR (thick), median ●", loc="left", fontsize=12)
    quantiles.grid(axis="x", alpha=0.18)

    fig.suptitle(
        f"Layer {layer}: main-body kurtosis of direct vocabulary projections",
        fontsize=17,
        y=0.975,
    )
    fig.text(
        0.5,
        0.935,
        r"$W_{unemb}d$ over the full vocabulary · raw unembedding · no final norm",
        ha="center",
        color="#526173",
        fontsize=10.5,
    )
    fig.subplots_adjust(left=0.14, right=0.98, bottom=0.09, top=0.86)
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.layer < 0 or args.direction_batch_size <= 0 or args.bins <= 1:
        raise ValueError("layer must be nonnegative and batch size/bins must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    lens = ICALens.from_pretrained(args.lens)
    if args.layer not in lens.available_layers:
        raise ValueError(f"Layer {args.layer} not in lens layers {lens.available_layers}")
    artifact = lens._get_layer(args.layer)
    ica_directions = torch.as_tensor(artifact.writing_matrix, dtype=torch.float32)

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
    if args.layer >= len(model.model.layers):
        raise ValueError(f"Layer {args.layer} outside model range")
    unembedding = model.get_output_embeddings().weight
    neuron_directions = model.model.layers[args.layer].mlp.down_proj.weight.detach()
    transformer_layer = model.model.layers[args.layer]
    if hasattr(transformer_layer, "self_attn"):
        attention_directions = transformer_layer.self_attn.o_proj.weight.detach()
        attention_projection_name = "self_attn.o_proj.weight"
    elif hasattr(transformer_layer, "linear_attn"):
        attention_directions = transformer_layer.linear_attn.out_proj.weight.detach()
        attention_projection_name = "linear_attn.out_proj.weight"
    else:
        raise RuntimeError(f"Layer {args.layer} has no recognized attention output projection")
    if ica_directions.shape[0] != unembedding.shape[1]:
        raise RuntimeError(
            f"ICA/unembedding mismatch: {ica_directions.shape} vs {unembedding.shape}"
        )
    if neuron_directions.shape[0] != unembedding.shape[1]:
        raise RuntimeError(
            f"MLP/unembedding mismatch: {neuron_directions.shape} vs {unembedding.shape}"
        )
    if attention_directions.shape[0] != unembedding.shape[1]:
        raise RuntimeError(
            f"Attention/unembedding mismatch: {attention_directions.shape} vs "
            f"{unembedding.shape}"
        )

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    random_directions = torch.randn(
        ica_directions.shape,
        generator=generator,
        dtype=torch.float32,
    )
    random_directions /= torch.linalg.vector_norm(
        random_directions, dim=0, keepdim=True
    ).clamp_min(1e-12)

    print(
        f"Projecting over {unembedding.shape[0]:,} vocabulary tokens: "
        f"{ica_directions.shape[1]:,} ICA, {random_directions.shape[1]:,} random, "
        f"{neuron_directions.shape[1]:,} neurons, "
        f"{attention_directions.shape[1]:,} attention-output directions",
        flush=True,
    )
    groups = {
        "ICA components": projection_kurtoses(
            unembedding,
            ica_directions,
            batch_size=args.direction_batch_size,
            label="ICA components",
        ),
        "Random directions": projection_kurtoses(
            unembedding,
            random_directions,
            batch_size=args.direction_batch_size,
            label="Random directions",
        ),
        "MLP neurons": projection_kurtoses(
            unembedding,
            neuron_directions,
            batch_size=args.direction_batch_size,
            label="MLP neurons",
        ),
        "Attention output": projection_kurtoses(
            unembedding,
            attention_directions,
            batch_size=args.direction_batch_size,
            label="Attention output",
        ),
    }
    summaries = {name: summarize(values) for name, values in groups.items()}

    torch.save(groups, args.output_dir / "kurtosis_values.pt")
    save_csv(args.output_dir / "kurtosis_values.csv", groups)
    metadata = {
        "lens": args.lens,
        "model": lens.model_id,
        "model_revision": lens.model_revision,
        "layer": args.layer,
        "projection_definition": "lm_head.weight @ direction",
        "final_normalization": False,
        "vocabulary_size": int(unembedding.shape[0]),
        "hidden_size": int(unembedding.shape[1]),
        "random_seed": args.seed,
        "random_definition": "independent isotropic Gaussian directions, L2 normalized",
        "ica_definition": "ICALens writing_matrix columns",
        "neuron_definition": "mlp.down_proj.weight columns",
        "attention_definition": f"{attention_projection_name} columns",
        "summaries": summaries,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    plot_histograms(
        args.output_dir / "kurtosis_histograms.png",
        groups,
        summaries,
        bins=args.bins,
        layer=args.layer,
    )
    print(json.dumps(summaries, indent=2), flush=True)
    print(f"Saved results to {args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
