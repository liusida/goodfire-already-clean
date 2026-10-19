from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from gb10_load_llm import load_model_to_cuda
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from measure_mlp_distributions import (
    DATASETS,
    MODEL_ID,
    MODEL_REVISION,
    make_batch,
    parse_layers,
    sample_sequences,
)


class LastLayerCaptured(Exception):
    """Stop after measuring the highest requested MLP layer."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure each MLP neuron's strongest positive and negative "
            "post-SwiGLU activation over English and/or Chinese web text."
        )
    )
    parser.add_argument(
        "--dataset",
        choices=("en", "cn", "both"),
        default="both",
        help=(
            "en=OpenWebText, cn=BAAI CCI3-Data, or both. --num-tokens is "
            "sampled independently from each selected corpus."
        ),
    )
    parser.add_argument(
        "--layers",
        default="0",
        help='Layer selection: "all", one layer, or a list such as "[0,1,2]".',
    )
    parser.add_argument("--num-tokens", type=int, default=1_000_000)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle-buffer", type=int, default=10_000)
    parser.add_argument("--bins", type=int, default=100)
    parser.add_argument(
        "--highlight-neurons",
        default="0",
        help='Comma-separated neuron indices to mark, for example "0,552".',
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/mlp_measurement"),
    )
    return parser.parse_args()


def parse_highlights(spec: str, size: int) -> list[int]:
    values = sorted({int(part.strip()) for part in spec.split(",") if part.strip()})
    invalid = [value for value in values if value < 0 or value >= size]
    if invalid:
        raise ValueError(f"Highlight neurons outside 0..{size - 1}: {invalid}")
    return values


def update_extremes(
    maxima: torch.Tensor,
    minima: torch.Tensor,
    activations: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    values = activations[mask].float()
    torch.maximum(maxima, values.amax(dim=0).cpu(), out=maxima)
    torch.minimum(minima, values.amin(dim=0).cpu(), out=minima)


def measure_corpus(
    *,
    model: AutoModelForCausalLM,
    sequences: list[list[int]],
    layers: list[int],
    intermediate_size: int,
    pad_token_id: int,
    batch_size: int,
    num_tokens: int,
    corpus_key: str,
) -> dict[int, dict[str, torch.Tensor]]:
    extrema = {
        layer: {
            "maximum": torch.full((intermediate_size,), float("-inf")),
            "minimum": torch.full((intermediate_size,), float("inf")),
        }
        for layer in layers
    }
    current_mask: torch.Tensor | None = None
    last_layer = layers[-1]

    def make_hook(layer: int):
        def capture(
            _module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]
        ) -> None:
            if current_mask is None:
                raise RuntimeError("Activation hook fired without a mask")
            update_extremes(
                extrema[layer]["maximum"],
                extrema[layer]["minimum"],
                inputs[0],
                current_mask,
            )
            if layer == last_layer:
                raise LastLayerCaptured

        return capture

    handles = [
        model.model.layers[layer].mlp.down_proj.register_forward_pre_hook(
            make_hook(layer)
        )
        for layer in layers
    ]
    progress = tqdm(total=num_tokens, unit="tok", desc=f"{corpus_key} · {layers}")
    try:
        for start in range(0, len(sequences), batch_size):
            batch_sequences = sequences[start : start + batch_size]
            input_ids, attention_mask = make_batch(batch_sequences, pad_token_id)
            current_mask = attention_mask.bool().to(model.device)
            try:
                with torch.inference_mode():
                    model.model(
                        input_ids=input_ids.to(model.device),
                        attention_mask=attention_mask.to(model.device),
                        use_cache=False,
                    )
            except LastLayerCaptured:
                pass
            progress.update(int(attention_mask.sum()))
    finally:
        progress.close()
        for handle in handles:
            handle.remove()

    for layer in layers:
        if not torch.isfinite(extrema[layer]["maximum"]).all():
            raise RuntimeError(f"Layer {layer} has unmeasured positive extrema")
        if not torch.isfinite(extrema[layer]["minimum"]).all():
            raise RuntimeError(f"Layer {layer} has unmeasured negative extrema")
    return extrema


def percentile(values: torch.Tensor, value: float) -> float:
    return float((values <= value).float().mean() * 100)


def summary(
    positive: torch.Tensor,
    negative: torch.Tensor,
    highlights: list[int],
) -> dict[str, object]:
    def describe(values: torch.Tensor) -> dict[str, float | int]:
        return {
            "minimum": float(values.min()),
            "median": float(values.median()),
            "mean": float(values.mean()),
            "p90": float(torch.quantile(values, 0.90)),
            "p99": float(torch.quantile(values, 0.99)),
            "maximum": float(values.max()),
            "zero_count": int(torch.count_nonzero(values == 0)),
        }

    return {
        "positive_peak": describe(positive),
        "negative_peak_magnitude": describe(negative),
        "highlights": {
            str(neuron): {
                "positive_peak": float(positive[neuron]),
                "positive_percentile": percentile(
                    positive, float(positive[neuron])
                ),
                "negative_peak": -float(negative[neuron]),
                "negative_peak_magnitude": float(negative[neuron]),
                "negative_magnitude_percentile": percentile(
                    negative, float(negative[neuron])
                ),
            }
            for neuron in highlights
        },
    }


def plot_extreme_magnitudes(
    positive: torch.Tensor,
    negative: torch.Tensor,
    *,
    highlights: list[int],
    title: str,
    output: Path,
    bins: int,
) -> None:
    positive_np = positive.numpy()
    negative_np = negative.numpy()
    upper = max(float(positive_np.max()), float(negative_np.max()))
    if upper <= 0:
        raise RuntimeError("No positive neuron-extreme magnitudes to plot")
    edges = np.linspace(0.0, upper, bins + 1)
    fig, ax = plt.subplots(figsize=(9.2, 5.2), constrained_layout=True)
    ax.hist(
        positive_np,
        bins=edges,
        alpha=0.58,
        color="#E45756",
        label="maximum positive activation",
    )
    ax.hist(
        negative_np,
        bins=edges,
        alpha=0.58,
        color="#4C78A8",
        label="maximum negative magnitude",
    )
    for index, neuron in enumerate(highlights):
        pos = float(positive[neuron])
        neg = float(negative[neuron])
        if pos > 0:
            ax.axvline(
                pos,
                color="#B22222",
                linestyle="--",
                linewidth=1.4,
                label=f"N{neuron} positive" if index == 0 else None,
            )
        if neg > 0:
            ax.axvline(
                neg,
                color="#2458A6",
                linestyle=":",
                linewidth=1.6,
                label=f"N{neuron} negative" if index == 0 else None,
            )
    ax.set_yscale("log")
    ax.set_xlabel("Per-neuron extreme activation magnitude")
    ax.set_ylabel("Neurons (log count)")
    ax.set_title(title)
    ax.grid(alpha=0.2)
    ax.legend()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_signed_extremes(
    maximum: torch.Tensor,
    minimum: torch.Tensor,
    *,
    highlights: list[int],
    title: str,
    output: Path,
    bins: int,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    panels = (
        (axes[0], maximum.numpy(), "Maximum activation", "#E45756"),
        (axes[1], minimum.numpy(), "Minimum activation", "#4C78A8"),
    )
    for ax, values, label, color in panels:
        ax.hist(values, bins=bins, color=color, alpha=0.82)
        for neuron in highlights:
            ax.axvline(
                float(maximum[neuron] if label.startswith("Maximum") else minimum[neuron]),
                color="black",
                linestyle="--",
                linewidth=1.2,
            )
        ax.set_xlabel(label)
        ax.set_ylabel("Neurons")
        ax.set_yscale("log")
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle(title)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def save_results(
    *,
    output_dir: Path,
    maximum: torch.Tensor,
    minimum: torch.Tensor,
    highlights: list[int],
    layer: int,
    corpus_key: str,
    corpus_label: str,
    sampled_tokens: int,
    bins: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    positive = maximum.clamp_min(0)
    negative = (-minimum).clamp_min(0)
    result_summary = summary(positive, negative, highlights)
    title = (
        f"Qwen3.5-0.8B-Base · Layer {layer} · all {len(maximum):,} neurons\n"
        f"Per-neuron corpus extremes · {corpus_label}"
    )
    plot_extreme_magnitudes(
        positive,
        negative,
        highlights=highlights,
        title=title,
        output=output_dir / "extreme_magnitude_distribution.png",
        bins=bins,
    )
    plot_signed_extremes(
        maximum,
        minimum,
        highlights=highlights,
        title=title,
        output=output_dir / "signed_extremes_distribution.png",
        bins=bins,
    )
    torch.save(
        {
            "maximum": maximum,
            "minimum": minimum,
            "positive_peak": positive,
            "negative_peak_magnitude": negative,
        },
        output_dir / "neuron_extremes.pt",
    )
    details = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "layer": layer,
        "dataset_key": corpus_key,
        "dataset_label": corpus_label,
        "sampled_tokens": sampled_tokens,
        "neuron_count": len(maximum),
        "activation_definition": "input to mlp.down_proj (post-SwiGLU)",
        "summary": result_summary,
    }
    (output_dir / "details.json").write_text(
        json.dumps(details, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    if args.num_tokens <= 0 or args.bins <= 0:
        raise ValueError("--num-tokens and --bins must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    corpus_keys = ("en", "cn") if args.dataset == "both" else (args.dataset,)
    sampled: dict[str, list[list[int]]] = {}
    for corpus_key in corpus_keys:
        corpus = DATASETS[corpus_key]
        print(f"Sampling {args.num_tokens:,} tokens from {corpus['label']}")
        sampled[corpus_key] = sample_sequences(
            tokenizer,
            dataset_id=corpus["dataset_id"],
            text_field=corpus["text_field"],
            dataset_label=corpus["label"],
            num_tokens=args.num_tokens,
            max_length=args.max_length,
            seed=args.seed,
            shuffle_buffer=args.shuffle_buffer,
        )

    model, touched = load_model_to_cuda(
        AutoModelForCausalLM,
        MODEL_ID,
        revision=MODEL_REVISION,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        touch="auto",
        return_touched=True,
    )
    model.eval()
    layers = parse_layers(args.layers, len(model.model.layers))
    intermediate_size = int(model.config.intermediate_size)
    highlights = parse_highlights(args.highlight_neurons, intermediate_size)
    corpus_extrema: dict[str, dict[int, dict[str, torch.Tensor]]] = {}

    for corpus_key in corpus_keys:
        corpus_extrema[corpus_key] = measure_corpus(
            model=model,
            sequences=sampled[corpus_key],
            layers=layers,
            intermediate_size=intermediate_size,
            pad_token_id=tokenizer.pad_token_id,
            batch_size=args.batch_size,
            num_tokens=args.num_tokens,
            corpus_key=corpus_key,
        )

    model_dir = args.output_root / MODEL_ID.rsplit("/", 1)[-1]
    for layer in layers:
        for corpus_key in corpus_keys:
            values = corpus_extrema[corpus_key][layer]
            save_results(
                output_dir=(
                    model_dir
                    / f"layer_{layer:02d}"
                    / "neuron_extremes"
                    / corpus_key
                ),
                maximum=values["maximum"],
                minimum=values["minimum"],
                highlights=highlights,
                layer=layer,
                corpus_key=corpus_key,
                corpus_label=DATASETS[corpus_key]["label"],
                sampled_tokens=args.num_tokens,
                bins=args.bins,
            )

        if len(corpus_keys) == 2:
            maximum = torch.maximum(
                corpus_extrema["en"][layer]["maximum"],
                corpus_extrema["cn"][layer]["maximum"],
            )
            minimum = torch.minimum(
                corpus_extrema["en"][layer]["minimum"],
                corpus_extrema["cn"][layer]["minimum"],
            )
            save_results(
                output_dir=(
                    model_dir / f"layer_{layer:02d}" / "neuron_extremes" / "both"
                ),
                maximum=maximum,
                minimum=minimum,
                highlights=highlights,
                layer=layer,
                corpus_key="both",
                corpus_label="OpenWebText + CCI3-Data",
                sampled_tokens=args.num_tokens * 2,
                bins=args.bins,
            )

    metadata = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "dataset": args.dataset,
        "dataset_ids": [DATASETS[key]["dataset_id"] for key in corpus_keys],
        "tokens_per_corpus": args.num_tokens,
        "total_tokens": args.num_tokens * len(corpus_keys),
        "layers": layers,
        "neuron_count": intermediate_size,
        "highlight_neurons": highlights,
        "seed": args.seed,
        "shuffle_buffer": args.shuffle_buffer,
        "cpu_pages_touched_gib": touched / 1024**3,
    }
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "neuron_extremes_run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved results: {model_dir.resolve()}")


if __name__ == "__main__":
    main()
