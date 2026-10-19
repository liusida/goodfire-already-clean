from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from gb10_load_llm import load_model_to_cuda
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from measure_mlp_distributions import (
    DATASETS,
    ICA_GAUSSIAN_LOG_COSH_EXPECTATION,
    make_batch,
    parse_layers,
    resolve_model,
    sample_sequences,
)


class LastLayerCaptured(Exception):
    """Stop the model after the highest requested MLP layer is captured."""


METRIC_LABELS = {
    "mean": "Mean",
    "variance": "Population variance",
    "skewness": "Population skewness",
    "kurtosis": "Population kurtosis",
    "ica_log_cosh_score": "ICA log-cosh score",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure per-neuron raw-distribution moments and standardized ICA "
            "log-cosh scores, then plot their distributions across neurons."
        )
    )
    parser.add_argument("--model", default="0.8B")
    parser.add_argument("--revision", default=None)
    parser.add_argument(
        "--dataset",
        choices=("en", "cn", "both"),
        default="en",
    )
    parser.add_argument(
        "--layers",
        default="0",
        help='Layer selection: "all", one layer, or a list such as "[0,7]".',
    )
    parser.add_argument("--num-tokens", type=int, default=100_000)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle-buffer", type=int, default=10_000)
    parser.add_argument("--bins", type=int, default=100)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/mlp_measurement"),
    )
    return parser.parse_args()


def measure_neuron_metrics(
    *,
    model: AutoModelForCausalLM,
    sequences: list[list[int]],
    layers: list[int],
    intermediate_size: int,
    pad_token_id: int,
    batch_size: int,
    token_count: int,
    label: str,
) -> dict[int, dict[str, torch.Tensor]]:
    moments = {
        layer: {
            name: torch.zeros(intermediate_size, dtype=torch.float64)
            for name in ("sum", "square_sum", "cube_sum", "fourth_sum")
        }
        for layer in layers
    }
    log_cosh_sums = {
        layer: torch.zeros(intermediate_size, dtype=torch.float64)
        for layer in layers
    }
    means: dict[int, torch.Tensor] = {}
    inverse_stds: dict[int, torch.Tensor] = {}
    nonconstant: dict[int, torch.Tensor] = {}
    current_mask: torch.Tensor | None = None
    phase = "moments"
    last_layer = layers[-1]

    def make_hook(layer: int):
        def capture(
            _module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]
        ) -> None:
            if current_mask is None:
                raise RuntimeError("Activation hook fired without a mask")
            values = inputs[0][current_mask].float()
            if phase == "moments":
                squared = values.square()
                batch_moments = {
                    "sum": values.sum(dim=0),
                    "square_sum": squared.sum(dim=0),
                    "cube_sum": (squared * values).sum(dim=0),
                    "fourth_sum": squared.square().sum(dim=0),
                }
                for name, batch_value in batch_moments.items():
                    moments[layer][name] += batch_value.cpu().double()
            else:
                standardized = (values - means[layer]) * inverse_stds[layer]
                log_cosh = (
                    torch.logaddexp(standardized, -standardized) - math.log(2.0)
                )
                log_cosh_sums[layer] += log_cosh.sum(dim=0).cpu().double()
            if layer == last_layer:
                raise LastLayerCaptured

        return capture

    handles = [
        model.model.layers[layer].mlp.down_proj.register_forward_pre_hook(
            make_hook(layer)
        )
        for layer in layers
    ]
    try:
        for pass_index in range(2):
            phase = "moments" if pass_index == 0 else "ICA log-cosh"
            if pass_index == 1:
                for layer in layers:
                    mean = moments[layer]["sum"] / token_count
                    variance = (
                        moments[layer]["square_sum"] / token_count - mean.square()
                    ).clamp_min(0.0)
                    valid = variance > 0
                    inverse_std = torch.zeros_like(variance)
                    inverse_std[valid] = variance[valid].rsqrt()
                    means[layer] = mean.to(model.device, dtype=torch.float32)
                    inverse_stds[layer] = inverse_std.to(
                        model.device, dtype=torch.float32
                    )
                    nonconstant[layer] = valid

            progress = tqdm(
                total=token_count,
                unit="tok",
                desc=f"{label} · {phase} · layers {layers}",
            )
            for start in range(0, len(sequences), batch_size):
                batch_sequences = sequences[start : start + batch_size]
                input_ids, attention_mask = make_batch(
                    batch_sequences, pad_token_id
                )
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
            progress.close()
    finally:
        for handle in handles:
            handle.remove()

    results: dict[int, dict[str, torch.Tensor]] = {}
    for layer in layers:
        mean = moments[layer]["sum"] / token_count
        second_raw = moments[layer]["square_sum"] / token_count
        third_raw = moments[layer]["cube_sum"] / token_count
        fourth_raw = moments[layer]["fourth_sum"] / token_count
        variance = (second_raw - mean.square()).clamp_min(0.0)
        central_third = third_raw - 3 * mean * second_raw + 2 * mean.pow(3)
        central_fourth = (
            fourth_raw
            - 4 * mean * third_raw
            + 6 * mean.square() * second_raw
            - 3 * mean.pow(4)
        ).clamp_min(0.0)
        valid = nonconstant[layer]
        skewness = torch.full_like(mean, torch.nan)
        kurtosis = torch.full_like(mean, torch.nan)
        skewness[valid] = central_third[valid] / variance[valid].pow(1.5)
        kurtosis[valid] = central_fourth[valid] / variance[valid].square()
        log_cosh_expectation = torch.full_like(mean, torch.nan)
        log_cosh_expectation[valid] = log_cosh_sums[layer][valid] / token_count
        log_cosh_contrast = (
            log_cosh_expectation - ICA_GAUSSIAN_LOG_COSH_EXPECTATION
        )
        results[layer] = {
            "mean": mean,
            "variance": variance,
            "skewness": skewness,
            "kurtosis": kurtosis,
            "excess_kurtosis": kurtosis - 3.0,
            "ica_log_cosh_expectation": log_cosh_expectation,
            "ica_log_cosh_contrast": log_cosh_contrast,
            "ica_log_cosh_score": log_cosh_contrast.square(),
        }
    return results


def describe(values: torch.Tensor) -> dict[str, float | int]:
    finite = values[torch.isfinite(values)]
    if not finite.numel():
        return {"finite_count": 0, "undefined_count": len(values)}
    return {
        "finite_count": int(finite.numel()),
        "undefined_count": int(values.numel() - finite.numel()),
        "minimum": float(finite.min()),
        "p01": float(torch.quantile(finite, 0.01)),
        "median": float(torch.median(finite)),
        "mean": float(finite.mean()),
        "p99": float(torch.quantile(finite, 0.99)),
        "maximum": float(finite.max()),
    }


def ranked_metric_extremes(
    metrics: dict[str, torch.Tensor],
    metric_name: str,
    count: int = 10,
) -> dict[str, list[dict[str, float | int]]]:
    values = metrics[metric_name]
    finite_indices = torch.isfinite(values).nonzero(as_tuple=False).flatten()
    finite_values = values[finite_indices]
    rank_count = min(count, int(finite_values.numel()))

    def records(*, largest: bool) -> list[dict[str, float | int]]:
        if rank_count == 0:
            return []
        _, positions = torch.topk(
            finite_values, rank_count, largest=largest, sorted=True
        )
        result: list[dict[str, float | int]] = []
        for rank, position in enumerate(positions.tolist(), start=1):
            neuron = int(finite_indices[position])
            record: dict[str, float | int] = {"rank": rank, "neuron": neuron}
            for name in METRIC_LABELS:
                record[name] = float(metrics[name][neuron])
            result.append(record)
        return result

    return {
        "lowest": records(largest=False),
        "highest": records(largest=True),
    }


def ranked_kurtosis_extremes(
    metrics: dict[str, torch.Tensor],
    count: int = 10,
) -> dict[str, list[dict[str, float | int]]]:
    """Backward-compatible wrapper for callers that rank kurtosis."""
    return ranked_metric_extremes(metrics, "kurtosis", count)


def plot_histograms(
    metrics: dict[str, torch.Tensor],
    *,
    title: str,
    output: Path,
    bins: int,
    central: bool,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    for ax, (name, label) in zip(axes.flat, METRIC_LABELS.items()):
        values = metrics[name]
        values = values[torch.isfinite(values)].numpy()
        clipped = 0
        if central and len(values) > 1:
            lower, upper = np.quantile(values, [0.005, 0.995])
            clipped = int(np.count_nonzero((values < lower) | (values > upper)))
            values = values[(values >= lower) & (values <= upper)]
        ax.hist(values, bins=bins, color="#4C78A8", alpha=0.88)
        ax.set_title(label)
        ax.set_ylabel("Neurons")
        ax.set_yscale("log")
        ax.grid(axis="y", alpha=0.2)
        if clipped:
            ax.text(
                0.98,
                0.95,
                f"{clipped:,} outside central 99%",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8,
                color="#555555",
            )
    axes.flat[-1].axis("off")
    fig.suptitle(title + (" · central 99%" if central else " · full range"))
    fig.savefig(output, dpi=180)
    plt.close(fig)


def save_layer(
    *,
    output_dir: Path,
    metrics: dict[str, torch.Tensor],
    model_name: str,
    model_id: str,
    model_revision: str | None,
    layer: int,
    dataset_key: str,
    dataset_label: str,
    token_count: int,
    bins: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    title = (
        f"{model_name} · Layer {layer} · {len(metrics['mean']):,} MLP neurons\n"
        f"Per-neuron raw-distribution metrics · {dataset_label}"
    )
    plot_histograms(
        metrics,
        title=title,
        output=output_dir / "metric_histograms.png",
        bins=bins,
        central=False,
    )
    plot_histograms(
        metrics,
        title=title,
        output=output_dir / "metric_histograms_central_99.png",
        bins=bins,
        central=True,
    )
    torch.save(metrics, output_dir / "neuron_metrics.pt")
    histogram_data: dict[str, dict[str, torch.Tensor]] = {}
    for name in METRIC_LABELS:
        values = metrics[name]
        values = values[torch.isfinite(values)].numpy()
        counts, edges = np.histogram(values, bins=bins)
        histogram_data[name] = {
            "counts": torch.from_numpy(counts),
            "edges": torch.from_numpy(edges),
        }
    torch.save(histogram_data, output_dir / "metric_histograms.pt")
    summaries = {name: describe(metrics[name]) for name in METRIC_LABELS}
    kurtosis_extremes = ranked_kurtosis_extremes(metrics)
    skewness_extremes = ranked_metric_extremes(metrics, "skewness")
    log_cosh_extremes = ranked_metric_extremes(
        metrics, "ica_log_cosh_score"
    )
    (output_dir / "kurtosis_extremes.json").write_text(
        json.dumps(
            {
                "definition": "population Pearson kurtosis",
                **kurtosis_extremes,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "skewness_extremes.json").write_text(
        json.dumps(
            {
                "definition": "population skewness (third standardized moment)",
                **skewness_extremes,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "log_cosh_extremes.json").write_text(
        json.dumps(
            {
                "definition": (
                    "squared ICA log-cosh contrast: "
                    "(E[log(cosh(z))] - E[log(cosh(v))])^2, "
                    "z standardized per neuron, v ~ N(0,1)"
                ),
                **log_cosh_extremes,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    details = {
        "model_id": model_id,
        "model_revision": model_revision,
        "layer": layer,
        "dataset_key": dataset_key,
        "dataset_label": dataset_label,
        "sampled_tokens": token_count,
        "neuron_count": len(metrics["mean"]),
        "activation_definition": "input to mlp.down_proj (post-SwiGLU)",
        "skewness_definition": "population skewness (third standardized moment)",
        "kurtosis_definition": "population Pearson kurtosis",
        "ica_log_cosh_definition": (
            "(E[log(cosh(z))] - E[log(cosh(v))])^2, "
            "z standardized per neuron, v ~ N(0,1)"
        ),
        "ica_gaussian_log_cosh_expectation": (
            ICA_GAUSSIAN_LOG_COSH_EXPECTATION
        ),
        "summaries": summaries,
        "skewness_extremes": skewness_extremes,
        "kurtosis_extremes": kurtosis_extremes,
        "log_cosh_extremes": log_cosh_extremes,
    }
    (output_dir / "details.json").write_text(
        json.dumps(details, indent=2) + "\n", encoding="utf-8"
    )
    rows = [
        "| Metric | Min | Median | Mean | P99 | Max |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, label in METRIC_LABELS.items():
        summary = summaries[name]
        rows.append(
            f"| {label} | {summary['minimum']:.6g} | "
            f"{summary['median']:.6g} | {summary['mean']:.6g} | "
            f"{summary['p99']:.6g} | {summary['maximum']:.6g} |"
        )
    kurtosis_rows = [
        "| Rank | Lowest neuron | Kurtosis | Highest neuron | Kurtosis |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for lowest, highest in zip(
        kurtosis_extremes["lowest"], kurtosis_extremes["highest"]
    ):
        kurtosis_rows.append(
            f"| {lowest['rank']} | N{lowest['neuron']} | "
            f"{lowest['kurtosis']:.6g} | N{highest['neuron']} | "
            f"{highest['kurtosis']:.6g} |"
        )
    skewness_rows = [
        "| Rank | Most negative neuron | Skewness | Most positive neuron | Skewness |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for lowest, highest in zip(
        skewness_extremes["lowest"], skewness_extremes["highest"]
    ):
        skewness_rows.append(
            f"| {lowest['rank']} | N{lowest['neuron']} | "
            f"{lowest['skewness']:.6g} | N{highest['neuron']} | "
            f"{highest['skewness']:.6g} |"
        )
    log_cosh_rows = [
        "| Rank | Lowest neuron | ICA score | Highest neuron | ICA score |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for lowest, highest in zip(
        log_cosh_extremes["lowest"], log_cosh_extremes["highest"]
    ):
        log_cosh_rows.append(
            f"| {lowest['rank']} | N{lowest['neuron']} | "
            f"{lowest['ica_log_cosh_score']:.6g} | "
            f"N{highest['neuron']} | "
            f"{highest['ica_log_cosh_score']:.6g} |"
        )
    report = [
        f"# Per-neuron MLP metrics: {model_name} · Layer {layer}",
        "",
        f"Corpus: **{dataset_label}** · sampled tokens: **{token_count:,}**",
        "",
        "![Full-range metric histograms](metric_histograms.png)",
        "",
        "![Central-99% metric histograms](metric_histograms_central_99.png)",
        "",
        "## Across-neuron summary",
        "",
        *rows,
        "",
        (
            "The ICA score is `(E[log(cosh(z))] - E[log(cosh(v))])²`, "
            "where each neuron is standardized independently and `v ~ N(0,1)`."
        ),
        "",
        "## Skewness extremes",
        "",
        *skewness_rows,
        "",
        "## Kurtosis extremes",
        "",
        *kurtosis_rows,
        "",
        "## ICA log-cosh score extremes",
        "",
        *log_cosh_rows,
        "",
    ]
    (output_dir / "report.md").write_text("\n".join(report), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.num_tokens <= 0 or args.bins <= 0:
        raise ValueError("--num-tokens and --bins must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    model_id, model_revision = resolve_model(args.model, args.revision)
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=model_revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    corpus_keys = ("en", "cn") if args.dataset == "both" else (args.dataset,)
    sequences: list[list[int]] = []
    for corpus_key in corpus_keys:
        corpus = DATASETS[corpus_key]
        print(f"Sampling {args.num_tokens:,} tokens from {corpus['label']}")
        sequences.extend(
            sample_sequences(
                tokenizer,
                dataset_id=corpus["dataset_id"],
                text_field=corpus["text_field"],
                dataset_label=corpus["label"],
                num_tokens=args.num_tokens,
                max_length=args.max_length,
                seed=args.seed,
                shuffle_buffer=args.shuffle_buffer,
            )
        )
    token_count = args.num_tokens * len(corpus_keys)
    dataset_label = " + ".join(DATASETS[key]["label"] for key in corpus_keys)

    model, touched = load_model_to_cuda(
        AutoModelForCausalLM,
        model_id,
        revision=model_revision,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        touch="auto",
        return_touched=True,
    )
    model.eval()
    layers = parse_layers(args.layers, len(model.model.layers))
    intermediate_size = int(model.model.layers[0].mlp.down_proj.in_features)
    metrics = measure_neuron_metrics(
        model=model,
        sequences=sequences,
        layers=layers,
        intermediate_size=intermediate_size,
        pad_token_id=tokenizer.pad_token_id,
        batch_size=args.batch_size,
        token_count=token_count,
        label=args.dataset,
    )

    model_name = model_id.rsplit("/", 1)[-1]
    for layer in layers:
        save_layer(
            output_dir=(
                args.output_root
                / model_name
                / f"layer_{layer:02d}"
                / "neuron_metrics"
                / args.dataset
            ),
            metrics=metrics[layer],
            model_name=model_name,
            model_id=model_id,
            model_revision=model_revision,
            layer=layer,
            dataset_key=args.dataset,
            dataset_label=dataset_label,
            token_count=token_count,
            bins=args.bins,
        )

    metadata = {
        "model_id": model_id,
        "model_revision": model_revision,
        "dataset": args.dataset,
        "tokens_per_corpus": args.num_tokens,
        "total_tokens": token_count,
        "layers": layers,
        "neuron_count": intermediate_size,
        "seed": args.seed,
        "shuffle_buffer": args.shuffle_buffer,
        "cpu_pages_touched_gib": touched / 1024**3,
    }
    model_dir = args.output_root / model_name
    (model_dir / "neuron_metrics_run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved per-neuron metrics under {model_dir.resolve()}")


if __name__ == "__main__":
    main()
