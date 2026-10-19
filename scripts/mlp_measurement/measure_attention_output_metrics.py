from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from gb10_load_llm import load_model_to_cuda
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from measure_mlp_distributions import DATASETS, make_batch, resolve_model, sample_sequences


class AttentionOutputCaptured(Exception):
    """Stop the model after the requested attention output input is captured."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure raw activation moments for channels entering an attention "
            "output projection (the coefficients of its output directions)."
        )
    )
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B-Base")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--dataset", choices=tuple(DATASETS), default="en")
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--num-tokens", type=int, default=1_000_000)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle-buffer", type=int, default=10_000)
    parser.add_argument("--extreme-count", type=int, default=10)
    parser.add_argument("--bins", type=int, default=100)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/mlp_measurement"),
    )
    return parser.parse_args()


def attention_output_projection(layer: torch.nn.Module) -> tuple[torch.nn.Module, str]:
    if hasattr(layer, "self_attn"):
        return layer.self_attn.o_proj, "self_attn.o_proj"
    if hasattr(layer, "linear_attn"):
        return layer.linear_attn.out_proj, "linear_attn.out_proj"
    raise RuntimeError("Layer has no recognized attention output projection")


def main() -> None:
    args = parse_args()
    if args.layer < 0 or args.num_tokens <= 0 or args.extreme_count <= 0:
        raise ValueError("layer must be nonnegative and counts must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    model_id, model_revision = resolve_model(args.model, args.revision)
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=model_revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    corpus = DATASETS[args.dataset]
    print(f"Sampling {args.num_tokens:,} tokens from {corpus['label']}", flush=True)
    sequences = sample_sequences(
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
        model_id,
        revision=model_revision,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        touch="auto",
        return_touched=True,
    )
    model.eval()
    if args.layer >= len(model.model.layers):
        raise ValueError(f"Layer {args.layer} outside model range")
    projection, projection_name = attention_output_projection(
        model.model.layers[args.layer]
    )
    direction_count = int(projection.in_features)
    sums = {
        name: torch.zeros(direction_count, dtype=torch.float64)
        for name in ("sum", "square", "cube", "fourth")
    }
    current_mask: torch.Tensor | None = None

    def capture(_module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        if current_mask is None:
            raise RuntimeError("Attention hook fired without a token mask")
        values = inputs[0][current_mask].float()
        squared = values.square()
        sums["sum"] += values.sum(dim=0).cpu().double()
        sums["square"] += squared.sum(dim=0).cpu().double()
        sums["cube"] += (squared * values).sum(dim=0).cpu().double()
        sums["fourth"] += squared.square().sum(dim=0).cpu().double()
        raise AttentionOutputCaptured

    handle = projection.register_forward_pre_hook(capture)
    progress = tqdm(total=args.num_tokens, unit="tok", desc=f"{args.dataset} · L{args.layer} {projection_name}")
    observed_tokens = 0
    try:
        for start in range(0, len(sequences), args.batch_size):
            batch_sequences = sequences[start : start + args.batch_size]
            input_ids, attention_mask = make_batch(batch_sequences, tokenizer.pad_token_id)
            current_mask = attention_mask.bool().to(model.device)
            try:
                with torch.inference_mode():
                    model.model(
                        input_ids=input_ids.to(model.device),
                        attention_mask=attention_mask.to(model.device),
                        use_cache=False,
                    )
            except AttentionOutputCaptured:
                pass
            count = int(attention_mask.sum())
            observed_tokens += count
            progress.update(count)
    finally:
        progress.close()
        handle.remove()
    if observed_tokens != args.num_tokens:
        raise RuntimeError(
            f"Expected {args.num_tokens:,} tokens but captured {observed_tokens:,}"
        )

    count = float(observed_tokens)
    mean = sums["sum"] / count
    second = sums["square"] / count
    third = sums["cube"] / count
    fourth = sums["fourth"] / count
    variance = (second - mean.square()).clamp_min(0.0)
    central_third = third - 3 * mean * second + 2 * mean.pow(3)
    central_fourth = (
        fourth
        - 4 * mean * third
        + 6 * mean.square() * second
        - 3 * mean.pow(4)
    ).clamp_min(0.0)
    valid = variance > 0
    skewness = torch.full_like(mean, torch.nan)
    kurtosis = torch.full_like(mean, torch.nan)
    skewness[valid] = central_third[valid] / variance[valid].pow(1.5)
    kurtosis[valid] = central_fourth[valid] / variance[valid].square()

    finite_ids = torch.nonzero(torch.isfinite(kurtosis), as_tuple=False).flatten()
    finite_values = kurtosis[finite_ids]
    extreme_count = min(args.extreme_count, int(finite_values.numel()))
    low_values, low_offsets = torch.topk(finite_values, extreme_count, largest=False)
    high_values, high_offsets = torch.topk(finite_values, extreme_count, largest=True)
    low_ids = finite_ids[low_offsets]
    high_ids = finite_ids[high_offsets]

    model_name = model_id.rsplit("/", 1)[-1]
    output_dir = (
        args.output_root
        / model_name
        / f"layer_{args.layer:02d}"
        / "attention_output_metrics"
        / args.dataset
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "mean": mean,
        "variance": variance,
        "skewness": skewness,
        "kurtosis": kurtosis,
        "excess_kurtosis": kurtosis - 3.0,
    }
    torch.save(metrics, output_dir / "metrics.pt")
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle_csv:
        writer = csv.writer(handle_csv)
        writer.writerow(["direction", "mean", "variance", "skewness", "kurtosis", "excess_kurtosis"])
        for direction in range(direction_count):
            writer.writerow(
                [direction] + [float(metrics[name][direction]) for name in metrics]
            )

    def rows(ids: torch.Tensor, values: torch.Tensor) -> list[dict[str, float | int]]:
        return [
            {
                "rank": rank,
                "direction": int(direction),
                "kurtosis": float(value),
                "mean": float(mean[direction]),
                "variance": float(variance[direction]),
                "skewness": float(skewness[direction]),
            }
            for rank, (direction, value) in enumerate(zip(ids, values), start=1)
        ]

    extremes = {"lowest": rows(low_ids, low_values), "highest": rows(high_ids, high_values)}
    (output_dir / "kurtosis_extremes.json").write_text(
        json.dumps(extremes, indent=2), encoding="utf-8"
    )
    summary = {
        "model": model_id,
        "layer": args.layer,
        "projection": projection_name,
        "activation_definition": f"input channels to {projection_name}",
        "dataset": args.dataset,
        "token_count": observed_tokens,
        "direction_count": direction_count,
        "seed": args.seed,
        "kurtosis": {
            "minimum": float(finite_values.min()),
            "p05": float(torch.quantile(finite_values, 0.05)),
            "median": float(torch.quantile(finite_values, 0.50)),
            "mean": float(finite_values.mean()),
            "p95": float(torch.quantile(finite_values, 0.95)),
            "maximum": float(finite_values.max()),
        },
        "cpu_pages_touched_gib": touched / 1024**3,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    figure, axis = plt.subplots(figsize=(8.5, 5.0))
    histogram_values = finite_values.numpy()
    log_bins = np.geomspace(
        float(histogram_values.min()),
        float(histogram_values.max()),
        args.bins + 1,
    )
    axis.hist(histogram_values, bins=log_bins, color="#b279a2", alpha=0.9)
    axis.axvline(float(torch.median(finite_values)), color="#202830", linestyle="--", linewidth=1.3)
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("Activation Pearson kurtosis (log scale)")
    axis.set_ylabel("Attention-output directions (log count)")
    axis.set_title(
        f"{model_name} · Layer {args.layer} {projection_name}\n"
        f"{observed_tokens:,} OpenWebText tokens"
    )
    axis.grid(axis="y", alpha=0.18)
    figure.tight_layout()
    figure.savefig(output_dir / "kurtosis_histogram.png", dpi=180, facecolor="white")
    plt.close(figure)

    report = [
        f"# Layer {args.layer} attention-output activation kurtosis",
        "",
        f"Projection: `{projection_name}`  ",
        f"Tokens: **{observed_tokens:,}**  ",
        f"Directions: **{direction_count:,}**",
        "",
        "## Lowest kurtosis",
        "",
        "| Rank | Direction | Kurtosis | Skewness |",
        "| ---: | ---: | ---: | ---: |",
    ]
    report.extend(
        f"| {row['rank']} | O{row['direction']} | {row['kurtosis']:.6g} | {row['skewness']:.6g} |"
        for row in extremes["lowest"]
    )
    report.extend(
        [
            "",
            "## Highest kurtosis",
            "",
            "| Rank | Direction | Kurtosis | Skewness |",
            "| ---: | ---: | ---: | ---: |",
        ]
    )
    report.extend(
        f"| {row['rank']} | O{row['direction']} | {row['kurtosis']:.6g} | {row['skewness']:.6g} |"
        for row in extremes["highest"]
    )
    (output_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print("\nLowest:", [(int(i), float(v)) for i, v in zip(low_ids, low_values)])
    print("Highest:", [(int(i), float(v)) for i, v in zip(high_ids, high_values)])
    print(f"Saved results to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
