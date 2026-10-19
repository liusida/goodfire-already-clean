from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from gb10_load_llm import load_model_to_cuda
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoTokenizer, Qwen3Model


MODEL_ID = "Qwen/Qwen3-32B"
MODEL_REVISION = "9216db5781bf21249d130ec9da846c4624c16137"
DATASET_ID = "NeelNanda/pile-10k"
DATASET_REVISION = "127bfedcd5047750df5ccf3a12979a47bfa0bafa"


class CapturedMlpOutput(RuntimeError):
    """Stop the model after recording the target layer's complete MLP output."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke-test projection of complete MLP output onto one neuron direction."
    )
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--neuron", type=int, default=15647)
    parser.add_argument("--max-tokens", type=int, default=50_000)
    parser.add_argument("--window-tokens", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--bins", type=int, default=180)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/layer0/neuron_15647_mlp_projection_smoke"),
    )
    args = parser.parse_args()
    if min(args.max_tokens, args.window_tokens, args.batch_size, args.bins) < 1:
        parser.error("token, batch, and bin counts must be positive")
    if args.layer < 0 or args.neuron < 0:
        parser.error("layer and neuron must be non-negative")
    return args


def distribution(values: np.ndarray) -> dict:
    values64 = values.astype(np.float64)
    mean = float(values64.mean())
    delta = values64 - mean
    moment_2 = float(np.mean(delta**2))
    std = math.sqrt(moment_2)
    skew = float(np.mean(delta**3) / std**3)
    pearson = float(np.mean(delta**4) / moment_2**2)
    levels = np.array([0, 0.001, 0.01, 0.5, 0.99, 0.999, 1.0])
    return {
        "count": int(values.size),
        "minimum": float(values64.min()),
        "maximum": float(values64.max()),
        "mean": mean,
        "population_standard_deviation": std,
        "population_skewness": skew,
        "pearson_kurtosis": pearson,
        "excess_kurtosis": pearson - 3.0,
        "quantiles": {
            f"{level:.3%}": float(value)
            for level, value in zip(levels, np.quantile(values64, levels), strict=True)
        },
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    config = AutoConfig.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    checkpoint_layer_count = int(config.num_hidden_layers)
    if args.layer >= checkpoint_layer_count:
        raise ValueError("Requested layer is outside the checkpoint")
    config.num_hidden_layers = args.layer + 1
    model, touched = load_model_to_cuda(
        Qwen3Model,
        MODEL_ID,
        config=config,
        revision=MODEL_REVISION,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        touch="auto",
        return_touched=True,
    )
    model.eval()
    mlp = model.layers[args.layer].mlp
    if args.neuron >= mlp.gate_proj.out_features:
        raise ValueError("Requested neuron is outside the MLP width")

    # down_proj maps each scalar MLP activation along one residual-stream column.
    direction = mlp.down_proj.weight[:, args.neuron].detach()
    direction_norm = torch.linalg.vector_norm(direction.float())
    unit_direction = (direction.float() / direction_norm).to(direction.dtype)
    direction_norm_squared = float(direction_norm**2)
    direct_batches: list[torch.Tensor] = []
    unit_projection_batches: list[torch.Tensor] = []

    def capture_direct_activation(
        _module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]
    ) -> None:
        direct_batches.append(inputs[0][..., args.neuron].detach().float().cpu())

    def capture_complete_output(
        _module: torch.nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        projection = F.linear(output, unit_direction)
        unit_projection_batches.append(projection.detach().float().cpu())
        raise CapturedMlpOutput

    pre_handle = mlp.down_proj.register_forward_pre_hook(capture_direct_activation)
    output_handle = mlp.register_forward_hook(capture_complete_output)
    dataset = load_dataset(DATASET_ID, split="train", revision=DATASET_REVISION)
    windows: list[list[int]] = []
    gathered_tokens = 0
    for example in dataset:
        if gathered_tokens >= args.max_tokens:
            break
        ids = tokenizer.encode(example["text"], add_special_tokens=False)
        ids = ids[: args.max_tokens - gathered_tokens]
        gathered_tokens += len(ids)
        windows.extend(
            ids[start : start + args.window_tokens]
            for start in range(0, len(ids), args.window_tokens)
        )

    try:
        for start in tqdm(
            range(0, len(windows), args.batch_size), desc="projection smoke batches"
        ):
            batch = windows[start : start + args.batch_size]
            max_length = max(map(len, batch))
            input_ids = torch.full(
                (len(batch), max_length), tokenizer.pad_token_id, dtype=torch.long
            )
            attention_mask = torch.zeros_like(input_ids)
            lengths = []
            for row, ids in enumerate(batch):
                lengths.append(len(ids))
                input_ids[row, : len(ids)] = torch.tensor(ids)
                attention_mask[row, : len(ids)] = 1
            before = len(direct_batches)
            with torch.inference_mode():
                try:
                    model(
                        input_ids=input_ids.to(model.device),
                        attention_mask=attention_mask.to(model.device),
                        use_cache=False,
                    )
                except CapturedMlpOutput:
                    pass
            if len(direct_batches) != before + 1 or len(unit_projection_batches) != before + 1:
                raise RuntimeError("Expected exactly one direct/projection capture per batch")
            direct_batches[-1] = torch.cat(
                [direct_batches[-1][row, :length] for row, length in enumerate(lengths)]
            )
            unit_projection_batches[-1] = torch.cat(
                [
                    unit_projection_batches[-1][row, :length]
                    for row, length in enumerate(lengths)
                ]
            )
    finally:
        pre_handle.remove()
        output_handle.remove()

    direct = torch.cat(direct_batches).numpy().astype("<f4", copy=False)
    unit_projection = (
        torch.cat(unit_projection_batches).numpy().astype("<f4", copy=False)
    )
    coefficient = unit_projection / float(direction_norm)
    interference = coefficient - direct
    if len(direct) != gathered_tokens or len(unit_projection) != gathered_tokens:
        raise RuntimeError("Captured score count does not match gathered tokens")

    np.savez(
        args.output_dir / "scores.npz",
        direct_activation=direct,
        unit_direction_projection=unit_projection,
        direction_coefficient=coefficient,
        cross_neuron_interference=interference,
    )
    statistics = {
        "direct_activation": distribution(direct),
        "unit_direction_projection": distribution(unit_projection),
        "direction_coefficient": distribution(coefficient),
        "cross_neuron_interference": distribution(interference),
    }
    summary = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "layer": args.layer,
        "neuron": args.neuron,
        "tokens": gathered_tokens,
        "direction_norm": float(direction_norm),
        "direction_norm_squared": direction_norm_squared,
        "projection_definitions": {
            "unit_direction_projection": "dot(complete_mlp_output, d_n / ||d_n||)",
            "direction_coefficient": "dot(complete_mlp_output, d_n) / ||d_n||^2",
            "d_n": "column N of down_proj.weight",
        },
        "statistics": statistics,
        "direct_projection_pearson_correlation": float(
            np.corrcoef(direct.astype(np.float64), coefficient.astype(np.float64))[0, 1]
        ),
        "touched_model_gib": touched / 1024**3,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), dpi=160)
    flat_axes = axes.ravel()
    panels = [
        (direct, "Direct neuron activation", "post-SwiGLU activation"),
        (coefficient, "MLP output projected onto neuron direction", "direction coefficient"),
    ]
    for ax, (values, title, xlabel) in zip(flat_axes[:2], panels, strict=True):
        ax.hist(values, bins=args.bins, color="#397792", log=True)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Token positions (log scale)")
        ax.grid(axis="y", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)

    projection_stats = statistics["direction_coefficient"]
    z = (coefficient - projection_stats["mean"]) / projection_stats[
        "population_standard_deviation"
    ]
    flat_axes[2].hist(
        z,
        bins=args.bins,
        range=(-6, 6),
        density=True,
        color="#397792",
        alpha=0.8,
        label="observed",
    )
    grid = np.linspace(-6, 6, 800)
    gaussian = np.exp(-(grid**2) / 2) / math.sqrt(2 * math.pi)
    flat_axes[2].plot(grid, gaussian, color="#b54720", linewidth=2, label="N(0,1)")
    flat_axes[2].set_title("Standardized projection vs Gaussian")
    flat_axes[2].set_xlabel("z-score")
    flat_axes[2].set_ylabel("density")
    flat_axes[2].legend(frameon=False)
    flat_axes[2].grid(alpha=0.2)
    flat_axes[2].spines[["top", "right"]].set_visible(False)

    interference_stats = statistics["cross_neuron_interference"]
    interference_z = (interference - interference_stats["mean"]) / interference_stats[
        "population_standard_deviation"
    ]
    flat_axes[3].hist(
        interference_z,
        bins=args.bins,
        range=(-6, 6),
        density=True,
        color="#397792",
        alpha=0.8,
        label="observed",
    )
    flat_axes[3].plot(grid, gaussian, color="#b54720", linewidth=2, label="N(0,1)")
    flat_axes[3].set_title("Cross-neuron interference vs Gaussian")
    flat_axes[3].set_xlabel("z-score")
    flat_axes[3].set_ylabel("density")
    flat_axes[3].legend(frameon=False)
    flat_axes[3].grid(alpha=0.2)
    flat_axes[3].spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        f"Qwen3-32B L{args.layer}/N{args.neuron} · {gathered_tokens:,} Pile tokens"
    )
    fig.tight_layout()
    plot_path = args.output_dir / "projection_distribution.png"
    fig.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Touched model pages: {touched / 1024**3:.2f} GiB")
    print(f"Tokens: {gathered_tokens:,}")
    for name, stats in statistics.items():
        print(
            f"{name}: mean={stats['mean']:.6g}, std={stats['population_standard_deviation']:.6g}, "
            f"skew={stats['population_skewness']:.6g}, excess_kurtosis={stats['excess_kurtosis']:.6g}"
        )
    print(f"Saved {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
