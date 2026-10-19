from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from gb10_load_llm import load_model_to_cuda
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Project normalized token embeddings onto one Layer-0 W_z gate row."
    )
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B-Base")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--direction", type=int, default=1825)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--chunk-size", type=int, default=8192)
    parser.add_argument("--bins", type=int, default=200)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/mlp_measurement"),
    )
    return parser.parse_args()


def display_token(tokenizer, token_id: int) -> str:
    return tokenizer.decode(
        [token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False
    ).replace("\n", "\\n").replace("\t", "\\t").replace("\r", "\\r")


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model, touched = load_model_to_cuda(
        AutoModelForCausalLM,
        args.model,
        revision=args.revision,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        touch="auto",
        return_touched=True,
    )
    model.eval()
    layer = model.model.layers[0]
    if not hasattr(layer, "linear_attn"):
        raise ValueError("Layer 0 does not have linear_attn")
    projection = layer.linear_attn.in_proj_z
    if not 0 <= args.direction < projection.out_features:
        raise ValueError(f"direction must be in [0, {projection.out_features})")

    embeddings = model.model.embed_tokens.weight
    norm = layer.input_layernorm
    # Qwen3.5 uses zero-centered RMSNorm: scale = 1 + learned weight.
    norm_weight = 1.0 + norm.weight.detach().float()
    epsilon = float(getattr(norm, "variance_epsilon", norm.eps))
    gate_row = projection.weight[args.direction].detach().float()
    gate_bias = (
        float(projection.bias[args.direction]) if projection.bias is not None else 0.0
    )
    scores = torch.empty(embeddings.shape[0], dtype=torch.float32)
    with torch.inference_mode():
        for start in range(0, embeddings.shape[0], args.chunk_size):
            stop = min(start + args.chunk_size, embeddings.shape[0])
            values = embeddings[start:stop].float()
            denominator = torch.sqrt(values.square().mean(dim=-1, keepdim=True) + epsilon)
            normalized = values / denominator * norm_weight
            scores[start:stop] = (normalized @ gate_row + gate_bias).cpu()

    count = min(args.top_k, scores.numel())
    high_values, high_ids = torch.topk(scores, count, largest=True)
    low_values, low_ids = torch.topk(scores, count, largest=False)
    model_name = args.model.rsplit("/", 1)[-1]
    output_dir = (
        args.output_root
        / model_name
        / "layer_00"
        / f"attention_output_O{args.direction}_wz_vocab_projection"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model": args.model,
            "direction": args.direction,
            "definition": "in_proj_z.weight[direction] @ input_layernorm(token_embedding)",
            "scores": scores,
        },
        output_dir / "scores.pt",
    )
    with (output_dir / "extremes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["tail", "rank", "token_id", "token", "z_preactivation", "silu_z"])
        for tail, ids, values in (
            ("positive", high_ids, high_values),
            ("negative", low_ids, low_values),
        ):
            for rank, (token_id, value) in enumerate(zip(ids.tolist(), values.tolist()), 1):
                writer.writerow(
                    [
                        tail,
                        rank,
                        token_id,
                        display_token(tokenizer, token_id),
                        value,
                        float(torch.nn.functional.silu(torch.tensor(value))),
                    ]
                )

    array = scores.numpy()
    summary = {
        "model": args.model,
        "layer": 0,
        "direction": args.direction,
        "vocabulary_size": int(scores.numel()),
        "mean": float(array.mean()),
        "standard_deviation": float(array.std()),
        "minimum": float(array.min()),
        "median": float(np.median(array)),
        "maximum": float(array.max()),
        "cpu_pages_touched_gib": touched / 1024**3,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(array, bins=args.bins, color="#4c78a8", alpha=0.88)
    ax.axvline(0, color="#333333", linewidth=1)
    ax.set_title(f"Layer 0 · Wz gate O{args.direction} · vocabulary projection")
    ax.set_xlabel(r"$W_z[O,:] \cdot \mathrm{RMSNorm}(E_{token})$")
    ax.set_ylabel("Vocabulary tokens")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "distribution.png", dpi=180)
    plt.close(fig)

    print(json.dumps(summary, indent=2))
    print("\nTop positive tokens:")
    for rank, (token_id, value) in enumerate(zip(high_ids.tolist(), high_values.tolist()), 1):
        print(f"{rank:3d}  {value:+.6f}  {token_id:6d}  {display_token(tokenizer, token_id)!r}")
    print("\nTop negative tokens:")
    for rank, (token_id, value) in enumerate(zip(low_ids.tolist(), low_values.tolist()), 1):
        print(f"{rank:3d}  {value:+.6f}  {token_id:6d}  {display_token(tokenizer, token_id)!r}")
    print(f"\nSaved to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
