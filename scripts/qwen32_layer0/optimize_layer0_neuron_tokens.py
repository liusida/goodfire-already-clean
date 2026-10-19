from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from gb10_load_llm import load_model_to_cuda
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoTokenizer, Qwen3Model


MODEL_ID = "Qwen/Qwen3-32B"
MODEL_REVISION = "9216db5781bf21249d130ec9da846c4624c16137"


class CapturedNeuron(RuntimeError):
    """Stop after target-layer attention and the selected scalar MLP activation."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exhaustively optimize a layer-0 neuron over the Qwen vocabulary."
    )
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--neuron", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--keep", type=int, default=50)
    parser.add_argument("--bins", type=int, default=300)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.layer < 0 or args.neuron < 0:
        parser.error("layer and neuron must be non-negative")
    if min(args.batch_size, args.keep, args.bins) < 1:
        parser.error("batch-size, keep, and bins must be positive")
    if args.output_dir is None:
        args.output_dir = Path(
            f"outputs/layer{args.layer}/neuron_{args.neuron}_vocabulary_optimization"
        )
    return args


def byte_decoder() -> dict[str, int]:
    byte_values = list(range(ord("!"), ord("~") + 1))
    byte_values += list(range(ord("¡"), ord("¬") + 1))
    byte_values += list(range(ord("®"), ord("ÿ") + 1))
    codepoints = byte_values.copy()
    extra = 0
    for value in range(256):
        if value not in byte_values:
            byte_values.append(value)
            codepoints.append(256 + extra)
            extra += 1
    return {chr(codepoint): value for value, codepoint in zip(byte_values, codepoints)}


def token_record(tokenizer, token_id: int, activation: float, rank: int, polarity: str) -> dict:
    piece = tokenizer.convert_ids_to_tokens(token_id)
    decoder = byte_decoder()
    try:
        raw_bytes = bytes(decoder[char] for char in piece)
        byte_hex = raw_bytes.hex(" ")
    except (KeyError, TypeError):
        byte_hex = ""
    return {
        "polarity": polarity,
        "rank": rank,
        "activation": activation,
        "token_id": token_id,
        "token_piece": piece,
        "decoded_token": tokenizer.decode(
            [token_id], clean_up_tokenization_spaces=False
        ),
        "token_bytes_hex": byte_hex,
        "is_special": token_id in tokenizer.all_special_ids,
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
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
    layer = model.layers[args.layer]
    width = int(layer.mlp.gate_proj.out_features)
    if args.neuron >= width:
        raise ValueError(f"Neuron {args.neuron} is outside MLP width {width}")
    gate_row = layer.mlp.gate_proj.weight[args.neuron].detach()
    up_row = layer.mlp.up_proj.weight[args.neuron].detach()
    captured: list[torch.Tensor] = []

    def capture(_module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        hidden = inputs[0]
        activation = F.silu(F.linear(hidden, gate_row)) * F.linear(hidden, up_row)
        captured.append(activation[:, -1].detach().float().cpu())
        raise CapturedNeuron

    handle = layer.mlp.register_forward_pre_hook(capture)
    vocab_size = int(model.embed_tokens.num_embeddings)
    scores = np.empty(vocab_size, dtype="<f4")
    try:
        for start in tqdm(
            range(0, vocab_size, args.batch_size), desc="vocabulary batches"
        ):
            stop = min(vocab_size, start + args.batch_size)
            input_ids = torch.arange(start, stop, dtype=torch.long).unsqueeze(1)
            captured.clear()
            with torch.inference_mode():
                try:
                    model(input_ids=input_ids.to(model.device), use_cache=False)
                except CapturedNeuron:
                    pass
            if len(captured) != 1:
                raise RuntimeError(f"Expected one capture, got {len(captured)}")
            scores[start:stop] = captured[0].numpy()
    finally:
        handle.remove()

    np.save(args.output_dir / "vocabulary_scores.npy", scores)
    keep = min(args.keep, vocab_size)
    positive_ids = np.argpartition(scores, -keep)[-keep:]
    positive_ids = positive_ids[np.argsort(scores[positive_ids])[::-1]]
    negative_ids = np.argpartition(scores, keep)[:keep]
    negative_ids = negative_ids[np.argsort(scores[negative_ids])]
    records = []
    for polarity, ids in (("positive", positive_ids), ("negative", negative_ids)):
        records.extend(
            token_record(tokenizer, int(token_id), float(scores[token_id]), rank, polarity)
            for rank, token_id in enumerate(ids, start=1)
        )

    fields = list(records[0])
    with (args.output_dir / "extreme_tokens.csv").open(
        "w", newline="", encoding="utf-8"
    ) as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    (args.output_dir / "extreme_tokens.json").write_text(
        json.dumps(records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    values = scores.astype(np.float64)
    mean = float(values.mean())
    delta = values - mean
    moment_2 = float(np.mean(delta**2))
    std = moment_2**0.5
    summary = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "checkpoint_layer_count": checkpoint_layer_count,
        "loaded_layer_count": int(model.config.num_hidden_layers),
        "layer": args.layer,
        "neuron": args.neuron,
        "search_space": "every embedding token ID as a one-token sequence",
        "vocabulary_size": vocab_size,
        "activation_definition": "silu(gate_proj(x)[N]) * up_proj(x)[N]",
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "mean": mean,
        "population_standard_deviation": std,
        "population_skewness": float(np.mean(delta**3) / std**3),
        "excess_kurtosis": float(np.mean(delta**4) / moment_2**2 - 3.0),
        "positive_optimum": records[0],
        "negative_optimum": records[keep],
        "touched_model_gib": touched / 1024**3,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    fig, ax = plt.subplots(figsize=(12, 5), dpi=170)
    ax.hist(values, bins=args.bins, color="#397792", log=True)
    ax.axvline(values[positive_ids[0]], color="#b54720", linestyle="--", label="positive optimum")
    ax.axvline(values[negative_ids[0]], color="#2463a5", linestyle="--", label="negative optimum")
    ax.set_title(
        f"Qwen3-32B L{args.layer}/N{args.neuron}: exhaustive one-token vocabulary search"
    )
    ax.set_xlabel("Post-SwiGLU activation")
    ax.set_ylabel("Vocabulary tokens (log scale)")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(args.output_dir / "vocabulary_distribution.png", bbox_inches="tight")
    plt.close(fig)

    print(f"Vocabulary size: {vocab_size:,}")
    print(
        f"Positive optimum: id={positive_ids[0]}, score={scores[positive_ids[0]]:.9g}, "
        f"piece={tokenizer.convert_ids_to_tokens(int(positive_ids[0]))!r}"
    )
    print(
        f"Negative optimum: id={negative_ids[0]}, score={scores[negative_ids[0]]:.9g}, "
        f"piece={tokenizer.convert_ids_to_tokens(int(negative_ids[0]))!r}"
    )
    print(f"Saved {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
