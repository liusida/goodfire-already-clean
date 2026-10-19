from __future__ import annotations

import argparse
import csv
import heapq
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
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
class CapturedMlpInput(RuntimeError):
    """Stop after target-layer attention, before computing the complete MLP."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit one Qwen3-32B MLP neuron on every token in pile-10k."
    )
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--neuron", type=int, default=15647)
    parser.add_argument("--threshold", type=float, default=8.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--window-tokens", type=int, default=1024)
    parser.add_argument("--overlap-tokens", type=int, default=128)
    parser.add_argument("--context-tokens", type=int, default=32)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/layer0/park_neuron_pile10k"),
    )
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.layer < 0 or args.neuron < 0:
        parser.error("--layer and --neuron must be non-negative")
    if args.window_tokens < 2:
        parser.error("--window-tokens must be at least 2")
    if not 0 <= args.overlap_tokens < args.window_tokens:
        parser.error("--overlap-tokens must be in [0, window-tokens)")
    if args.context_tokens < 0:
        parser.error("--context-tokens must be non-negative")
    return args


def json_default(value: object) -> object:
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run the Qwen3 layer-0 audit")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.output_dir / "hits.jsonl"
    csv_path = args.output_dir / "hits.csv"
    summary_path = args.output_dir / "summary.json"
    plot_path = args.output_dir / "hit_token_counts.png"
    scores_path = args.output_dir / "scores.f32"
    documents_path = args.output_dir / "documents.jsonl"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = AutoConfig.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    checkpoint_layer_count = int(config.num_hidden_layers)
    if args.layer >= checkpoint_layer_count:
        raise ValueError(
            f"Layer {args.layer} is outside the {checkpoint_layer_count}-layer checkpoint"
        )
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
    neuron_count = int(layer.mlp.gate_proj.out_features)
    if args.neuron >= neuron_count:
        raise ValueError(f"Neuron {args.neuron} is outside width {neuron_count}")
    gate_row = layer.mlp.gate_proj.weight[args.neuron].detach()
    up_row = layer.mlp.up_proj.weight[args.neuron].detach()
    captured: list[torch.Tensor] = []

    def capture_one_neuron(
        _module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]
    ) -> None:
        normalized_hidden = inputs[0]
        gate = F.linear(normalized_hidden, gate_row)
        up = F.linear(normalized_hidden, up_row)
        activation = F.silu(gate) * up
        captured.append(activation.detach().float().cpu())
        raise CapturedMlpInput

    handle = layer.mlp.register_forward_pre_hook(capture_one_neuron)
    dataset = load_dataset(
        DATASET_ID,
        split="train",
        revision=DATASET_REVISION,
    )

    batch: list[dict] = []
    total_tokens = 0
    scored_tokens = 0
    total_windows = 0
    hit_count = 0
    token_counts: Counter[tuple[int, str, str]] = Counter()
    token_activation_sum: Counter[tuple[int, str, str]] = Counter()
    token_activation_min: dict[tuple[int, str, str], float] = {}
    token_activation_max: dict[tuple[int, str, str], float] = {}
    token_examples: defaultdict[tuple[int, str, str], list[dict]] = defaultdict(list)
    top_hits: list[tuple[float, int, dict]] = []
    heap_serial = 0

    csv_fields = [
        "dataset_index",
        "pile_set_name",
        "document_token_index",
        "token_id",
        "token_piece",
        "target_token",
        "activation",
        "left_context",
        "right_context",
        "marked_context",
    ]

    with (
        jsonl_path.open("w", encoding="utf-8") as jsonl_file,
        csv_path.open("w", newline="", encoding="utf-8") as csv_file,
        scores_path.open("wb") as scores_file,
        documents_path.open("w", encoding="utf-8") as documents_file,
    ):
        csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
        csv_writer.writeheader()

        def process_batch() -> None:
            nonlocal batch, scored_tokens, total_windows, hit_count, heap_serial
            if not batch:
                return
            max_length = max(len(item["window_ids"]) for item in batch)
            input_ids = torch.full(
                (len(batch), max_length),
                tokenizer.pad_token_id,
                dtype=torch.long,
            )
            attention_mask = torch.zeros_like(input_ids)
            for row, item in enumerate(batch):
                length = len(item["window_ids"])
                input_ids[row, :length] = torch.tensor(
                    item["window_ids"], dtype=torch.long
                )
                attention_mask[row, :length] = 1

            captured.clear()
            try:
                with torch.inference_mode():
                    try:
                        model(
                            input_ids=input_ids.to(model.device),
                            attention_mask=attention_mask.to(model.device),
                            use_cache=False,
                        )
                    except CapturedMlpInput:
                        pass
            finally:
                del input_ids, attention_mask
            if len(captured) != 1:
                raise RuntimeError(f"Expected one MLP-input capture, got {len(captured)}")
            traces = captured.pop()

            for row, item in enumerate(batch):
                score_start = item["score_start"]
                length = len(item["window_ids"])
                trace = traces[row, score_start:length].contiguous()
                trace.numpy().astype("<f4", copy=False).tofile(scores_file)
                scored_tokens += int(trace.numel())
                hit_offsets = torch.nonzero(trace > args.threshold).flatten().tolist()
                for relative_offset in hit_offsets:
                    window_position = score_start + relative_offset
                    document_position = item["window_start"] + window_position
                    activation = float(traces[row, window_position])
                    token_id = int(item["document_ids"][document_position])
                    target_token = tokenizer.decode(
                        [token_id], clean_up_tokenization_spaces=False
                    )
                    token_piece = tokenizer.convert_ids_to_tokens(token_id)
                    context_start = max(0, document_position - args.context_tokens)
                    context_end = min(
                        len(item["document_ids"]),
                        document_position + args.context_tokens + 1,
                    )
                    left_context = tokenizer.decode(
                        item["document_ids"][context_start:document_position],
                        clean_up_tokenization_spaces=False,
                    )
                    right_context = tokenizer.decode(
                        item["document_ids"][document_position + 1 : context_end],
                        clean_up_tokenization_spaces=False,
                    )
                    record = {
                        "dataset_index": item["dataset_index"],
                        "pile_set_name": item["pile_set_name"],
                        "document_token_index": document_position,
                        "token_id": token_id,
                        "token_piece": token_piece,
                        "target_token": target_token,
                        "activation": activation,
                        "left_context": left_context,
                        "right_context": right_context,
                        "marked_context": (
                            left_context + "⟦" + target_token + "⟧" + right_context
                        ),
                    }
                    jsonl_file.write(
                        json.dumps(record, ensure_ascii=False, default=json_default)
                        + "\n"
                    )
                    csv_writer.writerow(record)
                    hit_count += 1
                    key = (token_id, token_piece, target_token)
                    token_counts[key] += 1
                    token_activation_sum[key] += activation
                    token_activation_min[key] = min(
                        activation, token_activation_min.get(key, activation)
                    )
                    token_activation_max[key] = max(
                        activation, token_activation_max.get(key, activation)
                    )
                    if len(token_examples[key]) < 5:
                        token_examples[key].append(record)
                    heap_serial += 1
                    heap_item = (activation, heap_serial, record)
                    if len(top_hits) < 100:
                        heapq.heappush(top_hits, heap_item)
                    elif activation > top_hits[0][0]:
                        heapq.heapreplace(top_hits, heap_item)

            total_windows += len(batch)
            batch = []

        try:
            for dataset_index, example in enumerate(
                tqdm(dataset, total=len(dataset), desc="pile-10k documents")
            ):
                text = example["text"]
                document_ids = tokenizer.encode(text, add_special_tokens=False)
                document_start = total_tokens
                total_tokens += len(document_ids)
                meta = example.get("meta") or {}
                pile_set_name = (
                    meta.get("pile_set_name", "") if isinstance(meta, dict) else str(meta)
                )
                documents_file.write(
                    json.dumps(
                        {
                            "dataset_index": dataset_index,
                            "score_start": document_start,
                            "token_count": len(document_ids),
                            "pile_set_name": pile_set_name,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                new_start = 0
                while new_start < len(document_ids):
                    window_start = max(0, new_start - args.overlap_tokens)
                    window_end = min(
                        len(document_ids), window_start + args.window_tokens
                    )
                    score_start = new_start - window_start
                    batch.append(
                        {
                            "dataset_index": dataset_index,
                            "pile_set_name": pile_set_name,
                            "document_ids": document_ids,
                            "window_ids": document_ids[window_start:window_end],
                            "window_start": window_start,
                            "score_start": score_start,
                        }
                    )
                    new_start = window_end
                    if len(batch) >= args.batch_size:
                        process_batch()
                if dataset_index % 100 == 0:
                    jsonl_file.flush()
                    csv_file.flush()
                    scores_file.flush()
                    documents_file.flush()
            process_batch()
        finally:
            handle.remove()

    if scored_tokens != total_tokens:
        raise RuntimeError(
            f"Token accounting mismatch: scored={scored_tokens}, total={total_tokens}"
        )

    token_summary = []
    for key, count in token_counts.most_common():
        token_id, token_piece, target_token = key
        token_summary.append(
            {
                "token_id": token_id,
                "token_piece": token_piece,
                "target_token": target_token,
                "count": count,
                "mean_activation": token_activation_sum[key] / count,
                "min_activation": token_activation_min[key],
                "max_activation": token_activation_max[key],
                "examples": token_examples[key],
            }
        )

    summary = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "dataset_rows": len(dataset),
        "checkpoint_layer_count": checkpoint_layer_count,
        "loaded_layer_count": int(model.config.num_hidden_layers),
        "layer": args.layer,
        "neuron": args.neuron,
        "activation_definition": "silu(gate_proj(x)[N]) * up_proj(x)[N], where x is the target layer's normalized MLP input",
        "threshold_rule": f"activation > {args.threshold}",
        "batch_size": args.batch_size,
        "window_tokens": args.window_tokens,
        "overlap_tokens": args.overlap_tokens,
        "context_tokens_each_side": args.context_tokens,
        "touched_model_gib": touched / 1024**3,
        "total_tokens": total_tokens,
        "scored_tokens": scored_tokens,
        "scores": {
            "path": scores_path.name,
            "format": "headerless little-endian IEEE-754 float32",
            "shape": [scored_tokens],
            "order": "dataset row, then document token position",
            "document_index_path": documents_path.name,
        },
        "total_windows": total_windows,
        "hit_count": hit_count,
        "hit_rate": hit_count / scored_tokens if scored_tokens else 0.0,
        "unique_hit_token_ids": len(token_counts),
        "hit_tokens": token_summary,
        "top_100_hits": [
            item[2] for item in sorted(top_hits, key=lambda item: item[0], reverse=True)
        ],
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=json_default) + "\n",
        encoding="utf-8",
    )

    if token_summary:
        shown = token_summary[: min(30, len(token_summary))]
        fig, ax = plt.subplots(figsize=(12, max(4.5, 0.34 * len(shown))), dpi=170)
        labels = [
            f"{entry['target_token']!r}  (id {entry['token_id']})" for entry in shown
        ]
        values = [entry["count"] for entry in shown]
        positions = list(range(len(shown)))
        ax.barh(positions, values, color="#28627b")
        ax.set_yticks(positions, labels)
        ax.invert_yaxis()
        ax.set_xlabel(
            f"Positions with L{args.layer}/N{args.neuron} activation > {args.threshold:g}"
        )
        ax.set_title(f"{DATASET_ID}: high-activation token counts")
        ax.grid(axis="x", alpha=0.25)
        ax.grid(axis="y", visible=False)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(plot_path, bbox_inches="tight")
        plt.close(fig)

    print(f"Touched model pages: {touched / 1024**3:.2f} GiB")
    print(f"Dataset rows: {len(dataset):,}")
    print(f"Scored tokens: {scored_tokens:,}")
    print(f"Windows: {total_windows:,}")
    print(f"Hits (activation > {args.threshold:g}): {hit_count:,}")
    print(f"Unique hit token IDs: {len(token_counts):,}")
    for entry in token_summary[:20]:
        print(
            f"{entry['count']:>7,}  {entry['target_token']!r:<18} "
            f"id={entry['token_id']:<7} mean={entry['mean_activation']:.3f} "
            f"range=[{entry['min_activation']:.3f}, {entry['max_activation']:.3f}]"
        )
    print(f"Saved {jsonl_path.resolve()}")
    print(f"Saved {csv_path.resolve()}")
    print(f"Saved {summary_path.resolve()}")
    print(f"Saved {scores_path.resolve()}")
    print(f"Saved {documents_path.resolve()}")
    if token_summary:
        print(f"Saved {plot_path.resolve()}")


if __name__ == "__main__":
    main()
