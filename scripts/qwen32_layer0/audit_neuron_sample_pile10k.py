from __future__ import annotations

import argparse
import bisect
import csv
import heapq
import json
import random
import shutil
from pathlib import Path

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


class CapturedMlpInput(RuntimeError):
    """Stop after target-layer attention, before computing its complete MLP."""


def parse_neuron_list(value: str) -> list[int]:
    try:
        neurons = [int(part) for part in value.split(",") if part.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError("neurons must be comma-separated integers") from error
    if not neurons or any(neuron < 0 for neuron in neurons):
        raise argparse.ArgumentTypeError("neurons must contain non-negative integers")
    if len(neurons) != len(set(neurons)):
        raise argparse.ArgumentTypeError("neurons must be unique")
    return neurons


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Score a sample of Qwen3-32B neurons on every Pile-10k token in one pass, "
            "saving all scores and each neuron's two-sided extremes."
        )
    )
    parser.add_argument("--layer", type=int, default=0)
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--neurons",
        type=parse_neuron_list,
        help="comma-separated explicit neuron IDs",
    )
    group.add_argument("--neuron-file", type=Path, help="one neuron ID per line")
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--extreme-count", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--window-tokens", type=int, default=1024)
    parser.add_argument("--overlap-tokens", type=int, default=128)
    parser.add_argument("--context-tokens", type=int, default=32)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing output directory",
    )
    args = parser.parse_args()
    if args.layer < 0:
        parser.error("--layer must be non-negative")
    if args.sample_size < 1 or args.extreme_count < 1 or args.batch_size < 1:
        parser.error("--sample-size, --extreme-count, and --batch-size must be positive")
    if args.window_tokens < 2:
        parser.error("--window-tokens must be at least 2")
    if not 0 <= args.overlap_tokens < args.window_tokens:
        parser.error("--overlap-tokens must be in [0, window-tokens)")
    if args.context_tokens < 0:
        parser.error("--context-tokens must be non-negative")
    return args


def load_explicit_neurons(args: argparse.Namespace) -> list[int] | None:
    if args.neurons is not None:
        return args.neurons
    if args.neuron_file is None:
        return None
    neurons = [
        int(line.strip())
        for line in args.neuron_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not neurons or any(neuron < 0 for neuron in neurons):
        raise ValueError("neuron file must contain non-negative neuron IDs")
    if len(neurons) != len(set(neurons)):
        raise ValueError("neuron file contains duplicate neuron IDs")
    return neurons


def candidate(
    score: float,
    item: dict,
    document_position: int,
    token_id: int,
    context_tokens: int,
    serial: int,
) -> tuple[float, int, dict]:
    document_ids = item["document_ids"]
    context_start = max(0, document_position - context_tokens)
    context_end = min(len(document_ids), document_position + context_tokens + 1)
    return (
        score,
        serial,
        {
            "dataset_index": item["dataset_index"],
            "pile_set_name": item["pile_set_name"],
            "document_token_index": document_position,
            "token_id": token_id,
            "left_ids": document_ids[context_start:document_position],
            "right_ids": document_ids[document_position + 1 : context_end],
        },
    )


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run the Qwen3 neuron audit")

    explicit_neurons = load_explicit_neurons(args)
    output_dir = args.output_dir or Path(
        f"outputs/layer{args.layer}/sample_{args.sample_size}_seed_{args.seed}_pile10k"
    )
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}. Use --overwrite to replace it."
            )
        if not (output_dir / "summary.json").is_file():
            raise FileExistsError(
                f"Refusing to replace {output_dir}: it is not a completed scanner output directory"
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = AutoConfig.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    checkpoint_layer_count = int(config.num_hidden_layers)
    if args.layer >= checkpoint_layer_count:
        raise ValueError(f"Layer {args.layer} is outside the checkpoint")
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
    neuron_width = int(layer.mlp.gate_proj.out_features)
    if explicit_neurons is None:
        if args.sample_size > neuron_width:
            raise ValueError(f"Cannot sample {args.sample_size} from width {neuron_width}")
        neurons = sorted(random.Random(args.seed).sample(range(neuron_width), args.sample_size))
        selection_method = "uniform_without_replacement"
    else:
        neurons = explicit_neurons
        selection_method = "explicit"
    if any(neuron >= neuron_width for neuron in neurons):
        raise ValueError(f"Neuron IDs must be below layer width {neuron_width}")

    neuron_index = torch.tensor(neurons, device=layer.mlp.gate_proj.weight.device)
    gate_rows = layer.mlp.gate_proj.weight.index_select(0, neuron_index).detach()
    up_rows = layer.mlp.up_proj.weight.index_select(0, neuron_index).detach()
    captured: list[torch.Tensor] = []

    def capture_neurons(
        _module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]
    ) -> None:
        hidden = inputs[0]
        activations = F.silu(F.linear(hidden, gate_rows)) * F.linear(hidden, up_rows)
        captured.append(activations.detach().float().cpu())
        raise CapturedMlpInput

    handle = layer.mlp.register_forward_pre_hook(capture_neurons)
    dataset = load_dataset(DATASET_ID, split="train", revision=DATASET_REVISION)

    scores_path = output_dir / "scores.f32"
    documents_path = output_dir / "documents.jsonl"
    extremes_jsonl_path = output_dir / "extremes.jsonl"
    extremes_csv_path = output_dir / "extremes.csv"
    thresholds_csv_path = output_dir / "thresholds.csv"
    summary_path = output_dir / "summary.json"
    neurons_path = output_dir / "neurons.json"
    neurons_path.write_text(json.dumps(neurons, indent=2) + "\n", encoding="utf-8")

    top_heaps: list[list[tuple[float, int, dict]]] = [[] for _ in neurons]
    # Negative score keys make these min-heaps retain the most negative values.
    bottom_heaps: list[list[tuple[float, int, dict]]] = [[] for _ in neurons]
    batch: list[dict] = []
    total_tokens = scored_tokens = total_windows = serial = 0

    with scores_path.open("wb") as scores_file, documents_path.open(
        "w", encoding="utf-8"
    ) as documents_file:

        def process_batch() -> None:
            nonlocal batch, scored_tokens, total_windows, serial
            if not batch:
                return
            max_length = max(len(item["window_ids"]) for item in batch)
            input_ids = torch.full(
                (len(batch), max_length), tokenizer.pad_token_id, dtype=torch.long
            )
            attention_mask = torch.zeros_like(input_ids)
            for row, item in enumerate(batch):
                length = len(item["window_ids"])
                input_ids[row, :length] = torch.tensor(item["window_ids"])
                attention_mask[row, :length] = 1

            captured.clear()
            with torch.inference_mode():
                try:
                    model(
                        input_ids=input_ids.to(model.device),
                        attention_mask=attention_mask.to(model.device),
                        use_cache=False,
                    )
                except CapturedMlpInput:
                    pass
            if len(captured) != 1:
                raise RuntimeError(f"Expected one capture, got {len(captured)}")
            traces = captured.pop()

            valid_traces = []
            trace_ends = []
            flat_length = 0
            for row, item in enumerate(batch):
                length = len(item["window_ids"])
                trace = traces[row, item["score_start"] : length].contiguous()
                trace.numpy().astype("<f4", copy=False).tofile(scores_file)
                scored_tokens += trace.shape[0]
                valid_traces.append(trace)
                flat_length += trace.shape[0]
                trace_ends.append(flat_length)

            # Merge all non-padding positions in the batch before top-k. This reduces
            # Python heap work from 40 candidates per window/neuron to 40 per batch/neuron.
            flat_trace = torch.cat(valid_traces, dim=0)
            local_count = min(args.extreme_count, flat_trace.shape[0])
            high_values, high_offsets = torch.topk(flat_trace, local_count, dim=0)
            low_values, low_offsets = torch.topk(
                flat_trace, local_count, dim=0, largest=False
            )
            for column in range(len(neurons)):
                for values, offsets, heap, polarity in (
                    (high_values, high_offsets, top_heaps[column], 1),
                    (low_values, low_offsets, bottom_heaps[column], -1),
                ):
                    for rank in range(local_count):
                        score = float(values[rank, column])
                        key = score if polarity > 0 else -score
                        if len(heap) >= args.extreme_count and key <= heap[0][0]:
                            continue
                        flat_offset = int(offsets[rank, column])
                        row = bisect.bisect_right(trace_ends, flat_offset)
                        row_start = 0 if row == 0 else trace_ends[row - 1]
                        offset = flat_offset - row_start
                        item = batch[row]
                        window_position = item["score_start"] + offset
                        document_position = item["window_start"] + window_position
                        token_id = int(item["document_ids"][document_position])
                        serial += 1
                        entry = candidate(
                            score,
                            item,
                            document_position,
                            token_id,
                            args.context_tokens,
                            serial,
                        )
                        keyed_entry = entry if polarity > 0 else (-score, entry[1], entry[2])
                        if len(heap) < args.extreme_count:
                            heapq.heappush(heap, keyed_entry)
                        else:
                            heapq.heapreplace(heap, keyed_entry)
            total_windows += len(batch)
            batch = []

        try:
            for dataset_index, example in enumerate(
                tqdm(dataset, total=len(dataset), desc="pile-10k documents")
            ):
                document_ids = tokenizer.encode(example["text"], add_special_tokens=False)
                meta = example.get("meta") or {}
                pile_set_name = meta.get("pile_set_name", "") if isinstance(meta, dict) else str(meta)
                documents_file.write(
                    json.dumps(
                        {
                            "dataset_index": dataset_index,
                            "score_start": total_tokens,
                            "token_count": len(document_ids),
                            "pile_set_name": pile_set_name,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                total_tokens += len(document_ids)
                new_start = 0
                while new_start < len(document_ids):
                    window_start = max(0, new_start - args.overlap_tokens)
                    window_end = min(len(document_ids), window_start + args.window_tokens)
                    batch.append(
                        {
                            "dataset_index": dataset_index,
                            "pile_set_name": pile_set_name,
                            "document_ids": document_ids,
                            "window_ids": document_ids[window_start:window_end],
                            "window_start": window_start,
                            "score_start": new_start - window_start,
                        }
                    )
                    new_start = window_end
                    if len(batch) >= args.batch_size:
                        process_batch()
                if dataset_index % 100 == 0:
                    scores_file.flush()
                    documents_file.flush()
            process_batch()
        finally:
            handle.remove()

    if scored_tokens != total_tokens:
        raise RuntimeError(f"Token accounting mismatch: {scored_tokens} != {total_tokens}")

    fields = [
        "layer", "neuron", "polarity", "rank", "activation", "dataset_index",
        "pile_set_name", "document_token_index", "token_id", "token_piece",
        "target_token", "left_context", "right_context", "marked_context",
    ]
    records = []
    for column, neuron in enumerate(neurons):
        sides = (
            ("positive", sorted(top_heaps[column], key=lambda item: item[0], reverse=True)),
            ("negative", sorted(bottom_heaps[column], key=lambda item: item[0], reverse=True)),
        )
        for polarity, entries in sides:
            for rank, heap_entry in enumerate(entries, start=1):
                score = heap_entry[0] if polarity == "positive" else -heap_entry[0]
                location = heap_entry[2]
                token_id = location["token_id"]
                target_token = tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
                left_context = tokenizer.decode(location["left_ids"], clean_up_tokenization_spaces=False)
                right_context = tokenizer.decode(location["right_ids"], clean_up_tokenization_spaces=False)
                records.append(
                    {
                        "layer": args.layer,
                        "neuron": neuron,
                        "polarity": polarity,
                        "rank": rank,
                        "activation": score,
                        "dataset_index": location["dataset_index"],
                        "pile_set_name": location["pile_set_name"],
                        "document_token_index": location["document_token_index"],
                        "token_id": token_id,
                        "token_piece": tokenizer.convert_ids_to_tokens(token_id),
                        "target_token": target_token,
                        "left_context": left_context,
                        "right_context": right_context,
                        "marked_context": left_context + "⟦" + target_token + "⟧" + right_context,
                    }
                )

    with extremes_jsonl_path.open("w", encoding="utf-8") as jsonl_file, extremes_csv_path.open(
        "w", newline="", encoding="utf-8"
    ) as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        for record in records:
            jsonl_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            writer.writerow(record)

    threshold_rows = []
    for column, neuron in enumerate(neurons):
        positive_scores = sorted(
            (entry[0] for entry in top_heaps[column]), reverse=True
        )
        negative_scores = sorted(-entry[0] for entry in bottom_heaps[column])
        threshold_rows.append(
            {
                "layer": args.layer,
                "neuron": neuron,
                "positive_cutoff_20th_largest": positive_scores[-1],
                "positive_maximum": positive_scores[0],
                "negative_cutoff_20th_smallest": negative_scores[-1],
                "negative_minimum": negative_scores[0],
            }
        )
    with thresholds_csv_path.open("w", newline="", encoding="utf-8") as threshold_file:
        threshold_writer = csv.DictWriter(
            threshold_file, fieldnames=list(threshold_rows[0])
        )
        threshold_writer.writeheader()
        threshold_writer.writerows(threshold_rows)

    summary = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "dataset_rows": len(dataset),
        "layer": args.layer,
        "neuron_width": neuron_width,
        "neurons": neurons,
        "neuron_count": len(neurons),
        "selection_method": selection_method,
        "sample_seed": args.seed if selection_method != "explicit" else None,
        "activation_definition": "silu(gate_proj(x)[N]) * up_proj(x)[N]",
        "extremes_per_side": args.extreme_count,
        "touched_model_gib": touched / 1024**3,
        "total_tokens": total_tokens,
        "scored_tokens_per_neuron": scored_tokens,
        "total_score_values": scored_tokens * len(neurons),
        "scores": {
            "path": scores_path.name,
            "format": "headerless little-endian IEEE-754 float32",
            "shape": [scored_tokens, len(neurons)],
            "order": "C order: dataset token position, then neuron column",
            "neuron_columns_path": neurons_path.name,
            "document_index_path": documents_path.name,
        },
        "extremes": {
            "jsonl_path": extremes_jsonl_path.name,
            "csv_path": extremes_csv_path.name,
            "thresholds_csv_path": thresholds_csv_path.name,
            "row_count": len(records),
            "selection_rule": (
                "exactly N largest and N smallest occurrences per neuron; "
                "boundary ties are resolved by encounter order"
            ),
        },
        "batch_size": args.batch_size,
        "window_tokens": args.window_tokens,
        "overlap_tokens": args.overlap_tokens,
        "context_tokens_each_side": args.context_tokens,
        "total_windows": total_windows,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    expected_bytes = scored_tokens * len(neurons) * np.dtype("<f4").itemsize
    if scores_path.stat().st_size != expected_bytes:
        raise RuntimeError("Dense score file size does not match its declared shape")
    print(f"Touched model pages: {touched / 1024**3:.2f} GiB")
    print(f"Neurons: {len(neurons):,}")
    print(f"Scored tokens per neuron: {scored_tokens:,}")
    print(f"Total score values: {scored_tokens * len(neurons):,}")
    print(f"Extreme records: {len(records):,}")
    print(f"Saved {output_dir.resolve()}")


if __name__ == "__main__":
    main()
