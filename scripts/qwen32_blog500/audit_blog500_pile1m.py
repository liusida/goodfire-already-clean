from __future__ import annotations

import argparse
import bisect
import csv
import heapq
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from gb10_load_llm import load_model_to_cuda
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoTokenizer, Qwen3Model


MODEL_ID = "Qwen/Qwen3-32B"
MODEL_REVISION = "9216db5781bf21249d130ec9da846c4624c16137"
DATASET_ID = "NeelNanda/pile-10k"
DATASET_REVISION = "127bfedcd5047750df5ccf3a12979a47bfa0bafa"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure the 500 Qwen3-32B neurons catalogued by the Already Clean "
            "blog on exactly one million Pile-10k tokens."
        )
    )
    parser.add_argument(
        "--neuron-csv",
        type=Path,
        default=Path("qwen3-32b-neuron-papers.csv"),
    )
    parser.add_argument("--num-tokens", type=int, default=1_000_000)
    parser.add_argument("--extreme-count", type=int, default=20)
    parser.add_argument("--bulk-absolute-quantile", type=float, default=0.90)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--window-tokens", type=int, default=512)
    parser.add_argument("--overlap-tokens", type=int, default=64)
    parser.add_argument("--context-tokens", type=int, default=32)
    parser.add_argument("--stats-chunk-rows", type=int, default=100_000)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/blog500_pile1m"),
    )
    parser.add_argument(
        "--keep-scores",
        action="store_true",
        help="retain the approximately 2 GB of per-layer float32 score matrices",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace a completed output directory",
    )
    args = parser.parse_args()
    if args.num_tokens < 1 or args.extreme_count < 1 or args.batch_size < 1:
        parser.error("--num-tokens, --extreme-count, and --batch-size must be positive")
    if args.window_tokens < 2:
        parser.error("--window-tokens must be at least 2")
    if not 0 <= args.overlap_tokens < args.window_tokens:
        parser.error("--overlap-tokens must be in [0, window-tokens)")
    if args.context_tokens < 0 or args.stats_chunk_rows < 1:
        parser.error("--context-tokens must be non-negative and --stats-chunk-rows positive")
    if not 0.5 <= args.bulk_absolute_quantile < 1:
        parser.error("--bulk-absolute-quantile must be in [0.5, 1)")
    return args


def read_catalog(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    required = {"model", "layer", "neuron", "id", "title", "url"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"{path} must contain columns {sorted(required)}")
    catalog = []
    seen: set[tuple[int, int]] = set()
    for row in rows:
        if row["model"] != "Qwen3-32B":
            raise ValueError(f"Unexpected model in catalog: {row['model']!r}")
        layer, neuron = int(row["layer"]), int(row["neuron"])
        key = (layer, neuron)
        if key in seen:
            raise ValueError(f"Duplicate catalog neuron L{layer}/N{neuron}")
        seen.add(key)
        catalog.append({**row, "layer": layer, "neuron": neuron})
    if len(catalog) != 500:
        raise ValueError(f"Expected 500 catalog neurons, found {len(catalog)}")
    return catalog


def prepare_output(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {path}")
        if not (path / "summary.json").is_file():
            raise FileExistsError(
                f"Refusing to replace {path}: no completed summary.json was found"
            )
        shutil.rmtree(path)
    path.mkdir(parents=True)


def context_candidate(
    *,
    item: dict,
    document_position: int,
    token_id: int,
    context_tokens: int,
) -> dict:
    ids = item["document_ids"]
    start = max(0, document_position - context_tokens)
    end = min(len(ids), document_position + context_tokens + 1)
    return {
        "dataset_index": item["dataset_index"],
        "pile_set_name": item["pile_set_name"],
        "document_token_index": document_position,
        "token_id": token_id,
        "left_ids": ids[start:document_position],
        "right_ids": ids[document_position + 1 : end],
    }


def safe_ratio(numerator: float, denominator: float) -> float | None:
    if not math.isfinite(numerator) or not math.isfinite(denominator) or denominator <= 0:
        return None
    return numerator / denominator


def distribution_metrics(
    scores_path: Path,
    *,
    rows: int,
    columns: int,
    chunk_rows: int,
    bulk_absolute_quantile: float,
) -> list[dict]:
    scores = np.memmap(scores_path, mode="r", dtype="<f4", shape=(rows, columns))
    total = np.zeros(columns, dtype=np.float64)
    minimum = np.full(columns, np.inf)
    maximum = np.full(columns, -np.inf)
    for start in range(0, rows, chunk_rows):
        chunk = np.asarray(scores[start : start + chunk_rows], dtype=np.float64)
        total += chunk.sum(axis=0, dtype=np.float64)
        minimum = np.minimum(minimum, chunk.min(axis=0))
        maximum = np.maximum(maximum, chunk.max(axis=0))
    mean = total / rows

    sums_2 = np.zeros(columns, dtype=np.float64)
    sums_3 = np.zeros(columns, dtype=np.float64)
    sums_4 = np.zeros(columns, dtype=np.float64)
    for start in range(0, rows, chunk_rows):
        chunk = np.asarray(scores[start : start + chunk_rows], dtype=np.float64)
        delta = chunk - mean
        delta_2 = delta * delta
        sums_2 += delta_2.sum(axis=0, dtype=np.float64)
        sums_3 += (delta_2 * delta).sum(axis=0, dtype=np.float64)
        sums_4 += (delta_2 * delta_2).sum(axis=0, dtype=np.float64)
    variance = sums_2 / rows
    std = np.sqrt(variance)
    with np.errstate(divide="ignore", invalid="ignore"):
        skewness = (sums_3 / rows) / std**3
        pearson_kurtosis = (sums_4 / rows) / variance**2

    # The largest catalog layer contains only 25 selected neurons, so loading one
    # layer's complete score matrix for an exact axis-wise quantile is modest.
    bulk_scale = np.quantile(
        np.abs(np.asarray(scores, dtype=np.float32)),
        bulk_absolute_quantile,
        axis=0,
    )
    del scores
    return [
        {
            "count": rows,
            "minimum": float(minimum[column]),
            "maximum": float(maximum[column]),
            "mean": float(mean[column]),
            "population_variance": float(variance[column]),
            "population_standard_deviation": float(std[column]),
            "population_skewness": (
                float(skewness[column]) if np.isfinite(skewness[column]) else None
            ),
            "pearson_kurtosis": (
                float(pearson_kurtosis[column])
                if np.isfinite(pearson_kurtosis[column])
                else None
            ),
            "excess_kurtosis": (
                float(pearson_kurtosis[column] - 3.0)
                if np.isfinite(pearson_kurtosis[column])
                else None
            ),
            "bulk_absolute_quantile": bulk_absolute_quantile,
            "bulk_absolute_scale": float(bulk_scale[column]),
        }
        for column in range(columns)
    ]


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    catalog = read_catalog(args.neuron_csv)
    prepare_output(args.output_dir, args.overwrite)
    catalog_snapshot_path = args.output_dir / "catalog.json"
    catalog_snapshot_path.write_text(
        json.dumps(catalog, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    by_layer: dict[int, list[dict]] = defaultdict(list)
    for row in catalog:
        by_layer[row["layer"]].append(row)
    selected_layers = sorted(by_layer)

    config = AutoConfig.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    if max(selected_layers) >= int(config.num_hidden_layers):
        raise ValueError("Catalog contains a layer outside the checkpoint")
    intermediate_size = int(config.intermediate_size)
    for layer, rows in by_layer.items():
        for row in rows:
            if not 0 <= row["neuron"] < intermediate_size:
                raise ValueError(f"L{layer}/N{row['neuron']} is outside the MLP width")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(
        "Loading the complete Qwen3-32B BF16 backbone. This requires roughly "
        "64 GiB for parameters before batch activations and loader overhead."
    )
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

    layer_neurons = {
        layer: [row["neuron"] for row in rows] for layer, rows in by_layer.items()
    }
    layer_indices = {
        layer: torch.tensor(neurons, device=model.device, dtype=torch.long)
        for layer, neurons in layer_neurons.items()
    }
    captures: dict[int, torch.Tensor] = {}

    def make_hook(layer: int):
        def capture(_module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
            if layer in captures:
                raise RuntimeError(f"Layer {layer} hook fired more than once")
            captures[layer] = (
                inputs[0]
                .index_select(-1, layer_indices[layer])
                .detach()
                .float()
                .cpu()
            )

        return capture

    handles = [
        model.layers[layer].mlp.down_proj.register_forward_pre_hook(make_hook(layer))
        for layer in selected_layers
    ]

    score_dir = args.output_dir / "scores_by_layer"
    score_dir.mkdir()
    score_paths = {layer: score_dir / f"layer_{layer:02d}.f32" for layer in selected_layers}
    score_files = {layer: path.open("wb") for layer, path in score_paths.items()}
    top_heaps = {
        layer: [[] for _ in rows] for layer, rows in by_layer.items()
    }
    bottom_heaps = {
        layer: [[] for _ in rows] for layer, rows in by_layer.items()
    }
    dataset = load_dataset(DATASET_ID, split="train", revision=DATASET_REVISION)
    documents_path = args.output_dir / "documents.jsonl"
    batch: list[dict] = []
    total_tokens = scored_tokens = total_windows = serial = 0

    with documents_path.open("w", encoding="utf-8") as documents_file:

        def process_batch() -> None:
            nonlocal batch, scored_tokens, total_windows, serial
            if not batch:
                return
            max_length = max(len(item["window_ids"]) for item in batch)
            input_ids = torch.full(
                (len(batch), max_length), tokenizer.pad_token_id, dtype=torch.long
            )
            attention_mask = torch.zeros_like(input_ids)
            for row_index, item in enumerate(batch):
                length = len(item["window_ids"])
                input_ids[row_index, :length] = torch.tensor(item["window_ids"])
                attention_mask[row_index, :length] = 1

            captures.clear()
            with torch.inference_mode():
                model(
                    input_ids=input_ids.to(model.device),
                    attention_mask=attention_mask.to(model.device),
                    use_cache=False,
                )
            if set(captures) != set(selected_layers):
                missing = sorted(set(selected_layers) - set(captures))
                raise RuntimeError(f"Missing MLP captures for layers {missing}")

            expected_batch_tokens = sum(
                len(item["window_ids"]) - item["score_start"] for item in batch
            )
            for layer in selected_layers:
                valid_traces = []
                trace_ends = []
                flat_length = 0
                traces = captures.pop(layer)
                for row_index, item in enumerate(batch):
                    length = len(item["window_ids"])
                    trace = traces[
                        row_index, item["score_start"] : length
                    ].contiguous()
                    trace.numpy().astype("<f4", copy=False).tofile(score_files[layer])
                    valid_traces.append(trace)
                    flat_length += trace.shape[0]
                    trace_ends.append(flat_length)
                if flat_length != expected_batch_tokens:
                    raise RuntimeError("Layer token accounting mismatch")

                flat_trace = torch.cat(valid_traces, dim=0)
                local_count = min(args.extreme_count, flat_trace.shape[0])
                high_values, high_offsets = torch.topk(
                    flat_trace, local_count, dim=0
                )
                low_values, low_offsets = torch.topk(
                    flat_trace, local_count, dim=0, largest=False
                )
                for column in range(len(by_layer[layer])):
                    for values, offsets, heap, polarity in (
                        (high_values, high_offsets, top_heaps[layer][column], 1),
                        (low_values, low_offsets, bottom_heaps[layer][column], -1),
                    ):
                        for local_rank in range(local_count):
                            score = float(values[local_rank, column])
                            key = score if polarity > 0 else -score
                            if len(heap) >= args.extreme_count and key <= heap[0][0]:
                                continue
                            flat_offset = int(offsets[local_rank, column])
                            batch_row = bisect.bisect_right(trace_ends, flat_offset)
                            row_start = 0 if batch_row == 0 else trace_ends[batch_row - 1]
                            offset = flat_offset - row_start
                            item = batch[batch_row]
                            window_position = item["score_start"] + offset
                            document_position = item["window_start"] + window_position
                            token_id = int(item["document_ids"][document_position])
                            serial += 1
                            location = context_candidate(
                                item=item,
                                document_position=document_position,
                                token_id=token_id,
                                context_tokens=args.context_tokens,
                            )
                            entry = (key, serial, location)
                            if len(heap) < args.extreme_count:
                                heapq.heappush(heap, entry)
                            else:
                                heapq.heapreplace(heap, entry)
            scored_tokens += expected_batch_tokens
            progress.update(expected_batch_tokens)
            total_windows += len(batch)
            batch = []

        try:
            progress = tqdm(total=args.num_tokens, unit="tok", desc="Pile-10k")
            for dataset_index, example in enumerate(dataset):
                remaining = args.num_tokens - total_tokens
                if remaining <= 0:
                    break
                # Some Pile documents exceed the model's advertised context length,
                # but they are always split into short windows below. Suppress the
                # tokenizer's whole-document warning; no oversized sequence is sent
                # to the model.
                document_ids = tokenizer(
                    example["text"], add_special_tokens=False, verbose=False
                )["input_ids"][:remaining]
                if not document_ids:
                    continue
                meta = example.get("meta") or {}
                pile_set_name = (
                    meta.get("pile_set_name", "")
                    if isinstance(meta, dict)
                    else str(meta)
                )
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
                    window_end = min(
                        len(document_ids), window_start + args.window_tokens
                    )
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
            process_batch()
            progress.close()
        finally:
            for handle in handles:
                handle.remove()
            for file in score_files.values():
                file.close()

    if total_tokens != args.num_tokens or scored_tokens != args.num_tokens:
        raise RuntimeError(
            f"Token accounting mismatch: source={total_tokens}, scored={scored_tokens}, "
            f"requested={args.num_tokens}"
        )

    metrics_by_key: dict[tuple[int, int], dict] = {}
    for layer in tqdm(selected_layers, desc="Exact distribution statistics"):
        rows = by_layer[layer]
        expected_bytes = args.num_tokens * len(rows) * np.dtype("<f4").itemsize
        if score_paths[layer].stat().st_size != expected_bytes:
            raise RuntimeError(f"Score size mismatch for layer {layer}")
        layer_metrics = distribution_metrics(
            score_paths[layer],
            rows=args.num_tokens,
            columns=len(rows),
            chunk_rows=args.stats_chunk_rows,
            bulk_absolute_quantile=args.bulk_absolute_quantile,
        )
        for column, (catalog_row, metrics) in enumerate(zip(rows, layer_metrics, strict=True)):
            positive = sorted(
                top_heaps[layer][column], key=lambda item: item[0], reverse=True
            )
            negative = sorted(
                bottom_heaps[layer][column], key=lambda item: item[0], reverse=True
            )
            positive_scores = [entry[0] for entry in positive]
            negative_scores = [-entry[0] for entry in negative]
            positive_raw = max(positive_scores[-1], 0.0)
            negative_raw = max(-negative_scores[-1], 0.0)
            bulk = metrics["bulk_absolute_scale"]
            positive_normalized = safe_ratio(positive_raw, bulk)
            negative_normalized = safe_ratio(negative_raw, bulk)
            unified = (
                max(positive_normalized, negative_normalized)
                if positive_normalized is not None and negative_normalized is not None
                else None
            )
            raw_sum = positive_raw + negative_raw
            metrics.update(
                {
                    "layer": layer,
                    "neuron": catalog_row["neuron"],
                    "id": catalog_row["id"],
                    "title": catalog_row["title"],
                    "status": catalog_row.get("status", ""),
                    "interest_score": catalog_row.get("interest_score", ""),
                    "cleanliness_score": catalog_row.get("cleanliness_score", ""),
                    "url": catalog_row["url"],
                    "positive_cutoff_kth_largest": positive_scores[-1],
                    "negative_cutoff_kth_smallest": negative_scores[-1],
                    "positive_maximum": positive_scores[0],
                    "negative_minimum": negative_scores[0],
                    "positive_zero_anchored_tail": positive_raw,
                    "negative_zero_anchored_tail": negative_raw,
                    "positive_normalized_zero_tail": positive_normalized,
                    "negative_normalized_zero_tail": negative_normalized,
                    "normalized_zero_anchored_tail_score": unified,
                    "tail_direction": (
                        "positive"
                        if positive_raw >= negative_raw
                        else "negative"
                    ),
                    "tail_asymmetry": (
                        (positive_raw - negative_raw) / raw_sum
                        if raw_sum > 0
                        else None
                    ),
                }
            )
            metrics_by_key[(layer, catalog_row["neuron"])] = metrics

    metric_rows = [metrics_by_key[(row["layer"], row["neuron"])] for row in catalog]
    metrics_json_path = args.output_dir / "neuron_metrics.json"
    metrics_csv_path = args.output_dir / "neuron_metrics.csv"
    metrics_json_path.write_text(
        json.dumps(metric_rows, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    with metrics_csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(metric_rows[0]))
        writer.writeheader()
        writer.writerows(metric_rows)

    extreme_records = []
    for catalog_row in catalog:
        layer, neuron = catalog_row["layer"], catalog_row["neuron"]
        column = layer_neurons[layer].index(neuron)
        for polarity, entries in (
            (
                "positive",
                sorted(top_heaps[layer][column], key=lambda item: item[0], reverse=True),
            ),
            (
                "negative",
                sorted(bottom_heaps[layer][column], key=lambda item: item[0], reverse=True),
            ),
        ):
            for rank, heap_entry in enumerate(entries, start=1):
                activation = heap_entry[0] if polarity == "positive" else -heap_entry[0]
                location = heap_entry[2]
                token_id = location["token_id"]
                target = tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
                left = tokenizer.decode(
                    location["left_ids"], clean_up_tokenization_spaces=False
                )
                right = tokenizer.decode(
                    location["right_ids"], clean_up_tokenization_spaces=False
                )
                extreme_records.append(
                    {
                        "layer": layer,
                        "neuron": neuron,
                        "id": catalog_row["id"],
                        "title": catalog_row["title"],
                        "status": catalog_row.get("status", ""),
                        "interest_score": catalog_row.get("interest_score", ""),
                        "cleanliness_score": catalog_row.get("cleanliness_score", ""),
                        "url": catalog_row["url"],
                        "polarity": polarity,
                        "rank": rank,
                        "activation": activation,
                        "dataset_index": location["dataset_index"],
                        "pile_set_name": location["pile_set_name"],
                        "document_token_index": location["document_token_index"],
                        "token_id": token_id,
                        "token_piece": tokenizer.convert_ids_to_tokens(token_id),
                        "target_token": target,
                        "left_context": left,
                        "right_context": right,
                        "marked_context": left + "⟦" + target + "⟧" + right,
                    }
                )

    extremes_jsonl_path = args.output_dir / "extremes.jsonl"
    extremes_csv_path = args.output_dir / "extremes.csv"
    fields = list(extreme_records[0])
    with extremes_jsonl_path.open("w", encoding="utf-8") as jsonl_file, extremes_csv_path.open(
        "w", newline="", encoding="utf-8"
    ) as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        for record in extreme_records:
            jsonl_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            writer.writerow(record)

    summary = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_dtype": "bfloat16",
        "full_model_memory_note": (
            "All 64 layers are required because the catalog spans the complete model; "
            "BF16 parameters require roughly 64 GiB before runtime overhead."
        ),
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "token_count": args.num_tokens,
        "catalog_path": str(args.neuron_csv),
        "catalog_snapshot_path": catalog_snapshot_path.name,
        "neuron_count": len(catalog),
        "layers": selected_layers,
        "extremes_per_side": args.extreme_count,
        "bulk_absolute_quantile": args.bulk_absolute_quantile,
        "normalized_tail_definition": (
            "max(max(-kth_smallest,0), max(kth_largest,0)) / "
            "quantile(abs(activation), bulk_absolute_quantile)"
        ),
        "activation_definition": "silu(gate_proj(x)[N]) * up_proj(x)[N]",
        "metrics": {
            "json_path": metrics_json_path.name,
            "csv_path": metrics_csv_path.name,
            "moment_convention": "population central moments; Pearson kurtosis and excess kurtosis",
        },
        "extremes": {
            "jsonl_path": extremes_jsonl_path.name,
            "csv_path": extremes_csv_path.name,
            "row_count": len(extreme_records),
        },
        "documents_path": documents_path.name,
        "batch_size": args.batch_size,
        "window_tokens": args.window_tokens,
        "overlap_tokens": args.overlap_tokens,
        "context_tokens_each_side": args.context_tokens,
        "total_windows": total_windows,
        "touched_model_gib": touched / 1024**3,
        "scores_retained": args.keep_scores,
        "scores_directory": score_dir.name if args.keep_scores else None,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    if not args.keep_scores:
        shutil.rmtree(score_dir)

    print(f"Measured {len(catalog)} neurons across {len(selected_layers)} layers")
    print(f"Scored {scored_tokens:,} tokens per neuron")
    print(f"Saved {len(extreme_records):,} extreme occurrences")
    print(f"Saved {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
