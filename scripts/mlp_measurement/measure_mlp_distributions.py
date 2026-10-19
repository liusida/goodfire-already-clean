from __future__ import annotations

import argparse
import ast
import bisect
import html
import json
import math
import random
from pathlib import Path
from typing import Iterator

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from datasets.exceptions import DatasetNotFoundError
from gb10_load_llm import load_model_to_cuda
from huggingface_hub import HfApi, HfFileSystem
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


DEFAULT_MODEL_ID = "Qwen/Qwen3.5-0.8B-Base"
DEFAULT_MODEL_REVISION = "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"
MODEL_ALIASES = {
    "0.8b": DEFAULT_MODEL_ID,
    "9b": "Qwen/Qwen3.5-9B-Base",
}
# Backward-compatible defaults used by the companion extremes script.
MODEL_ID = DEFAULT_MODEL_ID
MODEL_REVISION = DEFAULT_MODEL_REVISION
EXTREME_OCCURRENCES = 10
# E[log(cosh(V))] for V ~ N(0, 1), evaluated by Gauss-Hermite quadrature.
ICA_GAUSSIAN_LOG_COSH_EXPECTATION = 0.37456720749143835
DATASETS = {
    "en": {
        "dataset_id": "Skylion007/openwebtext",
        "text_field": "text",
        "label": "OpenWebText (English)",
    },
    "cn": {
        "dataset_id": "BAAI/CCI3-Data",
        "text_field": "content",
        "label": "CCI3-Data (Chinese)",
    },
}
CCI3_DENYLIST = {
    "data/part-00705-6f0afd98-d375-4d7f-8299-ac5e070bf4fc-c000.jsonl",
}


class LastLayerCaptured(Exception):
    """Stop the model after the highest requested activation is measured."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure raw and magnitude distributions of selected "
            "post-SwiGLU MLP layers."
        )
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_ID,
        help=(
            "Hugging Face model ID, or a Qwen alias such as 0.8B or 9B "
            f"(default: {DEFAULT_MODEL_ID})."
        ),
    )
    parser.add_argument(
        "--revision",
        default=None,
        help=(
            "Optional Hugging Face revision. The default 0.8B model uses its "
            "pinned project revision; other models default to their main revision."
        ),
    )
    parser.add_argument(
        "--layers",
        default="0",
        help='Layer selection: "all", one layer ("7"), or a list ("[0,1,2]" or "0,1,2").',
    )
    parser.add_argument(
        "--neurons",
        default="all",
        help=(
            'Neuron selection: "all", one neuron ("2026"), or an independently '
            'measured list ("[2,279]" or "2,279"). "all" remains pooled.'
        ),
    )
    parser.add_argument(
        "--dataset",
        choices=("en", "cn", "both"),
        default="en",
        help=(
            "Corpus selection: en=OpenWebText, cn=BAAI CCI3-Data, or both. "
            "--num-tokens is sampled independently from each selected corpus."
        ),
    )
    parser.add_argument("--num-tokens", type=int, default=10_000)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--context-tokens",
        type=int,
        default=24,
        help="Number of source tokens to retain on each side of an extreme token.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle-buffer", type=int, default=10_000)
    parser.add_argument("--bins", type=int, default=400)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/mlp_measurement"),
    )
    return parser.parse_args()


def resolve_model(model: str, revision: str | None) -> tuple[str, str | None]:
    model_id = MODEL_ALIASES.get(model.strip().lower(), model.strip())
    if not model_id:
        raise ValueError("--model must not be empty")
    if revision is None and model_id == DEFAULT_MODEL_ID:
        revision = DEFAULT_MODEL_REVISION
    return model_id, revision


def parse_selection(spec: str, size: int, name: str) -> list[int]:
    cleaned = spec.strip()
    if cleaned.lower() == "all":
        return list(range(size))

    try:
        value = ast.literal_eval(cleaned)
    except (SyntaxError, ValueError):
        value = [part.strip() for part in cleaned.split(",")]

    if isinstance(value, int):
        requested = [value]
    elif isinstance(value, (list, tuple)):
        requested = []
        for item in value:
            if isinstance(item, bool):
                raise ValueError(f"Invalid {name} selection: {spec!r}")
            requested.append(int(item))
    else:
        raise ValueError(f"Invalid {name} selection: {spec!r}")

    selected = sorted(set(requested))
    if not selected:
        raise ValueError(f"At least one {name} is required")
    invalid = [index for index in selected if index < 0 or index >= size]
    if invalid:
        raise ValueError(
            f"{name.capitalize()}s {invalid} are outside the valid range 0..{size - 1}"
        )
    return selected


def parse_layers(spec: str, num_layers: int) -> list[int]:
    return parse_selection(spec, num_layers, "layer")


def parse_neurons(spec: str, intermediate_size: int) -> list[int]:
    return parse_selection(spec, intermediate_size, "neuron")


def neuron_measurement_groups(spec: str, neurons: list[int]) -> list[list[int]]:
    """Split explicit multi-neuron selections into independent measurements."""
    if spec.strip().lower() == "all" or len(neurons) <= 1:
        return [neurons]
    return [[neuron] for neuron in neurons]


def neuron_title(neurons: list[int], intermediate_size: int) -> str:
    if len(neurons) == intermediate_size:
        return f"all {intermediate_size:,} neurons"
    if len(neurons) <= 8:
        return "neurons " + ", ".join(f"N{neuron}" for neuron in neurons)
    preview = ", ".join(f"N{neuron}" for neuron in neurons[:8])
    return f"{len(neurons):,} selected neurons ({preview}, …)"


def iter_cci3_texts(*, seed: int, shuffle_buffer: int) -> Iterator[str]:
    """Stream CCI3 JSONL robustly despite its stale schema and bad shards."""
    repo_id = "BAAI/CCI3-Data"
    api = HfApi()
    filesystem = HfFileSystem()
    files = [
        name
        for name in api.list_repo_files(repo_id, repo_type="dataset")
        if name.startswith("data/part-") and name.endswith(".jsonl")
        and name not in CCI3_DENYLIST
    ]
    rng = random.Random(seed)
    rng.shuffle(files)
    buffer: list[str] = []

    for name in files:
        path = f"datasets/{repo_id}/{name}"
        with filesystem.open(path, "rb") as handle:
            prefix = handle.read(64)
            if not prefix or prefix.strip(b"\x00") == b"":
                continue
            handle.seek(0)
            for raw_line in handle:
                try:
                    row = json.loads(raw_line.decode("utf-8", errors="replace"))
                except (json.JSONDecodeError, UnicodeError):
                    continue
                text = str(row.get("content", ""))
                if not text.strip():
                    continue
                if len(buffer) < shuffle_buffer:
                    buffer.append(text)
                    continue
                position = rng.randrange(shuffle_buffer + 1)
                if position == shuffle_buffer:
                    yield text
                else:
                    yield buffer[position]
                    buffer[position] = text

    rng.shuffle(buffer)
    yield from buffer


def sample_sequences(
    tokenizer: AutoTokenizer,
    *,
    dataset_id: str,
    text_field: str,
    dataset_label: str,
    num_tokens: int,
    max_length: int,
    seed: int,
    shuffle_buffer: int,
) -> list[list[int]]:
    if dataset_id == "BAAI/CCI3-Data":
        texts = iter_cci3_texts(seed=seed, shuffle_buffer=shuffle_buffer)
    else:
        try:
            stream = load_dataset(dataset_id, split="train", streaming=True).shuffle(
                seed=seed,
                buffer_size=shuffle_buffer,
            )
        except DatasetNotFoundError as error:
            raise RuntimeError(f"Unable to access dataset {dataset_id}") from error
        texts = (str(row.get(text_field, "")) for row in stream)
    sequences: list[list[int]] = []
    remaining = num_tokens

    for text in texts:
        if not text.strip():
            continue
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        for start in range(0, len(token_ids), max_length):
            if remaining == 0:
                return sequences
            sequence = token_ids[start : start + min(max_length, remaining)]
            if sequence:
                sequences.append(sequence)
                remaining -= len(sequence)

    raise RuntimeError(
        f"{dataset_label} ended after {num_tokens - remaining:,} tokens; "
        f"requested {num_tokens:,}"
    )


def make_batch(
    sequences: list[list[int]], pad_token_id: int
) -> tuple[torch.Tensor, torch.Tensor]:
    width = max(map(len, sequences))
    input_ids = torch.full(
        (len(sequences), width), pad_token_id, dtype=torch.long
    )
    attention_mask = torch.zeros_like(input_ids)
    for row, sequence in enumerate(sequences):
        length = len(sequence)
        input_ids[row, :length] = torch.tensor(sequence, dtype=torch.long)
        attention_mask[row, :length] = 1
    return input_ids, attention_mask


def histogram_plot(
    edges: np.ndarray,
    counts: np.ndarray,
    *,
    title: str,
    xlabel: str,
    output: Path,
    log_x: bool = False,
) -> None:
    centers = (edges[:-1] + edges[1:]) / 2
    widths = np.diff(edges)
    fig, ax = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
    ax.bar(centers, counts, width=widths, color="#4C78A8", edgecolor="none")
    if log_x:
        ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Activation values (log count)")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.2)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def serialize_extreme_context(
    tokenizer: AutoTokenizer,
    record: dict[str, object],
) -> dict[str, object]:
    token_ids = [int(token_id) for token_id in record["token_ids"]]
    target_offset = int(record["target_offset"])
    context_activations = [
        float(value) for value in record.get("context_activations", [])
    ]
    if not context_activations:
        context_activations = [0.0] * len(token_ids)
        context_activations[target_offset] = float(record["activation"])
    if len(context_activations) != len(token_ids):
        raise ValueError("Context activation count does not match context token count")
    return {
        "activation": float(record["activation"]),
        "sequence_index": int(record["sequence_index"]),
        "token_position": int(record["token_position"]),
        "target_offset": target_offset,
        "target_token_id": token_ids[target_offset],
        "target_token": tokenizer.decode(
            [token_ids[target_offset]],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        ),
        "context_token_ids": token_ids,
        "context_activations": context_activations,
        "context_tokens": [
            tokenizer.decode(
                [token_id],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            for token_id in token_ids
        ],
        "context_text": tokenizer.decode(
            token_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        ),
    }


def decoded_token_segments(
    tokenizer: AutoTokenizer,
    token_ids: list[int],
) -> list[tuple[int, int, str]]:
    """Decode the smallest spans that form valid display text.

    Byte-level BPE tokenizers can split one Unicode character across multiple
    token IDs. Decoding those IDs independently produces U+FFFD replacement
    characters even though decoding the complete span is valid. Each returned
    tuple is ``(start, end, text)`` in token-offset coordinates.
    """
    segments: list[tuple[int, int, str]] = []
    start = 0
    while start < len(token_ids):
        chosen: tuple[int, str] | None = None
        for end in range(start + 1, len(token_ids) + 1):
            text = tokenizer.decode(
                token_ids[start:end],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            if "\ufffd" not in text:
                chosen = (end, text)
                break

        if chosen is None:
            end = start + 1
            # A clipped context window can begin or end in the middle of a
            # Unicode character. That missing byte cannot be reconstructed
            # from the cached IDs, so show a boundary ellipsis rather than a
            # misleading replacement character.
            text = "…"
        else:
            end, text = chosen
        segments.append((start, end, text))
        start = end
    return segments


def target_display(
    tokenizer: AutoTokenizer,
    record: dict[str, object],
) -> tuple[str, int, int]:
    token_ids = [int(token_id) for token_id in record["context_token_ids"]]
    target_offset = int(record["target_offset"])
    for start, end, text in decoded_token_segments(tokenizer, token_ids):
        if start <= target_offset < end:
            return text, target_offset - start + 1, end - start
    raise ValueError(f"Target offset {target_offset} is outside the saved context")


def activation_background(value: float, scale: float) -> str:
    strength = min(abs(value) / scale, 1.0) if scale > 0.0 else 0.0
    alpha = 0.04 + 0.76 * strength**0.5
    rgb = "245, 158, 11" if value >= 0.0 else "59, 130, 246"
    return f"rgba({rgb}, {alpha:.3f})"


def context_html(
    tokenizer: AutoTokenizer,
    record: dict[str, object],
    activation_scale: float,
) -> str:
    token_ids = [int(token_id) for token_id in record["context_token_ids"]]
    target_offset = int(record["target_offset"])
    activations = [float(value) for value in record.get("context_activations", [])]
    if not activations:
        activations = [0.0] * len(token_ids)
        activations[target_offset] = float(record["activation"])
    spans: list[str] = []

    for start, end, text in decoded_token_segments(tokenizer, token_ids):
        segment_values = activations[start:end]
        representative = max(segment_values, key=abs)
        is_target = start <= target_offset < end
        token_details = ", ".join(
            f"ID {token_ids[index]}: {activations[index]:+.6f}"
            for index in range(start, end)
        )
        style = (
            f"background:{activation_background(representative, activation_scale)};"
            "border-radius:3px;padding:1px 0;"
        )
        if is_target:
            style += "box-shadow:inset 0 0 0 2px #b77900;font-weight:700;"
        spans.append(
            f'<span style="{style}" title="{html.escape(token_details, quote=True)}">'
            f"{html.escape(text)}</span>"
        )
    return f"<pre>{''.join(spans)}</pre>"


def raw_distribution_metrics(
    summary: dict[str, float | int],
    count: int,
) -> dict[str, float | None]:
    """Return population moments from raw running sums."""
    mean = float(summary["sum"]) / count
    second_raw = float(summary["square_sum"]) / count
    third_raw = float(summary["cube_sum"]) / count
    fourth_raw = float(summary["fourth_sum"]) / count
    variance = max(second_raw - mean * mean, 0.0)
    central_third = third_raw - 3.0 * mean * second_raw + 2.0 * mean**3
    central_fourth = max(
        fourth_raw
        - 4.0 * mean * third_raw
        + 6.0 * mean * mean * second_raw
        - 3.0 * mean**4,
        0.0,
    )
    skewness = central_third / variance**1.5 if variance > 0.0 else None
    kurtosis = central_fourth / variance**2 if variance > 0.0 else None
    return {
        "mean": mean,
        "variance": variance,
        "skewness": skewness,
        "kurtosis": kurtosis,
        "excess_kurtosis": kurtosis - 3.0 if kurtosis is not None else None,
    }


def write_markdown_report(
    *,
    tokenizer: AutoTokenizer,
    output: Path,
    model_id: str,
    layer: int,
    corpus_label: str,
    sampled_tokens: int,
    distribution_metrics: dict[str, float | int | None],
    neuron_contexts: dict[str, dict[str, list[dict[str, object]]]],
) -> None:
    kurtosis = distribution_metrics["kurtosis"]
    skewness = distribution_metrics["skewness"]
    kurtosis_text = "undefined" if kurtosis is None else f"{kurtosis:.6g}"
    skewness_text = "undefined" if skewness is None else f"{skewness:.6g}"
    log_cosh_score = distribution_metrics["ica_log_cosh_score"]
    log_cosh_text = (
        "undefined" if log_cosh_score is None else f"{log_cosh_score:.6g}"
    )
    context_values = [
        abs(float(value))
        for tails in neuron_contexts.values()
        for tail in ("maximum", "minimum")
        for record in tails[tail]
        for value in record.get("context_activations", [record["activation"]])
    ]
    activation_scale = max(context_values, default=1.0)
    lines = [
        f"# MLP activation report: {model_id.rsplit('/', 1)[-1]} · Layer {layer}",
        "",
        f"Corpus: **{corpus_label}** · sampled tokens: **{sampled_tokens:,}**",
        "",
        "## Distributions",
        "",
        "![Raw activation distribution](raw_distribution.png)",
        "",
        "## Raw-distribution statistics",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Mean | {float(distribution_metrics['mean']):.6g} |",
        f"| Population variance | {float(distribution_metrics['variance']):.6g} |",
        f"| Population skewness | {skewness_text} |",
        f"| Population kurtosis | {kurtosis_text} |",
        f"| ICA log-cosh score | {log_cosh_text} |",
        "",
        (
            "ICA log-cosh score is the unscaled negentropy approximation "
            "`(E[log(cosh(z))] - E[log(cosh(v))])²`, where `z` is the "
            "standardized activation and `v ~ N(0,1)`."
        ),
        "",
        f"## Top {EXTREME_OCCURRENCES} occurrences by neuron and tail",
        "",
        (
            "Every context token is colored by this neuron's activation: orange "
            "is positive, blue is negative, and intensity uses one shared scale "
            f"(±{activation_scale:.6g}). The ranked extreme token has an outline; "
            "hover any token for its exact value."
        ),
        "",
    ]
    expand = " open" if len(neuron_contexts) <= 8 else ""
    for neuron, tails in neuron_contexts.items():
        lines.extend([f"<details{expand}>", f"<summary><strong>N{neuron}</strong></summary>", ""])
        for tail, heading in (("maximum", "Positive / maximum tail"), ("minimum", "Negative / minimum tail")):
            lines.extend([f"### {heading}", ""])
            for rank, record in enumerate(tails[tail], start=1):
                activation = float(record["activation"])
                display, piece, pieces = target_display(tokenizer, record)
                token = html.escape(display)
                target_id = int(record["target_token_id"])
                piece_note = (
                    f"token ID {target_id}, piece {piece}/{pieces}"
                    if pieces > 1
                    else f"token ID {target_id}"
                )
                sequence = int(record["sequence_index"])
                position = int(record["token_position"])
                lines.extend(
                    [
                        (
                            f"{rank}. **{activation:+.6g}** at <code>{token}</code>"
                            f" <small>({piece_note})</small> "
                            f"(sequence {sequence:,}, token position {position:,})"
                        ),
                        "",
                        context_html(tokenizer, record, activation_scale),
                        "",
                    ]
                )
        lines.extend(["</details>", ""])
    output.write_text("\n".join(lines), encoding="utf-8")


def measure_individual_neurons_corpus(
    *,
    args: argparse.Namespace,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    touched: int,
    corpus_key: str,
    corpus: dict[str, str],
    sequences: list[list[int]],
    layers: list[int],
    neurons: list[int],
    intermediate_size: int,
    model_id: str,
    model_revision: str | None,
) -> None:
    """Capture selected neurons together, then report each one independently."""
    neuron_index = torch.tensor(neurons, device=model.device)
    captured: dict[int, list[torch.Tensor]] = {layer: [] for layer in layers}
    current_mask: torch.Tensor | None = None
    last_layer = layers[-1]

    def make_hook(layer: int):
        def capture(
            _module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]
        ) -> None:
            if current_mask is None:
                raise RuntimeError("Activation hook fired without an attention mask")
            selected = inputs[0].index_select(-1, neuron_index)[current_mask]
            captured[layer].append(selected.detach().to(device="cpu", dtype=torch.bfloat16))
            if layer == last_layer:
                raise LastLayerCaptured

        return capture

    handles = [
        model.model.layers[layer].mlp.down_proj.register_forward_pre_hook(
            make_hook(layer)
        )
        for layer in layers
    ]
    progress = tqdm(
        total=args.num_tokens,
        unit="tok",
        desc=(
            f"{corpus_key} · capture · {len(neurons)} individual neurons · "
            f"layers {layers}"
        ),
    )
    try:
        for start in range(0, len(sequences), args.batch_size):
            batch_sequences = sequences[start : start + args.batch_size]
            input_ids, attention_mask = make_batch(
                batch_sequences, tokenizer.pad_token_id
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
    finally:
        progress.close()
        for handle in handles:
            handle.remove()

    sequence_ends: list[int] = []
    running = 0
    for sequence in sequences:
        running += len(sequence)
        sequence_ends.append(running)
    if running != args.num_tokens:
        raise RuntimeError(
            f"Captured token mapping has {running:,} tokens; expected {args.num_tokens:,}"
        )

    def occurrence(
        neuron_values: torch.Tensor,
        flat_position: int,
        activation: float,
    ) -> dict[str, object]:
        sequence_index = bisect.bisect_right(sequence_ends, flat_position)
        sequence_start = sequence_ends[sequence_index - 1] if sequence_index else 0
        token_position = flat_position - sequence_start
        sequence = sequences[sequence_index]
        left = max(0, token_position - args.context_tokens)
        right = min(len(sequence), token_position + args.context_tokens + 1)
        return {
            "activation": activation,
            "sequence_index": sequence_index,
            "token_position": token_position,
            "target_offset": token_position - left,
            "token_ids": sequence[left:right],
            "context_activations": neuron_values[
                sequence_start + left : sequence_start + right
            ].tolist(),
        }

    model_name = model_id.rsplit("/", 1)[-1]
    model_dir = args.output_root / model_name
    for layer in layers:
        layer_values = torch.cat(captured[layer], dim=0)
        if layer_values.shape != (args.num_tokens, len(neurons)):
            raise RuntimeError(
                f"Layer {layer} captured shape {tuple(layer_values.shape)}; expected "
                f"({args.num_tokens}, {len(neurons)})"
            )
        for selected_index, neuron in enumerate(neurons):
            values = layer_values[:, selected_index].float()
            values64 = values.double()
            squared = values64.square()
            summary: dict[str, float | int] = {
                "sum": float(values64.sum()),
                "square_sum": float(squared.sum()),
                "cube_sum": float((squared * values64).sum()),
                "fourth_sum": float(squared.square().sum()),
            }
            metrics = raw_distribution_metrics(summary, args.num_tokens)
            mean = float(metrics["mean"])
            variance = float(metrics["variance"])
            if variance > 0.0:
                standardized = (values - mean) / math.sqrt(variance)
                log_cosh_expectation = float(
                    (
                        torch.logaddexp(standardized, -standardized)
                        - math.log(2.0)
                    ).mean()
                )
                log_cosh_contrast = (
                    log_cosh_expectation - ICA_GAUSSIAN_LOG_COSH_EXPECTATION
                )
                log_cosh_score = log_cosh_contrast**2
            else:
                log_cosh_expectation = None
                log_cosh_contrast = None
                log_cosh_score = None

            raw_min = float(values.min())
            raw_max = float(values.max())
            if raw_min == raw_max:
                raw_min -= 0.5
                raw_max += 0.5
            raw_edges = np.linspace(raw_min, raw_max, args.bins + 1)
            raw_counts, _ = np.histogram(values.numpy(), bins=raw_edges)
            magnitudes = values.abs()
            positive_magnitudes = magnitudes[magnitudes > 0]
            if positive_magnitudes.numel():
                min_magnitude = float(positive_magnitudes.min())
                max_magnitude = float(positive_magnitudes.max())
                log_min = math.log10(min_magnitude)
                log_max = math.log10(max_magnitude)
                if log_min == log_max:
                    log_min -= 0.5
                    log_max += 0.5
                magnitude_edges = np.logspace(log_min, log_max, args.bins + 1)
                magnitude_counts, _ = np.histogram(
                    positive_magnitudes.numpy(), bins=magnitude_edges
                )
            else:
                magnitude_edges = np.logspace(-1.0, 1.0, args.bins + 1)
                magnitude_counts = np.zeros(args.bins, dtype=np.int64)

            output_dir = (
                model_dir
                / f"layer_{layer:02d}"
                / f"neuron_{neuron:04d}"
                / corpus_key
            )
            output_dir.mkdir(parents=True, exist_ok=True)
            histogram_plot(
                raw_edges,
                raw_counts,
                title=(
                    f"{model_name} · Layer {layer} · neuron N{neuron}\n"
                    f"Post-SwiGLU activations · {corpus['label']}"
                ),
                xlabel="Raw activation",
                output=output_dir / "raw_distribution.png",
            )
            torch.save(
                {
                    "raw_edges": torch.from_numpy(raw_edges),
                    "raw_counts": torch.from_numpy(raw_counts),
                    "magnitude_edges": torch.from_numpy(magnitude_edges),
                    "magnitude_counts": torch.from_numpy(magnitude_counts),
                },
                output_dir / "histograms.pt",
            )

            top_k = min(EXTREME_OCCURRENCES, args.num_tokens)
            maximum_values, maximum_positions = torch.topk(values, top_k)
            minimum_values, minimum_positions = torch.topk(
                values, top_k, largest=False
            )
            neuron_contexts = {
                str(neuron): {
                    "maximum": [
                        serialize_extreme_context(
                            tokenizer,
                            occurrence(values, int(position), float(activation)),
                        )
                        for activation, position in zip(
                            maximum_values, maximum_positions
                        )
                    ],
                    "minimum": [
                        serialize_extreme_context(
                            tokenizer,
                            occurrence(values, int(position), float(activation)),
                        )
                        for activation, position in zip(
                            minimum_values, minimum_positions
                        )
                    ],
                }
            }
            details = {
                "dataset_key": corpus_key,
                "dataset_id": corpus["dataset_id"],
                "dataset_label": corpus["label"],
                "text_field": corpus["text_field"],
                "layer": layer,
                "activation_definition": "input to mlp.down_proj (post-SwiGLU)",
                "sampled_tokens": args.num_tokens,
                "intermediate_size": intermediate_size,
                "neurons": [neuron],
                "selected_neuron_count": 1,
                "activation_values": args.num_tokens,
                "minimum": float(values.min()),
                "maximum": float(values.max()),
                "mean": mean,
                "variance": variance,
                "standard_deviation": variance**0.5,
                "skewness": metrics["skewness"],
                "kurtosis": metrics["kurtosis"],
                "excess_kurtosis": metrics["excess_kurtosis"],
                "ica_log_cosh_expectation": log_cosh_expectation,
                "ica_gaussian_log_cosh_expectation": (
                    ICA_GAUSSIAN_LOG_COSH_EXPECTATION
                ),
                "ica_log_cosh_contrast": log_cosh_contrast,
                "ica_log_cosh_score": log_cosh_score,
                "zero_count": int(torch.count_nonzero(values == 0)),
                "negative_count": int(torch.count_nonzero(values < 0)),
                "positive_count": int(torch.count_nonzero(values > 0)),
                "bins": args.bins,
            }
            (output_dir / "details.json").write_text(
                json.dumps(details, indent=2) + "\n", encoding="utf-8"
            )
            context_payload = {
                "model_id": model_id,
                "model_revision": model_revision,
                "dataset_key": corpus_key,
                "dataset_id": corpus["dataset_id"],
                "dataset_label": corpus["label"],
                "layer": layer,
                "activation_definition": "input to mlp.down_proj (post-SwiGLU)",
                "sampled_tokens": args.num_tokens,
                "context_tokens_each_side": args.context_tokens,
                "occurrences_per_tail": EXTREME_OCCURRENCES,
                "neurons": neuron_contexts,
            }
            (output_dir / "extreme_contexts.json").write_text(
                json.dumps(context_payload, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            write_markdown_report(
                tokenizer=tokenizer,
                output=output_dir / "report.md",
                model_id=model_id,
                layer=layer,
                corpus_label=corpus["label"],
                sampled_tokens=args.num_tokens,
                distribution_metrics=details,
                neuron_contexts=neuron_contexts,
            )

    run_metadata = {
        "model_id": model_id,
        "model_revision": model_revision,
        "dataset_key": corpus_key,
        "dataset_id": corpus["dataset_id"],
        "dataset_label": corpus["label"],
        "sampling": "streaming shuffle",
        "num_tokens": args.num_tokens,
        "layers": layers,
        "neurons": neurons,
        "neuron_output_mode": "individual",
        "execution_mode": "single shared capture pass",
        "retained_activation_dtype": "bfloat16",
        "cpu_pages_touched_gib": touched / 1024**3,
    }
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / f"run_metadata_{corpus_key}.json").write_text(
        json.dumps(run_metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Measured {args.num_tokens:,} {corpus_key} tokens once for "
        f"{len(neurons):,} individual neurons across layers {layers}"
    )


def measure_corpus(
    *,
    args: argparse.Namespace,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    touched: int,
    corpus_key: str,
    corpus: dict[str, str],
    sequences: list[list[int]],
    layers: list[int],
    neurons: list[int],
    intermediate_size: int,
    model_id: str,
    model_revision: str | None,
    requested_neurons: list[int],
    measurement_groups: list[list[int]],
) -> None:
    neuron_label = neuron_title(neurons, intermediate_size)
    neuron_index = (
        None
        if len(neurons) == intermediate_size
        else torch.tensor(neurons, device=model.device)
    )
    last_layer = layers[-1]
    stats = {
        layer: {
            "count": 0,
            "sum": 0.0,
            "square_sum": 0.0,
            "cube_sum": 0.0,
            "fourth_sum": 0.0,
            "standardized_log_cosh_sum": 0.0,
            "minimum": float("inf"),
            "maximum": float("-inf"),
            "minimum_positive_magnitude": float("inf"),
            "maximum_magnitude": 0.0,
            "negative_count": 0,
            "zero_count": 0,
            "positive_count": 0,
        }
        for layer in layers
    }
    raw_counts = {
        layer: torch.zeros(args.bins, dtype=torch.int64) for layer in layers
    }
    magnitude_counts = {
        layer: torch.zeros(args.bins, dtype=torch.int64) for layer in layers
    }
    raw_ranges: dict[int, tuple[float, float]] = {}
    log_magnitude_ranges: dict[int, tuple[float, float]] = {}
    first_pass_metrics: dict[int, dict[str, float | None]] = {}
    extreme_records = {
        layer: {
            "maximum": torch.full(
                (EXTREME_OCCURRENCES, len(neurons)), float("-inf")
            ),
            "minimum": torch.full(
                (EXTREME_OCCURRENCES, len(neurons)), float("inf")
            ),
            "maximum_context": [
                [None] * len(neurons) for _ in range(EXTREME_OCCURRENCES)
            ],
            "minimum_context": [
                [None] * len(neurons) for _ in range(EXTREME_OCCURRENCES)
            ],
        }
        for layer in layers
    }
    phase = "stats"
    current_mask: torch.Tensor | None = None
    current_input_ids: torch.Tensor | None = None
    current_attention_mask: torch.Tensor | None = None
    current_sequence_start = 0

    def make_hook(layer: int):
        def measure(
            _module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]
        ) -> None:
            if current_mask is None:
                raise RuntimeError("Activation hook fired without an attention mask")
            selected = inputs[0][current_mask]
            if neuron_index is not None:
                selected = selected.index_select(-1, neuron_index)
            selected = selected.float()
            values = selected.reshape(-1)
            magnitudes = values.abs()
            positive_magnitudes = magnitudes[magnitudes > 0]

            if phase == "stats":
                summary = stats[layer]
                summary["count"] += values.numel()
                summary["sum"] += values.sum().item()
                squared = values.square()
                summary["square_sum"] += squared.sum().item()
                summary["cube_sum"] += (squared * values).sum().item()
                summary["fourth_sum"] += squared.square().sum().item()
                summary["minimum"] = min(summary["minimum"], values.min().item())
                summary["maximum"] = max(summary["maximum"], values.max().item())
                summary["maximum_magnitude"] = max(
                    summary["maximum_magnitude"], magnitudes.max().item()
                )
                if positive_magnitudes.numel():
                    summary["minimum_positive_magnitude"] = min(
                        summary["minimum_positive_magnitude"],
                        positive_magnitudes.min().item(),
                    )
                summary["negative_count"] += torch.count_nonzero(values < 0).item()
                summary["zero_count"] += torch.count_nonzero(values == 0).item()
                summary["positive_count"] += torch.count_nonzero(values > 0).item()

                if current_input_ids is None or current_attention_mask is None:
                    raise RuntimeError("Activation hook fired without source tokens")
                local_k = min(EXTREME_OCCURRENCES, selected.shape[0])
                local_maximum, local_maximum_positions = selected.topk(
                    local_k, dim=0, largest=True, sorted=True
                )
                local_minimum, local_minimum_positions = selected.topk(
                    local_k, dim=0, largest=False, sorted=True
                )
                local_maximum = local_maximum.cpu()
                local_minimum = local_minimum.cpu()
                local_maximum_positions = local_maximum_positions.cpu()
                local_minimum_positions = local_minimum_positions.cpu()
                valid_positions = current_mask.nonzero(as_tuple=False).cpu()
                records = extreme_records[layer]

                for tail, local_values, local_positions, largest in (
                    (
                        "maximum",
                        local_maximum,
                        local_maximum_positions,
                        True,
                    ),
                    (
                        "minimum",
                        local_minimum,
                        local_minimum_positions,
                        False,
                    ),
                ):
                    old_values = records[tail]
                    candidates = torch.cat((old_values, local_values), dim=0)
                    new_values, sources = candidates.topk(
                        EXTREME_OCCURRENCES,
                        dim=0,
                        largest=largest,
                        sorted=True,
                    )
                    unchanged_sources = torch.arange(EXTREME_OCCURRENCES).unsqueeze(1)
                    changed = (
                        sources != unchanged_sources
                    ).any(dim=0).nonzero(as_tuple=False).flatten().tolist()
                    contexts = records[f"{tail}_context"]
                    for selected_index in changed:
                        old_contexts = [
                            contexts[rank][selected_index]
                            for rank in range(EXTREME_OCCURRENCES)
                        ]
                        new_contexts = []
                        for rank in range(EXTREME_OCCURRENCES):
                            source = int(sources[rank, selected_index])
                            if source < EXTREME_OCCURRENCES:
                                new_contexts.append(old_contexts[source])
                                continue
                            local_rank = source - EXTREME_OCCURRENCES
                            batch_row, token_position = valid_positions[
                                int(local_positions[local_rank, selected_index])
                            ].tolist()
                            sequence_length = int(
                                current_attention_mask[batch_row].sum()
                            )
                            left = max(0, token_position - args.context_tokens)
                            right = min(
                                sequence_length,
                                token_position + args.context_tokens + 1,
                            )
                            new_contexts.append(
                                {
                                    "activation": float(
                                        local_values[local_rank, selected_index]
                                    ),
                                    "sequence_index": current_sequence_start
                                    + batch_row,
                                    "token_position": token_position,
                                    "target_offset": token_position - left,
                                    "token_ids": current_input_ids[
                                        batch_row, left:right
                                    ].tolist(),
                                    "context_activations": inputs[0][
                                        batch_row,
                                        left:right,
                                        neurons[selected_index],
                                    ]
                                    .detach()
                                    .float()
                                    .cpu()
                                    .tolist(),
                                }
                            )
                        for rank, context in enumerate(new_contexts):
                            contexts[rank][selected_index] = context
                    records[tail] = new_values
            else:
                metrics = first_pass_metrics[layer]
                variance = float(metrics["variance"])
                if variance > 0.0:
                    standardized = (values - float(metrics["mean"])) / math.sqrt(
                        variance
                    )
                    stats[layer]["standardized_log_cosh_sum"] += (
                        torch.logaddexp(standardized, -standardized)
                        - math.log(2.0)
                    ).sum().item()
                raw_min, raw_max = raw_ranges[layer]
                log_min, log_max = log_magnitude_ranges[layer]
                raw_counts[layer] += torch.histc(
                    values, bins=args.bins, min=raw_min, max=raw_max
                ).to(device="cpu", dtype=torch.int64)
                if positive_magnitudes.numel():
                    log_magnitudes = positive_magnitudes.log10().clamp_(
                        min=log_min, max=log_max
                    )
                    magnitude_counts[layer] += torch.histc(
                        log_magnitudes,
                        bins=args.bins,
                        min=log_min,
                        max=log_max,
                    ).to(device="cpu", dtype=torch.int64)

            if layer == last_layer:
                raise LastLayerCaptured

        return measure

    handles = [
        model.model.layers[layer].mlp.down_proj.register_forward_pre_hook(
            make_hook(layer)
        )
        for layer in layers
    ]

    try:
        for pass_index in range(2):
            phase = "stats" if pass_index == 0 else "histograms"
            if phase == "histograms":
                for layer in layers:
                    summary = stats[layer]
                    first_pass_metrics[layer] = raw_distribution_metrics(
                        summary, int(summary["count"])
                    )
                    raw_min = float(summary["minimum"])
                    raw_max = float(summary["maximum"])
                    if raw_min == raw_max:
                        raw_min -= 0.5
                        raw_max += 0.5
                    raw_ranges[layer] = (raw_min, raw_max)
                    min_mag = float(summary["minimum_positive_magnitude"])
                    max_mag = float(summary["maximum_magnitude"])
                    if not np.isfinite(min_mag) or max_mag <= 0:
                        raise RuntimeError(f"Layer {layer} has no nonzero activations")
                    log_magnitude_ranges[layer] = (
                        float(np.log10(min_mag)),
                        float(np.log10(max_mag)),
                    )

            progress = tqdm(
                total=args.num_tokens,
                unit="tok",
                desc=(
                    f"{corpus_key} · {phase} · {neuron_label} · layers {layers}"
                ),
            )
            for start in range(0, len(sequences), args.batch_size):
                batch_sequences = sequences[start : start + args.batch_size]
                input_ids, attention_mask = make_batch(
                    batch_sequences, tokenizer.pad_token_id
                )
                current_sequence_start = start
                current_input_ids = input_ids
                current_attention_mask = attention_mask
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
                progress.update(int(attention_mask.sum().item()))
            progress.close()
    finally:
        for handle in handles:
            handle.remove()

    model_name = model_id.rsplit("/", 1)[-1]
    model_dir = args.output_root / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    expected_count = args.num_tokens * len(neurons)

    for layer in layers:
        summary = stats[layer]
        if summary["count"] != expected_count:
            raise RuntimeError(
                f"Layer {layer} measured {summary['count']:,} values; "
                f"expected {expected_count:,}"
            )
        if int(raw_counts[layer].sum()) != expected_count:
            raise RuntimeError(f"Layer {layer} raw histogram count mismatch")
        nonzero_count = expected_count - int(summary["zero_count"])
        if int(magnitude_counts[layer].sum()) != nonzero_count:
            raise RuntimeError(f"Layer {layer} magnitude histogram count mismatch")

        raw_min, raw_max = raw_ranges[layer]
        log_min, log_max = log_magnitude_ranges[layer]
        raw_edges = np.linspace(raw_min, raw_max, args.bins + 1)
        magnitude_edges = np.logspace(log_min, log_max, args.bins + 1)
        layer_dir = model_dir / f"layer_{layer:02d}"
        output_dir = layer_dir
        if len(neurons) == 1:
            output_dir = layer_dir / f"neuron_{neurons[0]:04d}"
        output_dir = output_dir / corpus_key
        output_dir.mkdir(parents=True, exist_ok=True)

        records = extreme_records[layer]

        histogram_plot(
            raw_edges,
            raw_counts[layer].numpy(),
            title=(
                f"{model_name} · Layer {layer} · {neuron_label}\n"
                f"Post-SwiGLU activations · {corpus['label']}"
            ),
            xlabel="Raw activation",
            output=output_dir / "raw_distribution.png",
        )
        stale_magnitude_plot = output_dir / "magnitude_distribution.png"
        if stale_magnitude_plot.exists():
            stale_magnitude_plot.unlink()
        torch.save(
            {
                "raw_edges": torch.from_numpy(raw_edges),
                "raw_counts": raw_counts[layer],
                "magnitude_edges": torch.from_numpy(magnitude_edges),
                "magnitude_counts": magnitude_counts[layer],
            },
            output_dir / "histograms.pt",
        )

        metrics = raw_distribution_metrics(summary, expected_count)
        mean = float(metrics["mean"])
        variance = float(metrics["variance"])
        if variance > 0.0:
            log_cosh_expectation = (
                float(summary["standardized_log_cosh_sum"]) / expected_count
            )
            log_cosh_contrast = (
                log_cosh_expectation - ICA_GAUSSIAN_LOG_COSH_EXPECTATION
            )
            log_cosh_score = log_cosh_contrast**2
        else:
            log_cosh_expectation = None
            log_cosh_contrast = None
            log_cosh_score = None
        details = {
            "dataset_key": corpus_key,
            "dataset_id": corpus["dataset_id"],
            "dataset_label": corpus["label"],
            "text_field": corpus["text_field"],
            "layer": layer,
            "activation_definition": "input to mlp.down_proj (post-SwiGLU)",
            "sampled_tokens": args.num_tokens,
            "intermediate_size": intermediate_size,
            "neurons": neurons,
            "selected_neuron_count": len(neurons),
            "activation_values": expected_count,
            "minimum": summary["minimum"],
            "maximum": summary["maximum"],
            "mean": mean,
            "variance": variance,
            "standard_deviation": variance**0.5,
            "skewness": metrics["skewness"],
            "kurtosis": metrics["kurtosis"],
            "excess_kurtosis": metrics["excess_kurtosis"],
            "ica_log_cosh_expectation": log_cosh_expectation,
            "ica_gaussian_log_cosh_expectation": (
                ICA_GAUSSIAN_LOG_COSH_EXPECTATION
            ),
            "ica_log_cosh_contrast": log_cosh_contrast,
            "ica_log_cosh_score": log_cosh_score,
            "zero_count": summary["zero_count"],
            "negative_count": summary["negative_count"],
            "positive_count": summary["positive_count"],
            "bins": args.bins,
        }
        (output_dir / "details.json").write_text(
            json.dumps(details, indent=2) + "\n", encoding="utf-8"
        )
        neuron_contexts = {
            str(neuron): {
                tail: [
                    serialize_extreme_context(tokenizer, context)
                    for context in (
                        records[f"{tail}_context"][rank][selected_index]
                        for rank in range(EXTREME_OCCURRENCES)
                    )
                    if context is not None
                ]
                for tail in ("maximum", "minimum")
            }
            for selected_index, neuron in enumerate(neurons)
        }
        context_payload = {
            "model_id": model_id,
            "model_revision": model_revision,
            "dataset_key": corpus_key,
            "dataset_id": corpus["dataset_id"],
            "dataset_label": corpus["label"],
            "layer": layer,
            "activation_definition": "input to mlp.down_proj (post-SwiGLU)",
            "sampled_tokens": args.num_tokens,
            "context_tokens_each_side": args.context_tokens,
            "occurrences_per_tail": EXTREME_OCCURRENCES,
            "neurons": neuron_contexts,
        }
        (output_dir / "extreme_contexts.json").write_text(
            json.dumps(context_payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        write_markdown_report(
            tokenizer=tokenizer,
            output=output_dir / "report.md",
            model_id=model_id,
            layer=layer,
            corpus_label=corpus["label"],
            sampled_tokens=args.num_tokens,
            distribution_metrics=details,
            neuron_contexts=neuron_contexts,
        )

    run_metadata = {
        "model_id": model_id,
        "model_revision": model_revision,
        "dataset_key": corpus_key,
        "dataset_id": corpus["dataset_id"],
        "dataset_label": corpus["label"],
        "text_field": corpus["text_field"],
        "sampling": "streaming shuffle",
        "seed": args.seed,
        "shuffle_buffer": args.shuffle_buffer,
        "num_tokens": args.num_tokens,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "context_tokens_each_side": args.context_tokens,
        "layers": layers,
        "neurons": requested_neurons,
        "neuron_output_mode": (
            "pooled"
            if len(measurement_groups) == 1 and len(measurement_groups[0]) > 1
            else "individual"
        ),
        "measurement_groups": measurement_groups,
        "cpu_pages_touched_gib": touched / 1024**3,
    }
    (model_dir / f"run_metadata_{corpus_key}.json").write_text(
        json.dumps(run_metadata, indent=2) + "\n", encoding="utf-8"
    )

    print(
        f"Measured {args.num_tokens:,} {corpus_key} tokens × "
        f"{len(neurons):,} neurons for layers {layers}"
    )


def main() -> None:
    args = parse_args()
    model_id, model_revision = resolve_model(args.model, args.revision)
    if args.num_tokens <= 0:
        raise ValueError("--num-tokens must be positive")
    if args.bins <= 0:
        raise ValueError("--bins must be positive")
    if args.context_tokens < 0:
        raise ValueError("--context-tokens must be nonnegative")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=model_revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    corpus_keys = ("en", "cn") if args.dataset == "both" else (args.dataset,)
    sampled_sequences: dict[str, list[list[int]]] = {}
    for corpus_key in corpus_keys:
        corpus = DATASETS[corpus_key]
        print(
            f"Sampling {args.num_tokens:,} tokens from {corpus['label']} "
            f"({corpus['dataset_id']})"
        )
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
        if sum(map(len, sequences)) != args.num_tokens:
            raise RuntimeError(
                f"{corpus['label']} sampled token count does not match --num-tokens"
            )
        sampled_sequences[corpus_key] = sequences

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
    neurons = parse_neurons(args.neurons, intermediate_size)
    measurement_groups = neuron_measurement_groups(args.neurons, neurons)

    for corpus_key in corpus_keys:
        if args.neurons.strip().lower() == "all":
            measure_corpus(
                args=args,
                tokenizer=tokenizer,
                model=model,
                touched=touched,
                corpus_key=corpus_key,
                corpus=DATASETS[corpus_key],
                sequences=sampled_sequences[corpus_key],
                layers=layers,
                neurons=neurons,
                intermediate_size=intermediate_size,
                model_id=model_id,
                model_revision=model_revision,
                requested_neurons=neurons,
                measurement_groups=measurement_groups,
            )
        else:
            measure_individual_neurons_corpus(
                args=args,
                tokenizer=tokenizer,
                model=model,
                touched=touched,
                corpus_key=corpus_key,
                corpus=DATASETS[corpus_key],
                sequences=sampled_sequences[corpus_key],
                layers=layers,
                neurons=neurons,
                intermediate_size=intermediate_size,
                model_id=model_id,
                model_revision=model_revision,
            )

    model_dir = args.output_root / model_id.rsplit("/", 1)[-1]
    total_tokens = args.num_tokens * len(corpus_keys)
    print(
        f"Completed {', '.join(corpus_keys)}: {total_tokens:,} total sampled tokens"
    )
    print(f"Saved results: {model_dir.resolve()}")


if __name__ == "__main__":
    main()
