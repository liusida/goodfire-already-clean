from __future__ import annotations

import argparse
import heapq
import html
import json
from pathlib import Path
from typing import Any

import torch
from gb10_load_llm import load_model_to_cuda
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from measure_attention_output_metrics import (
    AttentionOutputCaptured,
    attention_output_projection,
)
from measure_mlp_distributions import DATASETS, make_batch, resolve_model, sample_sequences


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Profile both activation tails of the lowest- and highest-kurtosis "
            "attention-output directions measured previously."
        )
    )
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B-Base")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--dataset", choices=tuple(DATASETS), default="en")
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--num-tokens", type=int, default=1_000_000)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle-buffer", type=int, default=10_000)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--direction-count",
        type=int,
        default=10,
        help="Number of lowest- and highest-kurtosis directions to profile.",
    )
    parser.add_argument("--context-tokens", type=int, default=24)
    parser.add_argument(
        "--output-root", type=Path, default=Path("outputs/mlp_measurement")
    )
    return parser.parse_args()


def tail_color(value: float, scale: float) -> str:
    strength = min(abs(value) / max(scale, 1e-12), 1.0)
    if value >= 0:
        base = (244, 177, 61)
    else:
        base = (84, 145, 207)
    rgb = tuple(round(255 + (channel - 255) * strength * 0.72) for channel in base)
    return f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"


def render_html(
    path: Path,
    *,
    tokenizer: AutoTokenizer,
    model_id: str,
    layer: int,
    projection_name: str,
    token_count: int,
    profiles: dict[int, dict[str, Any]],
) -> None:
    all_values = [
        abs(float(value))
        for profile in profiles.values()
        for tail in ("positive", "negative")
        for record in profile[tail]
        for value in record["activations"]
    ]
    scale = sorted(all_values)[max(0, int(len(all_values) * 0.98) - 1)] if all_values else 1.0

    sections: list[str] = []
    for direction, profile in profiles.items():
        primary = "positive" if float(profile["skewness"]) >= 0 else "negative"
        tail_sections: list[str] = []
        for tail in (primary, "negative" if primary == "positive" else "positive"):
            cards: list[str] = []
            for rank, record in enumerate(profile[tail], start=1):
                token_spans: list[str] = []
                for offset, (token_id, activation) in enumerate(
                    zip(record["token_ids"], record["activations"])
                ):
                    token = tokenizer.decode(
                        [int(token_id)],
                        skip_special_tokens=False,
                        clean_up_tokenization_spaces=False,
                    )
                    classes = "token target" if offset == record["target_offset"] else "token"
                    token_spans.append(
                        f'<span class="{classes}" style="background:{tail_color(float(activation), scale)}" '
                        f'title="activation {float(activation):+.6g}">{html.escape(token) or "∅"}</span>'
                    )
                cards.append(
                    f"""
<article class="sample">
  <header><b>#{rank} &nbsp; {float(record['value']):+.6g}</b><span>sample {record['sample_index']:,} · token {record['token_position']}</span></header>
  <div class="context">{''.join(token_spans)}</div>
</article>"""
                )
            primary_badge = '<span class="primary-badge">skewness-selected primary tail</span>' if tail == primary else ""
            tail_sections.append(
                f'<section class="tail {"primary" if tail == primary else "secondary"}">'
                f'<h3>{tail.title()} tail {primary_badge}</h3>{"".join(cards)}</section>'
            )
        sections.append(
            f"""
<section class="direction">
  <h2>L{layer}:O{direction} <small>{profile['selection']} #{profile['selection_rank']} activation kurtosis · κ={float(profile['kurtosis']):.6g} · skew={float(profile['skewness']):+.6g}</small></h2>
  {''.join(tail_sections)}
</section>"""
        )

    document = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Attention-output extreme profiles</title>
<style>
body{{font-family:Inter,system-ui,sans-serif;margin:28px;color:#17202a;background:#fff}}
h1{{margin-bottom:4px}} .subtitle{{color:#667587;margin-bottom:28px}}
.direction{{margin:30px 0 42px}} h2{{border-bottom:1px solid #dce3ea;padding-bottom:9px}}
h2 small{{font-size:14px;font-weight:400;color:#667587;margin-left:10px}}
.tail{{margin:20px 0;padding-left:14px;border-left:3px solid #d8e0e8}}
.tail.primary{{border-left-color:#d18b00}} .primary-badge{{font-size:12px;color:#955f00;background:#fff1c9;padding:3px 7px;border-radius:10px;margin-left:8px}}
.sample{{border:1px solid #d9e1e8;border-radius:10px;margin:10px 0;overflow:hidden}}
.sample header{{display:flex;justify-content:space-between;background:#f4f7f9;padding:8px 12px;color:#405064}}
.context{{padding:12px;line-height:1.85;font-family:ui-monospace,SFMono-Regular,monospace}}
.token{{padding:2px 1px;border-radius:3px;white-space:pre-wrap}}
.target{{outline:2px solid #c88300;font-weight:700;position:relative;z-index:1}}
</style></head><body>
<h1>{html.escape(model_id)} · Layer {layer} attention-output extremes</h1>
<div class="subtitle">Input to <code>{html.escape(projection_name)}</code> · {token_count:,} OpenWebText tokens · both tails shown · token color is activation</div>
{''.join(sections)}
</body></html>"""
    path.write_text(document, encoding="utf-8")


def main() -> None:
    args = parse_args()
    model_id, model_revision = resolve_model(args.model, args.revision)
    model_name = model_id.rsplit("/", 1)[-1]
    metrics_dir = (
        args.output_root
        / model_name
        / f"layer_{args.layer:02d}"
        / "attention_output_metrics"
        / args.dataset
    )
    metrics_path = metrics_dir / "metrics.pt"
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"Missing {metrics_path}; run measure_attention_output_metrics.py first"
        )
    metrics = torch.load(metrics_path, map_location="cpu", weights_only=True)
    finite = torch.isfinite(metrics["kurtosis"])
    valid_ids = torch.nonzero(finite, as_tuple=False).flatten()
    valid_kurtosis = metrics["kurtosis"][valid_ids]
    direction_count = min(args.direction_count, int(valid_ids.numel()))
    _, low_offsets = torch.topk(valid_kurtosis, direction_count, largest=False)
    _, high_offsets = torch.topk(valid_kurtosis, direction_count, largest=True)
    low_directions = valid_ids[low_offsets].tolist()
    high_directions = valid_ids[high_offsets].tolist()
    selected = [int(direction) for direction in low_directions + high_directions]
    selection_info = {
        int(direction): ("lowest", rank)
        for rank, direction in enumerate(low_directions, start=1)
    }
    selection_info.update(
        {
            int(direction): ("highest", rank)
            for rank, direction in enumerate(high_directions, start=1)
        }
    )

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
    model, _ = load_model_to_cuda(
        AutoModelForCausalLM,
        model_id,
        revision=model_revision,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        touch="auto",
        return_touched=True,
    )
    model.eval()
    projection, projection_name = attention_output_projection(model.model.layers[args.layer])

    heaps: dict[int, dict[str, list[tuple[float, int, dict[str, Any]]]]] = {
        direction: {"positive": [], "negative": []} for direction in selected
    }
    current_mask: torch.Tensor | None = None
    current_sequences: list[list[int]] = []
    sample_base = 0
    tie_counter = 0

    def maybe_add(
        heap: list[tuple[float, int, dict[str, Any]]],
        key: float,
        record_factory: Any,
    ) -> None:
        nonlocal tie_counter
        if len(heap) >= args.top_k and key <= heap[0][0]:
            return
        record = record_factory()
        item = (key, tie_counter, record)
        tie_counter += 1
        if len(heap) < args.top_k:
            heapq.heappush(heap, item)
        else:
            heapq.heapreplace(heap, item)

    def capture(_module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        if current_mask is None:
            raise RuntimeError("Hook fired without mask")
        values = inputs[0][..., selected].float().cpu()
        lengths = current_mask.sum(dim=1).cpu().tolist()
        for batch_index, length_value in enumerate(lengths):
            length = int(length_value)
            token_ids = current_sequences[batch_index]
            for selected_offset, direction in enumerate(selected):
                channel = values[batch_index, :length, selected_offset]
                positive_value, positive_position = torch.max(channel, dim=0)
                negative_value, negative_position = torch.min(channel, dim=0)
                for tail, value_tensor, position_tensor, key in (
                    ("positive", positive_value, positive_position, float(positive_value)),
                    ("negative", negative_value, negative_position, -float(negative_value)),
                ):
                    position = int(position_tensor)

                    def record_factory(
                        position: int = position,
                        value: float = float(value_tensor),
                        channel: torch.Tensor = channel,
                        token_ids: list[int] = token_ids,
                        batch_index: int = batch_index,
                    ) -> dict[str, Any]:
                        left = max(0, position - args.context_tokens)
                        right = min(length, position + args.context_tokens + 1)
                        return {
                            "sample_index": sample_base + batch_index,
                            "token_position": position,
                            "value": value,
                            "token_ids": token_ids[left:right],
                            "activations": channel[left:right].tolist(),
                            "target_offset": position - left,
                        }

                    maybe_add(heaps[direction][tail], key, record_factory)
        raise AttentionOutputCaptured

    handle = projection.register_forward_pre_hook(capture)
    progress = tqdm(total=args.num_tokens, unit="tok", desc=f"profile L{args.layer} O{selected}")
    observed = 0
    try:
        for start in range(0, len(sequences), args.batch_size):
            current_sequences = sequences[start : start + args.batch_size]
            sample_base = start
            input_ids, attention_mask = make_batch(current_sequences, tokenizer.pad_token_id)
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
            observed += count
            progress.update(count)
    finally:
        progress.close()
        handle.remove()

    profiles: dict[int, dict[str, Any]] = {}
    for direction in selected:
        selection, selection_rank = selection_info[direction]
        profiles[direction] = {
            "selection": selection,
            "selection_rank": selection_rank,
            "kurtosis": float(metrics["kurtosis"][direction]),
            "skewness": float(metrics["skewness"][direction]),
        }
        for tail in ("positive", "negative"):
            profiles[direction][tail] = [
                item[2] for item in sorted(heaps[direction][tail], reverse=True)
            ]

    output_dir = metrics_dir / "extreme_profiles"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "profiles.json").write_text(
        json.dumps(
            {
                "model": model_id,
                "layer": args.layer,
                "projection": projection_name,
                "token_count": observed,
                "directions": profiles,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    render_html(
        output_dir / "report.html",
        tokenizer=tokenizer,
        model_id=model_id,
        layer=args.layer,
        projection_name=projection_name,
        token_count=observed,
        profiles=profiles,
    )
    print("Lowest directions:")
    for direction in low_directions:
        print(
            f"  O{direction}: κ={profiles[direction]['kurtosis']:.6g}, "
            f"skew={profiles[direction]['skewness']:+.6g}"
        )
    print("Highest directions:")
    for direction in high_directions:
        print(
            f"  O{direction}: κ={profiles[direction]['kurtosis']:.6g}, "
            f"skew={profiles[direction]['skewness']:+.6g}"
        )
    print(f"Saved profiles to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
