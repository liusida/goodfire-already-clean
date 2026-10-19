from __future__ import annotations

import argparse
import csv
import gc
import heapq
import html
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from gb10_load_llm import load_model_to_cuda
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "Qwen/Qwen3.5-0.8B-Base"
MODEL_REVISION = "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"
DEFAULT_FEATURES = Path("outputs/weekdays/qwen35_08b_weekday_layerwise_top_0.002pct_activations.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile selected Qwen MLP neurons on a shuffled OpenWebText stream."
    )
    parser.add_argument("--num-samples", type=int, default=100_000)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--context-tokens", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle-buffer", type=int, default=100_000)
    parser.add_argument("--checkpoint-every", type=int, default=1_000)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/openwebtext/openwebtext_100k_neuron_profiles.json"),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from --output using the same sampling/tokenization settings.",
    )
    return parser.parse_args()


def load_features(path: Path) -> list[tuple[int, int]]:
    with path.open(newline="") as handle:
        rows = csv.DictReader(handle)
        features = {
            (int(row["layer"]), int(row["neuron"]))
            for row in rows
            if row["is_weekday_token"] == "True"
        }
    if not features:
        raise RuntimeError(f"No weekday-token features found in {path}")
    return sorted(features)


def render_context(
    tokenizer: Any,
    token_ids: list[int],
    activation_values: list[float],
    position: int,
    radius: int,
) -> dict[str, Any]:
    start = max(0, position - radius)
    end = min(len(token_ids), position + radius + 1)
    left_ids = token_ids[start:position]
    active_id = token_ids[position]
    right_ids = token_ids[position + 1 : end]
    left = tokenizer.decode(left_ids, skip_special_tokens=False)
    active = tokenizer.decode([active_id], skip_special_tokens=False)
    right = tokenizer.decode(right_ids, skip_special_tokens=False)
    context_tokens = [
        {
            "text": tokenizer.decode([token_id], skip_special_tokens=False),
            "token_id": token_id,
            "token_position": token_position,
            "activation": activation_values[token_position],
            "is_active": token_position == position,
        }
        for token_position, token_id in enumerate(token_ids[start:end], start=start)
    ]
    return {
        "context_start": start,
        "context_end": end,
        "left_context": left,
        "active_token": active,
        "right_context": right,
        "context": f"{left}⟦{active}⟧{right}",
        "token_id": active_id,
        "context_tokens": context_tokens,
    }


def write_html(payload: dict[str, Any], path: Path) -> None:
    options = "".join(
        f'<option value="{html.escape(key)}">{html.escape(key)}</option>'
        for key in payload["profiles"]
    )
    data = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    document = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>OpenWebText neuron profiles</title>
<style>
*{{box-sizing:border-box}}body{{margin:0;background:#f3f5f7;color:#17212b;font-family:Inter,system-ui,sans-serif}}
header{{background:#17212b;color:#fff;padding:24px 30px}}h1{{margin:0 0 7px;font-size:23px}}header p{{margin:4px 0;color:#cbd4dd}}
.toolbar{{position:sticky;top:0;z-index:3;padding:12px 24px;background:#fff;border-bottom:1px solid #d8dee5}}select{{margin-left:8px;padding:7px 25px 7px 9px;border:1px solid #bdc7d1;border-radius:6px}}
main{{max-width:1200px;margin:auto;padding:18px}}.summary{{margin:0 0 12px;color:#5e6b78}}table{{width:100%;border-collapse:separate;border-spacing:0;background:#fff;border:1px solid #d8dee5;border-radius:8px;overflow:hidden}}
th,td{{padding:9px 11px;border-bottom:1px solid #e2e6ea;vertical-align:top}}th{{background:#edf2f6;text-align:left;position:sticky;top:59px;z-index:2}}td.rank{{width:55px;color:#73808d;text-align:right}}td.value{{width:110px;font-family:ui-monospace,monospace;font-weight:700;color:#8b2c0b}}td.meta{{width:150px;font-size:11px;color:#65727f}}
.context{{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;line-height:1.8;white-space:pre-wrap;overflow-wrap:anywhere}}.token-heat{{border-radius:2px;padding:2px 0}}.token-heat.active{{font-weight:800;outline:2px solid #c47b00;outline-offset:1px}}
@media(max-width:700px){{header{{padding:20px 16px}}main{{padding:10px 5px}}td.meta{{display:none}}th:nth-child(4){{display:none}}}}
</style></head><body><header><h1>OpenWebText neuron profiles</h1>
<p id="run-summary"></p></header><div class="toolbar"><label>Neuron<select id="feature">{options}</select></label></div>
<main><p class="summary" id="feature-summary"></p><table><thead><tr><th>Rank</th><th>Activation</th><th>Context</th><th>Source</th></tr></thead><tbody id="rows"></tbody></table></main>
<script>const DATA={data};const select=document.getElementById('feature'),body=document.getElementById('rows');
const esc=s=>String(s).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;');
document.getElementById('run-summary').textContent=`${{DATA.metadata.samples_processed.toLocaleString()}} / ${{DATA.metadata.num_samples.toLocaleString()}} sampled documents processed · max ${{DATA.metadata.max_length}} tokens/document · top ${{DATA.metadata.top_k}} occurrences retained`;
function heat(value,max){{const t=Math.max(0,Math.min(1,value/max)),g=Math.round(255-42*t),b=Math.round(255-176*t);return `rgb(255,${{g}},${{b}})`}}
function render(){{const key=select.value,items=DATA.profiles[key],max=Math.max(...items.map(x=>x.activation),1e-9);document.getElementById('feature-summary').textContent=`${{key}} · ${{items.length}} retained occurrences · token color scale: white at ≤0 to yellow at ${{max.toFixed(4)}}`;
body.innerHTML=items.map((x,i)=>{{const context=x.context_tokens.map(t=>`<span class="token-heat${{t.is_active?' active':''}}" style="background:${{heat(t.activation,max)}}" title="activation ${{t.activation.toFixed(6)}} · token ${{t.token_position}} · id ${{t.token_id}}">${{esc(t.text)}}</span>`).join('');return `<tr><td class="rank">${{i+1}}</td><td class="value">${{x.activation.toFixed(6)}}</td><td class="context">${{context}}</td><td class="meta">sample ${{x.sample_index.toLocaleString()}}<br>token ${{x.token_position}} · id ${{x.token_id}}</td></tr>`}}).join('')}}
select.addEventListener('change',render);render();</script></body></html>"""
    path.write_text(document)


def save_outputs(
    args: argparse.Namespace,
    features: list[tuple[int, int]],
    heaps: dict[tuple[int, int], list[tuple[float, int, dict[str, Any]]]],
    samples_processed: int,
    tokens_processed: int,
    started_at: float,
) -> None:
    profiles = {}
    for feature in features:
        key = f"L{feature[0]}:N{feature[1]}"
        profiles[key] = [
            item
            for _, _, item in sorted(heaps[feature], key=lambda entry: entry[0], reverse=True)
        ]
    payload = {
        "metadata": {
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "dataset_id": "Skylion007/openwebtext",
            "dataset_split": "train",
            "sampling": "streaming shuffle buffer followed by take",
            "seed": args.seed,
            "shuffle_buffer": args.shuffle_buffer,
            "num_samples": args.num_samples,
            "samples_processed": samples_processed,
            "tokens_processed": tokens_processed,
            "max_length": args.max_length,
            "batch_size": args.batch_size,
            "top_k": args.top_k,
            "context_tokens_each_side": args.context_tokens,
            "activation_definition": "input to mlp.down_proj (post-SwiGLU)",
            "elapsed_seconds_this_run": time.monotonic() - started_at,
            "complete": samples_processed >= args.num_samples,
            "features": [f"L{layer}:N{neuron}" for layer, neuron in features],
        },
        "profiles": profiles,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(args.output)
    write_html(payload, args.output.with_suffix(".html"))


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    features = load_features(args.features)
    by_layer: dict[int, list[int]] = defaultdict(list)
    for layer, neuron in features:
        by_layer[layer].append(neuron)

    heaps: dict[tuple[int, int], list[tuple[float, int, dict[str, Any]]]] = {
        feature: [] for feature in features
    }
    samples_processed = 0
    tokens_processed = 0
    serial = 0
    if args.resume:
        prior = json.loads(args.output.read_text())
        metadata = prior["metadata"]
        expected = {
            "seed": args.seed,
            "shuffle_buffer": args.shuffle_buffer,
            "num_samples": args.num_samples,
            "max_length": args.max_length,
            "top_k": args.top_k,
        }
        for key, value in expected.items():
            if metadata[key] != value:
                raise RuntimeError(f"Resume mismatch for {key}: {metadata[key]} != {value}")
        samples_processed = int(metadata["samples_processed"])
        tokens_processed = int(metadata["tokens_processed"])
        for feature in features:
            key = f"L{feature[0]}:N{feature[1]}"
            for item in prior["profiles"][key]:
                serial += 1
                heapq.heappush(heaps[feature], (float(item["activation"]), serial, item))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    model, touched = load_model_to_cuda(
        AutoModelForCausalLM,
        MODEL_ID,
        revision=MODEL_REVISION,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        touch="auto",
        return_touched=True,
    )
    model.eval()
    captured: dict[int, torch.Tensor] = {}
    handles = []
    for layer, neurons in by_layer.items():
        index = torch.tensor(neurons, device=model.device)

        def hook(_module: torch.nn.Module, inputs: tuple[torch.Tensor, ...], *, layer=layer, index=index) -> None:
            captured[layer] = inputs[0].detach().index_select(-1, index).float().cpu()

        handles.append(model.model.layers[layer].mlp.down_proj.register_forward_pre_hook(hook))

    stream = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
    stream = stream.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)
    if samples_processed:
        stream = stream.skip(samples_processed)
    remaining = args.num_samples - samples_processed
    iterator = iter(stream.take(remaining))
    started_at = time.monotonic()
    next_checkpoint = samples_processed + args.checkpoint_every

    try:
        while samples_processed < args.num_samples:
            batch_rows = []
            for _ in range(min(args.batch_size, args.num_samples - samples_processed)):
                try:
                    row = next(iterator)
                except StopIteration:
                    break
                if row["text"].strip():
                    batch_rows.append(row)
            if not batch_rows:
                break

            encoded = tokenizer(
                [row["text"] for row in batch_rows],
                add_special_tokens=False,
                truncation=True,
                max_length=args.max_length,
                padding=True,
                return_tensors="pt",
            )
            captured.clear()
            with torch.inference_mode():
                model.model(
                    input_ids=encoded.input_ids.to(model.device),
                    attention_mask=encoded.attention_mask.to(model.device),
                    use_cache=False,
                )

            batch_base = samples_processed
            valid_tokens = int(encoded.attention_mask.sum().item())
            tokens_processed += valid_tokens
            for layer, neurons in by_layer.items():
                layer_values = captured[layer]
                for local_neuron, neuron in enumerate(neurons):
                    values = layer_values[:, :, local_neuron].clone()
                    values[encoded.attention_mask == 0] = -torch.inf
                    local_k = min(args.top_k, valid_tokens)
                    top_values, top_offsets = torch.topk(values.reshape(-1), local_k)
                    for activation, offset in zip(top_values.tolist(), top_offsets.tolist(), strict=True):
                        heap = heaps[(layer, neuron)]
                        if len(heap) >= args.top_k and activation <= heap[0][0]:
                            continue
                        batch_index = offset // values.shape[1]
                        position = offset % values.shape[1]
                        sample_index = batch_base + batch_index
                        token_ids = encoded.input_ids[batch_index, : encoded.attention_mask[batch_index].sum()].tolist()
                        item = {
                            "activation": activation,
                            "sample_index": sample_index,
                            "token_position": position,
                            **render_context(
                                tokenizer,
                                token_ids,
                                values[batch_index, : len(token_ids)].tolist(),
                                position,
                                args.context_tokens,
                            ),
                        }
                        serial += 1
                        candidate = (activation, serial, item)
                        if len(heap) < args.top_k:
                            heapq.heappush(heap, candidate)
                        elif activation > heap[0][0]:
                            heapq.heapreplace(heap, candidate)

            samples_processed += len(batch_rows)
            if samples_processed >= next_checkpoint or samples_processed >= args.num_samples:
                save_outputs(
                    args, features, heaps, samples_processed, tokens_processed, started_at
                )
                elapsed = max(time.monotonic() - started_at, 1e-6)
                print(
                    f"{samples_processed:,}/{args.num_samples:,} documents · "
                    f"{tokens_processed:,} tokens · "
                    f"{(samples_processed - (args.num_samples - remaining)) / elapsed:.1f} docs/s"
                )
                next_checkpoint += args.checkpoint_every
    finally:
        for handle in handles:
            handle.remove()

    save_outputs(args, features, heaps, samples_processed, tokens_processed, started_at)
    print(f"Features: {len(features)}")
    print(f"Touched CPU pages: {touched / 1024**3:.2f} GiB")
    print(f"Saved profiles: {args.output.resolve()}")
    print(f"Saved browser: {args.output.with_suffix('.html').resolve()}")
    captured.clear()
    del iterator, stream, model
    gc.collect()
    torch.cuda.empty_cache()
    # PyArrow's streaming filesystem can leave a native callback alive during
    # CPython finalization on this GB10 image. All outputs are already atomically
    # saved, so exit without running extension-module destructors.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
