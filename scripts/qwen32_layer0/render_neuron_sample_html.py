from __future__ import annotations

import argparse
import csv
import html
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import AutoTokenizer


def byte_decoder() -> dict[str, int]:
    """Inverse of the GPT-2/Qwen byte-to-visible-Unicode vocabulary mapping."""
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a tabbed HTML explorer for a multi-neuron Pile-10k scan."
    )
    parser.add_argument("scan_dir", type=Path)
    parser.add_argument("--chunk-rows", type=int, default=100_000)
    parser.add_argument("--histogram-bins", type=int, default=180)
    parser.add_argument("--recompute-stats", action="store_true")
    parser.add_argument("--recompute-histograms", action="store_true")
    parser.add_argument("--refresh-contexts", action="store_true")
    return parser.parse_args()


def reconstruct_contexts(records: list[dict], summary: dict) -> list[dict]:
    """Decode context as original text, avoiding UTF-8 damage at token boundaries."""
    tokenizer = AutoTokenizer.from_pretrained(
        summary["model_id"], revision=summary["model_revision"]
    )
    dataset = load_dataset(
        summary["dataset_id"],
        split="train",
        revision=summary["dataset_revision"],
    )
    by_document: defaultdict[int, list[dict]] = defaultdict(list)
    for record in records:
        by_document[int(record["dataset_index"])].append(record)
    context_tokens = int(summary["context_tokens_each_side"])
    decode_byte = byte_decoder()
    for dataset_index, document_records in by_document.items():
        text = dataset[dataset_index]["text"]
        encoded = tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
            truncation=False,
        )
        offsets = encoded["offset_mapping"]
        input_ids = encoded["input_ids"]
        for record in document_records:
            position = int(record["document_token_index"])
            if position >= len(offsets):
                raise RuntimeError(
                    f"Token position {position} is outside document {dataset_index}"
                )
            context_start = max(0, position - context_tokens)
            context_end = min(len(offsets), position + context_tokens + 1)
            left_start = offsets[context_start][0]
            target_start, target_end = offsets[position]
            right_end = offsets[context_end - 1][1]
            span_first = position
            while span_first > 0 and (
                offsets[span_first - 1][0] < target_end
                and offsets[span_first - 1][1] > target_start
            ):
                span_first -= 1
            span_last = position
            while span_last + 1 < len(offsets) and (
                offsets[span_last + 1][0] < target_end
                and offsets[span_last + 1][1] > target_start
            ):
                span_last += 1
            span_part = position - span_first + 1
            span_count = span_last - span_first + 1
            pieces = tokenizer.convert_ids_to_tokens(input_ids[span_first : span_last + 1])
            piece_bytes = [bytes(decode_byte[char] for char in piece) for piece in pieces]
            group_start = min(offsets[index][0] for index in range(span_first, span_last + 1))
            prefix_bytes = len(text[group_start:target_start].encode("utf-8"))
            piece_start = sum(len(value) for value in piece_bytes[: span_part - 1])
            piece_end = piece_start + len(piece_bytes[span_part - 1])
            byte_count = len(text[target_start:target_end].encode("utf-8"))
            byte_start = max(piece_start, prefix_bytes) - prefix_bytes + 1
            byte_end = min(piece_end, prefix_bytes + byte_count) - prefix_bytes
            # Byte-fallback pieces belonging to one Unicode character can share an
            # offset. Highlighting that complete source span is preferable to �.
            record["left_context"] = text[left_start:target_start]
            record["target_token"] = text[target_start:target_end]
            record["right_context"] = text[target_end:right_end]
            record["display_span"] = text[target_start:target_end]
            record["span_token_part"] = span_part
            record["span_token_count"] = span_count
            record["span_byte_start"] = byte_start
            record["span_byte_end"] = byte_end
            record["span_byte_count"] = byte_count
            record["token_bytes_hex"] = piece_bytes[span_part - 1].hex(" ")
            record["marked_context"] = (
                record["left_context"]
                + "⟦"
                + record["target_token"]
                + "⟧"
                + record["right_context"]
            )
    return records


def compute_stats(
    scores: np.memmap, neurons: list[int], chunk_rows: int
) -> list[dict]:
    rows, columns = scores.shape
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
    moment_2 = sums_2 / rows
    std = np.sqrt(moment_2)
    with np.errstate(divide="ignore", invalid="ignore"):
        skewness = (sums_3 / rows) / std**3
        pearson_kurtosis = (sums_4 / rows) / moment_2**2

    return [
        {
            "neuron": neuron,
            "count": rows,
            "minimum": float(minimum[column]),
            "maximum": float(maximum[column]),
            "mean": float(mean[column]),
            "population_standard_deviation": float(std[column]),
            "population_skewness": float(skewness[column]),
            "pearson_kurtosis": float(pearson_kurtosis[column]),
            "excess_kurtosis": float(pearson_kurtosis[column] - 3.0),
        }
        for column, neuron in enumerate(neurons)
    ]


def compute_histograms(
    scores: np.memmap, stats: list[dict], bins: int, chunk_rows: int
) -> list[dict]:
    rows, columns = scores.shape
    minima = np.array([row["minimum"] for row in stats], dtype=np.float64)
    maxima = np.array([row["maximum"] for row in stats], dtype=np.float64)
    widths = maxima - minima
    widths[widths == 0] = 1.0
    scales = bins / widths
    counts = np.zeros((columns, bins), dtype=np.int64)
    column_offsets = np.arange(columns, dtype=np.int64) * bins
    for start in range(0, rows, chunk_rows):
        chunk = np.asarray(scores[start : start + chunk_rows], dtype=np.float64)
        indices = ((chunk - minima) * scales).astype(np.int64)
        np.clip(indices, 0, bins - 1, out=indices)
        keys = indices + column_offsets
        counts += np.bincount(
            keys.ravel(), minlength=columns * bins
        ).reshape(columns, bins)
    return [
        {
            "neuron": stats[column]["neuron"],
            "minimum": float(minima[column]),
            "maximum": float(maxima[column]),
            "bin_count": bins,
            "counts": counts[column].tolist(),
            "y_scale": "log10(count + 1)",
        }
        for column in range(columns)
    ]


def render_histogram_images(
    scan_dir: Path, histograms: list[dict], thresholds: dict[int, dict]
) -> dict[str, str]:
    image_dir = scan_dir / "histograms"
    image_dir.mkdir(exist_ok=True)
    paths = {}
    for histogram in histograms:
        neuron = int(histogram["neuron"])
        counts = np.asarray(histogram["counts"], dtype=np.int64)
        edges = np.linspace(
            histogram["minimum"], histogram["maximum"], len(counts) + 1
        )
        fig, ax = plt.subplots(figsize=(10.5, 3.2), dpi=140)
        ax.stairs(counts, edges, fill=True, color="#397792", alpha=0.9)
        ax.set_yscale("log")
        ax.axvline(0, color="#59636c", linewidth=1, linestyle=":", label="zero")
        threshold = thresholds[neuron]
        ax.axvline(
            threshold["negative_cutoff_20th_smallest"],
            color="#2463a5",
            linewidth=1.2,
            linestyle="--",
            label="20th-lowest cutoff",
        )
        ax.axvline(
            threshold["positive_cutoff_20th_largest"],
            color="#b54720",
            linewidth=1.2,
            linestyle="--",
            label="20th-highest cutoff",
        )
        ax.set_title(f"Layer 0 / Neuron {neuron} — all token activations")
        ax.set_xlabel("Post-SwiGLU activation")
        ax.set_ylabel("Token positions (log scale)")
        ax.grid(axis="y", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper right")
        fig.tight_layout()
        filename = f"neuron_{neuron}.png"
        fig.savefig(image_dir / filename, bbox_inches="tight")
        plt.close(fig)
        paths[str(neuron)] = f"histograms/{filename}"
    return paths


def main() -> None:
    args = parse_args()
    summary = json.loads((args.scan_dir / "summary.json").read_text(encoding="utf-8"))
    neurons = [int(value) for value in summary["neurons"]]
    shape = tuple(int(value) for value in summary["scores"]["shape"])
    if shape[1] != len(neurons):
        raise RuntimeError("Score columns do not match the neuron list")
    scores_path = args.scan_dir / summary["scores"]["path"]
    expected_bytes = int(np.prod(shape)) * np.dtype("<f4").itemsize
    if scores_path.stat().st_size != expected_bytes:
        raise RuntimeError("Score file size does not match summary.json")

    stats_path = args.scan_dir / "distribution_stats.json"
    if stats_path.exists() and not args.recompute_stats:
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
    else:
        scores = np.memmap(scores_path, mode="r", dtype="<f4", shape=shape)
        stats = compute_stats(scores, neurons, args.chunk_rows)
        stats_path.write_text(
            json.dumps(stats, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )

    histogram_path = args.scan_dir / "distribution_histograms.json"
    if histogram_path.exists() and not args.recompute_histograms:
        histograms = json.loads(histogram_path.read_text(encoding="utf-8"))
    else:
        scores = np.memmap(scores_path, mode="r", dtype="<f4", shape=shape)
        histograms = compute_histograms(
            scores, stats, args.histogram_bins, args.chunk_rows
        )
        histogram_path.write_text(
            json.dumps(histograms, separators=(",", ":")) + "\n", encoding="utf-8"
        )

    thresholds = {}
    with (args.scan_dir / summary["extremes"]["thresholds_csv_path"]).open(
        newline="", encoding="utf-8"
    ) as threshold_file:
        for row in csv.DictReader(threshold_file):
            thresholds[int(row["neuron"])] = {
                key: float(value)
                for key, value in row.items()
                if key not in {"layer", "neuron"}
            }
    histogram_images = render_histogram_images(
        args.scan_dir, histograms, thresholds
    )

    display_records_path = args.scan_dir / "display_extremes.json"
    if display_records_path.exists() and not args.refresh_contexts:
        records = json.loads(display_records_path.read_text(encoding="utf-8"))
    else:
        with (args.scan_dir / summary["extremes"]["jsonl_path"]).open(
            encoding="utf-8"
        ) as extremes_file:
            records = [json.loads(line) for line in extremes_file]
        records = reconstruct_contexts(records, summary)
        display_records_path.write_text(
            json.dumps(records, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    extremes: defaultdict[int, dict[str, list[dict]]] = defaultdict(
        lambda: {"negative": [], "positive": []}
    )
    for record in records:
        extremes[int(record["neuron"])][record["polarity"]].append(record)

    stats_by_neuron = {str(row["neuron"]): row for row in stats}
    display_neurons = sorted(
        neurons,
        key=lambda neuron: stats_by_neuron[str(neuron)]["excess_kurtosis"],
        reverse=True,
    )
    payload = {
        "summary": {
            "layer": summary["layer"],
            "model_id": summary["model_id"],
            "dataset_id": summary["dataset_id"],
            "tokens": summary["scored_tokens_per_neuron"],
            "sample_seed": summary["sample_seed"],
            "histogram_bins": histograms[0]["bin_count"],
        },
        "neurons": display_neurons,
        "stats": stats_by_neuron,
        "histogram_images": histogram_images,
        "thresholds": {str(key): value for key, value in thresholds.items()},
        "extremes": {str(key): value for key, value in extremes.items()},
    }
    payload_json = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    title = html.escape(
        f"{summary['model_id']} layer {summary['layer']} — {len(neurons)} neurons"
    )
    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
:root {{ color-scheme: light; --ink:#182027; --muted:#64717d; --line:#dce3e8;
  --paper:#fff; --wash:#f4f7f8; --neg:#2463a5; --neg-wash:#edf5fd;
  --pos:#b54720; --pos-wash:#fff2ec; --accent:#173f5f; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--wash); color:var(--ink); font:14px/1.45 system-ui,-apple-system,sans-serif; }}
header {{ padding:24px 28px 18px; background:var(--paper); border-bottom:1px solid var(--line); }}
h1 {{ margin:0 0 5px; font-size:23px; }} .sub {{ color:var(--muted); }}
.layout {{ display:grid; grid-template-columns:210px minmax(0,1fr); min-height:calc(100vh - 92px); }}
nav {{ position:sticky; top:0; height:100vh; overflow:auto; padding:14px; border-right:1px solid var(--line); background:#f9fbfc; }}
#filter {{ width:100%; padding:9px 10px; border:1px solid #cbd5dc; border-radius:7px; margin-bottom:10px; }}
.tab {{ display:block; width:100%; border:0; background:transparent; text-align:left; padding:7px 10px; border-radius:6px; color:#34424e; cursor:pointer; }}
.tab:hover {{ background:#eaf0f3; }} .tab.active {{ background:var(--accent); color:white; font-weight:650; }}
main {{ padding:22px 26px 50px; min-width:0; }}
.neuron-title {{ margin:0 0 14px; font-size:22px; }}
.stats {{ display:grid; grid-template-columns:repeat(8,minmax(105px,1fr)); gap:8px; margin-bottom:20px; }}
.stat {{ background:var(--paper); border:1px solid var(--line); border-radius:8px; padding:10px 11px; }}
.stat label {{ display:block; color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:.04em; margin-bottom:3px; }}
.stat strong {{ font:600 14px ui-monospace,SFMono-Regular,monospace; }}
.distribution {{ background:var(--paper); border:1px solid var(--line); border-radius:9px; padding:12px 14px 8px; margin-bottom:20px; }}
.distribution h2 {{ margin:0 0 2px; font-size:16px; }} .chart-note {{ color:var(--muted); font-size:12px; margin-bottom:8px; }}
.histogram {{ display:block; width:100%; height:auto; }}
.columns {{ display:grid; grid-template-columns:1fr 1fr; gap:18px; align-items:start; }}
.column h2 {{ position:sticky; top:0; z-index:2; margin:0 0 10px; padding:10px 12px; border-radius:8px; font-size:16px; backdrop-filter:blur(8px); }}
.negative h2 {{ color:var(--neg); background:rgba(237,245,253,.94); }}
.positive h2 {{ color:var(--pos); background:rgba(255,242,236,.94); }}
.card {{ background:var(--paper); border:1px solid var(--line); border-radius:9px; margin-bottom:10px; overflow:hidden; }}
.card-head {{ display:flex; gap:9px; align-items:baseline; padding:8px 11px; border-bottom:1px solid var(--line); }}
.rank {{ color:var(--muted); }} .score {{ font:700 14px ui-monospace,SFMono-Regular,monospace; }}
.token {{ font:600 13px ui-monospace,SFMono-Regular,monospace; overflow-wrap:anywhere; }}
.piece {{ color:var(--muted); font:12px ui-monospace,SFMono-Regular,monospace; }}
.source {{ margin-left:auto; color:var(--muted); font-size:12px; text-align:right; }}
.context {{ margin:0; padding:11px; white-space:pre-wrap; overflow-wrap:anywhere; font:12px/1.52 ui-monospace,SFMono-Regular,monospace; }}
mark {{ padding:1px 2px; border-radius:3px; font:inherit; }} .negative mark {{ background:#bfdbf7; }} .positive mark {{ background:#ffd0bd; }}
@media (max-width:1100px) {{ .stats {{ grid-template-columns:repeat(4,1fr); }} }}
@media (max-width:760px) {{ .layout {{ display:block; }} nav {{ position:relative; height:auto; max-height:230px; border-right:0; border-bottom:1px solid var(--line); }} main {{ padding:16px; }} .columns {{ grid-template-columns:1fr; }} .stats {{ grid-template-columns:repeat(2,1fr); }} }}
</style>
</head>
<body>
<header><h1>{title}</h1><div class="sub">{summary['dataset_id']} · {summary['scored_tokens_per_neuron']:,} tokens per neuron · seed {summary['sample_seed']} · population moments</div></header>
<div class="layout"><nav><input id="filter" placeholder="Find neuron…" autocomplete="off"><div id="tabs"></div></nav><main id="content"></main></div>
<script id="data" type="application/json">{payload_json}</script>
<script>
const data=JSON.parse(document.getElementById('data').textContent); let active=data.neurons[0];
const esc=s=>String(s).replace(/[&<>\"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;',"'":'&#39;'}}[c]));
const num=x=>Number(x).toLocaleString(undefined,{{maximumSignificantDigits:7}});
const kurt=x=>Number(x).toLocaleString(undefined,{{maximumSignificantDigits:4}});
function tabs(q=''){{ const box=document.getElementById('tabs'); box.innerHTML=''; const needle=q.trim();
 data.neurons.filter(n=>!needle||String(n).includes(needle)).forEach(n=>{{const b=document.createElement('button'); b.className='tab'+(n===active?' active':''); b.textContent='L'+data.summary.layer+' / N'+n+' (k='+kurt(data.stats[String(n)].excess_kurtosis)+')'; b.onclick=()=>{{active=n;tabs(document.getElementById('filter').value);render();}}; box.appendChild(b);}}); }}
function card(r,side){{const parts=r.span_token_count||1; const byteRange=r.span_byte_start===r.span_byte_end?`${{r.span_byte_start}}`:`${{r.span_byte_start}}–${{r.span_byte_end}}`; const part=parts>1?` · component ${{r.span_token_part}}/${{parts}} of ${{JSON.stringify(r.display_span)}} · UTF-8 byte ${{byteRange}}/${{r.span_byte_count}} (${{r.token_bytes_hex}})`:''; return `<article class="card"><div class="card-head"><span class="rank">#${{r.rank}}</span><span class="score">${{num(r.activation)}}</span><span class="token">token ${{r.token_id}}</span><span class="piece">piece ${{esc(JSON.stringify(r.token_piece))}}${{esc(part)}}</span><span class="source">${{esc(r.pile_set_name||'Unknown')}} · doc ${{r.dataset_index}}, token ${{r.document_token_index}}</span></div><pre class="context">${{esc(r.left_context)}}<mark>${{esc(r.target_token)}}</mark>${{esc(r.right_context)}}</pre></article>`;}}
function stat(label,value){{return `<div class="stat"><label>${{label}}</label><strong>${{num(value)}}</strong></div>`;}}
function render(){{const key=String(active), s=data.stats[key], t=data.thresholds[key], e=data.extremes[key];
 document.getElementById('content').innerHTML=`<h2 class="neuron-title">Layer ${{data.summary.layer}} · Neuron ${{active}}</h2><section class="stats">${{stat('minimum',s.minimum)}}${{stat('negative cutoff',t.negative_cutoff_20th_smallest)}}${{stat('mean',s.mean)}}${{stat('std. deviation',s.population_standard_deviation)}}${{stat('skewness',s.population_skewness)}}${{stat('excess kurtosis',s.excess_kurtosis)}}${{stat('positive cutoff',t.positive_cutoff_20th_largest)}}${{stat('maximum',s.maximum)}}</section><section class="distribution"><h2>Activation distribution</h2><div class="chart-note">All ${{data.summary.tokens.toLocaleString()}} token positions · ${{data.summary.histogram_bins}} bins · logarithmic count axis</div><img class="histogram" src="${{esc(data.histogram_images[key])}}" alt="Activation histogram for neuron ${{active}}"></section><section class="columns"><div class="column negative"><h2>Negative extremes · lowest 20</h2>${{e.negative.map(r=>card(r,'negative')).join('')}}</div><div class="column positive"><h2>Positive extremes · highest 20</h2>${{e.positive.map(r=>card(r,'positive')).join('')}}</div></section>`; }}
document.getElementById('filter').addEventListener('input',e=>tabs(e.target.value)); tabs(); render();
</script>
</body></html>
"""
    output_path = args.scan_dir / "neurons.html"
    output_path.write_text(document, encoding="utf-8")
    print(f"Statistics: {stats_path.resolve()}")
    print(f"Histograms: {histogram_path.resolve()}")
    print(f"HTML: {output_path.resolve()}")


if __name__ == "__main__":
    main()
