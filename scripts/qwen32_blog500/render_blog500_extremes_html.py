from __future__ import annotations

import argparse
import csv
import html
import json
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render an interactive highest/lowest report for the blog-500 audit."
    )
    parser.add_argument(
        "result_dir",
        nargs="?",
        type=Path,
        default=Path("outputs/blog500_pile1m"),
    )
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--contexts-per-side", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.count < 1 or args.contexts_per_side < 0:
        parser.error("--count must be positive and --contexts-per-side non-negative")
    return args


def main() -> None:
    args = parse_args()
    summary = json.loads((args.result_dir / "summary.json").read_text())
    metrics = json.loads((args.result_dir / "neuron_metrics.json").read_text())
    if len(metrics) != int(summary["neuron_count"]):
        raise RuntimeError("Metric count does not match summary.json")

    contexts: defaultdict[str, dict[str, list[dict]]] = defaultdict(
        lambda: {"negative": [], "positive": []}
    )
    with (args.result_dir / "extremes.csv").open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            rank = int(row["rank"])
            if rank > args.contexts_per_side:
                continue
            key = f"{row['layer']}:{row['neuron']}"
            contexts[key][row["polarity"]].append(
                {
                    "rank": rank,
                    "activation": float(row["activation"]),
                    "target_token": row["target_token"],
                    "token_piece": row["token_piece"],
                    "left_context": row["left_context"],
                    "right_context": row["right_context"],
                    "pile_set_name": row["pile_set_name"],
                    "dataset_index": int(row["dataset_index"]),
                    "document_token_index": int(row["document_token_index"]),
                }
            )
    for sides in contexts.values():
        for records in sides.values():
            records.sort(key=lambda row: row["rank"])

    payload = {
        "summary": {
            "model_id": summary["model_id"],
            "dataset_id": summary["dataset_id"],
            "token_count": summary["token_count"],
            "neuron_count": summary["neuron_count"],
            "extremes_per_side": summary["extremes_per_side"],
            "bulk_absolute_quantile": summary["bulk_absolute_quantile"],
        },
        "display_count": min(args.count, len(metrics) // 2),
        "contexts_per_side": args.contexts_per_side,
        "metrics": metrics,
        "contexts": dict(contexts),
    }
    payload_json = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    title = html.escape(
        f"{summary['model_id']} blog neurons — highest and lowest metrics"
    )
    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
:root {{ color-scheme:light; --ink:#17212a; --muted:#687581; --line:#dce3e8;
 --paper:#fff; --wash:#f3f6f8; --low:#2463a5; --low-wash:#edf5fd;
 --high:#b54720; --high-wash:#fff2ec; --accent:#173f5f; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--wash); color:var(--ink); font:14px/1.45 system-ui,-apple-system,sans-serif; }}
header {{ position:sticky; top:0; z-index:10; display:flex; gap:24px; align-items:end; padding:18px 24px; background:rgba(255,255,255,.96); border-bottom:1px solid var(--line); backdrop-filter:blur(8px); }}
.heading {{ flex:1; min-width:0; }} h1 {{ margin:0 0 4px; font-size:22px; }} .sub {{ color:var(--muted); font-size:13px; }}
.control {{ min-width:270px; }} .control label {{ display:block; margin-bottom:4px; color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:.05em; }}
#metric {{ width:100%; padding:9px 10px; border:1px solid #c8d2d9; border-radius:7px; background:white; }}
.columns {{ display:grid; grid-template-columns:1fr 1fr; gap:18px; padding:18px; align-items:start; }}
.column-title {{ position:sticky; top:92px; z-index:5; margin:0 0 10px; padding:10px 13px; border-radius:8px; font-size:16px; backdrop-filter:blur(8px); }}
.low .column-title {{ color:var(--low); background:rgba(237,245,253,.96); }} .high .column-title {{ color:var(--high); background:rgba(255,242,236,.96); }}
.card {{ margin-bottom:11px; overflow:hidden; border:1px solid var(--line); border-radius:10px; background:var(--paper); box-shadow:0 1px 2px rgba(20,30,40,.035); }}
.card-head {{ display:flex; gap:10px; align-items:baseline; padding:10px 12px; border-bottom:1px solid var(--line); }}
.rank {{ color:var(--muted); }} .neuron {{ font:700 15px ui-monospace,SFMono-Regular,monospace; }} .value {{ margin-left:auto; font:700 15px ui-monospace,SFMono-Regular,monospace; }}
.blog {{ padding:9px 12px 5px; }} .blog a {{ color:var(--accent); font-weight:650; text-decoration:none; }} .blog a:hover {{ text-decoration:underline; }}
.stats {{ display:flex; flex-wrap:wrap; gap:5px 12px; padding:4px 12px 10px; color:var(--muted); font:11px ui-monospace,SFMono-Regular,monospace; }}
.tails {{ display:grid; grid-template-columns:1fr 1fr; gap:1px; border-top:1px solid var(--line); background:var(--line); }}
.tail {{ min-width:0; background:white; }} .tail h3 {{ margin:0; padding:7px 10px; font-size:12px; }} .tail.negative h3 {{ color:var(--low); background:var(--low-wash); }} .tail.positive h3 {{ color:var(--high); background:var(--high-wash); }}
.sample {{ border-top:1px solid #edf0f2; }} .sample-meta {{ display:flex; gap:8px; padding:5px 9px; color:var(--muted); font:10px ui-monospace,SFMono-Regular,monospace; }} .sample-meta strong {{ color:var(--ink); }}
.context {{ margin:0; padding:7px 9px 9px; max-height:120px; overflow:auto; white-space:pre-wrap; overflow-wrap:anywhere; font:10.5px/1.45 ui-monospace,SFMono-Regular,monospace; }}
mark {{ padding:1px 2px; border-radius:3px; font:inherit; }} .negative mark {{ background:#bfdbf7; }} .positive mark {{ background:#ffd0bd; }}
.empty {{ padding:10px; color:var(--muted); font-size:12px; }}
@media (max-width:1050px) {{ header {{ align-items:stretch; flex-direction:column; gap:10px; }} .control {{ min-width:0; }} .columns {{ grid-template-columns:1fr; }} .column-title {{ position:static; }} }}
@media (max-width:650px) {{ .tails {{ grid-template-columns:1fr; }} }}
</style>
</head>
<body>
<header><div class="heading"><h1>{title}</h1><div class="sub">{html.escape(summary['dataset_id'])} · {summary['token_count']:,} tokens per neuron · {summary['neuron_count']} blog neurons · exact population moments</div></div><div class="control"><label for="metric">Rank highest and lowest by</label><select id="metric"></select></div></header>
<main class="columns"><section class="column low"><h2 class="column-title" id="lowTitle"></h2><div id="low"></div></section><section class="column high"><h2 class="column-title" id="highTitle"></h2><div id="high"></div></section></main>
<script id="data" type="application/json">{payload_json}</script>
<script>
const data=JSON.parse(document.getElementById('data').textContent);
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));
const fmt=x=>x==null?'n/a':Number(x).toLocaleString(undefined,{{maximumSignificantDigits:6}});
const options=[
 ['normalized_zero_anchored_tail_score','Normalized zero-anchored tail',r=>r.normalized_zero_anchored_tail_score],
 ['excess_kurtosis','Excess kurtosis',r=>r.excess_kurtosis],
 ['pearson_kurtosis','Pearson kurtosis',r=>r.pearson_kurtosis],
 ['absolute_skewness','Absolute skewness',r=>Math.abs(r.population_skewness)],
 ['population_skewness','Signed skewness',r=>r.population_skewness],
 ['raw_zero_tail','Raw zero-anchored tail',r=>Math.max(r.negative_zero_anchored_tail,r.positive_zero_anchored_tail)]
];
const config=key=>options.find(option=>option[0]===key);
function sample(record,side){{return `<div class="sample"><div class="sample-meta"><span>#${{record.rank}}</span><strong>${{fmt(record.activation)}}</strong><span>${{esc(record.pile_set_name)}} · doc ${{record.dataset_index}}, token ${{record.document_token_index}}</span></div><pre class="context">${{esc(record.left_context)}}<mark>${{esc(record.target_token)}}</mark>${{esc(record.right_context)}}</pre></div>`;}}
function tail(records,side){{return `<section class="tail ${{side}}"><h3>${{side==='negative'?'Negative':'Positive'}} extremes</h3>${{records.length?records.map(record=>sample(record,side)).join(''):'<div class="empty">No contexts retained</div>'}}</section>`;}}
function card(row,rank,value){{const key=row.layer+':'+row.neuron, examples=data.contexts[key]||{{negative:[],positive:[]}};return `<article class="card"><div class="card-head"><span class="rank">#${{rank}}</span><span class="neuron">L${{row.layer}} / N${{row.neuron}}</span><span class="value">${{fmt(value)}}</span></div><div class="blog"><a href="${{esc(row.url)}}" target="_blank" rel="noreferrer">${{esc(row.title)}}</a></div><div class="stats"><span>tail=${{fmt(row.normalized_zero_anchored_tail_score)}} (${{esc(row.tail_direction)}})</span><span>k=${{fmt(row.excess_kurtosis)}}</span><span>skew=${{fmt(row.population_skewness)}}</span><span>bulk=${{fmt(row.bulk_absolute_scale)}}</span></div><div class="tails">${{tail(examples.negative,'negative')}}${{tail(examples.positive,'positive')}}</div></article>`;}}
function render(){{const selected=config(document.getElementById('metric').value), valid=data.metrics.map(row=>[selected[2](row),row]).filter(([value])=>value!=null&&Number.isFinite(value)).sort((a,b)=>a[0]-b[0]), count=Math.min(data.display_count,Math.floor(valid.length/2)), low=valid.slice(0,count), high=valid.slice(-count).reverse();document.getElementById('lowTitle').textContent=`Lowest ${{count}} · ${{selected[1]}}`;document.getElementById('highTitle').textContent=`Highest ${{count}} · ${{selected[1]}}`;document.getElementById('low').innerHTML=low.map(([value,row],index)=>card(row,index+1,value)).join('');document.getElementById('high').innerHTML=high.map(([value,row],index)=>card(row,index+1,value)).join('');}}
const selector=document.getElementById('metric');options.forEach(option=>{{const node=document.createElement('option');node.value=option[0];node.textContent=option[1];selector.appendChild(node);}});selector.addEventListener('change',render);render();
</script>
</body>
</html>
"""
    output = args.output or args.result_dir / "highest_lowest_report.html"
    output.write_text(document, encoding="utf-8")
    print(f"Wrote {output.resolve()}")
    print(f"Metrics: {len(metrics)}")
    print(f"Contexts embedded: {sum(len(records) for sides in contexts.values() for records in sides.values())}")


if __name__ == "__main__":
    main()
