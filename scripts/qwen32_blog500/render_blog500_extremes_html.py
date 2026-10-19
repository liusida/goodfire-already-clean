from __future__ import annotations

import argparse
import csv
import html
import json
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a sortable 500-neuron explorer for the blog audit."
    )
    parser.add_argument(
        "result_dir", nargs="?", type=Path, default=Path("outputs/blog500_pile1m")
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = json.loads((args.result_dir / "summary.json").read_text())
    metric_rows = json.loads((args.result_dir / "neuron_metrics.json").read_text())
    if len(metric_rows) != int(summary["neuron_count"]):
        raise RuntimeError("Metric count does not match summary.json")
    metrics = {f"{row['layer']}:{row['neuron']}": row for row in metric_rows}
    if len(metrics) != len(metric_rows):
        raise RuntimeError("Duplicate layer/neuron pair in neuron_metrics.json")

    extremes: defaultdict[str, dict[str, list[dict]]] = defaultdict(
        lambda: {"negative": [], "positive": []}
    )
    with (args.result_dir / "extremes.csv").open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            key = f"{row['layer']}:{row['neuron']}"
            extremes[key][row["polarity"]].append(
                {
                    "rank": int(row["rank"]),
                    "activation": float(row["activation"]),
                    "dataset_index": int(row["dataset_index"]),
                    "pile_set_name": row["pile_set_name"],
                    "document_token_index": int(row["document_token_index"]),
                    "token_id": int(row["token_id"]),
                    "token_piece": row["token_piece"],
                    "target_token": row["target_token"],
                    "left_context": row["left_context"],
                    "right_context": row["right_context"],
                }
            )
    for key, sides in extremes.items():
        for side, records in sides.items():
            records.sort(key=lambda record: record["rank"])
            if len(records) != int(summary["extremes_per_side"]):
                raise RuntimeError(
                    f"Expected {summary['extremes_per_side']} {side} extremes "
                    f"for {key}, found {len(records)}"
                )
    if set(extremes) != set(metrics):
        raise RuntimeError("Extreme-context neurons do not match metric neurons")

    payload = {
        "summary": {
            "model_id": summary["model_id"],
            "dataset_id": summary["dataset_id"],
            "token_count": summary["token_count"],
            "neuron_count": summary["neuron_count"],
            "extremes_per_side": summary["extremes_per_side"],
        },
        "neurons": list(metrics),
        "metrics": metrics,
        "extremes": dict(extremes),
    }
    payload_json = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    title = html.escape(f"{summary['model_id']} — 500 blog neurons")
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{title}</title>
<style>
:root {{ color-scheme:light;--ink:#182027;--muted:#64717d;--line:#dce3e8;--paper:#fff;--wash:#f4f7f8;--neg:#2463a5;--neg-wash:#edf5fd;--pos:#b54720;--pos-wash:#fff2ec;--accent:#173f5f }}
* {{ box-sizing:border-box }} body {{ margin:0;background:var(--wash);color:var(--ink);font:14px/1.45 system-ui,-apple-system,sans-serif }}
header {{ padding:20px 26px 16px;background:var(--paper);border-bottom:1px solid var(--line) }} h1 {{ margin:0 0 4px;font-size:23px }} .sub {{ color:var(--muted) }}
.layout {{ display:grid;grid-template-columns:285px minmax(0,1fr);min-height:calc(100vh - 86px) }} nav {{ position:sticky;top:0;height:100vh;overflow:auto;padding:14px;border-right:1px solid var(--line);background:#f9fbfc }}
.nav-controls {{ position:sticky;top:-14px;z-index:5;margin:-14px -14px 7px;padding:14px 14px 7px;background:#f9fbfc }} .control-label {{ display:block;margin:0 0 4px;color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.04em }}
#metric,#filter {{ width:100%;padding:9px 10px;border:1px solid #cbd5dc;border-radius:7px;background:white }} #metric {{ margin-bottom:9px }} #filter {{ margin-bottom:3px }}
.pill {{ display:flex;width:100%;gap:7px;align-items:center;border:0;border-radius:7px;padding:7px 9px;background:transparent;color:#34424e;cursor:pointer;text-align:left }} .pill:hover {{ background:#eaf0f3 }} .pill.active {{ background:var(--accent);color:white }}
.pill-id {{ flex:1;font:650 12px ui-monospace,SFMono-Regular,monospace }} .pill-value {{ font:11px ui-monospace,SFMono-Regular,monospace;opacity:.8 }} main {{ min-width:0;padding:22px 25px 50px }}
.neuron-head {{ margin-bottom:14px }} .neuron-head h2 {{ margin:0 0 5px;font-size:22px }} .blog-link {{ color:var(--accent);font-weight:650;text-decoration:none }} .blog-link:hover {{ text-decoration:underline }} .blog-meta {{ margin-top:5px;color:var(--muted);font-size:12px }}
.stats {{ display:grid;grid-template-columns:repeat(5,minmax(120px,1fr));gap:8px;margin-bottom:18px }} .stat {{ min-width:0;padding:10px 11px;border:1px solid var(--line);border-radius:8px;background:var(--paper) }} .stat label {{ display:block;margin-bottom:3px;color:var(--muted);font-size:10px;text-transform:uppercase;letter-spacing:.04em }} .stat strong {{ font:600 13px ui-monospace,SFMono-Regular,monospace;overflow-wrap:anywhere }} .note {{ margin:-8px 0 18px;color:var(--muted);font-size:12px }}
.columns {{ display:grid;grid-template-columns:1fr 1fr;gap:18px;align-items:start }} .column h2 {{ position:sticky;top:0;z-index:3;margin:0 0 10px;padding:10px 12px;border-radius:8px;font-size:16px;backdrop-filter:blur(8px) }} .negative h2 {{ color:var(--neg);background:rgba(237,245,253,.95) }} .positive h2 {{ color:var(--pos);background:rgba(255,242,236,.95) }}
.card {{ margin-bottom:10px;overflow:hidden;border:1px solid var(--line);border-radius:9px;background:var(--paper) }} .card-head {{ display:flex;gap:9px;align-items:baseline;padding:8px 11px;border-bottom:1px solid var(--line) }} .rank {{ color:var(--muted) }} .score {{ font:700 14px ui-monospace,SFMono-Regular,monospace }} .token {{ font:600 12px ui-monospace,SFMono-Regular,monospace;overflow-wrap:anywhere }} .piece {{ color:var(--muted);font:11px ui-monospace,SFMono-Regular,monospace }} .source {{ margin-left:auto;color:var(--muted);font-size:11px;text-align:right }}
.context {{ margin:0;padding:11px;white-space:pre-wrap;overflow-wrap:anywhere;font:11.5px/1.5 ui-monospace,SFMono-Regular,monospace }} mark {{ padding:1px 2px;border-radius:3px;font:inherit }} .negative mark {{ background:#bfdbf7 }} .positive mark {{ background:#ffd0bd }}
@media(max-width:1150px){{.stats{{grid-template-columns:repeat(3,1fr)}}}} @media(max-width:800px){{.layout{{display:block}}nav{{position:relative;height:auto;max-height:300px;border-right:0;border-bottom:1px solid var(--line)}}.nav-controls{{top:0}}main{{padding:16px}}.columns{{grid-template-columns:1fr}}.stats{{grid-template-columns:repeat(2,1fr)}}}}
</style></head><body>
<header><h1>{title}</h1><div class="sub">{html.escape(summary['dataset_id'])} · {summary['token_count']:,} tokens per neuron · all {summary['neuron_count']} neurons · {summary['extremes_per_side']} extremes per side</div></header>
<div class="layout"><nav><div class="nav-controls"><label class="control-label" for="metric">Show and order by</label><select id="metric"></select><input id="filter" placeholder="Find layer, neuron, or title…" autocomplete="off"></div><div id="pills"></div></nav><main id="content"></main></div>
<script id="data" type="application/json">{payload_json}</script><script>
const data=JSON.parse(document.getElementById('data').textContent);let active=null,selectedMetric='normalized_zero_anchored_tail_score';
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c])); const fmt=x=>x==null?'n/a':Number(x).toLocaleString(undefined,{{maximumSignificantDigits:6}});
const metricOptions=[['normalized_zero_anchored_tail_score','Normalized zero-anchored tail','tail / bulk',r=>r.normalized_zero_anchored_tail_score],['excess_kurtosis','Excess kurtosis','k',r=>r.excess_kurtosis],['pearson_kurtosis','Pearson kurtosis','K',r=>r.pearson_kurtosis],['absolute_skewness','Absolute skewness','|skew|',r=>Math.abs(r.population_skewness)],['population_skewness','Signed skewness','skew',r=>r.population_skewness],['raw_zero_tail','Raw zero-anchored tail','raw tail',r=>Math.max(r.negative_zero_anchored_tail,r.positive_zero_anchored_tail)]];
const config=()=>metricOptions.find(option=>option[0]===selectedMetric); function ordered(){{const getter=config()[3];return [...data.neurons].sort((a,b)=>{{const av=getter(data.metrics[a]),bv=getter(data.metrics[b]);if(av==null)return 1;if(bv==null)return -1;return bv-av||data.metrics[a].layer-data.metrics[b].layer||data.metrics[a].neuron-data.metrics[b].neuron}})}}
function pills(){{const query=document.getElementById('filter').value.trim().toLowerCase(),cfg=config(),box=document.getElementById('pills');box.innerHTML='';ordered().filter(key=>{{const row=data.metrics[key];return !query||key.includes(query)||('l'+row.layer).includes(query)||('n'+row.neuron).includes(query)||row.title.toLowerCase().includes(query)}}).forEach(key=>{{const row=data.metrics[key],button=document.createElement('button');button.className='pill'+(key===active?' active':'');button.innerHTML=`<span class="pill-id">L${{row.layer}} / N${{row.neuron}}</span><span class="pill-value">${{esc(cfg[2])}}=${{fmt(cfg[3](row))}}</span>`;button.onclick=()=>{{active=key;pills();render()}};box.appendChild(button)}})}}
function stat(label,value){{return `<div class="stat"><label>${{label}}</label><strong>${{fmt(value)}}</strong></div>`}} function textStat(label,value){{return `<div class="stat"><label>${{label}}</label><strong>${{esc(value)}}</strong></div>`}}
function card(record){{return `<article class="card"><div class="card-head"><span class="rank">#${{record.rank}}</span><span class="score">${{fmt(record.activation)}}</span><span class="token">token ${{record.token_id}}</span><span class="piece">piece ${{esc(JSON.stringify(record.token_piece))}}</span><span class="source">${{esc(record.pile_set_name||'Unknown')}} · doc ${{record.dataset_index}}, token ${{record.document_token_index}}</span></div><pre class="context">${{esc(record.left_context)}}<mark>${{esc(record.target_token)}}</mark>${{esc(record.right_context)}}</pre></article>`}}
function render(){{const row=data.metrics[active],sides=data.extremes[active];document.getElementById('content').innerHTML=`<section class="neuron-head"><h2>Layer ${{row.layer}} · Neuron ${{row.neuron}}</h2><a class="blog-link" href="${{esc(row.url)}}" target="_blank" rel="noreferrer">${{esc(row.title)}}</a><div class="blog-meta">${{esc(row.status)}} · interest ${{esc(row.interest_score)}} · cleanliness ${{esc(row.cleanliness_score)}} · ${{esc(row.id)}}</div></section><section class="stats">${{stat('normalized tail score',row.normalized_zero_anchored_tail_score)}}${{textStat('tail direction',row.tail_direction)}}${{stat('tail asymmetry',row.tail_asymmetry)}}${{stat('negative normalized tail',row.negative_normalized_zero_tail)}}${{stat('positive normalized tail',row.positive_normalized_zero_tail)}}${{stat('bulk |activation| q90',row.bulk_absolute_scale)}}${{stat('negative raw tail',row.negative_zero_anchored_tail)}}${{stat('positive raw tail',row.positive_zero_anchored_tail)}}${{stat('excess kurtosis',row.excess_kurtosis)}}${{stat('Pearson kurtosis',row.pearson_kurtosis)}}${{stat('skewness',row.population_skewness)}}${{stat('standard deviation',row.population_standard_deviation)}}${{stat('mean',row.mean)}}${{stat('minimum',row.minimum)}}${{stat('maximum',row.maximum)}}</section><div class="note">Normalized tail = recurrent 20th-extreme magnitude from zero ÷ Q0.90(|activation|). Population moments and bulk scale use all ${{row.count.toLocaleString()}} scored positions.</div><section class="columns"><div class="column negative"><h2>Negative extremes · lowest ${{sides.negative.length}}</h2>${{sides.negative.map(card).join('')}}</div><div class="column positive"><h2>Positive extremes · highest ${{sides.positive.length}}</h2>${{sides.positive.map(card).join('')}}</div></section>`}}
const selector=document.getElementById('metric');metricOptions.forEach(option=>{{const node=document.createElement('option');node.value=option[0];node.textContent=option[1]+' (descending)';selector.appendChild(node)}});selector.value=selectedMetric;selector.addEventListener('change',event=>{{selectedMetric=event.target.value;pills()}});document.getElementById('filter').addEventListener('input',pills);active=ordered()[0];pills();render();
</script></body></html>"""
    output = args.output or args.result_dir / "highest_lowest_report.html"
    output.write_text(document, encoding="utf-8")
    context_count = sum(
        len(records) for sides in extremes.values() for records in sides.values()
    )
    print(f"Wrote {output.resolve()}")
    print(f"Neurons: {len(metrics)}")
    print(f"Extreme contexts embedded: {context_count}")


if __name__ == "__main__":
    main()
