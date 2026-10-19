#!/usr/bin/env python3
"""Project the sampled Layer-0 MLP directions into vocabulary-logit space."""

from __future__ import annotations

import csv
import html
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from gb10_load_llm import load_model_to_cuda
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "Qwen/Qwen3-32B"
MODEL_REVISION = "9216db5781bf21249d130ec9da846c4624c16137"
SOURCE_DIR = Path("outputs/layer0/sample_100_seed_0_pile10k")
OUTPUT_DIR = SOURCE_DIR / "logit_projection"
TOP_K = 20
HISTOGRAM_BINS = 180


def token_text(tokenizer: AutoTokenizer, token_id: int) -> str:
    value = tokenizer.decode(
        [token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    return value.replace("\n", "\\n").replace("\r", "\\r") or "∅"


def ranked_tokens(
    values: torch.Tensor, tokenizer: AutoTokenizer, *, largest: bool
) -> list[dict]:
    scores, indices = torch.topk(values, TOP_K, largest=largest, sorted=True)
    return [
        {
            "rank": rank,
            "token_id": int(token_id),
            "token": token_text(tokenizer, int(token_id)),
            "projection": float(score),
        }
        for rank, (score, token_id) in enumerate(zip(scores, indices), start=1)
    ]


def render_histogram(values: np.ndarray, neuron: int, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.3), constrained_layout=True)
    ax.hist(values, bins=HISTOGRAM_BINS, color="#4b86a0", linewidth=0)
    ax.set_yscale("log")
    ax.axvline(0, color="#7c8790", linestyle=":", linewidth=1)
    ax.set_xlabel("Direct vocabulary-logit projection")
    ax.set_ylabel("Vocabulary tokens (log count)")
    ax.set_title(f"Layer 0 · Neuron {neuron} — full-vocabulary projection")
    ax.grid(axis="y", alpha=0.2)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)


def render_report(rows: list[dict], output: Path) -> None:
    payload = json.dumps(rows, ensure_ascii=False).replace("</", "<\\/")
    document = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Layer-0 logit projection kurtosis</title><style>
:root{{--ink:#182027;--muted:#66727d;--line:#dce3e8;--paper:#fff;--wash:#f4f7f8;--accent:#173f5f}}*{{box-sizing:border-box}}body{{margin:0;background:var(--wash);color:var(--ink);font:14px/1.45 system-ui,sans-serif}}header{{padding:20px 26px;background:white;border-bottom:1px solid var(--line)}}h1{{margin:0 0 4px;font-size:23px}}.sub{{color:var(--muted)}}.layout{{display:grid;grid-template-columns:280px minmax(0,1fr);min-height:calc(100vh - 90px)}}nav{{position:sticky;top:0;height:100vh;overflow:auto;padding:13px;border-right:1px solid var(--line);background:#f9fbfc}}.controls{{position:sticky;top:-13px;z-index:3;margin:-13px -13px 7px;padding:13px;background:#f9fbfc}}select{{width:100%;padding:8px;border:1px solid #cbd5dc;border-radius:7px;background:white}}button{{display:flex;width:100%;gap:7px;padding:7px 9px;border:0;border-radius:7px;background:transparent;color:#34424e;text-align:left;cursor:pointer}}button:hover{{background:#eaf0f3}}button.active{{background:var(--accent);color:white}}button span:first-child{{flex:1;font:650 12px ui-monospace,monospace}}button span:last-child{{font:11px ui-monospace,monospace}}main{{min-width:0;padding:22px 25px 50px}}h2{{margin:0 0 13px}}.stats{{display:grid;grid-template-columns:repeat(5,minmax(120px,1fr));gap:8px;margin-bottom:15px}}.stat{{padding:9px 10px;border:1px solid var(--line);border-radius:8px;background:white}}.stat label{{display:block;color:var(--muted);font-size:10px;text-transform:uppercase}}.stat strong{{font:600 12px ui-monospace,monospace}}img{{display:block;width:min(100%,1050px);border:1px solid var(--line);border-radius:9px;background:white}}.tokens{{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:15px}}section{{padding:12px;border:1px solid var(--line);border-radius:9px;background:white}}h3{{margin:0 0 7px;font-size:14px}}ol{{margin:0;padding-left:27px}}li{{padding:3px 4px;border-bottom:1px solid #edf0f2}}li code{{display:inline-block;min-width:130px;white-space:pre-wrap}}li span{{float:right;font:12px ui-monospace,monospace}}.neg span{{color:#2463a5}}.pos span{{color:#b54720}}@media(max-width:800px){{.layout{{display:block}}nav{{position:relative;height:auto;max-height:280px}}.stats{{grid-template-columns:repeat(2,1fr)}}.tokens{{grid-template-columns:1fr}}}}
</style></head><body><header><h1>Layer 0 · sampled neurons — vocabulary-logit projections</h1><div class="sub"><code>lm_head.weight @ down_proj.weight[:, neuron]</code> · full vocabulary · raw unembedding · no final RMSNorm</div></header><div class="layout"><nav><div class="controls"><select id="metric"><option value="pearson_kurtosis">Pearson kurtosis</option><option value="excess_kurtosis">Excess kurtosis</option><option value="absolute_skewness">Absolute skewness</option><option value="std">Standard deviation</option></select></div><div id="pills"></div></nav><main id="content"></main></div><script id="data" type="application/json">{payload}</script><script>
const data=JSON.parse(document.getElementById('data').textContent);let active=null;const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));const fmt=x=>Number(x).toLocaleString(undefined,{{maximumSignificantDigits:6}});const value=(r,k)=>k==='absolute_skewness'?Math.abs(r.population_skewness):r[k];function order(){{const k=document.getElementById('metric').value;return [...data].sort((a,b)=>value(b,k)-value(a,k))}}function pills(){{const k=document.getElementById('metric').value;document.getElementById('pills').innerHTML=order().map(r=>`<button data-n="${{r.neuron}}" class="${{r.neuron===active?'active':''}}"><span>L0 / N${{r.neuron}}</span><span>${{fmt(value(r,k))}}</span></button>`).join('');document.querySelectorAll('button').forEach(b=>b.onclick=()=>{{active=Number(b.dataset.n);pills();render()}})}}function list(rows,cls){{return `<ol class="${{cls}}">${{rows.map(r=>`<li><code>${{esc(JSON.stringify(r.token))}}</code><span>${{fmt(r.projection)}}</span></li>`).join('')}}</ol>`}}function render(){{const r=data.find(x=>x.neuron===active);document.getElementById('content').innerHTML=`<h2>Layer 0 · Neuron ${{r.neuron}}</h2><div class="stats"><div class="stat"><label>Pearson kurtosis</label><strong>${{fmt(r.pearson_kurtosis)}}</strong></div><div class="stat"><label>Excess kurtosis</label><strong>${{fmt(r.excess_kurtosis)}}</strong></div><div class="stat"><label>Skewness</label><strong>${{fmt(r.population_skewness)}}</strong></div><div class="stat"><label>Mean</label><strong>${{fmt(r.mean)}}</strong></div><div class="stat"><label>Std</label><strong>${{fmt(r.std)}}</strong></div><div class="stat"><label>Minimum</label><strong>${{fmt(r.minimum)}}</strong></div><div class="stat"><label>Maximum</label><strong>${{fmt(r.maximum)}}</strong></div><div class="stat"><label>Direction L2 norm</label><strong>${{fmt(r.direction_l2_norm)}}</strong></div><div class="stat"><label>Vocabulary</label><strong>${{r.count.toLocaleString()}}</strong></div></div><img src="histograms/neuron_${{r.neuron}}.png"><div class="tokens"><section><h3>Most negative tokens</h3>${{list(r.tokens.negative,'neg')}}</section><section><h3>Most positive tokens</h3>${{list(r.tokens.positive,'pos')}}</section></div>`}}document.getElementById('metric').onchange=pills;active=order()[0].neuron;pills();render();
</script></body></html>"""
    output.write_text(document, encoding="utf-8")


def main() -> None:
    neurons = json.loads((SOURCE_DIR / "neurons.json").read_text())
    if len(neurons) != 100 or len(set(neurons)) != 100:
        raise ValueError("Expected 100 unique neurons")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    print("Loading pinned Qwen3-32B BF16 checkpoint...")
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
    down_weight = model.model.layers[0].mlp.down_proj.weight
    unembedding = model.get_output_embeddings().weight
    index = torch.tensor(neurons, device=down_weight.device, dtype=torch.long)
    directions = down_weight.index_select(1, index)
    with torch.inference_mode():
        projections = torch.matmul(unembedding, directions)
    projections = projections.transpose(0, 1).detach().float().cpu()
    direction_norms = directions.detach().float().norm(dim=0).cpu()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    histogram_dir = OUTPUT_DIR / "histograms"
    histogram_dir.mkdir(exist_ok=True)
    rows = []
    for row_index, neuron in enumerate(neurons):
        values = projections[row_index]
        values64 = values.double()
        centered = values64 - values64.mean()
        variance = centered.square().mean()
        std = variance.sqrt()
        skewness = centered.pow(3).mean() / std.pow(3)
        pearson = centered.pow(4).mean() / variance.square()
        array = values.numpy()
        render_histogram(array, neuron, histogram_dir / f"neuron_{neuron}.png")
        rows.append(
            {
                "neuron": neuron,
                "count": int(values.numel()),
                "mean": float(values64.mean()),
                "std": float(std),
                "population_skewness": float(skewness),
                "pearson_kurtosis": float(pearson),
                "excess_kurtosis": float(pearson - 3),
                "minimum": float(values.min()),
                "maximum": float(values.max()),
                "direction_l2_norm": float(direction_norms[row_index]),
                "tokens": {
                    "negative": ranked_tokens(values, tokenizer, largest=False),
                    "positive": ranked_tokens(values, tokenizer, largest=True),
                },
            }
        )

    metadata = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "layer": 0,
        "projection_definition": "lm_head.weight @ down_proj.weight[:, neuron]",
        "final_norm_applied": False,
        "vocabulary_size": int(unembedding.shape[0]),
        "neuron_count": len(neurons),
        "touched_model_gib": touched / 1024**3,
        "neurons": rows,
    }
    (OUTPUT_DIR / "results.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    with (OUTPUT_DIR / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
        keys = [key for key in rows[0] if key != "tokens"]
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        writer.writerows({key: row[key] for key in keys} for row in rows)
    render_report(rows, OUTPUT_DIR / "report.html")
    print(f"Projected {len(neurons)} neurons over {unembedding.shape[0]:,} tokens")
    print(f"Touched model GiB: {touched / 1024**3:.3f}")
    print(f"Wrote {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
