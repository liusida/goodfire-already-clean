from __future__ import annotations

import csv
import html
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "outputs/layer0/park_neuron_pile10k/hits.csv"
OUTPUT = ROOT / "outputs/layer0/park_neuron_pile10k/hits.html"


def main() -> None:
    with INPUT.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    for row in rows:
        row["dataset_index"] = int(row["dataset_index"])
        row["document_token_index"] = int(row["document_token_index"])
        row["token_id"] = int(row["token_id"])
        row["activation"] = float(row["activation"])

    token_counts = Counter(row["target_token"] for row in rows)
    source_counts = Counter(row["pile_set_name"] for row in rows)
    data = json.dumps(rows, ensure_ascii=False).replace("</", "<\\/")
    token_options = "".join(
        f'<option value="{html.escape(token, quote=True)}">{html.escape(repr(token))} ({count})</option>'
        for token, count in token_counts.most_common()
    )
    source_options = "".join(
        f'<option value="{html.escape(source, quote=True)}">{html.escape(source)} ({count})</option>'
        for source, count in source_counts.most_common()
    )

    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Qwen3-32B L0/N15647 · Pile-10k hits</title>
<style>
:root{{--ink:#20272b;--muted:#68757c;--line:#dce3e6;--paper:#fff;--wash:#f4f7f8;--accent:#28627b;--hot:#d9480f}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--wash);color:var(--ink);font:14px/1.5 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}}
header,main{{width:min(1180px,calc(100% - 32px));margin:auto}} header{{padding:34px 0 20px}} h1{{font-size:25px;margin:0 0 6px}} .sub{{color:var(--muted);margin:0}}
.stats{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;margin:20px 0}} .stat{{background:var(--paper);border:1px solid var(--line);border-radius:8px;padding:13px 15px}} .stat b{{display:block;font-size:21px}} .stat span{{color:var(--muted);font-size:12px}}
.tools{{position:sticky;top:0;z-index:2;display:grid;grid-template-columns:2fr 1fr 1fr 1fr;gap:8px;padding:12px;background:rgba(244,247,248,.96);border-bottom:1px solid var(--line);backdrop-filter:blur(8px)}}
input,select{{width:100%;padding:9px 10px;border:1px solid #c8d2d6;border-radius:6px;background:#fff;color:var(--ink);font:inherit}} .status{{color:var(--muted);margin:14px 2px}}
.hit{{background:var(--paper);border:1px solid var(--line);border-radius:9px;margin:0 0 10px;overflow:hidden}} .head{{display:grid;grid-template-columns:90px 90px 1fr auto;align-items:center;gap:12px;padding:11px 14px;cursor:pointer}}
.score{{font:700 17px ui-monospace,SFMono-Regular,Menlo,monospace;color:var(--hot)}} .token{{font:600 15px ui-monospace,SFMono-Regular,Menlo,monospace}} .meta{{color:var(--muted);font-size:12px}} .source{{color:var(--accent);font-weight:600}}
.context{{display:none;border-top:1px solid var(--line);padding:15px 17px 17px;white-space:pre-wrap;overflow-wrap:anywhere;font:14px/1.65 ui-monospace,SFMono-Regular,Menlo,monospace;background:#fbfcfc}} .hit.open .context{{display:block}} mark{{background:#ffd8bd;color:#7b2d00;border-radius:3px;padding:1px 2px}}
.pager{{display:flex;justify-content:center;align-items:center;gap:10px;padding:18px 0 36px}} button{{border:1px solid #bdcbd1;background:#fff;color:var(--accent);border-radius:6px;padding:8px 13px;font:inherit;cursor:pointer}} button:disabled{{opacity:.4;cursor:default}}
@media(max-width:760px){{.stats{{grid-template-columns:repeat(2,1fr)}}.tools{{grid-template-columns:1fr 1fr}}.head{{grid-template-columns:72px 1fr}}.head .meta{{grid-column:1/-1}}}}
</style>
</head>
<body>
<header>
  <h1>Qwen3-32B · L0/N15647 · Pile-10k hits</h1>
  <p class="sub">Every token position with post-SwiGLU activation &gt; 8. Click a row to expand its 32-token context.</p>
  <div class="stats">
    <div class="stat"><b>{len(rows):,}</b><span>high-activation positions</span></div>
    <div class="stat"><b>{len(token_counts)}</b><span>unique token IDs</span></div>
    <div class="stat"><b>15,392,600</b><span>corpus tokens scanned</span></div>
    <div class="stat"><b>{max(row['activation'] for row in rows):.3f}</b><span>maximum activation</span></div>
  </div>
</header>
<main>
  <div class="tools">
    <input id="search" type="search" placeholder="Search contexts…">
    <select id="token"><option value="">All tokens</option>{token_options}</select>
    <select id="source"><option value="">All sources</option>{source_options}</select>
    <select id="sort"><option value="activation-desc">Strongest first</option><option value="activation-asc">Weakest first</option><option value="corpus">Corpus order</option></select>
  </div>
  <p class="status" id="status"></p>
  <section id="hits"></section>
  <div class="pager"><button id="prev">Previous</button><span id="page"></span><button id="next">Next</button></div>
</main>
<script>const DATA={data};</script>
<script>
const PAGE_SIZE=100, $=s=>document.querySelector(s); let page=0, filtered=[];
const esc=s=>String(s).replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));
function marked(r){{return esc(r.left_context)+'<mark>'+esc(r.target_token)+'</mark>'+esc(r.right_context)}}
function apply(){{
  const q=$('#search').value.toLowerCase(), token=$('#token').value, source=$('#source').value;
  filtered=DATA.filter(r=>(!token||r.target_token===token)&&(!source||r.pile_set_name===source)&&(!q||r.marked_context.toLowerCase().includes(q)));
  const sort=$('#sort').value;
  if(sort==='activation-desc') filtered.sort((a,b)=>b.activation-a.activation);
  else if(sort==='activation-asc') filtered.sort((a,b)=>a.activation-b.activation);
  else filtered.sort((a,b)=>a.dataset_index-b.dataset_index||a.document_token_index-b.document_token_index);
  page=0; render();
}}
function render(){{
  const pages=Math.max(1,Math.ceil(filtered.length/PAGE_SIZE)); page=Math.min(page,pages-1);
  const slice=filtered.slice(page*PAGE_SIZE,(page+1)*PAGE_SIZE);
  $('#status').textContent=`${{filtered.length.toLocaleString()}} matching hits`;
  $('#hits').innerHTML=slice.map(r=>`<article class="hit"><div class="head"><span class="score">${{r.activation.toFixed(3)}}</span><span class="token">${{esc(JSON.stringify(r.target_token))}}</span><span class="meta"><span class="source">${{esc(r.pile_set_name)}}</span> · document ${{r.dataset_index.toLocaleString()}} · token ${{r.document_token_index.toLocaleString()}} · ID ${{r.token_id}}</span><span>⌄</span></div><div class="context">${{marked(r)}}</div></article>`).join('');
  document.querySelectorAll('.hit .head').forEach(h=>h.addEventListener('click',()=>h.parentElement.classList.toggle('open')));
  $('#page').textContent=`Page ${{page+1}} of ${{pages}}`; $('#prev').disabled=page===0; $('#next').disabled=page>=pages-1;
}}
$('#search').addEventListener('input',apply); $('#token').addEventListener('change',apply); $('#source').addEventListener('change',apply); $('#sort').addEventListener('change',apply);
$('#prev').addEventListener('click',()=>{{page--;render();scrollTo(0,0)}}); $('#next').addEventListener('click',()=>{{page++;render();scrollTo(0,0)}}); apply();
</script>
</body>
</html>"""
    OUTPUT.write_text(document, encoding="utf-8")
    print(f"Read {len(rows):,} records from {INPUT}")
    print(f"Saved {OUTPUT} ({OUTPUT.stat().st_size / 1024**2:.2f} MiB)")


if __name__ == "__main__":
    main()
