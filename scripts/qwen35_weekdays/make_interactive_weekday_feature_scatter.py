from __future__ import annotations

import csv
import json
from pathlib import Path


DATA = Path("outputs/weekdays/qwen35_08b_weekday_feature_matrix_values.csv")
OUTPUT = Path("outputs/weekdays/qwen35_08b_weekday_interactive_feature_scatter.html")


def main() -> None:
    with DATA.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    features = sorted({(int(row["layer"]), int(row["neuron"])) for row in rows})
    points: dict[tuple[int, str], dict[str, object]] = {}
    cutoffs: dict[str, float] = {}
    for row in rows:
        feature = f"L{int(row['layer'])}:N{int(row['neuron'])}"
        key = (int(row["template_index"]), row["weekday"])
        point = points.setdefault(
            key,
            {
                "templateIndex": int(row["template_index"]),
                "template": row["template"],
                "weekday": row["weekday"],
                "values": {},
            },
        )
        point["values"][feature] = float(row["value"])
        cutoffs[feature] = float(row["layer_cutoff"])

    payload = {
        "features": [f"L{layer}:N{neuron}" for layer, neuron in features],
        "cutoffs": cutoffs,
        "points": list(points.values()),
        "initialX": "L1:N2",
        "initialY": "L4:N279",
    }
    payload_json = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")

    html = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Interactive weekday feature scatter</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#f3f5f7;color:#17212b;font-family:Inter,system-ui,sans-serif}
header{background:#17212b;color:#fff;padding:24px 30px}h1{font-size:23px;margin:0 0 7px}header p{color:#cbd4dd;margin:0}
main{max-width:1300px;margin:auto;padding:18px}.panel{background:#fff;border:1px solid #d8dee5;border-radius:10px;box-shadow:0 1px 3px #0000000d;overflow:hidden}
.controls{display:flex;align-items:center;gap:18px;padding:13px 18px;border-bottom:1px solid #dfe4e9;background:#fafbfd;flex-wrap:wrap}
label{font-size:13px;font-weight:650}select{margin-left:7px;padding:7px 28px 7px 9px;border:1px solid #bdc7d1;border-radius:6px;background:#fff;color:#17212b;font-size:13px}
.hint{margin-left:auto;color:#697684;font-size:12px}.chart-wrap{position:relative;padding:8px 12px 16px}svg{display:block;width:100%;height:auto;min-height:580px}
.grid{stroke:#dfe4e9;stroke-width:1}.axis{stroke:#687684;stroke-width:1.2}.tick{fill:#53616f;font-size:12px}.axis-label{fill:#24313d;font-size:14px;font-weight:650}.title{fill:#17212b;font-size:17px;font-weight:700}
.cutoff{stroke-width:1.5;stroke-dasharray:6 5;opacity:.72}.point{stroke:#fff;stroke-width:1.3;opacity:.86;cursor:pointer;transition:r .1s,opacity .1s}.point:hover{r:7;opacity:1;stroke:#17212b;stroke-width:1.5}
.legend text{font-size:12px;fill:#354351}.tooltip{position:absolute;display:none;pointer-events:none;background:#17212b;color:#fff;padding:9px 11px;border-radius:6px;font-size:12px;line-height:1.45;box-shadow:0 4px 14px #0004;max-width:360px;z-index:5}.tooltip b{color:#fff}.tooltip span{color:#cbd4dd}
@media(max-width:700px){header{padding:20px 16px}main{padding:8px}.controls{gap:9px}.hint{width:100%;margin-left:0}svg{min-height:470px}}
</style>
</head>
<body>
<header><h1>Interactive weekday-token feature scatter</h1><p>Choose any discovered layer:neuron feature for each axis. Points are colored by weekday; hover for prompt details.</p></header>
<main><section class="panel"><div class="controls">
<label>X axis<select id="x-select"></select></label>
<button id="swap" type="button" title="Swap axes">⇄ Swap</button>
<label>Y axis<select id="y-select"></select></label>
<span class="hint">Dashed lines show each selected layer’s top-0.002% cutoff.</span>
</div><div class="chart-wrap"><svg id="chart" viewBox="0 0 1100 720" role="img"></svg><div id="tooltip" class="tooltip"></div></div></section></main>
<script>
const DATA = __PAYLOAD__;
const colors={Monday:'#4C78A8',Tuesday:'#F58518',Wednesday:'#54A24B',Thursday:'#E45756',Friday:'#B279A2',Saturday:'#FF9DA6',Sunday:'#9D755D'};
const days=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'];
const xSelect=document.getElementById('x-select'), ySelect=document.getElementById('y-select');
for(const feature of DATA.features){for(const select of [xSelect,ySelect]){const option=document.createElement('option');option.value=feature;option.textContent=feature;select.appendChild(option)}}
xSelect.value=DATA.initialX;ySelect.value=DATA.initialY;
const svg=document.getElementById('chart'), tooltip=document.getElementById('tooltip'), NS='http://www.w3.org/2000/svg';
function node(tag,attrs={},text=''){const n=document.createElementNS(NS,tag);for(const [k,v] of Object.entries(attrs))n.setAttribute(k,v);if(text)n.textContent=text;return n}
function niceExtent(values){let lo=Math.min(...values),hi=Math.max(...values),span=hi-lo||1;return [lo-span*.08,hi+span*.08]}
function ticks(lo,hi,n=6){const out=[];for(let i=0;i<=n;i++)out.push(lo+(hi-lo)*i/n);return out}
function render(){
  svg.replaceChildren();const xf=xSelect.value,yf=ySelect.value;
  const W=1100,H=720,m={l:95,r:190,t:65,b:80},pw=W-m.l-m.r,ph=H-m.t-m.b;
  const xs=DATA.points.map(p=>p.values[xf]),ys=DATA.points.map(p=>p.values[yf]);
  const [xmin,xmax]=niceExtent(xs),[ymin,ymax]=niceExtent(ys);
  const sx=v=>m.l+(v-xmin)/(xmax-xmin)*pw, sy=v=>m.t+ph-(v-ymin)/(ymax-ymin)*ph;
  svg.append(node('text',{x:m.l+pw/2,y:28,'text-anchor':'middle',class:'title'},`${xf} vs ${yf} at weekday token`));
  for(const v of ticks(xmin,xmax)){const x=sx(v);svg.append(node('line',{x1:x,y1:m.t,x2:x,y2:m.t+ph,class:'grid'}));svg.append(node('text',{x,y:m.t+ph+23,'text-anchor':'middle',class:'tick'},v.toFixed(3)))}
  for(const v of ticks(ymin,ymax)){const y=sy(v);svg.append(node('line',{x1:m.l,y1:y,x2:m.l+pw,y2:y,class:'grid'}));svg.append(node('text',{x:m.l-12,y:y+4,'text-anchor':'end',class:'tick'},v.toFixed(3)))}
  svg.append(node('line',{x1:m.l,y1:m.t+ph,x2:m.l+pw,y2:m.t+ph,class:'axis'}));svg.append(node('line',{x1:m.l,y1:m.t,x2:m.l,y2:m.t+ph,class:'axis'}));
  svg.append(node('text',{x:m.l+pw/2,y:H-18,'text-anchor':'middle',class:'axis-label'},`${xf} activation`));
  svg.append(node('text',{x:22,y:m.t+ph/2,'text-anchor':'middle',transform:`rotate(-90 22 ${m.t+ph/2})`,class:'axis-label'},`${yf} activation`));
  const xc=DATA.cutoffs[xf],yc=DATA.cutoffs[yf];
  if(xc>=xmin&&xc<=xmax)svg.append(node('line',{x1:sx(xc),y1:m.t,x2:sx(xc),y2:m.t+ph,class:'cutoff',stroke:'#4C78A8'}));
  if(yc>=ymin&&yc<=ymax)svg.append(node('line',{x1:m.l,y1:sy(yc),x2:m.l+pw,y2:sy(yc),class:'cutoff',stroke:'#E45756'}));
  for(const p of DATA.points){const c=node('circle',{cx:sx(p.values[xf]),cy:sy(p.values[yf]),r:5.3,fill:colors[p.weekday],class:'point'});
    c.addEventListener('mouseenter',()=>{tooltip.innerHTML=`<b>T${p.templateIndex+1} · ${p.weekday}</b><br><span>${p.template.replace('{weekday}',p.weekday)}</span><br>${xf}: ${p.values[xf].toFixed(6)}<br>${yf}: ${p.values[yf].toFixed(6)}`;tooltip.style.display='block'});
    c.addEventListener('mousemove',ev=>{const box=svg.parentElement.getBoundingClientRect();tooltip.style.left=`${ev.clientX-box.left+14}px`;tooltip.style.top=`${ev.clientY-box.top+14}px`});c.addEventListener('mouseleave',()=>tooltip.style.display='none');svg.append(c)}
  const legend=node('g',{class:'legend'});legend.append(node('text',{x:m.l+pw+35,y:m.t+4,'font-weight':'700'},'Weekday'));
  days.forEach((day,i)=>{const y=m.t+30+i*27;legend.append(node('circle',{cx:m.l+pw+43,cy:y,r:5.5,fill:colors[day]}));legend.append(node('text',{x:m.l+pw+57,y:y+4},day))});svg.append(legend);
}
xSelect.addEventListener('change',render);ySelect.addEventListener('change',render);document.getElementById('swap').addEventListener('click',()=>{const x=xSelect.value;xSelect.value=ySelect.value;ySelect.value=x;render()});render();
</script></body></html>"""
    OUTPUT.write_text(html.replace("__PAYLOAD__", payload_json))
    print(f"Features: {len(features)}")
    print(f"Points: {len(points)}")
    print(f"Saved interactive plot: {OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
