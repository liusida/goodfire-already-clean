from __future__ import annotations

import argparse
import csv
import html
import json
from collections import defaultdict
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an HTML token-by-layer report of significant MLP activations."
    )
    parser.add_argument(
        "--activations",
        type=Path,
        default=Path("outputs/weekdays/qwen35_08b_weekday_activations.pt"),
    )
    parser.add_argument(
        "--significant",
        type=Path,
        default=Path(
            "outputs/weekdays/qwen35_08b_weekday_layerwise_top_0.001pct_activations.csv"
        ),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path(
            "outputs/weekdays/qwen35_08b_weekday_layerwise_top_0.001pct_activations.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/weekdays/qwen35_08b_weekday_top_0.001pct_report.html"),
    )
    return parser.parse_args()


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def display_token(token: str) -> str:
    return token.replace(" ", "·").replace("\n", "↵")


def main() -> None:
    args = parse_args()
    artifact = torch.load(args.activations, map_location="cpu", weights_only=False)
    summary = json.loads(args.summary.read_text())

    significant: dict[tuple[str, int, int], list[dict[str, str]]] = defaultdict(list)
    with args.significant.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        key = (row["sample_id"], int(row["layer"]), int(row["token_position"]))
        significant[key].append(row)
    for entries in significant.values():
        entries.sort(key=lambda row: float(row["value"]), reverse=True)

    layer_cutoffs = {
        int(item["layer"]): float(item["top_fraction_cutoff"])
        for item in summary["layer_stats"]
    }
    max_value = max(float(row["value"]) for row in rows)
    num_layers = artifact["num_layers"]

    sections: list[str] = []
    for record_number, record in enumerate(artifact["records"], start=1):
        sample_id = record["sample_id"]
        tokens = record["tokens"]
        token_headers = []
        for position, (token, token_id) in enumerate(
            zip(tokens, record["input_ids"].tolist(), strict=True)
        ):
            weekday_class = " weekday-token" if position == record["weekday_position"] else ""
            token_headers.append(
                f'<th class="token{weekday_class}" title="token id {token_id}">'
                f'<span class="token-index">{position}</span>'
                f'<span class="token-text">{esc(display_token(token))}</span></th>'
            )

        layer_rows = []
        prompt_event_count = 0
        for layer in range(num_layers):
            cells = []
            layer_event_count = 0
            for position in range(len(tokens)):
                entries = significant.get((sample_id, layer, position), [])
                layer_event_count += len(entries)
                if not entries:
                    cells.append('<td class="empty">·</td>')
                    continue
                activation_lines = []
                strongest = max(float(entry["value"]) for entry in entries)
                intensity = min(strongest / max_value, 1.0)
                for entry in entries:
                    activation_lines.append(
                        '<span class="activation" '
                        f'title="layer percentile ≥ {float(entry["layer_percentile_lower_bound"]):.7f}; '
                        f'z = {float(entry["layer_z_score"]):.2f}">'
                        f'N{int(entry["neuron"])} '
                        f'<strong>{float(entry["value"]):.5g}</strong></span>'
                    )
                cells.append(
                    f'<td class="hit" style="--intensity:{intensity:.4f}">'
                    + "".join(activation_lines)
                    + "</td>"
                )
            prompt_event_count += layer_event_count
            layer_rows.append(
                f'<tr><th class="layer" title="raw cutoff {layer_cutoffs[layer]:.6g}">'
                f'L{layer}<span class="layer-count">{layer_event_count or ""}</span></th>'
                + "".join(cells)
                + "</tr>"
            )

        open_attribute = " open" if record_number == 1 else ""
        sections.append(
            f'<details class="prompt-card" data-weekday="{esc(record["weekday"].lower())}"'
            f' data-template="{record["template_index"]}"{open_attribute}>'
            f'<summary><span class="sample-number">{record_number:02d}</span> '
            f'<span class="sample-id">{esc(sample_id)}</span>'
            f'<span class="badge">{prompt_event_count} significant activations</span></summary>'
            f'<p class="prompt-text">{esc(record["prompt"])}</p>'
            '<div class="table-scroll"><table><thead><tr><th class="corner">Layer / token</th>'
            + "".join(token_headers)
            + "</tr></thead><tbody>"
            + "".join(layer_rows)
            + "</tbody></table></div></details>"
        )

    style = """
    :root { color-scheme: light; font-family: Inter, ui-sans-serif, system-ui, sans-serif; }
    * { box-sizing: border-box; }
    body { margin: 0; background: #f4f6f8; color: #17202a; }
    header { padding: 28px 32px 22px; background: #17202a; color: white; }
    h1 { margin: 0 0 8px; font-size: 24px; }
    header p { margin: 4px 0; color: #cbd5e1; max-width: 1050px; line-height: 1.45; }
    .toolbar { position: sticky; top: 0; z-index: 20; display: flex; gap: 10px; align-items: center;
      padding: 10px 32px; background: rgba(255,255,255,.96); border-bottom: 1px solid #d8dee6; }
    .toolbar button, .toolbar select { border: 1px solid #c3cad4; border-radius: 6px; background: white;
      padding: 7px 10px; color: #263442; }
    #visible-count { margin-left: auto; color: #5b6775; font-size: 13px; }
    main { padding: 18px 24px 60px; max-width: 100%; }
    .prompt-card { margin: 0 auto 14px; background: white; border: 1px solid #d9dee5;
      border-radius: 9px; box-shadow: 0 1px 2px rgba(0,0,0,.04); overflow: clip; }
    summary { cursor: pointer; padding: 13px 16px; font-weight: 650; background: #fbfcfd; }
    summary:hover { background: #f5f8fb; }
    .sample-number { display: inline-block; width: 30px; color: #8a96a3; font-variant-numeric: tabular-nums; }
    .sample-id { color: #263442; }
    .badge { float: right; font-size: 12px; font-weight: 600; color: #42627f; background: #e9f2fa;
      border-radius: 999px; padding: 3px 9px; }
    .prompt-text { margin: 13px 17px; color: #465463; }
    .table-scroll { overflow-x: auto; border-top: 1px solid #e0e5eb; }
    table { border-collapse: separate; border-spacing: 0; font-size: 11px; min-width: 100%; }
    th, td { border-right: 1px solid #e5e9ee; border-bottom: 1px solid #e5e9ee; min-width: 82px;
      max-width: 130px; height: 34px; padding: 4px 6px; text-align: center; vertical-align: middle; }
    thead th { position: sticky; top: 0; z-index: 4; background: #f4f7fa; }
    .corner, .layer { position: sticky; left: 0; z-index: 5; min-width: 96px; background: #f4f7fa; }
    .corner { z-index: 6; }
    .token-text { display: block; font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 12px; overflow-wrap: anywhere; }
    .token-index { display: block; color: #8995a3; font-size: 9px; font-weight: 500; }
    .weekday-token { background: #fff2cc; box-shadow: inset 0 -3px #e8b923; }
    .layer { text-align: left; padding-left: 10px; color: #405264; font-variant-numeric: tabular-nums; }
    .layer-count { float: right; min-width: 17px; color: #a84923; font-size: 10px; text-align: right; }
    td.empty { color: #dfe4e9; }
    td.hit { background: rgba(230, 92, 43, calc(.12 + var(--intensity) * .55)); color: #4c1d0b; }
    .activation { display: block; white-space: nowrap; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
    .activation + .activation { margin-top: 2px; border-top: 1px solid rgba(108,42,17,.16); padding-top: 2px; }
    .hidden { display: none; }
    @media (max-width: 700px) { header { padding: 22px 18px; } .toolbar { padding: 8px 18px; }
      main { padding: 12px 8px 40px; } .badge { display: none; } }
    """

    script = """
    const cards = [...document.querySelectorAll('.prompt-card')];
    const weekday = document.getElementById('weekday-filter');
    const count = document.getElementById('visible-count');
    function applyFilter() {
      let visible = 0;
      for (const card of cards) {
        const show = weekday.value === 'all' || card.dataset.weekday === weekday.value;
        card.classList.toggle('hidden', !show);
        if (show) visible++;
      }
      count.textContent = `${visible} / ${cards.length} prompts`;
    }
    weekday.addEventListener('change', applyFilter);
    document.getElementById('expand').addEventListener('click', () => cards.forEach(c => {
      if (!c.classList.contains('hidden')) c.open = true;
    }));
    document.getElementById('collapse').addEventListener('click', () => cards.forEach(c => c.open = false));
    applyFilter();
    """

    weekdays = artifact["weekdays"]
    options = "".join(
        f'<option value="{esc(day.lower())}">{esc(day)}</option>' for day in weekdays
    )
    document = (
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>Qwen3.5 weekday MLP activation report</title>"
        f"<style>{style}</style></head><body>"
        "<header><h1>Qwen3.5-0.8B weekday MLP activation report</h1>"
        f"<p>{len(artifact['records'])} prompts · {num_layers} layers · "
        f"{artifact['intermediate_size']:,} neurons per layer · "
        f"{len(rows):,} entries selected by the top 0.001% within each layer.</p>"
        "<p>Each table divides one prompt into token columns. Cells show significant "
        "post-SwiGLU activations as neuron index and raw value. Yellow headers mark the weekday token; "
        "hover an activation for its layer-relative percentile and z-score.</p></header>"
        '<div class="toolbar"><label>Weekday <select id="weekday-filter">'
        f'<option value="all">All weekdays</option>{options}</select></label>'
        '<button id="expand" type="button">Expand visible</button>'
        '<button id="collapse" type="button">Collapse all</button>'
        '<span id="visible-count"></span></div>'
        f"<main>{''.join(sections)}</main><script>{script}</script></body></html>"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(document)
    print(f"Prompts: {len(artifact['records'])}")
    print(f"Layers per prompt: {num_layers}")
    print(f"Significant activation entries: {len(rows)}")
    print(f"Populated token/layer cells: {len(significant)}")
    print(f"Saved report: {args.output.resolve()}")


if __name__ == "__main__":
    main()
