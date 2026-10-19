from __future__ import annotations

import csv
import html
import json
from collections import defaultdict
from pathlib import Path

import torch


ACTIVATIONS = Path("outputs/weekdays/qwen35_08b_weekday_activations.pt")
SIGNIFICANT = Path("outputs/weekdays/qwen35_08b_weekday_layerwise_top_0.002pct_activations.csv")
SUMMARY = Path("outputs/weekdays/qwen35_08b_weekday_layerwise_top_0.002pct_activations.json")
OUTPUT = Path("outputs/weekdays/qwen35_08b_weekday_top_0.002pct_grouped_report.html")


def e(value: object) -> str:
    return html.escape(str(value), quote=True)


def token_label(token: str) -> str:
    return token.replace(" ", "·").replace("\n", "↵")


def main() -> None:
    artifact = torch.load(ACTIVATIONS, map_location="cpu", weights_only=False)
    summary = json.loads(SUMMARY.read_text())
    percentage = float(summary["percentage_per_layer"])
    with SIGNIFICANT.open(newline="") as handle:
        selected_rows = list(csv.DictReader(handle))

    selected: dict[tuple[str, int, int], list[dict[str, str]]] = defaultdict(list)
    for row in selected_rows:
        selected[(row["sample_id"], int(row["layer"]), int(row["token_position"]))].append(row)
    for entries in selected.values():
        entries.sort(key=lambda row: float(row["value"]), reverse=True)

    cutoffs = {int(s["layer"]): float(s["top_fraction_cutoff"]) for s in summary["layer_stats"]}
    max_value = max(float(row["value"]) for row in selected_rows)
    num_layers = artifact["num_layers"]

    def table(record: dict, start: int, end: int, *, shared: bool = False) -> tuple[str, int]:
        headers = []
        for position in range(start, end):
            token = record["tokens"][position]
            weekday = position == record["weekday_position"]
            headers.append(
                f'<th class="token{" weekday" if weekday else ""}" title="token id '
                f'{record["input_ids"][position].item()}"><small>{position}</small>'
                f'<code>{e(token_label(token))}</code></th>'
            )
        body = []
        event_count = 0
        for layer in range(num_layers):
            cells = []
            row_count = 0
            for position in range(start, end):
                entries = selected.get((record["sample_id"], layer, position), [])
                row_count += len(entries)
                if not entries:
                    cells.append('<td class="empty">·</td>')
                    continue
                strongest = max(float(item["value"]) for item in entries)
                lines = "".join(
                    f'<span title="z={float(item["layer_z_score"]):.2f}">N{int(item["neuron"])} '
                    f'<b>{float(item["value"]):.5g}</b></span>'
                    for item in entries
                )
                cells.append(
                    f'<td class="hit" style="--heat:{min(strongest / max_value, 1):.3f}">{lines}</td>'
                )
            event_count += row_count
            body.append(
                f'<tr><th class="layer" title="layer cutoff {cutoffs[layer]:.6g}">L{layer}'
                f'<i>{row_count or ""}</i></th>{"".join(cells)}</tr>'
            )
        kind = "shared-table" if shared else "branch-table"
        return (
            f'<div class="scroll {kind}"><table><thead><tr><th class="corner">Layer / token</th>'
            f'{"".join(headers)}</tr></thead><tbody>{"".join(body)}</tbody></table></div>',
            event_count,
        )

    def branch_comparison(records: list[dict], branch_at: int) -> tuple[str, int]:
        weekday_headers = []
        token_headers = []
        for record in records:
            branch_length = len(record["tokens"]) - branch_at
            weekday_headers.append(
                f'<th class="weekday-group" colspan="{branch_length}">{e(record["weekday"])}</th>'
            )
            for position in range(branch_at, len(record["tokens"])):
                token_headers.append(
                    f'<th class="token{" weekday" if position == branch_at else ""}" '
                    f'title="token id {record["input_ids"][position].item()}">'
                    f'<small>{position}</small><code>{e(token_label(record["tokens"][position]))}</code></th>'
                )

        body = []
        total_events = 0
        for layer in range(num_layers):
            cells = []
            layer_events = 0
            for record in records:
                for position in range(branch_at, len(record["tokens"])):
                    entries = selected.get((record["sample_id"], layer, position), [])
                    layer_events += len(entries)
                    if not entries:
                        cells.append('<td class="empty">·</td>')
                        continue
                    strongest = max(float(item["value"]) for item in entries)
                    lines = "".join(
                        f'<span title="z={float(item["layer_z_score"]):.2f}">N{int(item["neuron"])} '
                        f'<b>{float(item["value"]):.5g}</b></span>'
                        for item in entries
                    )
                    cells.append(
                        f'<td class="hit" style="--heat:{min(strongest / max_value, 1):.3f}">{lines}</td>'
                    )
            total_events += layer_events
            body.append(
                f'<tr><th class="layer" title="layer cutoff {cutoffs[layer]:.6g}">L{layer}'
                f'<i>{layer_events or ""}</i></th>{"".join(cells)}</tr>'
            )
        return (
            '<div class="scroll comparison"><table><thead>'
            f'<tr><th class="corner" rowspan="2">Layer / token</th>{"".join(weekday_headers)}</tr>'
            f'<tr>{"".join(token_headers)}</tr></thead><tbody>{"".join(body)}</tbody></table></div>',
            total_events,
        )

    template_sections = []
    for template_index in range(len(artifact["templates"])):
        records = [r for r in artifact["records"] if r["template_index"] == template_index]
        records.sort(key=lambda r: artifact["weekdays"].index(r["weekday"]))
        canonical = records[0]
        branch_at = canonical["weekday_position"]
        shared_table, shared_count = table(canonical, 0, branch_at, shared=True)

        comparison_table, branch_count = branch_comparison(records, branch_at)

        prefix_text = "".join(canonical["tokens"][:branch_at])
        template_sections.append(
            f'<details class="template-card" open><summary><span>Template {template_index + 1}</span> '
            f'<code>{e(artifact["templates"][template_index])}</code></summary>'
            f'<details class="shared-details"><summary>Shared causal prefix '
            f'<em>{shared_count} significant</em> <code>{e(prefix_text)}…</code></summary>'
            f'<div class="shared"><p>{e(prefix_text)}<mark>…</mark></p>{shared_table}</div></details>'
            f'<div class="branches"><h2>Weekday branches side by side '
            f'<em>{branch_count} significant</em></h2>{comparison_table}</div>'
            f'</details>'
        )

    css = """
    *{box-sizing:border-box}body{margin:0;background:#f3f5f7;color:#17212b;font-family:Inter,system-ui,sans-serif}
    header{background:#17212b;color:white;padding:28px 32px}header h1{margin:0 0 8px;font-size:25px}header p{margin:5px 0;color:#cad3dd;max-width:1100px;line-height:1.45}
    main{padding:20px 24px 60px}.template-card{background:white;border:1px solid #d7dde4;border-radius:10px;margin:0 0 20px;overflow:hidden;box-shadow:0 1px 3px #0000000d}
    summary{padding:15px 18px;cursor:pointer;background:#f8fafc;font-weight:700;border-bottom:1px solid #e3e7eb}summary span{margin-right:12px}summary code{font-weight:400;color:#475666}
    .shared,.branches{padding:16px 18px}.shared-details{background:#fbfdff;border-bottom:3px solid #d8e8f5}.shared-details>summary{padding:11px 18px;background:#edf5fb;border-bottom:0;font-size:14px}.shared-details[open]>summary{border-bottom:1px solid #d8e8f5}.shared-details>summary code{font-weight:400;color:#617282;margin-left:9px}.shared h2,.branches>h2{font-size:16px;margin:0 0 7px}.shared p{margin:5px 0 13px;color:#405266}.shared mark{background:#fff0ad;padding:0 4px}
    h2 em,h3 em{font-style:normal;font-weight:500;color:#6d7884;font-size:11px;margin-left:8px}
    .scroll{overflow-x:auto;border:1px solid #dfe4e9;border-radius:7px}table{border-collapse:separate;border-spacing:0;font-size:11px;min-width:100%}
    th,td{height:33px;min-width:84px;max-width:132px;padding:4px 6px;text-align:center;border-right:1px solid #e6eaee;border-bottom:1px solid #e6eaee}thead th{background:#edf2f6;position:sticky;top:0;z-index:3}
    .corner,.layer{position:sticky;left:0;z-index:4;min-width:94px;background:#edf2f6}.corner{z-index:5}.layer{text-align:left;padding-left:10px;color:#405264}.layer i{float:right;font-size:10px;color:#a44725;font-style:normal}
    .token small{display:block;color:#8b96a1;font-size:9px}.token code{display:block;font-size:12px;overflow-wrap:anywhere}.weekday{background:#ffe69a;box-shadow:inset 0 -3px #dba900}.weekday-group{background:#fff5cf;font-size:12px;border-bottom:1px solid #e0c663}.weekday-group+ .weekday-group{border-left:3px solid #d3b442}.comparison td:nth-child(2n+2),.comparison thead tr:nth-child(2) th:nth-child(2n+1){border-left:3px solid #d3b442}
    td.empty{color:#e0e4e8}.hit{background:rgba(225,82,31,calc(.13 + var(--heat)*.58));color:#511e0a}.hit span{display:block;white-space:nowrap;font-family:ui-monospace,monospace}.hit span+span{border-top:1px solid #7e331f33;margin-top:2px;padding-top:2px}
    @media(max-width:700px){header{padding:22px 16px}main{padding:12px 7px}.shared,.branches{padding:13px 8px}summary code{display:block;margin-top:6px}}
    """
    document = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" '
        f'content="width=device-width,initial-scale=1"><title>Grouped weekday activation report</title><style>{css}</style></head><body>'
        '<header><h1>Qwen3.5-0.8B: grouped weekday MLP activations</h1>'
        '<p>Each template shows its shared prefix once. Separate weekday branches begin at the weekday token and include every later token.</p>'
        f'<p>Significance: upper {percentage:g}% within each layer, including all cutoff ties · '
        f'{len(selected_rows):,} selected activation entries.</p></header>'
        f'<main>{"".join(template_sections)}</main></body></html>'
    )
    OUTPUT.write_text(document)
    print(f"Templates: {len(template_sections)}")
    print(f"Weekday branches: {len(artifact['records'])}")
    print(f"Selected entries in source: {len(selected_rows)}")
    print(f"Saved grouped report: {OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
