from __future__ import annotations

import csv
import html
import json
from collections import Counter
from pathlib import Path

import torch


ACTIVATIONS = Path("outputs/weekdays/qwen35_08b_weekday_activations.pt")
SIGNIFICANT = Path("outputs/weekdays/qwen35_08b_weekday_layerwise_top_0.002pct_activations.csv")
SUMMARY = Path("outputs/weekdays/qwen35_08b_weekday_layerwise_top_0.002pct_activations.json")
OUTPUT = Path("outputs/weekdays/qwen35_08b_weekday_feature_matrix_report.html")
DATA_OUTPUT = Path("outputs/weekdays/qwen35_08b_weekday_feature_matrix_values.csv")


def e(value: object) -> str:
    return html.escape(str(value), quote=True)


def main() -> None:
    artifact = torch.load(ACTIVATIONS, map_location="cpu", weights_only=False)
    summary = json.loads(SUMMARY.read_text())
    with SIGNIFICANT.open(newline="") as handle:
        selected = list(csv.DictReader(handle))

    weekday_selected = [row for row in selected if row["is_weekday_token"] == "True"]
    features = sorted(
        {(int(row["layer"]), int(row["neuron"])) for row in weekday_selected}
    )
    discovery_counts = Counter(
        (int(row["layer"]), int(row["neuron"])) for row in weekday_selected
    )
    cutoffs = {
        int(item["layer"]): float(item["top_fraction_cutoff"])
        for item in summary["layer_stats"]
    }
    layer_stats = {int(item["layer"]): item for item in summary["layer_stats"]}
    records_by_key = {
        (record["template_index"], record["weekday"]): record
        for record in artifact["records"]
    }

    data_rows: list[dict[str, object]] = []
    for layer, neuron in features:
        cutoff = cutoffs[layer]
        mean = float(layer_stats[layer]["mean"])
        std = float(layer_stats[layer]["std"])
        for template_index, template in enumerate(artifact["templates"]):
            for weekday in artifact["weekdays"]:
                record = records_by_key[(template_index, weekday)]
                position = record["weekday_position"]
                value = float(record["activations"][layer, position, neuron].item())
                data_rows.append(
                    {
                        "layer": layer,
                        "neuron": neuron,
                        "template_index": template_index,
                        "template": template,
                        "weekday": weekday,
                        "token_position": position,
                        "value": value,
                        "layer_z_score": (value - mean) / std,
                        "layer_cutoff": cutoff,
                        "is_significant": value >= cutoff,
                    }
                )

    DATA_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with DATA_OUTPUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(data_rows[0]))
        writer.writeheader()
        writer.writerows(data_rows)

    rows_by_feature: dict[tuple[int, int], list[dict[str, object]]] = {}
    for feature in features:
        rows_by_feature[feature] = [
            row for row in data_rows if (row["layer"], row["neuron"]) == feature
        ]

    overview_rows = []
    feature_sections = []
    for feature_index, (layer, neuron) in enumerate(features):
        feature_rows = rows_by_feature[(layer, neuron)]
        values = [float(row["value"]) for row in feature_rows]
        significant_count = sum(bool(row["is_significant"]) for row in feature_rows)
        feature_min, feature_max = min(values), max(values)
        span = feature_max - feature_min
        feature_id = f"l{layer}-n{neuron}"
        overview_rows.append(
            f'<tr><td><a href="#{feature_id}">L{layer}:N{neuron}</a></td>'
            f'<td>{discovery_counts[(layer, neuron)]} / 70</td>'
            f'<td>{cutoffs[layer]:.6g}</td><td>{feature_min:.6g}</td><td>{feature_max:.6g}</td></tr>'
        )

        matrix_rows = []
        for template_index, template in enumerate(artifact["templates"]):
            cells = []
            for weekday in artifact["weekdays"]:
                row = next(
                    item
                    for item in feature_rows
                    if item["template_index"] == template_index and item["weekday"] == weekday
                )
                value = float(row["value"])
                normalized = 0.5 if span == 0 else (value - feature_min) / span
                significant_class = " significant" if row["is_significant"] else ""
                cells.append(
                    f'<td class="value{significant_class}" style="--heat:{normalized:.4f}" '
                    f'title="z={float(row["layer_z_score"]):.3f}; layer cutoff={cutoffs[layer]:.6g}">'
                    f'{value:.6g}</td>'
                )
            rendered_prompt = template.replace("{weekday}", "<weekday>")
            matrix_rows.append(
                f'<tr><th><span>T{template_index + 1}</span><small>{e(rendered_prompt)}</small></th>'
                f'{"".join(cells)}</tr>'
            )

        feature_sections.append(
            f'<section class="feature" id="{feature_id}"><h2>L{layer}:N{neuron}'
            f'<span>{significant_count}/70 significant · cutoff {cutoffs[layer]:.6g}</span></h2>'
            '<div class="scroll"><table class="matrix"><thead><tr><th>Template</th>'
            + "".join(f'<th>{e(day)}</th>' for day in artifact["weekdays"])
            + f'</tr></thead><tbody>{"".join(matrix_rows)}</tbody></table></div></section>'
        )

    css = """
    *{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:#f3f5f7;color:#17212b;font-family:Inter,system-ui,sans-serif}
    header{background:#17212b;color:#fff;padding:28px 32px}header h1{margin:0 0 8px;font-size:25px}header p{margin:5px 0;color:#cbd4dd;line-height:1.45;max-width:1050px}
    main{padding:20px 24px 60px;max-width:1450px;margin:auto}.overview,.feature{background:#fff;border:1px solid #d7dde4;border-radius:9px;margin-bottom:18px;box-shadow:0 1px 3px #0000000c;overflow:hidden}
    h2{font-size:16px;margin:0;padding:13px 16px;background:#f8fafc;border-bottom:1px solid #dfe4e9}h2 span{font-size:11px;font-weight:500;color:#687583;margin-left:10px}
    .scroll{overflow-x:auto}table{border-collapse:separate;border-spacing:0;width:100%;font-size:12px}th,td{padding:8px 10px;border-right:1px solid #e3e7eb;border-bottom:1px solid #e3e7eb;text-align:center;font-variant-numeric:tabular-nums}
    thead th{background:#edf2f6;position:sticky;top:0;z-index:2}.overview table td:first-child{font-weight:700}.overview a{color:#245d8c;text-decoration:none}.overview a:hover{text-decoration:underline}
    .matrix tbody th{position:sticky;left:0;z-index:1;background:#f8fafc;min-width:310px;max-width:420px;text-align:left}.matrix tbody th span{display:inline-block;width:28px;color:#758290}.matrix tbody th small{font-weight:400;color:#465564}
    td.value{min-width:105px;color:#57210d;background:rgba(230,83,31,calc(.07 + var(--heat)*.70))}.value.significant{font-weight:750;color:#491704;box-shadow:inset 0 0 0 2px #b94319}
    .legend{display:flex;gap:18px;align-items:center;margin-top:12px;font-size:12px}.swatch{display:inline-block;width:16px;height:12px;margin-right:5px;vertical-align:-2px;background:#f39a78;border:2px solid #dc6b3f}
    @media(max-width:700px){header{padding:22px 16px}main{padding:12px 7px}.matrix tbody th{min-width:220px}}
    """
    document = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" '
        f'content="width=device-width,initial-scale=1"><title>Weekday feature matrices</title><style>{css}</style></head><body>'
        '<header><h1>Weekday-token significant feature matrices</h1>'
        f'<p>Union of {len(features)} layer:neuron features that reached the layer-relative top 0.002% '
        'at a weekday token in at least one of the 70 prompts. Every matrix contains the raw activation '
        'for that feature at every template × weekday combination, including nonsignificant values.</p>'
        '<div class="legend"><span><i class="swatch"></i>At or above that layer’s significance cutoff</span>'
        '<span>Hover any value for its layer z-score and cutoff.</span></div></header>'
        '<main><section class="overview"><h2>Recorded features</h2><div class="scroll"><table><thead><tr>'
        '<th>Feature</th><th>Significant prompts</th><th>Layer cutoff</th><th>Minimum</th><th>Maximum</th>'
        f'</tr></thead><tbody>{"".join(overview_rows)}</tbody></table></div></section>'
        f'{"".join(feature_sections)}</main></body></html>'
    )
    OUTPUT.write_text(document)
    print(f"Features: {len(features)}")
    print(f"Matrix values: {len(data_rows)}")
    print(f"Saved values: {DATA_OUTPUT.resolve()}")
    print(f"Saved report: {OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
