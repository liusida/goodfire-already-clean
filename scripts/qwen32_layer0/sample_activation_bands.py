from __future__ import annotations

import argparse
import bisect
import csv
import html
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample token contexts from equal activation-score bands."
    )
    parser.add_argument("scan_dir", type=Path)
    parser.add_argument("--neuron", type=int, required=True)
    parser.add_argument("--centers", default="-7,-6,-5,-4,-3,-2,-1")
    parser.add_argument("--band-width", type=float, default=1.0)
    parser.add_argument("--samples-per-band", type=int, default=8)
    parser.add_argument("--context-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--chunk-rows", type=int, default=1_000_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    centers = [float(value) for value in args.centers.split(",")]
    if args.band_width <= 0 or args.samples_per_band < 1:
        raise ValueError("band width and sample count must be positive")
    summary = json.loads((args.scan_dir / "summary.json").read_text(encoding="utf-8"))
    neurons = [int(value) for value in summary["neurons"]]
    if args.neuron not in neurons:
        raise ValueError(f"Neuron {args.neuron} is not present in this scan")
    column = neurons.index(args.neuron)
    shape = tuple(int(value) for value in summary["scores"]["shape"])
    scores = np.memmap(
        args.scan_dir / summary["scores"]["path"],
        mode="r",
        dtype="<f4",
        shape=shape,
    )

    half_width = args.band_width / 2
    indices_by_center: dict[float, list[np.ndarray]] = {center: [] for center in centers}
    for start in range(0, shape[0], args.chunk_rows):
        values = np.asarray(scores[start : start + args.chunk_rows, column])
        for center in centers:
            local = np.flatnonzero(
                (values >= center - half_width) & (values < center + half_width)
            )
            if local.size:
                indices_by_center[center].append(local + start)

    rng = np.random.default_rng(args.seed)
    selected: list[tuple[float, int]] = []
    band_counts = {}
    for center in centers:
        candidates = (
            np.concatenate(indices_by_center[center])
            if indices_by_center[center]
            else np.empty(0, dtype=np.int64)
        )
        band_counts[center] = int(candidates.size)
        count = min(args.samples_per_band, candidates.size)
        if count:
            chosen = rng.choice(candidates, size=count, replace=False)
            chosen = chosen[np.argsort(np.abs(scores[chosen, column] - center))]
            selected.extend((center, int(index)) for index in chosen)

    documents = [
        json.loads(line)
        for line in (
            args.scan_dir / summary["scores"]["document_index_path"]
        ).open(encoding="utf-8")
    ]
    document_starts = [int(record["score_start"]) for record in documents]
    tokenizer = AutoTokenizer.from_pretrained(
        summary["model_id"], revision=summary["model_revision"]
    )
    dataset = load_dataset(
        summary["dataset_id"],
        split="train",
        revision=summary["dataset_revision"],
    )
    selected_by_document: defaultdict[int, list[tuple[float, int]]] = defaultdict(list)
    for center, global_index in selected:
        document_slot = bisect.bisect_right(document_starts, global_index) - 1
        selected_by_document[document_slot].append((center, global_index))

    records = []
    for document_slot, items in selected_by_document.items():
        document = documents[document_slot]
        dataset_index = int(document["dataset_index"])
        text = dataset[dataset_index]["text"]
        encoded = tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
            truncation=False,
        )
        input_ids = encoded["input_ids"]
        offsets = encoded["offset_mapping"]
        for center, global_index in items:
            token_position = global_index - int(document["score_start"])
            context_start = max(0, token_position - args.context_tokens)
            context_end = min(len(offsets), token_position + args.context_tokens + 1)
            target_start, target_end = offsets[token_position]
            left_start = offsets[context_start][0]
            right_end = offsets[context_end - 1][1]
            token_id = int(input_ids[token_position])
            records.append(
                {
                    "band_center": center,
                    "band_lower": center - half_width,
                    "band_upper": center + half_width,
                    "band_population": band_counts[center],
                    "activation": float(scores[global_index, column]),
                    "global_score_index": global_index,
                    "dataset_index": dataset_index,
                    "pile_set_name": document["pile_set_name"],
                    "document_token_index": token_position,
                    "token_id": token_id,
                    "token_piece": tokenizer.convert_ids_to_tokens(token_id),
                    "target_text": text[target_start:target_end],
                    "left_context": text[left_start:target_start],
                    "right_context": text[target_end:right_end],
                }
            )
    records.sort(key=lambda row: (-row["band_center"], row["activation"]))

    stem = f"activation_bands_N{args.neuron}"
    json_path = args.scan_dir / f"{stem}.json"
    csv_path = args.scan_dir / f"{stem}.csv"
    html_path = args.scan_dir / f"{stem}.html"
    json_path.write_text(
        json.dumps(
            {
                "layer": summary["layer"],
                "neuron": args.neuron,
                "band_width": args.band_width,
                "samples_per_band": args.samples_per_band,
                "seed": args.seed,
                "band_populations": {str(key): value for key, value in band_counts.items()},
                "records": records,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    sections = []
    for center in centers:
        band_records = [record for record in records if record["band_center"] == center]
        cards = []
        for record in band_records:
            context = (
                html.escape(record["left_context"])
                + "<mark>"
                + html.escape(record["target_text"])
                + "</mark>"
                + html.escape(record["right_context"])
            )
            cards.append(
                f"""<article><header><strong>{record['activation']:.7g}</strong>
                <code>token {record['token_id']} · {html.escape(repr(record['token_piece']))}</code>
                <span>{html.escape(record['pile_set_name'] or 'Unknown')} · doc {record['dataset_index']} · position {record['document_token_index']}</span></header>
                <pre>{context}</pre></article>"""
            )
        sections.append(
            f"""<section><h2>Centered at {center:g} <small>[{center-half_width:g}, {center+half_width:g}) · {band_counts[center]:,} total positions · {len(band_records)} sampled</small></h2>{''.join(cards)}</section>"""
        )
    page = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>L{summary['layer']}/N{args.neuron} activation bands</title><style>
body{{margin:0;background:#f3f6f7;color:#182027;font:14px/1.45 system-ui,sans-serif}}main{{max-width:1300px;margin:auto;padding:26px}}h1{{margin:0 0 4px}}.intro{{color:#63717c;margin-bottom:24px}}section{{margin:0 0 28px}}h2{{position:sticky;top:0;background:rgba(243,246,247,.95);padding:9px 0;margin:0;z-index:2}}h2 small{{color:#687680;font-size:13px;font-weight:400;margin-left:10px}}article{{background:white;border:1px solid #dbe2e6;border-radius:9px;margin:9px 0;overflow:hidden}}article header{{display:flex;gap:13px;align-items:baseline;padding:8px 12px;border-bottom:1px solid #e0e6e9}}article header strong{{font:700 14px ui-monospace,monospace}}article header code{{color:#34444f}}article header span{{margin-left:auto;color:#6b7881;font-size:12px}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;margin:0;padding:12px;font:12.5px/1.55 ui-monospace,monospace}}mark{{background:#ffd0bd;border-radius:3px;padding:1px 2px}}@media(max-width:700px){{article header{{display:block}}article header span{{display:block;margin:3px 0 0}}}}
</style></head><body><main><h1>Layer {summary['layer']} · Neuron {args.neuron}</h1><div class="intro">Equal-width activation bands · width {args.band_width:g} · up to {args.samples_per_band} reproducible samples per band · seed {args.seed}</div>{''.join(sections)}</main></body></html>"""
    html_path.write_text(page, encoding="utf-8")
    for center in centers:
        print(
            f"center {center:>5g}: {band_counts[center]:>6,} positions, "
            f"sampled {min(args.samples_per_band, band_counts[center])}"
        )
    print(f"Saved {html_path.resolve()}")
    print(f"Saved {csv_path.resolve()}")
    print(f"Saved {json_path.resolve()}")


if __name__ == "__main__":
    main()
