from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a compact HTML gallery from individual-neuron distribution "
            "artifacts produced by measure_mlp_distributions.py."
        )
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3.5-9B-Base",
        help="Model ID or output-directory name.",
    )
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--dataset", choices=("en", "cn"), default="en")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/mlp_measurement"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="HTML output path; defaults inside the selected layer directory.",
    )
    return parser.parse_args()


def discover_neurons(
    model_dir: Path,
    layer_dir: Path,
    *,
    layer: int,
    dataset: str,
) -> tuple[list[int], str]:
    metadata_path = model_dir / f"run_metadata_{dataset}.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata_layers = [int(value) for value in metadata.get("layers", [])]
        metadata_neurons = [int(value) for value in metadata.get("neurons", [])]
        if (
            layer in metadata_layers
            and metadata.get("neuron_output_mode") == "individual"
            and metadata_neurons
        ):
            return sorted(set(metadata_neurons)), metadata_path.name

    neurons = []
    for details_path in layer_dir.glob(f"neuron_*/{dataset}/details.json"):
        try:
            details = json.loads(details_path.read_text(encoding="utf-8"))
            neuron_values = details.get("neurons", [])
            if len(neuron_values) == 1:
                neurons.append(int(neuron_values[0]))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
    if not neurons:
        raise FileNotFoundError(
            f"No individual-neuron results found beneath {layer_dir}"
        )
    return sorted(set(neurons)), "layer-directory discovery"


def render_square_histogram(
    histogram_path: Path,
    output_path: Path,
    *,
    neuron: int,
) -> None:
    payload = torch.load(histogram_path, map_location="cpu", weights_only=True)
    edges = payload["raw_edges"].double().numpy()
    counts = payload["raw_counts"].double().numpy()
    centers = (edges[:-1] + edges[1:]) / 2
    widths = np.diff(edges)

    fig, ax = plt.subplots(figsize=(2.4, 2.4), constrained_layout=True)
    ax.bar(centers, counts, width=widths, color="#4c78a8", linewidth=0)
    ax.set_yscale("log")
    ax.axvline(0.0, color="#d1495b", linewidth=0.8, alpha=0.8)
    ax.set_title(f"N{neuron}", fontsize=11, pad=5)
    ax.set_xlabel("Activation", fontsize=8)
    ax.set_ylabel("Count", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(axis="y", alpha=0.18, linewidth=0.6)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, facecolor="white")
    plt.close(fig)


def fmt(value: object, *, signed: bool = False) -> str:
    if value is None:
        return "—"
    number = float(value)
    return f"{number:+.4g}" if signed else f"{number:.4g}"


def main() -> None:
    args = parse_args()
    if args.layer < 0:
        raise ValueError("--layer must be nonnegative")

    model_name = args.model.rstrip("/").rsplit("/", 1)[-1]
    model_dir = args.output_root / model_name
    layer_dir = model_dir / f"layer_{args.layer:02d}"
    neurons, discovery_source = discover_neurons(
        model_dir,
        layer_dir,
        layer=args.layer,
        dataset=args.dataset,
    )
    output = args.output or (
        layer_dir / f"neuron_histogram_gallery_{args.dataset}.html"
    )
    assets_dir = output.parent / f"{output.stem}_assets"

    records: list[dict[str, object]] = []
    missing: list[int] = []
    dataset_label = args.dataset
    sampled_tokens: int | None = None
    for neuron in neurons:
        result_dir = layer_dir / f"neuron_{neuron:04d}" / args.dataset
        details_path = result_dir / "details.json"
        histogram_path = result_dir / "histograms.pt"
        if not details_path.exists() or not histogram_path.exists():
            missing.append(neuron)
            continue
        details = json.loads(details_path.read_text(encoding="utf-8"))
        dataset_label = str(details.get("dataset_label", args.dataset))
        sampled_tokens = int(details.get("sampled_tokens", 0)) or sampled_tokens
        image_path = assets_dir / f"neuron_{neuron:04d}.png"
        render_square_histogram(
            histogram_path,
            image_path,
            neuron=neuron,
        )
        relative_image = image_path.relative_to(output.parent).as_posix()
        relative_report = (result_dir / "report.md").relative_to(output.parent).as_posix()
        records.append(
            {
                "neuron": neuron,
                "details": details,
                "relative_image": relative_image,
                "relative_report": relative_report,
            }
        )

    if not records:
        raise FileNotFoundError(
            "The run has not produced any complete neuron histogram artifacts yet."
        )

    skew_scale = max(
        abs(float(record["details"].get("skewness") or 0.0))
        for record in records
    ) or 1.0
    kurtosis_scale = max(
        float(record["details"].get("kurtosis") or 0.0)
        for record in records
    ) or 1.0
    log_cosh_scale = max(
        float(record["details"].get("ica_log_cosh_score") or 0.0)
        for record in records
    ) or 1.0

    cards: list[str] = []
    for record in records:
        neuron = int(record["neuron"])
        details = record["details"]
        skewness = float(details.get("skewness") or 0.0)
        kurtosis = max(0.0, float(details.get("kurtosis") or 0.0))
        log_cosh = max(0.0, float(details.get("ica_log_cosh_score") or 0.0))
        skew_width = 50.0 * min(abs(skewness) / skew_scale, 1.0)
        skew_left = 50.0 if skewness >= 0 else 50.0 - skew_width
        kurtosis_width = 100.0 * min(kurtosis / kurtosis_scale, 1.0)
        log_cosh_width = 100.0 * min(log_cosh / log_cosh_scale, 1.0)
        cards.append(
            f"""
<article class="card">
  <a class="plot-link" href="{html.escape(str(record['relative_report']))}">
    <img src="{html.escape(str(record['relative_image']))}" alt="Histogram for L{args.layer}:N{neuron}">
  </a>
  <div class="metrics">
    <span class="metric skewness bipolar"><i style="left:{skew_left:.2f}%;width:{skew_width:.2f}%"></i><em>skewness</em><b>{fmt(details.get('skewness'), signed=True)}</b></span>
    <span class="metric kurtosis"><i style="width:{kurtosis_width:.2f}%"></i><em>kurtosis</em><b>{fmt(details.get('kurtosis'))}</b></span>
    <span class="metric log-cosh"><i style="width:{log_cosh_width:.2f}%"></i><em>log-cosh</em><b>{fmt(details.get('ica_log_cosh_score'))}</b></span>
  </div>
</article>"""
        )
    missing_note = (
        f" · skipped incomplete neurons: {', '.join(f'N{n}' for n in missing)}"
        if missing
        else ""
    )
    token_note = f"{sampled_tokens:,} sampled tokens" if sampled_tokens else "sampled corpus"
    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(model_name)} · L{args.layer} neuron histograms</title>
<style>
:root {{ color-scheme: light; font-family: Inter, ui-sans-serif, system-ui, sans-serif; }}
body {{ margin: 0; color: #17202a; background: #f5f7fa; }}
main {{ max-width: 1500px; margin: 0 auto; padding: 28px; }}
h1 {{ margin: 0 0 5px; font-size: 25px; letter-spacing: -.02em; }}
.subtitle {{ margin: 0 0 22px; color: #687486; font-size: 14px; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(230px, 1fr)); gap: 13px; }}
.card {{ min-width: 0; padding: 10px; border: 1px solid #dce2e9; border-radius: 12px; background: white; box-shadow: 0 1px 2px rgba(20,30,45,.04); }}
.plot-link {{ display: block; aspect-ratio: 1 / 1; }}
.plot-link img {{ display: block; width: 100%; height: 100%; object-fit: contain; }}
.metrics {{ display: grid; gap: 5px; margin-top: 7px; }}
.metric {{ position: relative; isolation: isolate; overflow: hidden; display: grid; grid-template-columns: 1fr auto; align-items: center; gap: 6px; padding: 5px 7px; border: 1px solid; border-radius: 7px; font-size: 11px; }}
.metric i {{ position: absolute; z-index: -1; inset-block: 0; left: 0; opacity: .27; }}
.metric em {{ font-style: normal; }}
.metric b {{ font-variant-numeric: tabular-nums; }}
.metric em,.metric b {{ position: relative; z-index: 1; }}
.metric.bipolar::after {{ content: ""; position: absolute; z-index: 0; left: 50%; top: 0; bottom: 0; width: 1px; background: currentColor; opacity: .28; }}
.skewness {{ color: #5c3fa3; background: #f2effb; border-color: #d9d0f2; }}
.skewness i {{ background: #7655c5; }}
.kurtosis {{ color: #9a4f13; background: #fff3e7; border-color: #f2d2b5; }}
.kurtosis i {{ background: #df7c28; }}
.log-cosh {{ color: #08706a; background: #eaf8f6; border-color: #bee7e2; }}
.log-cosh i {{ background: #15958b; }}
@media (max-width: 520px) {{ main {{ padding: 16px; }} .grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; }} }}
</style>
</head>
<body><main>
<h1>{html.escape(model_name)} · Layer {args.layer}</h1>
<p class="subtitle">{len(cards)} individual post-SwiGLU neuron distributions · {html.escape(dataset_label)} · {token_note} · source: {html.escape(discovery_source)}{html.escape(missing_note)}</p>
<section class="grid">{''.join(cards)}</section>
</main></body>
</html>
"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding="utf-8")
    print(f"Rendered {len(cards)} neuron histograms")
    print(f"Saved: {output.resolve()}")
    if missing:
        print(f"Skipped {len(missing)} incomplete neurons: {missing}")


if __name__ == "__main__":
    main()
