from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from gb10_load_llm import load_model_to_cuda
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Project low- and high-kurtosis MLP down-projection directions "
            "directly through the unembedding matrix."
        )
    )
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B-Base")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--dataset", choices=("en", "cn", "both"), default="en")
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--top-k-tokens", type=int, default=10)
    parser.add_argument("--bins", type=int, default=180)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/mlp_measurement"),
    )
    return parser.parse_args()


def token_text(tokenizer: AutoTokenizer, token_id: int) -> str:
    text = tokenizer.decode(
        [token_id],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    return text.replace("\n", "\\n").replace("\r", "\\r") or "∅"


def ranked_tokens(
    values: torch.Tensor,
    tokenizer: AutoTokenizer,
    count: int,
) -> dict[str, list[dict[str, object]]]:
    count = min(count, int(values.numel()))

    def records(*, largest: bool) -> list[dict[str, object]]:
        scores, ids = torch.topk(values, count, largest=largest, sorted=True)
        return [
            {
                "rank": rank,
                "token_id": int(token_id),
                "token": token_text(tokenizer, int(token_id)),
                "projection": float(score),
            }
            for rank, (score, token_id) in enumerate(zip(scores, ids), start=1)
        ]

    return {"negative": records(largest=False), "positive": records(largest=True)}


def render_histogram(
    values: torch.Tensor,
    output: Path,
    *,
    neuron: int,
    input_kurtosis: float,
    projection_kurtosis: float,
    x_limit: float,
    bins: int,
) -> None:
    array = values.numpy()
    fig, ax = plt.subplots(figsize=(2.55, 2.55), constrained_layout=True)
    ax.hist(
        array,
        bins=bins,
        range=(-x_limit, x_limit),
        color="#4c78a8",
        linewidth=0,
    )
    ax.set_yscale("log")
    ax.axvline(0.0, color="#d1495b", linewidth=0.8, alpha=0.8)
    ax.set_xlim(-x_limit, x_limit)
    ax.set_title(
        f"N{neuron}\ninput κ={input_kurtosis:.4g} · projection κ={projection_kurtosis:.4g}",
        fontsize=8.5,
        pad=5,
    )
    ax.set_xlabel("Vocabulary projection", fontsize=8)
    ax.set_ylabel("Tokens", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(axis="y", alpha=0.18, linewidth=0.6)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, facecolor="white")
    plt.close(fig)


def token_list(rows: list[dict[str, object]], css_class: str) -> str:
    return "".join(
        f'<li><code>{html.escape(str(row["token"]))}</code>'
        f'<span class="{css_class}">{float(row["projection"]):+.5g}</span></li>'
        for row in rows
    )


def main() -> None:
    args = parse_args()
    if args.layer < 0 or args.count <= 0 or args.top_k_tokens <= 0 or args.bins <= 0:
        raise ValueError("Layer must be nonnegative and counts/bins must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    model_name = args.model.rstrip("/").rsplit("/", 1)[-1]
    layer_dir = args.output_root / model_name / f"layer_{args.layer:02d}"
    metrics_dir = layer_dir / "neuron_metrics" / args.dataset
    extremes_path = metrics_dir / "kurtosis_extremes.json"
    if not extremes_path.exists():
        raise FileNotFoundError(
            f"Missing {extremes_path}; run measure_neuron_metrics.py first"
        )
    extremes = json.loads(extremes_path.read_text(encoding="utf-8"))
    selections: list[dict[str, object]] = []
    for tail in ("lowest", "highest"):
        for row in extremes[tail][: args.count]:
            selections.append(
                {
                    "tail": tail,
                    "rank": int(row["rank"]),
                    "neuron": int(row["neuron"]),
                    "input_activation_kurtosis": float(row["kurtosis"]),
                }
            )
    neurons = [int(row["neuron"]) for row in selections]
    if len(neurons) != len(set(neurons)):
        raise ValueError("A neuron appears in both kurtosis tails")

    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model, touched = load_model_to_cuda(
        AutoModelForCausalLM,
        args.model,
        revision=args.revision,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        touch="auto",
        return_touched=True,
    )
    model.eval()
    if args.layer >= len(model.model.layers):
        raise ValueError(
            f"Layer {args.layer} is outside 0..{len(model.model.layers) - 1}"
        )

    down_weight = model.model.layers[args.layer].mlp.down_proj.weight
    unembedding_weight = model.get_output_embeddings().weight
    if down_weight.shape[0] != unembedding_weight.shape[1]:
        raise RuntimeError(
            f"Incompatible shapes: down_proj={tuple(down_weight.shape)}, "
            f"unembedding={tuple(unembedding_weight.shape)}"
        )
    invalid = [neuron for neuron in neurons if neuron >= down_weight.shape[1]]
    if invalid:
        raise ValueError(f"Neuron indices outside down_proj: {invalid}")

    neuron_index = torch.tensor(neurons, device=down_weight.device)
    directions = down_weight.index_select(1, neuron_index)
    with torch.inference_mode():
        projections = torch.matmul(unembedding_weight, directions)
    projections = projections.transpose(0, 1).detach().float().cpu()
    directions = directions.transpose(0, 1).detach().float().cpu()

    output_dir = layer_dir / "down_projection_kurtosis_extremes" / args.dataset
    assets_dir = output_dir / "histograms"
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "neurons": torch.tensor(neurons, dtype=torch.long),
            "down_directions": directions,
            "vocabulary_projections": projections,
        },
        output_dir / "projections.pt",
    )

    x_limit = float(projections.abs().max())
    cards: list[str] = []
    result_rows: list[dict[str, object]] = []
    for index, selection in enumerate(selections):
        neuron = int(selection["neuron"])
        values = projections[index]
        centered = values.double() - values.double().mean()
        variance = centered.square().mean()
        projection_kurtosis = float(centered.pow(4).mean() / variance.square())
        image_path = assets_dir / f"neuron_{neuron:04d}.png"
        render_histogram(
            values,
            image_path,
            neuron=neuron,
            input_kurtosis=float(selection["input_activation_kurtosis"]),
            projection_kurtosis=projection_kurtosis,
            x_limit=x_limit,
            bins=args.bins,
        )
        tokens = ranked_tokens(values, tokenizer, args.top_k_tokens)
        row = {
            **selection,
            "vocabulary_projection_kurtosis": projection_kurtosis,
            "direction_l2_norm": float(directions[index].norm()),
            "projection_minimum": float(values.min()),
            "projection_maximum": float(values.max()),
            "projection_mean": float(values.mean()),
            "projection_std": float(values.std(unbiased=False)),
            "tokens": tokens,
        }
        result_rows.append(row)
        cards.append(
            f"""
<article class="card {selection['tail']}">
  <header><b>L{args.layer}:N{neuron}</b><span>{selection['tail']} input κ · rank {selection['rank']} · projection κ={projection_kurtosis:.4g}</span></header>
  <img src="histograms/{html.escape(image_path.name)}" alt="Vocabulary projection histogram for N{neuron}">
  <div class="tokens">
    <section><h3>Most negative</h3><ol>{token_list(tokens['negative'], 'negative-score')}</ol></section>
    <section><h3>Most positive</h3><ol>{token_list(tokens['positive'], 'positive-score')}</ol></section>
  </div>
</article>"""
        )

    metadata = {
        "model_id": args.model,
        "model_revision": args.revision,
        "layer": args.layer,
        "dataset": args.dataset,
        "selection_source": str(extremes_path.resolve()),
        "projection_definition": "lm_head.weight @ down_proj.weight[:, neuron]",
        "final_norm_applied": False,
        "shared_histogram_x_limit": x_limit,
        "vocabulary_size": int(unembedding_weight.shape[0]),
        "hidden_size": int(unembedding_weight.shape[1]),
        "cpu_pages_touched_gib": touched / 1024**3,
        "neurons": result_rows,
    }
    (output_dir / "results.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(model_name)} L{args.layer} down-direction vocabulary projections</title>
<style>
:root{{font-family:Inter,ui-sans-serif,system-ui,sans-serif;color:#1d2733}}body{{margin:0;background:#f5f7fa}}main{{max-width:1500px;margin:auto;padding:28px}}h1{{font-size:25px;margin:0 0 5px}}.subtitle{{color:#687486;margin:0 0 22px}}.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:14px}}.card{{background:#fff;border:1px solid #dce2e9;border-radius:12px;overflow:hidden;box-shadow:0 1px 2px #1825360a}}.card.lowest{{border-top:4px solid #4c78a8}}.card.highest{{border-top:4px solid #e07b39}}header{{display:flex;justify-content:space-between;padding:9px 12px;background:#f8fafc}}header span{{color:#687486;font-size:12px}}img{{display:block;width:100%;aspect-ratio:1;object-fit:contain}}.tokens{{display:grid;grid-template-columns:1fr 1fr;gap:8px;padding:0 10px 12px}}h3{{font-size:11px;color:#687486;margin:4px 0}}ol{{list-style:none;margin:0;padding:0}}li{{display:flex;justify-content:space-between;gap:5px;padding:2px 3px;border-bottom:1px solid #edf0f3;font-size:10px}}code{{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}li span{{font-variant-numeric:tabular-nums}}.negative-score{{color:#3465a4}}.positive-score{{color:#b64b2a}}
</style></head><body><main>
<h1>{html.escape(model_name)} · Layer {args.layer} down directions</h1>
<p class="subtitle">Direct projection: <code>lm_head.weight @ down_proj.weight[:, neuron]</code> · no final norm · {len(selections)} kurtosis-extreme neurons · shared x-axis ±{x_limit:.5g}</p>
<section class="grid">{''.join(cards)}</section>
</main></body></html>"""
    (output_dir / "report.html").write_text(document, encoding="utf-8")
    print(f"Projected {len(selections)} directions over {unembedding_weight.shape[0]:,} tokens")
    print(f"Saved: {(output_dir / 'report.html').resolve()}")


if __name__ == "__main__":
    main()
