from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from gb10_load_llm import load_model_to_cuda
from transformers import AutoModelForCausalLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare matched GPT-2 MLP input/output directions with shuffled and nearest pairs."
    )
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--revision", default=None)
    parser.add_argument(
        "--output-root", type=Path, default=Path("outputs/direction_consumptions")
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    model, _ = load_model_to_cuda(
        AutoModelForCausalLM,
        args.model,
        revision=args.revision,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        touch="auto",
        return_touched=True,
    )
    model.eval()
    if not hasattr(model, "transformer") or not hasattr(model.transformer, "h"):
        raise ValueError("This script expects a GPT-2 architecture")

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    neuron_rows = []
    layer_rows = []
    with torch.inference_mode():
        for layer_index, block in enumerate(model.transformer.h):
            # Conv1D weights are [input, output].
            input_directions = block.mlp.c_fc.weight.detach().float()
            output_directions = block.mlp.c_proj.weight.detach().float().T
            input_unit = input_directions / input_directions.norm(dim=0, keepdim=True).clamp_min(1e-12)
            output_unit = output_directions / output_directions.norm(dim=0, keepdim=True).clamp_min(1e-12)

            matched = (input_unit * output_unit).sum(dim=0).cpu()
            permutation = torch.randperm(input_unit.shape[1], generator=generator)
            shuffled = (input_unit[:, permutation.to(input_unit.device)] * output_unit).sum(dim=0).cpu()
            all_cosines = (input_unit.T @ output_unit).abs()
            nearest_values, nearest_indices = all_cosines.max(dim=0)
            nearest = nearest_values.cpu()
            nearest_indices = nearest_indices.cpu()
            for neuron in range(matched.numel()):
                neuron_rows.append(
                    {
                        "layer": layer_index,
                        "neuron": neuron,
                        "matched_cosine": float(matched[neuron]),
                        "matched_absolute_cosine": float(matched[neuron].abs()),
                        "shuffled_cosine": float(shuffled[neuron]),
                        "shuffled_absolute_cosine": float(shuffled[neuron].abs()),
                        "nearest_absolute_cosine": float(nearest[neuron]),
                        "nearest_input_neuron": int(nearest_indices[neuron]),
                        "matched_is_nearest": int(nearest_indices[neuron] == neuron),
                    }
                )
            layer_rows.append(
                {
                    "layer": layer_index,
                    "neurons": matched.numel(),
                    "matched_mean_signed": float(matched.mean()),
                    "matched_positive_fraction": float((matched > 0).float().mean()),
                    "matched_mean_absolute": float(matched.abs().mean()),
                    "matched_median_absolute": float(matched.abs().median()),
                    "matched_p95_absolute": float(torch.quantile(matched.abs(), 0.95)),
                    "shuffled_mean_absolute": float(shuffled.abs().mean()),
                    "nearest_mean_absolute": float(nearest.mean()),
                    "matched_is_nearest_fraction": float(
                        (nearest_indices == torch.arange(matched.numel())).float().mean()
                    ),
                }
            )

    output_dir = args.output_root / args.model.rsplit("/", 1)[-1] / "mlp_matched_input_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "per_neuron.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=neuron_rows[0].keys())
        writer.writeheader()
        writer.writerows(neuron_rows)
    with (output_dir / "per_layer.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=layer_rows[0].keys())
        writer.writeheader()
        writer.writerows(layer_rows)

    layers = np.array([row["layer"] for row in layer_rows])
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ax0.plot(layers, [row["matched_mean_signed"] for row in layer_rows], "o-", label="matched mean signed")
    ax0.axhline(0, color="#667085", linewidth=0.8)
    ax0.set_ylabel("Signed cosine")
    ax0.legend()
    ax0.grid(alpha=0.22)

    ax1.plot(layers, [row["matched_mean_absolute"] for row in layer_rows], "o-", label="matched neuron")
    ax1.plot(layers, [row["shuffled_mean_absolute"] for row in layer_rows], "o-", label="shuffled pairing")
    ax1.plot(layers, [row["nearest_mean_absolute"] for row in layer_rows], "o-", label="nearest any neuron")
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Mean |cosine|")
    ax1.set_xticks(layers)
    ax1.set_ylim(0, 1)
    ax1.legend()
    ax1.grid(alpha=0.22)
    fig.suptitle("GPT-2 MLP input/output direction alignment")
    fig.tight_layout()
    fig.savefig(output_dir / "matched_vs_shuffled_vs_nearest.png", dpi=190)
    plt.close(fig)

    summary = {
        "model": args.model,
        "definition": "cos(c_fc.weight[:,j], c_proj.weight[j,:])",
        "layers": layer_rows,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for row in layer_rows:
        print(
            f"L{row['layer']:2d} matched signed={row['matched_mean_signed']:+.4f}  "
            f"matched |cos|={row['matched_mean_absolute']:.4f}  "
            f"shuffled={row['shuffled_mean_absolute']:.4f}  "
            f"nearest={row['nearest_mean_absolute']:.4f}  "
            f"same-nearest={row['matched_is_nearest_fraction']:.1%}  "
            f"positive={row['matched_positive_fraction']:.1%}"
        )
    print(f"Saved to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
