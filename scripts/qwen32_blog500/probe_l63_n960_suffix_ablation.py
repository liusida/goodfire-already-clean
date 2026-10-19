#!/usr/bin/env python3
"""Remove one leading token at a time and measure L63/N960 on the final dash."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from gb10_load_llm import load_model_to_cuda
from transformers import AutoConfig, AutoTokenizer, Qwen3Model


MODEL_ID = "Qwen/Qwen3-32B"
MODEL_REVISION = "9216db5781bf21249d130ec9da846c4624c16137"
PROMPT = "#NetNeutrality is STILL in danger - Click here to help. DEAL: For $28 - "
LAYER = 63
NEURON = 960
OUTPUT_DIR = Path("outputs/blog500_pile1m/l63_n960_suffix_ablation")


def main() -> None:
    config = AutoConfig.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    full_ids = tokenizer(PROMPT, add_special_tokens=False)["input_ids"]
    pieces = [
        tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
        for token_id in full_ids
    ]
    # Ignore a tokenizer-dependent trailing-space token if present. Every tested
    # suffix must end on the dash token, and the final suffix is exactly "8 -".
    dash_index = max(i for i, piece in enumerate(pieces) if piece == " -")
    full_ids = full_ids[: dash_index + 1]
    pieces = pieces[: dash_index + 1]
    digit_index = max(i for i, piece in enumerate(pieces) if piece == "8")
    starts = list(range(digit_index + 1))
    suffixes = [full_ids[start:] for start in starts]

    max_length = len(full_ids)
    input_ids = torch.full(
        (len(suffixes), max_length), tokenizer.pad_token_id, dtype=torch.long
    )
    attention_mask = torch.zeros_like(input_ids)
    for row, suffix in enumerate(suffixes):
        input_ids[row, : len(suffix)] = torch.tensor(suffix)
        attention_mask[row, : len(suffix)] = 1

    print("Loading pinned Qwen3-32B BF16 checkpoint...")
    model, touched = load_model_to_cuda(
        Qwen3Model,
        MODEL_ID,
        config=config,
        revision=MODEL_REVISION,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        touch="auto",
        return_touched=True,
    )
    model.eval()
    captured: dict[str, torch.Tensor] = {}

    def hook(_module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        captured["activation"] = inputs[0].detach().float().cpu()

    handle = model.layers[LAYER].mlp.down_proj.register_forward_pre_hook(hook)
    with torch.inference_mode():
        model(
            input_ids=input_ids.to(model.device),
            attention_mask=attention_mask.to(model.device),
            use_cache=False,
        )
    handle.remove()

    rows = []
    for row, start in enumerate(starts):
        target_position = len(suffixes[row]) - 1
        activation = float(captured["activation"][row, target_position, NEURON])
        rows.append(
            {
                "removed_token_count": start,
                "removed_token": "" if start == 0 else pieces[start - 1],
                "first_retained_token": pieces[start],
                "retained_token_count": len(suffixes[row]),
                "target_token": pieces[-1],
                "target_activation": activation,
                "suffix": tokenizer.decode(
                    suffixes[row], clean_up_tokenization_spaces=False
                ),
            }
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "results.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    with (OUTPUT_DIR / "results.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)

    fig, ax = plt.subplots(figsize=(11, 5.8), constrained_layout=True)
    x = [row["removed_token_count"] for row in rows]
    y = [row["target_activation"] for row in rows]
    ax.plot(x, y, marker="o", color="#246b86", linewidth=1.8)
    ax.axhline(0, color="#77838d", linewidth=1, linestyle=":")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [repr(row["first_retained_token"]) for row in rows],
        rotation=55,
        ha="right",
        fontsize=8.5,
    )
    ax.set_xlabel("First retained token (progressively remove tokens from the left)")
    ax.set_ylabel("L63/N960 activation on final ' -'")
    ax.set_title("Which left context supports the $28-dash activation?")
    ax.grid(axis="y", alpha=0.22)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.savefig(OUTPUT_DIR / "suffix_ablation.png", dpi=220)
    plt.close(fig)

    print(f"Touched model GiB: {touched / 1024**3:.3f}")
    print("removed\tfirst retained\tactivation\tsuffix")
    for row in rows:
        print(
            f"{row['removed_token_count']:>3}\t"
            f"{row['first_retained_token']!r}\t"
            f"{row['target_activation']:>9.4f}\t{row['suffix']!r}"
        )
    print(f"Wrote {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
