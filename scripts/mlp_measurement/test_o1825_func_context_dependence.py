from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from gb10_load_llm import load_model_to_cuda
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL = "Qwen/Qwen3.5-0.8B-Base"
LAYER = 0
DIRECTION = 1825
TARGET = " func"
PREFIXES = [
    "import Foundation\nprivate",
    "The weather forecast predicts rain tomorrow.",
    "A historian examined letters from ancient Rome.",
    "The chef mixed garlic, tomatoes, and olive oil.",
    "Quantum mechanics describes microscopic systems.",
    "The football team practiced before the match.",
    "She opened the novel and read the first chapter.",
    "Economic growth slowed during the final quarter.",
    "Mountains surrounded the quiet alpine village.",
    "The patient discussed the symptoms with a doctor.",
    "A telescope captured light from a distant galaxy.",
    "The orchestra performed a symphony on Friday.",
    "Researchers collected water samples from the river.",
    "The election results were announced after midnight.",
    "Coffee beans were roasted and packed for shipping.",
    "The museum displayed paintings from the Renaissance.",
    "Children played games in the neighborhood park.",
    "The train arrived at the central station on time.",
    "Photosynthesis converts light energy into chemical energy.",
    "The contract was reviewed by several attorneys.",
    "def greet():\n    print('hello')\n# Python then",
    "function greet() { console.log('hello'); } // JavaScript then",
    "public static void main(String[] args) {} // Java then",
    "fn main() { println!(\"hello\"); } // Rust then",
    "SELECT name FROM users WHERE active = true; -- SQL then",
    "<html><body>Hello</body></html> <!-- markup then -->",
    "1 + 1 = 2. Therefore, by induction,",
    "Once upon a time, in a kingdom far away,",
    "北京是一座历史悠久的城市。",
    "Η επιστήμη εξετάζει τον φυσικό κόσμο.",
    "123 456 789 ! ? ; :",
    "The previous text is deliberately unrelated to programming.",
]


class Captured(Exception):
    pass


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    target_ids = tokenizer.encode(TARGET, add_special_tokens=False)
    if len(target_ids) != 1:
        raise RuntimeError(f"{TARGET!r} is not one token: {target_ids}")
    target_id = target_ids[0]
    model, _ = load_model_to_cuda(
        AutoModelForCausalLM,
        MODEL,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        touch="auto",
        return_touched=True,
    )
    model.eval()
    mixer = model.model.layers[LAYER].linear_attn
    head = DIRECTION // mixer.head_v_dim
    coordinate = DIRECTION % mixer.head_v_dim
    current: dict[str, torch.Tensor | int] = {}

    def norm_pre_hook(_module, inputs):
        sequence_length = int(current["sequence_length"])
        current["core"] = inputs[0].detach().float().reshape(
            sequence_length, -1, mixer.head_v_dim
        ).cpu()
        current["gate"] = inputs[1].detach().float().reshape(
            sequence_length, -1, mixer.head_v_dim
        ).cpu()

    def output_pre_hook(_module, inputs):
        current["coefficient"] = inputs[0].detach().float().cpu()[0]
        raise Captured

    norm_handle = mixer.norm.register_forward_pre_hook(norm_pre_hook)
    output_handle = mixer.out_proj.register_forward_pre_hook(output_pre_hook)
    rows = []
    try:
        for case_id, prefix in enumerate(PREFIXES, 1):
            prompt = prefix.rstrip() + TARGET + " demo() {}"
            encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
            positions = [
                i for i, token_id in enumerate(encoded.input_ids[0].tolist())
                if token_id == target_id
            ]
            if len(positions) != 1:
                raise RuntimeError(f"Case {case_id} has target positions {positions}")
            position = positions[0]
            current.clear()
            current["sequence_length"] = encoded.input_ids.shape[1]
            try:
                with torch.inference_mode():
                    model.model(
                        input_ids=encoded.input_ids.to(model.device),
                        attention_mask=encoded.attention_mask.to(model.device),
                        use_cache=False,
                    )
            except Captured:
                pass
            core = current["core"][position, head]
            gate_z = current["gate"][position, head, coordinate]
            rms = torch.sqrt(core.square().mean() + mixer.norm.variance_epsilon)
            contextual = core[coordinate] / rms * mixer.norm.weight[coordinate].detach().float().cpu()
            silu_gate = F.silu(gate_z)
            coefficient = current["coefficient"][position, DIRECTION]
            rows.append(
                {
                    "case": case_id,
                    "prefix": prefix,
                    "token_position": position,
                    "z": float(gate_z),
                    "silu_z": float(silu_gate),
                    "contextual_readout": float(contextual),
                    "factorized_coefficient": float(contextual * silu_gate),
                    "captured_coefficient": float(coefficient),
                }
            )
    finally:
        norm_handle.remove()
        output_handle.remove()

    z = np.array([row["z"] for row in rows])
    context = np.array([row["contextual_readout"] for row in rows])
    coefficient = np.array([row["captured_coefficient"] for row in rows])
    summary = {
        "model": MODEL,
        "layer": LAYER,
        "direction": DIRECTION,
        "target_token": TARGET,
        "target_token_id": target_id,
        "cases": len(rows),
        "z": {
            "mean": float(z.mean()),
            "std": float(z.std()),
            "min": float(z.min()),
            "max": float(z.max()),
        },
        "contextual_readout": {
            "mean": float(context.mean()),
            "std": float(context.std()),
            "min": float(context.min()),
            "max": float(context.max()),
            "positive_fraction": float((context > 0).mean()),
        },
        "coefficient": {
            "mean": float(coefficient.mean()),
            "std": float(coefficient.std()),
            "min": float(coefficient.min()),
            "max": float(coefficient.max()),
            "positive_fraction": float((coefficient > 0).mean()),
        },
    }
    output_dir = Path(
        "outputs/mlp_measurement/Qwen3.5-0.8B-Base/layer_00/"
        "attention_output_O1825_func_context_test"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print("\nCases sorted by contextual readout:")
    for row in sorted(rows, key=lambda item: item["contextual_readout"]):
        print(
            f"{row['contextual_readout']:+.4f}  a={row['captured_coefficient']:+.4f}  "
            f"{row['prefix'][:72]!r}"
        )
    print(f"\nSaved to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
