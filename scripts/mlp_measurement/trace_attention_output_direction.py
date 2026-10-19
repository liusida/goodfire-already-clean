from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

import torch
from gb10_load_llm import load_model_to_cuda
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_PROMPT = '''import Foundation

private func greet() {
    print("Hello, world!")
}

greet()'''


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Trace one attention-output direction through a single transformer block."
    )
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B-Base")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--direction", type=int, default=1825)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/mlp_measurement"),
    )
    return parser.parse_args()


def cpu_float(value: torch.Tensor) -> torch.Tensor:
    return value.detach().float().cpu()[0]


def token_label(tokenizer: Any, token_id: int) -> str:
    piece = tokenizer.decode(
        [token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    return piece.replace("\n", "\\n").replace("\t", "\\t")


def cosine_rows(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    numerator = (left * right).sum(dim=-1)
    denominator = left.norm(dim=-1) * right.norm(dim=-1)
    return numerator / denominator.clamp_min(1e-12)


def render_report(
    output_path: Path,
    metadata: dict[str, Any],
    tokens: list[str],
    token_ids: list[int],
    tensors: dict[str, torch.Tensor],
    diagnostics: dict[str, Any],
) -> None:
    coeff = tensors["attention_out_proj_input"][:, metadata["direction"]]
    z_score = tensors.get("direction_z_score")
    isolated = tensors["direction_contribution"]
    attention = tensors["attention_output"]
    post_attn = tensors["post_attention_residual"]
    mlp = tensors["mlp_output"]
    final = tensors["layer_output"]
    max_abs = max(float(coeff.abs().max()), 1e-12)

    rows = []
    for position, (token, token_id) in enumerate(zip(tokens, token_ids)):
        value = float(coeff[position])
        strength = min(abs(value) / max_abs, 1.0)
        if value >= 0:
            color = f"rgba(222, 126, 32, {0.12 + 0.70 * strength:.3f})"
        else:
            color = f"rgba(43, 103, 178, {0.12 + 0.70 * strength:.3f})"
        z_cell = f"{float(z_score[position]):+.2f}σ" if z_score is not None else "—"
        rows.append(
            "<tr>"
            f"<td>{position}</td><td><code>{html.escape(token)}</code></td><td>{token_id}</td>"
            f"<td class='number' style='background:{color}'>{value:+.6f}</td>"
            f"<td class='number'>{z_cell}</td>"
            f"<td class='number'>{float(isolated[position].norm()):.6f}</td>"
            f"<td class='number'>{float(attention[position].norm()):.6f}</td>"
            f"<td class='number'>{float(post_attn[position].norm()):.6f}</td>"
            f"<td class='number'>{float(mlp[position].norm()):.6f}</td>"
            f"<td class='number'>{float(final[position].norm()):.6f}</td>"
            "</tr>"
        )

    stages = [
        ("embedding", "Token embedding / Layer-0 residual input", "residual"),
        ("input_layernorm", "RMSNorm before token mixer", "residual"),
        ("linear_attn_in_proj_qkv", "Projected Q/K/V mixture", "mixer"),
        ("linear_attn_in_proj_z", "Mixer output gate z", "mixer"),
        ("linear_attn_in_proj_b", "Mixer update gate b", "mixer"),
        ("linear_attn_in_proj_a", "Mixer decay parameter a", "mixer"),
        ("attention_out_proj_input", "aₜ: vector entering W_O", "mixer"),
        (
            "direction_contribution",
            f"aₜ,{metadata['direction']} W_O[:,{metadata['direction']}]",
            "residual",
        ),
        ("attention_output", "W_O aₜ + bias", "residual"),
        ("post_attention_residual", "Embedding + attention output", "residual"),
        ("post_attention_layernorm", "RMSNorm before MLP", "residual"),
        ("mlp_gate_projection", "MLP gate projection", "mlp"),
        ("mlp_up_projection", "MLP up projection", "mlp"),
        ("mlp_post_swiglu", "SiLU(gate) ⊙ up; input to W_down", "mlp"),
        ("mlp_output", "W_down · post-SwiGLU", "residual"),
        ("layer_output", "Post-MLP Layer-0 residual", "residual"),
    ]
    stage_rows = "".join(
        f"<tr><td><code>{name}</code></td><td>{html.escape(label)}</td>"
        f"<td>{kind}</td><td>{' × '.join(map(str, tensors[name].shape))}</td></tr>"
        for name, label, kind in stages
    )

    prompt = html.escape(metadata["prompt"])
    doc = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>L{metadata['layer']}:O{metadata['direction']} trace</title>
<style>
body {{ font-family: Inter, system-ui, sans-serif; color:#18212b; margin:32px auto; max-width:1200px; padding:0 18px; }}
h1 {{ margin-bottom:4px; }} .sub {{ color:#657386; margin-top:0; }}
pre {{ background:#f5f7fa; border:1px solid #dce3eb; border-radius:8px; padding:14px; white-space:pre-wrap; }}
table {{ border-collapse:collapse; width:100%; margin:14px 0 28px; font-size:14px; }}
th,td {{ border-bottom:1px solid #e1e6ec; padding:7px 9px; text-align:left; }}
th {{ background:#eef2f6; position:sticky; top:0; }} .number {{ text-align:right; font-variant-numeric:tabular-nums; }}
.formula {{ background:#f8fafc; border-left:4px solid #73849a; padding:11px 14px; }}
.ok {{ color:#16724a; font-weight:600; }} code {{ font-family:"SFMono-Regular",Consolas,monospace; }}
</style></head><body>
<h1>L{metadata['layer']}:O{metadata['direction']} · Swift trace</h1>
<p class="sub">{html.escape(metadata['model'])} · {len(tokens)} tokens · {html.escape(metadata['projection'])}</p>
<pre>{prompt}</pre>
<p class="formula"><code>attention_output[t] = W_O @ a[t] + bias</code><br>
<code>O{metadata['direction']}_contribution[t] = a[t,{metadata['direction']}] · W_O[:,{metadata['direction']}]</code></p>
<p class="ok">Maximum reconstruction error: {diagnostics['attention_reconstruction_max_abs_error']:.3e}</p>
<h2>O{metadata['direction']} across tokens</h2>
<table><thead><tr><th>Pos</th><th>Token</th><th>ID</th><th>Coefficient</th><th>Corpus z</th><th>O{metadata['direction']} Δr norm</th><th>Full attn Δr norm</th><th>Post-attn r norm</th><th>MLP Δr norm</th><th>Final r norm</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
<h2>Saved vectors</h2>
<table><thead><tr><th>Tensor key</th><th>Meaning</th><th>Space</th><th>Shape</th></tr></thead><tbody>{stage_rows}</tbody></table>
<p>All values are stored in <code>trace.pt</code> as float32 CPU tensors. The isolated direction excludes the output-projection bias.</p>
</body></html>"""
    output_path.write_text(doc, encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    encoded = tokenizer(args.prompt, return_tensors="pt", add_special_tokens=False)
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
    if not 0 <= args.layer < len(model.model.layers):
        raise ValueError(f"Layer {args.layer} is outside the model")
    layer = model.model.layers[args.layer]
    if not hasattr(layer, "linear_attn"):
        raise ValueError("This tracer currently expects a Qwen3.5 linear-attention layer")
    mixer = layer.linear_attn
    if not 0 <= args.direction < mixer.out_proj.in_features:
        raise ValueError(
            f"Direction {args.direction} outside [0, {mixer.out_proj.in_features})"
        )

    captured: dict[str, torch.Tensor] = {}
    handles: list[Any] = []

    def output_hook(name: str):
        def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            value = output[0] if isinstance(output, tuple) else output
            captured[name] = cpu_float(value)
        return hook

    def input_hook(name: str):
        def hook(_module: torch.nn.Module, inputs: tuple[Any, ...]) -> None:
            captured[name] = cpu_float(inputs[0])
        return hook

    def mixer_norm_input_hook(_module: torch.nn.Module, inputs: tuple[Any, ...]) -> None:
        head_dim = int(mixer.head_v_dim)
        sequence_length = int(encoded.input_ids.shape[1])
        captured["linear_attn_core_before_norm"] = (
            inputs[0].detach().float().cpu().reshape(sequence_length, -1, head_dim)
        )
        captured["linear_attn_output_gate"] = (
            inputs[1].detach().float().cpu().reshape(sequence_length, -1, head_dim)
        )

    handles.extend(
        [
            model.model.embed_tokens.register_forward_hook(output_hook("embedding")),
            layer.input_layernorm.register_forward_hook(output_hook("input_layernorm")),
            mixer.in_proj_qkv.register_forward_hook(output_hook("linear_attn_in_proj_qkv")),
            mixer.in_proj_z.register_forward_hook(output_hook("linear_attn_in_proj_z")),
            mixer.in_proj_b.register_forward_hook(output_hook("linear_attn_in_proj_b")),
            mixer.in_proj_a.register_forward_hook(output_hook("linear_attn_in_proj_a")),
            mixer.norm.register_forward_pre_hook(mixer_norm_input_hook),
            mixer.out_proj.register_forward_pre_hook(input_hook("attention_out_proj_input")),
            mixer.out_proj.register_forward_hook(output_hook("attention_output")),
            layer.post_attention_layernorm.register_forward_hook(output_hook("post_attention_layernorm")),
            layer.mlp.gate_proj.register_forward_hook(output_hook("mlp_gate_projection")),
            layer.mlp.up_proj.register_forward_hook(output_hook("mlp_up_projection")),
            layer.mlp.down_proj.register_forward_pre_hook(input_hook("mlp_post_swiglu")),
            layer.mlp.register_forward_hook(output_hook("mlp_output")),
            layer.register_forward_hook(output_hook("layer_output")),
        ]
    )
    try:
        with torch.inference_mode():
            model.model(
                input_ids=encoded.input_ids.to(model.device),
                attention_mask=encoded.attention_mask.to(model.device),
                use_cache=False,
            )
    finally:
        for handle in handles:
            handle.remove()

    captured["post_attention_residual"] = captured["embedding"] + captured["attention_output"]
    direction_vector = mixer.out_proj.weight[:, args.direction].detach().float().cpu()
    coefficient = captured["attention_out_proj_input"][:, args.direction]
    captured["direction_contribution"] = coefficient[:, None] * direction_vector[None, :]
    captured["direction_vector"] = direction_vector

    value_head = args.direction // mixer.head_v_dim
    head_coordinate = args.direction % mixer.head_v_dim
    core_coordinate = captured["linear_attn_core_before_norm"][:, value_head, head_coordinate]
    gate_coordinate = captured["linear_attn_output_gate"][:, value_head, head_coordinate]
    core_heads = captured["linear_attn_core_before_norm"][:, value_head]
    rms_denominator = torch.sqrt(
        core_heads.square().mean(dim=-1) + float(mixer.norm.variance_epsilon)
    )
    norm_weight = mixer.norm.weight[head_coordinate].detach().float().cpu()
    normalized_core_coordinate = core_coordinate / rms_denominator * norm_weight
    silu_gate_coordinate = torch.nn.functional.silu(gate_coordinate)
    captured["direction_core_coordinate"] = core_coordinate
    captured["direction_core_rms_denominator"] = rms_denominator
    captured["direction_normalized_core_coordinate"] = normalized_core_coordinate
    captured["direction_gate_coordinate"] = gate_coordinate
    captured["direction_silu_gate_coordinate"] = silu_gate_coordinate

    out_input = captured["attention_out_proj_input"]
    weight = mixer.out_proj.weight.detach().float().cpu()
    reconstructed = out_input @ weight.T
    if mixer.out_proj.bias is not None:
        reconstructed += mixer.out_proj.bias.detach().float().cpu()
    reconstruction_error = float((reconstructed - captured["attention_output"]).abs().max())
    residual_error = float(
        (
            captured["post_attention_residual"]
            + captured["mlp_output"]
            - captured["layer_output"]
        ).abs().max()
    )
    isolated_share = (
        captured["direction_contribution"].norm(dim=-1)
        / captured["attention_output"].norm(dim=-1).clamp_min(1e-12)
    )
    diagnostics = {
        "attention_reconstruction_max_abs_error": reconstruction_error,
        "layer_residual_reconstruction_max_abs_error": residual_error,
        "direction_coefficient_min": float(coefficient.min()),
        "direction_coefficient_max": float(coefficient.max()),
        "direction_coefficient_max_abs": float(coefficient.abs().max()),
        "direction_value_head": value_head,
        "direction_head_coordinate": head_coordinate,
        "direction_factorization_max_abs_error": float(
            (normalized_core_coordinate * silu_gate_coordinate - coefficient).abs().max()
        ),
        "direction_contribution_share_of_attention_norm_max": float(isolated_share.max()),
        "direction_contribution_attention_cosine": cosine_rows(
            captured["direction_contribution"], captured["attention_output"]
        ).tolist(),
        "cpu_pages_touched_gib": touched / 1024**3,
    }

    input_ids = encoded.input_ids[0].tolist()
    tokens = [token_label(tokenizer, token_id) for token_id in input_ids]
    model_name = args.model.rsplit("/", 1)[-1]
    output_dir = (
        args.output_root
        / model_name
        / f"layer_{args.layer:02d}"
        / f"attention_output_O{args.direction}_swift_trace"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir.parent / "attention_output_metrics" / "en" / "metrics.pt"
    calibration = None
    if metrics_path.exists():
        metrics = torch.load(metrics_path, map_location="cpu", weights_only=True)
        mean = metrics["mean"][args.direction].float()
        std = metrics["variance"][args.direction].float().sqrt()
        captured["direction_z_score"] = (coefficient - mean) / std.clamp_min(1e-12)
        calibration = {
            "source": str(metrics_path.resolve()),
            "mean": float(mean),
            "standard_deviation": float(std),
        }
    metadata = {
        "model": args.model,
        "revision": args.revision,
        "layer": args.layer,
        "direction": args.direction,
        "projection": "linear_attn.out_proj",
        "prompt": args.prompt,
        "tokens": tokens,
        "token_ids": input_ids,
        "calibration": calibration,
    }
    torch.save({"metadata": metadata, "tensors": captured, "diagnostics": diagnostics}, output_dir / "trace.pt")
    (output_dir / "summary.json").write_text(
        json.dumps({"metadata": metadata, "diagnostics": diagnostics}, indent=2),
        encoding="utf-8",
    )
    render_report(output_dir / "report.html", metadata, tokens, input_ids, captured, diagnostics)

    print(json.dumps(diagnostics, indent=2))
    print(f"Saved trace to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
