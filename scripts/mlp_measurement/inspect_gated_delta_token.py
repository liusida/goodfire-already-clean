from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from gb10_load_llm import load_model_to_cuda
from transformers import AutoModelForCausalLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconstruct and inspect one token/head in a Qwen3.5 GatedDeltaNet trace."
    )
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B-Base")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--direction", type=int, default=1825)
    parser.add_argument("--token", default=" func")
    parser.add_argument(
        "--trace",
        type=Path,
        default=Path(
            "outputs/mlp_measurement/Qwen3.5-0.8B-Base/layer_00/"
            "attention_output_O1825_swift_trace/trace.pt"
        ),
    )
    return parser.parse_args()


def scalar(value: torch.Tensor) -> float:
    return float(value.detach().float().cpu())


def main() -> None:
    args = parse_args()
    trace = torch.load(args.trace, map_location="cpu", weights_only=False)
    tokens = trace["metadata"]["tokens"]
    matches = [index for index, token in enumerate(tokens) if token == args.token]
    if len(matches) != 1:
        raise ValueError(f"Expected one {args.token!r} token, found positions {matches}")
    target = matches[0]

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
    mixer = model.model.layers[args.layer].linear_attn
    head = args.direction // mixer.head_v_dim
    coordinate = args.direction % mixer.head_v_dim
    device = model.device

    # Reproduce the depthwise causal convolution applied to projected Q/K/V.
    projected = trace["tensors"]["linear_attn_in_proj_qkv"].to(device=device, dtype=torch.bfloat16)
    conv_input = projected.T.unsqueeze(0)
    kernel = mixer.conv1d.weight.squeeze(1)
    conv = F.conv1d(
        conv_input,
        weight=kernel.unsqueeze(1),
        bias=mixer.conv1d.bias,
        padding=kernel.shape[-1] - 1,
        groups=conv_input.shape[1],
    )[:, :, : projected.shape[0]]
    conv = F.silu(conv).transpose(1, 2)
    query, key, value = torch.split(
        conv,
        [mixer.key_dim, mixer.key_dim, mixer.value_dim],
        dim=-1,
    )
    query = query.reshape(1, -1, mixer.num_k_heads, mixer.head_k_dim)
    key = key.reshape(1, -1, mixer.num_k_heads, mixer.head_k_dim)
    value = value.reshape(1, -1, mixer.num_v_heads, mixer.head_v_dim)
    if mixer.num_v_heads // mixer.num_k_heads > 1:
        repeats = mixer.num_v_heads // mixer.num_k_heads
        query = query.repeat_interleave(repeats, dim=2)
        key = key.repeat_interleave(repeats, dim=2)

    b_pre = trace["tensors"]["linear_attn_in_proj_b"].to(device=device, dtype=torch.bfloat16)
    a_pre = trace["tensors"]["linear_attn_in_proj_a"].to(device=device, dtype=torch.bfloat16)
    beta = b_pre.sigmoid().float()
    decay_log = -mixer.A_log.float().exp() * F.softplus(a_pre.float() + mixer.dt_bias)

    q_raw = query.float()
    k_raw = key.float()
    q = F.normalize(q_raw, dim=-1, eps=1e-6) / (mixer.head_k_dim**0.5)
    k = F.normalize(k_raw, dim=-1, eps=1e-6)
    v = value.float()
    state = torch.zeros(
        (mixer.num_v_heads, mixer.head_k_dim, mixer.head_v_dim),
        dtype=torch.float32,
        device=device,
    )
    selected: dict[str, float | int | str] | None = None
    reconstructed_core = []
    for position in range(projected.shape[0]):
        q_t, k_t, v_t = q[0, position], k[0, position], v[0, position]
        state_before = state
        state_decayed = state_before * decay_log[position].exp()[:, None, None]
        prediction = (state_decayed * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - prediction) * beta[position].unsqueeze(-1)
        update = k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        state = state_decayed + update
        core_from_old_state = (state_decayed * q_t.unsqueeze(-1)).sum(dim=-2)
        core_from_current_update = (update * q_t.unsqueeze(-1)).sum(dim=-2)
        core = core_from_old_state + core_from_current_update
        reconstructed_core.append(core)
        if position == target:
            core_head = core[head]
            rms = torch.sqrt(core_head.square().mean() + mixer.norm.variance_epsilon)
            norm_weight = mixer.norm.weight[coordinate].float()
            normalized_coordinate = core_head[coordinate] / rms * norm_weight
            z = trace["tensors"]["direction_gate_coordinate"][position].to(device)
            gate = F.silu(z)
            old_core_head = core_from_old_state[head]
            old_core_rms = torch.sqrt(
                old_core_head.square().mean() + mixer.norm.variance_epsilon
            )
            old_normalized_coordinate = (
                old_core_head[coordinate] / old_core_rms * norm_weight
            )
            update_core_head = core_from_current_update[head]
            update_only_rms = torch.sqrt(
                update_core_head.square().mean() + mixer.norm.variance_epsilon
            )
            update_only_normalized_coordinate = (
                update_core_head[coordinate] / update_only_rms * norm_weight
            )
            selected = {
                "token": args.token,
                "position": position,
                "direction": args.direction,
                "value_head": head,
                "head_coordinate": coordinate,
                "input_rmsnorm_norm": scalar(
                    trace["tensors"]["input_layernorm"][position].norm()
                ),
                "q_raw_norm": scalar(q_raw[0, position, head].norm()),
                "k_raw_norm": scalar(k_raw[0, position, head].norm()),
                "v_norm": scalar(v_t[head].norm()),
                "q_normalized_scaled_norm": scalar(q_t[head].norm()),
                "k_normalized_norm": scalar(k_t[head].norm()),
                "b_preactivation": scalar(b_pre[position, head]),
                "beta": scalar(beta[position, head]),
                "a_preactivation": scalar(a_pre[position, head]),
                "decay_log_g": scalar(decay_log[position, head]),
                "decay_multiplier_exp_g": scalar(decay_log[position, head].exp()),
                "state_before_norm": scalar(state_before[head].norm()),
                "state_after_decay_norm": scalar(state_decayed[head].norm()),
                "predicted_value_norm": scalar(prediction[head].norm()),
                "value_residual_before_beta_norm": scalar((v_t[head] - prediction[head]).norm()),
                "delta_norm": scalar(delta[head].norm()),
                "state_update_norm": scalar(update[head].norm()),
                "state_after_update_norm": scalar(state[head].norm()),
                "old_state_core_readout_norm": scalar(old_core_head.norm()),
                "old_state_core_coordinate": scalar(old_core_head[coordinate]),
                "current_update_core_readout_norm": scalar(update_core_head.norm()),
                "current_update_core_coordinate": scalar(update_core_head[coordinate]),
                "core_readout_norm": scalar(core_head.norm()),
                "core_coordinate": scalar(core_head[coordinate]),
                "core_rms_denominator": scalar(rms),
                "normalized_core_coordinate": scalar(normalized_coordinate),
                "z_preactivation": scalar(z),
                "silu_z": scalar(gate),
                "counterfactual_no_current_update_normalized_core_coordinate": scalar(
                    old_normalized_coordinate
                ),
                "counterfactual_no_current_update_coefficient": scalar(
                    old_normalized_coordinate * gate
                ),
                "counterfactual_current_update_only_normalized_core_coordinate": scalar(
                    update_only_normalized_coordinate
                ),
                "counterfactual_current_update_only_coefficient": scalar(
                    update_only_normalized_coordinate * gate
                ),
                "factorized_coefficient": scalar(normalized_coordinate * gate),
                "captured_coefficient": scalar(
                    trace["tensors"]["attention_out_proj_input"][position, args.direction]
                ),
                "direction_residual_update_norm": scalar(
                    trace["tensors"]["direction_contribution"][position].norm()
                ),
            }

    if selected is None:
        raise RuntimeError("Target token was not inspected")
    reconstructed_core_tensor = torch.stack(reconstructed_core).cpu()
    captured_core = trace["tensors"]["linear_attn_core_before_norm"]
    selected["core_reconstruction_max_abs_error"] = float(
        (reconstructed_core_tensor - captured_core).abs().max()
    )
    output = args.trace.parent / f"O{args.direction}_{args.token.strip()}_delta_details.json"
    output.write_text(json.dumps(selected, indent=2), encoding="utf-8")
    print(json.dumps(selected, indent=2))
    print(f"Saved to {output.resolve()}")


if __name__ == "__main__":
    main()
