from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from gb10_load_llm import load_model_to_cuda
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "Qwen/Qwen3.5-0.8B-Base"
MODEL_REVISION = "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"


class TargetCaptured(Exception):
    """Stop the forward pass immediately after capturing the target neuron."""


@dataclass
class ContinuousResult:
    tail: str
    restart: int
    activation: float
    activation_trace: torch.Tensor
    embeddings: torch.Tensor
    initial_ids: torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Maximize a Qwen3.5-0.8B-Base post-SwiGLU neuron using continuous "
            "input embeddings, then project and refine an actual token sequence."
        )
    )
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--neuron", type=int, default=0)
    parser.add_argument("--sequence-length", type=int, default=8)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--restarts", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument(
        "--alternating-rounds",
        type=int,
        default=3,
        help=(
            "Number of continuous-optimization -> nearest-token-snap rounds "
            "per restart. Each round runs --steps optimization steps."
        ),
    )
    parser.add_argument(
        "--tail",
        choices=("both", "positive", "negative"),
        default="both",
    )
    parser.add_argument("--nearest-tokens", type=int, default=32)
    parser.add_argument("--refinement-passes", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/mlp_measurement"),
    )
    parser.add_argument(
        "--output-name",
        default="activation_maximization.json",
        help="JSON filename inside the model/layer/neuron directory.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace, num_layers: int, num_neurons: int) -> None:
    if not 0 <= args.layer < num_layers:
        raise ValueError(f"--layer must be in 0..{num_layers - 1}")
    if not 0 <= args.neuron < num_neurons:
        raise ValueError(f"--neuron must be in 0..{num_neurons - 1}")
    if args.sequence_length <= 0:
        raise ValueError("--sequence-length must be positive")
    if args.steps <= 0 or args.restarts <= 0 or args.alternating_rounds <= 0:
        raise ValueError(
            "--steps, --restarts, and --alternating-rounds must be positive"
        )
    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be positive")
    if args.nearest_tokens <= 0 or args.refinement_passes < 0:
        raise ValueError("Invalid discrete-refinement settings")
    if Path(args.output_name).name != args.output_name:
        raise ValueError("--output-name must be a filename, not a path")


def safe_token_text(tokenizer: AutoTokenizer, token_id: int) -> str:
    return tokenizer.decode(
        [token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False
    )


def make_capture_hook(
    captured: dict[str, torch.Tensor], neuron: int
):
    def hook(
        _module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]
    ) -> None:
        captured["activation"] = inputs[0][:, :, neuron]
        raise TargetCaptured

    return hook


def run_to_target(
    model: AutoModelForCausalLM,
    captured: dict[str, torch.Tensor],
    *,
    input_ids: torch.Tensor | None = None,
    inputs_embeds: torch.Tensor | None = None,
) -> torch.Tensor:
    captured.clear()
    source = input_ids if input_ids is not None else inputs_embeds
    if source is None:
        raise ValueError("input_ids or inputs_embeds is required")
    attention_mask = torch.ones(source.shape[:2], dtype=torch.long, device=model.device)
    try:
        model.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=False,
        )
    except TargetCaptured:
        pass
    if "activation" not in captured:
        raise RuntimeError("Target-neuron hook did not fire")
    return captured["activation"]


def random_initial_ids(
    tokenizer: AutoTokenizer,
    length: int,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    special_ids = set(tokenizer.all_special_ids)
    while True:
        ids = torch.randint(
            0,
            len(tokenizer),
            (length,),
            generator=generator,
            device="cpu",
        )
        if all(int(token_id) not in special_ids for token_id in ids):
            return ids.unsqueeze(0).to(device)


def optimize_continuous(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    captured: dict[str, torch.Tensor],
    *,
    tail: str,
    restart: int,
    sequence_length: int,
    steps: int,
    learning_rate: float,
    generator: torch.Generator,
    initial_ids: torch.Tensor | None = None,
    round_index: int = 0,
) -> ContinuousResult:
    embedding_table = model.get_input_embeddings().weight
    if initial_ids is None:
        initial_ids = random_initial_ids(
            tokenizer, sequence_length, generator, model.device
        )
    else:
        initial_ids = initial_ids.to(model.device)
    initial = embedding_table[initial_ids].detach().float()
    target_norms = initial.norm(dim=-1, keepdim=True)
    optimized = torch.nn.Parameter(initial.clone())
    optimizer = torch.optim.Adam([optimized], lr=learning_rate)
    direction = 1.0 if tail == "positive" else -1.0
    best_score = float("-inf")
    best_activation = float("nan")
    best_activation_trace = torch.empty(sequence_length)
    best_embeddings = optimized.detach().cpu().clone()

    progress = tqdm(
        range(steps),
        desc=f"{tail} restart {restart} round {round_index + 1}",
        leave=False,
    )
    for _step in progress:
        optimizer.zero_grad(set_to_none=True)
        activation_trace = run_to_target(
            model,
            captured,
            inputs_embeds=optimized.to(dtype=model.dtype),
        )[0]
        activation = activation_trace.mean()
        score = direction * activation
        with torch.no_grad():
            value = float(activation.detach())
            signed_score = direction * value
            if signed_score > best_score:
                best_score = signed_score
                best_activation = value
                best_activation_trace = activation_trace.detach().float().cpu().clone()
                best_embeddings = optimized.detach().cpu().clone()
        (-score).backward()
        optimizer.step()

        with torch.no_grad():
            norms = optimized.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            optimized.mul_(target_norms / norms)
        if _step % 10 == 0 or _step + 1 == steps:
            progress.set_postfix(activation=f"{value:+.4f}")

    with torch.no_grad():
        final_trace = run_to_target(
            model,
            captured,
            inputs_embeds=optimized.to(dtype=model.dtype),
        )[0]
        final_activation = float(final_trace.mean())
        if direction * final_activation > best_score:
            best_activation = final_activation
            best_activation_trace = final_trace.detach().float().cpu().clone()
            best_embeddings = optimized.detach().cpu().clone()

    return ContinuousResult(
        tail=tail,
        restart=restart,
        activation=best_activation,
        activation_trace=best_activation_trace,
        embeddings=best_embeddings,
        initial_ids=initial_ids.detach().cpu(),
    )


def nearest_vocabulary_tokens(
    model: AutoModelForCausalLM,
    query_embeddings: torch.Tensor,
    *,
    top_k: int,
    chunk_size: int = 4096,
) -> tuple[torch.Tensor, torch.Tensor]:
    table = model.get_input_embeddings().weight.detach()
    queries = F.normalize(query_embeddings[0].to(model.device).float(), dim=-1)
    top_scores = torch.full(
        (queries.shape[0], top_k),
        float("-inf"),
        device=model.device,
    )
    top_ids = torch.zeros(
        (queries.shape[0], top_k), dtype=torch.long, device=model.device
    )

    for start in range(0, table.shape[0], chunk_size):
        chunk = F.normalize(table[start : start + chunk_size].float(), dim=-1)
        scores = queries @ chunk.T
        chunk_k = min(top_k, scores.shape[1])
        chunk_scores, chunk_ids = scores.topk(chunk_k, dim=-1)
        chunk_ids += start
        combined_scores = torch.cat((top_scores, chunk_scores), dim=-1)
        combined_ids = torch.cat((top_ids, chunk_ids), dim=-1)
        top_scores, positions = combined_scores.topk(top_k, dim=-1)
        top_ids = combined_ids.gather(-1, positions)

    return top_ids.cpu(), top_scores.cpu()


def evaluate_token_ids(
    model: AutoModelForCausalLM,
    captured: dict[str, torch.Tensor],
    input_ids: torch.Tensor,
) -> torch.Tensor:
    with torch.inference_mode():
        return run_to_target(
            model, captured, input_ids=input_ids.to(model.device)
        ).detach().cpu()


def refine_discrete(
    model: AutoModelForCausalLM,
    captured: dict[str, torch.Tensor],
    initial_ids: torch.Tensor,
    candidates: torch.Tensor,
    *,
    tail: str,
    passes: int,
) -> tuple[torch.Tensor, float, torch.Tensor]:
    direction = 1.0 if tail == "positive" else -1.0
    current = initial_ids.clone()
    current_trace = evaluate_token_ids(model, captured, current)[0]
    current_activation = float(current_trace.mean())

    for _pass in range(passes):
        changed = False
        for position in range(current.shape[1]):
            position_candidates = torch.unique(
                torch.cat((current[:, position].cpu(), candidates[position]))
            )
            batch = current.cpu().repeat(len(position_candidates), 1)
            batch[:, position] = position_candidates
            activation_traces = evaluate_token_ids(model, captured, batch)
            activations = activation_traces.mean(dim=-1)
            scores = direction * activations
            best = int(scores.argmax())
            best_activation = float(activations[best])
            best_id = int(position_candidates[best])
            if best_id != int(current[0, position]):
                current[0, position] = best_id
                changed = True
            current_activation = best_activation
            current_trace = activation_traces[best]
        if not changed:
            break

    return current, current_activation, current_trace


def result_payload(
    tokenizer: AutoTokenizer,
    result: ContinuousResult,
    nearest_ids: torch.Tensor,
    nearest_scores: torch.Tensor,
    projected_activation: float,
    projected_trace: torch.Tensor,
    refined_ids: torch.Tensor,
    refined_activation: float,
    refined_trace: torch.Tensor,
) -> dict[str, object]:
    projected_ids = nearest_ids[:, 0].unsqueeze(0)
    return {
        "tail": result.tail,
        "restart": result.restart,
        "continuous_activation": result.activation,
        "continuous_activation_trace": result.activation_trace.tolist(),
        "initial_token_ids": result.initial_ids[0].tolist(),
        "initial_text": tokenizer.decode(
            result.initial_ids[0], skip_special_tokens=False
        ),
        "projected_token_ids": projected_ids[0].tolist(),
        "projected_tokens": [
            safe_token_text(tokenizer, int(token_id)) for token_id in projected_ids[0]
        ],
        "projected_text": tokenizer.decode(
            projected_ids[0], skip_special_tokens=False
        ),
        "projected_activation": projected_activation,
        "projected_activation_trace": projected_trace.tolist(),
        "refined_token_ids": refined_ids[0].tolist(),
        "refined_tokens": [
            safe_token_text(tokenizer, int(token_id)) for token_id in refined_ids[0]
        ],
        "refined_text": tokenizer.decode(refined_ids[0], skip_special_tokens=False),
        "refined_activation": refined_activation,
        "refined_activation_trace": refined_trace.tolist(),
        "nearest_tokens_by_position": [
            [
                {
                    "token_id": int(token_id),
                    "token": safe_token_text(tokenizer, int(token_id)),
                    "cosine_similarity": float(score),
                }
                for token_id, score in zip(ids, scores, strict=True)
            ]
            for ids, scores in zip(nearest_ids, nearest_scores, strict=True)
        ],
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    model, touched = load_model_to_cuda(
        AutoModelForCausalLM,
        MODEL_ID,
        revision=MODEL_REVISION,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        touch="auto",
        return_touched=True,
    )
    model.eval()
    model.requires_grad_(False)
    validate_args(
        args,
        num_layers=len(model.model.layers),
        num_neurons=int(model.config.intermediate_size),
    )

    captured: dict[str, torch.Tensor] = {}
    handle = model.model.layers[args.layer].mlp.down_proj.register_forward_pre_hook(
        make_capture_hook(captured, args.neuron)
    )
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    tails = ("positive", "negative") if args.tail == "both" else (args.tail,)
    payloads: list[dict[str, object]] = []

    try:
        for tail in tails:
            direction = 1.0 if tail == "positive" else -1.0
            restart_results: list[dict[str, object]] = []
            for restart in range(args.restarts):
                snapped_ids: torch.Tensor | None = None
                rounds: list[dict[str, object]] = []
                final_result: ContinuousResult | None = None
                final_nearest_ids: torch.Tensor | None = None
                final_nearest_scores: torch.Tensor | None = None
                final_snapped_trace: torch.Tensor | None = None

                for round_index in range(args.alternating_rounds):
                    result = optimize_continuous(
                        model,
                        tokenizer,
                        captured,
                        tail=tail,
                        restart=restart,
                        sequence_length=args.sequence_length,
                        steps=args.steps,
                        learning_rate=args.learning_rate,
                        generator=generator,
                        initial_ids=snapped_ids,
                        round_index=round_index,
                    )
                    nearest_ids, nearest_scores = nearest_vocabulary_tokens(
                        model,
                        result.embeddings,
                        top_k=min(args.nearest_tokens, len(tokenizer)),
                    )
                    snapped_ids = nearest_ids[:, 0].unsqueeze(0)
                    snapped_trace = evaluate_token_ids(
                        model, captured, snapped_ids
                    )[0]
                    snapped_activation = float(snapped_trace.mean())
                    rounds.append(
                        {
                            "round": round_index + 1,
                            "continuous_activation": result.activation,
                            "continuous_activation_trace": (
                                result.activation_trace.tolist()
                            ),
                            "snapped_token_ids": snapped_ids[0].tolist(),
                            "snapped_tokens": [
                                safe_token_text(tokenizer, int(token_id))
                                for token_id in snapped_ids[0]
                            ],
                            "snapped_text": tokenizer.decode(
                                snapped_ids[0], skip_special_tokens=False
                            ),
                            "snapped_activation": snapped_activation,
                            "snapped_activation_trace": snapped_trace.tolist(),
                        }
                    )
                    final_result = result
                    final_nearest_ids = nearest_ids
                    final_nearest_scores = nearest_scores
                    final_snapped_trace = snapped_trace

                assert final_result is not None
                assert final_nearest_ids is not None
                assert final_nearest_scores is not None
                assert final_snapped_trace is not None
                restart_results.append(
                    {
                        "result": final_result,
                        "nearest_ids": final_nearest_ids,
                        "nearest_scores": final_nearest_scores,
                        "snapped_ids": snapped_ids,
                        "snapped_trace": final_snapped_trace,
                        "snapped_activation": float(final_snapped_trace.mean()),
                        "rounds": rounds,
                    }
                )

            best_restart = max(
                restart_results,
                key=lambda item: direction * float(item["snapped_activation"]),
            )
            best = best_restart["result"]
            nearest_ids = best_restart["nearest_ids"]
            nearest_scores = best_restart["nearest_scores"]
            projected_ids = best_restart["snapped_ids"]
            projected_trace = best_restart["snapped_trace"]
            assert isinstance(best, ContinuousResult)
            assert isinstance(nearest_ids, torch.Tensor)
            assert isinstance(nearest_scores, torch.Tensor)
            assert isinstance(projected_ids, torch.Tensor)
            assert isinstance(projected_trace, torch.Tensor)
            projected_activation = float(best_restart["snapped_activation"])
            refined_ids, refined_activation, refined_trace = refine_discrete(
                model,
                captured,
                projected_ids,
                nearest_ids,
                tail=tail,
                passes=args.refinement_passes,
            )
            payload = result_payload(
                tokenizer,
                best,
                nearest_ids,
                nearest_scores,
                projected_activation,
                projected_trace,
                refined_ids,
                refined_activation,
                refined_trace,
            )
            payload["alternating_rounds"] = best_restart["rounds"]
            payload["restart_final_snapped_activations"] = [
                float(item["snapped_activation"]) for item in restart_results
            ]
            payloads.append(payload)
            print(
                f"{tail} mean across {args.sequence_length} positions: "
                f"continuous={best.activation:+.6f}, "
                f"projected={projected_activation:+.6f}, "
                f"refined={refined_activation:+.6f}"
            )
            print(
                f"{tail} refined trace: "
                f"{[round(float(value), 6) for value in refined_trace]}"
            )
            print(f"{tail} refined tokens: {payload['refined_tokens']}")
            print(f"{tail} refined text: {payload['refined_text']!r}")
    finally:
        handle.remove()

    output_dir = (
        args.output_root
        / MODEL_ID.rsplit("/", 1)[-1]
        / f"layer_{args.layer:02d}"
        / f"neuron_{args.neuron:04d}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / args.output_name
    artifact = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "layer": args.layer,
        "neuron": args.neuron,
        "activation_definition": "input to mlp.down_proj (post-SwiGLU)",
        "position_objective": "mean activation across all sequence positions",
        "sequence_length": args.sequence_length,
        "steps": args.steps,
        "alternating_rounds": args.alternating_rounds,
        "restarts": args.restarts,
        "learning_rate": args.learning_rate,
        "nearest_tokens": args.nearest_tokens,
        "refinement_passes": args.refinement_passes,
        "seed": args.seed,
        "cpu_pages_touched_gib": touched / 1024**3,
        "important_note": (
            "Continuous embeddings are an off-token-manifold upper-bound search. "
            "Only projected/refined token sequences are realizable token inputs."
        ),
        "results": payloads,
    }
    output.write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Saved: {output.resolve()}")


if __name__ == "__main__":
    main()
