from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from gb10_load_llm import load_model_to_cuda
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "Qwen/Qwen3.5-9B-Base"
OUTPUT_DIR = Path(
    "outputs/mlp_measurement/Qwen3.5-9B-Base/michael_jordan_neuron_search"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Adversarially search every post-SwiGLU neuron for selectivity to "
            "Michael Jordan over known lexical, Jordan, basketball, and entity controls."
        )
    )
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def build_prompts() -> list[dict[str, object]]:
    prompts: list[dict[str, object]] = []

    def add(group: str, label: str, text: str, target: str) -> None:
        target_start = text.index(target)
        prompts.append(
            {
                "id": f"P{len(prompts) + 1:03d}",
                "group": group,
                "label": label,
                "text": text,
                "target_text": target,
                "target_start": target_start,
                "target_end": target_start + len(target),
            }
        )

    positive_texts = [
        "Michael Jordan led the Chicago Bulls to six NBA championships during the 1990s.",
        "Fans still debate whether Michael Jordan is the greatest basketball player ever.",
        "The documentary follows Michael Jordan through the Bulls' final championship season.",
        "As a rookie, Michael Jordan quickly became one of the NBA's most exciting scorers.",
        "Nike's partnership with Michael Jordan transformed signature basketball shoes.",
        "After retiring, Michael Jordan became the principal owner of the Charlotte Hornets.",
        "Collectors paid a record price for a jersey worn by Michael Jordan in the 1998 Finals.",
        "The coach recalled how Michael Jordan demanded intense effort at every practice.",
        "Young athletes around the world tried to imitate Michael Jordan on local courts.",
        "The biography explains how Michael Jordan became a globally recognized athlete.",
        "Chicago basketball fans celebrated when Michael Jordan returned from retirement.",
        "In the final seconds of the game, Michael Jordan took the decisive shot.",
        "During the championship parade, Michael Jordan thanked his teammates and coaches.",
        "The Bulls built their 1990s dynasty around Michael Jordan and Scottie Pippen.",
        "Sports historians frequently rank Michael Jordan above other NBA superstars.",
        "At the United Center, a statue honors Michael Jordan for his basketball career.",
        "The television interview revisited Michael Jordan's famous competitive drive.",
        "In sneaker culture, Michael Jordan remains an enormously influential figure.",
        "The Hall of Fame ceremony celebrated Michael Jordan and his impact on basketball.",
        "Opposing defenders struggled to contain Michael Jordan throughout his prime.",
        "The 1992 Olympic team featured Michael Jordan alongside many other NBA legends.",
        "A newspaper profile described Michael Jordan as both an athlete and businessman.",
        "During the 1997 Finals, Michael Jordan delivered another memorable performance.",
        "Basketball analysts studied how Michael Jordan created space for his jump shot.",
        "The museum displayed shoes and uniforms once worn by Michael Jordan.",
        "Teammates said Michael Jordan raised the intensity of every training session.",
        "The advertisement made Michael Jordan one of the world's most visible endorsers.",
        "For many Chicago residents, Michael Jordan symbolizes the Bulls' golden era.",
        "The podcast devoted an episode to Michael Jordan and the history of the NBA.",
        "A generation remembers watching Michael Jordan win his sixth championship.",
    ]
    for index, text in enumerate(positive_texts, start=1):
        add("mj_positive", f"Michael Jordan context {index}", text, "Michael Jordan")

    same_surname = [
        ("Michael B. Jordan", "The actor Michael B. Jordan starred in Creed and Black Panther."),
        ("Michael B. Jordan", "A magazine interviewed Michael B. Jordan about directing his first film."),
        ("Michael B. Jordan", "Critics praised Michael B. Jordan for his dramatic performance."),
        ("Michael B. Jordan", "At the premiere, Michael B. Jordan greeted fans on the red carpet."),
        ("Leslie Jordan", "The comedian Leslie Jordan became known for his distinctive Southern humor."),
        ("Barbara Jordan", "The congresswoman Barbara Jordan delivered a celebrated convention speech."),
        ("DeAndre Jordan", "The center DeAndre Jordan finished the play with a powerful dunk."),
        ("Eddie Jordan", "The coach Eddie Jordan discussed his team's defensive strategy."),
        ("Neil Jordan", "The filmmaker Neil Jordan directed several acclaimed dramas."),
        ("June Jordan", "The poet June Jordan wrote about politics, identity, and justice."),
    ]
    for name, text in same_surname:
        add("other_surname_jordan", name, text, name)

    first_name_jordan = [
        ("Jordan Peele", "The director Jordan Peele wrote and directed the film Get Out."),
        ("Jordan Peele", "Horror audiences praised Jordan Peele for his original screenplay."),
        ("Jordan Peterson", "The psychologist Jordan Peterson became known for lectures and books."),
        ("Jordan Peterson", "During the interview, Jordan Peterson discussed personal responsibility."),
        ("Jordan Poole", "The guard Jordan Poole made a long three-point shot."),
        ("Jordan Clarkson", "The basketball player Jordan Clarkson came off the bench to score."),
        ("Jordan Spieth", "The golfer Jordan Spieth prepared for the final round."),
        ("Jordan Chiles", "The gymnast Jordan Chiles competed for the United States."),
    ]
    for label, text in first_name_jordan:
        add("other_firstname_jordan", label, text, "Jordan")

    geography = [
        ("Jordan country", "The country Jordan is located at the crossroads of Asia and Africa."),
        ("Jordan country", "In the Middle East, Jordan shares borders with several countries."),
        ("Jordan country", "Tourists visit Jordan to see Petra and Wadi Rum."),
        ("Jordan country", "The capital of Jordan is Amman."),
        ("Jordan River", "The Jordan River flows south toward the Dead Sea."),
        ("Jordan River", "Pilgrims gathered beside the Jordan River early in the morning."),
        ("Jordan River", "Water levels in the Jordan River have declined over time."),
        ("Jordan surname", "The surname is Jordan."),
        ("Jordan word", "The word Jordan appeared twice in the document."),
        ("Jordan standalone", "The answer was simply Jordan."),
    ]
    for label, text in geography:
        add("geographic_or_generic_jordan", label, text, "Jordan")

    associated_references = [
        ("Air Jordan", "Nike introduced the Air Jordan sneaker line in the 1980s."),
        ("Air Jordan", "Collectors waited overnight for the newest Air Jordan release."),
        ("Air Jordan", "The store displayed a rare pair of Air Jordan basketball shoes."),
        ("Air Jordan", "Sneaker historians examined the original Air Jordan design."),
        ("Air Jordan", "The athlete wore an Air Jordan shoe in the advertisement."),
        ("Air Jordan", "Resale prices rose sharply for the limited Air Jordan model."),
        ("His Airness", "Basketball fans called the soaring Chicago star His Airness."),
        ("His Airness", "The arena announcer introduced the Bulls legend as His Airness."),
        ("Jumpman", "Nike used the silhouette of the airborne player for its Jumpman logo."),
        ("Jumpman", "Collectors recognized the basketball icon on the Jumpman brand."),
        ("number 23", "Chicago fans still associate the Bulls' number 23 with six championships."),
        ("number 23", "The team retired number 23 to honor its greatest player."),
        ("Space Jam", "The basketball superstar crossed into Hollywood as the star of Space Jam."),
        ("The Last Dance", "The documentary series The Last Dance chronicled the final Bulls title run."),
        ("Flu Game", "Fans remember the exhausted star's performance as the famous Flu Game."),
        ("six-time Bulls champion", "The six-time Bulls champion became a global sporting icon."),
    ]
    for index, (target, text) in enumerate(associated_references, start=1):
        add("mj_associated", f"Associated reference {index}: {target}", text, target)

    bulls_people = [
        "Scottie Pippen helped the Chicago Bulls win six championships.",
        "Dennis Rodman dominated the boards during his years with the Bulls.",
        "Toni Kukoc provided scoring and playmaking from the bench.",
        "Horace Grant wore protective goggles while playing power forward.",
        "Steve Kerr hit a crucial shot in the 1997 NBA Finals.",
        "Phil Jackson coached the Bulls through six championship seasons.",
        "Derrick Rose became the youngest player to win the league MVP award.",
        "Joakim Noah brought energy and defense to Chicago's frontcourt.",
        "Ron Harper played an important defensive role for the Bulls.",
        "Luc Longley started at center during Chicago's second three-peat.",
    ]
    for text in bulls_people:
        name = text.split(" ", 2)[:2]
        target = " ".join(name)
        add("bulls_people", target, text, target)

    basketball_people = [
        "Kobe Bryant scored eighty-one points in a single NBA game.",
        "LeBron James won championships with three different franchises.",
        "Vince Carter became famous for spectacular dunks.",
        "Dominique Wilkins earned the nickname the Human Highlight Film.",
        "Magic Johnson directed the Lakers' fast-paced offense.",
        "Larry Bird starred for the Boston Celtics.",
        "Shaquille O'Neal overpowered defenders near the basket.",
        "Stephen Curry changed basketball with his long-range shooting.",
        "Kareem Abdul-Jabbar perfected the skyhook.",
        "Wilt Chamberlain once scored one hundred points.",
    ]
    for text in basketball_people:
        target = text.split(" scored", 1)[0].split(" won", 1)[0].split(" became", 1)[0]
        target = target.split(" earned", 1)[0].split(" directed", 1)[0]
        target = target.split(" starred", 1)[0].split(" overpowered", 1)[0]
        target = target.split(" changed", 1)[0].split(" perfected", 1)[0]
        target = target.split(" once", 1)[0]
        add("other_basketball_people", target, text, target)

    dunk_texts = [
        ("dunk", "The guard surprised everyone with a powerful dunk late in the game."),
        ("dunked", "The center dunked over two defenders in transition."),
        ("dunking", "The crowd cheered as the rookie began dunking during warmups."),
        ("slam dunk", "The fast break ended with a spectacular slam dunk."),
        ("layup", "The player avoided contact and finished a soft layup."),
        ("jump shot", "The defender contested the jump shot from the baseline."),
        ("three-pointer", "The shooter made a three-pointer at the buzzer."),
        ("dunk", "She used a quick motion to dunk the cookie into her coffee."),
    ]
    for target, text in dunk_texts:
        add("basketball_action", target, text, target)

    entity_completions = [
        ("Ross Perot", "The independent candidate Ross Perot won nearly twenty percent of the vote."),
        ("John Anderson", "The politician John Anderson ran as an independent candidate in 1980."),
        ("Ronald Reagan", "The president Ronald Reagan addressed the nation on television."),
        ("Ferrari 328", "The owner insured a red Ferrari 328 from the late 1980s."),
        ("Ferrari 308", "The collector restored a classic Ferrari 308 in his garage."),
        ("BMW 328", "The dealership advertised a used BMW 328 at a reduced price."),
        ("328", "The number printed on the ticket was 328."),
        ("Pippen", "The surname Pippen appeared on the back of the jersey."),
        ("Perot", "The surname Perot appeared in the election archive."),
        ("Jordan", "The surname Jordan appeared in the directory."),
    ]
    for target, text in entity_completions:
        add("known_false_positive_entities", target, text, target)

    famous_people = [
        "Michael Jackson became one of the best-selling recording artists in history.",
        "Brad Pitt attended the film premiere with the cast.",
        "Elon Musk discussed rockets and electric vehicles.",
        "Taylor Swift announced another international concert tour.",
        "Barack Obama delivered a speech at the university.",
        "Serena Williams won another major tennis championship.",
        "Cristiano Ronaldo scored for his national team.",
        "Oprah Winfrey interviewed the author on television.",
        "Steve Jobs introduced a new device on stage.",
        "Muhammad Ali became an icon beyond the sport of boxing.",
    ]
    for text in famous_people:
        target = text.split(" became", 1)[0].split(" attended", 1)[0]
        target = target.split(" discussed", 1)[0].split(" announced", 1)[0]
        target = target.split(" delivered", 1)[0].split(" won", 1)[0]
        target = target.split(" scored", 1)[0].split(" interviewed", 1)[0]
        target = target.split(" introduced", 1)[0]
        add("other_famous_people", target, text, target)

    return prompts


def locate_target_token(
    tokenizer: AutoTokenizer,
    text: str,
    target_end: int,
) -> tuple[int, int, str]:
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    candidates = [
        index
        for index, (start, end) in enumerate(encoded.offset_mapping)
        if start < target_end <= end
    ]
    if not candidates:
        raise ValueError(f"Could not locate target ending at {target_end} in {text!r}")
    position = candidates[-1]
    token_id = int(encoded.input_ids[position])
    token = tokenizer.decode(
        [token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    return position, token_id, token


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.top_k <= 0:
        raise ValueError("--batch-size and --top-k must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    prompts = build_prompts()
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    for prompt in prompts:
        position, token_id, token = locate_target_token(
            tokenizer, str(prompt["text"]), int(prompt["target_end"])
        )
        prompt["target_position"] = position
        prompt["target_token_id"] = token_id
        prompt["target_token"] = token

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
    layer_count = len(model.model.layers)
    neuron_count = int(model.model.layers[0].mlp.down_proj.in_features)
    layer_batches: list[list[torch.Tensor]] = [[] for _ in range(layer_count)]
    current_positions: torch.Tensor | None = None
    current_rows: torch.Tensor | None = None

    def make_hook(layer: int):
        def hook(_module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
            if current_positions is None or current_rows is None:
                raise RuntimeError("Target positions were not initialized")
            layer_batches[layer].append(
                inputs[0][current_rows, current_positions]
                .detach()
                .to(dtype=torch.float32, device="cpu")
            )

        return hook

    handles = [
        layer.mlp.down_proj.register_forward_pre_hook(make_hook(layer_index))
        for layer_index, layer in enumerate(model.model.layers)
    ]
    try:
        for start in range(0, len(prompts), args.batch_size):
            batch = prompts[start : start + args.batch_size]
            encoded = tokenizer(
                [str(prompt["text"]) for prompt in batch],
                add_special_tokens=False,
                padding=True,
                return_tensors="pt",
            )
            current_positions = torch.tensor(
                [int(prompt["target_position"]) for prompt in batch],
                dtype=torch.long,
                device=model.device,
            )
            current_rows = torch.arange(len(batch), device=model.device)
            with torch.inference_mode():
                model.model(
                    input_ids=encoded.input_ids.to(model.device),
                    attention_mask=encoded.attention_mask.to(model.device),
                    use_cache=False,
                )
            print(f"Captured {min(start + args.batch_size, len(prompts))}/{len(prompts)} prompts")
    finally:
        for handle in handles:
            handle.remove()

    activations = torch.stack(
        [torch.cat(batches, dim=0) for batches in layer_batches], dim=1
    )
    if activations.shape != (len(prompts), layer_count, neuron_count):
        raise RuntimeError(f"Unexpected activation shape: {tuple(activations.shape)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(activations, args.output_dir / "activations.pt")
    metadata = {
        "model_id": args.model,
        "model_revision": args.revision,
        "activation_definition": "input to mlp.down_proj (post-SwiGLU)",
        "shape": list(activations.shape),
        "cpu_pages_touched_gib": touched / 1024**3,
        "prompts": prompts,
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    layer_rms = activations.square().mean(dim=(0, 2)).sqrt().clamp_min(1e-8)
    normalized = activations / layer_rms[None, :, None]
    groups = sorted({str(prompt["group"]) for prompt in prompts})
    group_indices = {
        group: [index for index, prompt in enumerate(prompts) if prompt["group"] == group]
        for group in groups
    }
    positive_indices = group_indices.pop("mj_positive")
    associated_indices = group_indices.pop("mj_associated")
    positive = normalized[positive_indices]
    positive_mean_signed = positive.mean(dim=0)
    polarity = torch.where(positive_mean_signed >= 0, 1.0, -1.0)
    positive_aligned = positive * polarity
    positive_mean = positive_aligned.mean(dim=0)
    positive_q10 = torch.quantile(positive_aligned, 0.10, dim=0)
    consistency = (positive_aligned > 0).float().mean(dim=0)
    associated_aligned = normalized[associated_indices] * polarity
    associated_mean = associated_aligned.mean(dim=0)
    associated_q25 = torch.quantile(associated_aligned, 0.25, dim=0)
    associated_consistency = (associated_aligned > 0).float().mean(dim=0)

    group_q90: dict[str, torch.Tensor] = {}
    group_mean: dict[str, torch.Tensor] = {}
    for group, indices in group_indices.items():
        aligned = normalized[indices] * polarity
        group_q90[group] = torch.quantile(aligned, 0.90, dim=0)
        group_mean[group] = aligned.mean(dim=0)
    group_names = list(group_q90)
    q90_stack = torch.stack([group_q90[group] for group in group_names])
    hardest_q90, hardest_group_index = q90_stack.max(dim=0)
    margin = positive_q10 - hardest_q90
    association_factor = associated_mean.clamp_min(0)
    score = (
        margin
        * positive_mean.clamp_min(0)
        * association_factor
        * consistency
        * associated_consistency
    )
    eligible = (
        (consistency >= 0.80)
        & (positive_q10 > 0)
        & (associated_consistency >= 0.50)
        & (associated_mean > 0)
    )
    score = torch.where(eligible, score, torch.full_like(score, -torch.inf))

    flat_score = score.reshape(-1)
    top_count = min(args.top_k, int(torch.isfinite(flat_score).sum()))
    top_indices = torch.topk(flat_score, top_count).indices
    rows: list[dict[str, object]] = []
    for rank, flat_index in enumerate(top_indices.tolist(), start=1):
        layer = flat_index // neuron_count
        neuron = flat_index % neuron_count
        hardest_group = group_names[int(hardest_group_index[layer, neuron])]
        row: dict[str, object] = {
            "rank": rank,
            "layer": layer,
            "neuron": neuron,
            "polarity": "positive" if polarity[layer, neuron] > 0 else "negative",
            "score": float(score[layer, neuron]),
            "positive_raw_mean": float(activations[positive_indices, layer, neuron].mean()),
            "positive_rms_mean": float(positive_mean[layer, neuron]),
            "positive_rms_q10": float(positive_q10[layer, neuron]),
            "positive_sign_consistency": float(consistency[layer, neuron]),
            "associated_rms_mean": float(associated_mean[layer, neuron]),
            "associated_rms_q25": float(associated_q25[layer, neuron]),
            "associated_sign_consistency": float(associated_consistency[layer, neuron]),
            "hardest_false_positive_group": hardest_group,
            "hardest_false_positive_rms_q90": float(hardest_q90[layer, neuron]),
            "rms_margin": float(margin[layer, neuron]),
        }
        for group in group_names:
            row[f"{group}_rms_mean"] = float(group_mean[group][layer, neuron])
            row[f"{group}_rms_q90"] = float(group_q90[group][layer, neuron])
        rows.append(row)
    write_csv(args.output_dir / "ranked_candidates.csv", rows)
    torch.save(layer_rms, args.output_dir / "layer_rms.pt")

    table_lines = [
        "| Rank | Neuron | Polarity | MJ raw mean | MJ RMS mean | MJ q10 | MJ consistency | Associated mean | Associated q25 | Associated consistency | Hardest false-positive group | FP q90 | Margin |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |",
    ]
    for row in rows[:50]:
        table_lines.append(
            "| {rank} | L{layer}:N{neuron} | {polarity} | {positive_raw_mean:+.4f} | "
            "{positive_rms_mean:.2f} | {positive_rms_q10:.2f} | "
            "{positive_sign_consistency:.0%} | {associated_rms_mean:.2f} | "
            "{associated_rms_q25:.2f} | {associated_sign_consistency:.0%} | "
            "{hardest_false_positive_group} | "
            "{hardest_false_positive_rms_q90:.2f} | {rms_margin:.2f} |".format(**row)
        )

    group_counts = "\n".join(
        f"- `{group}`: {len(indices)}"
        for group, indices in {
            "mj_positive": positive_indices,
            "mj_associated": associated_indices,
            **group_indices,
        }.items()
    )
    report = f"""# Adversarial Michael Jordan neuron search

Model: **{args.model}**  
Activation: **input to `mlp.down_proj` (post-SwiGLU)**  
Tensor shape: **{tuple(activations.shape)}**

## Prompt groups

{group_counts}

## Ranking method

Raw activations are divided by the RMS over all prompts and neurons within each
layer. Each neuron's polarity is chosen from its mean response across the
explicit Michael Jordan positives. The score rewards a large, sign-consistent
response to those core positives; a same-polarity response to associated
references such as Air Jordan, His Airness, Jumpman, No. 23, Space Jam, and The
Last Dance; and a positive margin between the **10th percentile** core response
and the **largest 90th-percentile response of any strict false-positive
group**. This is deliberately conservative: a candidate must generalize beyond
the literal name while surviving every known confound family.

This ranking is an adversarial prompt-set result, not proof of monosemanticity.
The highest candidates still require corpus profiling and intervention tests.

## Top candidates

{chr(10).join(table_lines)}

## Files

- `activations.pt`: complete raw float32 activation tensor
- `metadata.json`: prompts, groups, target positions, and target tokens
- `ranked_candidates.csv`: top {top_count} candidates with all group statistics
- `layer_rms.pt`: layer scales used for comparison
"""
    (args.output_dir / "report.md").write_text(report, encoding="utf-8")

    print("\nTop adversarially selective candidates:")
    for row in rows[:20]:
        print(
            f"L{row['layer']}:N{row['neuron']} {row['polarity']:<8} "
            f"MJ={row['positive_raw_mean']:+.4f} q10={row['positive_rms_q10']:.2f} "
            f"assoc={row['associated_rms_mean']:.2f}/{row['associated_sign_consistency']:.0%} "
            f"FP={row['hardest_false_positive_group']}:{row['hardest_false_positive_rms_q90']:.2f} "
            f"margin={row['rms_margin']:.2f}"
        )
    print(f"\nSaved: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
