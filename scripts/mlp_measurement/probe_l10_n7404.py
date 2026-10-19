from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path

import torch
from gb10_load_llm import load_model_to_cuda
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "Qwen/Qwen3.5-9B-Base"
LAYER = 10
NEURON = 7404
RANK_TAIL = "positive"
OUTPUT_DIR = Path("outputs/mlp_measurement/Qwen3.5-9B-Base/l10_n7404_probe")

HYPOTHESIS = (
    "L10:N7404 is a selective detector for the identity of a small set of "
    "scientific journals, not for journal names or the suffixes Evolution and "
    "Brain Res in general. Its Molecular Phylogenetics and Evolution response "
    "is robust to abbreviation, missing or changed conjunctions, punctuation, "
    "line breaks, markup, casing, and full-width Unicode, suggesting a fuzzy "
    "journal-identity feature rather than an exact token-sequence feature. A "
    "bare MPE acronym or DOI-only citation largely avoids the response."
)

TEST_PROMPTS = [
    {
        "id": "T01",
        "group": "positive-template",
        "purpose": "Near-replication of the strongest observed citation.",
        "text": (
            "References\nSmith, J. and Lee, R. 2000. Phylogenetic signal in "
            "pigeons and doves. Molecular Phylogenetics and Evolution 14, "
            "141-151."
        ),
    },
    {
        "id": "T02",
        "group": "positive-template",
        "purpose": "Near-replication of the Brain Res citation pattern.",
        "text": (
            "Miller, A. et al. (2004) A topography of executive functions "
            "revealed by functional magnetic resonance imaging. Brain Res "
            "Cogn Brain Res 20: 132-143."
        ),
    },
    {
        "id": "T03",
        "group": "positive-template",
        "purpose": "Near-replication of the Psychophysiol citation pattern.",
        "text": (
            "Jones, P. (2005) Attention deficit and impulsivity: selecting, "
            "shifting, and stopping. Int J Psychophysiol 58: 59-70."
        ),
    },
    {
        "id": "T04",
        "group": "bibliography",
        "purpose": "Dense references with journal, volume, pages, and DOI.",
        "text": (
            "References\nChen L, Patel S. Neural control of inhibition. "
            "J Neurosci. 2019;39(12):2201-2214. doi:10.1000/jn.2019.42.\n"
            "Garcia M. Comparative cognition in birds. Anim Behav. "
            "2021;174:88-102."
        ),
    },
    {
        "id": "T05",
        "group": "bibliography",
        "purpose": "Reference boilerplate seen near the original maxima.",
        "text": (
            "Google Scholar | Crossref | Medline | ISI\nFernandez, D. 2012. "
            "Response inhibition and attention. Brain Research 1442, 44-57."
        ),
    },
    {
        "id": "T06",
        "group": "bibliography",
        "purpose": "APA-like citation without abbreviated journal name.",
        "text": (
            "Williams, K. (2018). Decision making under uncertainty. "
            "Journal of Experimental Psychology, 147(6), 812-829."
        ),
    },
    {
        "id": "T07",
        "group": "bibliography",
        "purpose": "Fabricated citation preserving scholarly surface form.",
        "text": (
            "Zorbax, Q. and Plim, V. (2097) Transdimensional memory transfer. "
            "Int J Synthetic Cognition 91: 7-19."
        ),
    },
    {
        "id": "T08",
        "group": "lexical-control",
        "purpose": "Evolution in ordinary semantic prose, no citation form.",
        "text": (
            "Evolution explains how populations change over generations through "
            "mutation, selection, genetic drift, and migration."
        ),
    },
    {
        "id": "T09",
        "group": "lexical-control",
        "purpose": "Exact journal-title words used as an ordinary title.",
        "text": (
            "Our seminar today is called Molecular Phylogenetics and Evolution, "
            "and everyone is welcome to discuss the topic informally."
        ),
    },
    {
        "id": "T10",
        "group": "format-control",
        "purpose": "Citation-like numbers and pages without scholarship.",
        "text": (
            "Invoice 2004-20 lists items 132-143. Order 58 contains boxes "
            "59-70 and should arrive on 14 March."
        ),
    },
    {
        "id": "T11",
        "group": "format-control",
        "purpose": "Scholarly prose with an inline author-year citation.",
        "text": (
            "Response inhibition is often impaired in ADHD, particularly when "
            "the stop signal is delayed (Miller et al., 2004)."
        ),
    },
    {
        "id": "T12",
        "group": "format-control",
        "purpose": "Bibliography structure with non-scientific books.",
        "text": (
            "Bibliography\nAusten, Jane. Pride and Prejudice. London: T. "
            "Egerton, 1813.\nBorges, Jorge Luis. Ficciones. Buenos Aires: "
            "Sur, 1944."
        ),
    },
    {
        "id": "T13",
        "group": "token-edge",
        "purpose": "Repeated standalone Res-like fragments without citations.",
        "text": "Res Res Res. Research resources resolve residual resistance.",
    },
    {
        "id": "T14",
        "group": "token-edge",
        "purpose": "Words likely to expose iol-like tokenizer fragments.",
        "text": (
            "physiology psychophysiology biology radiobiology sociolinguistics "
            "and aerosolization"
        ),
    },
    {
        "id": "T15",
        "group": "token-edge",
        "purpose": "Journal abbreviations without surrounding citation syntax.",
        "text": "Brain Res. Int J Psychophysiol. Mol Phylogenet Evol. J Neurosci.",
    },
    {
        "id": "T16",
        "group": "token-edge",
        "purpose": "Citation compressed without whitespace.",
        "text": (
            "SmithJ(2004)ExecutiveFunctions.BrainResCognBrainRes20:132-143;"
            "JonesP(2005)Attention.IntJPsychophysiol58:59-70."
        ),
    },
    {
        "id": "T17",
        "group": "negative-control",
        "purpose": "Ordinary narrative with no scholarly cues.",
        "text": (
            "The children packed sandwiches, walked to the river, skipped "
            "stones, and returned home before sunset."
        ),
    },
    {
        "id": "T18",
        "group": "negative-control",
        "purpose": "Technical source code with words resembling references.",
        "text": (
            "class ResourceResolver:\n    def resolve(self, result):\n"
            "        return self.cache.get(result.id)"
        ),
    },
    {
        "id": "T19",
        "group": "multilingual-edge",
        "purpose": "Chinese prose containing a conventional journal citation.",
        "text": (
            "参考文献：王明、李华（2020）注意控制与反应抑制。《心理学报》"
            "52卷，第101—118页。"
        ),
    },
    {
        "id": "T20",
        "group": "adversarial-edge",
        "purpose": "Explicitly denies being a citation while copying its form.",
        "text": (
            "This is not a bibliography and not a scientific citation: "
            "Brain Res Cogn Brain Res 20: 132-143; Int J Psychophysiol 58: 59-70."
        ),
    },
    {
        "id": "T21",
        "group": "factorial-context",
        "purpose": "Exact Evolution venue under a minimal References heading.",
        "text": "References\nMolecular Phylogenetics and Evolution",
    },
    {
        "id": "T22",
        "group": "factorial-context",
        "purpose": "Exact Evolution venue at the beginning with no left context.",
        "text": "Molecular Phylogenetics and Evolution",
    },
    {
        "id": "T23",
        "group": "factorial-lexical",
        "purpose": "Replace and with an ampersand inside citation context.",
        "text": (
            "References\nSmith, J. 2000. Avian relationships. "
            "Molecular Phylogenetics & Evolution 14, 141-151."
        ),
    },
    {
        "id": "T24",
        "group": "factorial-lexical",
        "purpose": "Reorder the exact venue words while preserving citation form.",
        "text": (
            "References\nSmith, J. 2000. Avian relationships. "
            "Molecular Evolution and Phylogenetics 14, 141-151."
        ),
    },
    {
        "id": "T25",
        "group": "factorial-lexical",
        "purpose": "Pluralize the diagnostic final venue word.",
        "text": (
            "References\nSmith, J. 2000. Avian relationships. "
            "Molecular Phylogenetics and Evolutions 14, 141-151."
        ),
    },
    {
        "id": "T26",
        "group": "factorial-case",
        "purpose": "Lowercase the exact Evolution venue in citation context.",
        "text": (
            "References\nSmith, J. 2000. Avian relationships. "
            "molecular phylogenetics and evolution 14, 141-151."
        ),
    },
    {
        "id": "T27",
        "group": "factorial-case",
        "purpose": "Uppercase the exact Evolution venue in citation context.",
        "text": (
            "REFERENCES\nSMITH, J. 2000. AVIAN RELATIONSHIPS. "
            "MOLECULAR PHYLOGENETICS AND EVOLUTION 14, 141-151."
        ),
    },
    {
        "id": "T28",
        "group": "factorial-context",
        "purpose": "Exact venue in a compact journal list but not a citation.",
        "text": (
            "Journals: Molecular Phylogenetics and Evolution; Animal Behaviour; "
            "The Auk; Systematic Biology."
        ),
    },
    {
        "id": "T29",
        "group": "factorial-brain-res",
        "purpose": "Exact Brain Res venue with a conventional reference heading.",
        "text": (
            "References\nMiller A. 2004. Executive function and inhibition. "
            "Brain Res Cogn Brain Res 20:132-143."
        ),
    },
    {
        "id": "T30",
        "group": "factorial-brain-res",
        "purpose": "Truncate the venue after its first Brain Res segment.",
        "text": (
            "References\nMiller A. 2004. Executive function and inhibition. "
            "Brain Res 20:132-143."
        ),
    },
    {
        "id": "T31",
        "group": "factorial-brain-res",
        "purpose": "Use only the Cogn Brain Res suffix of the venue.",
        "text": (
            "References\nMiller A. 2004. Executive function and inhibition. "
            "Cogn Brain Res 20:132-143."
        ),
    },
    {
        "id": "T32",
        "group": "factorial-brain-res",
        "purpose": "Shuffle the abbreviated venue components.",
        "text": (
            "References\nMiller A. 2004. Executive function and inhibition. "
            "Brain Cogn Res Brain 20:132-143."
        ),
    },
    {
        "id": "T33",
        "group": "factorial-brain-res",
        "purpose": "Expand every abbreviation in the Brain Res venue.",
        "text": (
            "References\nMiller A. 2004. Executive function and inhibition. "
            "Brain Research Cognitive Brain Research 20:132-143."
        ),
    },
    {
        "id": "T34",
        "group": "factorial-psychophysiol",
        "purpose": "Exact Psychophysiol venue under a References heading.",
        "text": (
            "References\nJones P. 2005. Attention deficit and impulsivity. "
            "Int J Psychophysiol 58:59-70."
        ),
    },
    {
        "id": "T35",
        "group": "factorial-psychophysiol",
        "purpose": "Diagnostic abbreviation alone without citation context.",
        "text": "Psychophysiol",
    },
    {
        "id": "T36",
        "group": "factorial-psychophysiol",
        "purpose": "Spell out the abbreviated Psychophysiol venue.",
        "text": (
            "References\nJones P. 2005. Attention deficit and impulsivity. "
            "International Journal of Psychophysiology 58:59-70."
        ),
    },
    {
        "id": "T37",
        "group": "factorial-psychophysiol",
        "purpose": "Introduce a typo in the diagnostic venue word.",
        "text": (
            "References\nJones P. 2005. Attention deficit and impulsivity. "
            "Int J Psychophysio1 58:59-70."
        ),
    },
    {
        "id": "T38",
        "group": "related-real-venue",
        "purpose": "Related real journal sharing Molecular and Evolution.",
        "text": (
            "References\nKim S. 2019. Selection across populations. "
            "Molecular Biology and Evolution 36:101-115."
        ),
    },
    {
        "id": "T39",
        "group": "related-real-venue",
        "purpose": "Real journal whose complete title is Evolution.",
        "text": (
            "References\nTaylor B. 2018. Adaptation in island birds. "
            "Evolution 72:400-412."
        ),
    },
    {
        "id": "T40",
        "group": "related-real-venue",
        "purpose": "Related real venue with evolutionary meaning but different title.",
        "text": (
            "References\nPatel R. 2021. Speciation and genomic divergence. "
            "Journal of Evolutionary Biology 34:88-102."
        ),
    },
    {
        "id": "T41",
        "group": "journal-screen",
        "purpose": "Controlled venue substitution: Molecular Ecology.",
        "text": (
            "References\nSmith J. 2004. Comparative analysis of biological "
            "variation. Molecular Ecology 20:132-143."
        ),
    },
    {
        "id": "T42",
        "group": "journal-screen",
        "purpose": "Controlled venue substitution: Journal of Molecular Evolution.",
        "text": (
            "References\nSmith J. 2004. Comparative analysis of biological "
            "variation. Journal of Molecular Evolution 20:132-143."
        ),
    },
    {
        "id": "T43",
        "group": "journal-screen",
        "purpose": "Controlled venue substitution: Systematic Biology.",
        "text": (
            "References\nSmith J. 2004. Comparative analysis of biological "
            "variation. Systematic Biology 20:132-143."
        ),
    },
    {
        "id": "T44",
        "group": "journal-screen",
        "purpose": "Controlled venue substitution: Evolutionary Ecology.",
        "text": (
            "References\nSmith J. 2004. Comparative analysis of biological "
            "variation. Evolutionary Ecology 20:132-143."
        ),
    },
    {
        "id": "T45",
        "group": "journal-screen",
        "purpose": "Controlled venue substitution: The Auk.",
        "text": (
            "References\nSmith J. 2004. Comparative analysis of biological "
            "variation. The Auk 20:132-143."
        ),
    },
    {
        "id": "T46",
        "group": "journal-screen",
        "purpose": "Controlled venue substitution: Animal Behaviour.",
        "text": (
            "References\nSmith J. 2004. Comparative analysis of biological "
            "variation. Animal Behaviour 20:132-143."
        ),
    },
    {
        "id": "T47",
        "group": "journal-screen",
        "purpose": "Controlled venue substitution: NeuroImage.",
        "text": (
            "References\nSmith J. 2004. Comparative analysis of biological "
            "variation. NeuroImage 20:132-143."
        ),
    },
    {
        "id": "T48",
        "group": "journal-screen",
        "purpose": "Controlled venue substitution: Journal of Neuroscience.",
        "text": (
            "References\nSmith J. 2004. Comparative analysis of biological "
            "variation. Journal of Neuroscience 20:132-143."
        ),
    },
    {
        "id": "T49",
        "group": "journal-screen",
        "purpose": "Controlled venue substitution: Cognitive Neuroscience.",
        "text": (
            "References\nSmith J. 2004. Comparative analysis of biological "
            "variation. Cognitive Neuroscience 20:132-143."
        ),
    },
    {
        "id": "T50",
        "group": "journal-screen",
        "purpose": "Controlled venue substitution: Nature.",
        "text": (
            "References\nSmith J. 2004. Comparative analysis of biological "
            "variation. Nature 20:132-143."
        ),
    },
    {
        "id": "T51",
        "group": "evolution-journal-screen",
        "purpose": "Controlled Evolution suffix: Molecular Biology and Evolution.",
        "text": (
            "References\nSmith J. 2004. Comparative analysis of biological "
            "variation. Molecular Biology and Evolution 20:132-143."
        ),
    },
    {
        "id": "T52",
        "group": "evolution-journal-screen",
        "purpose": "Controlled Evolution suffix: Journal of Human Evolution.",
        "text": (
            "References\nSmith J. 2004. Comparative analysis of biological "
            "variation. Journal of Human Evolution 20:132-143."
        ),
    },
    {
        "id": "T53",
        "group": "evolution-journal-screen",
        "purpose": "Controlled Evolution suffix: Genome Biology and Evolution.",
        "text": (
            "References\nSmith J. 2004. Comparative analysis of biological "
            "variation. Genome Biology and Evolution 20:132-143."
        ),
    },
    {
        "id": "T54",
        "group": "evolution-journal-screen",
        "purpose": "Controlled Evolution suffix: Ecology and Evolution.",
        "text": (
            "References\nSmith J. 2004. Comparative analysis of biological "
            "variation. Ecology and Evolution 20:132-143."
        ),
    },
    {
        "id": "T55",
        "group": "evolution-journal-screen",
        "purpose": "Controlled Evolution suffix: Trends in Ecology & Evolution.",
        "text": (
            "References\nSmith J. 2004. Comparative analysis of biological "
            "variation. Trends in Ecology & Evolution 20:132-143."
        ),
    },
    {
        "id": "T56",
        "group": "brain-res-journal-screen",
        "purpose": "Controlled Brain Res suffix: Dev Brain Res.",
        "text": (
            "References\nSmith J. 2004. Comparative analysis of biological "
            "variation. Dev Brain Res 20:132-143."
        ),
    },
    {
        "id": "T57",
        "group": "brain-res-journal-screen",
        "purpose": "Controlled Brain Res suffix: Behav Brain Res.",
        "text": (
            "References\nSmith J. 2004. Comparative analysis of biological "
            "variation. Behav Brain Res 20:132-143."
        ),
    },
    {
        "id": "T58",
        "group": "brain-res-journal-screen",
        "purpose": "Controlled Brain Res suffix: Exp Brain Res.",
        "text": (
            "References\nSmith J. 2004. Comparative analysis of biological "
            "variation. Exp Brain Res 20:132-143."
        ),
    },
    {
        "id": "T59",
        "group": "brain-res-journal-screen",
        "purpose": "Controlled Brain Res suffix: Mol Brain Res.",
        "text": (
            "References\nSmith J. 2004. Comparative analysis of biological "
            "variation. Mol Brain Res 20:132-143."
        ),
    },
    {
        "id": "T60",
        "group": "brain-res-journal-screen",
        "purpose": "Controlled Brain Res suffix: Cogn Brain Res.",
        "text": (
            "References\nSmith J. 2004. Comparative analysis of biological "
            "variation. Cogn Brain Res 20:132-143."
        ),
    },
    {
        "id": "T61",
        "group": "mpe-form-evasion",
        "purpose": "Standard abbreviated journal title: Mol Phylogenet Evol.",
        "text": (
            "References\nSmith J. 2004. Comparative analysis of biological "
            "variation. Mol Phylogenet Evol 20:132-143."
        ),
    },
    {
        "id": "T62",
        "group": "mpe-form-evasion",
        "purpose": "Initialism only: MPE.",
        "text": (
            "References\nSmith J. 2004. Comparative analysis of biological "
            "variation. MPE 20:132-143."
        ),
    },
    {
        "id": "T63",
        "group": "mpe-form-evasion",
        "purpose": "Omit the conjunction from the full title.",
        "text": (
            "References\nSmith J. 2004. Comparative analysis of biological "
            "variation. Molecular Phylogenetics Evolution 20:132-143."
        ),
    },
    {
        "id": "T64",
        "group": "mpe-form-evasion",
        "purpose": "Replace the conjunction with a slash.",
        "text": (
            "References\nSmith J. 2004. Comparative analysis of biological "
            "variation. Molecular Phylogenetics/Evolution 20:132-143."
        ),
    },
    {
        "id": "T65",
        "group": "mpe-form-evasion",
        "purpose": "Replace the conjunction with a comma.",
        "text": (
            "References\nSmith J. 2004. Comparative analysis of biological "
            "variation. Molecular Phylogenetics, Evolution 20:132-143."
        ),
    },
    {
        "id": "T66",
        "group": "mpe-form-evasion",
        "purpose": "Hyphenate all words in the title.",
        "text": (
            "References\nSmith J. 2004. Comparative analysis of biological "
            "variation. Molecular-Phylogenetics-and-Evolution 20:132-143."
        ),
    },
    {
        "id": "T67",
        "group": "mpe-form-evasion",
        "purpose": "Insert a line break immediately before Evolution.",
        "text": (
            "References\nSmith J. 2004. Comparative analysis of biological "
            "variation. Molecular Phylogenetics and\nEvolution 20:132-143."
        ),
    },
    {
        "id": "T68",
        "group": "mpe-form-evasion",
        "purpose": "Parenthesize the final title word.",
        "text": (
            "References\nSmith J. 2004. Comparative analysis of biological "
            "variation. Molecular Phylogenetics and (Evolution) 20:132-143."
        ),
    },
    {
        "id": "T69",
        "group": "mpe-form-evasion",
        "purpose": "Put the intact title in quotation marks.",
        "text": (
            "References\nSmith J. 2004. Comparative analysis of biological "
            "variation. \"Molecular Phylogenetics and Evolution\" 20:132-143."
        ),
    },
    {
        "id": "T70",
        "group": "mpe-form-evasion",
        "purpose": "Wrap the intact title in Markdown italics.",
        "text": (
            "References\nSmith J. 2004. Comparative analysis of biological "
            "variation. *Molecular Phylogenetics and Evolution* 20:132-143."
        ),
    },
    {
        "id": "T71",
        "group": "mpe-form-evasion",
        "purpose": "Insert HTML emphasis tags around Evolution.",
        "text": (
            "References\nSmith J. 2004. Comparative analysis of biological "
            "variation. Molecular Phylogenetics and <em>Evolution</em> 20:132-143."
        ),
    },
    {
        "id": "T72",
        "group": "mpe-form-evasion",
        "purpose": "Insert the word journal before the final title word.",
        "text": (
            "References\nSmith J. 2004. Comparative analysis of biological "
            "variation. Molecular Phylogenetics and journal Evolution 20:132-143."
        ),
    },
    {
        "id": "T73",
        "group": "mpe-form-evasion",
        "purpose": "Write the ampersand as the literal HTML entity &amp;.",
        "text": (
            "References\nSmith J. 2004. Comparative analysis of biological "
            "variation. Molecular Phylogenetics &amp; Evolution 20:132-143."
        ),
    },
    {
        "id": "T74",
        "group": "mpe-form-evasion",
        "purpose": "Use full-width Unicode letters for Evolution.",
        "text": (
            "References\nSmith J. 2004. Comparative analysis of biological "
            "variation. Molecular Phylogenetics and Ｅｖｏｌｕｔｉｏｎ 20:132-143."
        ),
    },
    {
        "id": "T75",
        "group": "mpe-form-evasion",
        "purpose": "Identify the same article by DOI without naming the journal.",
        "text": (
            "References\nSmith J. 2004. Comparative analysis of biological "
            "variation. doi:10.1016/j.ympev.2004.01.001."
        ),
    },
]


class ActivationCaptured(Exception):
    """Stop the forward pass immediately after the target activation is read."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            f"Probe {MODEL_ID} Layer {LAYER} neuron {NEURON} "
            f"on {len(TEST_PROMPTS)} contrasts."
        )
    )
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
    )
    return parser.parse_args()


def token_text(tokenizer: AutoTokenizer, token_id: int) -> str:
    text = tokenizer.decode(
        [token_id],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    return text.replace("\n", "\\n") or "∅"


def measure(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    *,
    batch_size: int,
    max_length: int,
) -> list[dict[str, object]]:
    captured: torch.Tensor | None = None

    def hook(_module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        nonlocal captured
        captured = inputs[0][..., NEURON].detach().float().cpu()
        raise ActivationCaptured

    handle = model.model.layers[LAYER].mlp.down_proj.register_forward_pre_hook(hook)
    results: list[dict[str, object]] = []
    try:
        progress = tqdm(
            range(0, len(TEST_PROMPTS), batch_size),
            desc=f"L{LAYER}:N{NEURON}",
            unit="batch",
        )
        for start in progress:
            batch = TEST_PROMPTS[start : start + batch_size]
            encoded = tokenizer(
                [item["text"] for item in batch],
                add_special_tokens=False,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            captured = None
            try:
                with torch.inference_mode():
                    model.model(
                        input_ids=encoded.input_ids.to(model.device),
                        attention_mask=encoded.attention_mask.to(model.device),
                        use_cache=False,
                    )
            except ActivationCaptured:
                pass
            if captured is None:
                raise RuntimeError("Target MLP hook did not fire")

            for row, item in enumerate(batch):
                length = int(encoded.attention_mask[row].sum())
                token_ids = encoded.input_ids[row, :length].tolist()
                activations = captured[row, :length].tolist()
                order = sorted(
                    range(length), key=lambda index: activations[index], reverse=True
                )
                top_positions = order[: min(5, length)]
                bottom_positions = list(reversed(order[-min(5, length) :]))
                results.append(
                    {
                        **item,
                        "truncated": length == max_length,
                        "token_ids": token_ids,
                        "tokens": [token_text(tokenizer, token_id) for token_id in token_ids],
                        "activations": activations,
                        "maximum": max(activations),
                        "minimum": min(activations),
                        "mean": sum(activations) / length,
                        "maximum_position": order[0],
                        "minimum_position": order[-1],
                        "top_tokens": [
                            {
                                "position": position,
                                "token_id": token_ids[position],
                                "token": token_text(tokenizer, token_ids[position]),
                                "activation": activations[position],
                            }
                            for position in top_positions
                        ],
                        "bottom_tokens": [
                            {
                                "position": position,
                                "token_id": token_ids[position],
                                "token": token_text(tokenizer, token_ids[position]),
                                "activation": activations[position],
                            }
                            for position in bottom_positions
                        ],
                    }
                )
    finally:
        handle.remove()
    return results


def color(value: float, scale: float) -> str:
    strength = min(abs(value) / scale, 1.0) if scale > 0 else 0.0
    alpha = 0.08 + 0.72 * strength
    rgb = "225, 88, 54" if value >= 0 else "60, 110, 190"
    return f"rgba({rgb}, {alpha:.3f})"


def write_html(path: Path, results: list[dict[str, object]]) -> None:
    global_scale = max(
        abs(float(value))
        for result in results
        for value in result["activations"]
    )
    cards: list[str] = []
    if RANK_TAIL == "negative":
        ranked_results = sorted(results, key=lambda result: float(result["minimum"]))
    else:
        ranked_results = sorted(
            results,
            key=lambda result: float(result["maximum"]),
            reverse=True,
        )
    for result in ranked_results:
        tokens = "".join(
            (
                f'<span class="token" style="background:{color(float(value), global_scale)}" '
                f'title="position {position} · token {token_id} · activation {float(value):+.6f}">'
                f"{html.escape(str(token))}</span>"
            )
            for position, (token_id, token, value) in enumerate(
                zip(result["token_ids"], result["tokens"], result["activations"])
            )
        )
        top_rows = "".join(
            "<tr>"
            f"<td>{entry['position']}</td>"
            f"<td><code>{html.escape(str(entry['token']))}</code></td>"
            f"<td>{entry['token_id']}</td>"
            f"<td>{float(entry['activation']):+.6f}</td>"
            "</tr>"
            for entry in result["top_tokens"]
        )
        bottom_rows = "".join(
            "<tr>"
            f"<td>{entry['position']}</td>"
            f"<td><code>{html.escape(str(entry['token']))}</code></td>"
            f"<td>{entry['token_id']}</td>"
            f"<td>{float(entry['activation']):+.6f}</td>"
            "</tr>"
            for entry in result["bottom_tokens"]
        )
        cards.append(
            f"""
            <section class="card">
              <header><b>{result['id']}</b><span>{html.escape(str(result['group']))}</span>
                <strong>min {float(result['minimum']):+.4f} · max {float(result['maximum']):+.4f}</strong></header>
              <p class="purpose">{html.escape(str(result['purpose']))}</p>
              <div class="trace">{tokens}</div>
              <details><summary>Top five token positions</summary>
                <table><thead><tr><th>Position</th><th>Token</th><th>ID</th><th>Activation</th></tr></thead>
                <tbody>{top_rows}</tbody></table>
              </details>
              <details><summary>Bottom five token positions</summary>
                <table><thead><tr><th>Position</th><th>Token</th><th>ID</th><th>Activation</th></tr></thead>
                <tbody>{bottom_rows}</tbody></table>
              </details>
            </section>
            """
        )
    document = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>L{LAYER}:N{NEURON} probe</title>
<style>
body{{font:15px/1.45 system-ui,sans-serif;max-width:1200px;margin:32px auto;padding:0 20px;color:#20242a}}
h1{{margin-bottom:4px}} .hypothesis{{background:#f4f6f8;padding:14px 18px;border-radius:10px}}
.legend{{color:#596575}} .card{{border:1px solid #d8dee6;border-radius:12px;margin:16px 0;overflow:hidden}}
.card header{{display:flex;gap:18px;align-items:center;background:#f7f9fb;padding:10px 14px}}
.card header strong{{margin-left:auto}} .purpose{{margin:12px 14px;color:#596575}}
.trace{{margin:12px 14px;padding:12px;background:#fbfcfd;border-radius:8px;white-space:pre-wrap;font-family:ui-monospace,monospace}}
.token{{border-radius:3px;padding:2px 1px}} details{{margin:10px 14px 14px}} table{{border-collapse:collapse}}
th,td{{padding:5px 12px;border-bottom:1px solid #e3e7ec;text-align:right}} th:nth-child(2),td:nth-child(2){{text-align:left}}
</style></head><body>
<h1>L{LAYER}:N{NEURON} targeted probe</h1>
<p class="legend">Post-SwiGLU activation · red = positive · blue = negative · one global color scale (±{global_scale:.4f})</p>
<div class="hypothesis"><b>Hypothesis:</b> {html.escape(HYPOTHESIS)}</div>
{''.join(cards)}
</body></html>"""
    path.write_text(document, encoding="utf-8")


def save_results(
    output_dir: Path,
    results: list[dict[str, object]],
    *,
    model_id: str,
    model_revision: str | None,
    touched: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_id": model_id,
        "model_revision": model_revision,
        "layer": LAYER,
        "neuron": NEURON,
        "activation_definition": "input to mlp.down_proj (post-SwiGLU)",
        "hypothesis": HYPOTHESIS,
        "cpu_pages_touched_gib": touched / 1024**3,
        "prompts": results,
    }
    (output_dir / "results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "id", "group", "maximum", "minimum", "mean",
                "max_token", "max_position", "min_token", "min_position",
            ]
        )
        for result in results:
            maximum_position = int(result["maximum_position"])
            minimum_position = int(result["minimum_position"])
            writer.writerow(
                [
                    result["id"],
                    result["group"],
                    result["maximum"],
                    result["minimum"],
                    result["mean"],
                    result["tokens"][maximum_position],
                    maximum_position,
                    result["tokens"][minimum_position],
                    minimum_position,
                ]
            )
    write_html(output_dir / "report.html", results)


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.max_length <= 0:
        raise ValueError("--batch-size and --max-length must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
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
    intermediate_size = int(model.model.layers[LAYER].mlp.down_proj.in_features)
    if NEURON >= intermediate_size:
        raise ValueError(
            f"N{NEURON} is outside Layer {LAYER}'s {intermediate_size} MLP neurons"
        )
    results = measure(
        model,
        tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    save_results(
        args.output_dir,
        results,
        model_id=args.model,
        model_revision=args.revision,
        touched=touched,
    )
    if RANK_TAIL == "negative":
        ranking = sorted(results, key=lambda result: float(result["minimum"]))
        value_key = "minimum"
        position_key = "minimum_position"
    else:
        ranking = sorted(results, key=lambda result: float(result["maximum"]), reverse=True)
        value_key = "maximum"
        position_key = "maximum_position"
    print(f"\n{RANK_TAIL.title()}-tail activation by prompt:")
    for result in ranking:
        position = int(result[position_key])
        print(
            f"{result['id']}  {float(result[value_key]):+9.4f}  "
            f"{result['tokens'][position]!r}  {result['group']}"
        )
    print(f"\nSaved results: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
