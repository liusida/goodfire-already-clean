from __future__ import annotations

from pathlib import Path

import probe_l10_n7404 as probe


def citation(venue: str) -> str:
    return (
        "References\nSmith J. 2004. Cognitive control and substance use. "
        f"{venue} 20:132-143.\nGoogle Scholar | Crossref | Medline | ISI"
    )


def case(
    test_id: str,
    group: str,
    purpose: str,
    text: str,
) -> dict[str, str]:
    return {"id": test_id, "group": group, "purpose": purpose, "text": text}


TEST_PROMPTS = [
    case("T01", "prev-sci", "Observed abbreviated journal.", citation("Prev Sci")),
    case("T02", "prev-sci", "Fully expanded journal name.", citation("Preventive Science")),
    case("T03", "prev-sci", "Expanded first word, abbreviated second.", citation("Preventive Sci")),
    case("T04", "prev-sci", "Abbreviated first word, expanded second.", citation("Prev Science")),
    case("T05", "prev-sci", "Same abbreviation outside a citation.", "The journal is called Prev Sci."),
    case("T06", "neuropsychol-rev", "Observed abbreviated journal.", citation("Neuropsychol Rev")),
    case("T07", "neuropsychol-rev", "Fully expanded journal name.", citation("Neuropsychology Review")),
    case("T08", "neuropsychol-rev", "Abbreviated first word only.", citation("Neuropsychol Review")),
    case("T09", "neuropsychol-rev", "Abbreviated second word only.", citation("Neuropsychology Rev")),
    case("T10", "lexical-control", "Unrelated phrase ending in review.", citation("Annual Book Review")),
    case("T11", "gambl-stud", "Observed abbreviated journal.", citation("J Gambl Stud")),
    case("T12", "gambl-stud", "Fully expanded journal name.", citation("Journal of Gambling Studies")),
    case("T13", "gambl-stud", "Expanded phrase without Journal of.", citation("Gambling Studies")),
    case("T14", "gambl-stud", "Incomplete abbreviated journal name.", citation("J Gambl")),
    case("T15", "psychopharmacology", "Observed title with Berlin qualifier.", citation("Psychopharmacology (Berl)")),
    case("T16", "psychopharmacology", "Journal title without qualifier.", citation("Psychopharmacology")),
    case("T17", "psychopharmacology", "Different journal with a similar name.", citation("Journal of Psychopharmacology")),
    case("T18", "psychopharmacology", "Lowercase journal title.", citation("psychopharmacology (berl)")),
    case(
        "T19",
        "psychopharmacology",
        "Same word in ordinary prose.",
        "Psychopharmacology examines how drugs affect behavior and cognition.",
    ),
    case("T20", "cognit-ther-res", "Observed abbreviated journal.", citation("Cognit Ther Res")),
    case("T21", "cognit-ther-res", "Fully expanded journal name.", citation("Cognitive Therapy and Research")),
    case("T22", "cognit-ther-res", "Expanded name without conjunction.", citation("Cognitive Therapy Research")),
    case("T23", "journal-control", "Unrelated abbreviated journal ending in Res.", citation("Brain Res")),
    case("T24", "journal-control", "Broad scientific journal.", citation("Science")),
    case("T25", "journal-control", "Broad scientific journal.", citation("Nature")),
    case("T26", "journal-control", "Unrelated abbreviated neuroscience journal.", citation("J Neurosci")),
    case("T27", "journal-control", "Fabricated journal with matching abbreviation form.", citation("J Widget Stud")),
    case(
        "T28",
        "lexical-control",
        "Final-token words in ordinary prose.",
        "Previous science review studies describe research in pharmacology.",
    ),
    case(
        "T29",
        "combined-positive",
        "All observed journal names in a compact venue list.",
        "Journals: Prev Sci; Neuropsychol Rev; J Gambl Stud; "
        "Psychopharmacology (Berl); Cognit Ther Res.",
    ),
    case(
        "T30",
        "format-control",
        "Citation metadata without a venue name.",
        "References\nSmith J. 2004. Cognitive control and substance use. "
        "doi:10.1000/example.2004.001.",
    ),
]


probe.NEURON = 1376
probe.RANK_TAIL = "negative"
probe.OUTPUT_DIR = Path(
    "outputs/mlp_measurement/Qwen3.5-9B-Base/l10_n1376_probe"
)
probe.HYPOTHESIS = (
    "L10:N1376 activates in the negative direction for a small set of specific "
    "behavioral-science title families: Prev Sci/Preventive Science, "
    "Neuropsychol Rev/Neuropsychology Review, J Gambl Stud/Journal of Gambling "
    "Studies, and Cognit Ther Res/Cognitive Therapy and Research. Generic "
    "citation formatting, suffix words, fake venues, and unrelated journals do "
    "not reproduce the effect. Psychopharmacology is an exception that also "
    "activates in ordinary prose, so the feature is strongly journal-associated "
    "but not exclusively a journal-name feature."
)
probe.TEST_PROMPTS = TEST_PROMPTS


if __name__ == "__main__":
    probe.main()
