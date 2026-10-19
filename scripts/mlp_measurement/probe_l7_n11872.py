from __future__ import annotations

from pathlib import Path

import probe_l10_n7404 as probe


def case(test_id: str, group: str, purpose: str, text: str) -> dict[str, str]:
    return {"id": test_id, "group": group, "purpose": purpose, "text": text}


TEST_PROMPTS = [
    # Michael Jordan identity, deliberately stripped of brands and endorsements.
    case("T01", "mj-identity", "Name in a biographical statement.",
         "Michael Jordan was born in Brooklyn and raised in North Carolina."),
    case("T02", "mj-identity", "Name in a basketball achievement.",
         "Michael Jordan led Chicago to six NBA championships."),
    case("T03", "mj-identity", "Name in a game action.",
         "Michael Jordan made the decisive shot at the buzzer."),
    case("T04", "mj-identity", "Name in a retirement statement.",
         "Michael Jordan retired from professional basketball for the final time in 2003."),
    case("T05", "mj-identity", "Name in a family statement.",
         "Michael Jordan attended the ceremony with members of his family."),
    case("T06", "mj-identity", "Name without basketball context.",
         "The interviewer asked Michael Jordan about his childhood."),
    case("T07", "mj-identity", "Pronoun continuation after establishing identity.",
         "Michael Jordan entered the room. He greeted the audience and sat down."),
    case("T08", "mj-identity", "Basketball description without the literal name.",
         "The six-time Chicago champion made the famous final shot against Utah."),

    # The same person in explicitly commercial contexts.
    case("T09", "mj-commercial", "Name beside sponsorship language.",
         "Michael Jordan signed a lucrative sponsorship agreement with Nike."),
    case("T10", "mj-commercial", "Name beside endorsement language.",
         "Michael Jordan became one of the most successful celebrity endorsers in history."),
    case("T11", "mj-commercial", "Name beside advertising language.",
         "The advertising campaign featured Michael Jordan promoting athletic shoes."),
    case("T12", "mj-commercial", "Name beside brand language.",
         "The company built a global brand around Michael Jordan."),
    case("T13", "mj-commercial", "Possessive name before sponsorship.",
         "Michael Jordan's sponsorship transformed the economics of athlete endorsements."),
    case("T14", "mj-commercial", "Sponsorship before the name.",
         "Nike's landmark sponsorship of Michael Jordan changed sports marketing."),

    # Exact corpus-like trigger without Michael Jordan.
    case("T15", "sports-sponsorship", "Cricket sponsorship.",
         "The cricket board announced a major sponsorship deal for the domestic league."),
    case("T16", "sports-sponsorship", "Football sponsorship.",
         "The football club renewed its sponsorship agreement with the airline."),
    case("T17", "sports-sponsorship", "Tennis sponsorship.",
         "The tennis player signed a sponsorship contract with a watch company."),
    case("T18", "sports-sponsorship", "Basketball sponsorship, no Jordan.",
         "The basketball tournament secured a new sponsorship deal."),
    case("T19", "sports-sponsorship", "Olympic sponsorship.",
         "The Olympic committee approved the beverage company's sponsorship proposal."),
    case("T20", "sports-sponsorship", "Athlete endorsement synonym.",
         "The sprinter accepted a lucrative endorsement agreement from an apparel brand."),
    case("T21", "sports-sponsorship", "Sports marketing without sponsorship token.",
         "The famous athlete appeared in a global advertising campaign."),

    # Sponsorship outside sports tests whether the feature is generic.
    case("T22", "non-sports-sponsorship", "Museum sponsorship.",
         "The museum received corporate sponsorship for its new exhibition."),
    case("T23", "non-sports-sponsorship", "Conference sponsorship.",
         "A software company provided sponsorship for the academic conference."),
    case("T24", "non-sports-sponsorship", "Arts sponsorship.",
         "The orchestra depended on private sponsorship to fund the concert season."),
    case("T25", "non-sports-sponsorship", "Charity sponsorship.",
         "Local businesses offered sponsorship for the charity fundraiser."),
    case("T26", "non-sports-sponsorship", "Broadcast sponsorship.",
         "The program was made possible through corporate sponsorship."),
    case("T27", "non-sports-sponsorship", "Visa/legal meaning.",
         "The employee needed company sponsorship to obtain a work visa."),
    case("T28", "non-sports-sponsorship", "Child sponsorship.",
         "The charity organized a child sponsorship program for donors."),

    # Brand and sneaker concepts without the person.
    case("T29", "jordan-brand", "Air Jordan product name.",
         "Collectors waited overnight for the latest Air Jordan release."),
    case("T30", "jordan-brand", "Jumpman mark.",
         "The shoe displayed a red Jumpman logo on the heel."),
    case("T31", "jordan-brand", "Jordan Brand without athlete.",
         "Jordan Brand introduced a new basketball uniform."),
    case("T32", "jordan-brand", "Generic Nike product.",
         "Nike released a lightweight running shoe for marathon training."),
    case("T33", "jordan-brand", "Nike sponsorship without Jordan.",
         "Nike announced a sponsorship agreement with a young tennis player."),
    case("T34", "jordan-brand", "Adidas sponsorship control.",
         "Adidas announced a sponsorship agreement with a young tennis player."),

    # Matched substitutions isolate identity from sentence structure.
    case("T35", "matched-name", "Michael Jordan in neutral template.",
         "The documentary devoted an entire episode to Michael Jordan."),
    case("T36", "matched-name", "Michael Jackson substitution.",
         "The documentary devoted an entire episode to Michael Jackson."),
    case("T37", "matched-name", "Michael B. Jordan substitution.",
         "The documentary devoted an entire episode to Michael B. Jordan."),
    case("T38", "matched-name", "Scottie Pippen substitution.",
         "The documentary devoted an entire episode to Scottie Pippen."),
    case("T39", "matched-name", "Kobe Bryant substitution.",
         "The documentary devoted an entire episode to Kobe Bryant."),
    case("T40", "matched-name", "Serena Williams substitution.",
         "The documentary devoted an entire episode to Serena Williams."),

    case("T41", "matched-sponsorship", "Michael Jordan sponsorship.",
         "The company offered Michael Jordan a sponsorship deal."),
    case("T42", "matched-sponsorship", "Scottie Pippen sponsorship.",
         "The company offered Scottie Pippen a sponsorship deal."),
    case("T43", "matched-sponsorship", "Serena Williams sponsorship.",
         "The company offered Serena Williams a sponsorship deal."),
    case("T44", "matched-sponsorship", "Unknown athlete sponsorship.",
         "The company offered the unknown athlete a sponsorship deal."),
    case("T45", "matched-sponsorship", "Musician sponsorship.",
         "The company offered the famous musician a sponsorship deal."),
    case("T46", "matched-sponsorship", "No person sponsorship.",
         "The company announced a sponsorship deal."),

    # Lexical controls establish whether the exact token dominates.
    case("T47", "lexical-control", "Plural form.",
         "The organization negotiated several sponsorships during the year."),
    case("T48", "lexical-control", "Hyphenated phrase.",
         "The proposal described a co-sponsorship arrangement."),
    case("T49", "lexical-control", "Endorsement synonym.",
         "The organization negotiated a commercial endorsement during the year."),
    case("T50", "lexical-control", "Funding synonym.",
         "The organization negotiated corporate funding during the year."),
]


probe.LAYER = 7
probe.NEURON = 11872
probe.RANK_TAIL = "negative"
probe.OUTPUT_DIR = Path(
    "outputs/mlp_measurement/Qwen3.5-9B-Base/l7_n11872_falsification_probe"
)
probe.HYPOTHESIS = (
    "Falsification test for the proposed Michael Jordan interpretation. If "
    "L7:N11872 represents the person, identity-only and indirect-reference "
    "prompts should remain strongly negative while unrelated sponsorship "
    "prompts remain weak. If sponsorship and commercial sports language drive "
    "the stronger negative tail regardless of entity, the person-neuron "
    "interpretation should be rejected."
)
probe.TEST_PROMPTS = TEST_PROMPTS


if __name__ == "__main__":
    probe.main()
