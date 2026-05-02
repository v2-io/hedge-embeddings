"""Canonical item-set definitions for the Mosteller-derived analysis set.

Source: docs/mosteller_youtz_1990_full.csv (53 rows; row 30 "Doubtful" has
missing median/percentile fields and is excluded everywhere). 52 entries are
usable; the four within-type partitions sum to 53 template-instance items
distributed across 43 unique CSV rows.

All percentages are Mosteller medians on a [0, 100] scale, except where flagged
as `author_estimated=True` in the manifest (used only by exploratory M15 and
the cross-lingual translation-pairing sets).

Conventions
-----------
* Predicative / noun-phrase / cross-lingual keys are capitalized as in the
  CSV; experiment scripts lowercase them for template substitution.
* Adverbial and modal keys are lowercased — they substitute directly into
  templates without case manipulation.
* Templates use the `{PHRASE}` token; `BARE_CLAIM` has no trailing period.

Set membership at a glance
--------------------------
PREDICATIVE_13      §4.1, §4.2, §4.4 predicative axis
ADVERBIAL_19        §4.1 frequency adverb axis
NOUN_PHRASE_11      §4.1 noun-phrase axis
MODAL_M10           Canonical 10-item Mosteller-grounded modal axis (§4.1
                    Table 1, §4.4 erasure, §4.6 null-hypothesis 4th group).
                    Byte-equivalent to experiment_11_concept_erasure.py.
MODAL_M15           Exploratory 15-item modal axis: 8-of-M10
                    (drops `very probably`, `improbably`) + 7 author-estimated
                    forms. Used by experiments 05/06/07/08 in their original
                    runs. Author-estimated entries are flagged.
CROSSLINGUAL_PRED_12   12-item subset for experiment_09 (drops
                       `Not unreasonable`).
CROSSLINGUAL_MODAL_10  10-item set for experiment_09: 7-of-M10 + 3
                       author-estimated forms (`definitely`, `perhaps`,
                       `undoubtedly`) for translation parity across
                       8 languages.

Vogel and Wintle cross-validation expressions are NOT included here — they
live with their respective experiments because their per-study medians are
specific to each meta-analysis.
"""

# ---------------------------------------------------------------------------
# Templates and bare claim
# ---------------------------------------------------------------------------

PREDICATIVE_TEMPLATE = "It is {PHRASE} that the experiment will succeed"
ADVERBIAL_TEMPLATE = "The experiment will {PHRASE} succeed"
NOUN_PHRASE_TEMPLATE = "There is a {PHRASE} that the experiment will succeed"
MODAL_TEMPLATE = "The experiment will {PHRASE} succeed"

BARE_CLAIM = "The experiment will succeed"  # no trailing period


# ---------------------------------------------------------------------------
# Canonical Mosteller-grounded sets
# ---------------------------------------------------------------------------

# §4.1 / §4.2 / §4.4 — predicative axis (n=13)
PREDICATIVE_13 = {
    "Certain": 99.6,
    "Almost certain": 90.2,
    "Very likely": 87.5,
    "Likely": 71.1,
    "Probable": 70.2,
    "Very probable": 89.7,
    "Possible": 38.5,
    "Unlikely": 17.2,
    "Very unlikely": 5.0,
    "Improbable": 12.5,
    "Very improbable": 4.8,
    "Impossible": 0.3,
    "Not unreasonable": 37.6,
}

# §4.1 frequency adverb axis (n=19)
ADVERBIAL_19 = {
    "Always": 99.7,
    "Almost always": 91.7,
    "Very often": 82.8,
    "Often": 72.5,
    "Usually": 75.1,
    "Sometimes": 25.0,
    "Occasionally": 20.0,
    "Seldom": 10.2,
    "Very seldom": 4.9,
    "Rarely": 7.2,
    "Very rarely": 3.0,
    "Almost never": 2.9,
    "Never": 0.3,
    "Not often": 19.7,
    "Not very often": 10.1,
    "As often as not": 50.0,
    "More often than not": 59.8,
    "Once in a while": 15.3,
    "Now and then": 15.1,
}

# §4.1 noun-phrase axis (n=11)
NOUN_PHRASE_11 = {
    "Very high probability": 92.5,
    "High probability": 82.3,
    "Moderate probability": 52.4,
    "Low probability": 15.0,
    "Very low probability": 4.9,
    "High chance": 80.4,
    "Poor chance": 10.3,
    "Low chance": 9.8,
    "Even chance": 50.0,
    "Better than even chance": 57.6,
    "Less than an even chance": 40.2,
}

# §4.1 / §4.4 / §4.6 canonical modal axis — 10 Mosteller-grounded adverbials.
# Byte-equivalent to experiment_11_concept_erasure.py:173-184. Each adverb
# inherits the median of its predicative source (e.g. "improbably" = "Improbable").
MODAL_M10 = {
    "certainly": 99.6,         # adverb of "Certain"
    "almost certainly": 90.2,  # adverb of "Almost certain"
    "very likely": 87.5,
    "likely": 71.1,
    "probably": 70.2,          # adverb of "Probable"
    "very probably": 89.7,     # adverb of "Very probable"
    "possibly": 38.5,          # adverb of "Possible"
    "unlikely": 17.2,
    "very unlikely": 5.0,
    "improbably": 12.5,        # adverb of "Improbable"
}


# ---------------------------------------------------------------------------
# Exploratory M15 (used by §4.5 robustness probes — experiments 05/06/07/08)
# ---------------------------------------------------------------------------
# 8-of-M10 (drops `very probably` and `improbably`) + 7 author-estimated forms.
# Byte-equivalent to experiment_05_ensemble.py:124-131,
# experiment_06_compound_hedges.py:60-67, experiment_07:113-120, experiment_08:67-74.
#
# Author-estimated values are clearly marked here. They were assigned by the
# authors based on closest-Mosteller-synonym intuitions and have NO independent
# psychometric grounding. M10 is preferred for any quantitative claim.
MODAL_M15 = {
    # 8-of-M10 (Mosteller-grounded)
    "certainly": 99.6,
    "almost certainly": 90.2,
    "very likely": 87.5,
    "likely": 71.1,
    "probably": 70.2,
    "possibly": 38.5,
    "unlikely": 17.2,
    "very unlikely": 5.0,
    # 7 author-estimated values
    "conceivably": 38.5,    # author-estimated (≈ "Possible")
    "definitely": 99.6,     # author-estimated (≈ "Certain")
    "perhaps": 38.5,        # author-estimated (≈ "Possible")
    "maybe": 38.5,          # author-estimated (≈ "Possible")
    "presumably": 70.2,     # author-estimated (≈ "Probable")
    "undoubtedly": 95.0,    # author-estimated (≈ between "Almost certain" and "Certain")
    "arguably": 55.0,       # author-estimated (no close Mosteller analogue)
}

# Set of M15 keys whose median is author-estimated (not from Mosteller).
MODAL_M15_AUTHOR_ESTIMATED = frozenset({
    "conceivably", "definitely", "perhaps", "maybe",
    "presumably", "undoubtedly", "arguably",
})


# ---------------------------------------------------------------------------
# Cross-lingual translation-pairing subsets (experiment_09)
# ---------------------------------------------------------------------------
# 12-item predicative subset: PREDICATIVE_13 minus "Not unreasonable" (no clean
# translation across all 8 target languages). Lowercased keys to match the
# format the cross-lingual experiment uses.
CROSSLINGUAL_PRED_12 = {
    "certain": 99.6,
    "almost certain": 90.2,
    "very likely": 87.5,
    "likely": 71.1,
    "probable": 70.2,
    "very probable": 89.7,
    "possible": 38.5,
    "unlikely": 17.2,
    "very unlikely": 5.0,
    "improbable": 12.5,
    "very improbable": 4.8,
    "impossible": 0.3,
}

# 10-item modal set used in experiment_09 for translation pairing across
# 8 languages. 7-of-M10 (drops `very probably`, `improbably`, `almost certainly`
# is kept; `very likely` and `unlikely` etc. are kept) + 3 author-estimated
# (`definitely`, `perhaps`, `undoubtedly`). Author-estimated entries are
# flagged in CROSSLINGUAL_MODAL_10_AUTHOR_ESTIMATED.
CROSSLINGUAL_MODAL_10 = {
    "certainly": 99.6,
    "almost certainly": 90.2,
    "probably": 70.2,
    "likely": 71.1,
    "possibly": 38.5,
    "unlikely": 17.2,
    "very unlikely": 5.0,
    "definitely": 99.6,    # author-estimated
    "perhaps": 38.5,       # author-estimated
    "undoubtedly": 95.0,   # author-estimated
}

CROSSLINGUAL_MODAL_10_AUTHOR_ESTIMATED = frozenset({
    "definitely", "perhaps", "undoubtedly",
})


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def _author_estimated_for(name):
    """Return frozenset of author-estimated keys for a named set, or empty."""
    if name == "MODAL_M15":
        return MODAL_M15_AUTHOR_ESTIMATED
    if name == "CROSSLINGUAL_MODAL_10":
        return CROSSLINGUAL_MODAL_10_AUTHOR_ESTIMATED
    return frozenset()


_SETS = [
    ("PREDICATIVE_13", PREDICATIVE_13, PREDICATIVE_TEMPLATE),
    ("ADVERBIAL_19", ADVERBIAL_19, ADVERBIAL_TEMPLATE),
    ("NOUN_PHRASE_11", NOUN_PHRASE_11, NOUN_PHRASE_TEMPLATE),
    ("MODAL_M10", MODAL_M10, MODAL_TEMPLATE),
    ("MODAL_M15", MODAL_M15, MODAL_TEMPLATE),
    ("CROSSLINGUAL_PRED_12", CROSSLINGUAL_PRED_12, PREDICATIVE_TEMPLATE),
    ("CROSSLINGUAL_MODAL_10", CROSSLINGUAL_MODAL_10, MODAL_TEMPLATE),
]


def print_manifest():
    """Print the contents of every set, marking author-estimated entries."""
    print("Mosteller-derived analysis set — manifest")
    print("=" * 72)
    print(f"BARE_CLAIM: {BARE_CLAIM!r}")
    print()
    for name, mapping, template in _SETS:
        ae = _author_estimated_for(name)
        n_ae = sum(1 for k in mapping if k in ae)
        n_grounded = len(mapping) - n_ae
        print(f"{name}  (n={len(mapping)}; Mosteller-grounded={n_grounded}, "
              f"author-estimated={n_ae})")
        print(f"  template: {template!r}")
        for k, v in mapping.items():
            tag = "  [author-est]" if k in ae else ""
            print(f"    {k:30s}  {v:6.1f}{tag}")
        print()


if __name__ == "__main__":
    print_manifest()
