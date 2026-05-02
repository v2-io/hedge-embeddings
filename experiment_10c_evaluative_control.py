"""
Experiment 10c — Evaluative Non-Epistemic Control (canonical-pipeline version)

Replaces the original PURE_NON_EPISTEMIC control in
`experiment_10_null_hypothesis.py` (lines 143–157) with a grammatically
clean, semantically non-probabilistic alternative — and validates that
the standalone protocol matches the paper's canonical pipeline against
a positive control before any conclusion is drawn.

WHY A REPLACEMENT (recap)
-------------------------
Codex flagged that the original PURE set ("expensive", "complicated",
"old", "fast", "rare", "common", ...) has two problems:

  1. Most of those adjectives don't grammatically take *that*-complement
     clauses. "It is expensive that the experiment will succeed" is
     structurally marginal; the script's own source-comment on
     experiment_10_null_hypothesis.py:161 already concedes this.
     A bad LOO ρ on this set is therefore confounded — could test
     absence of probability content, OR could just test ungrammaticality.

  2. Two items ("rare", "common") are frequency/probability-adjacent.

PROTOCOL DRIFT NOTE
-------------------
A previous version of this script reimplemented `train_axis`, the bare
claim string, and the phrase-substitution step itself. That reimplemen-
tation drifted from the canonical pipeline in three places:

  (a) BARE_CLAIM had a trailing period; canonical does not.
  (b) Phrase substitution did not lowercase the dict key; canonical does.
  (c) train_axis centered medians without dividing by std; canonical
      does (medians - mean) / std.

These together broke the positive control: Mosteller predicative LOO ρ
came back at -0.022 instead of the canonical 0.874. To eliminate the
risk of any further drift, this version imports `embed_texts`,
`train_axis`, `BARE_CLAIM`, `PREDICATIVE`, and `PREDICATIVE_TEMPLATE`
directly from `paper_validation.py`. The only thing this script defines
locally is the new EVALUATIVE adjective set and a thin LOO/in-sample
evaluation wrapper that mirrors `paper_validation.run_loo`.

POSITIVE CONTROL
----------------
The script always evaluates Mosteller predicative first, as a positive
control. If the LOO ρ on that does not come back ~0.85–0.88 on
nomic-embed-text:v1.5, the script aborts with a clear message — the
EVALUATIVE result is not trustable until the canonical pipeline yields
the canonical answer.
"""

import sys
import numpy as np
from scipy import stats

# Import the canonical pipeline. paper_validation.py has an
# `if __name__ == "__main__":` guard, so importing it does not run
# the validation main(). It does set MODEL from sys.argv[1] at module
# load time, which is the same convention every script in this repo
# uses, so this works correctly when invoked as
# `python3 experiment_10c_evaluative_control.py [model_name]`.
from paper_validation import (
    embed_texts,
    train_axis,
    BARE_CLAIM,
    PREDICATIVE,
    PREDICATIVE_TEMPLATE,
    MODEL,
)

# ----- Adjective sets to compare -----

# Original PURE_NON_EPISTEMIC from experiment_10 — included for direct
# comparison. Codex's grammar-mismatch critique target.
ORIG_PURE = {
    "expensive": 50, "complicated": 45, "old": 55, "new": 50,
    "large": 50, "small": 50, "fast": 55, "slow": 45,
    "difficult": 40, "easy": 60, "popular": 55, "rare": 30, "common": 70,
}

# Original NON_EPISTEMIC_ADJECTIVES from experiment_10 — kept as-is in
# the paper as the epistemic-leakage probe. Included here so all four
# numbers come from the same model run and are directly comparable.
ORIG_MIXED = {
    "important": 50, "interesting": 45, "surprising": 30, "well-known": 60,
    "obvious": 90, "unfortunate": 25, "noteworthy": 55, "remarkable": 40,
    "controversial": 35, "expected": 70, "documented": 65,
    "unprecedented": 20, "acknowledged": 60,
}

# NEW EVALUATIVE control — grammatical *that*-complement, non-probabilistic.
# Labels are deliberately scrambled across 20–80 so they do NOT track the
# natural negative→positive valence of the adjectives. The test is whether
# the protocol fits these arbitrary labels via LOO; it should not.
EVALUATIVE = {
    "tragic": 60, "regrettable": 35, "unfortunate": 75, "disappointing": 20,
    "concerning": 50, "troubling": 80, "interesting": 30, "noteworthy": 65,
    "fitting": 25, "encouraging": 70, "satisfying": 40, "fortunate": 55,
    "wonderful": 45,
}


def evaluate_set(name, adj_dict, lowercase_keys=True):
    """Evaluate one adjective set under the canonical pipeline.

    `lowercase_keys` mirrors `paper_validation.run_baselines:177`, which
    lowercases dict keys before substitution. Mosteller predicative keys
    are capitalized ("Certain"); the canonical pipeline produces "It is
    certain that the experiment will succeed". The non-epistemic control
    sets here use already-lowercase keys so the flag is a no-op for them
    but is set the same way for consistency with the canonical path.
    """
    print(f"\n--- {name}  (n = {len(adj_dict)}) ---")

    phrases = list(adj_dict.keys())
    medians = np.array([adj_dict[p] for p in phrases], dtype=np.float64)

    # Canonical sentence construction.
    sentences = [
        PREDICATIVE_TEMPLATE.replace(
            "{PHRASE}", p.lower() if lowercase_keys else p)
        for p in phrases
    ]

    # Canonical embed: BARE_CLAIM first, then all template sentences in
    # one batch call. Match `paper_validation.run_baselines` exactly.
    bare_emb = embed_texts([BARE_CLAIM])[0]
    template_embs = embed_texts(sentences)
    diffs = template_embs - bare_emb

    # In-sample fit via canonical train_axis.
    w, slope, intercept = train_axis(diffs, medians)
    proj = diffs @ w
    in_preds = np.clip(slope * proj + intercept, 0, 100)
    in_rho, _ = stats.spearmanr(proj, medians)
    in_mae = float(np.mean(np.abs(in_preds - medians)))

    # LOO via the same loop as `paper_validation.run_loo`.
    n = len(medians)
    loo_preds = np.zeros(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        w_i, slope_i, intercept_i = train_axis(diffs[mask], medians[mask])
        loo_preds[i] = float(np.clip(
            slope_i * (diffs[i] @ w_i) + intercept_i, 0, 100))
    loo_rho, _ = stats.spearmanr(loo_preds, medians)
    loo_mae = float(np.mean(np.abs(loo_preds - medians)))

    print(f"    In-sample ρ : {in_rho:+.4f}")
    print(f"    In-sample MAE: {in_mae:.2f}")
    print(f"    LOO ρ        : {loo_rho:+.4f}")
    print(f"    LOO MAE      : {loo_mae:.2f}")
    return loo_rho


# ----- Run -----

print("=" * 72)
print(f"Experiment 10c — evaluative non-epistemic control (canonical pipeline)")
print(f"Model: {MODEL}")
print(f"Template: \"{PREDICATIVE_TEMPLATE}\"")
print(f"Bare claim: \"{BARE_CLAIM}\"  (no trailing period — canonical)")
print("=" * 72)

results = {}

# Positive control — Mosteller predicative. If this does not come back
# in the 0.85–0.88 range on nomic-embed-text:v1.5, the protocol is still
# drifting and any other result here is untrustable.
results["real"] = evaluate_set(
    "Mosteller predicative — positive ceiling", PREDICATIVE)

POSITIVE_CONTROL_FLOOR = 0.80
if results["real"] < POSITIVE_CONTROL_FLOOR:
    print(f"\n*** POSITIVE CONTROL FAILED: Mosteller predicative LOO ρ = "
          f"{results['real']:+.4f}, expected ≥ {POSITIVE_CONTROL_FLOOR}.")
    print("    The standalone pipeline is still drifting from canonical.")
    print("    Aborting before reporting EVALUATIVE; result would be untrustable.")
    sys.exit(1)
else:
    print(f"\n[positive control PASSED at LOO ρ = {results['real']:+.4f}, "
          f"≥ {POSITIVE_CONTROL_FLOOR}]")

results["orig_pure"]  = evaluate_set(
    "Original PURE — ungrammatical (Codex critique target)", ORIG_PURE)
results["orig_mixed"] = evaluate_set(
    "Original MIXED — epistemic leakage probe", ORIG_MIXED)
results["evaluative"] = evaluate_set(
    "NEW EVALUATIVE — clean grammatical control", EVALUATIVE)

print()
print("=" * 72)
print("SUMMARY")
print("=" * 72)
print(f"  {'Set':<55} {'LOO ρ':>10}")
print(f"  {'-' * 55} {'-' * 10}")
for label, name in [("real", "Mosteller predicative (ceiling)"),
                    ("orig_pure", "Original PURE (ungrammatical)"),
                    ("orig_mixed", "Original MIXED (leakage probe)"),
                    ("evaluative", "EVALUATIVE (new clean control)")]:
    print(f"  {name:<55} {results[label]:>+10.4f}")

print()
print("Reading:")
print("  - The protocol is selective for actual probability semantics if")
print("    EVALUATIVE LOO ρ is near zero (or anywhere comfortably below the")
print("    ORIG_PURE 0.31 baseline).")
print("  - If EVALUATIVE LOO ρ rises substantially above zero, the protocol")
print("    fits arbitrary labels on grammatical adjectives in this template,")
print("    which would weaken the §4.6 control story.")
print("  - Mosteller predicative is the positive ceiling. ORIG_MIXED's")
print("    elevated LOO ρ is the epistemic-leakage signal §6 L4 already")
print("    documents — kept as-is in the paper.")
