"""
Experiment 10: Null Hypothesis Testing

Do our results reflect genuine epistemic structure, or could ANY set of
adjectives in the same templates produce similar-looking results?

Three null hypothesis tests:

1. PERMUTATION TEST: Keep the real hedge expressions but randomly shuffle
   their Mosteller medians (breaking the true phrase→probability mapping).
   Re-train the axis. Measure LOO ρ. Repeat 1000 times. If the real LOO ρ
   is above the 95th percentile, the result is significant beyond template
   artifacts.

2. NON-EPISTEMIC ADJECTIVE CONTROL: Take 8-13 non-epistemic adjectives
   (e.g., "interesting", "old", "expensive", "complex"), put them in the
   same predicative template, assign them arbitrary "probability" values,
   and see if the axis achieves comparable LOO ρ. This tests whether the
   template structure itself creates the signal.

3. RANDOM LABEL CONTROL: Keep the real expressions, assign random
   probabilities drawn from Uniform(0, 100). How often does LOO ρ exceed
   our observed values by chance?

If our real LOO ρ is in the top 1% of the permutation distribution, the
result is not an artifact of the method. If the non-epistemic adjectives
produce LOW ρ with their true (non-probability) semantic ordering but
HIGH ρ with random labels, it means the method is prone to overfitting
— a warning sign.

Usage:
  python3 experiment_10_null_hypothesis.py [model_name]
  Default model: nomic-embed-text:v1.5
"""

import sys
import numpy as np
import requests
from scipy import stats

MODEL = sys.argv[1] if len(sys.argv) > 1 else "nomic-embed-text:v1.5"
OLLAMA_URL = "http://localhost:11434/api/embed"


def embed_texts(texts, model=MODEL):
    resp = requests.post(OLLAMA_URL, json={"model": model, "input": texts})
    resp.raise_for_status()
    return np.array(resp.json()["embeddings"], dtype=np.float64)


def normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def train_axis(diffs, medians, lam=0.1):
    medians_c = (medians - medians.mean()) / medians.std()
    dim = diffs.shape[1]
    w = np.linalg.solve(diffs.T @ diffs + lam * np.eye(dim), diffs.T @ medians_c)
    w = normalize(w)
    projections = diffs @ w
    slope, intercept, _, _, _ = stats.linregress(projections, medians)
    return w, slope, intercept


def loo_rho(diffs, medians, lam=0.1):
    """Leave-one-out cross-validated Spearman ρ."""
    n = len(medians)
    if n < 4:
        return 0.0
    loo_preds = np.zeros(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        w, slope, intercept = train_axis(diffs[mask], medians[mask], lam)
        loo_preds[i] = float(np.clip(slope * (diffs[i] @ w) + intercept, 0, 100))
    r, _ = stats.spearmanr(loo_preds, medians)
    return r


BARE_CLAIM = "The experiment will succeed"

# Real epistemic expressions (our actual data)
PREDICATIVE = {
    "Certain": 99.6, "Almost certain": 90.2, "Very likely": 87.5,
    "Likely": 71.1, "Probable": 70.2, "Very probable": 89.7,
    "Possible": 38.5, "Unlikely": 17.2, "Very unlikely": 5.0,
    "Improbable": 12.5, "Very improbable": 4.8, "Impossible": 0.3,
    "Not unreasonable": 37.6,
}
PREDICATIVE_TEMPLATE = "It is {PHRASE} that the experiment will succeed"

ADVERBIAL = {
    "Always": 99.7, "Almost always": 91.7, "Very often": 82.8,
    "Often": 72.5, "Usually": 75.1, "Sometimes": 25.0,
    "Occasionally": 20.0, "Seldom": 10.2, "Very seldom": 4.9,
    "Rarely": 7.2, "Very rarely": 3.0, "Almost never": 2.9,
    "Never": 0.3, "Not often": 19.7, "Not very often": 10.1,
    "As often as not": 50.0, "More often than not": 59.8,
    "Once in a while": 15.3, "Now and then": 15.1,
}
ADVERBIAL_TEMPLATE = "The experiment will {PHRASE} succeed"

NOUN_PHRASE = {
    "Very high probability": 92.5, "High probability": 82.3,
    "Moderate probability": 52.4, "Low probability": 15.0,
    "Very low probability": 4.9, "High chance": 80.4,
    "Poor chance": 10.3, "Low chance": 9.8, "Even chance": 50.0,
    "Better than even chance": 57.6, "Less than an even chance": 40.2,
}
NOUN_PHRASE_TEMPLATE = "There is a {PHRASE} that the experiment will succeed"

# Modal: 10-item Mosteller-derived set, unified with experiment_11 / paper_validation
MODAL = {
    "certainly": 99.6,         "almost certainly": 90.2,
    "very likely": 87.5,       "likely": 71.1,
    "probably": 70.2,          "very probably": 89.7,
    "possibly": 38.5,          "unlikely": 17.2,
    "very unlikely": 5.0,      "improbably": 12.5,
}
MODAL_TEMPLATE = "The experiment will {PHRASE} succeed"

# Non-epistemic adjectives for control — same template, no probability content
NON_EPISTEMIC_ADJECTIVES = {
    # Using same predicative template: "It is {X} that the experiment will succeed"
    # These are grammatical but don't convey probability
    "important": 50,       # arbitrary assigned value
    "interesting": 45,
    "surprising": 30,
    "well-known": 60,
    "obvious": 90,         # note: "obvious" MIGHT carry epistemic weight
    "unfortunate": 25,
    "noteworthy": 55,
    "remarkable": 40,
    "controversial": 35,
    "expected": 70,        # note: this one IS somewhat epistemic
    "documented": 65,
    "unprecedented": 20,
    "acknowledged": 60,
}

# A cleaner set: adjectives with NO epistemic content whatsoever
PURE_NON_EPISTEMIC = {
    "expensive": 50,
    "complicated": 45,
    "old": 55,
    "new": 50,
    "large": 50,
    "small": 50,
    "fast": 55,
    "slow": 45,
    "difficult": 40,
    "easy": 60,
    "popular": 55,
    "rare": 30,            # note: "rare" has a frequency meaning
    "common": 70,
}
# Template for these: "It is {X} that the experiment will succeed"
# Most of these are grammatically awkward in this frame, which is the point —
# they don't belong in this template because they're not epistemic.
# But ridge regression might still find structure in the awkwardness.


def run_permutation_test(diffs, medians, n_perms=1000, seed=42):
    """Permutation test: shuffle labels, measure LOO ρ distribution."""
    rng = np.random.RandomState(seed)
    real_rho = loo_rho(diffs, medians)

    perm_rhos = np.zeros(n_perms)
    for i in range(n_perms):
        shuffled = rng.permutation(medians)
        perm_rhos[i] = loo_rho(diffs, shuffled)

    p_value = np.mean(np.abs(perm_rhos) >= np.abs(real_rho))
    percentile = np.mean(np.abs(perm_rhos) < np.abs(real_rho)) * 100

    return real_rho, perm_rhos, p_value, percentile


def run_random_label_test(diffs, n_trials=1000, seed=42):
    """Assign random Uniform(0,100) labels, measure LOO ρ distribution."""
    rng = np.random.RandomState(seed)
    n = len(diffs)

    random_rhos = np.zeros(n_trials)
    for i in range(n_trials):
        random_labels = rng.uniform(0, 100, size=n)
        random_rhos[i] = loo_rho(diffs, random_labels)

    return random_rhos


def main():
    print()
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║  Experiment 10: Null Hypothesis Testing                            ║")
    print("╠══════════════════════════════════════════════════════════════════════╣")
    print(f"║  Model: {MODEL:<58s}║")
    print("║                                                                    ║")
    print("║  Could ANY set of adjectives in the same templates produce         ║")
    print("║  similar results? Or is the signal genuinely epistemic?            ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print()

    try:
        test = embed_texts(["test"])
        dim = test.shape[1]
        print(f"Model loaded. Dim: {dim}")
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    bare_emb = embed_texts([BARE_CLAIM])[0]

    # ═══════════════════════════════════════════════════════════════════
    # Embed real epistemic expressions
    # ═══════════════════════════════════════════════════════════════════

    groups = {
        "Predicative": (PREDICATIVE, PREDICATIVE_TEMPLATE),
        "Adverbial": (ADVERBIAL, ADVERBIAL_TEMPLATE),
        "Noun phrase": (NOUN_PHRASE, NOUN_PHRASE_TEMPLATE),
        "Modal": (MODAL, MODAL_TEMPLATE),
    }

    group_diffs = {}
    group_medians = {}

    for gname, (exprs, template) in groups.items():
        phrases = list(exprs.keys())
        medians = np.array([exprs[p] for p in phrases])
        sentences = [template.replace("{PHRASE}", p.lower()) for p in phrases]
        embs = embed_texts(sentences)
        diffs = embs - bare_emb
        group_diffs[gname] = diffs
        group_medians[gname] = medians

    # ═══════════════════════════════════════════════════════════════════
    # TEST 1: PERMUTATION TEST
    # ═══════════════════════════════════════════════════════════════════
    print()
    print("═" * 78)
    print("TEST 1: PERMUTATION TEST (1000 permutations)")
    print("  Shuffle Mosteller medians across expressions. Re-train axis.")
    print("  If real LOO ρ >> shuffled LOO ρ, the signal is in the")
    print("  phrase-probability MAPPING, not in the template structure.")
    print("═" * 78)

    for gname in groups:
        diffs = group_diffs[gname]
        medians = group_medians[gname]

        real_rho, perm_rhos, p_value, percentile = run_permutation_test(
            diffs, medians, n_perms=1000)

        print(f"\n  {gname} (n={len(medians)}):")
        print(f"    Real LOO ρ:           {real_rho:+.4f}")
        print(f"    Permuted LOO ρ mean:  {np.mean(perm_rhos):+.4f} "
              f"(std: {np.std(perm_rhos):.4f})")
        print(f"    Permuted LOO ρ max:   {np.max(np.abs(perm_rhos)):+.4f}")
        print(f"    Permuted 95th %%ile:   {np.percentile(np.abs(perm_rhos), 95):+.4f}")
        print(f"    Permuted 99th %%ile:   {np.percentile(np.abs(perm_rhos), 99):+.4f}")
        print(f"    Real ρ percentile:    {percentile:.1f}th")
        print(f"    p-value (two-sided):  {p_value:.4f}")

        if p_value < 0.01:
            print(f"    → SIGNIFICANT (p < 0.01): The phrase→probability mapping")
            print(f"      is NOT an artifact of template structure.")
        elif p_value < 0.05:
            print(f"    → SIGNIFICANT (p < 0.05): Moderate evidence against null.")
        else:
            print(f"    → NOT SIGNIFICANT: The result could arise from shuffled labels.")

    # ═══════════════════════════════════════════════════════════════════
    # TEST 2: NON-EPISTEMIC ADJECTIVE CONTROL
    # ═══════════════════════════════════════════════════════════════════
    print()
    print("═" * 78)
    print("TEST 2: NON-EPISTEMIC ADJECTIVE CONTROL")
    print("  Same template, but with adjectives that carry NO probability")
    print("  content. If ridge regression achieves high LOO ρ on these,")
    print("  the method is prone to overfitting on template structure.")
    print("═" * 78)

    for label, adj_set in [("Mixed (some epistemic leakage)", NON_EPISTEMIC_ADJECTIVES),
                            ("Pure non-epistemic", PURE_NON_EPISTEMIC)]:
        phrases = list(adj_set.keys())
        assigned_vals = np.array([adj_set[p] for p in phrases])
        sentences = [PREDICATIVE_TEMPLATE.replace("{PHRASE}", p) for p in phrases]
        embs = embed_texts(sentences)
        diffs = embs - bare_emb

        # LOO with the arbitrary assigned values
        r_assigned = loo_rho(diffs, assigned_vals)

        # LOO with random labels (1000 trials)
        random_rhos = run_random_label_test(diffs, n_trials=1000)

        # In-sample fit
        w, sl, intc = train_axis(diffs, assigned_vals)
        preds = np.clip(sl * (diffs @ w) + intc, 0, 100)
        r_is, _ = stats.spearmanr(preds, assigned_vals)

        print(f"\n  {label} (n={len(phrases)}):")
        print(f"    In-sample ρ (arbitrary labels):  {r_is:+.4f}")
        print(f"    LOO ρ (arbitrary labels):        {r_assigned:+.4f}")
        print(f"    Random-label LOO ρ mean:         {np.mean(random_rhos):+.4f} "
              f"(std: {np.std(random_rhos):.4f})")
        print(f"    Random-label LOO ρ 95th %%ile:    {np.percentile(np.abs(random_rhos), 95):+.4f}")

        # Compare to real epistemic predicative
        real_loo = loo_rho(group_diffs["Predicative"], group_medians["Predicative"])
        print(f"    Real epistemic predicative LOO ρ: {real_loo:+.4f}")

        if abs(r_assigned) < np.percentile(np.abs(random_rhos), 95):
            print(f"    → GOOD: Non-epistemic adjectives don't produce structured axes.")
            print(f"      The method is not prone to template-structure overfitting.")
        else:
            print(f"    → WARNING: Non-epistemic adjectives show some structure.")
            print(f"      Investigate whether this reflects genuine semantic ordering.")

    # ═══════════════════════════════════════════════════════════════════
    # TEST 3: RANDOM LABEL CONTROL ON REAL EXPRESSIONS
    # ═══════════════════════════════════════════════════════════════════
    print()
    print("═" * 78)
    print("TEST 3: RANDOM LABELS ON REAL EPISTEMIC EXPRESSIONS (1000 trials)")
    print("  Keep the real hedge words but assign random Uniform(0,100)")
    print("  probabilities. How often does LOO ρ match our real values?")
    print("═" * 78)

    for gname in groups:
        diffs = group_diffs[gname]
        medians = group_medians[gname]
        real_rho = loo_rho(diffs, medians)
        random_rhos = run_random_label_test(diffs, n_trials=1000)

        p_value = np.mean(np.abs(random_rhos) >= np.abs(real_rho))

        print(f"\n  {gname} (n={len(medians)}):")
        print(f"    Real LOO ρ:             {real_rho:+.4f}")
        print(f"    Random-label mean:      {np.mean(random_rhos):+.4f} "
              f"(std: {np.std(random_rhos):.4f})")
        print(f"    Random-label 95th %%ile:  {np.percentile(np.abs(random_rhos), 95):+.4f}")
        print(f"    Random-label max:       {np.max(np.abs(random_rhos)):+.4f}")
        print(f"    p-value:                {p_value:.4f}")

    # ═══════════════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════════════
    print()
    print("═" * 78)
    print("SUMMARY")
    print("═" * 78)
    print()
    print("  Test 1 (Permutation): Do shuffled labels produce comparable LOO ρ?")
    print("  Test 2 (Non-epistemic): Do non-probability adjectives produce axes?")
    print("  Test 3 (Random labels): Do random targets produce comparable LOO ρ?")
    print()
    print("  If all three fail to match our real LOO ρ values, the signal is")
    print("  genuinely epistemic — it's in the phrase meanings, not the method.")
    print()


if __name__ == "__main__":
    main()
