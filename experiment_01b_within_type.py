"""
Experiment 01b: Within-type analysis to remove syntactic confound.

Follow-up to experiment_01. The main finding was that PC1 across all 52
Mosteller expressions captures syntactic type (noun phrase vs adverb vs
predicative), NOT probability. But PC2 showed r=-0.65 with probability,
suggesting the signal exists behind the syntactic variation.

This script tests:
  1. Within each syntactic type (holding syntax constant), does PC1
     correlate with probability? (Removes the syntactic confound.)
  2. Does the Part A "probably" direction align with the probability-
     correlated axis (PC2 of the full set)?
  3. What does a supervised "probability direction" look like, and
     does it generalize across syntactic types?

References:
  - experiment_01_hedge_direction.py (initial findings)
  - EXPERIMENT-PLAN.md (overall experimental design)
  - docs/mosteller_youtz_1990_full.csv (ground truth)
"""

import sys
import json
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.decomposition import PCA

# Import configuration from experiment_01
# (Duplicating the relevant parts here for self-containedness — a future agent
# should not need to parse experiment_01 to understand this script.)

MODEL = sys.argv[1] if len(sys.argv) > 1 else "nomic-embed-text:v1.5"
OLLAMA_URL = "http://localhost:11434/api/embed"
MOSTELLER_CSV = Path(__file__).parent / "docs" / "mosteller_youtz_1990_full.csv"
VOGEL_CSV = Path(__file__).parent / "docs" / "vogel_2022_systematic_review.csv"


def embed_texts(texts, model=MODEL):
    resp = requests.post(OLLAMA_URL, json={"model": model, "input": texts})
    resp.raise_for_status()
    return np.array(resp.json()["embeddings"], dtype=np.float64)


def normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


# ---------------------------------------------------------------------------
# Expression sets by syntactic type
#
# Each group uses ONE template, eliminating syntactic variation entirely.
# Within a group, the ONLY thing that changes is the hedge phrase.
# If probability is a linear direction, it should be PC1 within each group.
# ---------------------------------------------------------------------------

PREDICATIVE_EXPRESSIONS = {
    # expression: Mosteller median probability
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
PREDICATIVE_TEMPLATE = "It is {PHRASE} that the experiment will succeed"
PREDICATIVE_BARE = "The experiment will succeed"

ADVERBIAL_EXPRESSIONS = {
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
ADVERBIAL_TEMPLATE = "The experiment will {PHRASE} succeed"
ADVERBIAL_BARE = "The experiment will succeed"

NOUN_PHRASE_EXPRESSIONS = {
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
NOUN_PHRASE_TEMPLATE = "There is a {PHRASE} that the experiment will succeed"
NOUN_PHRASE_BARE = "The experiment will succeed"

# ---------------------------------------------------------------------------
# Additional claim contexts for cross-content validation
#
# If the probability axis is content-independent, the same direction should
# emerge when we use different bare claims.
# ---------------------------------------------------------------------------

EXTRA_CONTEXTS = [
    {
        "predicative": "It is {PHRASE} that the treatment will be effective",
        "adverbial": "The treatment will {PHRASE} be effective",
        "noun_phrase": "There is a {PHRASE} that the treatment will be effective",
        "bare": "The treatment will be effective",
    },
    {
        "predicative": "It is {PHRASE} that the prediction will be correct",
        "adverbial": "The prediction will {PHRASE} be correct",
        "noun_phrase": "There is a {PHRASE} that the prediction will be correct",
        "bare": "The prediction will be correct",
    },
]


def analyze_group(name, expressions, template, bare, extra_contexts=None):
    """
    Analyze a single syntactic group: does PCA on difference vectors reveal
    a probability axis?
    """
    print(f"\n{'─'*60}")
    print(f"  {name} (n={len(expressions)})")
    print(f"  Template: \"{template}\"")
    print(f"{'─'*60}")

    # Build sentences
    phrases = list(expressions.keys())
    medians = np.array([expressions[p] for p in phrases])

    sentences = [template.replace("{PHRASE}", p.lower()) for p in phrases]
    all_to_embed = [bare] + sentences
    embeddings = embed_texts(all_to_embed)

    e_bare = embeddings[0]
    e_phrases = embeddings[1:]
    diffs = e_phrases - e_bare

    # --- PCA ---
    n_components = min(5, len(phrases) - 1)
    pca = PCA(n_components=n_components)
    projected = pca.fit_transform(diffs)

    print(f"\n  PCA explained variance:")
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    for i in range(min(3, n_components)):
        print(f"    PC{i+1}: {pca.explained_variance_ratio_[i]*100:.1f}% "
              f"(cum: {cumvar[i]*100:.1f}%)")

    # --- Correlations with probability ---
    print(f"\n  Correlation with Mosteller median:")
    best_pc = 0
    best_r = 0
    for i in range(min(3, n_components)):
        pc = projected[:, i]
        r_s, p_s = stats.spearmanr(pc, medians)
        r_p, p_p = stats.pearsonr(pc, medians)
        marker = ""
        if abs(r_s) > abs(best_r):
            best_r = r_s
            best_pc = i
            marker = " ← BEST"
        print(f"    PC{i+1}: Spearman ρ={r_s:+.4f} (p={p_s:.2e}), "
              f"Pearson r={r_p:+.4f} (p={p_p:.2e}){marker}")

    # The best probability-correlated direction
    prob_axis = pca.components_[best_pc]

    # --- Supervised direction: regression directly on medians ---
    # Find the direction in embedding space that best predicts probability.
    # This is the "best case" — if even this doesn't work, the signal isn't there.
    # Use least squares: find w such that diffs @ w ≈ medians (scaled)
    medians_centered = (medians - medians.mean()) / medians.std()
    # Regularized least squares (ridge, small lambda for stability)
    dim = diffs.shape[1]
    lam = 0.1
    w_supervised = np.linalg.solve(
        diffs.T @ diffs + lam * np.eye(dim),
        diffs.T @ medians_centered
    )
    w_supervised = normalize(w_supervised)

    supervised_proj = diffs @ w_supervised
    r_sup, p_sup = stats.spearmanr(supervised_proj, medians)
    print(f"\n  Supervised direction (ridge regression on probability):")
    print(f"    Spearman ρ={r_sup:+.4f} (p={p_sup:.2e})")
    print(f"    Alignment with PC{best_pc+1}: "
          f"{abs(float(w_supervised @ prob_axis)):.4f}")

    # --- Cross-context validation (if available) ---
    if extra_contexts:
        print(f"\n  Cross-context validation:")
        for ctx_i, ctx in enumerate(extra_contexts):
            ctx_template = ctx.get(name.lower().replace(" ", "_"))
            if ctx_template is None:
                # Try to match
                for key in ctx:
                    if key != "bare" and name.lower().startswith(key[:4]):
                        ctx_template = ctx[key]
                        break
            if ctx_template is None:
                continue

            ctx_sentences = [ctx_template.replace("{PHRASE}", p.lower())
                            for p in phrases]
            ctx_all = [ctx["bare"]] + ctx_sentences
            ctx_emb = embed_texts(ctx_all)
            ctx_bare_e = ctx_emb[0]
            ctx_diffs = ctx_emb[1:] - ctx_bare_e

            # Project onto the supervised direction from the primary context
            ctx_proj = ctx_diffs @ w_supervised
            r_ctx, p_ctx = stats.spearmanr(ctx_proj, medians)
            print(f"    Context {ctx_i+1}: ρ={r_ctx:+.4f} (p={p_ctx:.2e}) "
                  f"{'✓ generalizes' if abs(r_ctx) > 0.5 else '✗ weak'}")

    # --- Print ranking for visual inspection ---
    sorted_idx = np.argsort(supervised_proj)[::-1]
    print(f"\n  Expressions ranked by supervised projection (should match probability):")
    for i in sorted_idx:
        bar_len = int(medians[i] / 100 * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        print(f"    {phrases[i]:28s} Median={medians[i]:5.1f}%  "
              f"Proj={supervised_proj[i]:+.3f}  {bar}")

    return {
        "name": name,
        "n": len(phrases),
        "pca_var_1": pca.explained_variance_ratio_[0],
        "best_pc": best_pc,
        "best_r": best_r,
        "supervised_r": r_sup,
        "prob_axis": prob_axis,
        "supervised_axis": w_supervised,
    }


# ---------------------------------------------------------------------------
# Part A direction alignment check
# ---------------------------------------------------------------------------

def check_partA_alignment(group_results):
    """
    Compute the 'probably' direction from Part A claims and check alignment
    with each group's probability axis.
    """
    print(f"\n{'═'*60}")
    print("  Part A 'probably' direction vs. group probability axes")
    print(f"{'═'*60}\n")

    # Re-embed Part A claims to get the 'probably' direction
    from experiment_01_hedge_direction import CLAIM_TRIPLES
    bare_texts = [t[0] for t in CLAIM_TRIPLES]
    hedge_texts = [t[1] for t in CLAIM_TRIPLES]

    all_texts = bare_texts + hedge_texts
    all_emb = embed_texts(all_texts)
    n = len(CLAIM_TRIPLES)
    diffs = all_emb[n:] - all_emb[:n]
    probably_dir = normalize(diffs.mean(axis=0))

    for result in group_results:
        align_pca = abs(float(probably_dir @ result["prob_axis"]))
        align_sup = abs(float(probably_dir @ result["supervised_axis"]))
        print(f"  {result['name']}:")
        print(f"    vs. best PCA axis: {align_pca:.4f}")
        print(f"    vs. supervised axis: {align_sup:.4f}")

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print()
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║  Experiment 01b: Within-Type Syntactic Control                     ║")
    print("╠══════════════════════════════════════════════════════════════════════╣")
    print(f"║  Model: {MODEL:<58s}║")
    print("║                                                                    ║")
    print("║  Question: When syntactic variation is removed, does the dominant  ║")
    print("║  direction within each phrase type predict probability?            ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")

    # Verify model
    try:
        test = embed_texts(["test"])
        print(f"\nModel loaded. Dim: {test.shape[1]}")
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    results = []

    # Analyze each syntactic group
    r_pred = analyze_group(
        "Predicative", PREDICATIVE_EXPRESSIONS,
        PREDICATIVE_TEMPLATE, PREDICATIVE_BARE,
        extra_contexts=EXTRA_CONTEXTS
    )
    results.append(r_pred)

    r_adv = analyze_group(
        "Adverbial", ADVERBIAL_EXPRESSIONS,
        ADVERBIAL_TEMPLATE, ADVERBIAL_BARE,
        extra_contexts=EXTRA_CONTEXTS
    )
    results.append(r_adv)

    r_np = analyze_group(
        "Noun_phrase", NOUN_PHRASE_EXPRESSIONS,
        NOUN_PHRASE_TEMPLATE, NOUN_PHRASE_BARE,
        extra_contexts=EXTRA_CONTEXTS
    )
    results.append(r_np)

    # Check Part A alignment
    try:
        check_partA_alignment(results)
    except Exception as e:
        print(f"\n  (Could not check Part A alignment: {e})")

    # --- Summary ---
    print(f"\n{'═'*60}")
    print("  SUMMARY")
    print(f"{'═'*60}\n")

    for r in results:
        print(f"  {r['name']:15s}  n={r['n']:2d}  "
              f"PC{r['best_pc']+1} ρ={r['best_r']:+.3f}  "
              f"Supervised ρ={r['supervised_r']:+.3f}  "
              f"PC1 var={r['pca_var_1']*100:.0f}%")

    print()
    any_strong = any(abs(r['supervised_r']) > 0.7 for r in results)
    any_moderate = any(abs(r['supervised_r']) > 0.5 for r in results)

    if any_strong:
        print("  POSITIVE: At least one syntactic type shows strong probability signal.")
        print("  The hypothesis holds within controlled syntax.")
        print("  → Proceed to multi-claim validation and cross-model testing.")
    elif any_moderate:
        print("  MODERATE: Some signal, but not overwhelmingly strong.")
        print("  → Investigate whether more claims or a larger model helps.")
    else:
        print("  NEGATIVE: No strong probability signal within any syntactic type.")
        print("  → Try a different model, or reconsider the hypothesis.")

    print()


if __name__ == "__main__":
    main()
