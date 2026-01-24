"""
Experiment 01: Does epistemic hedging have consistent linear structure
in sentence embedding space?

Two complementary tests in one script:

Part A - Consistency Test:
  Does "probably" produce a consistent difference vector regardless of
  what claim it modifies? Compare to a control adverb ("reportedly")
  which modifies epistemic stance differently (evidential, not confidence).

Part B - Mosteller Projection:
  Do the 53 Mosteller verbal probability expressions, when embedded in
  a fixed claim context, project onto a low-dimensional subspace whose
  principal axis correlates with empirically calibrated probability?

Design philosophy:
  - Start with one model (nomic-embed-text:v1.5), easily swappable
  - Compute both tests from the same embedding calls where possible
  - Report results clearly enough that a future agent can decide
    whether to proceed to Experiment 2 without re-reading all the context
  - All ground truth comes from docs/mosteller_youtz_1990_full.csv

References:
  - EXPERIMENT-PLAN.md (experimental design and rationale)
  - docs/epistemic-geometry-hedging-as-linear-structure.md (hypothesis)
  - docs/verbal-probability-calibration.md (calibration context)
  - docs/mosteller_youtz_1990_full.csv (ground truth data)

Usage:
  python3 experiment_01_hedge_direction.py [model_name]
  Default model: nomic-embed-text:v1.5
"""

import sys
import json
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.decomposition import PCA

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Model to use (override via command line argument)
MODEL = sys.argv[1] if len(sys.argv) > 1 else "nomic-embed-text:v1.5"

OLLAMA_URL = "http://localhost:11434/api/embed"

# Ground truth data
MOSTELLER_CSV = Path(__file__).parent / "docs" / "mosteller_youtz_1990_full.csv"
VOGEL_CSV = Path(__file__).parent / "docs" / "vogel_2022_systematic_review.csv"

# ---------------------------------------------------------------------------
# Part A: Claim set for consistency testing
#
# Each claim must be:
#   - Declarative (asserting something)
#   - Grammatically natural with "probably" and "reportedly" inserted
#   - Diverse in domain
#   - Not about probability itself
#   - Roughly similar in token length
#
# The template is: "The {subject} is {adverb} {predicate}"
# or adapted forms that keep insertion position consistent.
# ---------------------------------------------------------------------------

# (bare_claim, hedged_with_probably, control_with_reportedly)
# Hand-crafted for grammatical naturalness with both modifiers.
CLAIM_TRIPLES = [
    # Science / physical world
    ("The bridge is structurally sound",
     "The bridge is probably structurally sound",
     "The bridge is reportedly structurally sound"),

    ("The glacier is retreating faster than predicted",
     "The glacier is probably retreating faster than predicted",
     "The glacier is reportedly retreating faster than predicted"),

    ("The vaccine reduces transmission significantly",
     "The vaccine probably reduces transmission significantly",
     "The vaccine reportedly reduces transmission significantly"),

    # Technology
    ("The algorithm converges within acceptable bounds",
     "The algorithm probably converges within acceptable bounds",
     "The algorithm reportedly converges within acceptable bounds"),

    ("The encryption protocol is vulnerable to side-channel attacks",
     "The encryption protocol is probably vulnerable to side-channel attacks",
     "The encryption protocol is reportedly vulnerable to side-channel attacks"),

    # Social / institutional
    ("The election results reflect the actual vote counts",
     "The election results probably reflect the actual vote counts",
     "The election results reportedly reflect the actual vote counts"),

    ("The policy reduces carbon emissions over a decade",
     "The policy probably reduces carbon emissions over a decade",
     "The policy reportedly reduces carbon emissions over a decade"),

    # Personal / human
    ("The patient is responding well to treatment",
     "The patient is probably responding well to treatment",
     "The patient is reportedly responding well to treatment"),

    ("The student understands the core concepts",
     "The student probably understands the core concepts",
     "The student reportedly understands the core concepts"),

    ("The witness is telling the truth",
     "The witness is probably telling the truth",
     "The witness is reportedly telling the truth"),

    # Abstract / conceptual
    ("The model captures the essential dynamics of the system",
     "The model probably captures the essential dynamics of the system",
     "The model reportedly captures the essential dynamics of the system"),

    ("The theory explains the observed anomalies",
     "The theory probably explains the observed anomalies",
     "The theory reportedly explains the observed anomalies"),

    # Everyday / physical
    ("The roof will last another ten years",
     "The roof will probably last another ten years",
     "The roof will reportedly last another ten years"),

    ("The restaurant serves authentic regional cuisine",
     "The restaurant probably serves authentic regional cuisine",
     "The restaurant reportedly serves authentic regional cuisine"),

    ("The water supply meets safety standards",
     "The water supply probably meets safety standards",
     "The water supply reportedly meets safety standards"),
]

# ---------------------------------------------------------------------------
# Part B: Mosteller expression templates
#
# Different syntactic types need different templates. We classify each
# Mosteller expression and embed it in an appropriate frame.
#
# The "bare" baseline for Part B is the unmodified template claim:
#   "The experiment will succeed"
#
# This is deliberately simple and neutral to minimize content-driven
# variation in the difference vectors.
# ---------------------------------------------------------------------------

# Template claim for Part B (the propositional content held constant)
PARTB_BARE = "The experiment will succeed"

# We need multiple bare claims for Part B to test across content too,
# but start with one for the initial signal check.
PARTB_BARE_EXTRA = [
    "The treatment will be effective",
    "The prediction will be correct",
    "The investment will be profitable",
]

# Classify Mosteller expressions by how they embed in a sentence.
# Returns (template_with_placeholder, syntactic_type)
# The placeholder is {PHRASE} in each template.
#
# NOTE: Some classifications might be wrong. This is my best guess at
# natural phrasing. If results look weird for specific expressions,
# check whether the template is producing ungrammatical sentences.

SYNTACTIC_TEMPLATES = {
    # "It is {PHRASE} that X will Y" — predicative adjectives
    "predicative": "It is {PHRASE} that the experiment will succeed",

    # "The experiment will {PHRASE} succeed" — adverbs modifying the verb
    "adverbial": "The experiment will {PHRASE} succeed",

    # "There is a {PHRASE} that the experiment will succeed" — noun phrases
    "noun_phrase": "There is a {PHRASE} that the experiment will succeed",

    # Complex expressions that don't fit neatly — use a looser frame
    "clausal": "The experiment {PHRASE}",
}

# Classification of each Mosteller expression.
# This is manual and could be wrong in places — noted for future review.
EXPRESSION_SYNTAX = {
    # Predicative adjectives (it is X that...)
    "Certain": "predicative",
    "Almost certain": "predicative",
    "Very likely": "predicative",
    "Likely": "predicative",
    "Probable": "predicative",
    "Very probable": "predicative",
    "Possible": "predicative",
    "Unlikely": "predicative",
    "Very unlikely": "predicative",
    "Improbable": "predicative",
    "Very improbable": "predicative",
    "Impossible": "predicative",
    "Doubtful": "predicative",

    # Adverbs (X will {adverb} Y)
    "Always": "adverbial",
    "Almost always": "adverbial",
    "Very often": "adverbial",
    "Often": "adverbial",
    "Frequently": "adverbial",  # listed as "Frequent" in data
    "Usually": "adverbial",
    "Sometimes": "adverbial",
    "Occasionally": "adverbial",
    "Seldom": "adverbial",
    "Very seldom": "adverbial",
    "Rarely": "adverbial",
    "Very rarely": "adverbial",
    "Almost never": "adverbial",
    "Never": "adverbial",
    "Not often": "adverbial",
    "Not very often": "adverbial",
    "Very infrequent": "adverbial",  # awkward as adverb but closest fit

    # Noun phrases (there is a X that...)
    "High chance": "noun_phrase",
    "High probability": "noun_phrase",
    "Very high probability": "noun_phrase",
    "Moderate probability": "noun_phrase",
    "Low probability": "noun_phrase",
    "Very low probability": "noun_phrase",
    "Poor chance": "noun_phrase",
    "Low chance": "noun_phrase",
    "Even chance": "noun_phrase",
    "Better than even chance": "noun_phrase",
    "Less than an even chance": "noun_phrase",

    # Frequency-as-adverb (awkward fits, but closest natural phrasing)
    "Very frequent": "adverbial",  # "will very frequently succeed"
    "Frequent": "adverbial",       # "will frequently succeed"
    "Infrequent": "adverbial",     # "will infrequently succeed"

    # Complex / clausal (experiment {PHRASE})
    "Once in a while": "adverbial",  # "will once in a while succeed" — awkward
    "Now and then": "adverbial",     # "will now and then succeed" — awkward
    "As often as not": "adverbial",
    "More often than not": "adverbial",
    "Less often than not": "adverbial",
    "Not unreasonable": "predicative",  # "it is not unreasonable that..."
    "Not infrequent": "adverbial",      # awkward but closest
    "Might happen": "clausal",          # "the experiment might happen"... wrong.
    "Liable to happen": "clausal",
    "Unusually": "adverbial",
}

# Some Mosteller expressions need their wording adapted for the template.
# "Frequent" in the data means the adjective, but we need "frequently" as adverb.
EXPRESSION_ADAPTATIONS = {
    "Frequent": "frequently",
    "Very frequent": "very frequently",
    "Infrequent": "infrequently",
    "Very infrequent": "very infrequently",
    "Not infrequent": "not infrequently",
    # These need the phrase lowercased for insertion into templates:
    # (handled generically below)
}


# ---------------------------------------------------------------------------
# Embedding helper
# ---------------------------------------------------------------------------

def embed_texts(texts: list[str], model: str = MODEL) -> np.ndarray:
    """Embed a list of texts via ollama API. Returns (n, dim) array."""
    resp = requests.post(OLLAMA_URL, json={"model": model, "input": texts})
    resp.raise_for_status()
    data = resp.json()
    embeddings = np.array(data["embeddings"], dtype=np.float64)
    return embeddings


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """L2-normalize each row."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)  # avoid division by zero
    return matrix / norms


def pairwise_cosines(vectors: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine similarity matrix for normalized vectors."""
    normed = normalize_rows(vectors)
    return normed @ normed.T


def mean_off_diagonal(matrix: np.ndarray) -> float:
    """Mean of off-diagonal elements (excludes self-similarity)."""
    n = matrix.shape[0]
    mask = ~np.eye(n, dtype=bool)
    return float(matrix[mask].mean())


def std_off_diagonal(matrix: np.ndarray) -> float:
    """Std of off-diagonal elements."""
    n = matrix.shape[0]
    mask = ~np.eye(n, dtype=bool)
    return float(matrix[mask].std())


# ---------------------------------------------------------------------------
# Part A: Consistency Test
# ---------------------------------------------------------------------------

def run_part_a():
    """
    Test: Does "probably" produce a more consistent direction than "reportedly"?

    If hedge vectors are significantly more consistent than control vectors,
    the direction is specifically epistemic, not just "adverb insertion."
    """
    print("=" * 70)
    print("PART A: Consistency of 'probably' vs 'reportedly' difference vectors")
    print(f"Model: {MODEL}")
    print(f"Claims: {len(CLAIM_TRIPLES)}")
    print("=" * 70)
    print()

    bare_texts = [t[0] for t in CLAIM_TRIPLES]
    hedge_texts = [t[1] for t in CLAIM_TRIPLES]
    control_texts = [t[2] for t in CLAIM_TRIPLES]

    # Embed all at once for efficiency
    all_texts = bare_texts + hedge_texts + control_texts
    print(f"Embedding {len(all_texts)} sentences...")
    all_embeddings = embed_texts(all_texts)

    n = len(CLAIM_TRIPLES)
    e_bare = all_embeddings[:n]
    e_hedge = all_embeddings[n:2*n]
    e_control = all_embeddings[2*n:]

    # Difference vectors (the raw geometric modification each adverb produces)
    diff_hedge = e_hedge - e_bare
    diff_control = e_control - e_bare

    # Consistency: how similar are the difference vectors to each other?
    cos_hedge = pairwise_cosines(diff_hedge)
    cos_control = pairwise_cosines(diff_control)

    mean_hedge = mean_off_diagonal(cos_hedge)
    std_hedge = std_off_diagonal(cos_hedge)
    mean_control = mean_off_diagonal(cos_control)
    std_control = std_off_diagonal(cos_control)

    print(f"Hedge ('probably') difference vectors:")
    print(f"  Mean pairwise cosine: {mean_hedge:.4f} (std: {std_hedge:.4f})")
    print(f"Control ('reportedly') difference vectors:")
    print(f"  Mean pairwise cosine: {mean_control:.4f} (std: {std_control:.4f})")
    print()

    # Direction comparison: is the hedge direction different from control?
    mean_hedge_dir = normalize_rows(diff_hedge.mean(axis=0, keepdims=True))[0]
    mean_control_dir = normalize_rows(diff_control.mean(axis=0, keepdims=True))[0]
    hedge_vs_control = float(mean_hedge_dir @ mean_control_dir)
    print(f"Cosine between mean hedge direction and mean control direction: "
          f"{hedge_vs_control:.4f}")
    print()

    # Interpretation
    print("--- Interpretation ---")
    if mean_hedge > 0.5:
        print("STRONG: 'Probably' produces a highly consistent direction.")
    elif mean_hedge > 0.3:
        print("MODERATE: 'Probably' has a discernible direction, with noise.")
    elif mean_hedge > 0.1:
        print("WEAK: Some consistency, but the direction is noisy.")
    else:
        print("NONE: No consistent direction detected for 'probably'.")

    if mean_hedge > mean_control + 0.1:
        print("GOOD: Hedge direction is more consistent than control.")
        print("      This suggests the direction is specifically epistemic.")
    elif abs(mean_hedge - mean_control) < 0.1:
        print("NEUTRAL: Hedge and control have similar consistency.")
        print("         The direction may be 'adverb insertion' generically.")
    else:
        print("UNEXPECTED: Control is more consistent than hedge.")
        print("            Investigate whether template phrasing is confounding.")

    if abs(hedge_vs_control) < 0.5:
        print("DISTINCT: Hedge and control point in different directions.")
    else:
        print("SIMILAR: Hedge and control point in similar directions.")
        print("         The modification may not be specifically epistemic.")

    print()

    # Also report the magnitude of difference vectors (are they large or tiny?)
    hedge_magnitudes = np.linalg.norm(diff_hedge, axis=1)
    control_magnitudes = np.linalg.norm(diff_control, axis=1)
    print(f"Hedge diff magnitude: mean={hedge_magnitudes.mean():.4f}, "
          f"std={hedge_magnitudes.std():.4f}")
    print(f"Control diff magnitude: mean={control_magnitudes.mean():.4f}, "
          f"std={control_magnitudes.std():.4f}")
    print()

    return {
        "mean_hedge_consistency": mean_hedge,
        "mean_control_consistency": mean_control,
        "hedge_vs_control_cosine": hedge_vs_control,
        "mean_hedge_direction": mean_hedge_dir,
    }


# ---------------------------------------------------------------------------
# Part B: Mosteller Projection
# ---------------------------------------------------------------------------

def build_mosteller_sentences():
    """
    For each Mosteller expression with data, construct a sentence using
    the appropriate syntactic template. Returns list of
    (expression, median, iqr, sentence, bare_sentence).
    """
    df = pd.read_csv(MOSTELLER_CSV)
    df = df.dropna(subset=["Median"])

    results = []
    skipped = []

    for _, row in df.iterrows():
        expr = row["Expression"]
        median = row["Median"]
        iqr = row["IQR"]

        # Look up syntactic type
        syntax = EXPRESSION_SYNTAX.get(expr)
        if syntax is None:
            skipped.append(expr)
            continue

        template = SYNTACTIC_TEMPLATES[syntax]

        # Adapt the phrase for insertion (lowercase, morphological changes)
        phrase = EXPRESSION_ADAPTATIONS.get(expr, expr.lower())

        sentence = template.replace("{PHRASE}", phrase)

        # The bare sentence is the template with the phrase removed.
        # We need a consistent baseline per syntactic type.
        if syntax == "predicative":
            bare = "The experiment will succeed"
        elif syntax == "adverbial":
            bare = "The experiment will succeed"
        elif syntax == "noun_phrase":
            bare = "The experiment will succeed"
        elif syntax == "clausal":
            bare = "The experiment will succeed"
        else:
            bare = "The experiment will succeed"

        results.append({
            "expression": expr,
            "median": median,
            "iqr": iqr,
            "sentence": sentence,
            "bare": bare,
            "syntax": syntax,
        })

    if skipped:
        print(f"  (Skipped {len(skipped)} expressions without syntax mapping: "
              f"{skipped[:5]}{'...' if len(skipped) > 5 else ''})")

    return results


def run_part_b(hedge_direction_from_a=None):
    """
    Test: Do Mosteller expressions project onto a low-dimensional subspace
    whose principal axis correlates with calibrated probability?
    """
    print("=" * 70)
    print("PART B: Mosteller Projection and PCA")
    print(f"Model: {MODEL}")
    print("=" * 70)
    print()

    items = build_mosteller_sentences()
    print(f"Expressions with valid templates: {len(items)}")
    print()

    # Embed all sentences + the bare baseline
    sentences = [item["sentence"] for item in items]
    bare_sentence = items[0]["bare"]  # Same for all in this version

    # Embed bare once, then all hedged sentences
    all_to_embed = [bare_sentence] + sentences
    print(f"Embedding {len(all_to_embed)} sentences...")
    all_embeddings = embed_texts(all_to_embed)

    e_bare = all_embeddings[0]  # single vector
    e_phrases = all_embeddings[1:]  # one per Mosteller expression

    # Difference vectors: how does each phrase modify the bare claim?
    diffs = e_phrases - e_bare  # (n_expressions, dim)

    medians = np.array([item["median"] for item in items])
    iqrs = np.array([item["iqr"] for item in items])
    expressions = [item["expression"] for item in items]

    # --- PCA on difference vectors ---
    pca = PCA(n_components=min(10, len(items)))
    pca.fit(diffs)

    print("PCA on difference vectors (how much structure exists):")
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    for i in range(min(5, len(cumvar))):
        print(f"  PC{i+1}: {pca.explained_variance_ratio_[i]*100:.1f}% "
              f"(cumulative: {cumvar[i]*100:.1f}%)")
    print()

    # --- Correlation of PC1 with Mosteller median ---
    projections = pca.transform(diffs)
    pc1 = projections[:, 0]

    # Pearson and Spearman correlations
    r_pearson, p_pearson = stats.pearsonr(pc1, medians)
    r_spearman, p_spearman = stats.spearmanr(pc1, medians)

    print(f"PC1 vs. Mosteller Median:")
    print(f"  Pearson r = {r_pearson:.4f} (p = {p_pearson:.2e})")
    print(f"  Spearman ρ = {r_spearman:.4f} (p = {p_spearman:.2e})")
    print()

    # Check if sign needs flipping (PC direction is arbitrary)
    if r_pearson < 0:
        print("  (PC1 is negatively correlated — flipping sign for interpretability)")
        pc1 = -pc1
        r_pearson = -r_pearson
        r_spearman = -r_spearman
        print(f"  After flip: Pearson r = {r_pearson:.4f}, Spearman ρ = {r_spearman:.4f}")
        print()

    # --- PC2 correlation (is there a second meaningful axis?) ---
    if projections.shape[1] >= 2:
        pc2 = projections[:, 1]
        r2_p, p2_p = stats.pearsonr(pc2, medians)
        r2_s, p2_s = stats.spearmanr(pc2, medians)
        print(f"PC2 vs. Mosteller Median:")
        print(f"  Pearson r = {r2_p:.4f} (p = {p2_p:.2e})")
        print(f"  Spearman ρ = {r2_s:.4f} (p = {p2_s:.2e})")
        print()

    # --- IQR prediction: does consistency predict ambiguity? ---
    # For this, we'd need per-expression consistency across multiple claims.
    # With a single bare sentence, we can't measure this directly.
    # Instead, check if IQR correlates with diff vector magnitude
    # (hypothesis: ambiguous phrases might produce smaller/larger modifications?)
    diff_magnitudes = np.linalg.norm(diffs, axis=1)
    r_iqr_mag, p_iqr_mag = stats.spearmanr(diff_magnitudes, iqrs)
    print(f"Diff vector magnitude vs. IQR:")
    print(f"  Spearman ρ = {r_iqr_mag:.4f} (p = {p_iqr_mag:.2e})")
    print(f"  (Positive = ambiguous phrases produce larger modifications)")
    print()

    # --- Cross-validate with Vogel if available ---
    vogel_corr = cross_validate_vogel(items, pc1, pca, diffs)

    # --- Compare Part A hedge direction with Part B PC1 ---
    if hedge_direction_from_a is not None:
        pc1_dir = pca.components_[0]
        alignment = float(np.abs(hedge_direction_from_a @ pc1_dir))
        print(f"Alignment of Part A 'probably' direction with Part B PC1: "
              f"{alignment:.4f}")
        if alignment > 0.7:
            print("  STRONG: 'Probably' direction aligns with the principal hedge axis.")
        elif alignment > 0.4:
            print("  MODERATE: Some alignment, but 'probably' doesn't fully capture PC1.")
        else:
            print("  WEAK: 'Probably' direction is not the same as the Mosteller PC1.")
            print("         This might mean the hypothesis is wrong, or that 'probably'")
            print("         is not representative of the general hedge axis.")
        print()

    # --- Report notable outliers ---
    print("--- Top 5 expressions by PC1 projection (most 'certain' end) ---")
    sorted_idx = np.argsort(pc1)[::-1]
    for i in sorted_idx[:5]:
        print(f"  {expressions[i]:25s}  PC1={pc1[i]:+.4f}  "
              f"Median={medians[i]:.1f}%  IQR={iqrs[i]:.1f}")

    print("--- Bottom 5 expressions by PC1 projection (most 'uncertain' end) ---")
    for i in sorted_idx[-5:]:
        print(f"  {expressions[i]:25s}  PC1={pc1[i]:+.4f}  "
              f"Median={medians[i]:.1f}%  IQR={iqrs[i]:.1f}")
    print()

    # --- Tier analysis: do Tier 1/2/3 phrases behave differently? ---
    tier1_mask = iqrs < 5
    tier2_mask = (iqrs >= 5) & (iqrs <= 20)
    tier3_mask = iqrs > 20

    print(f"By tier (IQR-based):")
    if tier1_mask.any():
        r_t1, _ = stats.spearmanr(pc1[tier1_mask], medians[tier1_mask]) \
            if tier1_mask.sum() > 2 else (float('nan'), None)
        print(f"  Tier 1 (IQR<5, n={tier1_mask.sum()}): "
              f"Spearman ρ(PC1, median) = {r_t1:.4f}")
    if tier2_mask.any():
        r_t2, _ = stats.spearmanr(pc1[tier2_mask], medians[tier2_mask]) \
            if tier2_mask.sum() > 2 else (float('nan'), None)
        print(f"  Tier 2 (5≤IQR≤20, n={tier2_mask.sum()}): "
              f"Spearman ρ(PC1, median) = {r_t2:.4f}")
    if tier3_mask.any():
        r_t3, _ = stats.spearmanr(pc1[tier3_mask], medians[tier3_mask]) \
            if tier3_mask.sum() > 2 else (float('nan'), None)
        print(f"  Tier 3 (IQR>20, n={tier3_mask.sum()}): "
              f"Spearman ρ(PC1, median) = {r_t3:.4f}")
    print()

    # --- Interpretation ---
    print("--- Interpretation ---")
    if abs(r_pearson) > 0.7:
        print("STRONG: PC1 is well-correlated with calibrated probability.")
        print("        The hedge subspace is likely real and calibrated.")
    elif abs(r_pearson) > 0.4:
        print("MODERATE: PC1 has meaningful correlation with probability.")
        print("          Structure exists but is noisy or multi-dimensional.")
    elif abs(r_pearson) > 0.2:
        print("WEAK: Some correlation, but not enough to rely on geometrically.")
    else:
        print("NONE: PC1 does not predict probability for this model.")

    if pca.explained_variance_ratio_[0] > 0.5:
        print("CONCENTRATED: >50% variance in PC1. Hedging is approximately 1D.")
    elif pca.explained_variance_ratio_[0] > 0.3:
        print("MODERATE SPREAD: PC1 dominant but other dimensions matter.")
    else:
        print("DIFFUSE: Hedge modifications are spread across many dimensions.")
        print("         The 'single axis' hypothesis is likely too strong.")

    print()
    return {
        "pca_explained": pca.explained_variance_ratio_[:5].tolist(),
        "pearson_r": r_pearson,
        "spearman_r": r_spearman,
        "n_expressions": len(items),
    }


def cross_validate_vogel(items, pc1, pca, diffs):
    """
    If Vogel data is available, test whether the projection function learned
    from Mosteller also predicts Vogel's independent means.
    """
    if not VOGEL_CSV.exists():
        return None

    vogel = pd.read_csv(VOGEL_CSV)
    vogel = vogel.dropna(subset=["Mean"])

    # Match Vogel expressions to our embedded items
    expr_to_idx = {item["expression"]: i for i, item in enumerate(items)}

    matched = []
    for _, row in vogel.iterrows():
        expr = row["Expression"]
        if expr in expr_to_idx:
            idx = expr_to_idx[expr]
            matched.append((expr, pc1[idx], row["Mean"]))

    if len(matched) < 5:
        print(f"Vogel cross-validation: only {len(matched)} matched expressions "
              f"(need ≥5). Skipping.")
        return None

    expressions, projections, vogel_means = zip(*matched)
    projections = np.array(projections)
    vogel_means = np.array(vogel_means)

    r_v, p_v = stats.spearmanr(projections, vogel_means)
    print(f"Vogel cross-validation ({len(matched)} matched expressions):")
    print(f"  Spearman ρ(PC1, Vogel mean) = {r_v:.4f} (p = {p_v:.2e})")
    if abs(r_v) > 0.6:
        print("  VALIDATES: Projection generalizes to independent meta-analysis data.")
    else:
        print("  DOES NOT VALIDATE: Projection doesn't generalize to Vogel data.")
    print()

    return r_v


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print()
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║  Experiment 01: Epistemic Hedging as Linear Structure              ║")
    print("╠══════════════════════════════════════════════════════════════════════╣")
    print(f"║  Model: {MODEL:<58s}║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print()

    # Verify ollama is running and model is available
    try:
        test = embed_texts(["test"], model=MODEL)
        dim = test.shape[1]
        print(f"Model loaded. Embedding dimension: {dim}")
        print()
    except Exception as e:
        print(f"ERROR: Could not connect to ollama or load model '{MODEL}'.")
        print(f"  {e}")
        print(f"  Is ollama running? Try: ollama serve")
        print(f"  Is the model pulled? Try: ollama pull {MODEL}")
        sys.exit(1)

    # Run Part A
    part_a_results = run_part_a()

    # Run Part B, passing the hedge direction from Part A for comparison
    part_b_results = run_part_b(
        hedge_direction_from_a=part_a_results.get("mean_hedge_direction")
    )

    # --- Summary ---
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print(f"Model: {MODEL}")
    print()
    print("Part A (Consistency):")
    print(f"  'Probably' consistency:   {part_a_results['mean_hedge_consistency']:.4f}")
    print(f"  'Reportedly' consistency: {part_a_results['mean_control_consistency']:.4f}")
    print(f"  Hedge-control separation: {part_a_results['hedge_vs_control_cosine']:.4f}")
    print()
    print("Part B (Mosteller Projection):")
    print(f"  PC1 explained variance:   {part_b_results['pca_explained'][0]*100:.1f}%")
    print(f"  PC1-Median Pearson r:     {part_b_results['pearson_r']:.4f}")
    print(f"  PC1-Median Spearman ρ:    {part_b_results['spearman_r']:.4f}")
    print()

    # Overall assessment
    print("--- Overall Assessment ---")
    a_ok = part_a_results['mean_hedge_consistency'] > 0.3
    b_ok = abs(part_b_results['pearson_r']) > 0.4
    separated = abs(part_a_results['hedge_vs_control_cosine']) < 0.7

    if a_ok and b_ok and separated:
        print("PROCEED: Both tests show positive signal.")
        print("  → Experiment 2 (multi-phrase, multi-claim) is warranted.")
    elif b_ok and not a_ok:
        print("INTERESTING: Mosteller projection works but single-phrase consistency")
        print("  is weak. The structure may be cross-phrase but not content-independent.")
        print("  → Investigate with more claims and phrases before concluding.")
    elif a_ok and not b_ok:
        print("PARTIAL: Consistent direction exists but doesn't predict probability.")
        print("  → The direction may be syntactic (adverb position) not semantic.")
        print("  → Try different syntactic templates or a different model.")
    elif not a_ok and not b_ok:
        print("NEGATIVE: No signal detected with this model.")
        print("  → Try a different model before abandoning the hypothesis.")
    else:
        print("MIXED: Results don't fit clean categories. Examine details above.")

    print()
    print("Next steps are documented in EXPERIMENT-PLAN.md")
    print()


if __name__ == "__main__":
    main()
