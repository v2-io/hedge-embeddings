"""
Experiment 04: IQR-Consistency Relationship

The novel prediction: phrases with low IQR (high human consensus about what
they mean) should produce more geometrically consistent difference vectors
across diverse claims than phrases with high IQR (high human disagreement).

Intuition: if "certain" means the same thing to everyone, embedding it in
different sentence contexts should produce similar geometric modifications.
If "possible" means different things to different people (Mosteller shows
bimodal distribution, IQR = 42.7), the geometric modification might be less
stable — reflecting the word's genuine semantic ambiguity.

This would mean the embedding model has learned not just WHAT phrases mean
on average, but HOW STABLE that meaning is across contexts. The geometry
would encode interpretive precision, not just central tendency.

Design:
  - 20 diverse claims, each embedded with each of 13 predicative phrases
  - For each phrase: compute all 20 difference vectors, measure pairwise cosine
  - Correlate per-phrase consistency with negative IQR from Mosteller

Prediction: Spearman ρ(consistency, -IQR) > 0.5

Caution: This prediction assumes the model's context-sensitivity reflects
human interpretive variance. This is NOT obvious — the model might encode
a single "meaning" regardless of human disagreement. A null result doesn't
invalidate the broader hypothesis, just this specific prediction.

References:
  - EXPERIMENT-PLAN.md (IQR prediction, "Remaining Questions" #2)
  - FINDINGS-01.md (initial evidence: diff magnitude vs IQR was ρ=-0.24, ns)
  - docs/mosteller_youtz_1990_full.csv (IQR data)
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


def normalize_rows(matrix):
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    return matrix / norms


# ---------------------------------------------------------------------------
# Claims: 20 diverse propositions, all grammatically natural with
# "It is {phrase} that [claim]"
#
# Criteria: diverse domains, not about probability, similar length,
# natural with all 13 predicative phrases.
# ---------------------------------------------------------------------------

CLAIMS = [
    "the experiment will succeed",
    "the surgery will be successful",
    "the negotiations will reach an agreement",
    "the crop will survive the drought",
    "the team will win the championship",
    "the startup will become profitable",
    "the bridge will withstand the earthquake",
    "the medication will reduce symptoms",
    "the algorithm will find the optimal solution",
    "the reform will improve the economy",
    "the student will pass the exam",
    "the satellite will reach orbit",
    "the vaccine will prevent infection",
    "the peace talks will end the conflict",
    "the investment will generate returns",
    "the engine will start in cold weather",
    "the species will adapt to the new climate",
    "the project will finish on schedule",
    "the election will be conducted fairly",
    "the prediction will prove accurate",
]

# Predicative phrases with their Mosteller IQR values
# Sorted by IQR for clarity in output
PHRASES = [
    # (phrase, median, IQR)
    ("Impossible", 0.3, 0.3),       # Tier 1 — very tight consensus
    ("Certain", 99.6, 1.1),         # Tier 1
    ("Very improbable", 4.8, 5.9),  # Tier 2 — moderate
    ("Very unlikely", 5.0, 7.1),    # Tier 2
    ("Almost certain", 90.2, 7.5),  # Tier 2
    ("Very probable", 89.7, 8.9),   # Tier 2
    ("Very likely", 87.5, 10.1),    # Tier 2
    ("Probable", 70.2, 13.0),       # Tier 2
    ("Unlikely", 17.2, 13.0),       # Tier 2
    ("Improbable", 12.5, 14.7),     # Tier 2
    ("Likely", 71.1, 15.0),         # Tier 2
    ("Not unreasonable", 37.6, 29.1), # Tier 3 — high ambiguity
    ("Possible", 38.5, 42.7),       # Tier 3 — VERY high ambiguity (bimodal!)
]

TEMPLATE = "It is {PHRASE} that {CLAIM}"


def compute_phrase_consistency(phrase, claims, bare_embeddings):
    """
    Embed the phrase with each claim, compute difference vectors,
    return mean pairwise cosine (consistency measure).
    """
    sentences = [TEMPLATE.replace("{PHRASE}", phrase.lower()).replace("{CLAIM}", c)
                 for c in claims]
    phrase_embeddings = embed_texts(sentences)
    diffs = phrase_embeddings - bare_embeddings
    diffs_normed = normalize_rows(diffs)

    # Pairwise cosine similarity
    cos_matrix = diffs_normed @ diffs_normed.T
    n = len(claims)
    mask = ~np.eye(n, dtype=bool)
    mean_cos = float(cos_matrix[mask].mean())
    std_cos = float(cos_matrix[mask].std())
    min_cos = float(cos_matrix[mask].min())

    return mean_cos, std_cos, min_cos


def main():
    print()
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║  Experiment 04: IQR-Consistency Relationship                       ║")
    print("╠══════════════════════════════════════════════════════════════════════╣")
    print(f"║  Model: {MODEL:<58s}║")
    print("║                                                                    ║")
    print("║  Prediction: Phrases with low IQR (high consensus) produce more   ║")
    print("║  consistent difference vectors across diverse claims.             ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print()

    try:
        test = embed_texts(["test"])
        dim = test.shape[1]
        print(f"Model loaded. Dim: {dim}")
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    # Embed all bare claims
    print(f"Claims: {len(CLAIMS)}")
    print(f"Phrases: {len(PHRASES)}")
    print(f"Total sentences to embed: {len(CLAIMS)} bare + {len(CLAIMS) * len(PHRASES)} hedged")
    print()

    bare_sentences = [f"It is true that {c}" for c in CLAIMS]
    # Actually — the bare sentence should be just the claim without hedging.
    # But we need a consistent frame. Two options:
    #   1. Bare = "The experiment will succeed" (no frame)
    #   2. Bare = "It is true that the experiment will succeed" (neutral frame)
    # Option 2 keeps the syntactic frame constant. "True" should be maximally
    # certain (no hedging). Let's use the raw claim as bare.
    bare_sentences = CLAIMS  # Actually just embed the raw claim
    # Wait — the template adds "It is {phrase} that {claim}". The bare version
    # should be the claim without the "It is {phrase} that" wrapper.
    # The difference vector then captures the full effect of the wrapper.
    # This is what we did in Experiment 1b.

    print("Embedding bare claims...")
    bare_embeddings = embed_texts(bare_sentences)

    # Compute consistency for each phrase
    print("Computing per-phrase consistency across 20 claims...")
    print()

    results = []
    print(f"{'Phrase':<20s}  {'IQR':>5s}  {'Median':>6s}  "
          f"{'Consistency':>11s}  {'Std':>5s}  {'Min':>5s}  Tier")
    print("─" * 75)

    for phrase, median, iqr in PHRASES:
        mean_cos, std_cos, min_cos = compute_phrase_consistency(
            phrase, CLAIMS, bare_embeddings)

        tier = "1" if iqr < 5 else ("2" if iqr <= 20 else "3")
        results.append({
            "phrase": phrase, "median": median, "iqr": iqr,
            "consistency": mean_cos, "std": std_cos, "min": min_cos, "tier": tier,
        })

        print(f"  {phrase:<18s}  {iqr:5.1f}  {median:6.1f}%  "
              f"{mean_cos:11.4f}  {std_cos:5.3f}  {min_cos:5.3f}  {tier}")

    # --- Correlation analysis ---
    print()
    print("═" * 75)
    print("CORRELATION ANALYSIS")
    print("═" * 75)
    print()

    iqrs = np.array([r["iqr"] for r in results])
    consistencies = np.array([r["consistency"] for r in results])
    medians = np.array([r["median"] for r in results])

    # Primary test: consistency vs. -IQR
    r_iqr, p_iqr = stats.spearmanr(consistencies, -iqrs)
    print(f"Consistency vs. -IQR (primary prediction):")
    print(f"  Spearman ρ = {r_iqr:+.4f} (p = {p_iqr:.4e})")

    if abs(r_iqr) > 0.5:
        print(f"  → SUPPORTED: Phrases with low IQR produce more consistent directions.")
    elif abs(r_iqr) > 0.3:
        print(f"  → MODERATE: Some evidence for the prediction.")
    else:
        print(f"  → NOT SUPPORTED: Consistency is not related to IQR.")

    # Secondary: is consistency confounded with probability extremity?
    # Phrases near 0% and 100% might be more consistent simply because
    # the model has less ambiguity about extreme values.
    extremity = np.abs(medians - 50)  # distance from 50%
    r_ext, p_ext = stats.spearmanr(consistencies, extremity)
    print(f"\nConsistency vs. probability extremity (confound check):")
    print(f"  Spearman ρ = {r_ext:+.4f} (p = {p_ext:.4e})")
    if abs(r_ext) > 0.5:
        print(f"  → NOTE: Consistency correlates with extremity.")
        print(f"     The IQR result may be confounded (extreme phrases have low IQR).")

    # Partial: IQR effect after controlling for extremity?
    # Simple approach: residualize consistency on extremity, then correlate with IQR
    slope_e, intercept_e, _, _, _ = stats.linregress(extremity, consistencies)
    resid_consistency = consistencies - (slope_e * extremity + intercept_e)
    r_partial, p_partial = stats.spearmanr(resid_consistency, -iqrs)
    print(f"\nConsistency vs. -IQR (after removing extremity effect):")
    print(f"  Spearman ρ = {r_partial:+.4f} (p = {p_partial:.4e})")
    if abs(r_partial) > 0.3:
        print(f"  → IQR effect survives controlling for extremity.")
    else:
        print(f"  → IQR effect does not survive controlling for extremity.")
        print(f"     Consistency may be driven by probability magnitude, not ambiguity.")

    # --- Also check: does consistency correlate with the Mosteller median? ---
    r_med, p_med = stats.spearmanr(consistencies, medians)
    print(f"\nConsistency vs. Mosteller median:")
    print(f"  Spearman ρ = {r_med:+.4f} (p = {p_med:.4e})")

    # --- Tier comparison ---
    print()
    print("═" * 75)
    print("TIER COMPARISON")
    print("═" * 75)
    print()

    for tier in ["1", "2", "3"]:
        tier_items = [r for r in results if r["tier"] == tier]
        if tier_items:
            tier_cons = [r["consistency"] for r in tier_items]
            tier_iqrs = [r["iqr"] for r in tier_items]
            print(f"  Tier {tier} (n={len(tier_items)}, "
                  f"IQR range {min(tier_iqrs):.1f}–{max(tier_iqrs):.1f}):")
            print(f"    Mean consistency: {np.mean(tier_cons):.4f} "
                  f"(range: {min(tier_cons):.4f}–{max(tier_cons):.4f})")

    # --- Visual: scatter-like display ---
    print()
    print("═" * 75)
    print("VISUAL: IQR vs. Consistency")
    print("═" * 75)
    print()
    print(f"  {'Phrase':<20s}  IQR{'':>4s}  Consistency (higher = more consistent)")
    for r in sorted(results, key=lambda x: x["iqr"]):
        bar_len = int(r["consistency"] * 50)
        bar = "█" * bar_len + "░" * (50 - bar_len)
        print(f"  {r['phrase']:<20s}  {r['iqr']:5.1f}  {bar} {r['consistency']:.3f}")

    # --- Summary ---
    print()
    print("═" * 75)
    print("SUMMARY")
    print("═" * 75)
    print()
    print(f"  Primary: ρ(consistency, -IQR) = {r_iqr:+.4f} (p = {p_iqr:.4e})")
    print(f"  Confound: ρ(consistency, extremity) = {r_ext:+.4f}")
    print(f"  Partial: ρ(consistency, -IQR | extremity) = {r_partial:+.4f}")
    print()

    if abs(r_iqr) > 0.5 and abs(r_partial) > 0.3:
        print("  FINDING: The embedding model encodes BOTH probability level AND")
        print("  interpretive precision. Ambiguous phrases are geometrically less")
        print("  stable across contexts. The geometry captures the very thing that")
        print("  makes Tier 3 phrases dangerous — their semantic instability.")
    elif abs(r_iqr) > 0.5 and abs(r_partial) < 0.3:
        print("  FINDING: Consistency correlates with IQR, but this is explained")
        print("  by probability extremity. Extreme phrases (near 0% or 100%) are")
        print("  both more consistent AND have lower IQR. The model encodes")
        print("  probability magnitude, but not specifically interpretive precision.")
    elif abs(r_iqr) < 0.3:
        print("  FINDING: No evidence that interpretive precision is encoded in")
        print("  the embedding geometry. The model assigns consistent directions")
        print("  regardless of human disagreement about phrase meaning.")
    else:
        print("  MIXED: Moderate evidence. Further investigation needed.")

    print()


if __name__ == "__main__":
    main()
