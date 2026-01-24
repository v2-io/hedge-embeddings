"""
Experiment 03: Modal Adverb Axis and Subspace Analysis

Key insight from Experiment 2: the adverbial axis (trained on frequency adverbs
like "always", "often", "rarely") doesn't calibrate modal adverbs (like
"probably", "certainly", "possibly") correctly. These share a syntactic slot
but encode different semantic types: base-rate frequency vs. epistemic confidence.

This experiment:
1. Trains a MODAL ADVERB axis using Mosteller predicative adjectives converted
   to their adverbial forms (probable→probably, certain→certainly, etc.)
2. Tests whether this new axis handles the natural hedges better
3. Analyzes the subspace structure: how are the 4 axes related?
4. Measures how much of each novel phrase's difference vector lies in the
   trained subspace (diagnostic for attenuation)

If the modal axis works, the practical system needs 4 types, not 3:
  - Predicative: "It is {likely/certain/possible} that X"
  - Frequency adverb: "X {always/often/rarely} happens"
  - Noun phrase: "There is a {high/low} chance that X"
  - Modal adverb: "X will {probably/certainly/possibly} happen"

References:
  - FINDINGS-02.md (compression problem for "probably", "certainly")
  - experiment_01b_within_type.py (original 3 axes)
"""

import sys
import numpy as np
import requests
from pathlib import Path
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


def train_axis(expressions, template, bare_emb):
    """Train supervised probability axis. Returns (axis, slope, intercept)."""
    phrases = list(expressions.keys())
    medians = np.array([expressions[p] for p in phrases])
    sentences = [template.replace("{PHRASE}", p.lower()) for p in phrases]
    embeddings = embed_texts(sentences)
    diffs = embeddings - bare_emb
    medians_centered = (medians - medians.mean()) / medians.std()
    dim = diffs.shape[1]
    w = np.linalg.solve(diffs.T @ diffs + 0.1 * np.eye(dim),
                        diffs.T @ medians_centered)
    w = normalize(w)
    projections = diffs @ w
    slope, intercept, _, _, _ = stats.linregress(projections, medians)
    r, _ = stats.spearmanr(projections, medians)
    return w, slope, intercept, r


# ---------------------------------------------------------------------------
# Axis training data
# ---------------------------------------------------------------------------

BARE_CLAIM = "The experiment will succeed"

# Original 3 axes from Experiment 1b
PREDICATIVE = {
    "Certain": 99.6, "Almost certain": 90.2, "Very likely": 87.5,
    "Likely": 71.1, "Probable": 70.2, "Very probable": 89.7,
    "Possible": 38.5, "Unlikely": 17.2, "Very unlikely": 5.0,
    "Improbable": 12.5, "Very improbable": 4.8, "Impossible": 0.3,
    "Not unreasonable": 37.6,
}
PREDICATIVE_TEMPLATE = "It is {PHRASE} that the experiment will succeed"

FREQUENCY = {
    "Always": 99.7, "Almost always": 91.7, "Very often": 82.8,
    "Often": 72.5, "Usually": 75.1, "Sometimes": 25.0,
    "Occasionally": 20.0, "Seldom": 10.2, "Very seldom": 4.9,
    "Rarely": 7.2, "Very rarely": 3.0, "Almost never": 2.9,
    "Never": 0.3, "Not often": 19.7, "Not very often": 10.1,
    "As often as not": 50.0, "More often than not": 59.8,
    "Once in a while": 15.3, "Now and then": 15.1,
}
FREQUENCY_TEMPLATE = "The experiment will {PHRASE} succeed"

NOUN_PHRASE = {
    "Very high probability": 92.5, "High probability": 82.3,
    "Moderate probability": 52.4, "Low probability": 15.0,
    "Very low probability": 4.9, "High chance": 80.4,
    "Poor chance": 10.3, "Low chance": 9.8, "Even chance": 50.0,
    "Better than even chance": 57.6, "Less than an even chance": 40.2,
}
NOUN_PHRASE_TEMPLATE = "There is a {PHRASE} that the experiment will succeed"

# NEW: Modal adverb axis
# These are the adverbial forms of the predicative Mosteller adjectives.
# Same Mosteller-calibrated probabilities, different syntactic frame.
MODAL_ADVERB = {
    "certainly": 99.6,       # from "Certain"
    "almost certainly": 90.2, # from "Almost certain"
    "very likely": 87.5,     # adverb form same as adjective
    "likely": 71.1,          # same
    "probably": 70.2,        # from "Probable"
    "possibly": 38.5,        # from "Possible"
    "unlikely": 17.2,        # adverb form (= adjective here)
    "very unlikely": 5.0,    # same
    "conceivably": 38.5,     # ≈ "Possible" (my estimate — NOT from Mosteller)
    "definitely": 99.6,      # ≈ "Certain" (my estimate — NOT from Mosteller)
    "perhaps": 38.5,         # ≈ "Possible" (my estimate)
    "maybe": 38.5,           # ≈ "Possible" (my estimate)
    "presumably": 70.2,      # ≈ "Probable" (my estimate)
    "undoubtedly": 95.0,     # ≈ high certainty (my estimate)
    "arguably": 55.0,        # moderate — debatable (my estimate)
}
MODAL_TEMPLATE = "The experiment will {PHRASE} succeed"

# Note: Some of these probabilities are MY ESTIMATES, not Mosteller data.
# I've marked them above. The Mosteller-derived ones are the core; the
# estimates extend coverage but introduce uncertainty. A future agent should
# consider whether the estimated values affect the axis quality.


# ---------------------------------------------------------------------------
# Novel phrases to test (same as Experiment 2, key subset)
# ---------------------------------------------------------------------------

NOVEL_PHRASES = [
    # Modal adverbs (should work better on modal axis)
    ("The experiment will probably succeed", "modal", 70.2),
    ("The experiment will certainly succeed", "modal", 99.6),
    ("The experiment will possibly succeed", "modal", 38.5),
    ("The experiment will likely succeed", "modal", 71.1),
    ("The experiment will definitely succeed", "modal", 99.6),
    ("The experiment will perhaps succeed", "modal", 38.5),
    ("The experiment will presumably succeed", "modal", 70.0),
    ("The experiment will conceivably succeed", "modal", 35.0),
    ("The experiment will undoubtedly succeed", "modal", 95.0),
    ("The experiment will arguably succeed", "modal", 55.0),

    # First-person frames (how do they project onto all 4 axes?)
    ("I think the experiment will succeed", "first_person", 65.0),
    ("I believe the experiment will succeed", "first_person", 70.0),
    ("I suspect the experiment will succeed", "first_person", 55.0),
    ("I doubt the experiment will succeed", "first_person", 20.0),
    ("I'm not sure the experiment will succeed", "first_person", 40.0),

    # Anti-hedges in sentence-initial position
    ("Obviously, the experiment will succeed", "anti_hedge", 95.0),
    ("Clearly, the experiment will succeed", "anti_hedge", 92.0),
    ("Without question, the experiment will succeed", "anti_hedge", 98.0),

    # Additional novel phrasings
    ("All indications suggest the experiment will succeed", "novel", 80.0),
    ("It's a safe bet that the experiment will succeed", "novel", 85.0),
    ("There's a slim chance the experiment will succeed", "novel", 15.0),
    ("The experiment might well succeed", "novel", 60.0),
    ("I wouldn't be surprised if the experiment succeeded", "novel", 55.0),
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print()
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║  Experiment 03: Modal Adverb Axis and Subspace Analysis            ║")
    print("╠══════════════════════════════════════════════════════════════════════╣")
    print(f"║  Model: {MODEL:<58s}║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print()

    try:
        test = embed_texts(["test"])
        dim = test.shape[1]
        print(f"Model loaded. Dim: {dim}\n")
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    bare_emb = embed_texts([BARE_CLAIM])[0]

    # --- Train all 4 axes ---
    print("Training axes...")
    ax_pred, sl_pred, int_pred, r_pred = train_axis(
        PREDICATIVE, PREDICATIVE_TEMPLATE, bare_emb)
    print(f"  Predicative:   ρ = {r_pred:.3f}")

    ax_freq, sl_freq, int_freq, r_freq = train_axis(
        FREQUENCY, FREQUENCY_TEMPLATE, bare_emb)
    print(f"  Frequency:     ρ = {r_freq:.3f}")

    ax_np, sl_np, int_np, r_np = train_axis(
        NOUN_PHRASE, NOUN_PHRASE_TEMPLATE, bare_emb)
    print(f"  Noun phrase:   ρ = {r_np:.3f}")

    ax_modal, sl_modal, int_modal, r_modal = train_axis(
        MODAL_ADVERB, MODAL_TEMPLATE, bare_emb)
    print(f"  Modal adverb:  ρ = {r_modal:.3f}")
    print()

    # --- Subspace analysis: how are the 4 axes related? ---
    print("─" * 60)
    print("  Axis relationships (cosine similarity)")
    print("─" * 60)
    axes = {"Pred": ax_pred, "Freq": ax_freq, "NP": ax_np, "Modal": ax_modal}
    ax_names = list(axes.keys())
    for i in range(len(ax_names)):
        for j in range(i+1, len(ax_names)):
            cos = float(axes[ax_names[i]] @ axes[ax_names[j]])
            print(f"  {ax_names[i]:5s} ↔ {ax_names[j]:5s}: {cos:+.4f}")
    print()

    # Are the axes spanning a subspace? Compute effective dimensionality
    ax_matrix = np.array([ax_pred, ax_freq, ax_np, ax_modal])
    _, sigmas, _ = np.linalg.svd(ax_matrix)
    print(f"  SVD of axis matrix: σ = [{', '.join(f'{s:.3f}' for s in sigmas)}]")
    print(f"  Effective rank: {(sigmas > 0.1).sum()} "
          f"(axes span a ~{(sigmas > 0.1).sum()}D subspace)")
    print()

    # --- Project novel phrases onto all 4 axes ---
    print("═" * 78)
    print(f"{'Phrase':<50s} {'Pred':>5s} {'Freq':>5s} {'NP':>5s} "
          f"{'Modal':>5s} {'Best':>5s} {'Intuit':>6s}")
    print("═" * 78)

    all_sentences = [p[0] for p in NOVEL_PHRASES]
    all_embs = embed_texts(all_sentences)
    all_diffs = all_embs - bare_emb

    results_by_cat = {}
    all_results = []

    for i, (sentence, category, intuition) in enumerate(NOVEL_PHRASES):
        diff = all_diffs[i]

        prob_pred = max(0, min(100, sl_pred * float(diff @ ax_pred) + int_pred))
        prob_freq = max(0, min(100, sl_freq * float(diff @ ax_freq) + int_freq))
        prob_np = max(0, min(100, sl_np * float(diff @ ax_np) + int_np))
        prob_modal = max(0, min(100, sl_modal * float(diff @ ax_modal) + int_modal))

        # Best axis: which one is closest to intuition?
        probs = [prob_pred, prob_freq, prob_np, prob_modal]
        errors = [abs(p - intuition) for p in probs]
        best_idx = np.argmin(errors)
        best_prob = probs[best_idx]
        best_label = ["Pred", "Freq", "NP", "Modal"][best_idx]

        # Subspace fraction: how much of diff lies in the 4-axis subspace?
        proj_onto_subspace = ax_matrix.T @ (ax_matrix @ diff)
        frac_in_subspace = np.linalg.norm(proj_onto_subspace) / np.linalg.norm(diff)

        result = {
            "sentence": sentence, "category": category, "intuition": intuition,
            "prob_pred": prob_pred, "prob_freq": prob_freq,
            "prob_np": prob_np, "prob_modal": prob_modal,
            "best_prob": best_prob, "best_axis": best_label,
            "subspace_frac": frac_in_subspace,
        }
        all_results.append(result)
        results_by_cat.setdefault(category, []).append(result)

        if i == 0 or NOVEL_PHRASES[i][1] != NOVEL_PHRASES[i-1][1]:
            print(f"--- {category} ---")

        short = sentence[:47] + "..." if len(sentence) > 50 else sentence
        print(f"  {short:<48s} {prob_pred:5.1f} {prob_freq:5.1f} "
              f"{prob_np:5.1f} {prob_modal:5.1f} {best_prob:5.1f} "
              f"{intuition:5.1f}%")

    # --- Category analysis ---
    print()
    print("═" * 78)
    print("ANALYSIS: Modal axis vs. others for modal adverbs")
    print("═" * 78)
    print()

    modal_items = results_by_cat.get("modal", [])
    if modal_items:
        # Compare: how well does each axis calibrate modal adverbs?
        intuitions = [it["intuition"] for it in modal_items]
        for ax_name, key in [("Predicative", "prob_pred"), ("Frequency", "prob_freq"),
                             ("Noun phrase", "prob_np"), ("Modal", "prob_modal")]:
            preds = [it[key] for it in modal_items]
            r, p = stats.spearmanr(preds, intuitions)
            mae = np.mean([abs(pr - in_) for pr, in_ in zip(preds, intuitions)])
            print(f"  {ax_name:12s}: ρ = {r:+.3f} (p={p:.3e}), MAE = {mae:.1f}%")
        print()

        # What does the MODAL axis give for each phrase?
        print("  Modal axis predictions for modal adverbs:")
        for it in sorted(modal_items, key=lambda x: -x["prob_modal"]):
            short = it["sentence"][:45]
            print(f"    {short:<45s}  Modal={it['prob_modal']:5.1f}%  "
                  f"Mosteller={it['intuition']:5.1f}%  "
                  f"Error={abs(it['prob_modal']-it['intuition']):4.1f}%")

    # --- First-person analysis ---
    print()
    print("═" * 78)
    print("ANALYSIS: First-person frames on each axis")
    print("═" * 78)
    print()

    fp_items = results_by_cat.get("first_person", [])
    if fp_items:
        intuitions = [it["intuition"] for it in fp_items]
        for ax_name, key in [("Predicative", "prob_pred"), ("Frequency", "prob_freq"),
                             ("Noun phrase", "prob_np"), ("Modal", "prob_modal")]:
            preds = [it[key] for it in fp_items]
            r, p = stats.spearmanr(preds, intuitions)
            mae = np.mean([abs(pr - in_) for pr, in_ in zip(preds, intuitions)])
            print(f"  {ax_name:12s}: ρ = {r:+.3f} (p={p:.3e}), MAE = {mae:.1f}%")

    # --- Subspace coverage ---
    print()
    print("═" * 78)
    print("SUBSPACE COVERAGE: Fraction of diff vector in trained 4-axis space")
    print("═" * 78)
    print()
    print(f"  {'Category':<15s}  {'Mean frac':>9s}  {'Min':>5s}  {'Max':>5s}  Interpretation")
    for cat in ["modal", "first_person", "anti_hedge", "novel"]:
        items = results_by_cat.get(cat, [])
        if items:
            fracs = [it["subspace_frac"] for it in items]
            mean_f = np.mean(fracs)
            interp = ("good coverage" if mean_f > 0.5 else
                     "moderate" if mean_f > 0.3 else "poor — off-axis")
            print(f"  {cat:<15s}  {mean_f:9.3f}  {min(fracs):5.3f}  "
                  f"{max(fracs):5.3f}  {interp}")

    # --- Overall summary ---
    print()
    print("═" * 78)
    print("SUMMARY")
    print("═" * 78)
    print()

    # Compare overall correlation using modal axis vs. frequency axis for all phrases
    all_intuitions = [r["intuition"] for r in all_results]

    # Strategy 1: always use predicative axis
    preds_1 = [r["prob_pred"] for r in all_results]
    r1, _ = stats.spearmanr(preds_1, all_intuitions)
    mae1 = np.mean([abs(p - i) for p, i in zip(preds_1, all_intuitions)])

    # Strategy 2: always use modal axis
    preds_2 = [r["prob_modal"] for r in all_results]
    r2, _ = stats.spearmanr(preds_2, all_intuitions)
    mae2 = np.mean([abs(p - i) for p, i in zip(preds_2, all_intuitions)])

    # Strategy 3: use best axis (cheating — oracle selection)
    preds_3 = [r["best_prob"] for r in all_results]
    r3, _ = stats.spearmanr(preds_3, all_intuitions)
    mae3 = np.mean([abs(p - i) for p, i in zip(preds_3, all_intuitions)])

    # Strategy 4: average all 4 axes
    preds_4 = [(r["prob_pred"] + r["prob_freq"] + r["prob_np"] + r["prob_modal"]) / 4
               for r in all_results]
    r4, _ = stats.spearmanr(preds_4, all_intuitions)
    mae4 = np.mean([abs(p - i) for p, i in zip(preds_4, all_intuitions)])

    print(f"  Strategy comparison (n={len(all_results)} phrases):")
    print(f"    {'Strategy':<25s}  {'ρ':>6s}  {'MAE':>6s}")
    print(f"    {'Predicative only':<25s}  {r1:+.3f}  {mae1:5.1f}%")
    print(f"    {'Modal only':<25s}  {r2:+.3f}  {mae2:5.1f}%")
    print(f"    {'Average of 4 axes':<25s}  {r4:+.3f}  {mae4:5.1f}%")
    print(f"    {'Best axis (oracle)':<25s}  {r3:+.3f}  {mae3:5.1f}%")
    print()

    if r2 > r1 and mae2 < mae1:
        print("  MODAL AXIS IS BETTER than predicative for this phrase set.")
    elif mae2 < mae1:
        print("  MODAL AXIS has lower MAE (better calibration) despite similar ranking.")
    else:
        print("  No clear winner between predicative and modal axes.")

    print()


if __name__ == "__main__":
    main()
