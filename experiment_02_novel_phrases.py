"""
Experiment 02: Novel Phrase Projection

Tests whether the probability axes discovered in Experiment 1b generalize
to phrases NOT in the Mosteller calibration data. These include:

1. Informal epistemic markers: "I think", "It seems to me", "I suspect"
2. Conversational hedges: "I wouldn't be surprised if", "I wonder though"
3. Anti-hedges (certainty markers): "Obviously", "It goes without saying"
4. Novel phrasings: "I'd wager that", "Signs point to", "It stands to reason"
5. Positive controls: Mosteller phrases (should project correctly)
6. Negative controls: Non-epistemic modifications ("Fortunately", "Interestingly")

Key question: Do the trained type-specific probability axes give reasonable
probability estimates for phrases they've never seen?

The experiment projects novel phrases onto ALL THREE trained axes (predicative,
adverbial, noun-phrase) to test whether the probability reading is robust across
axes or type-dependent. Consistent readings across axes would mean the geometry
captures probability per se, not just within-type variation.

Design note: Novel phrases don't fit cleanly into the template slots used for
training. We embed them in natural sentence form and compute the difference
from the bare claim. This tests whether the axes generalize to natural syntax.

References:
  - experiment_01b_within_type.py (trained axes and calibration data)
  - FINDINGS-01.md (prior results and context)
  - docs/mosteller_youtz_1990_full.csv (calibration data for mapping)
"""

import sys
import numpy as np
import requests
from pathlib import Path
from scipy import stats

MODEL = sys.argv[1] if len(sys.argv) > 1 else "nomic-embed-text:v1.5"
OLLAMA_URL = "http://localhost:11434/api/embed"
MOSTELLER_CSV = Path(__file__).parent / "docs" / "mosteller_youtz_1990_full.csv"


def embed_texts(texts, model=MODEL):
    resp = requests.post(OLLAMA_URL, json={"model": model, "input": texts})
    resp.raise_for_status()
    return np.array(resp.json()["embeddings"], dtype=np.float64)


def normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


# ---------------------------------------------------------------------------
# Training data: recompute the supervised probability axes from Experiment 1b
#
# We retrain here (rather than loading saved axes) so this script is
# self-contained and can be run with any model.
# ---------------------------------------------------------------------------

PREDICATIVE_EXPRESSIONS = {
    "Certain": 99.6, "Almost certain": 90.2, "Very likely": 87.5,
    "Likely": 71.1, "Probable": 70.2, "Very probable": 89.7,
    "Possible": 38.5, "Unlikely": 17.2, "Very unlikely": 5.0,
    "Improbable": 12.5, "Very improbable": 4.8, "Impossible": 0.3,
    "Not unreasonable": 37.6,
}
PREDICATIVE_TEMPLATE = "It is {PHRASE} that the experiment will succeed"

ADVERBIAL_EXPRESSIONS = {
    "Always": 99.7, "Almost always": 91.7, "Very often": 82.8,
    "Often": 72.5, "Usually": 75.1, "Sometimes": 25.0,
    "Occasionally": 20.0, "Seldom": 10.2, "Very seldom": 4.9,
    "Rarely": 7.2, "Very rarely": 3.0, "Almost never": 2.9,
    "Never": 0.3, "Not often": 19.7, "Not very often": 10.1,
    "As often as not": 50.0, "More often than not": 59.8,
    "Once in a while": 15.3, "Now and then": 15.1,
}
ADVERBIAL_TEMPLATE = "The experiment will {PHRASE} succeed"

NOUN_PHRASE_EXPRESSIONS = {
    "Very high probability": 92.5, "High probability": 82.3,
    "Moderate probability": 52.4, "Low probability": 15.0,
    "Very low probability": 4.9, "High chance": 80.4,
    "Poor chance": 10.3, "Low chance": 9.8, "Even chance": 50.0,
    "Better than even chance": 57.6, "Less than an even chance": 40.2,
}
NOUN_PHRASE_TEMPLATE = "There is a {PHRASE} that the experiment will succeed"

BARE_CLAIM = "The experiment will succeed"


def train_axis(expressions, template, bare_embedding):
    """
    Train a supervised probability axis for a syntactic type.
    Returns: (axis_vector, slope, intercept) where
      probability ≈ slope * (diff @ axis) + intercept
    """
    phrases = list(expressions.keys())
    medians = np.array([expressions[p] for p in phrases])

    sentences = [template.replace("{PHRASE}", p.lower()) for p in phrases]
    embeddings = embed_texts(sentences)
    diffs = embeddings - bare_embedding

    # Ridge regression to find probability-maximizing direction
    medians_centered = (medians - medians.mean()) / medians.std()
    dim = diffs.shape[1]
    lam = 0.1
    w = np.linalg.solve(diffs.T @ diffs + lam * np.eye(dim), diffs.T @ medians_centered)
    w = normalize(w)

    # Calibration: linear mapping from projection to probability
    projections = diffs @ w
    slope, intercept, _, _, _ = stats.linregress(projections, medians)

    return w, slope, intercept, projections, medians


def project_to_probability(diff_vector, axis, slope, intercept):
    """Project a difference vector onto a trained axis and map to probability."""
    proj = float(diff_vector @ axis)
    prob = slope * proj + intercept
    return proj, prob


# ---------------------------------------------------------------------------
# Novel phrases to test
#
# Each entry: (natural_sentence, category, intuitive_probability_guess)
#
# The intuitive_probability_guess is my best estimate of what probability
# the phrase implies. This is NOT ground truth — it's a sanity check.
# If the geometric prediction is wildly different from intuition, investigate.
# If it's close, the geometry might be right (or my intuition might be wrong).
# ---------------------------------------------------------------------------

NOVEL_PHRASES = [
    # --- Informal epistemic markers (conversational hedging) ---
    ("I think the experiment will succeed",
     "informal_hedge", 65),
    ("It seems to me that the experiment will succeed",
     "informal_hedge", 60),
    ("I believe the experiment will succeed",
     "informal_hedge", 70),
    ("I suspect the experiment will succeed",
     "informal_hedge", 55),
    ("I imagine the experiment will succeed",
     "informal_hedge", 50),

    # --- Doubt / skepticism markers ---
    ("I wonder if the experiment will succeed",
     "doubt", 35),
    ("I doubt the experiment will succeed",
     "doubt", 20),
    ("I'm not sure the experiment will succeed",
     "doubt", 40),
    ("I wouldn't be surprised if the experiment succeeded",
     "doubt", 55),
    ("I have my doubts about whether the experiment will succeed",
     "doubt", 25),

    # --- Anti-hedges (certainty markers) ---
    ("Obviously, the experiment will succeed",
     "anti_hedge", 95),
    ("It goes without saying that the experiment will succeed",
     "anti_hedge", 95),
    ("Clearly, the experiment will succeed",
     "anti_hedge", 92),
    ("Without question, the experiment will succeed",
     "anti_hedge", 98),
    ("There is no doubt that the experiment will succeed",
     "anti_hedge", 97),
    ("Undoubtedly, the experiment will succeed",
     "anti_hedge", 93),

    # --- Novel phrasings (non-standard hedges) ---
    ("I'd wager that the experiment will succeed",
     "novel", 70),
    ("Signs point to the experiment succeeding",
     "novel", 65),
    ("It stands to reason that the experiment will succeed",
     "novel", 75),
    ("The odds are good that the experiment will succeed",
     "novel", 70),
    ("All indications suggest the experiment will succeed",
     "novel", 80),
    ("It remains to be seen whether the experiment will succeed",
     "novel", 45),
    ("There's a slim chance the experiment will succeed",
     "novel", 15),
    ("It's a safe bet that the experiment will succeed",
     "novel", 85),
    ("The experiment might well succeed",
     "novel", 60),
    ("The experiment could conceivably succeed",
     "novel", 35),

    # --- Positive controls (Mosteller phrases in natural form) ---
    # These should project close to their known Mosteller values.
    ("The experiment will probably succeed",
     "control_mosteller", 70.2),  # "Probable" = 70.2%
    ("The experiment will certainly succeed",
     "control_mosteller", 99.6),  # "Certain" = 99.6% (close)
    ("It is unlikely that the experiment will succeed",
     "control_mosteller", 17.2),  # "Unlikely" = 17.2%
    ("It is very likely that the experiment will succeed",
     "control_mosteller", 87.5),  # "Very likely" = 87.5%
    ("The experiment will rarely succeed",
     "control_mosteller", 7.2),   # "Rarely" = 7.2%

    # --- Negative controls (non-epistemic modifiers) ---
    # These should NOT project cleanly onto the probability axis.
    # Predicted probabilities should be near 50% (uninformative) or erratic.
    ("Fortunately, the experiment will succeed",
     "control_nonepistemic", None),
    ("Interestingly, the experiment will succeed",
     "control_nonepistemic", None),
    ("Surprisingly, the experiment will succeed",
     "control_nonepistemic", None),
    ("Importantly, the experiment will succeed",
     "control_nonepistemic", None),
]

# Additional bare claims for cross-content check of novel phrases
EXTRA_BARE_CLAIMS = [
    "The treatment will be effective",
    "The prediction will be correct",
]

# Novel phrases adapted for extra contexts (spot check, not exhaustive)
CROSS_CONTENT_PHRASES = [
    ("I think the treatment will be effective", "informal_hedge", 65),
    ("Obviously, the treatment will be effective", "anti_hedge", 95),
    ("I doubt the prediction will be correct", "doubt", 20),
    ("It stands to reason that the prediction will be correct", "novel", 75),
]


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def main():
    print()
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║  Experiment 02: Novel Phrase Projection                            ║")
    print("╠══════════════════════════════════════════════════════════════════════╣")
    print(f"║  Model: {MODEL:<58s}║")
    print("║                                                                    ║")
    print("║  Question: Do trained probability axes give reasonable estimates   ║")
    print("║  for phrases not in the Mosteller calibration data?               ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print()

    # Verify model
    try:
        test = embed_texts(["test"])
        dim = test.shape[1]
        print(f"Model loaded. Dim: {dim}")
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    # --- Train axes ---
    print("\nTraining probability axes on Mosteller data...")
    bare_emb = embed_texts([BARE_CLAIM])[0]

    ax_pred, sl_pred, int_pred, proj_pred, med_pred = train_axis(
        PREDICATIVE_EXPRESSIONS, PREDICATIVE_TEMPLATE, bare_emb)
    ax_adv, sl_adv, int_adv, proj_adv, med_adv = train_axis(
        ADVERBIAL_EXPRESSIONS, ADVERBIAL_TEMPLATE, bare_emb)
    ax_np, sl_np, int_np, proj_np, med_np = train_axis(
        NOUN_PHRASE_EXPRESSIONS, NOUN_PHRASE_TEMPLATE, bare_emb)

    # Report training fit
    print(f"  Predicative axis: projection range [{proj_pred.min():.3f}, {proj_pred.max():.3f}]")
    print(f"    Calibration: prob = {sl_pred:.1f} * proj + {int_pred:.1f}")
    print(f"  Adverbial axis: projection range [{proj_adv.min():.3f}, {proj_adv.max():.3f}]")
    print(f"    Calibration: prob = {sl_adv:.1f} * proj + {int_adv:.1f}")
    print(f"  Noun phrase axis: projection range [{proj_np.min():.3f}, {proj_np.max():.3f}]")
    print(f"    Calibration: prob = {sl_np:.1f} * proj + {int_np:.1f}")

    # --- Embed novel phrases ---
    print(f"\nEmbedding {len(NOVEL_PHRASES)} novel phrases...")
    novel_sentences = [p[0] for p in NOVEL_PHRASES]
    novel_embeddings = embed_texts(novel_sentences)
    novel_diffs = novel_embeddings - bare_emb

    # --- Project onto all three axes ---
    print()
    print("=" * 78)
    print(f"{'Phrase':<55s} {'Pred':>5s} {'Adv':>5s} {'NP':>5s} {'Avg':>5s} {'Intuit':>6s}")
    print("=" * 78)

    categories = {}
    all_results = []

    for i, (sentence, category, intuition) in enumerate(NOVEL_PHRASES):
        diff = novel_diffs[i]

        _, prob_pred = project_to_probability(diff, ax_pred, sl_pred, int_pred)
        _, prob_adv = project_to_probability(diff, ax_adv, sl_adv, int_adv)
        _, prob_np = project_to_probability(diff, ax_np, sl_np, int_np)

        # Clip to [0, 100] for display
        prob_pred = max(0, min(100, prob_pred))
        prob_adv = max(0, min(100, prob_adv))
        prob_np = max(0, min(100, prob_np))
        avg = (prob_pred + prob_adv + prob_np) / 3

        # Track by category
        if category not in categories:
            categories[category] = []
        categories[category].append({
            "sentence": sentence,
            "prob_pred": prob_pred,
            "prob_adv": prob_adv,
            "prob_np": prob_np,
            "avg": avg,
            "intuition": intuition,
        })

        all_results.append({
            "sentence": sentence,
            "category": category,
            "prob_pred": prob_pred,
            "prob_adv": prob_adv,
            "prob_np": prob_np,
            "avg": avg,
            "intuition": intuition,
        })

        # Print with category separators
        if i == 0 or NOVEL_PHRASES[i][1] != NOVEL_PHRASES[i-1][1]:
            print(f"--- {category} ---")

        intuit_str = f"{intuition:5.1f}%" if intuition is not None else "  n/a"
        # Shortened sentence for display
        short = sentence[:52] + "..." if len(sentence) > 55 else sentence
        print(f"  {short:<53s} {prob_pred:5.1f} {prob_adv:5.1f} "
              f"{prob_np:5.1f} {avg:5.1f} {intuit_str}")

    # --- Analysis by category ---
    print()
    print("=" * 78)
    print("ANALYSIS BY CATEGORY")
    print("=" * 78)

    for cat_name in ["informal_hedge", "doubt", "anti_hedge", "novel",
                     "control_mosteller", "control_nonepistemic"]:
        items = categories.get(cat_name, [])
        if not items:
            continue

        print(f"\n--- {cat_name} (n={len(items)}) ---")

        avgs = [it["avg"] for it in items]
        print(f"  Average geometric probability: {np.mean(avgs):.1f}% "
              f"(range: {min(avgs):.1f}% – {max(avgs):.1f}%)")

        # Cross-axis consistency: std across the three axis projections
        cross_axis_stds = [np.std([it["prob_pred"], it["prob_adv"], it["prob_np"]])
                          for it in items]
        print(f"  Cross-axis consistency: mean std = {np.mean(cross_axis_stds):.1f}%")
        if np.mean(cross_axis_stds) < 10:
            print(f"    → Good: axes agree (std < 10%)")
        elif np.mean(cross_axis_stds) < 20:
            print(f"    → Moderate: some axis disagreement")
        else:
            print(f"    → Poor: axes disagree substantially")

        # Correlation with intuition (for categories that have it)
        intuitions = [it["intuition"] for it in items if it["intuition"] is not None]
        geom_probs = [it["avg"] for it in items if it["intuition"] is not None]
        if len(intuitions) >= 3:
            r, p = stats.spearmanr(geom_probs, intuitions)
            print(f"  Correlation with intuition: ρ = {r:.3f} (p = {p:.3e})")
            if abs(r) > 0.7:
                print(f"    → Good: geometry tracks intuition")
            elif abs(r) > 0.4:
                print(f"    → Moderate: partial agreement")
            else:
                print(f"    → Weak/no agreement")

    # --- Positive control accuracy ---
    print()
    print("=" * 78)
    print("POSITIVE CONTROL: Mosteller phrases in natural syntax")
    print("=" * 78)
    print()
    controls = categories.get("control_mosteller", [])
    if controls:
        errors = [abs(it["avg"] - it["intuition"]) for it in controls]
        print(f"  Mean absolute error (geometric avg vs. Mosteller): {np.mean(errors):.1f}%")
        print(f"  Max error: {max(errors):.1f}%")
        for it in controls:
            short = it["sentence"][:50]
            print(f"    {short:<50s}  Geom={it['avg']:5.1f}%  "
                  f"Mosteller={it['intuition']:5.1f}%  "
                  f"Error={abs(it['avg']-it['intuition']):4.1f}%")

    # --- Negative control behavior ---
    print()
    print("=" * 78)
    print("NEGATIVE CONTROL: Non-epistemic modifiers")
    print("=" * 78)
    print()
    neg_controls = categories.get("control_nonepistemic", [])
    if neg_controls:
        avgs = [it["avg"] for it in neg_controls]
        stds = [np.std([it["prob_pred"], it["prob_adv"], it["prob_np"]])
                for it in neg_controls]
        print(f"  Average probability: {np.mean(avgs):.1f}% (should be near 50% or erratic)")
        print(f"  Cross-axis std: {np.mean(stds):.1f}% (should be high if axes disagree)")
        for it in neg_controls:
            short = it["sentence"][:50]
            print(f"    {short:<50s}  Pred={it['prob_pred']:5.1f}%  "
                  f"Adv={it['prob_adv']:5.1f}%  NP={it['prob_np']:5.1f}%")

    # --- Cross-content check ---
    print()
    print("=" * 78)
    print("CROSS-CONTENT: Same phrases, different propositions")
    print("=" * 78)
    print()

    for extra_bare in EXTRA_BARE_CLAIMS:
        extra_bare_emb = embed_texts([extra_bare])[0]
        print(f"  Bare claim: \"{extra_bare}\"")

        cross_phrases = [p for p in CROSS_CONTENT_PHRASES
                        if extra_bare.split()[1] in p[0]]  # rough match
        if not cross_phrases:
            # Use all cross-content phrases for this bare claim
            cross_phrases = [p for p in CROSS_CONTENT_PHRASES
                            if any(w in p[0] for w in extra_bare.split()[1:3])]

        if cross_phrases:
            cross_sentences = [p[0] for p in cross_phrases]
            cross_embs = embed_texts(cross_sentences)
            for j, (sent, cat, intuit) in enumerate(cross_phrases):
                diff = cross_embs[j] - extra_bare_emb
                _, pp = project_to_probability(diff, ax_pred, sl_pred, int_pred)
                _, pa = project_to_probability(diff, ax_adv, sl_adv, int_adv)
                _, pn = project_to_probability(diff, ax_np, sl_np, int_np)
                pp, pa, pn = [max(0, min(100, x)) for x in [pp, pa, pn]]
                avg = (pp + pa + pn) / 3
                short = sent[:50]
                print(f"    {short:<50s}  Avg={avg:5.1f}% (intuit={intuit}%)")
        print()

    # --- Overall summary ---
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print()

    # Key questions answered:
    epistemic_cats = ["informal_hedge", "doubt", "anti_hedge", "novel"]
    all_epistemic = [it for cat in epistemic_cats
                     for it in categories.get(cat, [])]
    all_intuitions = [it["intuition"] for it in all_epistemic
                     if it["intuition"] is not None]
    all_geometric = [it["avg"] for it in all_epistemic
                    if it["intuition"] is not None]

    if len(all_intuitions) >= 5:
        r_overall, p_overall = stats.spearmanr(all_geometric, all_intuitions)
        print(f"Overall correlation (geometry vs. intuition, n={len(all_intuitions)}):")
        print(f"  Spearman ρ = {r_overall:.3f} (p = {p_overall:.3e})")
        print()

    cross_axis_all = [np.std([it["prob_pred"], it["prob_adv"], it["prob_np"]])
                     for it in all_epistemic]
    print(f"Cross-axis consistency (all epistemic phrases):")
    print(f"  Mean std across 3 axes: {np.mean(cross_axis_all):.1f}%")
    print()

    control_errors = [abs(it["avg"] - it["intuition"])
                     for it in categories.get("control_mosteller", [])]
    if control_errors:
        print(f"Positive control (Mosteller phrases in natural syntax):")
        print(f"  Mean absolute error: {np.mean(control_errors):.1f}%")
        print()

    print("--- Assessment ---")
    if len(all_intuitions) >= 5 and abs(r_overall) > 0.7:
        print("POSITIVE: Geometry predicts probability for novel phrases.")
        print("  The trained axes generalize beyond Mosteller vocabulary.")
    elif len(all_intuitions) >= 5 and abs(r_overall) > 0.5:
        print("MODERATE: Some predictive power for novel phrases.")
        print("  The axes capture something, but not perfectly.")
    else:
        print("WEAK/NEGATIVE: Limited generalization to novel phrases.")
        print("  The axes may be too specific to Mosteller vocabulary.")
    print()


if __name__ == "__main__":
    main()
