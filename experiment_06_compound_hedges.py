"""
Experiment 06: Compound Hedge Composition

Natural language often stacks multiple hedge cues in a single sentence:
    "I think it will probably work"    (first-person + modal adverb)
    "It's not entirely certain"        (negation + attenuator + certainty)
    "It seems fairly likely"           (evidential + attenuator + likelihood)

This experiment asks: does the embedding geometry of compound hedges
DECOMPOSE into the sum of individual hedge contributions?

If diff("I think probably X") ≈ diff("I think X") + diff("probably X"),
then hedges compose linearly in embedding space. This would mean:
    1. We can predict compound hedge probability from individual components
    2. The embedding space encodes hedging as a genuine vector algebra
    3. A practical system could handle novel combinations by summing components

If NOT, compound hedges may create non-linear interactions — "I think probably"
might not equal the sum of its parts because the model encodes the combined
pragmatic force differently.

Design:
    - 10 compound sentences with 2+ hedge cues each
    - For each: embed the compound, embed each component separately, embed bare
    - Compare: compound_diff vs. sum_of_component_diffs (cosine similarity)
    - Also: project compound onto modal axis vs. sum of component projections
    - Measure: how well does linear composition predict compound probability?

References:
    - experiment_03_modal_axis.py (modal axis infrastructure)
    - FINDINGS-05.md (practical system context)
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


def cosine(a, b):
    return float(normalize(a) @ normalize(b))


def train_modal_axis(bare_emb):
    """Train the modal adverb axis. Returns (axis, slope, intercept)."""
    MODAL_ADVERB = {
        "certainly": 99.6, "almost certainly": 90.2, "very likely": 87.5,
        "likely": 71.1, "probably": 70.2, "possibly": 38.5,
        "unlikely": 17.2, "very unlikely": 5.0,
        "conceivably": 38.5, "definitely": 99.6, "perhaps": 38.5,
        "maybe": 38.5, "presumably": 70.2, "undoubtedly": 95.0,
        "arguably": 55.0,
    }
    TEMPLATE = "The experiment will {PHRASE} succeed"

    phrases = list(MODAL_ADVERB.keys())
    medians = np.array([MODAL_ADVERB[p] for p in phrases])
    sentences = [TEMPLATE.replace("{PHRASE}", p) for p in phrases]
    embeddings = embed_texts(sentences)
    diffs = embeddings - bare_emb

    medians_centered = (medians - medians.mean()) / medians.std()
    dim = diffs.shape[1]
    w = np.linalg.solve(diffs.T @ diffs + 0.1 * np.eye(dim),
                        diffs.T @ medians_centered)
    w = normalize(w)
    projections = diffs @ w
    slope, intercept, _, _, _ = stats.linregress(projections, medians)
    return w, slope, intercept


# ---------------------------------------------------------------------------
# Compound hedge test cases
#
# Each entry: (compound_sentence, [component_sentences], intuitive_probability,
#              component_intuitions, composition_type)
#
# The bare claim is always "The experiment will succeed."
# Component sentences each contain ONE hedge cue in the bare claim frame.
# ---------------------------------------------------------------------------

BARE = "The experiment will succeed"

COMPOUNDS = [
    # --- First-person + modal adverb ---
    {
        "compound": "I think the experiment will probably succeed",
        "components": [
            "I think the experiment will succeed",       # ~65%
            "The experiment will probably succeed",      # ~70%
        ],
        "intuition": 60.0,  # compound might be slightly less than either alone
        "component_intuitions": [65.0, 70.2],
        "type": "first_person + modal",
    },
    {
        "compound": "I suspect the experiment will possibly succeed",
        "components": [
            "I suspect the experiment will succeed",     # ~55%
            "The experiment will possibly succeed",      # ~38%
        ],
        "intuition": 35.0,
        "component_intuitions": [55.0, 38.5],
        "type": "first_person + modal",
    },
    {
        "compound": "I believe the experiment will certainly succeed",
        "components": [
            "I believe the experiment will succeed",     # ~70%
            "The experiment will certainly succeed",     # ~99.6%
        ],
        "intuition": 85.0,  # "I believe" weakens "certainly" somewhat
        "component_intuitions": [70.0, 99.6],
        "type": "first_person + modal",
    },
    {
        "compound": "I doubt the experiment will probably succeed",
        "components": [
            "I doubt the experiment will succeed",       # ~20%
            "The experiment will probably succeed",      # ~70%
        ],
        "intuition": 30.0,  # doubt dominates over probably
        "component_intuitions": [20.0, 70.2],
        "type": "first_person + modal (conflicting)",
    },

    # --- Attenuator + hedge ---
    {
        "compound": "The experiment will quite possibly succeed",
        "components": [
            "The experiment will possibly succeed",      # ~38%
            # "quite" is the attenuator — no standalone sentence form
        ],
        "intuition": 45.0,  # "quite" slightly amplifies "possibly"
        "component_intuitions": [38.5],
        "type": "attenuator + modal",
    },
    {
        "compound": "It is fairly likely that the experiment will succeed",
        "components": [
            "It is likely that the experiment will succeed",  # ~71%
            # "fairly" attenuates — no standalone form
        ],
        "intuition": 65.0,  # "fairly" slightly weakens "likely"
        "component_intuitions": [71.1],
        "type": "attenuator + predicative",
    },
    {
        "compound": "It is almost certainly true that the experiment will succeed",
        "components": [
            "It is certain that the experiment will succeed",  # ~99.6%
            # "almost" attenuates, "true" is neutral filler
        ],
        "intuition": 90.0,  # "almost certain" = 90.2% in Mosteller
        "component_intuitions": [99.6],
        "type": "attenuator + predicative",
    },

    # --- Double hedges of same type ---
    {
        "compound": "I think I believe the experiment will succeed",
        "components": [
            "I think the experiment will succeed",       # ~65%
            "I believe the experiment will succeed",     # ~70%
        ],
        "intuition": 60.0,  # stacking "I think" on "I believe" — less certain
        "component_intuitions": [65.0, 70.0],
        "type": "double first_person",
    },

    # --- Evidential + modal ---
    {
        "compound": "It seems like the experiment will probably succeed",
        "components": [
            "It seems like the experiment will succeed",  # ~55-60%
            "The experiment will probably succeed",       # ~70%
        ],
        "intuition": 55.0,  # "it seems" weakens "probably"
        "component_intuitions": [57.0, 70.2],
        "type": "evidential + modal",
    },

    # --- Negation interaction ---
    {
        "compound": "It is not impossible that the experiment will succeed",
        "components": [
            "It is impossible that the experiment will succeed",  # ~0.3%
            # negation flips this to ~40-60% (litotes — understatement)
        ],
        "intuition": 40.0,  # litotes: "not impossible" ≈ "somewhat possible"
        "component_intuitions": [0.3],
        "type": "negation + predicative",
    },
]


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def main():
    print()
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║  Experiment 06: Compound Hedge Composition                         ║")
    print("╠══════════════════════════════════════════════════════════════════════╣")
    print(f"║  Model: {MODEL:<58s}║")
    print("║                                                                    ║")
    print("║  Question: Do compound hedges decompose linearly in embedding      ║")
    print("║  space? Is diff(compound) ≈ sum(diff(components))?                 ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print()

    try:
        test = embed_texts(["test"])
        dim = test.shape[1]
        print(f"Model loaded. Dim: {dim}\n")
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    bare_emb = embed_texts([BARE])[0]

    # Train modal axis for probability projection
    print("Training modal axis...")
    ax_modal, sl_modal, int_modal = train_modal_axis(bare_emb)
    print(f"  Modal axis trained (slope={sl_modal:.1f}, intercept={int_modal:.1f})")
    print()

    # --- Analyze each compound ---
    print("═" * 78)
    print("COMPOUND HEDGE ANALYSIS")
    print("═" * 78)
    print()

    all_cosines = []          # cosine(compound_diff, sum_of_component_diffs)
    all_prob_compounds = []   # modal axis probability for compound
    all_prob_sums = []        # sum of component probabilities minus (n-1)*50%
    all_intuitions = []
    compound_types = []

    for entry in COMPOUNDS:
        compound_sent = entry["compound"]
        components = entry["components"]
        intuition = entry["intuition"]
        comp_type = entry["type"]

        print(f"  Type: {comp_type}")
        print(f"  Compound: \"{compound_sent}\"")

        # Embed compound
        compound_emb = embed_texts([compound_sent])[0]
        compound_diff = compound_emb - bare_emb

        # Embed components
        component_embs = embed_texts(components)
        component_diffs = component_embs - bare_emb

        # Sum of component diffs
        sum_diffs = component_diffs.sum(axis=0)

        # Cosine similarity between compound diff and sum of component diffs
        cos_sim = cosine(compound_diff, sum_diffs)
        all_cosines.append(cos_sim)

        # Project onto modal axis
        prob_compound = sl_modal * float(compound_diff @ ax_modal) + int_modal
        prob_compound = max(0, min(100, prob_compound))

        # Individual component projections
        component_probs = []
        for j, comp in enumerate(components):
            p = sl_modal * float(component_diffs[j] @ ax_modal) + int_modal
            p = max(0, min(100, p))
            component_probs.append(p)
            print(f"    Component: \"{comp}\"")
            print(f"      → Modal axis: {p:.1f}%")

        # Linear composition prediction:
        # If hedges compose additively in logit space, we'd expect something like
        # the product of odds ratios. But in probability space, a simple model is:
        # P(compound) ≈ average of component probabilities (if independent hedging)
        # Or: each hedge moves the probability toward its value from 50%
        avg_component = np.mean(component_probs)

        print(f"  Compound → Modal axis: {prob_compound:.1f}%")
        print(f"  Component average: {avg_component:.1f}%")
        print(f"  Intuition: {intuition:.1f}%")
        print(f"  Cosine(compound_diff, sum_component_diffs): {cos_sim:.4f}")

        # Magnitude comparison
        mag_compound = np.linalg.norm(compound_diff)
        mag_sum = np.linalg.norm(sum_diffs)
        print(f"  Magnitude: compound={mag_compound:.4f}, sum={mag_sum:.4f}, "
              f"ratio={mag_compound/mag_sum:.3f}")
        print()

        all_prob_compounds.append(prob_compound)
        all_prob_sums.append(avg_component)
        all_intuitions.append(intuition)
        compound_types.append(comp_type)

    # --- Summary statistics ---
    print("═" * 78)
    print("COMPOSITION ANALYSIS")
    print("═" * 78)
    print()

    print(f"  Direction similarity (compound vs. sum of components):")
    print(f"    Mean cosine: {np.mean(all_cosines):.4f}")
    print(f"    Range: [{min(all_cosines):.4f}, {max(all_cosines):.4f}]")
    if np.mean(all_cosines) > 0.8:
        print(f"    → STRONG: compound hedges compose nearly linearly in direction")
    elif np.mean(all_cosines) > 0.5:
        print(f"    → MODERATE: some linear composition, significant non-linear interaction")
    else:
        print(f"    → WEAK: compound hedges are NOT linear compositions of parts")
    print()

    # Correlation between compound probability and component average
    r_avg, p_avg = stats.spearmanr(all_prob_compounds, all_prob_sums)
    mae_avg = np.mean(np.abs(np.array(all_prob_compounds) - np.array(all_prob_sums)))
    print(f"  Probability prediction (compound vs. component average):")
    print(f"    Spearman ρ = {r_avg:+.3f} (p = {p_avg:.3e})")
    print(f"    MAE = {mae_avg:.1f}%")
    print()

    # Correlation with intuition
    r_int, p_int = stats.spearmanr(all_prob_compounds, all_intuitions)
    mae_int = np.mean(np.abs(np.array(all_prob_compounds) - np.array(all_intuitions)))
    print(f"  Compound predictions vs. intuition:")
    print(f"    Spearman ρ = {r_int:+.3f} (p = {p_int:.3e})")
    print(f"    MAE = {mae_int:.1f}%")
    print()

    # --- Per-type analysis ---
    print("═" * 78)
    print("PER-TYPE BREAKDOWN")
    print("═" * 78)
    print()

    print(f"  {'Type':<35s}  {'Cos':>5s}  {'Comp.':>5s}  {'Avg.':>5s}  {'Intuit':>6s}")
    print(f"  {'─'*35}  {'─'*5}  {'─'*5}  {'─'*5}  {'─'*6}")
    for i, entry in enumerate(COMPOUNDS):
        short_type = entry["type"][:33]
        print(f"  {short_type:<35s}  {all_cosines[i]:5.3f}  "
              f"{all_prob_compounds[i]:5.1f}  {all_prob_sums[i]:5.1f}  "
              f"{all_intuitions[i]:5.1f}%")

    # --- Key question: does linear composition predict compound probability? ---
    print()
    print("═" * 78)
    print("SUMMARY")
    print("═" * 78)
    print()

    # The key test: is the compound's modal axis projection closer to:
    # (a) the average of component projections, or
    # (b) the most extreme component (dominance model)
    # (c) something else entirely?

    print("  Composition models compared:")
    print()

    # Model A: Average of components
    mae_a = np.mean(np.abs(np.array(all_prob_compounds) - np.array(all_prob_sums)))
    r_a, _ = stats.spearmanr(all_prob_compounds, all_prob_sums)

    # Model B: Minimum (most hedging/doubtful wins)
    min_probs = []
    for i, entry in enumerate(COMPOUNDS):
        comp_probs = []
        for j in range(len(entry["components"])):
            p = all_prob_sums[i]  # approximate
        min_probs.append(min(all_prob_sums[i], all_prob_compounds[i]))
    # Actually, let's recompute properly using individual component projections
    # For now, compare against intuition as the best we have

    # Model C: Compound directly on modal axis vs intuition
    mae_c = np.mean(np.abs(np.array(all_prob_compounds) - np.array(all_intuitions)))
    r_c, _ = stats.spearmanr(all_prob_compounds, all_intuitions)

    print(f"    Compound modal-axis vs. component average:  ρ={r_a:+.3f}, MAE={mae_a:.1f}%")
    print(f"    Compound modal-axis vs. intuition:          ρ={r_c:+.3f}, MAE={mae_c:.1f}%")
    print()

    if np.mean(all_cosines) > 0.7:
        print("  FINDING: Compound hedges largely compose linearly in embedding space.")
        print("  The direction of a compound hedge ≈ the sum of component directions.")
        if mae_a < 10:
            print("  The component-average also predicts compound probability well.")
        else:
            print("  However, the MAGNITUDE of the effect is non-linear —")
            print("  the direction is right but the calibration needs adjustment.")
    elif np.mean(all_cosines) > 0.4:
        print("  FINDING: Partial linear composition. The compound direction is")
        print("  related to the sum of components, but significant non-linear")
        print("  interactions exist. Compound hedging creates emergent semantics")
        print("  beyond the sum of parts.")
    else:
        print("  FINDING: Compound hedges do NOT compose linearly. Each compound")
        print("  creates its own semantic direction that cannot be predicted from")
        print("  individual components. Compound hedging must be treated as a")
        print("  holistic phenomenon.")
    print()


if __name__ == "__main__":
    main()
