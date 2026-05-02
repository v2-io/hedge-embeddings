"""
Experiment 08: "Possible" Bimodality Probe + Wintle Cross-Validation

Two quick experiments for the paper:

PART A — Bimodality Probe:
  "Possible" is bimodal in humans (Mosteller IQR=42.7). Our models encode
  a single stable direction (Experiment 4). But do MODIFIERS that disambiguate
  "possible" toward one reading or the other move the projection accordingly?

  If "barely possible" projects low (~10%) and "entirely possible" projects
  high (~55%), the geometry encodes the full semantic continuum — it just
  defaults to one point for the bare word.

PART B — Wintle et al. (2019) Cross-Validation:
  A third independent validation source (n≈924) beyond Mosteller and Vogel.
  Train on Mosteller, test on Wintle. Three-dataset convergence.

Usage:
  python3 experiment_08_bimodality_wintle.py [model_name]
  Default model: nomic-embed-text:v1.5
"""

import sys
import numpy as np
import requests
from scipy import stats

from data.mosteller import (
    PREDICATIVE_13, PREDICATIVE_TEMPLATE,
    ADVERBIAL_19, ADVERBIAL_TEMPLATE,
    NOUN_PHRASE_11, NOUN_PHRASE_TEMPLATE,
    MODAL_M10, MODAL_M15, MODAL_TEMPLATE,
    BARE_CLAIM,
)

# CLI: positional model name + optional --modal=m10|m15
_args = [a for a in sys.argv[1:] if not a.startswith("--")]
_flags = {a.split("=", 1)[0]: a.split("=", 1)[1] if "=" in a else ""
          for a in sys.argv[1:] if a.startswith("--")}

MODEL = _args[0] if _args else "nomic-embed-text:v1.5"
MODAL_SET_NAME = _flags.get("--modal", "m10").lower()
if MODAL_SET_NAME not in {"m10", "m15"}:
    print(f"ERROR: --modal must be m10 or m15, got {MODAL_SET_NAME!r}")
    sys.exit(2)

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


# Local aliases pointing at canonical sets in data/mosteller.py.
# MODAL toggles between M10 (canonical) and M15 (legacy) via --modal flag.
PREDICATIVE = PREDICATIVE_13
ADVERBIAL = ADVERBIAL_19
NOUN_PHRASE = NOUN_PHRASE_11
MODAL = MODAL_M10 if MODAL_SET_NAME == "m10" else MODAL_M15


# ═══════════════════════════════════════════════════════════════════════
# PART A: Bimodality Probe
# ═══════════════════════════════════════════════════════════════════════

# Disambiguating variants of "possible" in predicative frame
POSSIBLE_VARIANTS_PRED = [
    # LOW-leaning (logical possibility, expected ~5-20%)
    ("barely possible", 10),
    ("just barely possible", 5),
    ("theoretically possible", 10),
    ("remotely possible", 8),
    ("just possible", 15),
    ("technically possible", 12),
    # NEUTRAL (bare form, Mosteller median 38.5%)
    ("possible", 38.5),
    # HIGH-leaning (reasonable chance, expected ~45-65%)
    ("entirely possible", 55),
    ("quite possible", 50),
    ("perfectly possible", 60),
    ("very possible", 55),
    ("eminently possible", 60),
]

# Modal frame variants of "possibly"
POSSIBLE_VARIANTS_MODAL = [
    ("barely possibly", 10),
    ("just possibly", 15),
    # bare
    ("possibly", 38.5),
    # high
    ("quite possibly", 50),
    ("very possibly", 55),
]

# Controls: same modifiers on unambiguous words
MODIFIER_CONTROLS = [
    # "Likely" (Mosteller 71.1%) — unambiguous, moderate-high
    ("barely likely", 55),
    ("likely", 71.1),
    ("entirely likely", 80),
    ("very likely", 87.5),
    # "Certain" (Mosteller 99.6%) — unambiguous, extreme
    ("barely certain", 85),
    ("certain", 99.6),
    ("entirely certain", 99),
    # "Impossible" with negation
    ("not entirely impossible", 25),
    ("not impossible", 40),
    ("not at all impossible", 45),
]


# ═══════════════════════════════════════════════════════════════════════
# PART B: Wintle et al. 2019 Cross-Validation
# ═══════════════════════════════════════════════════════════════════════

# Wintle et al. 2019 (n≈924) — median probability estimates
# Source: "Verbal probabilities: Very likely to be somewhat more confusing
# than numbers" (PLoS ONE)
# These are the standard ODNI/ICD-203 terms plus additional expressions.
# Values are approximate medians from their Figure 2 / Table data.
WINTLE_PREDICATIVE = {
    # Expression: Wintle median probability (%)
    "Almost certain": 93,
    "Very likely": 85,
    "Likely": 73,
    "Probable": 72,
    "Very unlikely": 8,
    "Unlikely": 15,
    "Impossible": 2,
    "Possible": 45,
    "Improbable": 12,
}

WINTLE_ADVERBIAL = {
    "Almost always": 90,
    "Usually": 78,
    "Rarely": 8,
}


def run_part_a(bare_emb, ax_pred, sl_pred, int_pred, ax_modal, sl_modal, int_modal):
    """Part A: Bimodality probe for 'possible'."""
    print()
    print("═" * 78)
    print("PART A: BIMODALITY PROBE — Does geometry encode the ambiguity")
    print("        spectrum of 'possible' when modifiers disambiguate?")
    print("═" * 78)

    # --- Predicative frame ---
    print(f"\n  PREDICATIVE FRAME: \"It is {{modifier}} possible that X\"")
    print(f"  {'Phrase':<30s}  {'Pred':>6s}  {'Modal':>6s}  {'Expect':>6s}")
    print(f"  {'─'*30}  {'─'*6}  {'─'*6}  {'─'*6}")

    pred_results = []
    for phrase, expected in POSSIBLE_VARIANTS_PRED:
        sentence = PREDICATIVE_TEMPLATE.replace("{PHRASE}", phrase)
        emb = embed_texts([sentence])[0]
        diff = emb - bare_emb
        prob_pred = max(0, min(100, sl_pred * float(diff @ ax_pred) + int_pred))
        prob_modal = max(0, min(100, sl_modal * float(diff @ ax_modal) + int_modal))
        pred_results.append((phrase, prob_pred, prob_modal, expected))
        print(f"  {phrase:<30s}  {prob_pred:5.1f}%  {prob_modal:5.1f}%  {expected:5.1f}%")

    # Compute spread and correlation
    low_phrases = [r for r in pred_results if r[3] <= 15]
    high_phrases = [r for r in pred_results if r[3] >= 45]
    bare = [r for r in pred_results if r[0] == "possible"][0]

    if low_phrases and high_phrases:
        low_mean = np.mean([r[1] for r in low_phrases])
        high_mean = np.mean([r[1] for r in high_phrases])
        spread = high_mean - low_mean
        print(f"\n  Low-leaning mean (pred axis): {low_mean:.1f}%")
        print(f"  High-leaning mean (pred axis): {high_mean:.1f}%")
        print(f"  Bare 'possible' (pred axis): {bare[1]:.1f}%")
        print(f"  Spread (high - low): {spread:.1f} percentage points")
        if spread > 20:
            print(f"  → POSITIVE: Modifiers traverse the ambiguity range.")
        elif spread > 10:
            print(f"  → MODERATE: Some movement, but less than expected.")
        else:
            print(f"  → NEGATIVE: Modifiers don't move the projection much.")

    # Correlation with expected values
    actuals = [r[1] for r in pred_results]
    expecteds = [r[3] for r in pred_results]
    r_pred, p_pred = stats.spearmanr(actuals, expecteds)
    print(f"\n  Spearman ρ (pred projection vs. expected): {r_pred:+.3f} (p={p_pred:.3e})")

    # --- Modal frame ---
    print(f"\n  MODAL FRAME: \"The experiment will {{modifier}} possibly succeed\"")
    print(f"  {'Phrase':<30s}  {'Modal':>6s}  {'Expect':>6s}")
    print(f"  {'─'*30}  {'─'*6}  {'─'*6}")

    modal_results = []
    for phrase, expected in POSSIBLE_VARIANTS_MODAL:
        sentence = MODAL_TEMPLATE.replace("{PHRASE}", phrase)
        emb = embed_texts([sentence])[0]
        diff = emb - bare_emb
        prob_modal = max(0, min(100, sl_modal * float(diff @ ax_modal) + int_modal))
        modal_results.append((phrase, prob_modal, expected))
        print(f"  {phrase:<30s}  {prob_modal:5.1f}%  {expected:5.1f}%")

    # --- Controls ---
    print(f"\n  CONTROLS: Same modifiers on unambiguous words")
    print(f"  {'Phrase':<30s}  {'Pred':>6s}  {'Expect':>6s}")
    print(f"  {'─'*30}  {'─'*6}  {'─'*6}")

    for phrase, expected in MODIFIER_CONTROLS:
        sentence = PREDICATIVE_TEMPLATE.replace("{PHRASE}", phrase)
        emb = embed_texts([sentence])[0]
        diff = emb - bare_emb
        prob_pred = max(0, min(100, sl_pred * float(diff @ ax_pred) + int_pred))
        print(f"  {phrase:<30s}  {prob_pred:5.1f}%  {expected:5.1f}%")

    return r_pred, spread if (low_phrases and high_phrases) else 0


def run_part_b(bare_emb, axes_data):
    """Part B: Wintle et al. 2019 cross-validation."""
    print()
    print("═" * 78)
    print("PART B: WINTLE ET AL. (2019) CROSS-VALIDATION")
    print("  Train on Mosteller (1990, n=238), test on Wintle (2019, n≈924)")
    print("  Third independent validation source.")
    print("═" * 78)

    results = {}

    for group_name, wintle_data, train_data, template, ax, sl, intc in axes_data:
        if not wintle_data:
            continue

        # Test Wintle expressions on Mosteller-trained axis
        wintle_phrases = list(wintle_data.keys())
        wintle_medians = np.array([wintle_data[p] for p in wintle_phrases])
        wintle_sentences = [template.replace("{PHRASE}", p.lower()) for p in wintle_phrases]
        wintle_embs = embed_texts(wintle_sentences)
        wintle_diffs = wintle_embs - bare_emb
        wintle_preds = np.clip(sl * (wintle_diffs @ ax) + intc, 0, 100)

        r_wintle, p_wintle = stats.spearmanr(wintle_preds, wintle_medians)
        mae_wintle = np.mean(np.abs(wintle_preds - wintle_medians))

        print(f"\n  {group_name} (n_train={len(train_data)}, n_test={len(wintle_data)}):")
        print(f"    Wintle Spearman ρ = {r_wintle:+.4f} (p = {p_wintle:.2e})")
        print(f"    Wintle MAE = {mae_wintle:.1f}%")
        print(f"    Per-phrase:")
        for i, phrase in enumerate(wintle_phrases):
            err = abs(wintle_preds[i] - wintle_medians[i])
            # Also show Mosteller value if the phrase exists there
            mosteller_val = train_data.get(phrase, None)
            most_str = f"  Most={mosteller_val:.1f}%" if mosteller_val else ""
            print(f"      {phrase:25s}  Pred={wintle_preds[i]:5.1f}%  "
                  f"Wintle={wintle_medians[i]:5.1f}%  Err={err:4.1f}%{most_str}")

        results[group_name] = {
            "rho": r_wintle, "p": p_wintle, "mae": mae_wintle,
            "n": len(wintle_data),
        }

    return results


def main():
    print()
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║  Experiment 08: Bimodality Probe + Wintle Cross-Validation         ║")
    print("╠══════════════════════════════════════════════════════════════════════╣")
    print(f"║  Model: {MODEL:<58s}║")
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

    # Train all axes on Mosteller data
    print("\nTraining axes on Mosteller data...")

    phrases_p = list(PREDICATIVE.keys())
    medians_p = np.array([PREDICATIVE[p] for p in phrases_p])
    sents_p = [PREDICATIVE_TEMPLATE.replace("{PHRASE}", p.lower()) for p in phrases_p]
    diffs_p = embed_texts(sents_p) - bare_emb
    ax_pred, sl_pred, int_pred = train_axis(diffs_p, medians_p)
    r_p, _ = stats.spearmanr(np.clip(sl_pred * (diffs_p @ ax_pred) + int_pred, 0, 100), medians_p)
    print(f"  Predicative: ρ = {r_p:.3f}")

    phrases_m = list(MODAL.keys())
    medians_m = np.array([MODAL[p] for p in phrases_m])
    sents_m = [MODAL_TEMPLATE.replace("{PHRASE}", p) for p in phrases_m]
    diffs_m = embed_texts(sents_m) - bare_emb
    ax_modal, sl_modal, int_modal = train_axis(diffs_m, medians_m)
    r_m, _ = stats.spearmanr(np.clip(sl_modal * (diffs_m @ ax_modal) + int_modal, 0, 100), medians_m)
    print(f"  Modal: ρ = {r_m:.3f}")

    phrases_a = list(ADVERBIAL.keys())
    medians_a = np.array([ADVERBIAL[p] for p in phrases_a])
    sents_a = [ADVERBIAL_TEMPLATE.replace("{PHRASE}", p.lower()) for p in phrases_a]
    diffs_a = embed_texts(sents_a) - bare_emb
    ax_adv, sl_adv, int_adv = train_axis(diffs_a, medians_a)
    r_a, _ = stats.spearmanr(np.clip(sl_adv * (diffs_a @ ax_adv) + int_adv, 0, 100), medians_a)
    print(f"  Adverbial: ρ = {r_a:.3f}")

    # --- Part A ---
    r_bimodal, spread = run_part_a(
        bare_emb, ax_pred, sl_pred, int_pred, ax_modal, sl_modal, int_modal)

    # --- Part B ---
    axes_data = [
        ("Predicative", WINTLE_PREDICATIVE, PREDICATIVE, PREDICATIVE_TEMPLATE,
         ax_pred, sl_pred, int_pred),
        ("Adverbial", WINTLE_ADVERBIAL, ADVERBIAL, ADVERBIAL_TEMPLATE,
         ax_adv, sl_adv, int_adv),
    ]
    wintle_results = run_part_b(bare_emb, axes_data)

    # --- Summary ---
    print()
    print("═" * 78)
    print("SUMMARY")
    print("═" * 78)
    print()
    print(f"  Part A — Bimodality probe:")
    print(f"    Predicative ρ (projection vs. expected): {r_bimodal:+.3f}")
    print(f"    Spread (high-leaning minus low-leaning): {spread:.1f}pp")
    print()
    print(f"  Part B — Wintle cross-validation (train Mosteller → test Wintle):")
    for gname, res in wintle_results.items():
        print(f"    {gname}: ρ = {res['rho']:+.3f}, MAE = {res['mae']:.1f}% (n={res['n']})")

    # Three-dataset comparison
    print()
    print("  THREE-DATASET COMPARISON (Predicative axis):")
    print("    Mosteller (1990, n=238) → Vogel (2022, 21 studies): see paper_validation.py")
    if "Predicative" in wintle_results:
        wr = wintle_results["Predicative"]
        print(f"    Mosteller (1990, n=238) → Wintle (2019, n≈924): ρ = {wr['rho']:+.3f}, MAE = {wr['mae']:.1f}%")
    print()


if __name__ == "__main__":
    main()
