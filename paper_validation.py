"""
Paper Validation: Baselines, Cross-Validation, Bootstrap CIs, and Vogel Validation

This script produces the statistical rigor needed for the paper:

1. BASELINES — Compare supervised ridge axis against:
   - Random direction (mean of 100 random unit vectors)
   - Mean-difference direction (unsupervised centroid)
   - PCA PC1 (unsupervised, best single component)
   - Supervised ridge (our method)

2. LEAVE-ONE-OUT CROSS-VALIDATION — For each expression, train the axis
   on all OTHER expressions, predict the held-out one. Reports LOO ρ and MAE.

3. BOOTSTRAP CONFIDENCE INTERVALS — 10,000 bootstrap resamples of the
   Spearman ρ between supervised projections and Mosteller medians.
   Reports 95% CIs.

4. VOGEL WITHIN-TYPE CROSS-VALIDATION — Train on Mosteller, test on Vogel
   (independent meta-analysis). Done within syntactic types.

5. MOSTELLER-ONLY MODAL AXIS — Retrain modal axis using ONLY expressions
   with Mosteller-derived probabilities (no author estimates).

Usage:
  python3 paper_validation.py [model_name]
  Default model: nomic-embed-text:v1.5
"""

import sys
import numpy as np
import requests
from scipy import stats
from pathlib import Path

from data.mosteller import (
    PREDICATIVE_13, PREDICATIVE_TEMPLATE,
    ADVERBIAL_19, ADVERBIAL_TEMPLATE,
    NOUN_PHRASE_11, NOUN_PHRASE_TEMPLATE,
    MODAL_M10, MODAL_M15, MODAL_TEMPLATE,
    BARE_CLAIM,
)

MODEL = sys.argv[1] if len(sys.argv) > 1 else "nomic-embed-text:v1.5"
OLLAMA_URL = "http://localhost:11434/api/embed"

VOGEL_CSV = Path(__file__).parent / "docs" / "vogel_2022_systematic_review.csv"


def embed_texts(texts, model=MODEL):
    resp = requests.post(OLLAMA_URL, json={"model": model, "input": texts})
    resp.raise_for_status()
    return np.array(resp.json()["embeddings"], dtype=np.float64)


def normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def train_axis(diffs, medians, lam=0.1):
    """Train supervised probability axis via ridge regression.
    Returns: (axis_unit_vector, slope, intercept)."""
    medians_c = (medians - medians.mean()) / medians.std()
    dim = diffs.shape[1]
    w = np.linalg.solve(diffs.T @ diffs + lam * np.eye(dim), diffs.T @ medians_c)
    w = normalize(w)
    projections = diffs @ w
    slope, intercept, _, _, _ = stats.linregress(projections, medians)
    return w, slope, intercept


def project_to_prob(diffs, axis, slope, intercept):
    """Project difference vectors onto axis and convert to probability."""
    proj = diffs @ axis
    return np.clip(slope * proj + intercept, 0, 100)


def spearman_ci_bootstrap(x, y, n_boot=10000, ci=0.95, seed=42):
    """Bootstrap confidence interval for Spearman ρ."""
    rng = np.random.RandomState(seed)
    n = len(x)
    rhos = np.zeros(n_boot)
    for i in range(n_boot):
        idx = rng.randint(0, n, size=n)
        rhos[i], _ = stats.spearmanr(x[idx], y[idx])
    alpha = (1 - ci) / 2
    lo = np.percentile(rhos, alpha * 100)
    hi = np.percentile(rhos, (1 - alpha) * 100)
    r_obs, p_obs = stats.spearmanr(x, y)
    return r_obs, p_obs, lo, hi


# ---------------------------------------------------------------------------
# Training data — sourced from data/mosteller.py
# MODAL_ALL is the legacy 15-item exploratory set (M15);
# MODAL_MOSTELLER_ONLY is now the canonical 10-item M10 set (was 8 items —
# extended to canonical M10 to match experiment_11 / paper §4.1 / §4.4).
# ---------------------------------------------------------------------------

PREDICATIVE = PREDICATIVE_13
ADVERBIAL = ADVERBIAL_19
NOUN_PHRASE = NOUN_PHRASE_11

MODAL_ALL = MODAL_M15
MODAL_MOSTELLER_ONLY = MODAL_M10

# Vogel expressions matched to our syntactic types
VOGEL_PREDICATIVE = {
    "Certain": 95.0, "Almost certain": 85.7, "Very likely": 82.3,
    "Very probable": 82.5, "Likely": 69.2, "Probable": 70.1,
    "Possible": 42.0, "Unlikely": 17.9, "Very unlikely": 11.1,
    "Improbable": 15.8, "Impossible": 7.24,
}

VOGEL_ADVERBIAL = {
    "Usually": 76.8, "Seldom": 11.7, "Rarely": 8.9,
}

# Syntactic groups for iteration
GROUPS = [
    ("Predicative", PREDICATIVE, PREDICATIVE_TEMPLATE),
    ("Adverbial", ADVERBIAL, ADVERBIAL_TEMPLATE),
    ("Noun phrase", NOUN_PHRASE, NOUN_PHRASE_TEMPLATE),
]


# ---------------------------------------------------------------------------
# Section 1: Baselines
# ---------------------------------------------------------------------------

def run_baselines(group_name, expressions, template, bare_emb, dim):
    """Compare supervised axis against baselines for one syntactic group."""
    phrases = list(expressions.keys())
    medians = np.array([expressions[p] for p in phrases])
    n = len(phrases)

    sentences = [template.replace("{PHRASE}", p.lower()) for p in phrases]
    embeddings = embed_texts(sentences)
    diffs = embeddings - bare_emb

    results = {}

    # --- Baseline 1: Random direction (average over 100 trials) ---
    rng = np.random.RandomState(42)
    random_rhos = []
    random_maes = []
    for _ in range(100):
        rand_dir = normalize(rng.randn(dim))
        proj = diffs @ rand_dir
        slope, intercept, _, _, _ = stats.linregress(proj, medians)
        preds = np.clip(slope * proj + intercept, 0, 100)
        r, _ = stats.spearmanr(preds, medians)
        mae = np.mean(np.abs(preds - medians))
        random_rhos.append(abs(r))
        random_maes.append(mae)
    results["Random direction"] = {
        "rho": np.mean(random_rhos),
        "rho_std": np.std(random_rhos),
        "mae": np.mean(random_maes),
        "mae_std": np.std(random_maes),
        "note": "mean of 100 random unit vectors",
    }

    # --- Baseline 2: Mean-difference direction (unsupervised centroid) ---
    mean_dir = normalize(diffs.mean(axis=0))
    proj_mean = diffs @ mean_dir
    slope_m, intercept_m, _, _, _ = stats.linregress(proj_mean, medians)
    preds_mean = np.clip(slope_m * proj_mean + intercept_m, 0, 100)
    r_mean, _ = stats.spearmanr(preds_mean, medians)
    mae_mean = np.mean(np.abs(preds_mean - medians))
    results["Mean-diff direction"] = {"rho": abs(r_mean), "mae": mae_mean}

    # --- Baseline 3: PCA PC1 (unsupervised) ---
    # Center the diffs
    diffs_centered = diffs - diffs.mean(axis=0)
    U, S, Vt = np.linalg.svd(diffs_centered, full_matrices=False)
    pc1 = Vt[0]
    proj_pc1 = diffs @ pc1
    slope_p, intercept_p, _, _, _ = stats.linregress(proj_pc1, medians)
    preds_pc1 = np.clip(slope_p * proj_pc1 + intercept_p, 0, 100)
    r_pc1, _ = stats.spearmanr(preds_pc1, medians)
    mae_pc1 = np.mean(np.abs(preds_pc1 - medians))
    var_explained_pc1 = S[0] ** 2 / np.sum(S ** 2)
    results["PCA PC1"] = {
        "rho": abs(r_pc1), "mae": mae_pc1,
        "var_explained": var_explained_pc1,
    }

    # --- Supervised ridge regression (our method, in-sample) ---
    w_sup, sl_sup, int_sup = train_axis(diffs, medians)
    preds_sup = project_to_prob(diffs, w_sup, sl_sup, int_sup)
    r_sup, _ = stats.spearmanr(preds_sup, medians)
    mae_sup = np.mean(np.abs(preds_sup - medians))
    results["Supervised ridge (in-sample)"] = {"rho": abs(r_sup), "mae": mae_sup}

    return results, diffs, medians, phrases


# ---------------------------------------------------------------------------
# Section 2: Leave-One-Out Cross-Validation
# ---------------------------------------------------------------------------

def run_loo(diffs, medians, phrases):
    """Leave-one-out cross-validation on the ridge regression axis."""
    n = len(medians)
    loo_preds = np.zeros(n)

    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        diffs_train = diffs[mask]
        medians_train = medians[mask]

        w, slope, intercept = train_axis(diffs_train, medians_train)
        loo_preds[i] = float(np.clip(
            slope * (diffs[i] @ w) + intercept, 0, 100))

    r_loo, p_loo = stats.spearmanr(loo_preds, medians)
    mae_loo = np.mean(np.abs(loo_preds - medians))
    max_err = np.max(np.abs(loo_preds - medians))

    return {
        "rho": r_loo, "p": p_loo, "mae": mae_loo, "max_err": max_err,
        "predictions": loo_preds,
    }


# ---------------------------------------------------------------------------
# Section 3: Bootstrap CIs
# ---------------------------------------------------------------------------

def run_bootstrap(diffs, medians):
    """Bootstrap CIs for supervised ridge Spearman ρ."""
    w, slope, intercept = train_axis(diffs, medians)
    preds = project_to_prob(diffs, w, slope, intercept)
    r_obs, p_obs, ci_lo, ci_hi = spearman_ci_bootstrap(preds, medians)
    return {"rho": r_obs, "p": p_obs, "ci_lo": ci_lo, "ci_hi": ci_hi}


# ---------------------------------------------------------------------------
# Section 4: Vogel Within-Type Cross-Validation
# ---------------------------------------------------------------------------

def run_vogel_validation(group_name, train_expressions, template, bare_emb,
                         vogel_expressions):
    """Train on Mosteller, test on Vogel (within same syntactic type)."""
    if not vogel_expressions:
        return None

    # Train axis on Mosteller
    train_phrases = list(train_expressions.keys())
    train_medians = np.array([train_expressions[p] for p in train_phrases])
    train_sentences = [template.replace("{PHRASE}", p.lower()) for p in train_phrases]
    train_embs = embed_texts(train_sentences)
    train_diffs = train_embs - bare_emb
    w, slope, intercept = train_axis(train_diffs, train_medians)

    # Test on Vogel
    vogel_phrases = list(vogel_expressions.keys())
    vogel_medians = np.array([vogel_expressions[p] for p in vogel_phrases])
    vogel_sentences = [template.replace("{PHRASE}", p.lower()) for p in vogel_phrases]
    vogel_embs = embed_texts(vogel_sentences)
    vogel_diffs = vogel_embs - bare_emb

    vogel_preds = project_to_prob(vogel_diffs, w, slope, intercept)
    r_vogel, p_vogel = stats.spearmanr(vogel_preds, vogel_medians)
    mae_vogel = np.mean(np.abs(vogel_preds - vogel_medians))

    return {
        "rho": r_vogel, "p": p_vogel, "mae": mae_vogel,
        "n": len(vogel_phrases),
        "predictions": list(zip(vogel_phrases, vogel_preds, vogel_medians)),
    }


# ---------------------------------------------------------------------------
# Section 5: Mosteller-Only Modal Axis
# ---------------------------------------------------------------------------

def run_modal_comparison(bare_emb):
    """Compare modal axis trained with ALL expressions vs. Mosteller-only."""
    # Train both axes
    results = {}

    for label, expressions in [("Modal (M15 incl. author-estimated)", MODAL_ALL),
                                ("Modal (Mosteller-only M10)", MODAL_MOSTELLER_ONLY)]:
        phrases = list(expressions.keys())
        medians = np.array([expressions[p] for p in phrases])
        sentences = [MODAL_TEMPLATE.replace("{PHRASE}", p) for p in phrases]
        embs = embed_texts(sentences)
        diffs = embs - bare_emb
        w, slope, intercept = train_axis(diffs, medians)

        # In-sample fit
        preds = project_to_prob(diffs, w, slope, intercept)
        r, _ = stats.spearmanr(preds, medians)
        mae = np.mean(np.abs(preds - medians))

        # LOO
        n = len(medians)
        loo_preds = np.zeros(n)
        for i in range(n):
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            w_i, sl_i, int_i = train_axis(diffs[mask], medians[mask])
            loo_preds[i] = float(np.clip(sl_i * (diffs[i] @ w_i) + int_i, 0, 100))
        r_loo, _ = stats.spearmanr(loo_preds, medians)
        mae_loo = np.mean(np.abs(loo_preds - medians))

        results[label] = {
            "n": len(phrases),
            "in_sample_rho": abs(r), "in_sample_mae": mae,
            "loo_rho": abs(r_loo), "loo_mae": mae_loo,
            "axis": w, "slope": slope, "intercept": intercept,
        }

    # Cross-test: train on Mosteller-only, test on the 7 author-estimated phrases
    author_only = {k: v for k, v in MODAL_ALL.items()
                   if k not in MODAL_MOSTELLER_ONLY}
    if author_only:
        w_m = results["Modal (Mosteller-only M10)"]["axis"]
        sl_m = results["Modal (Mosteller-only M10)"]["slope"]
        int_m = results["Modal (Mosteller-only M10)"]["intercept"]

        a_phrases = list(author_only.keys())
        a_medians = np.array([author_only[p] for p in a_phrases])
        a_sentences = [MODAL_TEMPLATE.replace("{PHRASE}", p) for p in a_phrases]
        a_embs = embed_texts(a_sentences)
        a_diffs = a_embs - bare_emb
        a_preds = project_to_prob(a_diffs, w_m, sl_m, int_m)
        r_cross, _ = stats.spearmanr(a_preds, a_medians)
        mae_cross = np.mean(np.abs(a_preds - a_medians))
        results["Mosteller-only → author-estimated"] = {
            "rho": abs(r_cross), "mae": mae_cross, "n": len(a_phrases),
            "predictions": list(zip(a_phrases, a_preds, a_medians)),
        }

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print()
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║  Paper Validation: Baselines, LOO, Bootstrap CIs, Vogel           ║")
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

    # ═══════════════════════════════════════════════════════════════════════
    # SECTION 1: BASELINES
    # ═══════════════════════════════════════════════════════════════════════
    print()
    print("═" * 78)
    print("SECTION 1: BASELINES (supervised ridge vs. alternatives)")
    print("═" * 78)

    vogel_map = {
        "Predicative": VOGEL_PREDICATIVE,
        "Adverbial": VOGEL_ADVERBIAL,
        "Noun phrase": {},
    }

    all_diffs = {}
    all_medians = {}
    all_phrases = {}

    for group_name, expressions, template in GROUPS:
        print(f"\n{'─'*60}")
        print(f"  {group_name} (n={len(expressions)})")
        print(f"{'─'*60}")

        baselines, diffs, medians, phrases = run_baselines(
            group_name, expressions, template, bare_emb, dim)

        all_diffs[group_name] = diffs
        all_medians[group_name] = medians
        all_phrases[group_name] = phrases

        print(f"\n  {'Method':<30s}  {'ρ':>6s}  {'MAE':>7s}  Notes")
        print(f"  {'─'*30}  {'─'*6}  {'─'*7}  {'─'*25}")
        for method, res in baselines.items():
            rho_str = f"{res['rho']:.3f}"
            mae_str = f"{res['mae']:.1f}%"
            note = res.get("note", "")
            if "rho_std" in res:
                rho_str += f"±{res['rho_std']:.3f}"
                mae_str = f"{res['mae']:.1f}±{res['mae_std']:.1f}%"
            if "var_explained" in res:
                note = f"PC1 var={res['var_explained']*100:.0f}%"
            print(f"  {method:<30s}  {rho_str:>10s}  {mae_str:>9s}  {note}")

    # ═══════════════════════════════════════════════════════════════════════
    # SECTION 2: LEAVE-ONE-OUT CROSS-VALIDATION
    # ═══════════════════════════════════════════════════════════════════════
    print()
    print("═" * 78)
    print("SECTION 2: LEAVE-ONE-OUT CROSS-VALIDATION")
    print("═" * 78)

    for group_name, _, _ in GROUPS:
        diffs = all_diffs[group_name]
        medians = all_medians[group_name]
        phrases = all_phrases[group_name]

        loo = run_loo(diffs, medians, phrases)
        print(f"\n  {group_name} (n={len(medians)}):")
        print(f"    LOO Spearman ρ = {loo['rho']:+.4f} (p = {loo['p']:.2e})")
        print(f"    LOO MAE = {loo['mae']:.1f}%")
        print(f"    LOO max error = {loo['max_err']:.1f}%")

        # Show worst LOO predictions
        errors = np.abs(loo["predictions"] - medians)
        worst_idx = np.argsort(errors)[::-1][:3]
        print(f"    Worst LOO predictions:")
        for idx in worst_idx:
            print(f"      {phrases[idx]:25s}  Pred={loo['predictions'][idx]:5.1f}%  "
                  f"True={medians[idx]:5.1f}%  Err={errors[idx]:4.1f}%")

    # ═══════════════════════════════════════════════════════════════════════
    # SECTION 3: BOOTSTRAP CONFIDENCE INTERVALS
    # ═══════════════════════════════════════════════════════════════════════
    print()
    print("═" * 78)
    print("SECTION 3: BOOTSTRAP 95% CONFIDENCE INTERVALS (10,000 resamples)")
    print("═" * 78)

    for group_name, _, _ in GROUPS:
        diffs = all_diffs[group_name]
        medians = all_medians[group_name]

        boot = run_bootstrap(diffs, medians)
        print(f"\n  {group_name}:")
        print(f"    Supervised ρ = {boot['rho']:+.4f}  "
              f"95% CI: [{boot['ci_lo']:+.4f}, {boot['ci_hi']:+.4f}]")

    # Also bootstrap the LOO predictions
    print(f"\n  LOO bootstrap CIs:")
    for group_name, _, _ in GROUPS:
        diffs = all_diffs[group_name]
        medians = all_medians[group_name]
        loo = run_loo(diffs, medians, all_phrases[group_name])
        r_obs, _, ci_lo, ci_hi = spearman_ci_bootstrap(
            loo["predictions"], medians)
        print(f"    {group_name}: LOO ρ = {r_obs:+.4f}  "
              f"95% CI: [{ci_lo:+.4f}, {ci_hi:+.4f}]")

    # ═══════════════════════════════════════════════════════════════════════
    # SECTION 4: VOGEL WITHIN-TYPE CROSS-VALIDATION
    # ═══════════════════════════════════════════════════════════════════════
    print()
    print("═" * 78)
    print("SECTION 4: VOGEL WITHIN-TYPE CROSS-VALIDATION")
    print("  (Train on Mosteller, test on Vogel — independent meta-analysis)")
    print("═" * 78)

    for group_name, expressions, template in GROUPS:
        vogel_exprs = vogel_map.get(group_name, {})
        if not vogel_exprs:
            print(f"\n  {group_name}: No Vogel expressions for this type. Skipped.")
            continue

        vogel_result = run_vogel_validation(
            group_name, expressions, template, bare_emb, vogel_exprs)

        if vogel_result:
            print(f"\n  {group_name} (n_train={len(expressions)}, "
                  f"n_test={vogel_result['n']}):")
            print(f"    Vogel Spearman ρ = {vogel_result['rho']:+.4f} "
                  f"(p = {vogel_result['p']:.2e})")
            print(f"    Vogel MAE = {vogel_result['mae']:.1f}%")
            print(f"    Per-phrase predictions:")
            for phrase, pred, true in vogel_result["predictions"]:
                err = abs(pred - true)
                print(f"      {phrase:25s}  Pred={pred:5.1f}%  "
                      f"Vogel={true:5.1f}%  Err={err:4.1f}%")

    # ═══════════════════════════════════════════════════════════════════════
    # SECTION 5: MOSTELLER-ONLY MODAL AXIS
    # ═══════════════════════════════════════════════════════════════════════
    print()
    print("═" * 78)
    print("SECTION 5: MODAL AXIS — ALL EXPRESSIONS vs. MOSTELLER-ONLY")
    print("═" * 78)

    modal_results = run_modal_comparison(bare_emb)

    for label, res in modal_results.items():
        if "predictions" in res:
            print(f"\n  {label} (n={res['n']}):")
            print(f"    ρ = {res['rho']:.3f}, MAE = {res['mae']:.1f}%")
            for phrase, pred, true in res["predictions"]:
                print(f"      {phrase:20s}  Pred={pred:5.1f}%  True={true:5.1f}%  "
                      f"Err={abs(pred-true):4.1f}%")
        else:
            print(f"\n  {label} (n={res['n']}):")
            print(f"    In-sample:  ρ = {res['in_sample_rho']:.3f}, "
                  f"MAE = {res['in_sample_mae']:.1f}%")
            print(f"    LOO:        ρ = {res['loo_rho']:.3f}, "
                  f"MAE = {res['loo_mae']:.1f}%")

    # ═══════════════════════════════════════════════════════════════════════
    # SUMMARY TABLE
    # ═══════════════════════════════════════════════════════════════════════
    print()
    print("═" * 78)
    print("SUMMARY TABLE FOR PAPER")
    print("═" * 78)
    print()
    print(f"Model: {MODEL}")
    print()

    # Run LOO for each group and collect
    print(f"  {'Group':<15s}  {'n':>3s}  {'In-sample ρ':>12s}  {'LOO ρ':>8s}  "
          f"{'LOO MAE':>8s}  {'Bootstrap 95% CI':>20s}  {'Random ρ':>10s}")
    print(f"  {'─'*15}  {'─'*3}  {'─'*12}  {'─'*8}  {'─'*8}  {'─'*20}  {'─'*10}")

    for group_name, expressions, template in GROUPS:
        diffs = all_diffs[group_name]
        medians = all_medians[group_name]
        phrases = all_phrases[group_name]
        n = len(medians)

        # In-sample
        w, sl, intc = train_axis(diffs, medians)
        preds_is = project_to_prob(diffs, w, sl, intc)
        r_is, _ = stats.spearmanr(preds_is, medians)

        # LOO
        loo = run_loo(diffs, medians, phrases)

        # Bootstrap on LOO
        _, _, ci_lo, ci_hi = spearman_ci_bootstrap(loo["predictions"], medians)

        # Random baseline
        rng = np.random.RandomState(42)
        rand_rhos = []
        for _ in range(100):
            rd = normalize(rng.randn(dim))
            rp = diffs @ rd
            rsl, rint, _, _, _ = stats.linregress(rp, medians)
            rpreds = np.clip(rsl * rp + rint, 0, 100)
            rr, _ = stats.spearmanr(rpreds, medians)
            rand_rhos.append(abs(rr))

        print(f"  {group_name:<15s}  {n:3d}  {abs(r_is):12.3f}  "
              f"{abs(loo['rho']):8.3f}  {loo['mae']:7.1f}%  "
              f"[{ci_lo:+.3f}, {ci_hi:+.3f}]  "
              f"{np.mean(rand_rhos):.3f}±{np.std(rand_rhos):.3f}")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
