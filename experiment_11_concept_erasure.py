"""
Experiment 11: Linear Concept Erasure (Functional vs. Geometric)

Recent linear-feature interpretability work (Ji et al. 2025; Valentin et al.
2025; Marks & Tegmark 2023) operates on decoder LLM internal states (residual
streams, unembedding matrices) and demonstrates *functional* content via
causal intervention (steering, activation patching). Sentence embedding
models lack a token-generation pathway, so those tools don't directly apply.

This experiment is the embedding-class analogue: representation-level
intervention (rank-1 projection erasure) instead of activation steering.
The reviewer concern this addresses is "the probability axis is just
geometric correlation, not functional structure."

For each ordered pair (A, B) with A ≠ B and A, B ∈ {Predicative,
Adverbial, NounPhrase, Modal}:

  1. Train v_A on type A and v_B on type B (ridge regression on diff-vectors
     vs. Mosteller medians).
  2. BASELINE: apply v_B's calibrated decoder to type-B diff vectors.
     Report Spearman ρ and MAE vs. Mosteller_B.
  3. TREATMENT: rank-1 erase v_A from type-B diff vectors:
        e' = e - (e · v_A) v_A
     Apply v_B (same slope/intercept). Report ρ and MAE.
  4. CONTROL (a) — uniform random: 100 random unit directions sampled from
     Gaussian + normalize. Erase each from type-B diffs. Distribution of ρ.
  5. CONTROL (b) — cosine-matched random: 100 unit directions r with
        |cos(r, v_B)| = |cos(v_A, v_B)|
     constructed as r = sign·c·v_B + sqrt(1-c²)·u_orth where u_orth is a
     random unit vector orthogonal to v_B and sign is ±1 randomly. This
     matches the geometric overlap with v_B, so any extra degradation from
     real-axis erasure is evidence of functional probability content beyond
     geometric overlap.
  6. Z-scores: z = (ρ_treatment − mean) / std for each control distribution.
     Negative z ⇒ real-axis erasure degrades more than the random baseline.

If z_matched << 0, then erasing the *probability* axis hurts probability
prediction more than erasing a random direction with the same geometric
overlap with v_B. That is the embedding-class analogue of "functional
intervention," not just geometric correlation.

Usage:
  python3 experiment_11_concept_erasure.py [model_name]
  Default model: mxbai-embed-large
  Recommended secondary: qwen3-embedding (architectural diversity)
"""

import sys
import numpy as np
import requests
from scipy import stats

MODEL = sys.argv[1] if len(sys.argv) > 1 else "mxbai-embed-large"
OLLAMA_URL = "http://localhost:11434/api/embed"

N_RANDOM = 100
SEED = 42


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


def erase(diffs, v):
    """Rank-1 projection erasure: e' = e - (e·v) v.  v assumed unit-norm."""
    return diffs - np.outer(diffs @ v, v)


def predict(diffs, w, slope, intercept):
    """Apply calibrated axis decoder to diffs. Returns clipped [0,100] preds."""
    return np.clip(slope * (diffs @ w) + intercept, 0.0, 100.0)


def metrics(preds, medians):
    rho, _ = stats.spearmanr(preds, medians)
    mae = float(np.mean(np.abs(preds - medians)))
    return rho, mae


def random_unit(rng, dim):
    v = rng.standard_normal(dim)
    return normalize(v)


def random_cosine_matched(rng, v_b, target_abs_cos):
    """
    Sample a unit vector r with |cos(r, v_b)| = target_abs_cos.
        r = sign·c·v_b + sqrt(1−c²)·u_orth
    where u_orth is a random unit vector orthogonal to v_b and sign is ±1.
    """
    dim = v_b.shape[0]
    c = float(np.clip(target_abs_cos, 0.0, 1.0))
    u = rng.standard_normal(dim)
    u_orth = u - (u @ v_b) * v_b
    u_orth = normalize(u_orth)
    sign = 1.0 if rng.random() < 0.5 else -1.0
    r = sign * c * v_b + np.sqrt(max(0.0, 1.0 - c * c)) * u_orth
    return normalize(r)


# ────────────────────────────────────────────────────────────────────────────
# Mosteller-derived expression sets (verbatim from experiment_10).
# Modal subset uses ONLY Mosteller-grounded values — estimated values used in
# experiment_03 (perhaps, conceivably, arguably, definitely, presumably,
# undoubtedly, maybe) are excluded so calibration noise doesn't enter the
# erasure analysis. n_modal = 10.
# ────────────────────────────────────────────────────────────────────────────

BARE_CLAIM = "The experiment will succeed"

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

# Modal: ONLY Mosteller-derived values (n=10).  See docstring header.
MODAL = {
    "certainly": 99.6,         # adverb of "Certain"
    "almost certainly": 90.2,  # adverb of "Almost certain"
    "very likely": 87.5,
    "likely": 71.1,
    "probably": 70.2,          # adverb of "Probable"
    "very probably": 89.7,     # adverb of "Very probable"
    "possibly": 38.5,          # adverb of "Possible"
    "unlikely": 17.2,
    "very unlikely": 5.0,
    "improbably": 12.5,        # adverb of "Improbable"
}
MODAL_TEMPLATE = "The experiment will {PHRASE} succeed"


def build_group_data(exprs, template, bare_emb):
    phrases = list(exprs.keys())
    medians = np.array([exprs[p] for p in phrases], dtype=np.float64)
    sentences = [template.replace("{PHRASE}", p.lower()) for p in phrases]
    embs = embed_texts(sentences)
    diffs = embs - bare_emb
    return diffs, medians


def main():
    print()
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║  Experiment 11: Linear Concept Erasure (Functional vs. Geometric)  ║")
    print("╠══════════════════════════════════════════════════════════════════════╣")
    print(f"║  Model: {MODEL:<58s}║")
    print("║                                                                    ║")
    print("║  Rank-1 erase v_A from type-B diffs; re-decode with v_B.           ║")
    print("║  Compare degradation against uniform-random and cosine-matched     ║")
    print("║  random direction baselines (N=100 each, seed=42).                 ║")
    print("║                                                                    ║")
    print("║  Negative z_matched ⇒ erasing the probability axis hurts more      ║")
    print("║  than erasing a random direction with matched geometric overlap    ║")
    print("║  ⇒ functional content, not just geometric correlation.             ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print()

    try:
        test = embed_texts(["test"])
        dim = test.shape[1]
        print(f"Model loaded. Dim: {dim}")
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    rng = np.random.RandomState(SEED)
    bare_emb = embed_texts([BARE_CLAIM])[0]

    # ═══════════════════════════════════════════════════════════════════════
    # Embed all four types and train per-type axes
    # ═══════════════════════════════════════════════════════════════════════

    groups = {
        "Predicative": (PREDICATIVE, PREDICATIVE_TEMPLATE),
        "Adverbial":   (ADVERBIAL,   ADVERBIAL_TEMPLATE),
        "NounPhrase":  (NOUN_PHRASE, NOUN_PHRASE_TEMPLATE),
        "Modal":       (MODAL,       MODAL_TEMPLATE),
    }

    diffs_by_type   = {}
    medians_by_type = {}
    axes_by_type    = {}   # name -> (w, slope, intercept)

    print()
    print("═" * 78)
    print("AXIS TRAINING (in-sample fit per type)")
    print("═" * 78)
    print(f"  {'Type':<12s} {'n':>4s}  {'in-sample ρ':>14s}  {'in-sample MAE':>15s}")

    for gname, (exprs, template) in groups.items():
        d, m = build_group_data(exprs, template, bare_emb)
        diffs_by_type[gname]   = d
        medians_by_type[gname] = m
        w, sl, intc = train_axis(d, m)
        axes_by_type[gname] = (w, sl, intc)
        preds = predict(d, w, sl, intc)
        rho_is, mae_is = metrics(preds, m)
        print(f"  {gname:<12s} {len(m):>4d}  {rho_is:>+14.4f}  {mae_is:>15.3f}")

    # ═══════════════════════════════════════════════════════════════════════
    # Pairwise erasure analysis (12 ordered pairs)
    # ═══════════════════════════════════════════════════════════════════════

    type_names = list(groups.keys())
    pair_records = []   # for summary aggregation

    print()
    print("═" * 78)
    print("PAIRWISE ERASURE ANALYSIS")
    print("  For each (eraser A, target B): erase v_A from type-B diffs;")
    print("  decode with v_B's calibrated regressor; compare vs. random baselines.")
    print("")
    print("  Two parallel views:")
    print("    ρ-based   : higher ρ_treat = better; functional ⇒ z_ρ   < −2")
    print("                verdicts: ↓↓ functional / ↓ trend / —")
    print("    MAE-based : lower MAE_treat = better; functional ⇒ z_MAE > +2")
    print("                verdicts: ↑↑ functional / ↑ trend / —")
    print("═" * 78)

    rho_header = (
        f"  {'A→B':<24s} {'cos':>7s} | "
        f"{'ρ_base':>7s} {'ρ_treat':>8s} | "
        f"{'ρ_unif μ±σ':>16s} {'ρ_match μ±σ':>16s} | "
        f"{'z_unif':>7s} {'z_match':>8s} | verdict"
    )
    mae_header = (
        f"  {'A→B':<24s} {'cos':>7s} | "
        f"{'MAE_b':>6s} {'MAE_t':>7s} {'ΔMAE':>7s} | "
        f"{'MAE_unif μ±σ':>18s} {'MAE_match μ±σ':>18s} | "
        f"{'z_uMAE':>7s} {'z_mMAE':>8s} | verdict"
    )

    for a_name in type_names:
        for b_name in type_names:
            if a_name == b_name:
                continue

            v_a, _, _              = axes_by_type[a_name]
            v_b, slope_b, intc_b   = axes_by_type[b_name]
            diffs_b                = diffs_by_type[b_name]
            medians_b              = medians_by_type[b_name]

            cos_ab = float(v_a @ v_b)         # both unit norm
            abs_cos_ab = abs(cos_ab)

            # Baseline: v_B applied to unmodified type-B diffs.
            preds_base = predict(diffs_b, v_b, slope_b, intc_b)
            rho_base, mae_base = metrics(preds_base, medians_b)

            # Treatment: erase v_A, then decode with v_B.
            diffs_b_treat = erase(diffs_b, v_a)
            preds_treat = predict(diffs_b_treat, v_b, slope_b, intc_b)
            rho_treat, mae_treat = metrics(preds_treat, medians_b)

            # Control (a): uniform random unit directions.
            unif_rng = np.random.RandomState(SEED)
            unif_rhos = np.zeros(N_RANDOM)
            unif_maes = np.zeros(N_RANDOM)
            for k in range(N_RANDOM):
                r = random_unit(unif_rng, dim)
                d_e = erase(diffs_b, r)
                p_e = predict(d_e, v_b, slope_b, intc_b)
                rho_k, mae_k = metrics(p_e, medians_b)
                unif_rhos[k] = rho_k if not np.isnan(rho_k) else 0.0
                unif_maes[k] = mae_k

            # Control (b): cosine-matched random unit directions.
            match_rng = np.random.RandomState(SEED + 1)
            match_rhos = np.zeros(N_RANDOM)
            match_maes = np.zeros(N_RANDOM)
            for k in range(N_RANDOM):
                r = random_cosine_matched(match_rng, v_b, abs_cos_ab)
                d_e = erase(diffs_b, r)
                p_e = predict(d_e, v_b, slope_b, intc_b)
                rho_k, mae_k = metrics(p_e, medians_b)
                match_rhos[k] = rho_k if not np.isnan(rho_k) else 0.0
                match_maes[k] = mae_k

            mu_u, sd_u = float(np.mean(unif_rhos)),  float(np.std(unif_rhos))
            mu_m, sd_m = float(np.mean(match_rhos)), float(np.std(match_rhos))
            z_u = (rho_treat - mu_u) / sd_u if sd_u > 1e-12 else 0.0
            z_m = (rho_treat - mu_m) / sd_m if sd_m > 1e-12 else 0.0

            mu_u_mae, sd_u_mae = float(np.mean(unif_maes)),  float(np.std(unif_maes))
            mu_m_mae, sd_m_mae = float(np.mean(match_maes)), float(np.std(match_maes))
            z_u_mae = (mae_treat - mu_u_mae) / sd_u_mae if sd_u_mae > 1e-12 else 0.0
            z_m_mae = (mae_treat - mu_m_mae) / sd_m_mae if sd_m_mae > 1e-12 else 0.0

            # ρ-based verdict (higher ρ better; functional ⇒ z << 0)
            if z_m < -2.0:
                rho_verdict = "↓↓ functional"
            elif z_m < -1.0:
                rho_verdict = "↓ trend"
            else:
                rho_verdict = "—"

            # MAE-based verdict (lower MAE better; functional ⇒ z >> 0)
            if z_m_mae > 2.0:
                mae_verdict = "↑↑ functional"
            elif z_m_mae > 1.0:
                mae_verdict = "↑ trend"
            else:
                mae_verdict = "—"

            pair_records.append({
                "a": a_name, "b": b_name, "cos": cos_ab,
                "rho_base": rho_base, "rho_treat": rho_treat,
                "mae_base": mae_base, "mae_treat": mae_treat,
                "unif_mu": mu_u, "unif_sd": sd_u,
                "match_mu": mu_m, "match_sd": sd_m,
                "z_unif": z_u, "z_match": z_m,
                "unif_mu_mae": mu_u_mae, "unif_sd_mae": sd_u_mae,
                "match_mu_mae": mu_m_mae, "match_sd_mae": sd_m_mae,
                "z_unif_mae": z_u_mae, "z_match_mae": z_m_mae,
                "rho_verdict": rho_verdict,
                "mae_verdict": mae_verdict,
                # legacy alias (kept for any downstream consumers)
                "verdict": rho_verdict,
            })

    # ---- Print ρ-based table -------------------------------------------------
    print()
    print("  ρ-based view (higher ρ_treat better; functional ⇒ z_match < −2)")
    print()
    print(rho_header)
    print("  " + "-" * (len(rho_header) - 2))
    for rec in pair_records:
        label = f"{rec['a']}→{rec['b']}"
        print(
            f"  {label:<24s} {rec['cos']:>+7.3f} | "
            f"{rec['rho_base']:>+7.3f} {rec['rho_treat']:>+8.3f} | "
            f"{rec['unif_mu']:>+7.3f}±{rec['unif_sd']:>4.2f}  "
            f"{rec['match_mu']:>+7.3f}±{rec['match_sd']:>4.2f} | "
            f"{rec['z_unif']:>+7.2f} {rec['z_match']:>+8.2f} | {rec['rho_verdict']}"
        )

    # ---- Print MAE-based table ----------------------------------------------
    print()
    print("  MAE-based view (lower MAE_treat better; functional ⇒ z_match_MAE > +2)")
    print()
    print(mae_header)
    print("  " + "-" * (len(mae_header) - 2))
    for rec in pair_records:
        label = f"{rec['a']}→{rec['b']}"
        delta_mae = rec['mae_treat'] - rec['mae_base']
        print(
            f"  {label:<24s} {rec['cos']:>+7.3f} | "
            f"{rec['mae_base']:>6.2f} {rec['mae_treat']:>7.2f} {delta_mae:>+7.2f} | "
            f"{rec['unif_mu_mae']:>8.2f}±{rec['unif_sd_mae']:>5.2f}    "
            f"{rec['match_mu_mae']:>8.2f}±{rec['match_sd_mae']:>5.2f}  | "
            f"{rec['z_unif_mae']:>+7.2f} {rec['z_match_mae']:>+8.2f} | {rec['mae_verdict']}"
        )

    # ═══════════════════════════════════════════════════════════════════════
    # Cross-pair correlations (across the 12 ordered pairs)
    # Tests whether geometric overlap (cos) predicts treatment effect size.
    # ═══════════════════════════════════════════════════════════════════════

    cos_arr           = np.array([r["cos"] for r in pair_records])
    delta_mae_arr     = np.array([r["mae_treat"] - r["mae_base"] for r in pair_records])
    delta_rho_arr     = np.array([r["rho_treat"] - r["rho_base"] for r in pair_records])
    z_match_mae_arr   = np.array([r["z_match_mae"] for r in pair_records])

    r_cos_dmae,    _ = stats.pearsonr(cos_arr, delta_mae_arr)
    r_cos_zmmae,   _ = stats.pearsonr(cos_arr, z_match_mae_arr)
    r_cos_drho,    _ = stats.pearsonr(cos_arr, delta_rho_arr)

    print()
    print("═" * 78)
    print("CROSS-PAIR CORRELATIONS")
    print("  Across the 12 ordered pairs:")
    print(f"    Pearson r(cos, ΔMAE_real)         = {r_cos_dmae:+.3f}")
    print(f"    Pearson r(cos, z_match_MAE)       = {r_cos_zmmae:+.3f}")
    print(f"    Pearson r(cos, ρ_treat - ρ_base)  = {r_cos_drho:+.3f}")
    print("═" * 78)

    # ═══════════════════════════════════════════════════════════════════════
    # Headline section: predicative ↔ modal (both directions)
    # Both axes are calibrated to the same Mosteller predicative medians,
    # one in adjective form, one in adverb form. High cosine + functional
    # erasure is the rhetorically load-bearing story.
    # ═══════════════════════════════════════════════════════════════════════

    print()
    print("═" * 78)
    print("HEADLINE: Predicative ↔ Modal")
    print("  Same Mosteller medians, different syntactic forms (adj vs. adv).")
    print("  These axes have the strongest a priori reason to share content.")
    print("═" * 78)

    headline_pairs = [("Predicative", "Modal"), ("Modal", "Predicative")]
    for a_name, b_name in headline_pairs:
        rec = next(r for r in pair_records if r["a"] == a_name and r["b"] == b_name)
        print()
        print(f"  {a_name} → {b_name}")
        print(f"    cos(v_A, v_B):          {rec['cos']:+.4f}")
        print(f"    ρ baseline:             {rec['rho_base']:+.4f}")
        print(f"    ρ after v_A erasure:    {rec['rho_treat']:+.4f}  "
              f"(Δ = {rec['rho_treat'] - rec['rho_base']:+.4f})")
        print(f"    MAE baseline:           {rec['mae_base']:.3f}")
        print(f"    MAE after v_A erasure:  {rec['mae_treat']:.3f}  "
              f"(Δ = {rec['mae_treat'] - rec['mae_base']:+.3f})")
        print(f"    Uniform-random MAE:     {rec['unif_mu_mae']:.3f} ± {rec['unif_sd_mae']:.3f}"
              f"   →  z_MAE = {rec['z_unif_mae']:+.2f}")
        print(f"    Cosine-matched MAE:     {rec['match_mu_mae']:.3f} ± {rec['match_sd_mae']:.3f}"
              f"   →  z_MAE = {rec['z_match_mae']:+.2f}")
        print(f"    MAE-based verdict:      {rec['mae_verdict']}")
        print(f"    Uniform-random ρ:       {rec['unif_mu']:+.4f} ± {rec['unif_sd']:.4f}"
              f"   →  z = {rec['z_unif']:+.2f}")
        print(f"    Cosine-matched ρ:       {rec['match_mu']:+.4f} ± {rec['match_sd']:.4f}"
              f"   →  z = {rec['z_match']:+.2f}")
        print(f"    ρ-based verdict:        {rec['rho_verdict']}")

    # ═══════════════════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════════════════

    print()
    print("═" * 78)
    print("SUMMARY")
    print("═" * 78)

    n_pairs = len(pair_records)

    def bucket_neg(records, key):
        # ρ-based (lower z = more functional)
        funcs  = sum(1 for r in records if r[key] < -2.0)
        trends = sum(1 for r in records if -2.0 <= r[key] < -1.0)
        neut   = n_pairs - funcs - trends
        return funcs, trends, neut

    def bucket_pos(records, key):
        # MAE-based (higher z = more functional)
        funcs  = sum(1 for r in records if r[key] > 2.0)
        trends = sum(1 for r in records if 1.0 < r[key] <= 2.0)
        neut   = n_pairs - funcs - trends
        return funcs, trends, neut

    f_u,    t_u,    n_u    = bucket_neg(pair_records, "z_unif")
    f_m,    t_m,    n_m    = bucket_neg(pair_records, "z_match")
    f_uM,   t_uM,   n_uM   = bucket_pos(pair_records, "z_unif_mae")
    f_mM,   t_mM,   n_mM   = bucket_pos(pair_records, "z_match_mae")

    print()
    print(f"  Pairs evaluated: {n_pairs} (4×3 ordered cross-type)")
    print()
    print(f"  ρ-based (higher ρ_treat = better; functional ⇒ z < −2)")
    print(f"  Vs. UNIFORM random baseline:")
    print(f"    ↓↓ functional (z < −2):    {f_u:>2d} / {n_pairs}")
    print(f"    ↓  trend     (−2 ≤ z<−1):  {t_u:>2d} / {n_pairs}")
    print(f"    —  neutral   (z ≥ −1):     {n_u:>2d} / {n_pairs}")
    print()
    print(f"  Vs. COSINE-MATCHED random baseline (the rigorous comparison):")
    print(f"    ↓↓ functional (z < −2):    {f_m:>2d} / {n_pairs}")
    print(f"    ↓  trend     (−2 ≤ z<−1):  {t_m:>2d} / {n_pairs}")
    print(f"    —  neutral   (z ≥ −1):     {n_m:>2d} / {n_pairs}")
    print()
    print(f"  MAE-based (lower MAE_treat = better; functional ⇒ z_MAE > +2)")
    print(f"  Vs. MAE-based UNIFORM random baseline:")
    print(f"    ↑↑ functional (z_MAE > +2):    {f_uM:>2d} / {n_pairs}")
    print(f"    ↑  trend     (1 ≤ z_MAE < 2):  {t_uM:>2d} / {n_pairs}")
    print(f"    —  neutral   (z_MAE < 1):      {n_uM:>2d} / {n_pairs}")
    print()
    print(f"  Vs. MAE-based COSINE-MATCHED random baseline:")
    print(f"    ↑↑ functional (z_MAE > +2):    {f_mM:>2d} / {n_pairs}")
    print(f"    ↑  trend     (1 ≤ z_MAE < 2):  {t_mM:>2d} / {n_pairs}")
    print(f"    —  neutral   (z_MAE < 1):      {n_mM:>2d} / {n_pairs}")
    print()
    print("  Interpretation:")
    print("    The cosine-matched control fixes the geometric overlap with v_B.")
    print("    If real-axis erasure still degrades ρ_B more than matched random")
    print("    erasure, the v_A direction carries probability content used by the")
    print("    v_B linear decoder — i.e., the axis is functional, not just a")
    print("    geometric correlate of lexical similarity. Pairs that cross the")
    print("    functional threshold against the matched-random null are the")
    print("    embedding-class analogue of causal-intervention evidence in")
    print("    decoder-LLM interpretability.")
    print()


if __name__ == "__main__":
    main()
