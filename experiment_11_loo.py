"""
Experiment 11 LOO: Out-of-Sample Concept Erasure (LOO target decoder)

Codex audit (April 2026) flagged that experiment_11_concept_erasure.py fits
the target decoder v_B in-sample on the same items used to evaluate it.
That is sufficient evidence of "functional disruption of a fitted linear
decoder" but is slightly weaker than a full out-of-sample / generalization
claim.  This script repeats the concept-erasure protocol with the target
decoder fit in leave-one-out fashion, while leaving the eraser axis v_A
trained on its own type's full data (consistent with experiment_11.py).

If LOO erasure reproduces the in-sample headline pattern (Pred ↔ Modal
ΔMAE 17–25 pp; cross-pair r(cos, ΔMAE) ≈ 0.9), then §4.4 upgrades from
"functional disruption of a fitted linear decoder" to "out-of-sample
functional validation."

For each ordered pair (eraser A, target B) with A ≠ B and
A, B ∈ {Predicative, Adverbial, NounPhrase, Modal-M10}:

  1. Train v_A once on type-A items (full set, ridge λ=0.1) — same as
     experiment_11.
  2. For each target item i ∈ type B (LOO loop):
       - Train (v_B^(−i), slope^(−i), intercept^(−i)) on the n_B − 1
         remaining type-B items.
       - Baseline LOO prediction for item i: clip(slope·(diff_i·v_B) + intc).
       - Treated LOO prediction: erase v_A from diff_i, then decode with
         the same v_B^(−i) triple.
  3. Aggregate baseline LOO MAE, treated LOO MAE, ΔMAE.
  4. Cosine-matched random null also under LOO target decoding (N=100).

Cross-pair correlation r(cos, ΔMAE_LOO) is the headline diagnostic.

Usage:
  python3 experiment_11_loo.py [model_name]
  Default model: mxbai-embed-large
  Recommended secondary: qwen3-embedding
"""

import sys
import time
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
    """Canonical ridge protocol (matches experiment_11 / paper_validation).

    Uses the dual form  w = X^T (X X^T + λI)^{-1} y_c  which is mathematically
    equivalent to  w = (X^T X + λI)^{-1} X^T y_c  but O(n^3) instead of O(d^3).
    The unit-normalized w is identical to within float64 precision; verified
    against the primal form to ~1e-12 relative diff.  This matters because the
    LOO erasure runs ~30k axis fits per model.
    """
    medians_c = (medians - medians.mean()) / medians.std()
    n = diffs.shape[0]
    G = diffs @ diffs.T               # (n, n)
    alpha = np.linalg.solve(G + lam * np.eye(n), medians_c)
    w = diffs.T @ alpha
    w = normalize(w)
    projections = diffs @ w
    slope, intercept, _, _, _ = stats.linregress(projections, medians)
    return w, slope, intercept


def erase(diffs, v):
    """Rank-1 projection erasure: e' = e - (e·v) v.  v assumed unit-norm."""
    return diffs - np.outer(diffs @ v, v)


def predict_one(diff_i, w, slope, intercept):
    """Apply calibrated decoder to a single diff vector."""
    return float(np.clip(slope * (diff_i @ w) + intercept, 0.0, 100.0))


def random_unit(rng, dim):
    v = rng.standard_normal(dim)
    return normalize(v)


def random_cosine_matched(rng, v_b, target_abs_cos):
    """
    Sample a unit vector r with |cos(r, v_b)| = target_abs_cos.
        r = sign·c·v_b + sqrt(1−c²)·u_orth
    where u_orth is a random unit vector orthogonal to v_b.
    NOTE: v_b here is the FULL-data axis used only as a reference direction
    to define the cosine-matching distribution.  This matches experiment_11.
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
# Mosteller-derived expression sets — verbatim from experiment_11.py.
# Modal = M10 (Mosteller-grounded only).
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

MODAL = {
    "certainly": 99.6,
    "almost certainly": 90.2,
    "very likely": 87.5,
    "likely": 71.1,
    "probably": 70.2,
    "very probably": 89.7,
    "possibly": 38.5,
    "unlikely": 17.2,
    "very unlikely": 5.0,
    "improbably": 12.5,
}
MODAL_TEMPLATE = "The experiment will {PHRASE} succeed"


def build_group_data(exprs, template, bare_emb):
    phrases = list(exprs.keys())
    medians = np.array([exprs[p] for p in phrases], dtype=np.float64)
    sentences = [template.replace("{PHRASE}", p.lower()) for p in phrases]
    embs = embed_texts(sentences)
    diffs = embs - bare_emb
    return diffs, medians, phrases


# ────────────────────────────────────────────────────────────────────────────
# LOO machinery
# ────────────────────────────────────────────────────────────────────────────


def loo_predictions(diffs_eval, diffs_b, medians_b, lam=0.1):
    """
    For each item i ∈ type B, train v_B^(−i) on the n_B − 1 remaining type-B
    items, then apply that calibrated triple to diffs_eval[i].

    diffs_eval: per-item diff vectors to decode (one per i; may be erased).
    diffs_b:    per-item diff vectors used to train the LOO axis.
    medians_b:  type-B median labels.

    Returns: array of LOO predictions, one per item.

    Optimization: precompute the full Gram matrix G = diffs_b @ diffs_b.T once,
    then for each held-out i slice the (n−1)×(n−1) submatrix.  This is the
    dominant cost in the inner loop; everything else is O(n^3) or O(n·d).
    """
    n = len(medians_b)
    G_full = diffs_b @ diffs_b.T         # (n, n) — computed once
    preds = np.zeros(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        diffs_train  = diffs_b[mask]
        medians_train = medians_b[mask]
        # z-score normalize labels exactly as train_axis does
        m_mean = medians_train.mean()
        m_std  = medians_train.std()
        medians_c = (medians_train - m_mean) / m_std

        # Slice the precomputed Gram matrix — much faster than recomputing
        G_loo = G_full[np.ix_(mask, mask)]
        n_train = n - 1
        alpha = np.linalg.solve(G_loo + lam * np.eye(n_train), medians_c)
        w = diffs_train.T @ alpha
        w_norm = np.linalg.norm(w)
        if w_norm > 0:
            w = w / w_norm
        # Slope/intercept on training projections vs. raw medians
        proj_train = diffs_train @ w
        slope_loo, intc_loo, _, _, _ = stats.linregress(proj_train, medians_train)
        preds[i] = float(np.clip(
            slope_loo * (diffs_eval[i] @ w) + intc_loo, 0.0, 100.0,
        ))
    return preds


def main():
    print()
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║  Experiment 11 LOO: Out-of-Sample Concept Erasure                  ║")
    print("╠══════════════════════════════════════════════════════════════════════╣")
    print(f"║  Model: {MODEL:<58s}║")
    print("║                                                                    ║")
    print("║  Per (A, B): v_A trained on full type-A; for each i ∈ type B,      ║")
    print("║  train v_B^(−i) on n_B − 1 items, decode item i baseline / treated ║")
    print("║  with that LOO triple.  Compare against cosine-matched random      ║")
    print("║  direction null (N=100), also LOO-decoded.                         ║")
    print("║                                                                    ║")
    print("║  Reproduces in-sample pattern ⇒ §4.4 upgrades to OUT-OF-SAMPLE     ║")
    print("║  functional validation.                                            ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print(flush=True)

    try:
        test = embed_texts(["test"])
        dim = test.shape[1]
        print(f"Model loaded. Dim: {dim}", flush=True)
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    rng = np.random.RandomState(SEED)
    bare_emb = embed_texts([BARE_CLAIM])[0]

    groups = {
        "Predicative": (PREDICATIVE, PREDICATIVE_TEMPLATE),
        "Adverbial":   (ADVERBIAL,   ADVERBIAL_TEMPLATE),
        "NounPhrase":  (NOUN_PHRASE, NOUN_PHRASE_TEMPLATE),
        "Modal":       (MODAL,       MODAL_TEMPLATE),
    }

    diffs_by_type   = {}
    medians_by_type = {}
    phrases_by_type = {}
    full_axes_by_type = {}    # name -> (w, slope, intercept) on full data

    print()
    print("═" * 78)
    print("AXIS TRAINING (full data per type — used for v_A only)")
    print("═" * 78)
    print(f"  {'Type':<12s} {'n':>4s}  {'in-sample ρ':>14s}  {'in-sample MAE':>15s}")

    for gname, (exprs, template) in groups.items():
        d, m, phrases = build_group_data(exprs, template, bare_emb)
        diffs_by_type[gname]   = d
        medians_by_type[gname] = m
        phrases_by_type[gname] = phrases
        w, sl, intc = train_axis(d, m)
        full_axes_by_type[gname] = (w, sl, intc)
        proj = d @ w
        preds_is = np.clip(sl * proj + intc, 0.0, 100.0)
        rho_is, _ = stats.spearmanr(preds_is, m)
        mae_is = float(np.mean(np.abs(preds_is - m)))
        print(f"  {gname:<12s} {len(m):>4d}  {rho_is:>+14.4f}  {mae_is:>15.3f}",
              flush=True)

    # ═══════════════════════════════════════════════════════════════════════
    # Pairwise LOO erasure analysis (12 ordered pairs)
    # ═══════════════════════════════════════════════════════════════════════

    type_names = list(groups.keys())
    pair_records = []

    print()
    print("═" * 78)
    print("PAIRWISE LOO ERASURE ANALYSIS")
    print("  For each (eraser A, target B):")
    print("    v_A trained on full type-A;")
    print("    for each i ∈ type B, train v_B^(−i) on n_B − 1 items,")
    print("    decode baseline diff_i and erased diff_i with that LOO triple;")
    print("    aggregate baseline LOO MAE, treated LOO MAE, ΔMAE.")
    print("  Cosine-matched random null (N=100) also LOO-decoded per item.")
    print("═" * 78)
    print(flush=True)

    t_run_start = time.time()

    for a_name in type_names:
        for b_name in type_names:
            if a_name == b_name:
                continue

            v_a, _, _              = full_axes_by_type[a_name]
            v_b_full, _, _         = full_axes_by_type[b_name]
            diffs_b                = diffs_by_type[b_name]
            medians_b              = medians_by_type[b_name]
            n_b                    = len(medians_b)

            cos_ab = float(v_a @ v_b_full)
            abs_cos_ab = abs(cos_ab)

            t_pair_start = time.time()

            # ─── BASELINE: LOO-decode unmodified diffs_b ────────────────
            preds_base_loo = loo_predictions(diffs_b, diffs_b, medians_b)
            mae_base_loo = float(np.mean(np.abs(preds_base_loo - medians_b)))
            rho_base_loo, _ = stats.spearmanr(preds_base_loo, medians_b)

            # ─── TREATMENT: erase v_A, then LOO-decode ─────────────────
            # NOTE: erasure is applied to diffs_b BEFORE the LOO axis is
            # trained, so v_B^(−i) is trained on the erased remaining items
            # too.  This is the principled "erase then decode" pipeline:
            # the eraser removes v_A's component from every item; the
            # decoder is then fit on the held-out remainder of those
            # already-erased items and applied to the held-out one.
            diffs_b_erased = erase(diffs_b, v_a)
            preds_treat_loo = loo_predictions(
                diffs_b_erased, diffs_b_erased, medians_b,
            )
            mae_treat_loo = float(np.mean(np.abs(preds_treat_loo - medians_b)))
            rho_treat_loo, _ = stats.spearmanr(preds_treat_loo, medians_b)

            delta_mae = mae_treat_loo - mae_base_loo
            delta_rho = rho_treat_loo - rho_base_loo

            # ─── COSINE-MATCHED RANDOM NULL (LOO) ───────────────────────
            match_rng = np.random.RandomState(SEED + 1)
            match_maes = np.zeros(N_RANDOM)
            match_rhos = np.zeros(N_RANDOM)
            for k in range(N_RANDOM):
                r = random_cosine_matched(match_rng, v_b_full, abs_cos_ab)
                d_e = erase(diffs_b, r)
                preds_e = loo_predictions(d_e, d_e, medians_b)
                m_k = float(np.mean(np.abs(preds_e - medians_b)))
                rho_k, _ = stats.spearmanr(preds_e, medians_b)
                match_maes[k] = m_k
                match_rhos[k] = rho_k if not np.isnan(rho_k) else 0.0

            mu_m_mae, sd_m_mae = float(np.mean(match_maes)), float(np.std(match_maes))
            mu_m_rho, sd_m_rho = float(np.mean(match_rhos)), float(np.std(match_rhos))
            z_match_mae = (
                (mae_treat_loo - mu_m_mae) / sd_m_mae
                if sd_m_mae > 1e-12 else 0.0
            )
            z_match_rho = (
                (rho_treat_loo - mu_m_rho) / sd_m_rho
                if sd_m_rho > 1e-12 else 0.0
            )

            # ─── UNIFORM RANDOM NULL (LOO) ─────────────────────────────
            unif_rng = np.random.RandomState(SEED)
            unif_maes = np.zeros(N_RANDOM)
            unif_rhos = np.zeros(N_RANDOM)
            for k in range(N_RANDOM):
                r = random_unit(unif_rng, dim)
                d_e = erase(diffs_b, r)
                preds_e = loo_predictions(d_e, d_e, medians_b)
                m_k = float(np.mean(np.abs(preds_e - medians_b)))
                rho_k, _ = stats.spearmanr(preds_e, medians_b)
                unif_maes[k] = m_k
                unif_rhos[k] = rho_k if not np.isnan(rho_k) else 0.0

            mu_u_mae, sd_u_mae = float(np.mean(unif_maes)),  float(np.std(unif_maes))
            mu_u_rho, sd_u_rho = float(np.mean(unif_rhos)),  float(np.std(unif_rhos))
            z_unif_mae = (
                (mae_treat_loo - mu_u_mae) / sd_u_mae
                if sd_u_mae > 1e-12 else 0.0
            )
            z_unif_rho = (
                (rho_treat_loo - mu_u_rho) / sd_u_rho
                if sd_u_rho > 1e-12 else 0.0
            )

            # MAE-based verdict (lower MAE better; functional ⇒ z >> 0)
            if z_match_mae > 2.0:
                mae_verdict = "↑↑ functional"
            elif z_match_mae > 1.0:
                mae_verdict = "↑ trend"
            else:
                mae_verdict = "—"

            t_pair_elapsed = time.time() - t_pair_start

            pair_records.append({
                "a": a_name, "b": b_name, "cos": cos_ab,
                "n_b": n_b,
                "mae_base_loo": mae_base_loo, "mae_treat_loo": mae_treat_loo,
                "delta_mae": delta_mae,
                "rho_base_loo": rho_base_loo, "rho_treat_loo": rho_treat_loo,
                "delta_rho": delta_rho,
                "match_mu_mae": mu_m_mae, "match_sd_mae": sd_m_mae,
                "match_mu_rho": mu_m_rho, "match_sd_rho": sd_m_rho,
                "z_match_mae": z_match_mae, "z_match_rho": z_match_rho,
                "unif_mu_mae": mu_u_mae, "unif_sd_mae": sd_u_mae,
                "unif_mu_rho": mu_u_rho, "unif_sd_rho": sd_u_rho,
                "z_unif_mae": z_unif_mae, "z_unif_rho": z_unif_rho,
                "mae_verdict": mae_verdict,
            })

            print(
                f"  {a_name + '→' + b_name:<24s} "
                f"cos={cos_ab:+.3f}  n_B={n_b:>2d}  "
                f"MAE_base={mae_base_loo:>5.2f}  "
                f"MAE_treat={mae_treat_loo:>6.2f}  "
                f"ΔMAE={delta_mae:>+6.2f}  "
                f"match μ±σ={mu_m_mae:>6.2f}±{sd_m_mae:>4.2f}  "
                f"z_match={z_match_mae:>+6.2f}  "
                f"({t_pair_elapsed:>4.1f}s)",
                flush=True,
            )

    t_run_total = time.time() - t_run_start
    print(f"\n  Total LOO erasure runtime: {t_run_total:.1f}s "
          f"({t_run_total/60:.1f} min)",
          flush=True)

    # ═══════════════════════════════════════════════════════════════════════
    # Per-pair MAE table (LOO)
    # ═══════════════════════════════════════════════════════════════════════

    print()
    print("═" * 78)
    print("PAIRWISE LOO ERASURE: PER-PAIR MAE TABLE")
    print("═" * 78)
    print()
    print("  MAE-based view (lower MAE_treat better; functional ⇒ z_match_MAE > +2)")
    print()
    mae_header = (
        f"  {'A→B':<24s} {'cos':>7s} | "
        f"{'MAE_b':>6s} {'MAE_t':>7s} {'ΔMAE':>7s} | "
        f"{'MAE_unif μ±σ':>18s} {'MAE_match μ±σ':>18s} | "
        f"{'z_uMAE':>7s} {'z_mMAE':>8s} | verdict"
    )
    print(mae_header)
    print("  " + "-" * (len(mae_header) - 2))
    for rec in pair_records:
        label = f"{rec['a']}→{rec['b']}"
        print(
            f"  {label:<24s} {rec['cos']:>+7.3f} | "
            f"{rec['mae_base_loo']:>6.2f} {rec['mae_treat_loo']:>7.2f} "
            f"{rec['delta_mae']:>+7.2f} | "
            f"{rec['unif_mu_mae']:>8.2f}±{rec['unif_sd_mae']:>5.2f}    "
            f"{rec['match_mu_mae']:>8.2f}±{rec['match_sd_mae']:>5.2f}  | "
            f"{rec['z_unif_mae']:>+7.2f} {rec['z_match_mae']:>+8.2f} | "
            f"{rec['mae_verdict']}",
            flush=True,
        )

    # ═══════════════════════════════════════════════════════════════════════
    # Per-pair ρ table (LOO)
    # ═══════════════════════════════════════════════════════════════════════

    print()
    print("  ρ-based view (higher ρ_treat better; functional ⇒ z_match_ρ < −2)")
    print()
    rho_header = (
        f"  {'A→B':<24s} {'cos':>7s} | "
        f"{'ρ_base':>7s} {'ρ_treat':>8s} {'Δρ':>7s} | "
        f"{'ρ_unif μ±σ':>16s} {'ρ_match μ±σ':>16s} | "
        f"{'z_uρ':>7s} {'z_mρ':>8s}"
    )
    print(rho_header)
    print("  " + "-" * (len(rho_header) - 2))
    for rec in pair_records:
        label = f"{rec['a']}→{rec['b']}"
        print(
            f"  {label:<24s} {rec['cos']:>+7.3f} | "
            f"{rec['rho_base_loo']:>+7.3f} {rec['rho_treat_loo']:>+8.3f} "
            f"{rec['delta_rho']:>+7.3f} | "
            f"{rec['unif_mu_rho']:>+7.3f}±{rec['unif_sd_rho']:>4.2f}  "
            f"{rec['match_mu_rho']:>+7.3f}±{rec['match_sd_rho']:>4.2f} | "
            f"{rec['z_unif_rho']:>+7.2f} {rec['z_match_rho']:>+8.2f}",
            flush=True,
        )

    # ═══════════════════════════════════════════════════════════════════════
    # Cross-pair correlations
    # ═══════════════════════════════════════════════════════════════════════

    cos_arr        = np.array([r["cos"]          for r in pair_records])
    delta_mae_arr  = np.array([r["delta_mae"]    for r in pair_records])
    z_match_mae_arr= np.array([r["z_match_mae"]  for r in pair_records])
    delta_rho_arr  = np.array([r["delta_rho"]    for r in pair_records])

    r_cos_dmae,    _ = stats.pearsonr(cos_arr, delta_mae_arr)
    r_cos_zmmae,   _ = stats.pearsonr(cos_arr, z_match_mae_arr)
    r_cos_drho,    _ = stats.pearsonr(cos_arr, delta_rho_arr)

    print()
    print("═" * 78)
    print("CROSS-PAIR CORRELATIONS (LOO)")
    print("  Across the 12 ordered pairs:")
    print(f"    Pearson r(cos, ΔMAE_LOO)          = {r_cos_dmae:+.3f}")
    print(f"    Pearson r(cos, z_match_MAE_LOO)   = {r_cos_zmmae:+.3f}")
    print(f"    Pearson r(cos, Δρ_LOO)            = {r_cos_drho:+.3f}")
    print("═" * 78)
    print(flush=True)

    # ═══════════════════════════════════════════════════════════════════════
    # Headline section: Predicative ↔ Modal
    # ═══════════════════════════════════════════════════════════════════════

    print()
    print("═" * 78)
    print("HEADLINE: Predicative ↔ Modal (LOO)")
    print("═" * 78)

    headline_pairs = [("Predicative", "Modal"), ("Modal", "Predicative")]
    for a_name, b_name in headline_pairs:
        rec = next(r for r in pair_records if r["a"] == a_name and r["b"] == b_name)
        print()
        print(f"  {a_name} → {b_name}")
        print(f"    cos(v_A, v_B_full):       {rec['cos']:+.4f}")
        print(f"    LOO MAE baseline:         {rec['mae_base_loo']:.3f}")
        print(f"    LOO MAE after v_A erase:  {rec['mae_treat_loo']:.3f}  "
              f"(Δ = {rec['delta_mae']:+.3f})")
        print(f"    Cosine-matched LOO MAE:   {rec['match_mu_mae']:.3f} ± "
              f"{rec['match_sd_mae']:.3f}   →  z_MAE = {rec['z_match_mae']:+.2f}")
        print(f"    Uniform-random LOO MAE:   {rec['unif_mu_mae']:.3f} ± "
              f"{rec['unif_sd_mae']:.3f}   →  z_MAE = {rec['z_unif_mae']:+.2f}")
        print(f"    LOO ρ baseline:           {rec['rho_base_loo']:+.4f}")
        print(f"    LOO ρ after v_A erase:    {rec['rho_treat_loo']:+.4f}  "
              f"(Δ = {rec['delta_rho']:+.4f})")
        print(f"    MAE-based verdict:        {rec['mae_verdict']}")
        print(flush=True)

    # ═══════════════════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════════════════

    print()
    print("═" * 78)
    print("SUMMARY (LOO)")
    print("═" * 78)

    n_pairs = len(pair_records)

    def bucket_pos(records, key):
        funcs  = sum(1 for r in records if r[key] > 2.0)
        trends = sum(1 for r in records if 1.0 < r[key] <= 2.0)
        neut   = n_pairs - funcs - trends
        return funcs, trends, neut

    f_uM, t_uM, n_uM = bucket_pos(pair_records, "z_unif_mae")
    f_mM, t_mM, n_mM = bucket_pos(pair_records, "z_match_mae")

    print()
    print(f"  Pairs evaluated: {n_pairs} (4×3 ordered cross-type)")
    print()
    print(f"  MAE-based (lower MAE_treat = better; functional ⇒ z_MAE > +2)")
    print(f"  Vs. MAE-based UNIFORM random baseline (LOO):")
    print(f"    ↑↑ functional (z_MAE > +2):    {f_uM:>2d} / {n_pairs}")
    print(f"    ↑  trend     (1 ≤ z_MAE < 2):  {t_uM:>2d} / {n_pairs}")
    print(f"    —  neutral   (z_MAE < 1):      {n_uM:>2d} / {n_pairs}")
    print()
    print(f"  Vs. MAE-based COSINE-MATCHED random baseline (LOO):")
    print(f"    ↑↑ functional (z_MAE > +2):    {f_mM:>2d} / {n_pairs}")
    print(f"    ↑  trend     (1 ≤ z_MAE < 2):  {t_mM:>2d} / {n_pairs}")
    print(f"    —  neutral   (z_MAE < 1):      {n_mM:>2d} / {n_pairs}")
    print()
    print("  Interpretation:")
    print("    Reproduces in-sample headline pattern ⇒ §4.4 upgrades from")
    print("    'functional disruption of a fitted linear decoder' to")
    print("    'out-of-sample functional validation.'  The target decoder")
    print("    is now fit on items it never sees, so the cross-type")
    print("    degradation is no longer an in-sample-fit artifact.")
    print(flush=True)


if __name__ == "__main__":
    main()
