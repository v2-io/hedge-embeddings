"""
Experiment 11b: Label-Permutation Null for Concept Erasure (Functional vs.
Geometric — complementary null to experiment_11.py).

Experiment 11's matched-random null (cosine-matched random direction) tests
whether a *direction* with the same geometric overlap with v_B as v_A is
functionally equivalent to v_A.  This is a geometry-controlled null but does
not directly test whether the *probability content* aligned with v_A is
doing the work, because the matched-random direction is, by construction,
not trained on probability labels at all.

This complementary null shuffles the Mosteller medians among the
expressions of type A, retrains v_A on the shuffled labels (same templates,
same diff-vector construction, same ridge regression with λ=0.1), and
erases that v_A_shuffled from the type-B difference vectors.  The target
axis v_B and its calibrated triple (slope_B, intercept_B) stay REAL — only
the eraser changes.  This isolates "is the probability content of v_A
doing work?" from "is the trained-axis-on-this-template-set geometry doing
work?": v_A_shuffled lives in the same template subspace, was trained on
the same diff vectors with the same regularizer, and lacks only the
probability-aligned label structure.

For each ordered pair (A, B) with A ≠ B and A, B ∈ {Predicative,
Adverbial, NounPhrase, Modal}, we compute ΔMAE under K label
permutations of A's labels and report:
  * real ΔMAE (read off the existing exp11 results at runtime if present,
    else recomputed once for the run)
  * shuffled-null mean ± std
  * percentile of the real ΔMAE in the shuffled-null distribution
  * empirical p-value = (1 + #{shuffled ΔMAE ≥ real ΔMAE}) / (K + 1)

Methodological notes:
  * Only the eraser axis (v_A) is retrained on shuffled labels.  v_B and
    its calibrated triple are real, fit once before the permutation loop.
  * Diff vectors are computed once and reused across permutations: only
    the labels feeding into the ridge solve change, so embeddings are
    stable across permutations.
  * The training protocol (template, ridge λ=0.1, normalize w to unit norm)
    is byte-identical to experiment_11.py's train_axis.
  * The empirical p-value adds the +1/+1 correction so it never returns
    exactly 0; a real ΔMAE strictly larger than every shuffled draw
    yields p = 1/(K+1).

Usage:
  python3 experiment_11b_label_permutation.py [model_name] [K]
  Default model: mxbai-embed-large
  Default K:     500   (drops to 200 if you pass it explicitly)
"""

import sys
import time
import numpy as np
import requests
from scipy import stats

MODEL = sys.argv[1] if len(sys.argv) > 1 else "mxbai-embed-large"
K_PERMS = int(sys.argv[2]) if len(sys.argv) > 2 else 500
OLLAMA_URL = "http://localhost:11434/api/embed"

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


def erase(diffs, v):
    """Rank-1 projection erasure: e' = e - (e·v) v.  v assumed unit-norm."""
    return diffs - np.outer(diffs @ v, v)


def predict(diffs, w, slope, intercept):
    """Apply calibrated axis decoder to diffs. Returns clipped [0,100] preds."""
    return np.clip(slope * (diffs @ w) + intercept, 0.0, 100.0)


def mae(preds, medians):
    return float(np.mean(np.abs(preds - medians)))


# ────────────────────────────────────────────────────────────────────────────
# Mosteller-derived expression sets — verbatim from experiment_11.py.
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
    return diffs, medians


# ────────────────────────────────────────────────────────────────────────────
# Existing (matched-random) verdicts and real ΔMAE values, recovered at
# runtime by recomputing rather than parsing exp11_*.txt.  This keeps the
# script independent of the formatted text output.
# ────────────────────────────────────────────────────────────────────────────


def matched_random_verdict_label(z_match_mae):
    if z_match_mae > 2.0:
        return "↑↑ functional"
    if z_match_mae > 1.0:
        return "↑ trend"
    return "—"


def main():
    print()
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║  Experiment 11b: Label-Permutation Null for Concept Erasure        ║")
    print("╠══════════════════════════════════════════════════════════════════════╣")
    print(f"║  Model: {MODEL:<58s}║")
    print(f"║  K (permutations per pair): {K_PERMS:<40d}║")
    print("║                                                                    ║")
    print("║  Shuffle Mosteller medians among type-A expressions; retrain v_A   ║")
    print("║  on shuffled labels (same template, same ridge protocol).  Erase   ║")
    print("║  v_A_shuffled from type-B diffs; decode with v_B's REAL calibrated ║")
    print("║  triple.  Compare real ΔMAE against the shuffled-null distribution.║")
    print("║                                                                    ║")
    print("║  Tests: is the *probability content* of v_A doing work — beyond    ║")
    print("║  the geometric/structural properties that survive label shuffling? ║")
    print("║                                                                    ║")
    print("║  Complementary to experiment_11.py's cosine-matched random null —  ║")
    print("║  not a replacement.  The two nulls answer different questions.     ║")
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

    groups = {
        "Predicative": (PREDICATIVE, PREDICATIVE_TEMPLATE),
        "Adverbial":   (ADVERBIAL,   ADVERBIAL_TEMPLATE),
        "NounPhrase":  (NOUN_PHRASE, NOUN_PHRASE_TEMPLATE),
        "Modal":       (MODAL,       MODAL_TEMPLATE),
    }

    diffs_by_type   = {}
    medians_by_type = {}
    real_axes       = {}     # name -> (w, slope, intercept) on REAL labels

    print()
    print("═" * 78)
    print("AXIS TRAINING (real labels, in-sample fit per type)")
    print("═" * 78)
    print(f"  {'Type':<12s} {'n':>4s}  {'in-sample ρ':>14s}  {'in-sample MAE':>15s}")

    for gname, (exprs, template) in groups.items():
        d, m = build_group_data(exprs, template, bare_emb)
        diffs_by_type[gname]   = d
        medians_by_type[gname] = m
        w, sl, intc = train_axis(d, m)
        real_axes[gname] = (w, sl, intc)
        preds = predict(d, w, sl, intc)
        rho_is, _ = stats.spearmanr(preds, m)
        mae_is = mae(preds, m)
        print(f"  {gname:<12s} {len(m):>4d}  {rho_is:>+14.4f}  {mae_is:>15.3f}")

    # ═══════════════════════════════════════════════════════════════════════
    # For each ordered pair (A, B): real ΔMAE, plus matched-random verdict
    # recomputed (cheap N=100 control reproducing experiment_11.py exactly),
    # plus shuffled-null distribution of K permutations.
    # ═══════════════════════════════════════════════════════════════════════

    type_names = list(groups.keys())
    pair_records = []

    print()
    print("═" * 78)
    print("LABEL-PERMUTATION NULL: PER-PAIR PROCESSING")
    print("═" * 78)
    print(f"  {K_PERMS} permutations per pair × 12 pairs.  Eraser v_A retrained on")
    print(f"  shuffled type-A labels each permutation; v_B's calibrated triple is real.")
    print()
    print(f"  Pair                            │ progress")
    print(f"  ────────────────────────────────┼──────────────────")

    t_run_start = time.time()

    for a_name in type_names:
        for b_name in type_names:
            if a_name == b_name:
                continue

            v_a_real, _, _              = real_axes[a_name]
            v_b, slope_b, intc_b        = real_axes[b_name]
            diffs_a                     = diffs_by_type[a_name]
            medians_a                   = medians_by_type[a_name]
            diffs_b                     = diffs_by_type[b_name]
            medians_b                   = medians_by_type[b_name]

            cos_ab = float(v_a_real @ v_b)

            # Real ΔMAE under v_A real eraser, decoded with v_B real triple.
            mae_base = mae(predict(diffs_b, v_b, slope_b, intc_b), medians_b)
            diffs_b_treat_real = erase(diffs_b, v_a_real)
            mae_treat_real = mae(
                predict(diffs_b_treat_real, v_b, slope_b, intc_b),
                medians_b,
            )
            real_delta_mae = mae_treat_real - mae_base

            # Cosine-matched random null (recomputes experiment_11.py's
            # MAE-side z_match_MAE so we can label per-pair verdicts here
            # with the same threshold logic the existing exp11 results use).
            match_rng = np.random.RandomState(SEED + 1)
            match_maes = np.zeros(100)
            abs_cos = abs(cos_ab)
            for k in range(100):
                u = match_rng.standard_normal(dim)
                u_orth = u - (u @ v_b) * v_b
                u_orth = normalize(u_orth)
                sign = 1.0 if match_rng.random() < 0.5 else -1.0
                r = sign * abs_cos * v_b + np.sqrt(max(0.0, 1.0 - abs_cos * abs_cos)) * u_orth
                r = normalize(r)
                d_e = erase(diffs_b, r)
                match_maes[k] = mae(predict(d_e, v_b, slope_b, intc_b), medians_b)
            mu_m_mae = float(np.mean(match_maes))
            sd_m_mae = float(np.std(match_maes))
            z_match_mae = (
                (mae_treat_real - mu_m_mae) / sd_m_mae
                if sd_m_mae > 1e-12 else 0.0
            )
            matched_verdict = matched_random_verdict_label(z_match_mae)

            # Shuffled-label null: shuffle medians_A only, retrain v_A,
            # erase from diffs_b, decode with v_B real triple, compute ΔMAE.
            perm_rng = np.random.RandomState(
                SEED + 100 * type_names.index(a_name) + type_names.index(b_name)
            )

            shuffled_delta_maes = np.zeros(K_PERMS)
            n_a = len(medians_a)
            t_pair_start = time.time()

            for k in range(K_PERMS):
                # Permute the labels among the type-A expressions only.
                perm = perm_rng.permutation(n_a)
                medians_a_shuf = medians_a[perm]

                # Retrain v_A on shuffled labels (same protocol as exp11).
                w_a_shuf, _, _ = train_axis(diffs_a, medians_a_shuf, lam=0.1)

                # Erase v_A_shuf from type-B diffs; decode with v_B real triple.
                diffs_b_treat = erase(diffs_b, w_a_shuf)
                preds_treat   = predict(diffs_b_treat, v_b, slope_b, intc_b)
                shuffled_delta_maes[k] = mae(preds_treat, medians_b) - mae_base

            t_pair_elapsed = time.time() - t_pair_start

            mu_s = float(np.mean(shuffled_delta_maes))
            sd_s = float(np.std(shuffled_delta_maes))
            # Percentile of real ΔMAE within the shuffled distribution
            # (fraction of shuffled draws ≤ real_delta_mae).
            pct_le = float(np.mean(shuffled_delta_maes <= real_delta_mae)) * 100.0
            # Empirical one-sided p-value: how often does shuffled match or
            # exceed real?  Add-one smoothing.
            n_ge = int(np.sum(shuffled_delta_maes >= real_delta_mae))
            p_value = (1 + n_ge) / (K_PERMS + 1)

            # Label-permutation verdict: real ΔMAE meaningfully exceeds the
            # shuffled-null upper tail.  Threshold at p ≤ 0.01 for ↑↑ and
            # 0.01 < p ≤ 0.05 for trend; matches the convention in §4.6.
            if p_value <= 0.01:
                perm_verdict = "↑↑ label-perm functional"
            elif p_value <= 0.05:
                perm_verdict = "↑ label-perm trend"
            else:
                perm_verdict = "—"

            pair_records.append({
                "a": a_name,
                "b": b_name,
                "cos": cos_ab,
                "real_delta_mae": real_delta_mae,
                "shuffled_mu": mu_s,
                "shuffled_sd": sd_s,
                "percentile": pct_le,
                "p_value": p_value,
                "n_ge": n_ge,
                "matched_verdict": matched_verdict,
                "z_match_mae": z_match_mae,
                "perm_verdict": perm_verdict,
                "shuffled_max": float(np.max(shuffled_delta_maes)),
                "shuffled_min": float(np.min(shuffled_delta_maes)),
            })

            print(f"  {a_name + '→' + b_name:<31s} │ "
                  f"K={K_PERMS}  {t_pair_elapsed:5.1f}s  "
                  f"ΔMAE_real={real_delta_mae:+6.2f}  "
                  f"null μ={mu_s:+5.2f}±{sd_s:.2f}  "
                  f"p={p_value:.4f}")

    t_run_total = time.time() - t_run_start
    print(f"\n  Total label-perm runtime: {t_run_total:.1f}s "
          f"({t_run_total/60:.1f} min)")

    # ═══════════════════════════════════════════════════════════════════════
    # Per-pair summary table.
    # ═══════════════════════════════════════════════════════════════════════

    print()
    print("═" * 78)
    print("LABEL-PERMUTATION NULL: PER-PAIR RESULTS")
    print("═" * 78)
    print()
    header = (
        f"  {'A→B':<24s} {'cos':>7s} | "
        f"{'real ΔMAE':>10s} {'null μ±σ':>16s} | "
        f"{'pct':>6s} {'p-val':>7s} | "
        f"{'matched':<14s} {'label-perm':<24s}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for rec in pair_records:
        label = f"{rec['a']}→{rec['b']}"
        null_str = f"{rec['shuffled_mu']:+6.2f}±{rec['shuffled_sd']:.2f}"
        print(
            f"  {label:<24s} {rec['cos']:>+7.3f} | "
            f"{rec['real_delta_mae']:>+10.3f} {null_str:>16s} | "
            f"{rec['percentile']:>5.1f}% {rec['p_value']:>7.4f} | "
            f"{rec['matched_verdict']:<14s} {rec['perm_verdict']:<24s}"
        )

    # ═══════════════════════════════════════════════════════════════════════
    # Cross-null agreement summary.
    # ═══════════════════════════════════════════════════════════════════════

    n_pairs = len(pair_records)

    matched_func = sum(
        1 for r in pair_records
        if r["matched_verdict"] == "↑↑ functional"
    )
    matched_trend = sum(
        1 for r in pair_records
        if r["matched_verdict"] == "↑ trend"
    )
    perm_func = sum(
        1 for r in pair_records
        if r["perm_verdict"].startswith("↑↑")
    )
    perm_trend = sum(
        1 for r in pair_records
        if r["perm_verdict"].startswith("↑ ")
    )

    matched_func_set = {
        (r["a"], r["b"]) for r in pair_records
        if r["matched_verdict"] == "↑↑ functional"
    }
    perm_func_set = {
        (r["a"], r["b"]) for r in pair_records
        if r["perm_verdict"].startswith("↑↑")
    }
    both_func = matched_func_set & perm_func_set
    only_matched = matched_func_set - perm_func_set
    only_perm = perm_func_set - matched_func_set
    neither = (
        {(r["a"], r["b"]) for r in pair_records}
        - matched_func_set - perm_func_set
    )

    print()
    print("═" * 78)
    print("CROSS-NULL AGREEMENT")
    print("═" * 78)
    print()
    print(f"  Pairs evaluated: {n_pairs}")
    print()
    print(f"  Matched-random null (MAE-side, exp11):")
    print(f"    ↑↑ functional:        {matched_func:>2d} / {n_pairs}")
    print(f"    ↑ trend:              {matched_trend:>2d} / {n_pairs}")
    print()
    print(f"  Label-permutation null (this experiment):")
    print(f"    ↑↑ functional (p≤.01):  {perm_func:>2d} / {n_pairs}")
    print(f"    ↑ trend (p≤.05):        {perm_trend:>2d} / {n_pairs}")
    print()
    print(f"  Cross-null per-pair agreement:")
    print(f"    Functional under both nulls:    {len(both_func):>2d} / {n_pairs}")
    print(f"    Functional only under matched:  {len(only_matched):>2d} / {n_pairs}")
    print(f"    Functional only under label-pm: {len(only_perm):>2d} / {n_pairs}")
    print(f"    Functional under neither:       {len(neither):>2d} / {n_pairs}")
    print()
    if both_func:
        print(f"  Pairs functional under BOTH nulls:")
        for a, b in sorted(both_func):
            print(f"    {a} → {b}")
    if only_matched:
        print()
        print(f"  Pairs functional only under matched-random:")
        for a, b in sorted(only_matched):
            print(f"    {a} → {b}")
    if only_perm:
        print()
        print(f"  Pairs functional only under label-permutation:")
        for a, b in sorted(only_perm):
            print(f"    {a} → {b}")
    print()
    print("  Headline pair (predicative ↔ modal, both directions):")
    for a, b in [("Predicative", "Modal"), ("Modal", "Predicative")]:
        rec = next(r for r in pair_records if r["a"] == a and r["b"] == b)
        print(f"    {a:<12s} → {b:<12s}  "
              f"cos={rec['cos']:+.3f}  "
              f"real ΔMAE={rec['real_delta_mae']:+6.2f}  "
              f"null μ±σ={rec['shuffled_mu']:+5.2f}±{rec['shuffled_sd']:.2f}  "
              f"p={rec['p_value']:.4f}")
    print()
    print("  Interpretation:")
    print("    The label-permutation null asks whether the *probability content*")
    print("    of v_A is the load-bearing input to the cross-type degradation,")
    print("    with the template subspace, regularizer, and diff-vector geometry")
    print("    held fixed.  The matched-random null in experiment_11.py asks")
    print("    whether the *direction* of v_A is more damaging than a random")
    print("    direction with the same geometric overlap with v_B.  The two")
    print("    nulls answer different questions; pairs functional under BOTH")
    print("    are the strongest functional-validation cases.")
    print()


if __name__ == "__main__":
    main()
