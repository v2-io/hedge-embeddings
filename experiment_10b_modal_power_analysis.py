"""
Experiment 10b: Modal Permutation-Test Power Analysis

Operationalizes the §6 L7 small-sample saturation argument as a quantitative
sample-size sweep. Reruns the §4.6 modal permutation test (1000 label
shuffles, retrain ridge axis per shuffle, LOO ρ on permuted labels) at
n ∈ {10, 12, 14, 17}, with the modal expression set grown by appending
seven *author-estimated* modal terms drawn from earlier exploratory
scripts (experiment_03_modal_axis.py, experiment_05_ensemble.py,
experiment_06_compound_hedges.py).

╔═══════════════════════════════════════════════════════════════════════╗
║  IMPORTANT METHODOLOGICAL CAVEAT                                       ║
║                                                                        ║
║  The seven appended modal terms ("conceivably", "definitely",          ║
║  "perhaps", "maybe", "presumably", "undoubtedly", "arguably") carry    ║
║  AUTHOR-ESTIMATED probability values, NOT Mosteller-calibrated         ║
║  values. They are explicitly excluded from the load-bearing modal      ║
║  axis training in the rest of the paper (§3.1, §6 L8) precisely        ║
║  because they are not psychometrically grounded.                       ║
║                                                                        ║
║  Their use here is appropriate ONLY because this experiment makes a    ║
║  *structural* claim — that small-n saturates rank-based metrics — not  ║
║  a calibration claim. The expanded set lets us walk the permutation    ║
║  test out of the small-sample saturation regime to verify the L7       ║
║  prediction; it does NOT calibrate a wider modal axis.                 ║
║                                                                        ║
║  Do NOT cite the n>10 results here as calibration evidence.            ║
╚═══════════════════════════════════════════════════════════════════════╝

Pipeline matches experiment_10_null_hypothesis.py:run_permutation_test
(same train_axis, same loo_rho, same 1000-shuffle loop, same RNG seed=42).

Usage:
  python3 experiment_10b_modal_power_analysis.py [model_name]
  Default model: nomic-embed-text:v1.5
"""

import sys
import numpy as np
import requests
from scipy import stats

MODEL = sys.argv[1] if len(sys.argv) > 1 else "nomic-embed-text:v1.5"
OLLAMA_URL = "http://localhost:11434/api/embed"

N_PERMS = 1000
SEED = 42
SAMPLE_SIZES = [10, 12, 14, 17]


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
    """Leave-one-out cross-validated Spearman ρ. Identical to experiment_10."""
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


def run_permutation_test(diffs, medians, n_perms=N_PERMS, seed=SEED):
    """Identical to experiment_10's run_permutation_test."""
    rng = np.random.RandomState(seed)
    real_rho = loo_rho(diffs, medians)
    perm_rhos = np.zeros(n_perms)
    for i in range(n_perms):
        shuffled = rng.permutation(medians)
        perm_rhos[i] = loo_rho(diffs, shuffled)
    p_value = np.mean(np.abs(perm_rhos) >= np.abs(real_rho))
    percentile = np.mean(np.abs(perm_rhos) < np.abs(real_rho)) * 100
    return real_rho, perm_rhos, p_value, percentile


# ──────────────────────────────────────────────────────────────────────────
# Modal expression sets
# ──────────────────────────────────────────────────────────────────────────

BARE_CLAIM = "The experiment will succeed"
MODAL_TEMPLATE = "The experiment will {PHRASE} succeed"

# 10-item Mosteller-grounded modal set (identical to experiment_10,
# experiment_11, paper_validation, and §3.1's load-bearing modal axis).
MODAL_MOSTELLER = [
    ("certainly",         99.6),
    ("almost certainly",  90.2),
    ("very likely",       87.5),
    ("likely",            71.1),
    ("probably",          70.2),
    ("very probably",     89.7),
    ("possibly",          38.5),
    ("unlikely",          17.2),
    ("very unlikely",      5.0),
    ("improbably",        12.5),
]

# Seven author-estimated modal terms. Values copied verbatim from
# experiment_03_modal_axis.py (and confirmed identical in
# experiment_05_ensemble.py and experiment_06_compound_hedges.py).
# THESE ARE NOT MOSTELLER-CALIBRATED. They are used here ONLY for
# sample-size variation in a structural power analysis.
MODAL_AUTHOR_ESTIMATED = [
    ("conceivably",  38.5),  # ≈ "Possible" (author estimate, not Mosteller)
    ("definitely",   99.6),  # ≈ "Certain"  (author estimate, not Mosteller)
    ("perhaps",      38.5),  # ≈ "Possible" (author estimate, not Mosteller)
    ("maybe",        38.5),  # ≈ "Possible" (author estimate, not Mosteller)
    ("presumably",   70.2),  # ≈ "Probable" (author estimate, not Mosteller)
    ("undoubtedly",  95.0),  # high certainty (author estimate, not Mosteller)
    ("arguably",     55.0),  # moderate (author estimate, not Mosteller)
]

# Maximum reachable n: 10 + 7 = 17.
MAX_N = len(MODAL_MOSTELLER) + len(MODAL_AUTHOR_ESTIMATED)


def build_modal_set(n, rng):
    """Return phrases, medians for first n items.

    n=10 returns the Mosteller set untouched.
    n>10 appends the first (n-10) author-estimated terms in the
    deterministic order they appear in MODAL_AUTHOR_ESTIMATED.

    A note on determinism: the task spec asks for "sampled deterministically
    (seed 42)". The MODAL_AUTHOR_ESTIMATED list is small (7 items) and the
    inclusion order is itself a confound when only some are added, so we
    use a deterministic seeded permutation of the 7 items and take the
    first (n-10). Seed 42 is shared with the outer permutation test, but
    is consumed here once before the permutation loop.
    """
    if n < 10 or n > MAX_N:
        raise ValueError(f"n must be in [10, {MAX_N}], got {n}")
    base = list(MODAL_MOSTELLER)
    extra_pool = list(MODAL_AUTHOR_ESTIMATED)
    if n > 10:
        order = rng.permutation(len(extra_pool))
        extras = [extra_pool[i] for i in order[: n - 10]]
        base = base + extras
    phrases = [p for p, _ in base]
    medians = np.array([m for _, m in base], dtype=np.float64)
    return phrases, medians


def main():
    print()
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║  Experiment 10b: Modal Permutation-Test Power Analysis             ║")
    print("╠══════════════════════════════════════════════════════════════════════╣")
    print(f"║  Model: {MODEL:<58s}║")
    print("║                                                                    ║")
    print("║  Operationalizes §6 L7: walks the modal permutation test out of    ║")
    print("║  the small-sample rank-saturation regime by appending up to 7      ║")
    print("║  author-estimated modal terms to the 10-item Mosteller set.        ║")
    print("║                                                                    ║")
    print("║  CAVEAT: author-estimated values are NOT Mosteller-calibrated.     ║")
    print("║  Used here ONLY for sample-size variation in a structural power    ║")
    print("║  analysis — NOT for any calibration claim. See §6 L8.              ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print()

    try:
        test = embed_texts(["test"])
        dim = test.shape[1]
        print(f"Model loaded. Dim: {dim}")
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    print()
    print("Modal expression composition:")
    print(f"  Mosteller-grounded (n={len(MODAL_MOSTELLER)}):")
    for p, m in MODAL_MOSTELLER:
        print(f"    {p:<22s}  {m:5.1f}  [Mosteller-calibrated]")
    print(f"  Author-estimated pool (n={len(MODAL_AUTHOR_ESTIMATED)}):")
    for p, m in MODAL_AUTHOR_ESTIMATED:
        print(f"    {p:<22s}  {m:5.1f}  [AUTHOR-ESTIMATED — NOT Mosteller-calibrated]")
    print()
    print(f"  Sample sizes swept: {SAMPLE_SIZES}")
    print(f"  Permutations per n: {N_PERMS}")
    print(f"  RNG seed:           {SEED}")
    print()

    bare_emb = embed_texts([BARE_CLAIM])[0]

    # We need a deterministic order for which author-estimated terms to add.
    # Use a dedicated RNG seeded from SEED so it does not interfere with the
    # permutation-test RNG (which is freshly seeded inside run_permutation_test).
    order_rng = np.random.RandomState(SEED)
    composition_order = order_rng.permutation(len(MODAL_AUTHOR_ESTIMATED))
    print("Author-estimated inclusion order (seeded permutation, seed=42):")
    for k, idx in enumerate(composition_order):
        phrase, val = MODAL_AUTHOR_ESTIMATED[idx]
        print(f"  position {k+1}: {phrase} ({val})")
    print()

    rows = []
    for n in SAMPLE_SIZES:
        if n > MAX_N:
            print(f"  Skipping n={n}: exceeds MAX_N={MAX_N}.")
            continue

        # Build the modal set deterministically (re-using the same RNG state
        # would advance it; instead we rebuild the set with a fresh RNG seeded
        # from SEED so n=12 / n=14 / n=17 always include the SAME first
        # (n-10) author-estimated terms in the SAME order).
        build_rng = np.random.RandomState(SEED)
        phrases, medians = build_modal_set(n, build_rng)

        sentences = [MODAL_TEMPLATE.replace("{PHRASE}", p) for p in phrases]
        embs = embed_texts(sentences)
        diffs = embs - bare_emb

        n_mosteller = min(10, n)
        n_estimated = max(0, n - 10)

        print("═" * 78)
        print(f"n = {n}  ({n_mosteller} Mosteller + {n_estimated} author-estimated)")
        print("─" * 78)
        print("  Modal expressions used:")
        for k, (p, m) in enumerate(zip(phrases, medians)):
            tag = "Mosteller" if k < 10 else "AUTHOR-ESTIMATED"
            print(f"    [{tag:<16s}] {p:<22s} {m:5.1f}")
        print()

        real_rho, perm_rhos, p_value, percentile = run_permutation_test(
            diffs, medians, n_perms=N_PERMS, seed=SEED)

        perm_mean = float(np.mean(perm_rhos))
        perm_std = float(np.std(perm_rhos))
        perm_max = float(np.max(np.abs(perm_rhos)))
        perm_95 = float(np.percentile(np.abs(perm_rhos), 95))
        perm_99 = float(np.percentile(np.abs(perm_rhos), 99))

        print(f"  Real-label LOO ρ:        {real_rho:+.4f}")
        print(f"  Permuted LOO ρ mean:     {perm_mean:+.4f}")
        print(f"  Permuted LOO ρ std:      {perm_std:.4f}")
        print(f"  Permuted |ρ| max:        {perm_max:.4f}")
        print(f"  Permuted |ρ| 95th %ile:  {perm_95:.4f}")
        print(f"  Permuted |ρ| 99th %ile:  {perm_99:.4f}")
        print(f"  Real-rank percentile:    {percentile:.1f}th")
        print(f"  p-value (two-sided):     {p_value:.4f}")
        print()

        rows.append({
            "n": n,
            "n_mosteller": n_mosteller,
            "n_estimated": n_estimated,
            "real_rho": real_rho,
            "perm_mean": perm_mean,
            "perm_std": perm_std,
            "perm_95": perm_95,
            "percentile": percentile,
            "p_value": p_value,
        })

    # ─── Summary table ────────────────────────────────────────────────────
    print()
    print("═" * 78)
    print("SUMMARY TABLE  —  Modal permutation-test power vs. sample size")
    print("═" * 78)
    print()
    print(f"  {'n':>3s}  {'Most.':>5s}  {'Est.':>4s}  {'real ρ':>8s}  "
          f"{'perm mean ± std':>17s}  {'perm 95%':>8s}  "
          f"{'%ile':>6s}  {'p-val':>7s}")
    print("  " + "─" * 74)
    for r in rows:
        ms = f"{r['perm_mean']:+.3f} ± {r['perm_std']:.3f}"
        print(f"  {r['n']:>3d}  {r['n_mosteller']:>5d}  {r['n_estimated']:>4d}  "
              f"{r['real_rho']:>+8.4f}  {ms:>17s}  "
              f"{r['perm_95']:>8.4f}  {r['percentile']:>5.1f}th  "
              f"{r['p_value']:>7.4f}")
    print()
    print("  Columns:")
    print("    n         total modal items in the regression")
    print("    Most.     count of Mosteller-grounded items (max 10)")
    print("    Est.      count of AUTHOR-ESTIMATED items appended (max 7)")
    print("    real ρ    LOO Spearman ρ on real labels")
    print("    perm      mean ± std of LOO ρ over 1000 label permutations")
    print("    perm 95%  95th percentile of |permuted ρ|")
    print("    %ile      where real |ρ| sits in the permuted |ρ| distribution")
    print("    p-val     fraction of permutations with |ρ| ≥ |real ρ|")
    print()
    print("  REMINDER: rows with n > 10 include AUTHOR-ESTIMATED modal values.")
    print("  Those rows do NOT support any calibration claim. They test the")
    print("  STRUCTURAL prediction that small-n saturates rank-based metrics")
    print("  (§6 L7) by varying n with all other pipeline elements held fixed.")
    print()


if __name__ == "__main__":
    main()
