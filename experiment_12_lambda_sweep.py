"""
Experiment 12: Ridge Regularizer (λ) Robustness Sweep.

§3.3 of the paper currently states that probability-axis recovery is robust
to the ridge regularizer λ across {0.001, 0.01, 0.1, 1, 10}, "informally
verified during initial development."  An external review flagged this line
as defensively weak given the underdetermined regime (d ∈ {768, 4096},
n ∈ {10, 19}); this experiment runs a real sweep with reported numbers so
the §3.3 prose can replace the hand-wave with a quantified claim.

For each combination of (model, syntactic type, λ) we run the canonical
in-sample fit + leave-one-out (LOO) evaluation byte-identically to
paper_validation.py:train_axis (same templates, same difference vectors,
same standardization-then-ridge construction, same OLS calibration, same
[0, 100] clip).  The only thing that varies is the regularizer.  Per
(model, type) we then report the worst-case LOO ρ drop and worst-case MAE
inflation across the five λ values relative to the λ = 0.1 reference used
throughout the paper.

Models swept:
  * nomic-embed-text:v1.5     (768d, default in §3.3)
  * mxbai-embed-large         (1024d, foregrounded in §4.4)

Syntactic types swept (within-type, exactly the four reported in Table 1):
  * Predicative              (n = 13)
  * Frequency adverb         (n = 19)
  * Noun phrase              (n = 11)
  * Modal (Mosteller-only)   (n = 10)

λ values swept:
  0.001, 0.01, 0.1, 1, 10  (five points spanning four orders of magnitude)

Usage:
  python3 experiment_12_lambda_sweep.py
  python3 experiment_12_lambda_sweep.py nomic-embed-text:v1.5

  Without arguments, sweeps both models in sequence.  With one model
  argument, sweeps just that model.

Output: structured stdout suitable for redirection to
results/exp12_lambda_sweep.txt.  No files written, no state mutated.
"""

import sys
import numpy as np
import requests
from scipy import stats


OLLAMA_URL = "http://localhost:11434/api/embed"

# ────────────────────────────────────────────────────────────────────────────
# Canonical training protocol — byte-identical to paper_validation.py.
# ────────────────────────────────────────────────────────────────────────────


def embed_texts(texts, model):
    resp = requests.post(OLLAMA_URL, json={"model": model, "input": texts})
    resp.raise_for_status()
    return np.array(resp.json()["embeddings"], dtype=np.float64)


def normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def train_axis(diffs, medians, lam=0.1):
    """Train supervised probability axis via ridge regression.

    Byte-identical to paper_validation.py:train_axis.
    Returns: (axis_unit_vector, slope, intercept).
    """
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


# ────────────────────────────────────────────────────────────────────────────
# Training data — verbatim from paper_validation.py / experiment_11.py.
# Modal uses the n=10 Mosteller-only set (the load-bearing axis in §3.1).
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

MODAL_MOSTELLER_ONLY = {
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

GROUPS = [
    ("Predicative",      PREDICATIVE,           PREDICATIVE_TEMPLATE),
    ("Frequency adverb", ADVERBIAL,             ADVERBIAL_TEMPLATE),
    ("Noun phrase",      NOUN_PHRASE,           NOUN_PHRASE_TEMPLATE),
    ("Modal",            MODAL_MOSTELLER_ONLY,  MODAL_TEMPLATE),
]

LAMBDAS = [0.001, 0.01, 0.1, 1.0, 10.0]
REFERENCE_LAMBDA = 0.1

DEFAULT_MODELS = ["nomic-embed-text:v1.5", "mxbai-embed-large"]


# ────────────────────────────────────────────────────────────────────────────
# Per-(model, type) evaluation for one λ.
# ────────────────────────────────────────────────────────────────────────────


def evaluate_lambda(diffs, medians, lam):
    """Run in-sample fit + LOO for one λ.

    Returns dict with in_sample_rho, loo_rho, loo_mae.  All ρ are absolute
    values to match Table 1 reporting (the axis sign is convention).
    """
    n = len(medians)

    # In-sample fit
    w, slope, intercept = train_axis(diffs, medians, lam=lam)
    preds_is = project_to_prob(diffs, w, slope, intercept)
    r_is, _ = stats.spearmanr(preds_is, medians)
    mae_is = float(np.mean(np.abs(preds_is - medians)))

    # Leave-one-out
    loo_preds = np.zeros(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        w_i, sl_i, int_i = train_axis(diffs[mask], medians[mask], lam=lam)
        loo_preds[i] = float(np.clip(sl_i * (diffs[i] @ w_i) + int_i, 0, 100))
    r_loo, _ = stats.spearmanr(loo_preds, medians)
    mae_loo = float(np.mean(np.abs(loo_preds - medians)))

    return {
        "in_sample_rho": abs(r_is),
        "in_sample_mae": mae_is,
        "loo_rho": abs(r_loo),
        "loo_mae": mae_loo,
    }


def build_diffs(group_name, expressions, template, bare_emb, model):
    phrases = list(expressions.keys())
    medians = np.array([expressions[p] for p in phrases], dtype=np.float64)
    sentences = [template.replace("{PHRASE}", p.lower()) for p in phrases]
    embs = embed_texts(sentences, model=model)
    diffs = embs - bare_emb
    return diffs, medians


# ────────────────────────────────────────────────────────────────────────────
# Sweep + reporting per model.
# ────────────────────────────────────────────────────────────────────────────


def sweep_model(model):
    print()
    print("=" * 78)
    print(f"  MODEL: {model}")
    print("=" * 78)

    # Probe model, get dim, embed bare claim once.
    test = embed_texts(["test"], model=model)
    dim = test.shape[1]
    print(f"  dim = {dim}")
    bare_emb = embed_texts([BARE_CLAIM], model=model)[0]

    # Embed each type's diff matrix once, reuse across λ.
    type_data = {}
    for group_name, expressions, template in GROUPS:
        diffs, medians = build_diffs(group_name, expressions, template,
                                     bare_emb, model)
        type_data[group_name] = (diffs, medians)
        print(f"  {group_name:<18s}  n = {len(medians)}")

    # Run sweep.
    # results[group_name][lam] = {in_sample_rho, in_sample_mae, loo_rho, loo_mae}
    results = {}
    for group_name, _, _ in GROUPS:
        diffs, medians = type_data[group_name]
        results[group_name] = {}
        for lam in LAMBDAS:
            results[group_name][lam] = evaluate_lambda(diffs, medians, lam)

    # ─── Per-(type) detailed table ──────────────────────────────────────────
    print()
    print("  " + "─" * 74)
    print("  Per-type ridge sweep (in-sample ρ, LOO ρ, in-sample MAE, LOO MAE)")
    print("  " + "─" * 74)
    for group_name, _, _ in GROUPS:
        print()
        print(f"    {group_name}")
        print(f"      {'λ':>8s}  {'in-sample ρ':>12s}  {'LOO ρ':>8s}  "
              f"{'in-sample MAE':>14s}  {'LOO MAE':>9s}")
        print(f"      {'─'*8}  {'─'*12}  {'─'*8}  {'─'*14}  {'─'*9}")
        for lam in LAMBDAS:
            r = results[group_name][lam]
            marker = "  ← reference" if lam == REFERENCE_LAMBDA else ""
            print(f"      {lam:>8.3f}  {r['in_sample_rho']:>12.4f}  "
                  f"{r['loo_rho']:>8.4f}  {r['in_sample_mae']:>13.2f}%  "
                  f"{r['loo_mae']:>8.2f}%{marker}")

    # ─── Compact LOO ρ matrix (rows = type, columns = λ) ───────────────────
    print()
    print("  " + "─" * 74)
    print("  LOO ρ matrix — rows = type, columns = λ")
    print("  " + "─" * 74)
    header = "    " + " " * 18 + "  " + "  ".join(f"{l:>8.3f}" for l in LAMBDAS)
    print(header)
    for group_name, _, _ in GROUPS:
        row = "    " + f"{group_name:<18s}  " + "  ".join(
            f"{results[group_name][l]['loo_rho']:>8.4f}" for l in LAMBDAS)
        print(row)

    # ─── Worst-case degradation vs λ = 0.1 reference ────────────────────────
    print()
    print("  " + "─" * 74)
    print(f"  Worst-case deviation across λ ∈ {LAMBDAS} relative to λ = "
          f"{REFERENCE_LAMBDA}")
    print("  " + "─" * 74)
    print(f"    {'Type':<18s}  {'ref LOO ρ':>10s}  {'min LOO ρ':>10s}  "
          f"{'Δρ (worst)':>11s}  {'ref MAE':>9s}  {'max MAE':>9s}  "
          f"{'ΔMAE (worst)':>12s}")
    print(f"    {'─'*18}  {'─'*10}  {'─'*10}  {'─'*11}  {'─'*9}  {'─'*9}  "
          f"{'─'*12}")

    # Also collect worst-of-worst across types for the model summary line.
    overall_worst_rho_drop = 0.0
    overall_worst_mae_inflation = 0.0
    for group_name, _, _ in GROUPS:
        ref = results[group_name][REFERENCE_LAMBDA]
        loo_rhos = np.array([results[group_name][l]["loo_rho"] for l in LAMBDAS])
        loo_maes = np.array([results[group_name][l]["loo_mae"] for l in LAMBDAS])
        min_rho = float(np.min(loo_rhos))
        max_rho = float(np.max(loo_rhos))
        max_mae = float(np.max(loo_maes))
        rho_drop = ref["loo_rho"] - min_rho            # positive ⇒ degraded
        rho_gain = max_rho - ref["loo_rho"]            # positive ⇒ improved
        mae_inflation = max_mae - ref["loo_mae"]       # positive ⇒ degraded

        overall_worst_rho_drop = max(overall_worst_rho_drop, rho_drop)
        overall_worst_mae_inflation = max(overall_worst_mae_inflation,
                                          mae_inflation)

        print(f"    {group_name:<18s}  {ref['loo_rho']:>10.4f}  "
              f"{min_rho:>10.4f}  {rho_drop:>+11.4f}  "
              f"{ref['loo_mae']:>8.2f}%  {max_mae:>8.2f}%  "
              f"{mae_inflation:>+11.2f}%")

        # Notes on direction of worst-case drift.
        argmin_lam = LAMBDAS[int(np.argmin(loo_rhos))]
        argmax_mae_lam = LAMBDAS[int(np.argmax(loo_maes))]
        notes = []
        notes.append(f"min LOO ρ at λ = {argmin_lam}")
        notes.append(f"max LOO MAE at λ = {argmax_mae_lam}")
        if rho_gain > 1e-4 and (LAMBDAS[int(np.argmax(loo_rhos))]
                                != REFERENCE_LAMBDA):
            notes.append(f"better at λ = {LAMBDAS[int(np.argmax(loo_rhos))]} "
                         f"(+{rho_gain:.4f} over reference)")
        print(f"      ({'; '.join(notes)})")

    print()
    print(f"  Worst LOO ρ drop across all (type, λ) for this model: "
          f"{overall_worst_rho_drop:+.4f}")
    print(f"  Worst LOO MAE inflation across all (type, λ) for this model: "
          f"{overall_worst_mae_inflation:+.2f}%")

    return results, overall_worst_rho_drop, overall_worst_mae_inflation


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────


def main():
    if len(sys.argv) > 1:
        models = [sys.argv[1]]
    else:
        models = list(DEFAULT_MODELS)

    print()
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║  Experiment 12: Ridge regularizer (λ) robustness sweep              ║")
    print("╠══════════════════════════════════════════════════════════════════════╣")
    print(f"║  Models: {', '.join(models):<58s}║")
    print(f"║  Types : Predicative, Frequency adverb, Noun phrase, Modal         ║")
    print(f"║  λ     : {', '.join(str(l) for l in LAMBDAS):<58s}║")
    print(f"║  Reference λ = {REFERENCE_LAMBDA} (paper's fixed setting)                          ║")
    print("║                                                                    ║")
    print("║  Protocol byte-identical to paper_validation.py:train_axis.        ║")
    print("║  Per (model, type, λ): in-sample ρ, LOO ρ, in-sample MAE, LOO MAE. ║")
    print("║  Per (model, type)   : worst-case LOO ρ drop and MAE inflation     ║")
    print("║                        across the five-decade λ sweep, vs λ=0.1.  ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print()

    summaries = []
    for model in models:
        try:
            _, worst_drho, worst_dmae = sweep_model(model)
            summaries.append((model, worst_drho, worst_dmae))
        except Exception as e:
            print(f"ERROR for model {model}: {e}", file=sys.stderr)
            raise

    # ─── Cross-model summary ────────────────────────────────────────────────
    if len(summaries) > 1:
        print()
        print("=" * 78)
        print("  CROSS-MODEL SUMMARY")
        print("=" * 78)
        print(f"  Worst-case LOO degradation across λ ∈ {LAMBDAS}, vs λ = "
              f"{REFERENCE_LAMBDA}:")
        print()
        print(f"    {'Model':<28s}  {'worst LOO Δρ':>14s}  "
              f"{'worst LOO ΔMAE':>16s}")
        print(f"    {'─'*28}  {'─'*14}  {'─'*16}")
        for model, drho, dmae in summaries:
            print(f"    {model:<28s}  {drho:>+14.4f}  {dmae:>+15.2f}%")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
