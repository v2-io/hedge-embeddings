"""
Experiment 07: Matryoshka Truncation Test

Does the probability axis survive dimensional truncation?

Matryoshka Representation Learning (MRL) trains models to place important
information in early dimensions. If the hedge axis is concentrated in the
first N dimensions, truncated embeddings suffice for probability extraction
— enabling fast inference (64-dim projection instead of 768-dim).

Test: For nomic-embed-text:v1.5 (which uses MRL training):
1. Train the supervised probability axis at full dimensionality (768d)
2. Truncate embeddings to 512, 256, 128, 64 dims
3. Retrain the axis at each truncation level
4. Measure: does the axis still predict Mosteller probability?

Also test mxbai-embed-large (1024d) for comparison — it was NOT trained
with MRL, so we'd expect worse degradation under truncation.

For non-MRL models, truncation is just taking the first N dimensions of
the embedding vector (which may or may not be meaningful).

Usage:
  python3 experiment_07_matryoshka_truncation.py [model_name] [--modal=m10|m15]
  Default model: nomic-embed-text:v1.5
  Default modal set: m10 (canonical 10-item Mosteller-grounded)
                     Pass --modal=m15 to reproduce the legacy 15-item run.
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
MODAL_EXPRESSIONS = MODAL_M10 if MODAL_SET_NAME == "m10" else MODAL_M15

OLLAMA_URL = "http://localhost:11434/api/embed"


def embed_texts(texts, model=MODEL):
    resp = requests.post(OLLAMA_URL, json={"model": model, "input": texts})
    resp.raise_for_status()
    return np.array(resp.json()["embeddings"], dtype=np.float64)


def normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def train_axis(diffs, medians, lam=0.1):
    """Train supervised probability axis via ridge regression."""
    medians_c = (medians - medians.mean()) / medians.std()
    dim = diffs.shape[1]
    w = np.linalg.solve(diffs.T @ diffs + lam * np.eye(dim), diffs.T @ medians_c)
    w = normalize(w)
    projections = diffs @ w
    slope, intercept, _, _, _ = stats.linregress(projections, medians)
    return w, slope, intercept


def loo_evaluate(diffs, medians, lam=0.1):
    """Leave-one-out cross-validation. Returns (loo_rho, loo_mae)."""
    n = len(medians)
    loo_preds = np.zeros(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        w, slope, intercept = train_axis(diffs[mask], medians[mask], lam)
        loo_preds[i] = float(np.clip(slope * (diffs[i] @ w) + intercept, 0, 100))
    r, p = stats.spearmanr(loo_preds, medians)
    mae = np.mean(np.abs(loo_preds - medians))
    return r, mae, loo_preds


# ---------------------------------------------------------------------------
# Training data — sourced from data/mosteller.py
# ---------------------------------------------------------------------------

GROUPS = {
    "Predicative": {"expressions": PREDICATIVE_13,  "template": PREDICATIVE_TEMPLATE},
    "Adverbial":   {"expressions": ADVERBIAL_19,    "template": ADVERBIAL_TEMPLATE},
    "Noun phrase": {"expressions": NOUN_PHRASE_11,  "template": NOUN_PHRASE_TEMPLATE},
    "Modal":       {"expressions": MODAL_EXPRESSIONS, "template": MODAL_TEMPLATE},
}


def main():
    print()
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║  Experiment 07: Matryoshka Truncation Test                         ║")
    print("╠══════════════════════════════════════════════════════════════════════╣")
    print(f"║  Model: {MODEL:<58s}║")
    modal_label = f"Modal set: {MODAL_SET_NAME.upper()} (n={len(MODAL_EXPRESSIONS)})"
    print(f"║  {modal_label:<66s}║")
    print("║                                                                    ║")
    print("║  Question: Does the probability axis survive dimensional           ║")
    print("║  truncation? How few dimensions suffice?                           ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print()

    try:
        test = embed_texts(["test"])
        full_dim = test.shape[1]
        print(f"Model loaded. Full dimension: {full_dim}")
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    # Truncation levels to test
    truncations = [d for d in [full_dim, 512, 256, 128, 64, 32] if d <= full_dim]

    # Embed everything at full dimensionality once
    print(f"\nEmbedding all sentences at full {full_dim}d...")
    bare_emb_full = embed_texts([BARE_CLAIM])[0]

    group_data = {}
    for gname, ginfo in GROUPS.items():
        phrases = list(ginfo["expressions"].keys())
        medians = np.array([ginfo["expressions"][p] for p in phrases])
        sentences = [ginfo["template"].replace("{PHRASE}", p.lower()) for p in phrases]
        embs = embed_texts(sentences)
        diffs = embs - bare_emb_full
        group_data[gname] = {
            "phrases": phrases, "medians": medians, "diffs_full": diffs,
            "bare_full": bare_emb_full,
        }

    # ═══════════════════════════════════════════════════════════════════════
    # Test each truncation level
    # ═══════════════════════════════════════════════════════════════════════
    print()
    print("═" * 78)
    print("TRUNCATION RESULTS")
    print("═" * 78)

    # Header
    print(f"\n  {'Group':<15s}", end="")
    for d in truncations:
        print(f"  {d:>5d}d", end="")
    print()
    print(f"  {'─'*15}", end="")
    for _ in truncations:
        print(f"  {'─'*6}", end="")
    print()

    # In-sample ρ at each truncation
    print(f"\n  IN-SAMPLE Spearman ρ:")
    all_results = {}
    for gname in GROUPS:
        medians = group_data[gname]["medians"]
        diffs_full = group_data[gname]["diffs_full"]

        print(f"  {gname:<15s}", end="")
        all_results[gname] = {}
        for d in truncations:
            diffs_trunc = diffs_full[:, :d]
            w, slope, intercept = train_axis(diffs_trunc, medians)
            preds = np.clip(slope * (diffs_trunc @ w) + intercept, 0, 100)
            r, _ = stats.spearmanr(preds, medians)
            mae = np.mean(np.abs(preds - medians))
            all_results[gname][d] = {"in_sample_rho": abs(r), "in_sample_mae": mae}
            print(f"  {abs(r):5.3f}", end="")
        print()

    # LOO ρ at each truncation
    print(f"\n  LEAVE-ONE-OUT Spearman ρ:")
    for gname in GROUPS:
        medians = group_data[gname]["medians"]
        diffs_full = group_data[gname]["diffs_full"]

        print(f"  {gname:<15s}", end="")
        for d in truncations:
            diffs_trunc = diffs_full[:, :d]
            r_loo, mae_loo, _ = loo_evaluate(diffs_trunc, medians)
            all_results[gname][d]["loo_rho"] = abs(r_loo)
            all_results[gname][d]["loo_mae"] = mae_loo
            print(f"  {abs(r_loo):5.3f}", end="")
        print()

    # LOO MAE at each truncation
    print(f"\n  LEAVE-ONE-OUT MAE:")
    for gname in GROUPS:
        print(f"  {gname:<15s}", end="")
        for d in truncations:
            mae = all_results[gname][d]["loo_mae"]
            print(f"  {mae:4.1f}%", end="")
        print()

    # ═══════════════════════════════════════════════════════════════════════
    # Axis alignment across truncation levels
    # ═══════════════════════════════════════════════════════════════════════
    print()
    print("═" * 78)
    print("AXIS STABILITY: cosine(full_axis[:d], truncated_axis)")
    print("═" * 78)

    for gname in GROUPS:
        medians = group_data[gname]["medians"]
        diffs_full = group_data[gname]["diffs_full"]

        # Train at full dim
        w_full, _, _ = train_axis(diffs_full, medians)

        print(f"\n  {gname:<15s}", end="")
        for d in truncations:
            if d == full_dim:
                print(f"  {'1.000':>6s}", end="")
                continue
            diffs_trunc = diffs_full[:, :d]
            w_trunc, _, _ = train_axis(diffs_trunc, medians)
            # Compare: truncate the full axis to d dims, normalize, compare
            w_full_trunc = normalize(w_full[:d])
            cos = abs(float(w_full_trunc @ w_trunc))
            print(f"  {cos:5.3f}", end="")
        print()

    # ═══════════════════════════════════════════════════════════════════════
    # Concentration analysis: where is the axis weight?
    # ═══════════════════════════════════════════════════════════════════════
    print()
    print("═" * 78)
    print("AXIS WEIGHT CONCENTRATION")
    print("  (fraction of axis L2 norm in first N dimensions)")
    print("═" * 78)

    for gname in GROUPS:
        medians = group_data[gname]["medians"]
        diffs_full = group_data[gname]["diffs_full"]
        w_full, _, _ = train_axis(diffs_full, medians)

        print(f"\n  {gname:<15s}", end="")
        full_norm = np.linalg.norm(w_full)
        for d in truncations:
            partial_norm = np.linalg.norm(w_full[:d])
            frac = partial_norm / full_norm
            print(f"  {frac:5.3f}", end="")
        print()

    # ═══════════════════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════════════════
    print()
    print("═" * 78)
    print("SUMMARY")
    print("═" * 78)
    print()

    # Find smallest dimension where LOO ρ stays above threshold for all groups
    for threshold in [0.9, 0.8, 0.7, 0.6]:
        for d in sorted(truncations):
            if all(all_results[gname][d]["loo_rho"] >= threshold for gname in GROUPS):
                print(f"  LOO ρ ≥ {threshold:.1f} for all groups: {d}d minimum")
                break
        else:
            print(f"  LOO ρ ≥ {threshold:.1f} for all groups: not achieved at any truncation")

    print()

    # Best truncation sweet spot
    print("  Degradation from full dimensionality (LOO ρ):")
    for d in truncations:
        if d == full_dim:
            continue
        worst_drop = 0
        for gname in GROUPS:
            full_rho = all_results[gname][full_dim]["loo_rho"]
            trunc_rho = all_results[gname][d]["loo_rho"]
            drop = full_rho - trunc_rho
            if drop > worst_drop:
                worst_drop = drop
        print(f"    {d:4d}d: worst-case drop = {worst_drop:+.3f}")

    print()

    # MRL verdict
    is_mrl = "nomic" in MODEL.lower()
    if is_mrl:
        print(f"  Model uses Matryoshka training: YES")
    else:
        print(f"  Model uses Matryoshka training: UNKNOWN/NO")

    # Check if 128d preserves signal
    all_128_ok = all(
        all_results[gname].get(128, {}).get("loo_rho", 0) > 0.6
        for gname in GROUPS
    ) if 128 in truncations else False

    if all_128_ok:
        print(f"  128d sufficient? YES — LOO ρ > 0.6 for all groups at 128d")
        print(f"  → Probability extraction is feasible with {full_dim//128}× fewer dimensions")
    else:
        if 128 in truncations:
            rhos_128 = [all_results[gname][128]["loo_rho"] for gname in GROUPS]
            print(f"  128d sufficient? MARGINAL — LOO ρ range: "
                  f"[{min(rhos_128):.3f}, {max(rhos_128):.3f}]")

    print()


if __name__ == "__main__":
    main()
