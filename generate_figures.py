"""
Generate all paper figures for CMCL 2026 / TACL submission.

Produces 6 publication-quality figures:
  1. Vogel cross-validation scatter (headline figure)
  2. LOO prediction scatter (3-panel)
  3. Baseline comparison (grouped bar chart)
  4. Cross-linguistic zero-shot transfer (bar chart + heatmap)
  5. Truncation curves (line plot)
  6. Three-dataset convergence scatter (Mosteller + Vogel + Wintle)

Figures 1-3, 6 use live Ollama embeddings (mxbai-embed-large).
Figures 4-5 use hardcoded results from pre-computed experiments.

Usage:
  python3 generate_figures.py [model_name]
  Default model: mxbai-embed-large

Requires: matplotlib, numpy, scipy, requests
Requires: Ollama running locally with the specified model
"""

import sys
import os
import numpy as np
import requests
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.ticker as mticker

MODEL = sys.argv[1] if len(sys.argv) > 1 else "mxbai-embed-large"
OLLAMA_URL = "http://localhost:11434/api/embed"
FIGURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIGURE_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers (self-contained, following project convention)
# ---------------------------------------------------------------------------

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


def project_to_prob(diffs, axis, slope, intercept):
    proj = diffs @ axis
    return np.clip(slope * proj + intercept, 0, 100)


def run_loo(diffs, medians):
    n = len(medians)
    loo_preds = np.zeros(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        w, slope, intercept = train_axis(diffs[mask], medians[mask])
        loo_preds[i] = float(np.clip(
            slope * (diffs[i] @ w) + intercept, 0, 100))
    return loo_preds


def savefig(fig, name):
    pdf_path = os.path.join(FIGURE_DIR, f"{name}.pdf")
    png_path = os.path.join(FIGURE_DIR, f"{name}.png")
    fig.savefig(pdf_path, bbox_inches="tight", dpi=300)
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    print(f"  Saved: {pdf_path}")
    print(f"  Saved: {png_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Style configuration
# ---------------------------------------------------------------------------

STYLE = {
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
}
plt.rcParams.update(STYLE)

COLORS = {
    "pred": "#2176AE",
    "adv": "#57B894",
    "noun": "#F18F01",
    "modal": "#C73E1D",
    "vogel": "#2176AE",
    "wintle": "#57B894",
    "random": "#AAAAAA",
    "mean_diff": "#999999",
    "pca": "#F18F01",
    "supervised": "#2176AE",
}


# ---------------------------------------------------------------------------
# Training data (Mosteller & Youtz 1990)
# ---------------------------------------------------------------------------

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

BARE_CLAIM = "The experiment will succeed"

# Vogel 2022 (meta-analysis, independent validation)
VOGEL_PREDICATIVE = {
    "Certain": 95.0, "Almost certain": 85.7, "Very likely": 82.3,
    "Very probable": 82.5, "Likely": 69.2, "Probable": 70.1,
    "Possible": 42.0, "Unlikely": 17.9, "Very unlikely": 11.1,
    "Improbable": 15.8, "Impossible": 7.24,
}

# Wintle et al. 2019 (n~924, third independent validation)
WINTLE_PREDICATIVE = {
    "Almost certain": 93, "Very likely": 85,
    "Likely": 73, "Probable": 72,
    "Very unlikely": 8, "Unlikely": 15,
    "Impossible": 2, "Possible": 45,
    "Improbable": 12,
}

GROUPS = [
    ("Predicative", PREDICATIVE, PREDICATIVE_TEMPLATE),
    ("Adverbial", ADVERBIAL, ADVERBIAL_TEMPLATE),
    ("Noun Phrase", NOUN_PHRASE, NOUN_PHRASE_TEMPLATE),
]


# ---------------------------------------------------------------------------
# Figure 1: Vogel Cross-Validation Scatter
# ---------------------------------------------------------------------------

def figure_1_vogel(bare_emb):
    print("\nFigure 1: Vogel Cross-Validation Scatter...")

    # Train on Mosteller predicative
    phrases = list(PREDICATIVE.keys())
    medians = np.array([PREDICATIVE[p] for p in phrases])
    sentences = [PREDICATIVE_TEMPLATE.replace("{PHRASE}", p.lower()) for p in phrases]
    embs = embed_texts(sentences)
    diffs = embs - bare_emb
    w, slope, intercept = train_axis(diffs, medians)

    # Predict Vogel
    vogel_phrases = list(VOGEL_PREDICATIVE.keys())
    vogel_true = np.array([VOGEL_PREDICATIVE[p] for p in vogel_phrases])
    vogel_sents = [PREDICATIVE_TEMPLATE.replace("{PHRASE}", p.lower()) for p in vogel_phrases]
    vogel_embs = embed_texts(vogel_sents)
    vogel_diffs = vogel_embs - bare_emb
    vogel_preds = project_to_prob(vogel_diffs, w, slope, intercept)

    rho, p_val = stats.spearmanr(vogel_preds, vogel_true)
    mae = np.mean(np.abs(vogel_preds - vogel_true))

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 100], [0, 100], "k--", alpha=0.3, linewidth=1, zorder=1)
    ax.scatter(vogel_true, vogel_preds, c=COLORS["vogel"], s=60,
               edgecolors="white", linewidth=0.5, zorder=3)

    for i, phrase in enumerate(vogel_phrases):
        offset_x, offset_y = 2.5, 2.5
        ha = "left"
        # Avoid overlaps for close points
        if phrase == "Probable":
            offset_y = -4
        elif phrase == "Very probable":
            offset_y = 4
        elif phrase == "Improbable":
            offset_x = -2.5
            ha = "right"
        elif phrase == "Very unlikely":
            offset_y = -4
        ax.annotate(phrase, (vogel_true[i], vogel_preds[i]),
                    xytext=(offset_x, offset_y), textcoords="offset points",
                    fontsize=8, ha=ha, va="bottom", color="#444444")

    ax.set_xlabel("Vogel et al. (2022) ground truth (%)")
    ax.set_ylabel("Model prediction (%)")
    ax.set_title("Cross-Validation: Mosteller-Trained Axis\nPredicts Independent Meta-Analysis")
    ax.set_xlim(-5, 105)
    ax.set_ylim(-5, 105)
    ax.set_aspect("equal")
    ax.text(0.05, 0.95, f"$\\rho$ = {rho:.3f}\nMAE = {mae:.1f}%",
            transform=ax.transAxes, fontsize=11, verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#CCCCCC", alpha=0.9))
    fig.tight_layout()
    savefig(fig, "fig1_vogel_crossval")
    return vogel_preds, vogel_true, vogel_phrases, w, slope, intercept


# ---------------------------------------------------------------------------
# Figure 2: LOO Prediction Scatter (3-panel)
# ---------------------------------------------------------------------------

def figure_2_loo(bare_emb):
    print("\nFigure 2: LOO Prediction Scatter...")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    group_colors = [COLORS["pred"], COLORS["adv"], COLORS["noun"]]

    all_loo_data = {}

    for idx, (group_name, expressions, template) in enumerate(GROUPS):
        ax = axes[idx]
        phrases = list(expressions.keys())
        medians = np.array([expressions[p] for p in phrases])
        sentences = [template.replace("{PHRASE}", p.lower()) for p in phrases]
        embs = embed_texts(sentences)
        diffs = embs - bare_emb

        loo_preds = run_loo(diffs, medians)
        rho, p_val = stats.spearmanr(loo_preds, medians)
        mae = np.mean(np.abs(loo_preds - medians))

        all_loo_data[group_name] = (loo_preds, medians, phrases)

        ax.plot([0, 100], [0, 100], "k--", alpha=0.3, linewidth=1, zorder=1)
        ax.scatter(medians, loo_preds, c=group_colors[idx], s=45,
                   edgecolors="white", linewidth=0.5, zorder=3)

        # Label points — skip if too crowded
        for i, phrase in enumerate(phrases):
            # Abbreviate long names
            label = phrase
            if len(label) > 16:
                label = label[:14] + ".."
            ax.annotate(label, (medians[i], loo_preds[i]),
                        xytext=(3, 3), textcoords="offset points",
                        fontsize=6.5, color="#555555", ha="left", va="bottom")

        ax.set_xlabel("Mosteller ground truth (%)")
        if idx == 0:
            ax.set_ylabel("LOO predicted (%)")
        ax.set_title(f"{group_name} (n={len(phrases)})")
        ax.set_xlim(-5, 105)
        ax.set_ylim(-5, 105)
        ax.set_aspect("equal")
        ax.text(0.05, 0.95,
                f"LOO $\\rho$ = {rho:.3f}\nMAE = {mae:.1f}%",
                transform=ax.transAxes, fontsize=9, verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="#CCCCCC", alpha=0.9))

    fig.suptitle(f"Leave-One-Out Cross-Validation ({MODEL})", fontsize=14, y=1.02)
    fig.tight_layout()
    savefig(fig, "fig2_loo_scatter")
    return all_loo_data


# ---------------------------------------------------------------------------
# Figure 3: Baseline Comparison (grouped bar chart)
# ---------------------------------------------------------------------------

def figure_3_baselines(bare_emb):
    print("\nFigure 3: Baseline Comparison...")

    fig, ax = plt.subplots(figsize=(8, 4.5))

    n_groups = len(GROUPS)
    n_methods = 4
    bar_width = 0.18
    x = np.arange(n_groups)

    method_data = {m: {"rhos": [], "errs": []} for m in
                   ["Random", "Mean-diff", "PCA PC1", "Supervised\n(LOO)"]}
    method_colors = [COLORS["random"], COLORS["mean_diff"],
                     COLORS["pca"], COLORS["supervised"]]

    for group_name, expressions, template in GROUPS:
        phrases = list(expressions.keys())
        medians = np.array([expressions[p] for p in phrases])
        n = len(phrases)
        sentences = [template.replace("{PHRASE}", p.lower()) for p in phrases]
        embs = embed_texts(sentences)
        diffs = embs - bare_emb
        dim = diffs.shape[1]

        # Random direction (100 trials)
        rng = np.random.RandomState(42)
        random_rhos = []
        for _ in range(100):
            rand_dir = normalize(rng.randn(dim))
            proj = diffs @ rand_dir
            sl, intc, _, _, _ = stats.linregress(proj, medians)
            preds = np.clip(sl * proj + intc, 0, 100)
            r, _ = stats.spearmanr(preds, medians)
            random_rhos.append(abs(r))
        method_data["Random"]["rhos"].append(np.mean(random_rhos))
        method_data["Random"]["errs"].append(np.std(random_rhos))

        # Mean-diff
        mean_dir = normalize(diffs.mean(axis=0))
        proj = diffs @ mean_dir
        sl, intc, _, _, _ = stats.linregress(proj, medians)
        preds = np.clip(sl * proj + intc, 0, 100)
        r, _ = stats.spearmanr(preds, medians)
        method_data["Mean-diff"]["rhos"].append(abs(r))
        method_data["Mean-diff"]["errs"].append(0)

        # PCA PC1
        diffs_c = diffs - diffs.mean(axis=0)
        _, S, Vt = np.linalg.svd(diffs_c, full_matrices=False)
        pc1 = Vt[0]
        proj = diffs @ pc1
        sl, intc, _, _, _ = stats.linregress(proj, medians)
        preds = np.clip(sl * proj + intc, 0, 100)
        r, _ = stats.spearmanr(preds, medians)
        method_data["PCA PC1"]["rhos"].append(abs(r))
        method_data["PCA PC1"]["errs"].append(0)

        # Supervised (LOO)
        loo_preds = run_loo(diffs, medians)
        r, _ = stats.spearmanr(loo_preds, medians)
        method_data["Supervised\n(LOO)"]["rhos"].append(abs(r))
        method_data["Supervised\n(LOO)"]["errs"].append(0)

    for i, (method, data) in enumerate(method_data.items()):
        offset = (i - n_methods / 2 + 0.5) * bar_width
        bars = ax.bar(x + offset, data["rhos"], bar_width, label=method,
                      color=method_colors[i], edgecolor="white", linewidth=0.5,
                      yerr=data["errs"] if any(e > 0 for e in data["errs"]) else None,
                      capsize=3, error_kw={"linewidth": 1})

    ax.set_xticks(x)
    ax.set_xticklabels([g[0] for g in GROUPS])
    ax.set_ylabel("Spearman $\\rho$")
    ax.set_title("Method Comparison: Supervised Ridge vs. Baselines")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", framealpha=0.9)
    ax.axhline(y=0.5, color="#CCCCCC", linestyle=":", linewidth=0.8)

    fig.tight_layout()
    savefig(fig, "fig3_baselines")


# ---------------------------------------------------------------------------
# Figure 4: Cross-Linguistic (bar chart + heatmap)
# ---------------------------------------------------------------------------

def figure_4_crosslingual():
    print("\nFigure 4: Cross-Linguistic Transfer...")

    # Hardcoded from exp09_bge-m3_8lang.txt
    languages = ["English", "German", "French", "Spanish", "Chinese",
                 "Japanese", "Korean", "Arabic", "Hindi"]
    lang_short = ["EN", "DE", "FR", "ES", "ZH", "JA", "KO", "AR", "HI"]

    # Zero-shot rho values (from SUMMARY table)
    zeroshot_pred = {
        "German": 0.952, "French": 0.952, "Spanish": 0.905,
        "Chinese": 0.952, "Japanese": 0.952, "Korean": 1.000,
        "Arabic": 0.833, "Hindi": 0.952,
    }
    zeroshot_modal = {
        "German": 0.918, "French": 0.946, "Spanish": 0.852,
        "Chinese": 0.991, "Japanese": 0.857, "Korean": 0.841,
        "Arabic": 0.943, "Hindi": 1.000,
    }

    # 9x9 pairwise axis alignment matrix (predicative)
    alignment = np.array([
        [1.000, 0.940, 0.941, 0.904, 0.779, 0.845, 0.865, 0.859, 0.845],
        [0.940, 1.000, 0.917, 0.926, 0.782, 0.828, 0.860, 0.853, 0.839],
        [0.941, 0.917, 1.000, 0.860, 0.766, 0.811, 0.833, 0.881, 0.815],
        [0.904, 0.926, 0.860, 1.000, 0.770, 0.823, 0.850, 0.812, 0.800],
        [0.779, 0.782, 0.766, 0.770, 1.000, 0.821, 0.821, 0.756, 0.813],
        [0.845, 0.828, 0.811, 0.823, 0.821, 1.000, 0.900, 0.800, 0.849],
        [0.865, 0.860, 0.833, 0.850, 0.821, 0.900, 1.000, 0.802, 0.871],
        [0.859, 0.853, 0.881, 0.812, 0.756, 0.800, 0.802, 1.000, 0.768],
        [0.845, 0.839, 0.815, 0.800, 0.813, 0.849, 0.871, 0.768, 1.000],
    ])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5),
                                    gridspec_kw={"width_ratios": [1.1, 1]})

    # Panel A: Zero-shot rho bars
    target_langs = [l for l in languages if l != "English"]
    target_short = [s for s, l in zip(lang_short, languages) if l != "English"]
    x = np.arange(len(target_langs))
    bar_width = 0.35

    pred_vals = [zeroshot_pred[l] for l in target_langs]
    modal_vals = [zeroshot_modal[l] for l in target_langs]

    ax1.bar(x - bar_width / 2, pred_vals, bar_width, label="Predicative",
            color=COLORS["pred"], edgecolor="white", linewidth=0.5)
    ax1.bar(x + bar_width / 2, modal_vals, bar_width, label="Modal",
            color=COLORS["modal"], edgecolor="white", linewidth=0.5)

    ax1.set_xticks(x)
    ax1.set_xticklabels(target_short, fontsize=10)
    ax1.set_ylabel("Zero-Shot Spearman $\\rho$")
    ax1.set_title("(a) Zero-Shot Transfer from English")
    ax1.set_ylim(0.7, 1.05)
    ax1.axhline(y=0.9, color="#CCCCCC", linestyle=":", linewidth=0.8)
    ax1.legend(loc="lower left", framealpha=0.9)

    # Add value labels on bars
    for i, (pv, mv) in enumerate(zip(pred_vals, modal_vals)):
        if pv == 1.0 or mv == 1.0:
            if pv == 1.0:
                ax1.text(i - bar_width / 2, pv + 0.005, "1.0",
                         ha="center", va="bottom", fontsize=7, fontweight="bold")
            if mv == 1.0:
                ax1.text(i + bar_width / 2, mv + 0.005, "1.0",
                         ha="center", va="bottom", fontsize=7, fontweight="bold")

    # Panel B: Heatmap
    im = ax2.imshow(alignment, cmap="YlOrRd", vmin=0.7, vmax=1.0, aspect="equal")
    ax2.set_xticks(range(len(lang_short)))
    ax2.set_xticklabels(lang_short, fontsize=9)
    ax2.set_yticks(range(len(lang_short)))
    ax2.set_yticklabels(lang_short, fontsize=9)
    ax2.set_title("(b) Pairwise Axis Alignment (Predicative)")
    ax2.spines[:].set_visible(False)
    ax2.grid(False)

    # Annotate cells
    for i in range(len(languages)):
        for j in range(len(languages)):
            val = alignment[i, j]
            color = "white" if val > 0.9 else "black"
            if i != j:
                ax2.text(j, i, f"{val:.2f}", ha="center", va="center",
                         fontsize=7, color=color)

    cbar = fig.colorbar(im, ax=ax2, shrink=0.8, label="Cosine Similarity")

    fig.suptitle("Cross-Linguistic Probability Geometry (bge-m3, 8 Languages)",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    savefig(fig, "fig4_crosslingual")


# ---------------------------------------------------------------------------
# Figure 5: Truncation Curves
# ---------------------------------------------------------------------------

def figure_5_truncation():
    print("\nFigure 5: Truncation Curves...")

    # Hardcoded from truncation_nomic-v1.5.txt
    dims = [768, 512, 256, 128, 64, 32]
    loo_rho = {
        "Predicative": [0.874, 0.885, 0.885, 0.852, 0.897, 0.852],
        "Adverbial":   [0.923, 0.907, 0.896, 0.893, 0.896, 0.807],
        "Noun Phrase":  [0.936, 0.918, 0.918, 0.900, 0.909, 0.936],
        "Modal":       [0.948, 0.951, 0.946, 0.906, 0.924, 0.960],
    }

    type_colors = {
        "Predicative": COLORS["pred"],
        "Adverbial": COLORS["adv"],
        "Noun Phrase": COLORS["noun"],
        "Modal": COLORS["modal"],
    }
    type_markers = {
        "Predicative": "o",
        "Adverbial": "s",
        "Noun Phrase": "D",
        "Modal": "^",
    }

    fig, ax = plt.subplots(figsize=(6, 4.5))

    for group_name, rhos in loo_rho.items():
        ax.plot(dims, rhos, marker=type_markers[group_name], markersize=7,
                color=type_colors[group_name], label=group_name,
                linewidth=2, markeredgecolor="white", markeredgewidth=0.5)

    ax.set_xlabel("Embedding Dimensions")
    ax.set_ylabel("LOO Spearman $\\rho$")
    ax.set_title("Matryoshka Truncation: Probability Axis Survives Compression\n(nomic-embed-text v1.5)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(dims)
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.set_ylim(0.75, 1.0)
    ax.legend(loc="lower left", framealpha=0.9)
    ax.axhline(y=0.8, color="#CCCCCC", linestyle=":", linewidth=0.8, label="_")
    ax.invert_xaxis()

    fig.tight_layout()
    savefig(fig, "fig5_truncation")


# ---------------------------------------------------------------------------
# Figure 6: Three-Dataset Convergence
# ---------------------------------------------------------------------------

def figure_6_three_datasets(bare_emb):
    print("\nFigure 6: Three-Dataset Convergence...")

    # Train on Mosteller predicative
    phrases = list(PREDICATIVE.keys())
    medians = np.array([PREDICATIVE[p] for p in phrases])
    sentences = [PREDICATIVE_TEMPLATE.replace("{PHRASE}", p.lower()) for p in phrases]
    embs = embed_texts(sentences)
    diffs = embs - bare_emb
    w, slope, intercept = train_axis(diffs, medians)

    # Predict Vogel
    vogel_phrases = list(VOGEL_PREDICATIVE.keys())
    vogel_true = np.array([VOGEL_PREDICATIVE[p] for p in vogel_phrases])
    vogel_sents = [PREDICATIVE_TEMPLATE.replace("{PHRASE}", p.lower()) for p in vogel_phrases]
    vogel_diffs = embed_texts(vogel_sents) - bare_emb
    vogel_preds = project_to_prob(vogel_diffs, w, slope, intercept)

    # Predict Wintle
    wintle_phrases = list(WINTLE_PREDICATIVE.keys())
    wintle_true = np.array([WINTLE_PREDICATIVE[p] for p in wintle_phrases])
    wintle_sents = [PREDICATIVE_TEMPLATE.replace("{PHRASE}", p.lower())
                    for p in wintle_phrases]
    wintle_diffs = embed_texts(wintle_sents) - bare_emb
    wintle_preds = project_to_prob(wintle_diffs, w, slope, intercept)

    rho_v, _ = stats.spearmanr(vogel_preds, vogel_true)
    rho_w, _ = stats.spearmanr(wintle_preds, wintle_true)
    mae_v = np.mean(np.abs(vogel_preds - vogel_true))
    mae_w = np.mean(np.abs(wintle_preds - wintle_true))

    fig, ax = plt.subplots(figsize=(6, 5.5))

    ax.plot([0, 100], [0, 100], "k--", alpha=0.3, linewidth=1, zorder=1)

    # Vogel points
    ax.scatter(vogel_true, vogel_preds, c=COLORS["vogel"], s=65,
               marker="o", edgecolors="white", linewidth=0.5, zorder=3,
               label=f"Vogel 2022 ($\\rho$={rho_v:.3f})")

    # Wintle points
    ax.scatter(wintle_true, wintle_preds, c=COLORS["wintle"], s=65,
               marker="s", edgecolors="white", linewidth=0.5, zorder=3,
               label=f"Wintle 2019 ($\\rho$={rho_w:.3f})")

    # Label Vogel points
    for i, phrase in enumerate(vogel_phrases):
        ax.annotate(phrase, (vogel_true[i], vogel_preds[i]),
                    xytext=(3, 4), textcoords="offset points",
                    fontsize=7, color=COLORS["vogel"], alpha=0.8)

    # Label Wintle points (offset to avoid overlap with Vogel)
    for i, phrase in enumerate(wintle_phrases):
        # Check if this phrase also appears in Vogel (shared expressions)
        if phrase in VOGEL_PREDICATIVE:
            ax.annotate(phrase, (wintle_true[i], wintle_preds[i]),
                        xytext=(3, -8), textcoords="offset points",
                        fontsize=7, color=COLORS["wintle"], alpha=0.8)
        else:
            ax.annotate(phrase, (wintle_true[i], wintle_preds[i]),
                        xytext=(3, 4), textcoords="offset points",
                        fontsize=7, color=COLORS["wintle"], alpha=0.8)

    ax.set_xlabel("Ground truth probability (%)")
    ax.set_ylabel("Model prediction (%)")
    ax.set_title("Three-Dataset Convergence\n(Mosteller-trained axis, predicative frame)")
    ax.set_xlim(-5, 105)
    ax.set_ylim(-5, 105)
    ax.set_aspect("equal")
    ax.legend(loc="lower right", framealpha=0.9)

    fig.tight_layout()
    savefig(fig, "fig6_three_datasets")


# ---------------------------------------------------------------------------
# Figure 7: Null Hypothesis Controls
# ---------------------------------------------------------------------------

def figure_7_null_hypothesis():
    print("\nFigure 7: Null Hypothesis Controls...")

    # Hardcoded from exp10_nomic-v1.5.txt
    groups = ["Predicative", "Adverbial", "Noun Phrase"]

    # Real LOO rho values
    real_loo = [0.874, 0.923, 0.936]

    # Permutation test (1000 shuffles): mean, std, 95th, 99th percentile
    perm_mean = [-0.159, -0.080, -0.147]
    perm_std = [0.356, 0.292, 0.414]
    perm_95 = [0.731, 0.603, 0.800]
    perm_99 = [0.834, 0.795, 0.918]
    perm_p = [0.003, 0.000, 0.005]

    # Non-epistemic control
    non_epistemic_loo = 0.306  # pure non-epistemic
    real_pred_loo = 0.874      # real predicative for comparison

    # Random-label p-values
    rand_p = [0.006, 0.000, 0.003]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5),
                                    gridspec_kw={"width_ratios": [1.5, 1]})

    # --- Panel A: Permutation test ---
    x = np.arange(len(groups))
    bar_width = 0.3

    # Null distribution range (mean to 95th percentile as bar, whisker to 99th)
    null_heights = [p95 - pm for p95, pm in zip(perm_95, perm_mean)]
    ax1.bar(x - bar_width / 2, null_heights, bar_width, bottom=perm_mean,
            color="#DDDDDD", edgecolor="#AAAAAA", linewidth=0.8,
            label="Permuted null (mean to 95th %ile)")

    # Add whiskers from 95th to 99th percentile
    for i in range(len(groups)):
        ax1.plot([x[i] - bar_width / 2, x[i] - bar_width / 2],
                 [perm_95[i], perm_99[i]], color="#888888", linewidth=1.5)
        ax1.plot([x[i] - bar_width / 2 - 0.05, x[i] - bar_width / 2 + 0.05],
                 [perm_99[i], perm_99[i]], color="#888888", linewidth=1.5)

    # Null distribution mean markers
    for i in range(len(groups)):
        ax1.plot(x[i] - bar_width / 2, perm_mean[i], "k_", markersize=10,
                 markeredgewidth=1.5)

    # Real LOO rho values
    ax1.bar(x + bar_width / 2, real_loo, bar_width,
            color=[COLORS["pred"], COLORS["adv"], COLORS["noun"]],
            edgecolor="white", linewidth=0.5,
            label="Real LOO $\\rho$")

    # p-value annotations
    for i in range(len(groups)):
        p = perm_p[i]
        p_str = f"p={p:.3f}" if p > 0 else "p<0.001"
        ax1.text(x[i], max(real_loo[i], perm_99[i]) + 0.03, p_str,
                 ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax1.set_xticks(x)
    ax1.set_xticklabels(groups)
    ax1.set_ylabel("LOO Spearman $\\rho$")
    ax1.set_title("(a) Permutation Test: Real vs. Shuffled Labels")
    ax1.set_ylim(-0.6, 1.15)
    ax1.axhline(y=0, color="#CCCCCC", linestyle="-", linewidth=0.5)
    ax1.legend(loc="lower left", framealpha=0.9, fontsize=9)

    # --- Panel B: Epistemic vs. Non-Epistemic ---
    categories = ["Epistemic\n(hedge words)", "Non-Epistemic\n(random adjectives)"]
    values = [real_pred_loo, non_epistemic_loo]
    colors_b = [COLORS["pred"], "#AAAAAA"]

    bars = ax2.bar(categories, values, width=0.5, color=colors_b,
                   edgecolor="white", linewidth=0.5)

    # Value labels on bars
    for bar, val in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=11,
                 fontweight="bold")

    ax2.set_ylabel("LOO Spearman $\\rho$")
    ax2.set_title("(b) Epistemic Specificity")
    ax2.set_ylim(0, 1.1)

    fig.suptitle("Null Hypothesis Controls (nomic-embed-text v1.5)",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    savefig(fig, "fig7_null_hypothesis")


# ---------------------------------------------------------------------------
# Figure 8: Syntactic Confound PCA
# ---------------------------------------------------------------------------

def figure_8_pca_confound(bare_emb):
    print("\nFigure 8: Syntactic Confound PCA...")

    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    # Embed all three types
    all_diffs = []
    all_types = []
    all_medians_list = []
    all_phrases_list = []

    for group_name, expressions, template in GROUPS:
        phrases = list(expressions.keys())
        medians = np.array([expressions[p] for p in phrases])
        sentences = [template.replace("{PHRASE}", p.lower()) for p in phrases]
        embs = embed_texts(sentences)
        diffs = embs - bare_emb
        all_diffs.append(diffs)
        all_types.extend([group_name] * len(phrases))
        all_medians_list.extend(medians.tolist())
        all_phrases_list.extend(phrases)

    mixed_diffs = np.vstack(all_diffs)
    all_medians_arr = np.array(all_medians_list)

    # PCA on mixed
    mixed_centered = mixed_diffs - mixed_diffs.mean(axis=0)
    U_m, S_m, Vt_m = np.linalg.svd(mixed_centered, full_matrices=False)
    pc_mixed = mixed_centered @ Vt_m[:2].T
    var_mixed = S_m[:2] ** 2 / np.sum(S_m ** 2)

    # PCA on predicative only
    pred_diffs = all_diffs[0]
    pred_medians = np.array([PREDICATIVE[p] for p in PREDICATIVE])
    pred_centered = pred_diffs - pred_diffs.mean(axis=0)
    U_p, S_p, Vt_p = np.linalg.svd(pred_centered, full_matrices=False)
    pc_pred = pred_centered @ Vt_p[:2].T
    var_pred = S_p[:2] ** 2 / np.sum(S_p ** 2)

    # Correlations
    rho_pc1_mixed, _ = stats.spearmanr(pc_mixed[:, 0], all_medians_arr)
    rho_pc2_mixed, _ = stats.spearmanr(pc_mixed[:, 1], all_medians_arr)
    rho_pc1_pred, _ = stats.spearmanr(pc_pred[:, 0], pred_medians)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # --- Panel A: Mixed types (PC1 = syntax) ---
    type_colors_map = {"Predicative": COLORS["pred"], "Adverbial": COLORS["adv"],
                       "Noun Phrase": COLORS["noun"]}
    for gname in ["Predicative", "Adverbial", "Noun Phrase"]:
        mask = [t == gname for t in all_types]
        idx = np.where(mask)[0]
        ax1.scatter(pc_mixed[idx, 0], pc_mixed[idx, 1],
                    c=type_colors_map[gname], s=50, label=gname,
                    edgecolors="white", linewidth=0.5, zorder=3)

    ax1.set_xlabel(f"PC1 ({var_mixed[0]*100:.0f}% var)")
    ax1.set_ylabel(f"PC2 ({var_mixed[1]*100:.0f}% var)")
    ax1.set_title("(a) All Types Mixed: PC1 = Syntax")
    ax1.legend(loc="best", framealpha=0.9, fontsize=9)
    ax1.text(0.05, 0.05,
             f"PC1 vs. prob: $\\rho$={rho_pc1_mixed:.2f}\nPC2 vs. prob: $\\rho$={rho_pc2_mixed:.2f}",
             transform=ax1.transAxes, fontsize=9, verticalalignment="bottom",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                       edgecolor="#CCCCCC", alpha=0.9))

    # --- Panel B: Predicative only (PC1 = probability) ---
    # viridis: perceptually uniform, colorblind-safe, grayscale-safe
    # (replaces RdYlGn which fails for red-green deficiency, ~8% of men)
    norm = Normalize(vmin=0, vmax=100)
    cmap = plt.cm.viridis
    sc = ax2.scatter(pc_pred[:, 0], pc_pred[:, 1],
                     c=pred_medians, cmap=cmap, norm=norm, s=60,
                     edgecolors="white", linewidth=0.5, zorder=3)

    pred_phrases = list(PREDICATIVE.keys())
    for i, phrase in enumerate(pred_phrases):
        label = phrase if len(phrase) <= 15 else phrase[:13] + ".."
        ax2.annotate(label, (pc_pred[i, 0], pc_pred[i, 1]),
                     xytext=(4, 4), textcoords="offset points",
                     fontsize=7, color="#444444")

    ax2.set_xlabel(f"PC1 ({var_pred[0]*100:.0f}% var)")
    ax2.set_ylabel(f"PC2 ({var_pred[1]*100:.0f}% var)")
    ax2.set_title("(b) Predicative Only: PC1 = Probability")
    cbar = fig.colorbar(sc, ax=ax2, shrink=0.8, label="Mosteller Median (%)")
    ax2.text(0.05, 0.05,
             f"PC1 vs. prob: $\\rho$={rho_pc1_pred:.2f}",
             transform=ax2.transAxes, fontsize=9, verticalalignment="bottom",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                       edgecolor="#CCCCCC", alpha=0.9))

    fig.suptitle("The Syntactic Confound: Why Within-Type Analysis Is Essential",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    savefig(fig, "fig8_pca_confound")


# ---------------------------------------------------------------------------
# Figure 9: Method Diagram
# ---------------------------------------------------------------------------

def figure_9_method():
    print("\nFigure 9: Method Diagram...")

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 4)
    ax.axis("off")
    ax.set_aspect("equal")

    box_style = dict(boxstyle="round,pad=0.4", facecolor="#F0F4F8",
                     edgecolor="#2176AE", linewidth=1.5)
    embed_style = dict(boxstyle="round,pad=0.3", facecolor="#E8F5E9",
                       edgecolor="#57B894", linewidth=1.5)
    result_style = dict(boxstyle="round,pad=0.4", facecolor="#FFF3E0",
                        edgecolor="#F18F01", linewidth=1.5)
    axis_style = dict(boxstyle="round,pad=0.3", facecolor="#FFEBEE",
                      edgecolor="#C73E1D", linewidth=1.5)

    # Step 1: Input sentences
    ax.text(1.0, 3.2, '"The experiment will succeed"',
            fontsize=9, ha="center", va="center", bbox=box_style,
            fontstyle="italic")
    ax.text(1.0, 3.7, "Bare claim", fontsize=8, ha="center",
            color="#666666", fontweight="bold")

    ax.text(1.0, 1.0, '"The experiment will probably succeed"',
            fontsize=9, ha="center", va="center", bbox=box_style,
            fontstyle="italic")
    ax.text(1.0, 0.5, "Hedged sentence", fontsize=8, ha="center",
            color="#666666", fontweight="bold")

    # Step 2: Embed
    ax.text(3.5, 3.2, "Embed", fontsize=10, ha="center", va="center",
            bbox=embed_style, fontweight="bold")
    ax.text(3.5, 1.0, "Embed", fontsize=10, ha="center", va="center",
            bbox=embed_style, fontweight="bold")

    # Arrows: input -> embed
    ax.annotate("", xy=(2.7, 3.2), xytext=(2.1, 3.2),
                arrowprops=dict(arrowstyle="->", color="#888888", lw=1.5))
    ax.annotate("", xy=(2.7, 1.0), xytext=(2.1, 1.0),
                arrowprops=dict(arrowstyle="->", color="#888888", lw=1.5))

    # Step 3: Vectors
    ax.text(5.2, 3.2, "$\\mathbf{e}_{bare}$", fontsize=13, ha="center",
            va="center", color="#2176AE")
    ax.text(5.2, 1.0, "$\\mathbf{e}_{hedged}$", fontsize=13, ha="center",
            va="center", color="#2176AE")

    # Arrows: embed -> vectors
    ax.annotate("", xy=(4.6, 3.2), xytext=(4.1, 3.2),
                arrowprops=dict(arrowstyle="->", color="#888888", lw=1.5))
    ax.annotate("", xy=(4.6, 1.0), xytext=(4.1, 1.0),
                arrowprops=dict(arrowstyle="->", color="#888888", lw=1.5))

    # Step 4: Subtract
    ax.text(6.6, 2.1, "$\\Delta = \\mathbf{e}_{hedged} - \\mathbf{e}_{bare}$",
            fontsize=11, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#E3F2FD",
                      edgecolor="#2176AE", linewidth=1.5),
            fontweight="bold")
    ax.text(6.6, 1.5, "Difference vector", fontsize=8, ha="center",
            color="#666666")

    # Arrows: vectors -> subtract
    ax.annotate("", xy=(5.8, 2.5), xytext=(5.6, 3.0),
                arrowprops=dict(arrowstyle="->", color="#888888", lw=1.5))
    ax.annotate("", xy=(5.8, 1.7), xytext=(5.6, 1.2),
                arrowprops=dict(arrowstyle="->", color="#888888", lw=1.5))

    # Step 5: Project onto axis
    ax.text(8.6, 2.1, "$p = \\Delta \\cdot \\mathbf{w}_{axis}$",
            fontsize=11, ha="center", va="center", bbox=axis_style,
            fontweight="bold")
    ax.text(8.6, 1.5, "Project onto\nprobability axis", fontsize=8,
            ha="center", color="#666666")

    # Arrow: subtract -> project
    ax.annotate("", xy=(7.7, 2.1), xytext=(7.4, 2.1),
                arrowprops=dict(arrowstyle="->", color="#888888", lw=1.5))

    # Step 6: Result
    ax.text(10.2, 2.1, "70.2%", fontsize=16, ha="center", va="center",
            bbox=result_style, fontweight="bold", color="#F18F01")
    ax.text(10.2, 1.5, "Calibrated\nprobability", fontsize=8,
            ha="center", color="#666666")

    # Arrow: project -> result
    ax.annotate("", xy=(9.5, 2.1), xytext=(9.3, 2.1),
                arrowprops=dict(arrowstyle="->", color="#888888", lw=1.5))

    # Title
    ax.text(5.5, 3.9, "Method: Extracting Calibrated Probability from Embeddings",
            fontsize=13, ha="center", va="center", fontweight="bold")

    fig.tight_layout()
    savefig(fig, "fig9_method_diagram")


# ---------------------------------------------------------------------------
# Figure 10: Bimodality Probe
# ---------------------------------------------------------------------------

def figure_10_bimodality(bare_emb):
    print("\nFigure 10: Bimodality Probe...")

    # Train predicative axis
    phrases = list(PREDICATIVE.keys())
    medians = np.array([PREDICATIVE[p] for p in phrases])
    sentences = [PREDICATIVE_TEMPLATE.replace("{PHRASE}", p.lower()) for p in phrases]
    embs = embed_texts(sentences)
    diffs = embs - bare_emb
    w, slope, intercept = train_axis(diffs, medians)

    # Possible variants (from experiment_08)
    variants = [
        ("barely possible", "low"),
        ("just barely possible", "low"),
        ("remotely possible", "low"),
        ("theoretically possible", "low"),
        ("technically possible", "low"),
        ("just possible", "low"),
        ("possible", "neutral"),
        ("quite possible", "high"),
        ("entirely possible", "high"),
        ("very possible", "high"),
        ("perfectly possible", "high"),
        ("eminently possible", "high"),
    ]

    projections = []
    for phrase, category in variants:
        sent = PREDICATIVE_TEMPLATE.replace("{PHRASE}", phrase)
        emb = embed_texts([sent])[0]
        diff = emb - bare_emb
        prob = float(np.clip(slope * (diff @ w) + intercept, 0, 100))
        projections.append((phrase, prob, category))

    # Sort by projection value
    projections.sort(key=lambda x: x[1])

    fig, ax = plt.subplots(figsize=(8, 5))

    cat_colors = {"low": "#2176AE", "neutral": "#888888", "high": "#C73E1D"}
    cat_labels = {"low": "Low-leaning", "neutral": "Bare form",
                  "high": "High-leaning"}

    y_positions = np.arange(len(projections))
    for i, (phrase, prob, cat) in enumerate(projections):
        ax.barh(i, prob, height=0.6, color=cat_colors[cat],
                edgecolor="white", linewidth=0.5, alpha=0.85)
        # Label on bar or to the right
        if prob > 50:
            ax.text(prob - 1, i, f"{prob:.0f}%", ha="right", va="center",
                    fontsize=9, color="white", fontweight="bold")
        else:
            ax.text(prob + 1, i, f"{prob:.0f}%", ha="left", va="center",
                    fontsize=9, color=cat_colors[cat], fontweight="bold")

    ax.set_yticks(y_positions)
    ax.set_yticklabels([p[0] for p in projections], fontsize=10)
    ax.set_xlabel("Predicted Probability (%)")
    ax.set_title('Bimodality Probe: Modifiers Disambiguate "Possible"\nAlong the Probability Axis')
    ax.set_xlim(0, 85)

    # Mosteller reference line
    ax.axvline(x=38.5, color="#888888", linestyle="--", linewidth=1, alpha=0.7)
    ax.text(39.5, len(projections) - 0.5,
            'Mosteller median\nfor "possible": 38.5%',
            fontsize=8, color="#666666", va="top")

    # Legend
    handles = [plt.Rectangle((0, 0), 1, 1, color=cat_colors[c])
               for c in ["low", "neutral", "high"]]
    ax.legend(handles, [cat_labels[c] for c in ["low", "neutral", "high"]],
              loc="lower right", framealpha=0.9, fontsize=9)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    savefig(fig, "fig10_bimodality")


# ---------------------------------------------------------------------------
# Figure 11: Compound Hedge Composition
# ---------------------------------------------------------------------------

def figure_11_compound(bare_emb):
    print("\nFigure 11: Compound Hedge Composition...")

    # Train modal axis
    MODAL = {
        "certainly": 99.6, "almost certainly": 90.2, "very likely": 87.5,
        "likely": 71.1, "probably": 70.2, "possibly": 38.5,
        "unlikely": 17.2, "very unlikely": 5.0,
        "conceivably": 38.5, "definitely": 99.6, "perhaps": 38.5,
        "maybe": 38.5, "presumably": 70.2, "undoubtedly": 95.0,
        "arguably": 55.0,
    }
    MODAL_TEMPLATE = "The experiment will {PHRASE} succeed"
    m_phrases = list(MODAL.keys())
    m_medians = np.array([MODAL[p] for p in m_phrases])
    m_sents = [MODAL_TEMPLATE.replace("{PHRASE}", p) for p in m_phrases]
    m_embs = embed_texts(m_sents)
    m_diffs = m_embs - bare_emb
    ax_modal, sl_modal, int_modal = train_axis(m_diffs, m_medians)

    # Compound test cases (subset for clean visualization)
    compounds = [
        {
            "label": "I think + probably",
            "compound": "I think the experiment will probably succeed",
            "components": [
                "I think the experiment will succeed",
                "The experiment will probably succeed",
            ],
            "type": "cooperative",
        },
        {
            "label": "I believe + certainly",
            "compound": "I believe the experiment will certainly succeed",
            "components": [
                "I believe the experiment will succeed",
                "The experiment will certainly succeed",
            ],
            "type": "cooperative",
        },
        {
            "label": "It seems + probably",
            "compound": "It seems like the experiment will probably succeed",
            "components": [
                "It seems like the experiment will succeed",
                "The experiment will probably succeed",
            ],
            "type": "cooperative",
        },
        {
            "label": "I suspect + possibly",
            "compound": "I suspect the experiment will possibly succeed",
            "components": [
                "I suspect the experiment will succeed",
                "The experiment will possibly succeed",
            ],
            "type": "cooperative",
        },
        {
            "label": "I doubt + probably",
            "compound": "I doubt the experiment will probably succeed",
            "components": [
                "I doubt the experiment will succeed",
                "The experiment will probably succeed",
            ],
            "type": "conflicting",
        },
        {
            "label": "not + impossible",
            "compound": "It is not impossible that the experiment will succeed",
            "components": [
                "It is impossible that the experiment will succeed",
            ],
            "type": "negation",
        },
    ]

    cosines = []
    mag_ratios = []
    comp_probs = []
    avg_probs = []
    labels = []
    types = []

    for entry in compounds:
        comp_emb = embed_texts([entry["compound"]])[0]
        comp_diff = comp_emb - bare_emb
        comp_embs = embed_texts(entry["components"])
        comp_diffs = comp_embs - bare_emb
        sum_diff = comp_diffs.sum(axis=0)

        cos = float(normalize(comp_diff) @ normalize(sum_diff))
        mag_r = np.linalg.norm(comp_diff) / np.linalg.norm(sum_diff)
        prob_comp = float(np.clip(sl_modal * (comp_diff @ ax_modal) + int_modal, 0, 100))
        prob_parts = [float(np.clip(sl_modal * (d @ ax_modal) + int_modal, 0, 100))
                      for d in comp_diffs]

        cosines.append(cos)
        mag_ratios.append(mag_r)
        comp_probs.append(prob_comp)
        avg_probs.append(np.mean(prob_parts))
        labels.append(entry["label"])
        types.append(entry["type"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # --- Panel A: Direction cosine (compound vs. sum of components) ---
    y = np.arange(len(labels))
    type_colors = {"cooperative": COLORS["pred"], "conflicting": COLORS["noun"],
                   "negation": COLORS["modal"]}
    bar_colors = [type_colors[t] for t in types]

    bars = ax1.barh(y, cosines, height=0.6, color=bar_colors,
                    edgecolor="white", linewidth=0.5)

    for i, (cos_val, label) in enumerate(zip(cosines, labels)):
        if cos_val > 0.5:
            ax1.text(cos_val - 0.02, i, f"{cos_val:.2f}", ha="right",
                     va="center", fontsize=9, color="white", fontweight="bold")
        else:
            ax1.text(cos_val + 0.02, i, f"{cos_val:.2f}", ha="left",
                     va="center", fontsize=9, color=bar_colors[i],
                     fontweight="bold")

    ax1.set_yticks(y)
    ax1.set_yticklabels(labels, fontsize=10)
    ax1.set_xlabel("Cosine Similarity")
    ax1.set_title("(a) Direction: compound vs. sum of parts")
    ax1.set_xlim(-0.1, 1.05)
    ax1.axvline(x=0.8, color="#CCCCCC", linestyle=":", linewidth=0.8)

    handles = [plt.Rectangle((0, 0), 1, 1, color=type_colors[t])
               for t in ["cooperative", "conflicting", "negation"]]
    ax1.legend(handles, ["Cooperative", "Conflicting", "Negation"],
               loc="lower right", framealpha=0.9, fontsize=9)

    # --- Panel B: Compound probability vs. component average ---
    ax2.plot([0, 100], [0, 100], "k--", alpha=0.3, linewidth=1, zorder=1)

    for i in range(len(labels)):
        ax2.scatter(avg_probs[i], comp_probs[i], c=bar_colors[i], s=80,
                    edgecolors="white", linewidth=0.5, zorder=3)
        ax2.annotate(labels[i], (avg_probs[i], comp_probs[i]),
                     xytext=(4, 4), textcoords="offset points",
                     fontsize=8, color="#444444")

    ax2.set_xlabel("Component average probability (%)")
    ax2.set_ylabel("Compound probability (%)")
    ax2.set_title("(b) Probability: sub-linear composition")
    ax2.set_xlim(-5, 105)
    ax2.set_ylim(-5, 105)
    ax2.set_aspect("equal")

    fig.suptitle("Compound Hedge Composition in Embedding Space",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    savefig(fig, "fig11_compound")


# ---------------------------------------------------------------------------
# Figure 12: Cross-Model Axis Alignment
# ---------------------------------------------------------------------------

def figure_12_crossmodel():
    print("\nFigure 12: Cross-Model Consistency...")

    # Hardcoded from FINDINGS-03.md: Modal axis applied to each type
    # Values are Spearman rho (modal axis -> target type)
    models = ["nomic\nv1.5", "gemma\n300m", "MoE\nv2", "mxbai\nlarge", "qwen3\n4096d"]
    types = ["Predicative", "Adverbial", "Noun Phrase", "Modal"]

    # Modal axis LOO rho when applied to each type (from FINDINGS-03 table)
    # These are in-sample rho for cross-type, LOO for modal-on-modal
    modal_rho = np.array([
        # nomic, gemma, moe, mxbai, qwen3
        [0.76, 0.80, 0.63, 0.70, 0.93],   # -> Predicative
        [0.73, 0.87, 0.81, 0.65, 0.93],   # -> Adverbial
        [0.67, 0.77, 0.20, 0.61, 0.62],   # -> Noun Phrase
        [0.90, 0.89, 0.96, 0.80, 0.92],   # -> Modal (native)
    ])

    # Modal axis MAE when applied to each type
    modal_mae = np.array([
        [22.4, 19.2, 16.4, 13.6, 12.1],   # -> Predicative
        [38.5, 42.9, 38.6, 30.1, 32.8],   # -> Adverbial
        [25.5, 26.8, 23.4, 20.4, 19.6],   # -> Noun Phrase
        [8.5,  3.2,  8.2,  8.3,  3.2],    # -> Modal (native)
    ])

    # Inter-type axis alignment (within model)
    # 4 types: Pred, Adv (Freq), NP, Modal
    alignment_models = {
        "nomic v1.5": np.array([
            [1.000, 0.364, 0.248, 0.492],
            [0.364, 1.000, 0.399, 0.458],
            [0.248, 0.399, 1.000, 0.358],
            [0.492, 0.458, 0.358, 1.000],
        ]),
        "mxbai": np.array([
            [1.000, 0.424, 0.469, 0.749],
            [0.424, 1.000, 0.411, 0.559],
            [0.469, 0.411, 1.000, 0.432],
            [0.749, 0.559, 0.432, 1.000],
        ]),
        "qwen3": np.array([
            [1.000, 0.444, 0.770, 0.807],
            [0.444, 1.000, 0.416, 0.583],
            [0.770, 0.416, 1.000, 0.641],
            [0.807, 0.583, 0.641, 1.000],
        ]),
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5),
                                    gridspec_kw={"width_ratios": [1.2, 1]})

    # --- Panel A: Modal axis MAE heatmap across models and types ---
    im = ax1.imshow(modal_mae, cmap="YlGn_r", aspect="auto",
                    vmin=0, vmax=45)

    ax1.set_xticks(range(len(models)))
    ax1.set_xticklabels(models, fontsize=9)
    ax1.set_yticks(range(len(types)))
    ax1.set_yticklabels(types, fontsize=10)
    ax1.set_title("(a) Modal Axis MAE (%) Across Models")
    ax1.spines[:].set_visible(False)
    ax1.grid(False)

    for i in range(len(types)):
        for j in range(len(models)):
            val = modal_mae[i, j]
            color = "white" if val > 25 else "black"
            ax1.text(j, i, f"{val:.1f}", ha="center", va="center",
                     fontsize=10, color=color, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax1, shrink=0.8, label="MAE (%)")

    # --- Panel B: Average inter-type alignment across 3 representative models ---
    avg_alignment = np.mean([alignment_models[m] for m in alignment_models], axis=0)
    type_short = ["Pred", "Adv", "NP", "Modal"]

    im2 = ax2.imshow(avg_alignment, cmap="Blues", aspect="equal",
                     vmin=0.2, vmax=1.0)
    ax2.set_xticks(range(4))
    ax2.set_xticklabels(type_short, fontsize=10)
    ax2.set_yticks(range(4))
    ax2.set_yticklabels(type_short, fontsize=10)
    ax2.set_title("(b) Inter-Type Axis Alignment\n(avg. 3 models)")
    ax2.spines[:].set_visible(False)
    ax2.grid(False)

    for i in range(4):
        for j in range(4):
            val = avg_alignment[i, j]
            color = "white" if val > 0.7 else "black"
            if i != j:
                ax2.text(j, i, f"{val:.2f}", ha="center", va="center",
                         fontsize=10, color=color)

    cbar2 = fig.colorbar(im2, ax=ax2, shrink=0.8, label="Cosine Similarity")

    fig.suptitle("Cross-Model Consistency: 5 Architectures, Same Probability Structure",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    savefig(fig, "fig12_crossmodel")


# ---------------------------------------------------------------------------
# Figure 13: Concept Erasure — ΔMAE vs. cos(v_A, v_B)
# ---------------------------------------------------------------------------

def _parse_exp11_pairs(path):
    """Parse the MAE-based pairwise erasure table from an exp11 results file.

    Returns a list of dicts with keys: pair, cos, mae_b, mae_t, dmae.
    The MAE-based table looks like::

        A→B                          cos |  MAE_b   MAE_t    ΔMAE | ...
        Predicative→Adverbial     +0.424 |   5.47   14.57   +9.10 | ...

    We grab the 12 ordered cross-type pairs by matching the leading "X→Y"
    label (lossy on Unicode arrow handling, so we accept either '→' or '->').
    """
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        in_mae_section = False
        for line in f:
            stripped = line.strip()
            if "MAE-based view" in stripped:
                in_mae_section = True
                continue
            if not in_mae_section:
                continue
            # End of MAE section
            if stripped.startswith("═") or stripped.startswith("CROSS-PAIR"):
                if rows:
                    break
            # Pair lines start with a capitalised type name
            if not stripped or stripped.startswith(("A→B", "---", "MAE", "cos")):
                continue
            # Look for an arrow in the first token group
            if "→" not in stripped and "->" not in stripped:
                continue
            # Split on '|' — this gives us 5 segments; we need segments 0 and 1
            parts = [p.strip() for p in line.split("|")]
            if len(parts) < 5:
                continue
            head = parts[0]  # "Predicative→Adverbial     +0.424"
            mae_part = parts[1]  # "5.47   14.57   +9.10"
            # Extract pair label and cosine
            head_tokens = head.split()
            if len(head_tokens) < 2:
                continue
            pair = head_tokens[0]
            # Cosine is the last token of head
            try:
                cos_val = float(head_tokens[-1])
            except ValueError:
                continue
            mae_tokens = mae_part.split()
            if len(mae_tokens) < 3:
                continue
            try:
                mae_b = float(mae_tokens[0])
                mae_t = float(mae_tokens[1])
                dmae = float(mae_tokens[2])
            except ValueError:
                continue
            rows.append({
                "pair": pair,
                "cos": cos_val,
                "mae_b": mae_b,
                "mae_t": mae_t,
                "dmae": dmae,
            })
    return rows


def figure_13_erasure_mae_cos():
    print("\nFigure 13: Concept Erasure ΔMAE vs. Cosine...")

    base = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "results")
    mxbai_rows = _parse_exp11_pairs(os.path.join(base, "exp11_mxbai.txt"))
    qwen3_rows = _parse_exp11_pairs(os.path.join(base, "exp11_qwen3.txt"))

    if len(mxbai_rows) != 12 or len(qwen3_rows) != 12:
        print(f"  WARNING: parsed {len(mxbai_rows)} mxbai / "
              f"{len(qwen3_rows)} qwen3 rows (expected 12 each)")

    # Reported in results files; recompute as a sanity check.
    cos_m = np.array([r["cos"] for r in mxbai_rows])
    dm_m = np.array([r["dmae"] for r in mxbai_rows])
    cos_q = np.array([r["cos"] for r in qwen3_rows])
    dm_q = np.array([r["dmae"] for r in qwen3_rows])
    r_m = float(np.corrcoef(cos_m, dm_m)[0, 1]) if len(cos_m) > 1 else float("nan")
    r_q = float(np.corrcoef(cos_q, dm_q)[0, 1]) if len(cos_q) > 1 else float("nan")
    print(f"  mxbai r(cos, ΔMAE) = {r_m:+.3f}  (results file: +0.922)")
    print(f"  qwen3 r(cos, ΔMAE) = {r_q:+.3f}  (results file: +0.857)")

    # Color/marker scheme: discrete, colorblind-safe (sampled from viridis)
    # — distinct from any of the rainbow palettes elsewhere in the figure set.
    color_mxbai = "#440154"   # viridis dark-purple
    color_qwen3 = "#1F9E89"   # viridis teal
    marker_mxbai = "o"
    marker_qwen3 = "s"

    fig, ax = plt.subplots(figsize=(7, 5.2))

    # Per-model best-fit lines (transparent dashed) — visualise the r values
    x_grid = np.linspace(0.30, 1.00, 50)
    for cos_v, dm_v, col in [(cos_m, dm_m, color_mxbai),
                             (cos_q, dm_q, color_qwen3)]:
        if len(cos_v) > 1:
            slope, intercept, _, _, _ = stats.linregress(cos_v, dm_v)
            ax.plot(x_grid, slope * x_grid + intercept,
                    color=col, linestyle="--", linewidth=1.2,
                    alpha=0.45, zorder=2)

    # Headline pair predicate: predicative <-> modal
    def is_headline(pair):
        return pair in ("Predicative→Modal", "Modal→Predicative")

    # Plot non-headline points
    for rows, col, mk, label_r in [
        (mxbai_rows, color_mxbai, marker_mxbai, r_m),
        (qwen3_rows, color_qwen3, marker_qwen3, r_q),
    ]:
        non_h = [r for r in rows if not is_headline(r["pair"])]
        ax.scatter([r["cos"] for r in non_h],
                   [r["dmae"] for r in non_h],
                   color=col, marker=mk, s=70,
                   edgecolors="white", linewidth=0.6, zorder=3, alpha=0.85)

    # Highlight headline pairs with larger markers + black edge
    # Use a small label offset table (per model × direction) to avoid overlap
    # at the dense top-right cluster of headline points.
    label_offsets = {
        # (model_key, pair) -> (dx, dy, ha, va)
        ("mxbai", "Predicative→Modal"): (+0.012, -0.4, "left", "top"),
        ("mxbai", "Modal→Predicative"): (+0.012, +0.4, "left", "bottom"),
        ("qwen3", "Predicative→Modal"): (-0.012, -0.4, "right", "top"),
        ("qwen3", "Modal→Predicative"): (-0.012, +0.4, "right", "bottom"),
    }
    for model_key, rows, col, mk in [
        ("mxbai", mxbai_rows, color_mxbai, marker_mxbai),
        ("qwen3", qwen3_rows, color_qwen3, marker_qwen3),
    ]:
        head = [r for r in rows if is_headline(r["pair"])]
        ax.scatter([r["cos"] for r in head],
                   [r["dmae"] for r in head],
                   color=col, marker=mk, s=170,
                   edgecolors="black", linewidth=1.4, zorder=5)
        for r in head:
            direction = "Pred→Modal" if r["pair"] == "Predicative→Modal" \
                else "Modal→Pred"
            dx, dy, ha, va = label_offsets[(model_key, r["pair"])]
            ax.annotate(
                direction,
                (r["cos"], r["dmae"]),
                xytext=(r["cos"] + dx, r["dmae"] + dy),
                fontsize=8.5, color=col, ha=ha, va=va,
                fontweight="bold", zorder=6,
            )

    ax.set_xlabel(r"$\cos(\mathbf{v}_A,\, \mathbf{v}_B)$  (eraser–target axis alignment)")
    ax.set_ylabel(r"$\Delta\mathrm{MAE}_{\mathrm{real}}$  (percentage points)")
    ax.set_title(
        "Concept Erasure: ΔMAE Tracks Inter-Axis Cosine Alignment\n"
        "(12 ordered cross-type pairs per model; predicative ↔ modal called out)"
    )
    ax.set_xlim(0.30, 1.00)
    ax.set_ylim(-1.0, 28.0)

    # Legend with Pearson r values per model + headline marker entry
    legend_handles = [
        Line2D([0], [0], marker=marker_mxbai, linestyle="none",
               markersize=9, markerfacecolor=color_mxbai,
               markeredgecolor="white", markeredgewidth=0.6,
               label=f"mxbai-embed-large (r = {r_m:.2f})"),
        Line2D([0], [0], marker=marker_qwen3, linestyle="none",
               markersize=9, markerfacecolor=color_qwen3,
               markeredgecolor="white", markeredgewidth=0.6,
               label=f"qwen3-embedding (r = {r_q:.2f})"),
        Line2D([0], [0], marker="o", linestyle="none",
               markersize=11, markerfacecolor="#888888",
               markeredgecolor="black", markeredgewidth=1.2,
               label="Headline pair (Pred ↔ Modal)"),
        Line2D([0], [0], color="#888888", linestyle="--", linewidth=1.2,
               alpha=0.6, label="Per-model linear fit"),
    ]
    ax.legend(handles=legend_handles, loc="upper left",
              framealpha=0.92, fontsize=9)

    fig.tight_layout()
    savefig(fig, "fig13_erasure_mae_cos")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print(f"  Generating paper figures")
    print(f"  Model: {MODEL}")
    print(f"  Output: {FIGURE_DIR}")
    print("=" * 60)

    # Test model availability
    try:
        test = embed_texts(["test"])
        dim = test.shape[1]
        print(f"\nModel loaded. Dim: {dim}")
    except Exception as e:
        print(f"\nERROR: Cannot connect to Ollama. Is it running?")
        print(f"  {e}")
        sys.exit(1)

    bare_emb = embed_texts([BARE_CLAIM])[0]

    # Figures requiring live embeddings
    figure_1_vogel(bare_emb)
    figure_2_loo(bare_emb)
    figure_3_baselines(bare_emb)
    figure_6_three_datasets(bare_emb)

    # Figures requiring live embeddings (continued)
    figure_8_pca_confound(bare_emb)
    figure_10_bimodality(bare_emb)
    figure_11_compound(bare_emb)

    # Figures from hardcoded results
    figure_4_crosslingual()
    figure_5_truncation()
    figure_7_null_hypothesis()
    figure_9_method()
    figure_12_crossmodel()
    figure_13_erasure_mae_cos()

    print("\n" + "=" * 60)
    print("  All figures generated successfully!")
    print(f"  Output directory: {FIGURE_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
