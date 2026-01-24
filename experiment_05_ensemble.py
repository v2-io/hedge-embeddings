"""
Experiment 05: Ensemble Axis Combination with Logistic Calibration

The axis selection problem: we have 4 trained probability axes (predicative,
frequency, noun phrase, modal), each optimal for its own syntactic type. The
oracle (which knows ground truth) picks the best axis per phrase and achieves
MAE 4.3-6.5%. Can we approach oracle performance without knowing the answer?

This experiment tests ensemble strategies that combine all 4 axis projections:

1. Modal only (current best single-axis fallback)
2. Average of 4 linear predictions (naive ensemble)
3. Magnitude-weighted average (weight by |projection|)
4. Max-projection routing (use axis with highest |projection|)
5. Logistic regression on 4 projections (soft ensemble with bounded output)
6. Logistic + magnitude features (8 features: projections + |projections|)
7. Oracle (cheating — picks best axis per phrase)

The logistic approach is the key innovation: by fitting
    prob = sigmoid(w · projections + bias)
we get:
    - Naturally bounded output in (0%, 100%)
    - Non-linear calibration that handles extremes better
    - Implicit soft axis selection via learned weights
    - Only 5 parameters — well-determined with ~58 training samples

Training data: All Mosteller-calibrated phrases from all 4 syntactic types,
projected onto all 4 axes. Cross-validated via leave-one-out.

Test data: Novel phrases from Experiments 02/03 (not in training set).

References:
    - experiment_03_modal_axis.py (4-axis infrastructure)
    - FINDINGS-03.md (oracle vs. single-axis results)
"""

import sys
import numpy as np
import requests
from scipy import stats
from scipy.optimize import minimize

MODEL = sys.argv[1] if len(sys.argv) > 1 else "nomic-embed-text:v1.5"
OLLAMA_URL = "http://localhost:11434/api/embed"


def embed_texts(texts, model=MODEL):
    resp = requests.post(OLLAMA_URL, json={"model": model, "input": texts})
    resp.raise_for_status()
    return np.array(resp.json()["embeddings"], dtype=np.float64)


def normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def sigmoid(x):
    """Numerically stable sigmoid."""
    return np.where(x >= 0,
                    1 / (1 + np.exp(-x)),
                    np.exp(x) / (1 + np.exp(x)))


def logit(p):
    """Logit transform, clipping to avoid infinities."""
    p = np.clip(p, 0.005, 0.995)
    return np.log(p / (1 - p))


def train_axis(expressions, template, bare_emb):
    """Train supervised probability axis. Returns (axis, slope, intercept)."""
    phrases = list(expressions.keys())
    medians = np.array([expressions[p] for p in phrases])
    sentences = [template.replace("{PHRASE}", p.lower()) for p in phrases]
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
# Training data: all 4 syntactic groups
# ---------------------------------------------------------------------------

BARE_CLAIM = "The experiment will succeed"

PREDICATIVE = {
    "Certain": 99.6, "Almost certain": 90.2, "Very likely": 87.5,
    "Likely": 71.1, "Probable": 70.2, "Very probable": 89.7,
    "Possible": 38.5, "Unlikely": 17.2, "Very unlikely": 5.0,
    "Improbable": 12.5, "Very improbable": 4.8, "Impossible": 0.3,
    "Not unreasonable": 37.6,
}
PREDICATIVE_TEMPLATE = "It is {PHRASE} that the experiment will succeed"

FREQUENCY = {
    "Always": 99.7, "Almost always": 91.7, "Very often": 82.8,
    "Often": 72.5, "Usually": 75.1, "Sometimes": 25.0,
    "Occasionally": 20.0, "Seldom": 10.2, "Very seldom": 4.9,
    "Rarely": 7.2, "Very rarely": 3.0, "Almost never": 2.9,
    "Never": 0.3, "Not often": 19.7, "Not very often": 10.1,
    "As often as not": 50.0, "More often than not": 59.8,
    "Once in a while": 15.3, "Now and then": 15.1,
}
FREQUENCY_TEMPLATE = "The experiment will {PHRASE} succeed"

NOUN_PHRASE = {
    "Very high probability": 92.5, "High probability": 82.3,
    "Moderate probability": 52.4, "Low probability": 15.0,
    "Very low probability": 4.9, "High chance": 80.4,
    "Poor chance": 10.3, "Low chance": 9.8, "Even chance": 50.0,
    "Better than even chance": 57.6, "Less than an even chance": 40.2,
}
NOUN_PHRASE_TEMPLATE = "There is a {PHRASE} that the experiment will succeed"

# Modal adverb axis: Mosteller-derived + estimated values
MODAL_ADVERB = {
    "certainly": 99.6, "almost certainly": 90.2, "very likely": 87.5,
    "likely": 71.1, "probably": 70.2, "possibly": 38.5,
    "unlikely": 17.2, "very unlikely": 5.0,
    "conceivably": 38.5, "definitely": 99.6, "perhaps": 38.5,
    "maybe": 38.5, "presumably": 70.2, "undoubtedly": 95.0,
    "arguably": 55.0,
}
MODAL_TEMPLATE = "The experiment will {PHRASE} succeed"


# ---------------------------------------------------------------------------
# Novel phrases for out-of-sample testing
# (same phrases from experiments 02/03, with intuitive probability estimates)
# ---------------------------------------------------------------------------

NOVEL_PHRASES = [
    # Modal adverbs (should work best on modal axis)
    ("The experiment will probably succeed", "modal", 70.2),
    ("The experiment will certainly succeed", "modal", 99.6),
    ("The experiment will possibly succeed", "modal", 38.5),
    ("The experiment will likely succeed", "modal", 71.1),
    ("The experiment will definitely succeed", "modal", 99.6),
    ("The experiment will perhaps succeed", "modal", 38.5),
    ("The experiment will presumably succeed", "modal", 70.0),
    ("The experiment will conceivably succeed", "modal", 35.0),
    ("The experiment will undoubtedly succeed", "modal", 95.0),
    ("The experiment will arguably succeed", "modal", 55.0),

    # First-person frames
    ("I think the experiment will succeed", "first_person", 65.0),
    ("I believe the experiment will succeed", "first_person", 70.0),
    ("I suspect the experiment will succeed", "first_person", 55.0),
    ("I doubt the experiment will succeed", "first_person", 20.0),
    ("I'm not sure the experiment will succeed", "first_person", 40.0),

    # Anti-hedges
    ("Obviously, the experiment will succeed", "anti_hedge", 95.0),
    ("Clearly, the experiment will succeed", "anti_hedge", 92.0),
    ("Without question, the experiment will succeed", "anti_hedge", 98.0),

    # Novel phrasings
    ("All indications suggest the experiment will succeed", "novel", 80.0),
    ("It's a safe bet that the experiment will succeed", "novel", 85.0),
    ("There's a slim chance the experiment will succeed", "novel", 15.0),
    ("The experiment might well succeed", "novel", 60.0),
    ("I wouldn't be surprised if the experiment succeeded", "novel", 55.0),
]

# Cross-context validation claims
CROSS_CONTEXTS = [
    {
        "bare": "The treatment will be effective",
        "predicative": "It is {PHRASE} that the treatment will be effective",
        "frequency": "The treatment will {PHRASE} be effective",
        "noun_phrase": "There is a {PHRASE} that the treatment will be effective",
        "modal": "The treatment will {PHRASE} be effective",
    },
    {
        "bare": "The prediction will be correct",
        "predicative": "It is {PHRASE} that the prediction will be correct",
        "frequency": "The prediction will {PHRASE} be correct",
        "noun_phrase": "There is a {PHRASE} that the prediction will be correct",
        "modal": "The prediction will {PHRASE} be correct",
    },
]


# ---------------------------------------------------------------------------
# Ensemble strategies
# ---------------------------------------------------------------------------

def strategy_modal_only(projections, axes_params, _weights=None):
    """Use only the modal axis with linear calibration."""
    sl, intc = axes_params[3]  # modal is index 3
    return np.clip(sl * projections[:, 3] + intc, 0, 100)


def strategy_average(projections, axes_params, _weights=None):
    """Average the 4 linear predictions."""
    preds = np.zeros((len(projections), 4))
    for i in range(4):
        sl, intc = axes_params[i]
        preds[:, i] = sl * projections[:, i] + intc
    return np.clip(preds.mean(axis=1), 0, 100)


def strategy_magnitude_weighted(projections, axes_params, _weights=None):
    """Weight each axis's prediction by the absolute projection magnitude."""
    preds = np.zeros((len(projections), 4))
    weights = np.abs(projections)
    for i in range(4):
        sl, intc = axes_params[i]
        preds[:, i] = sl * projections[:, i] + intc
    # Normalize weights per sample
    weight_sums = weights.sum(axis=1, keepdims=True)
    weight_sums = np.where(weight_sums == 0, 1, weight_sums)
    weights_normed = weights / weight_sums
    result = (preds * weights_normed).sum(axis=1)
    return np.clip(result, 0, 100)


def strategy_max_projection(projections, axes_params, _weights=None):
    """Use the axis with the highest absolute projection."""
    preds = np.zeros((len(projections), 4))
    for i in range(4):
        sl, intc = axes_params[i]
        preds[:, i] = sl * projections[:, i] + intc
    best_idx = np.argmax(np.abs(projections), axis=1)
    return np.clip(preds[np.arange(len(preds)), best_idx], 0, 100)


def strategy_logistic_raw(projections, _axes_params, weights):
    """Logistic regression on raw projections: prob = sigmoid(w·proj + bias)*100."""
    w, bias = weights[:-1], weights[-1]
    logits = projections @ w + bias
    return sigmoid(logits) * 100


def strategy_linear_ensemble(projections, axes_params, weights):
    """Linear combination of calibrated probabilities (ridge regression).
    weights = [w1, w2, w3, w4, bias]
    """
    # First compute calibrated probabilities from each axis
    probs = np.zeros((len(projections), 4))
    for i in range(4):
        sl, intc = axes_params[i]
        probs[:, i] = sl * projections[:, i] + intc
    w, bias = weights[:4], weights[4]
    result = probs @ w + bias
    return np.clip(result, 0, 100)


def strategy_logistic_calibrated(projections, axes_params, weights):
    """Logistic applied to calibrated probabilities (sigmoid on probs, not raw projections).
    prob = sigmoid(w · calibrated_probs + bias) * 100
    """
    probs = np.zeros((len(projections), 4))
    for i in range(4):
        sl, intc = axes_params[i]
        probs[:, i] = sl * projections[:, i] + intc
    # Normalize probs to [0, 1] range for logistic input
    probs_normed = probs / 100.0
    w, bias = weights[:4], weights[4]
    logits = probs_normed @ w + bias
    return sigmoid(logits) * 100


def strategy_oracle(projections, axes_params, _weights, targets):
    """Cheating: pick the axis closest to ground truth per phrase."""
    preds = np.zeros((len(projections), 4))
    for i in range(4):
        sl, intc = axes_params[i]
        preds[:, i] = np.clip(sl * projections[:, i] + intc, 0, 100)
    errors = np.abs(preds - targets[:, np.newaxis])
    best_idx = np.argmin(errors, axis=1)
    return preds[np.arange(len(preds)), best_idx]


# ---------------------------------------------------------------------------
# Logistic regression fitting
# ---------------------------------------------------------------------------

def fit_logistic(X, y_prob, lam=0.01):
    """
    Fit logistic regression: prob = sigmoid(w·X + bias) * 100.
    Minimizes MSE in probability space with L2 regularization.

    X: (n, d) feature matrix
    y_prob: (n,) target probabilities in [0, 100]
    lam: L2 regularization strength

    Returns: parameter array [w1, ..., wd, bias]
    """
    d = X.shape[1]
    y = y_prob / 100.0  # normalize to [0, 1]

    def loss(params):
        w, bias = params[:d], params[d]
        logits = X @ w + bias
        pred = sigmoid(logits)
        mse = np.mean((pred - y) ** 2)
        reg = lam * np.sum(w ** 2)
        return mse + reg

    def grad(params):
        w, bias = params[:d], params[d]
        logits = X @ w + bias
        pred = sigmoid(logits)
        d_logits = 2 / len(y) * (pred - y) * pred * (1 - pred)
        d_w = X.T @ d_logits + 2 * lam * w
        d_bias = d_logits.sum()
        return np.concatenate([d_w, [d_bias]])

    x0 = np.zeros(d + 1)
    result = minimize(loss, x0, jac=grad, method='L-BFGS-B',
                      options={'maxiter': 5000, 'ftol': 1e-12})
    return result.x


def fit_linear_ensemble(X, y_prob, lam=0.1):
    """
    Ridge regression: prob = w·X + bias.
    X should be calibrated probabilities from each axis.

    Returns: parameter array [w1, ..., wd, bias]
    """
    n, d = X.shape
    # Add bias column
    X_aug = np.hstack([X, np.ones((n, 1))])
    # Ridge: (X'X + λI)^-1 X'y, with no regularization on bias
    reg = lam * np.eye(d + 1)
    reg[-1, -1] = 0  # don't regularize bias
    params = np.linalg.solve(X_aug.T @ X_aug + reg, X_aug.T @ y_prob)
    return params


def calibrated_features(projections, axes_params):
    """Convert raw projections to calibrated probability estimates."""
    probs = np.zeros((len(projections), 4))
    for i in range(4):
        sl, intc = axes_params[i]
        probs[:, i] = sl * projections[:, i] + intc
    return probs


def leave_one_out_evaluate(X, y_prob, fit_fn, predict_fn, lam=0.01):
    """Leave-one-out cross-validation. Returns array of LOO predictions."""
    n = len(y_prob)
    loo_preds = np.zeros(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        X_train, y_train = X[mask], y_prob[mask]
        weights = fit_fn(X_train, y_train, lam)
        X_test = X[i:i+1]
        loo_preds[i] = predict_fn(X_test, weights)
    return loo_preds


def predict_linear(X, weights):
    """Predict using fitted linear ensemble weights."""
    X_aug = np.hstack([X, np.ones((len(X), 1))])
    return np.clip((X_aug @ weights)[0], 0, 100)


def predict_logistic_cal(X, weights):
    """Predict using fitted logistic on calibrated features."""
    X_normed = X / 100.0
    w, bias = weights[:X.shape[1]], weights[X.shape[1]]
    logits = X_normed @ w + bias
    return sigmoid(logits)[0] * 100


def fit_logistic_calibrated(X, y_prob, lam=0.01):
    """Fit logistic on calibrated prob features (normalized to [0,1])."""
    X_normed = X / 100.0
    return fit_logistic(X_normed, y_prob, lam)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print()
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║  Experiment 05: Ensemble Axis Combination                          ║")
    print("╠══════════════════════════════════════════════════════════════════════╣")
    print(f"║  Model: {MODEL:<58s}║")
    print("║                                                                    ║")
    print("║  Question: Can a logistic ensemble of 4 axes approach oracle       ║")
    print("║  performance without knowing the correct axis per phrase?          ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print()

    try:
        test = embed_texts(["test"])
        dim = test.shape[1]
        print(f"Model loaded. Dim: {dim}\n")
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    bare_emb = embed_texts([BARE_CLAIM])[0]

    # --- Train 4 axes ---
    print("Training axes...")
    axes = []
    axes_params = []  # (slope, intercept) for each axis

    for name, exprs, template in [
        ("Predicative", PREDICATIVE, PREDICATIVE_TEMPLATE),
        ("Frequency", FREQUENCY, FREQUENCY_TEMPLATE),
        ("Noun phrase", NOUN_PHRASE, NOUN_PHRASE_TEMPLATE),
        ("Modal", MODAL_ADVERB, MODAL_TEMPLATE),
    ]:
        w, sl, intc = train_axis(exprs, template, bare_emb)
        axes.append(w)
        axes_params.append((sl, intc))
        print(f"  {name:12s}: slope={sl:7.1f}, intercept={intc:5.1f}")

    axes_matrix = np.array(axes)  # (4, dim)

    # --- Build training set: all Mosteller phrases projected onto all 4 axes ---
    print("\nBuilding training set...")

    all_phrases = []  # (phrase_text, template_key, median)
    groups = [
        ("predicative", PREDICATIVE, PREDICATIVE_TEMPLATE),
        ("frequency", FREQUENCY, FREQUENCY_TEMPLATE),
        ("noun_phrase", NOUN_PHRASE, NOUN_PHRASE_TEMPLATE),
        ("modal", MODAL_ADVERB, MODAL_TEMPLATE),
    ]

    for group_name, exprs, template in groups:
        for phrase, median in exprs.items():
            sentence = template.replace("{PHRASE}", phrase.lower())
            all_phrases.append((sentence, group_name, phrase, median))

    # Embed all training phrases
    all_sentences = [p[0] for p in all_phrases]
    all_embeddings = embed_texts(all_sentences)
    all_diffs = all_embeddings - bare_emb

    # Project onto all 4 axes
    # X: (n_phrases, 4) — the 4 projections per phrase
    X_train = all_diffs @ axes_matrix.T
    y_train = np.array([p[3] for p in all_phrases])
    group_labels = [p[1] for p in all_phrases]

    n_train = len(all_phrases)
    print(f"  {n_train} phrases from 4 groups, projected onto 4 axes")
    print(f"  Target range: {y_train.min():.1f}% – {y_train.max():.1f}%")

    # --- Compute calibrated features for ensemble training ---
    X_cal = calibrated_features(X_train, axes_params)  # (n, 4) in [0, 100] range

    # --- Fit ensembles ---
    print("\nFitting ensembles on calibrated probability features...")

    # Linear ensemble (ridge on calibrated probs)
    weights_linear = fit_linear_ensemble(X_cal, y_train, lam=0.1)
    print(f"  Linear ensemble: weights = [{', '.join(f'{w:+.4f}' for w in weights_linear[:4])}]")
    print(f"    bias = {weights_linear[4]:+.2f}")

    # Logistic on calibrated probs
    weights_logcal = fit_logistic_calibrated(X_cal, y_train, lam=0.01)
    print(f"  Logistic (calibrated): weights = [{', '.join(f'{w:+.3f}' for w in weights_logcal[:4])}]")
    print(f"    bias = {weights_logcal[4]:+.3f}")

    # --- Evaluate all strategies on training data ---
    print("\n" + "═" * 78)
    print("TRAINING SET PERFORMANCE (in-sample, n={})".format(n_train))
    print("═" * 78 + "\n")

    strategies = [
        ("Modal only (linear)", lambda P, AP, W: strategy_modal_only(P, AP, W)),
        ("Average (4 linear)", lambda P, AP, W: strategy_average(P, AP, W)),
        ("Magnitude-weighted", lambda P, AP, W: strategy_magnitude_weighted(P, AP, W)),
        ("Max-projection", lambda P, AP, W: strategy_max_projection(P, AP, W)),
        ("Linear ensemble", lambda P, AP, W: strategy_linear_ensemble(P, AP, weights_linear)),
        ("Logistic (calibrated)", lambda P, AP, W: strategy_logistic_calibrated(P, AP, weights_logcal)),
        ("Oracle", lambda P, AP, W: strategy_oracle(P, AP, W, y_train)),
    ]

    print(f"  {'Strategy':<25s}  {'ρ':>6s}  {'MAE':>6s}  {'Max err':>7s}  Notes")
    print(f"  {'─'*25}  {'─'*6}  {'─'*6}  {'─'*7}  {'─'*20}")

    for name, fn in strategies:
        preds = fn(X_train, axes_params, None)
        r, _ = stats.spearmanr(preds, y_train)
        mae = np.mean(np.abs(preds - y_train))
        max_err = np.max(np.abs(preds - y_train))
        note = ""
        if "Oracle" in name:
            note = "(cheating)"
        elif "Logistic" in name:
            note = "(in-sample)"
        print(f"  {name:<25s}  {r:+.3f}  {mae:5.1f}%  {max_err:6.1f}%  {note}")

    # --- Leave-one-out cross-validation for logistic strategies ---
    print("\n" + "═" * 78)
    print("LEAVE-ONE-OUT CROSS-VALIDATION (n={})".format(n_train))
    print("═" * 78 + "\n")

    print("  Computing LOO predictions for ensemble strategies...")

    loo_linear = leave_one_out_evaluate(X_cal, y_train, fit_linear_ensemble,
                                         predict_linear, lam=0.1)
    loo_logcal = leave_one_out_evaluate(X_cal, y_train, fit_logistic_calibrated,
                                         predict_logistic_cal, lam=0.01)

    print()
    print(f"  {'Strategy':<25s}  {'ρ':>6s}  {'MAE':>6s}  {'Max err':>7s}")
    print(f"  {'─'*25}  {'─'*6}  {'─'*6}  {'─'*7}")

    # Non-ensemble strategies (no LOO needed — no fitted params beyond axes)
    for name, fn in strategies[:4]:  # modal, average, mag-weighted, max-proj
        preds = fn(X_train, axes_params, None)
        r, _ = stats.spearmanr(preds, y_train)
        mae = np.mean(np.abs(preds - y_train))
        max_err = np.max(np.abs(preds - y_train))
        print(f"  {name:<25s}  {r:+.3f}  {mae:5.1f}%  {max_err:6.1f}%")

    r_lin, _ = stats.spearmanr(loo_linear, y_train)
    mae_lin = np.mean(np.abs(loo_linear - y_train))
    max_lin = np.max(np.abs(loo_linear - y_train))
    print(f"  {'Linear ensemble (LOO)':<25s}  {r_lin:+.3f}  {mae_lin:5.1f}%  {max_lin:6.1f}%")

    r_lc, _ = stats.spearmanr(loo_logcal, y_train)
    mae_lc = np.mean(np.abs(loo_logcal - y_train))
    max_lc = np.max(np.abs(loo_logcal - y_train))
    print(f"  {'Logistic-cal (LOO)':<25s}  {r_lc:+.3f}  {mae_lc:5.1f}%  {max_lc:6.1f}%")

    # Oracle for comparison
    oracle_preds = strategy_oracle(X_train, axes_params, None, y_train)
    r_o, _ = stats.spearmanr(oracle_preds, y_train)
    mae_o = np.mean(np.abs(oracle_preds - y_train))
    max_o = np.max(np.abs(oracle_preds - y_train))
    print(f"  {'Oracle (upper bound)':<25s}  {r_o:+.3f}  {mae_o:5.1f}%  {max_o:6.1f}%")

    # --- Per-group breakdown for best LOO strategy ---
    print("\n" + "═" * 78)
    print("PER-GROUP BREAKDOWN (LOO linear ensemble)")
    print("═" * 78 + "\n")

    for group in ["predicative", "frequency", "noun_phrase", "modal"]:
        mask = np.array([g == group for g in group_labels])
        if mask.sum() == 0:
            continue
        group_linear = loo_linear[mask]
        group_targets = y_train[mask]
        r_l, _ = stats.spearmanr(group_linear, group_targets)
        mae_l = np.mean(np.abs(group_linear - group_targets))
        # Also show modal-only for comparison
        modal_preds = strategy_modal_only(X_train[mask], axes_params, None)
        mae_m = np.mean(np.abs(modal_preds - group_targets))
        print(f"  {group:15s} (n={mask.sum():2d}):  Linear MAE={mae_l:5.1f}%  "
              f"Modal MAE={mae_m:5.1f}%  Δ={mae_l-mae_m:+5.1f}%")

    # --- Novel phrase evaluation ---
    print("\n" + "═" * 78)
    print("NOVEL PHRASE EVALUATION (out-of-sample, n={})".format(len(NOVEL_PHRASES)))
    print("═" * 78 + "\n")

    novel_sentences = [p[0] for p in NOVEL_PHRASES]
    novel_embs = embed_texts(novel_sentences)
    novel_diffs = novel_embs - bare_emb
    novel_projs = novel_diffs @ axes_matrix.T  # (n_novel, 4)
    novel_targets = np.array([p[2] for p in NOVEL_PHRASES])
    novel_cats = [p[1] for p in NOVEL_PHRASES]

    # Predictions from each strategy (using weights fitted on full training set)
    novel_preds = {}
    novel_preds["Modal only"] = strategy_modal_only(novel_projs, axes_params, None)
    novel_preds["Average"] = strategy_average(novel_projs, axes_params, None)
    novel_preds["Mag-weighted"] = strategy_magnitude_weighted(novel_projs, axes_params, None)
    novel_preds["Max-proj"] = strategy_max_projection(novel_projs, axes_params, None)
    novel_preds["Linear ens."] = strategy_linear_ensemble(novel_projs, axes_params, weights_linear)
    novel_preds["Logistic-cal"] = strategy_logistic_calibrated(novel_projs, axes_params, weights_logcal)
    novel_preds["Oracle"] = strategy_oracle(novel_projs, axes_params, None, novel_targets)

    print(f"  {'Strategy':<25s}  {'ρ':>6s}  {'MAE':>6s}  {'Max err':>7s}")
    print(f"  {'─'*25}  {'─'*6}  {'─'*6}  {'─'*7}")
    for name in ["Modal only", "Average", "Mag-weighted", "Max-proj",
                 "Linear ens.", "Logistic-cal", "Oracle"]:
        preds = novel_preds[name]
        r, _ = stats.spearmanr(preds, novel_targets)
        mae = np.mean(np.abs(preds - novel_targets))
        max_err = np.max(np.abs(preds - novel_targets))
        print(f"  {name:<25s}  {r:+.3f}  {mae:5.1f}%  {max_err:6.1f}%")

    # --- Per-category breakdown for novel phrases ---
    print("\n  Per-category (Linear ensemble vs. Modal only):")
    print(f"  {'Category':<15s}  {'Modal MAE':>9s}  {'Lin. MAE':>9s}  {'Δ':>6s}")
    for cat in ["modal", "first_person", "anti_hedge", "novel"]:
        mask = np.array([c == cat for c in novel_cats])
        if mask.sum() == 0:
            continue
        mae_modal = np.mean(np.abs(novel_preds["Modal only"][mask] - novel_targets[mask]))
        mae_lin = np.mean(np.abs(novel_preds["Linear ens."][mask] - novel_targets[mask]))
        delta = mae_lin - mae_modal
        print(f"  {cat:<15s}  {mae_modal:8.1f}%  {mae_lin:8.1f}%  {delta:+5.1f}%")

    # --- Detailed predictions for novel phrases ---
    print("\n" + "═" * 78)
    print("DETAILED NOVEL PREDICTIONS (Linear ens. vs. Modal only)")
    print("═" * 78 + "\n")

    lin_preds = novel_preds["Linear ens."]
    modal_preds = novel_preds["Modal only"]
    print(f"  {'Phrase':<48s} {'Lin':>5s} {'Modal':>5s} {'Target':>6s} {'Err-L':>5s}")
    print(f"  {'─'*48} {'─'*5} {'─'*5} {'─'*6} {'─'*5}")

    current_cat = None
    for i, (sentence, cat, target) in enumerate(NOVEL_PHRASES):
        if cat != current_cat:
            current_cat = cat
            print(f"  --- {cat} ---")
        short = sentence[:45] + "..." if len(sentence) > 48 else sentence
        err = abs(lin_preds[i] - target)
        print(f"  {short:<48s} {lin_preds[i]:5.1f} {modal_preds[i]:5.1f} "
              f"{target:5.1f}% {err:4.1f}%")

    # --- Cross-context generalization ---
    print("\n" + "═" * 78)
    print("CROSS-CONTEXT: Does the ensemble generalize to different claims?")
    print("═" * 78 + "\n")

    # Use a subset of Mosteller phrases that exist in all groups for testing
    # (predicative expressions, projected in a different context)
    for ctx in CROSS_CONTEXTS:
        ctx_bare_emb = embed_texts([ctx["bare"]])[0]
        print(f"  Context: \"{ctx['bare']}\"")

        # Embed predicative phrases in this context
        ctx_phrases = []
        for phrase, median in list(PREDICATIVE.items())[:5]:  # subset for speed
            sentence = ctx["predicative"].replace("{PHRASE}", phrase.lower())
            ctx_phrases.append((sentence, median))

        ctx_sentences = [p[0] for p in ctx_phrases]
        ctx_embs = embed_texts(ctx_sentences)
        ctx_diffs = ctx_embs - ctx_bare_emb
        ctx_projs = ctx_diffs @ axes_matrix.T
        ctx_targets = np.array([p[1] for p in ctx_phrases])

        # Compare strategies
        ctx_modal = strategy_modal_only(ctx_projs, axes_params, None)
        ctx_linear = strategy_linear_ensemble(ctx_projs, axes_params, weights_linear)

        mae_m = np.mean(np.abs(ctx_modal - ctx_targets))
        mae_l = np.mean(np.abs(ctx_linear - ctx_targets))
        print(f"    Modal-only MAE: {mae_m:5.1f}%,  Linear ens. MAE: {mae_l:5.1f}%")

        for i, (sent, med) in enumerate(ctx_phrases):
            short_phrase = list(PREDICATIVE.keys())[i]
            print(f"      {short_phrase:20s}  Modal={ctx_modal[i]:5.1f}%  "
                  f"Lin.={ctx_linear[i]:5.1f}%  Target={med:5.1f}%")
        print()

    # --- Summary ---
    print("═" * 78)
    print("SUMMARY")
    print("═" * 78 + "\n")

    print("  Linear ensemble weights:")
    axis_names = ["Pred", "Freq", "NP", "Modal"]
    for i, name in enumerate(axis_names):
        print(f"    {name:8s}: {weights_linear[i]:+.4f}")
    print(f"    {'Bias':8s}: {weights_linear[4]:+.2f}")
    print()

    best_novel_name = min(["Modal only", "Mag-weighted", "Max-proj",
                           "Linear ens.", "Logistic-cal"],
                          key=lambda n: np.mean(np.abs(novel_preds[n] - novel_targets)))
    best_novel_mae = np.mean(np.abs(novel_preds[best_novel_name] - novel_targets))
    oracle_mae = np.mean(np.abs(novel_preds["Oracle"] - novel_targets))

    print(f"  Best non-oracle strategy (novel phrases): {best_novel_name}")
    print(f"    MAE = {best_novel_mae:.1f}% (oracle = {oracle_mae:.1f}%)")
    print(f"    Gap from oracle: {best_novel_mae - oracle_mae:.1f}%")
    print()

    # Did the ensemble beat modal-only?
    mae_modal_novel = np.mean(np.abs(novel_preds["Modal only"] - novel_targets))
    mae_lin_novel = np.mean(np.abs(novel_preds["Linear ens."] - novel_targets))
    mae_logcal_novel = np.mean(np.abs(novel_preds["Logistic-cal"] - novel_targets))
    best_ens = min(mae_lin_novel, mae_logcal_novel)
    best_ens_name = "Linear ensemble" if mae_lin_novel <= mae_logcal_novel else "Logistic-calibrated"
    if best_ens < mae_modal_novel:
        print(f"  RESULT: {best_ens_name} IMPROVES over modal-only by "
              f"{mae_modal_novel - best_ens:.1f}% MAE on novel phrases.")
    else:
        print(f"  RESULT: Neither ensemble improves over modal-only.")
        print(f"    Modal: {mae_modal_novel:.1f}%,  Linear: {mae_lin_novel:.1f}%,  "
              f"Logistic: {mae_logcal_novel:.1f}%")
    print()


if __name__ == "__main__":
    main()
