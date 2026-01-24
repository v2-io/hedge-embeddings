# Findings 05: Ensemble Axis Combination (Modest Negative)

**Date:** 2026-01-23

**Script:** `experiment_05_ensemble.py`

**Models:** mxbai-embed-large (1024d), qwen3-embedding (4096d), embeddinggemma:300m (768d)

---

## The Question

The oracle (which knows ground truth) picks the best axis per phrase and achieves
MAE 4.3-6.5%. Can we approach oracle performance without knowing the answer,
by combining the 4 axis projections with learned weights?

## Result: MODAL-ONLY REMAINS BEST

No ensemble strategy beats the modal axis alone on novel (out-of-sample) phrases.

| Strategy | mxbai MAE | qwen3 MAE | gemma MAE |
|----------|-----------|-----------|-----------|
| Modal only | **9.7%** | **5.9%** | **5.6%** |
| Average of 4 | 15.9% | 15.4% | 18.8% |
| Magnitude-weighted | 14.6% | 14.3% | 16.2% |
| Max-projection | 10.6% | 9.1% | 18.8% |
| Linear ensemble (ridge) | 11.4% | 8.3% | 7.5% |
| Logistic (calibrated) | 15.4% | 14.3% | 14.5% |
| **Oracle** | **5.9%** | **4.3%** | **4.5%** |

Gap from oracle: 1.1-3.8 percentage points. The modal axis alone captures
most of the achievable accuracy.

---

## Strategies Tested

1. **Modal only**: Use only the modal axis with linear calibration. No training
   beyond the axis itself.

2. **Average of 4**: Mean of the 4 linear probability estimates. Naive but
   parameter-free.

3. **Magnitude-weighted**: Weight each axis's estimate by the absolute
   projection magnitude. The intuition: strong projection = high relevance.

4. **Max-projection**: Use the axis with the highest |projection|. Hard routing
   based on signal strength.

5. **Linear ensemble (ridge)**: Ridge regression on the 4 calibrated probability
   estimates. 5 parameters (4 weights + bias), fitted on the 58 Mosteller
   training phrases.

6. **Logistic (calibrated)**: Sigmoid applied to a linear combination of the 4
   calibrated probability estimates. Same 5 parameters, but output bounded in
   (0%, 100%) via the sigmoid.

7. **Oracle**: Cheating — picks the axis whose prediction is closest to truth.

---

## Why the Logistic Approach Failed

The sigmoid function compresses extreme values toward 50%. But the linear
calibration already produces correct extreme predictions (99.6% for "definitely",
0.3% for "impossible") when the right axis is used. The logistic curve REMOVES
information that the linear mapping preserves.

The compression is visible in the predictions: logistic-calibrated outputs
cluster in the 40-60% range while modal-only correctly reaches 95%+ for
certainty markers.

**Lesson:** Logistic regression is appropriate when the raw features are
unbounded and need to be mapped to probabilities. Here, the features (calibrated
axis predictions) are ALREADY calibrated probabilities. The sigmoid adds
unwanted distortion.

---

## Why the Linear Ensemble Didn't Help

The learned weights reveal the problem:

| Model | Pred weight | Freq weight | NP weight | Modal weight | Bias |
|-------|-------------|-------------|-----------|-------------|------|
| mxbai | +0.42 | +0.49 | +0.18 | +0.16 | -1.5 |
| qwen3 | +1.16 | +0.22 | -0.68 | +0.21 | +12.4 |
| gemma | +0.22 | +0.23 | +0.22 | +0.57 | -0.6 |

The weights are unstable across models. Only gemma correctly identifies the
modal axis as most important (weight 0.57). The others put more weight on
predicative and frequency axes, reflecting the training data composition
(predicative: 13, frequency: 19, noun phrase: 11, modal: 15). The ensemble
overfits to the training distribution.

**Per-group training performance confirms this:**

On predicative and frequency phrases (which dominate training), the linear
ensemble matches or slightly beats modal-only. On modal phrases (the novel
test distribution), modal-only wins by 2-3%.

---

## Cross-Context Generalization

On qwen3, the linear ensemble does SLIGHTLY beat modal-only for cross-context
predicative phrases:
- "The treatment will be effective": Linear 5.3% vs Modal 6.4%
- "The prediction will be correct": Linear 8.8% vs Modal 10.7%

This is the one scenario where ensemble adds value: predicative phrases embedded
in unfamiliar contexts get a small boost from combining axis predictions. But the
improvement (1-2%) doesn't justify the added complexity.

---

## What This Means for the Practical System

The axis selection problem from FINDINGS-03 ("Remaining Gaps #1") is now
partially resolved — but not through ensemble methods:

**The modal axis IS the practical solution.** For a production calibration system:

1. Use the modal axis as the universal fallback
2. If you can detect that a phrase is predicative ("It is {X} that..."), use
   the predicative axis (slight improvement for that syntactic type)
3. No ensemble is needed — the modal axis alone is within 1-4% of oracle
   performance

**Why this works:** Most natural hedging uses modal adverbs ("probably",
"certainly", "possibly"), first-person frames ("I think", "I believe"), or
clause-level modifiers that are semantically closest to the modal adverb space.
Predicative adjectives ("it is likely that") and frequency adverbs ("always",
"rarely") are rarer in natural text.

---

## Remaining Gap Analysis

The remaining 1-4% gap from oracle comes from specific failure modes:

1. **"Conceivably"** — projects to 50-66% on all models (Mosteller: 35%).
   No axis or ensemble helps; this word may genuinely encode higher probability
   in adverbial position.

2. **"I'm not sure" / "I doubt"** — these are best on the predicative axis
   (negation-like syntax) but the ensemble doesn't learn to route them there.

3. **Anti-hedges in sentence-initial position** — "Obviously, X" projects to
   75-88% on the modal axis but should be 95%. The remaining gap is a ceiling
   on what single-axis projection can achieve.

For these edge cases, a rule-based routing system (detect negation → predicative;
detect sentence-initial → rewrite) would likely be more effective than a learned
ensemble. The data is too sparse for the ensemble to learn these patterns.
