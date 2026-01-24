# Findings 04: IQR-Consistency Prediction (Negative Result)

**Date:** 2026-01-23

**Script:** `experiment_04_iqr_consistency.py`

**Models:** mxbai-embed-large (1024d), qwen3-embedding (4096d),
nomic-embed-text-v2-moe (768d), embeddinggemma:300m (768d)

---

## The Prediction

Phrases with low IQR (high human consensus about meaning) should produce more
geometrically consistent difference vectors across diverse claims than phrases
with high IQR (high human disagreement). This would mean the embedding geometry
encodes interpretive precision — not just what a phrase means on average, but
how stable that meaning is.

## Result: NOT SUPPORTED

| Model (dim) | ρ(consistency, -IQR) | p-value |
|-------------|---------------------|---------|
| embeddinggemma (768d) | -0.041 | 0.89 |
| nomic-v2-moe (768d) | +0.028 | 0.93 |
| mxbai-embed-large (1024d) | +0.047 | 0.88 |
| qwen3-embedding (4096d) | -0.195 | 0.52 |

No significant relationship between IQR and geometric consistency on any model.
Tested across 4 architectures spanning 768-4096 dimensions.

---

## Full Output: mxbai-embed-large (1024d)

```
Phrase               IQR   Median  Consistency   Tier
Impossible           0.3     0.3%       0.709    1
Certain              1.1    99.6%       0.691    1
Very improbable      5.9     4.8%       0.769    2
Very unlikely        7.1     5.0%       0.681    2
Almost certain       7.5    90.2%       0.704    2
Very probable        8.9    89.7%       0.675    2
Very likely         10.1    87.5%       0.683    2
Probable            13.0    70.2%       0.638    2
Unlikely            13.0    17.2%       0.653    2
Improbable          14.7    12.5%       0.754    2
Likely              15.0    71.1%       0.656    2
Not unreasonable    29.1    37.6%       0.745    3
Possible            42.7    38.5%       0.715    3

Primary: ρ(consistency, -IQR) = +0.047 (p = 0.88)
Extremity confound:  ρ(consistency, extremity) = +0.193
After partialing:    ρ(consistency, -IQR | extremity) = +0.047
Median correlation:  ρ(consistency, median) = -0.368

Tier comparison:
  Tier 1 (IQR<5):   mean consistency = 0.700
  Tier 2 (5≤IQR≤20): mean consistency = 0.690
  Tier 3 (IQR>20):  mean consistency = 0.730
```

## Full Output: qwen3-embedding (4096d)

```
Phrase               IQR   Median  Consistency   Tier
Impossible           0.3     0.3%       0.569    1
Certain              1.1    99.6%       0.660    1
Very improbable      5.9     4.8%       0.555    2
Very unlikely        7.1     5.0%       0.530    2
Almost certain       7.5    90.2%       0.626    2
Very probable        8.9    89.7%       0.638    2
Very likely         10.1    87.5%       0.598    2
Probable            13.0    70.2%       0.664    2
Unlikely            13.0    17.2%       0.526    2
Improbable          14.7    12.5%       0.541    2
Likely              15.0    71.1%       0.585    2
Not unreasonable    29.1    37.6%       0.605    3
Possible            42.7    38.5%       0.669    3

Primary: ρ(consistency, -IQR) = -0.195 (p = 0.52)
Extremity confound:  ρ(consistency, extremity) = -0.308
After partialing:    ρ(consistency, -IQR | extremity) = +0.165
Median correlation:  ρ(consistency, median) = +0.654 (p = 0.015)
```

## Full Output: nomic-embed-text-v2-moe (768d)

```
Phrase               IQR   Median  Consistency   Tier
Impossible           0.3     0.3%       0.778    1
Certain              1.1    99.6%       0.706    1
Very improbable      5.9     4.8%       0.780    2
Very unlikely        7.1     5.0%       0.777    2
Almost certain       7.5    90.2%       0.735    2
Very probable        8.9    89.7%       0.733    2
Very likely         10.1    87.5%       0.732    2
Probable            13.0    70.2%       0.713    2
Unlikely            13.0    17.2%       0.774    2
Improbable          14.7    12.5%       0.775    2
Likely              15.0    71.1%       0.715    2
Not unreasonable    29.1    37.6%       0.781    3
Possible            42.7    38.5%       0.766    3

Primary: ρ(consistency, -IQR) = +0.028 (p = 0.93)
Extremity confound:  ρ(consistency, extremity) = +0.107
After partialing:    ρ(consistency, -IQR | extremity) = +0.028
Median correlation:  ρ(consistency, median) = -0.802 (p = 0.001)
```

## Full Output: embeddinggemma:300m (768d)

```
Phrase               IQR   Median  Consistency   Tier
Impossible           0.3     0.3%       0.734    1
Certain              1.1    99.6%       0.653    1
Very improbable      5.9     4.8%       0.758    2
Very unlikely        7.1     5.0%       0.705    2
Almost certain       7.5    90.2%       0.714    2
Very probable        8.9    89.7%       0.681    2
Very likely         10.1    87.5%       0.666    2
Probable            13.0    70.2%       0.683    2
Unlikely            13.0    17.2%       0.691    2
Improbable          14.7    12.5%       0.748    2
Likely              15.0    71.1%       0.670    2
Not unreasonable    29.1    37.6%       0.738    3
Possible            42.7    38.5%       0.708    3

Primary: ρ(consistency, -IQR) = -0.041 (p = 0.89)
Extremity confound:  ρ(consistency, extremity) = +0.050
After partialing:    ρ(consistency, -IQR | extremity) = -0.041
Median correlation:  ρ(consistency, median) = -0.703 (p = 0.007)
```

---

## Interpretation

### What the negative result means:

The embedding model encodes a **single, stable meaning** for each phrase,
regardless of how much humans disagree about what it means. "Possible"
(bimodal distribution in Mosteller, IQR 42.7) has just as consistent a
geometric direction as "Certain" (tight consensus, IQR 1.1).

This tells us:
1. The model learns ONE interpretation per phrase, not a distribution
2. Human interpretive variance (IQR) is a property of the POPULATION, not of
   the language itself — the geometry captures language, not population variance
3. If we want to encode interpretive precision in the calibration system, we
   must bring it in externally (from the IQR data), not read it from geometry
4. The Tier 1/2/3 classification from `docs/verbal-probability-calibration.md`
   remains essential — it provides information the geometry cannot

### Which interpretation did the models choose for "possible/possibly"?

"Possible" is bimodal in humans: one group interprets it as logical possibility
("not impossible" ≈ 5-10%), another as reasonable chance ("decent probability"
≈ 40-50%). The Mosteller P25=7.5% and P75=50.2% reflect these two clusters.

On the modal axis (Experiment 3), "possibly" projections are size-dependent:

| Model (dim) | "possibly" projection | Error from Mosteller 38.5% |
|-------------|----------------------|---------------------------|
| nomic-v1.5 (768d) | 54.2% | +15.7% |
| nomic-v2-moe (768d) | 48.6% | +10.1% |
| embeddinggemma (768d) | 43.7% | +5.2% |
| mxbai-embed-large (1024d) | 45.3% | +6.8% |
| **qwen3-embedding (4096d)** | **38.6%** | **+0.1%** |

The larger model (qwen3) recovers the Mosteller median exactly. The smaller
models show varying degrees of upward bias toward the "reasonable chance"
interpretation. This suggests the "logical possibility" meaning of "possibly"
is a more subtle semantic distinction that requires higher representational
capacity to preserve.

Notably, "perhaps" shows the same pattern: 49.1% (nomic) → 49.7% (MoE) →
36.8% (gemma) → 46.6% (mxbai) → 38.5% (qwen3). The gemma and qwen3 models
both recover near-Mosteller values, while the nomic variants are biased high.

A calibration system should be aware that smaller embedding models will
systematically overestimate the probability conveyed by "possibly" and
"perhaps" — treating them as moderate probability rather than as the weaker
epistemic markers they often are.

### Secondary finding: median-consistency correlation (dimensionality-dependent)

The correlation between consistency and Mosteller median FLIPS depending on
model dimensionality:

| Model (dim) | ρ(consistency, median) | p-value | Direction |
|-------------|----------------------|---------|-----------|
| nomic-v2-moe (768d) | **-0.802** | 0.001 | uncertainty more consistent |
| embeddinggemma (768d) | **-0.703** | 0.007 | uncertainty more consistent |
| mxbai-embed-large (1024d) | -0.368 | ns | intermediate |
| qwen3-embedding (4096d) | **+0.654** | 0.015 | certainty more consistent |

In lower-dimensional spaces (768d), low-probability phrases produce more
geometrically consistent directions across claims. In higher-dimensional space
(4096d), the pattern reverses: high-probability phrases are more consistent.

Possible explanation: In 768 dimensions, high-probability phrases ("certain",
"very likely", "probable") compete for representational capacity — there are more
fine-grained distinctions to encode among certainty expressions, leading to more
context-sensitive directions. Low-probability phrases are semantically simpler
("won't happen" modifies all sentences similarly regardless of content). In 4096
dimensions, there is enough capacity for certainty distinctions to be encoded
stably, and instead the lower-probability phrases show more content-sensitivity
(perhaps because what counts as "unlikely" depends more on domain — unlikely
for a surgery vs. unlikely for a prediction engage different semantic facets).

### Dimensionality affects absolute consistency

Absolute consistency values decrease with model dimensionality:
- nomic-v2-moe (768d): 0.71-0.78
- embeddinggemma (768d): 0.65-0.76
- MXBAI (1024d): 0.64-0.77
- Qwen3 (4096d): 0.53-0.67

All are well above zero (the direction is meaningful in all cases). Higher-
dimensional models allow more content-specific variation in difference vectors —
they can encode subtle interactions between the hedge phrase and what is being
hedged. This is not noise: the MoE model (768d) shows the HIGHEST consistency
range while also achieving excellent modal axis calibration (MAE = 8.2%),
suggesting its consistency comes from architectural specialization, not reduced
capacity.

---

## What this tells us about the overall approach

The geometry encodes:
- ✓ What probability a phrase implies (Experiments 1-3)
- ✓ The relative ordering of phrases by probability (ρ > 0.84)
- ✓ Enough structure for calibrated extraction (MAE ~8% with right axis)
- ✗ How ambiguous/precise the phrase is to human interpreters

This means a practical calibration system needs TWO information sources:
1. **Geometric projection** → probability level (what the phrase means)
2. **Mosteller IQR lookup** → interpretive precision (how ambiguous the phrase
   is across people)

The geometry replaces phrase-lookup for probability estimation but not for
ambiguity flagging. The tier system from the verbal-probability-calibration
guide remains essential for the second dimension.
