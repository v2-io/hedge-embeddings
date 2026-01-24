# Findings 06: Compound Hedge Composition

**Date:** 2026-01-23

**Script:** `experiment_06_compound_hedges.py`

**Models:** qwen3-embedding (4096d), mxbai-embed-large (1024d)

---

## The Question

Natural language often stacks multiple hedge cues: "I think it will probably
work", "It seems fairly likely", "I doubt it will probably succeed."

Do compound hedges decompose linearly in embedding space? That is, does
diff("I think probably X") ≈ diff("I think X") + diff("probably X")?

## Result: YES (with caveats)

| Metric | qwen3 | mxbai |
|--------|-------|-------|
| Mean cosine (direction) | 0.82 | 0.89 |
| ρ (compound vs. component avg) | +0.71 | +0.96 |
| MAE (compound vs. component avg) | 10.0% | 9.4% |
| ρ (compound vs. intuition) | +0.79 | +0.91 |
| MAE (compound vs. intuition) | 8.2% | 12.8% |

Compound hedges compose **approximately linearly in direction** but with
**sub-linear magnitude** (compound magnitude / sum magnitude ≈ 0.70-0.87).

---

## Key Findings

### 1. Direction Linearity Is Strong

For most compound types, the compound difference vector points in approximately
the same direction as the sum of individual component vectors:

| Compound type | qwen3 cos | mxbai cos |
|---------------|-----------|-----------|
| First-person + modal | 0.92-0.96 | 0.97-0.99 |
| Evidential + modal | 0.96 | 0.97 |
| Attenuator + predicative | 0.66-0.92 | 0.84-0.97 |
| Attenuator + modal | 0.76 | 0.85 |
| Double first-person | 0.88 | 0.96 |
| **Negation + predicative** | **0.22** | **0.64** |

First-person + modal combinations are nearly perfectly linear (cos > 0.92).
This means the model treats "I think" and "probably" as independent semantic
contributions that simply add together.

### 2. Negation Is the Exception

"It is not impossible that X" does NOT decompose as negation + impossibility.
On qwen3, the cosine drops to 0.22 — the compound direction is essentially
unrelated to the "impossible" direction. This makes sense: negation doesn't
just scale or shift the impossibility vector — it creates a qualitatively
different meaning (litotes: understated possibility).

Practical implication: a calibration system cannot handle "not impossible" by
inverting the "impossible" axis. It needs to be treated as its own semantic
entity.

### 3. Sub-Linear Magnitude

The compound difference vector is always SHORTER than the sum of components:

- Typical ratio: 0.70-0.87 (qwen3), 0.60-0.90 (mxbai)
- One exception: "It is fairly likely" on qwen3 has ratio 1.39 (the "fairly"
  frame adds more structure than "likely" alone)

This sub-linearity is semantically correct: "I think probably" SHOULD convey
less total modification than the sum of "I think" + "probably" separately.
Multiple hedge cues partially overlap in semantic content — they don't
independently stack to full effect.

### 4. Conflicting Hedges: The More Specific Cue Dominates

"I doubt the experiment will probably succeed":
- Component average: 35-39% (average of "doubt" ~12-20% and "probably" ~61-65%)
- Actual compound: 11-21%
- "Doubt" dominates decisively over "probably"

The model gives precedence to the more specific, emotionally-charged hedge
cue ("doubt") over the neutral probabilistic marker ("probably"). This
matches linguistic intuition: "I doubt" frames the entire subsequent clause
as skepticism, overriding the affirmative force of "probably".

### 5. "Quite" Is an Intensifier, Not an Attenuator

"The experiment will quite possibly succeed" → 59-67% (vs "possibly" alone: 39-45%)

Both models treat "quite" as INCREASING the probability of "possibly," consistent
with the British English intensive usage ("quite good" = "really good") rather
than the American English attenuative usage ("quite good" = "only fairly good").
This is a training-data bias worth noting for production systems.

### 6. "Almost Certainly" Is Correctly Attenuated

"It is almost certainly true that X" → 82-90% (Mosteller "Almost certain": 90.2%)

Both models correctly handle this complex construction: "almost" attenuates
"certainly" to produce a value near Mosteller's independent calibration for
"almost certain." This shows the geometry captures the attenuating semantics
of "almost" even in a multi-word construction.

---

## Practical Implications

For a production calibration system handling compound hedges:

1. **Simple projection works**: Just embed the compound sentence and project
   onto the modal axis. The result is already reasonable (MAE 8-13% vs intuition).
   No decomposition needed.

2. **Component-average as sanity check**: If you CAN decompose the compound into
   components, their average probability predicts the compound's projection well
   (ρ = 0.71-0.96).

3. **Negation requires special handling**: "Not X" cannot be handled by any
   linear operation on the "X" embedding. Detect negation and treat the negated
   expression as a distinct lookup.

4. **Conflicting cues**: When hedge cues conflict (doubt + probably), the
   compound projects closer to the MORE EXTREME component. The system should
   be aware that doubt-like markers override probability markers.

---

## What This Means for the Geometric Hypothesis

The linear composition finding strengthens the overall thesis from
`docs/epistemic-geometry-hedging-as-linear-structure.md`:

**Hedging is not just a linear DIRECTION — it's a linear ALGEBRA.** Multiple
hedge cues add approximately as vectors, with the result being a direction
that's the vector sum of individual contributions (scaled down in magnitude).
This is stronger than the original hypothesis, which only predicted a single
linear direction for hedging. The space actually supports vector arithmetic
on epistemic modifiers.

The exception (negation) is linguistically principled: negation is not a
scaling operation on the original meaning — it creates a qualitatively different
meaning. The geometry correctly encodes this distinction.

---

## Appendix: Negation Geometry (Quick Probe)

A follow-up test asked: is there a consistent "negation vector" in embedding
space? If diff("not X") - diff("X") is the same for all X, then negation IS
a linear operation (just one that's orthogonal to probability).

**Result: Negation is NOT a single direction (mean pairwise cosine of negation
vectors = -0.06).** But it IS highly structured:

| Pair | Cosine |
|------|--------|
| not-likely vs not-probable | **+0.96** |
| not-possible vs not-probable | +0.78 |
| not-possible vs not-impossible | **-0.93** |
| not-likely vs not-unlikely | **-0.89** |
| not-impossible vs not-unlikely | +0.77 |

The pattern: negating a high-probability word (certain, likely, probable) produces
vectors that are mutually aligned (+0.39 to +0.96) but ANTI-CORRELATED with
vectors from negating low-probability words (impossible, unlikely). The cosines
between these groups are -0.63 to -0.93.

**Interpretation:** Negation acts as a **probability-reflecting** operation.
It pushes the meaning toward the opposite side of the probability spectrum:
- "Not certain" pushes DOWN from 99% toward center
- "Not impossible" pushes UP from 0% toward center
- The two push in OPPOSITE directions

There is no single "not" vector. Instead, negation is context-dependent in
a structured way: it inverts the probability direction of the negated word.
This is why "not impossible" (cos 0.22 with the "impossible" direction in
Experiment 06) creates a seemingly unrelated direction — it's pushing in the
mirror-image direction from where "impossible" would push.

**Practical implication:** A calibration system cannot handle negation by
adding a fixed vector. Negated hedges must be treated as distinct lexical
items, or the system must detect negation and approximately invert the
probability estimate (reflect through ~50%).
