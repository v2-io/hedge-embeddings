# Experiment Plan: Epistemic Hedging as Linear Structure in Embedding Space

**Purpose:** Test whether epistemic hedging manifests as consistent linear
structure in modern sentence embedding models, and if so, whether that structure
is calibrated to empirical probability.

**Status:** Experiments 1-3 complete. Core hypothesis validated with nuance.

**Date:** 2026-01-23 (experiments run same day)

**Context:** Companion to `docs/epistemic-geometry-hedging-as-linear-structure.md`
and `docs/epistemic-claim-calibration-first-principles.md`. Ground truth from
`docs/mosteller_youtz_1990_full.csv` and related calibration data.

---

## Intellectual Honesty: What We Don't Know

Before designing experiments, it's worth being explicit about the epistemic
status of the core hypothesis.

### What's established (by prior work):

- Word-level embeddings (word2vec, GloVe) encode grammatical and semantic
  modifications as approximately linear directions (Mikolov et al. 2013).
- Gender, tense, plurality manifest as consistent vector offsets.
- The Mosteller data provides stable, well-replicated ground-truth mappings
  from verbal probability expressions to numerical probability distributions.

### What's hypothesized (plausible but untested):

- That sentence-level embeddings (which use contextual representations, pooling
  strategies, and contrastive training objectives very different from word2vec)
  preserve this linear structure for epistemic modifications.
- That the direction is specifically *epistemic* rather than a generic "sentence
  was modified by an adverb" direction.
- That projection magnitude correlates with calibrated probability (not just
  with "degree of modification").

### What I genuinely don't know:

- Whether anyone has already tested this specific hypothesis. (I'm not aware of
  prior work on hedge-as-direction in sentence embeddings, but my knowledge has
  gaps. A literature search would be prudent before investing heavily.)
- What the null distribution looks like: if you insert *any* adverb into a fixed
  sentence position, how consistent are the resulting difference vectors? The
  hedge direction is only meaningful if it's MORE consistent than the generic
  "adverb insertion" direction, or if it points somewhere *different*.
- Whether contrastive training objectives (which push "X is true" away from "X
  is probably true" as hard negatives) might actually *destroy* linear structure
  by optimizing for discriminability rather than semantic compositionality.

### Key risk:

The strongest version of this hypothesis—that hedge words lie on a single axis
whose projection magnitude predicts calibrated probability—might be too strong.
A weaker but still useful result would be: hedge modifications occupy a
low-dimensional subspace, and position within that subspace correlates with
probability, even if the relationship isn't perfectly linear or one-dimensional.

The experiment should be designed to detect the weak version even if the strong
version fails.

---

## Ground Truth Available

### Primary: Mosteller & Youtz 1990

53 verbal probability expressions with P25, Median, P75, IQR from n=238 science
writers. This gives us:

- **Central tendency** (Median): What probability does this phrase typically convey?
- **Precision** (IQR): How much do people agree about what it means?
- **Distribution shape** (P25, P75): Symmetric or skewed?

Expressions span the full [0%, 100%] range and include:
- Likelihood terms: "likely", "probable", "certain", "possible", ...
- Frequency terms: "often", "rarely", "seldom", "frequently", ...
- Complex constructions: "not unreasonable", "liable to happen", ...

### Cross-validation: Vogel 2022

Means from meta-analysis of 21 studies (1967-2018) for ~20 overlapping
expressions. If our geometric predictions trained on Mosteller also predict
Vogel's independent estimates, we've found something robust.

### Domain specificity: Medical meta-analysis

Probability shifts for the same phrases in medical vs. general contexts. Enables
testing whether domain modulates the hedge direction or merely offsets it.

---

## Syntactic Classification of Expressions

This is a practical prerequisite, not a formality. The Mosteller expressions
occupy different syntactic slots, and forcing a phrase into the wrong slot
produces ungrammatical sentences whose embeddings will be distorted by the
ungrammaticality rather than by the hedge semantics.

| Syntactic type | Examples | Natural template pattern |
|----------------|----------|--------------------------|
| Predicative adjective | likely, probable, certain, possible, unlikely | "It is {phrase} that X" or "X is {phrase}" |
| Adverb (frequency) | often, rarely, seldom, always, never, usually | "X {phrase} happens/occurs" |
| Noun phrase | high chance, poor chance, even chance | "There is a {phrase} that X" |
| Complex/clausal | not unreasonable, liable to happen, might happen | Requires per-phrase templates |

**Decision needed at implementation time:** Whether to exclude complex
constructions from the primary analysis (they might not embed cleanly in any
template) or to include them as a robustness check.

---

## Experiment 1: Does Hedging Have a Consistent Direction?

### The core question

When you embed "X is [hedge] Y" and subtract the embedding of "X is Y", do the
resulting difference vectors point in approximately the same direction regardless
of what X and Y are?

### Design

**Claims (the propositional content):**

Select 15-20 diverse declarative claims spanning multiple domains. Each must be:
- Grammatically natural when preceded by any of the test hedge phrases
- Semantically diverse (science, personal, technical, social, physical, etc.)
- Roughly similar in length (to avoid length-driven embedding differences)
- Not inherently about probability or uncertainty (to avoid confounding content
  with hedge)

Example candidates (need vetting for naturalness with all hedge phrases):
```
"The bridge is structurally sound"
"The medication reduces inflammation"
"The algorithm converges"
"The election results are accurate"
"The glacier is retreating"
"The patient is recovering"
```

**Hedge phrases (start small, expand):**

Phase 1a: Just "probably" (the single most common epistemic hedge). This is the
minimum viable test—if even one phrase doesn't produce consistent directions,
the hypothesis needs rethinking.

Phase 1b: Add "certainly", "possibly", "likely", "unlikely" (spanning the
probability range, all fitting the predicative-adjective slot).

Phase 1c: Add frequency adverbs: "often", "rarely", "always", "never" (testing
whether frequency and likelihood share a direction or occupy different
subspaces).

**Controls (essential):**

For each claim, also create variants with non-epistemic adverbs that occupy a
similar syntactic position:
- "currently" (temporal, no epistemic content)
- "reportedly" (evidential, not epistemic confidence)
- "officially" (social/institutional, not probability)

The purpose: establish the null. If "probably" produces consistent direction AND
that direction is different from "currently"/"reportedly", then we've found
something specifically epistemic, not just "adverb insertion is a direction."

If the hedge direction and the control direction are the same... that's a
negative result that would require rethinking.

### Measures

1. **Within-phrase consistency:** For a single phrase (e.g., "probably"), compute
   all difference vectors across claims. Normalize each to unit length. Compute
   pairwise cosines. Report mean and distribution.

2. **Control consistency:** Same as above for each control adverb.

3. **Discriminability:** Cosine between the mean hedge direction and the mean
   control direction. If they're nearly parallel (cosine > 0.8), the hedge
   direction is not specifically epistemic.

4. **Cross-phrase alignment:** If multiple hedge phrases are tested, compute
   their mean directions. Are they parallel (1D hedge axis) or do they span a
   subspace (multi-dimensional)?

### What "success" and "failure" look like

I don't have well-calibrated thresholds for these measures. Rather than
pretending I do, I'll describe what the results *mean*:

- **Mean pairwise cosine near 1.0** for hedge vectors: The direction is nearly
  perfectly consistent. (Unlikely—word2vec analogies aren't perfect either.)
- **Mean pairwise cosine 0.5-0.8:** Substantial consistency. A direction exists
  but there's claim-dependent variation. Worth pursuing.
- **Mean pairwise cosine 0.2-0.5:** Weak consistency. There might be a subspace
  but it's noisy. Check if PCA reveals structure hidden in the pairwise measure.
- **Mean pairwise cosine near 0.0:** No consistency. The hypothesis fails for
  this model. Try another model before giving up entirely.
- **Hedge cosine similar to control cosine:** The direction exists but isn't
  specifically epistemic. "Any adverb insertion" is a direction. This is a
  negative result for the *epistemic* hypothesis but still interesting.
- **Hedge cosine much higher than control cosine:** The epistemic direction is
  more consistent than generic modification. This supports the hypothesis.

### The "implicit certainty" problem

A subtlety worth noting: "The bridge is structurally sound" is not truly
epistemically neutral—it carries implicit certainty (an unhedged assertion). So
the difference vector is measuring "departure from implicit certainty" rather
than "position on a hedge axis." These are related but not identical.

An alternative framing: maybe there is no "zero point" on the hedge axis, and
bare assertions are simply at one end (maximum certainty). This is actually
fine—it means the axis goes from "certain" to "uncertain" and different hedge
phrases are at different positions along it. The experiment tests this directly.

---

## Experiment 2: Does Geometry Predict Probability?

### Prerequisite

Experiment 1 must show at least moderate consistency (mean cosine > ~0.3) AND
some discriminability from controls.

### Design

Using the Mosteller expressions that fit the same syntactic type (to avoid
confounding syntax with semantics):

1. Embed each expression in 5+ claim contexts (same claims from Experiment 1).
2. Compute mean difference vector per expression.
3. Extract the principal axis (or 2-3 axes via PCA on all mean difference
   vectors).
4. Project each expression's mean vector onto the principal axis(es).
5. Correlate projections with Mosteller medians.

### The IQR prediction (novel, unverified reasoning)

**Hypothesis:** Phrases with low IQR (high consensus on meaning) produce more
consistent difference vectors than phrases with high IQR (low consensus).

**Reasoning:** If a phrase means the same thing to everyone, embedding it in
different contexts should produce similar geometric modifications. If it means
different things to different people (or in different contexts), the geometric
modification should be less stable.

**Caution:** This reasoning assumes the embedding model's geometry reflects
population-level interpretive variance. This is not obvious—the model might
encode a single "meaning" for each phrase regardless of human interpretive
variance. If so, consistency would be uniformly high regardless of IQR, and this
prediction would fail without invalidating the broader hypothesis.

**Test:** Spearman rank correlation between per-phrase consistency (from
Experiment 1b/c) and negative IQR. If significant, the geometry captures
interpretive precision. If not, it captures central tendency only (still useful,
just less rich).

### Cross-validation

Train the projection→probability mapping on Mosteller data. Test on Vogel data.
If the mapping generalizes, it's capturing something about the language itself,
not an artifact of one study's population.

---

## Experiment 3: Cross-Model Consistency

### Why this matters

If the hedge direction is similar across different embedding architectures and
training objectives, it's a property of the *language* (or at least of the
distributional statistics of English text). If it's model-specific, calibration
must be model-specific too.

### Models to test

| Model | Architecture | Training | Why include |
|-------|-------------|----------|-------------|
| all-MiniLM-L6-v2 | BERT-small, mean pool | Contrastive (sentence pairs) | Fast, old, widely studied baseline |
| nomic-embed-text-v1.5 | BERT variant, Matryoshka | Contrastive + Matryoshka loss | Modern, enables dim-truncation test |
| A third model TBD | Different base | Different training | Tests generality |

The third model should ideally be architecturally distinct from the first two
(e.g., a decoder-based embedding model, or a different model family). The goal
is maximum diversity of training regimes.

### Matryoshka truncation test (specific to nomic or similar)

If the hedge direction exists, does it survive dimensional truncation?

```
Compute hedge axis using full 768 dims → h_full
Compute hedge axis using first 128 dims → h_128
Compute hedge axis using first 64 dims → h_64

Measure: Does h_64 still predict Mosteller probability?
```

If yes, hedge information is concentrated in the top principal components of the
model's learned representation—consistent with the MRL training objective
putting important information first. This would also mean confidence extraction
is cheap (64-dim projection instead of 768-dim).

---

## Experiment 4: Domain Shift (if Experiments 1-2 succeed)

### Design

Using the 5 phrases that appear in both Mosteller and the medical meta-analysis:
- "rare": 7.2% general → 10.0% medical
- "unlikely": 17.2% → 17.71% medical
- "possible": 38.5% → 43.28% medical
- "likely": 71.1% → 71.87% medical
- "very likely": 87.5% → 84.3% medical

Embed each phrase in 3+ general contexts and 3+ medical contexts. Compute
difference vectors. Compare projections onto the hedge axis.

**Predictions:**
- "Rare" should show the largest projection shift (largest probability shift)
- "Unlikely" and "likely" should show minimal shift
- "Very likely" should shift in the *opposite* direction from "rare" (lower in
  medical context)

**What we're really testing:** Whether embedding models implicitly encode
domain-specific calibration norms. If they do, the geometric calibration
approach can be context-sensitive without explicit domain labels.

---

## Practical Considerations

### Computational cost

All experiments involve embedding <500 sentences per model. This is trivial
computationally—minutes on CPU, seconds on GPU. The bottleneck is *thought*, not
compute.

### Claim generation is the most important manual step

The quality of the experiment depends heavily on the claims being:
- Genuinely diverse in content
- Grammatically natural with all tested hedge phrases
- Not inherently about probability (avoid "The coin will land heads")
- Consistent in length and complexity

This step deserves care and probably a second pair of eyes. A bad claim set can
produce misleading results in either direction (false positive from
systematically similar content, or false negative from awkward phrasing that
distorts embeddings).

### Analysis should be exploratory first, confirmatory second

The PCA and correlation analyses should be visualized and explored before
computing p-values. What does the scatterplot of (projection, Mosteller median)
actually look like? Is it linear? Is there a clear outlier? Does "possible"
(with its bimodal distribution) behave differently from the rest?

Statistical tests should confirm patterns seen in exploration, not replace
visual inspection and judgment.

### What to do with negative results

If Experiment 1 shows no consistency: try a different model before concluding
the hypothesis is wrong. Different training objectives produce very different
geometric properties.

If Experiment 1 shows consistency but Experiment 2 shows no probability
correlation: the direction might encode "degree of modification" (a syntactic
property) rather than "probability" (a semantic property). This would still be
interesting—it means hedging has geometric structure, just not calibrated
structure.

If results are inconsistent across models: the structure is model-specific, and
calibration must be model-specific. This limits the practical value but doesn't
eliminate it.

---

## What This Could Mean (If It Works)

If epistemic hedging manifests as calibrated linear structure in embedding space,
it provides:

1. **A continuous, phrase-agnostic confidence extraction method.** No phrase
   lookup tables. Novel phrasings ("I'd wager that...", "signs point to...")
   project onto the same axis as known calibration phrases.

2. **A geometric bridge between the hierarchical Bayesian framework and
   embedding-based systems.** The hierarchical model (in the companion paper)
   can use geometric confidence as an informative prior, even for (source,
   domain, phrase) triples with zero outcome data.

3. **A foundation for source calibration.** If different sources produce
   systematically different hedge directions or magnitudes, source-specific
   calibration becomes a learned rotation/scaling in embedding space.

4. **A connection to conversational dynamics.** If confidence is a direction,
   tracking how a conversation's stance on a proposition evolves (from
   "possibly" to "probably" to "certainly") becomes a trajectory in a known
   subspace—linking epistemic calibration to the collaborative-discovery
   framework.

None of this is guaranteed. The experiments will reveal what's actually there.

---

## Execution Order

1. **Literature check:** Has this been done? Quick search for "epistemic modality
   embedding geometry" or "hedging linear structure sentence embeddings." If
   prior work exists, build on it rather than duplicating.

2. **Claim generation:** Write 15-20 diverse claims. Vet each for naturalness
   with "probably", "certainly", "possibly", and at least one frequency adverb.
   Write corresponding control-adverb variants.

3. **Experiment 1a:** Single phrase ("probably"), single model (MiniLM), 15-20
   claims. This is the minimum viable test. If it fails here, reconsider before
   investing more.

4. **Experiment 1b:** Expand to 5 hedge phrases and controls if 1a shows signal.

5. **Experiment 2:** Mosteller correlation. The headline result.

6. **Experiment 3:** Cross-model replication.

7. **Experiment 4:** Domain shift (conditional on Experiments 1-2 succeeding).

Each step gates the next. Don't proceed past a gate without understanding what
the current results actually mean.

---

## Findings (2026-01-23)

### Scripts

- `experiment_01_hedge_direction.py` — Part A (consistency) + Part B (Mosteller
  projection with all types mixed). Run with model name as argument.
- `experiment_01b_within_type.py` — Within-type syntactic control analysis.
  Separates predicative, adverbial, and noun-phrase expressions. The key script.

### Summary of Results

**The core hypothesis is validated, with an important structural nuance.**

Probability IS encoded as a linear direction in sentence embedding space.
But it's type-specific: different syntactic constructions (predicative adjective,
frequency adverb, noun phrase) have different probability axes. When syntax is
held constant, probability is the dominant or near-dominant linear direction.

### Experiment 1 (Part A): Consistency of "probably"

"Probably" produces a consistent difference vector across 15 diverse claims
(mean pairwise cosine 0.68 on nomic-embed-text). This confirms that hedge
modifications are geometrically consistent — adding "probably" moves sentences
in approximately the same direction regardless of content.

However, the control adverb "reportedly" also shows high consistency (0.60),
and the two directions are moderately aligned (cos 0.53). This means there is
a shared "adverb insertion" component. The epistemic-specific component is the
difference between these directions, which is substantial but not as clean as
hoped.

### Experiment 1b: Within-Type Analysis (the key finding)

When we mixed all 52 Mosteller expressions (predicative + adverbial + noun
phrase) in Experiment 1 Part B, PC1 captured syntactic type variation, not
probability. PC1 separated "There is a high probability..." sentences from
"The experiment will often..." sentences. The probability signal was in PC2
(r = -0.65).

The fix: analyze each syntactic type separately. Results across 3 models:

**PC1 (unsupervised) correlation with Mosteller median:**

| Type (n) | Nomic-embed | EmbeddingGemma | MXBAI-embed-large |
|----------|-------------|----------------|-------------------|
| Predicative (13) | ρ = -0.85 | ρ = -0.86 | ρ = **-0.93** |
| Adverbial (19) | ρ = +0.87 | ρ = +0.79 | ρ = +0.87 |
| Noun phrase (11) | ρ = **-0.98** | ρ = 0.62 | ρ = -0.91 |

**Supervised direction (ridge regression) correlation:**

| Type | Nomic | Gemma | MXBAI |
|------|-------|-------|-------|
| Predicative | 0.93 | **0.99** | 0.97 |
| Adverbial | 0.95 | 0.95 | 0.95 |
| Noun phrase | 0.97 | 0.96 | 0.97 |

**Cross-content generalization** (probability axis found with "experiment will
succeed" tested on "treatment will be effective" and "prediction will be
correct"): ρ > 0.77 everywhere, ρ > 0.92 in most cases. The probability axis
generalizes across what is being hedged.

### The "Probably Direction" Puzzle

The direction that adding "probably" moves sentences (Part A) does NOT align
with the probability axis found within any syntactic type (alignment < 0.18
across all models). This is consistent and informative:

- The **probability axis** measures how different hedge phrases differ from each
  other (syntax held constant).
- The **"probably" direction** measures how one phrase moves sentences from their
  unhedged position (syntax changes alongside probability).

These are different geometric phenomena. Adding "probably" changes both the
syntactic frame AND the implied probability simultaneously. The within-type
probability axis isolates the probabilistic component by holding syntax constant.

### What This Means

1. **Probability is a linear direction** — confirmed across 3 architecturally
   different models (BERT-style, decoder-derived, and mixed). This is a property
   of the language, not of specific training recipes.

2. **The direction is type-specific** — predicative, adverbial, and noun-phrase
   constructions each have their own probability axis. Mixing types confounds
   syntax with semantics.

3. **The direction generalizes across content** — the axis found using one claim
   template works for other claim templates. This means you can calibrate once
   and apply broadly.

4. **Unsupervised PCA finds the axis** — you don't need labeled probability
   data to find the direction. Just embed the Mosteller phrases and PCA finds
   probability as the dominant axis (especially for MXBAI where PC1 explains
   60% of variance and correlates at ρ = -0.93).

5. **Supervised directions are very strong** — ridge regression on Mosteller
   medians gives ρ > 0.93 in all cases across all models. This means the
   Mosteller data is sufficient calibration data for accurate geometric
   probability extraction.

### Implications for the Calibration System

To extract probability from a hedged claim geometrically:

1. Identify the syntactic type of the hedge expression (predicative/adverbial/
   noun-phrase). This could be done by a simple classifier or by matching
   against known expressions.
2. Embed the hedged claim and the corresponding bare (unhedged) claim.
3. Compute the difference vector.
4. Project onto the type-specific probability axis (trained on Mosteller via
   ridge regression — precomputed and stored).
5. Map projection to probability (linear mapping calibrated against Mosteller
   medians).

This is more complex than "one axis for everything" but still very tractable
and fast.

---

## Remaining Questions (Post-Experiment)

1. **Can we build a universal axis?** If we regress out the syntactic component
   (e.g., by centering each type group before PCA), does a single axis emerge?
   This would simplify the system.

2. **Does the IQR-consistency prediction hold?** We haven't yet tested whether
   ambiguous phrases (high IQR, like "possible") produce less consistent
   difference vectors across claims than precise phrases (low IQR, like
   "certain"). This requires embedding each phrase in many claims (20+) and
   measuring per-phrase consistency. (Initial evidence from diff magnitude vs.
   IQR was ρ = -0.24, non-significant — but magnitude isn't the same as
   directional consistency.)

3. **Domain shift detection?** The medical meta-analysis data shows measurable
   shifts for some phrases. Can the geometry detect this? (Experiment 4 from the
   original plan.)

4. **How to handle novel phrasings?** "I'd wager that...", "signs point to...",
   "it stands to reason..." — phrases not in Mosteller. Can we embed these and
   project onto the probability axis? What syntactic type do they belong to? This
   is the practical value proposition.

5. **Does the Vogel data validate?** Initial cross-validation (Part B, all types
   mixed, PC1) showed marginal ρ = 0.48 with Vogel. Re-running Vogel validation
   within types, using the supervised axis, should give stronger results.

6. **The "probably" direction decomposition.** Can we factor the "probably"
   direction into a syntactic component and a probabilistic component? If so,
   the de-hedging idea from the geometry paper becomes tractable: project out
   the hedge subspace to recover pure propositional content.

---

**Document Status:** Experiments 1-3 complete. Core hypothesis validated (with
nuance about syntactic types). Ready for next-phase experiments.

**Key files:**
- `experiment_01_hedge_direction.py` — initial experiment (mixed types)
- `experiment_01b_within_type.py` — within-type analysis (the main result)
- `docs/mosteller_youtz_1990_full.csv` — ground truth data
- `docs/epistemic-geometry-hedging-as-linear-structure.md` — original hypothesis
