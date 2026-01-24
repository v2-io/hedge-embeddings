# Epistemic Geometry: Hedging as Linear Structure in Embedding Space

**Purpose:** Exploration of the hypothesis that epistemic hedging, like grammatical gender or tense, is encoded as a consistent vector direction in embedding space — and the implications for automated epistemic calibration.

**Status:** Hypotheses and experimental directions

**Last Updated:** 2026-01-23

**Companion to:** `epistemic-claim-calibration-first-principles.md`

---

## Table of Contents

1. [The Core Analogy](#the-core-analogy)
2. [The Decomposition Hypothesis](#the-decomposition-hypothesis)
3. [Testable Predictions](#testable-predictions)
4. [Implications for Epistemic Calibration](#implications-for-epistemic-calibration)
5. [Connections to Embedding Model Training](#connections-to-embedding-model-training)
6. [Experimental Path](#experimental-path)
7. [Open Questions](#open-questions)

---

## The Core Analogy

### The Word2Vec Discovery

Mikolov et al. (2013) demonstrated that semantic relationships are encoded as consistent vector directions in embedding space:

```
vector("king") - vector("man") + vector("woman") ≈ vector("queen")
```

The direction from "man" to "woman" is approximately the same as the direction from "king" to "queen", from "waiter" to "waitress", from "actor" to "actress". Gender is not a function of the word's content — it's a **linear transformation** applied consistently across the vocabulary.

This finding generalized: tense, plurality, comparative forms, and other grammatical and semantic modifications all manifest as approximately linear directions.

### The Hypothesis: Epistemic Hedging is a Linear Direction

If grammatical modifications are linear, perhaps epistemic modifications are too:

```
vector("X is true") → vector("X is probably true")
```

...should be approximately the same transformation as:

```
vector("Y is true") → vector("Y is probably true")
```

...for any propositions X and Y.

If this holds:

- **"Probably" is a direction**, not a content-dependent function
- **Different hedge words are different magnitudes** along this direction (or nearby directions)
- **Confidence can be read off** as a projection onto this axis
- **Propositional content can be recovered** by projecting out the hedge component

---

## The Decomposition Hypothesis

### First-Order Decomposition

Any hedged claim embedding decomposes as:

```
e(hedged_claim) = e(bare_proposition) + α · h
```

Where:
- `e(bare_proposition)` — the propositional content, independent of confidence
- `h` — the hedge direction (unit vector)
- `α` — the hedge magnitude (larger = more uncertain)

The bare proposition is what is being claimed. The hedge direction encodes epistemic stance. The magnitude encodes degree of hedging.

### Second-Order Decomposition

If we allow for additional linear factors:

```
e(claim) = e(core_content) + α · h + β · s + γ · d + ε
```

Where:
- `h` — hedge direction (epistemic confidence)
- `s` — source signature direction (stylistic/idiolectal)
- `d` — domain direction (field-specific framing)
- `ε` — residual / noise

This is a linear factor model. The factors may not be perfectly orthogonal, but if they're substantially independent, standard linear algebra can separate them.

### The Subspace View

More generally, hedging might not be a single direction but a low-dimensional subspace. "Probably," "possibly," "likely," "perhaps" might span a 2-3 dimensional hedge subspace rather than lying on a single axis.

```
e(hedged_claim) = e(bare_proposition) + H · α
```

Where:
- `H` — a matrix whose columns span the hedge subspace (e.g., 768 × 3)
- `α` — coordinates within that subspace (e.g., 3 × 1)

The subspace view accommodates:
- Different "flavors" of hedging (epistemic vs. evidential vs. deontic)
- Non-linear relationships between hedge words
- Finer-grained confidence distinctions

---

## Testable Predictions

### Prediction 1: Hedge Vectors Are Consistent Across Content

If hedging is a direction, then the vector from a claim to its hedged version should be similar regardless of what the claim is about.

**Test:**
```
For many diverse claims C₁, C₂, ..., Cₙ:
  Compute dᵢ = embed(hedge(Cᵢ)) - embed(Cᵢ)
  
Measure: pairwise cosine similarity among {d₁, d₂, ..., dₙ}

Prediction: mean pairwise cosine > 0.7
```

If the hedge vectors point in random directions, the mean cosine will be near zero. If they're consistent, it will be high.

### Prediction 2: Different Hedge Words Lie Along a Common Axis

"Probably," "possibly," "definitely," "certainly" should all modify embeddings along approximately the same direction, differing primarily in magnitude.

**Test:**
```
For a fixed claim C:
  For each hedge phrase p in {probably, possibly, definitely, certainly, ...}:
    Compute dₚ = embed(apply(C, p)) - embed(C)
    
Measure: Do all dₚ lie in a low-dimensional subspace?

Method: PCA on {dₚ}. 
Prediction: First 1-3 components explain >90% of variance.
```

### Prediction 3: Projection Magnitude Correlates with Calibrated Probability

If the hedge direction is meaningful, projection onto it should predict the empirical probability associated with each hedge phrase.

**Test:**
```
Extract hedge axis h (mean of normalized hedge vectors)

For each Mosteller phrase p with known probability P(p):
  Compute projection πₚ = embed(apply(C, p)) · h
  
Measure: correlation(π, 1 - P)

Prediction: |correlation| > 0.7
```

The sign depends on convention (whether h points toward certainty or uncertainty), but the magnitude of correlation indicates how well the geometric projection captures calibrated confidence.

### Prediction 4: De-Hedging Improves Content Similarity

Two claims that differ only in hedge phrasing should become more similar when the hedge component is removed.

**Test:**
```
Create pairs: (C with "definitely", C with "possibly") for many claims C

raw_similarity = cosine(embed(C_definitely), embed(C_possibly))
dehedged_similarity = cosine(project_out(embed(C_definitely), h), 
                             project_out(embed(C_possibly), h))

Prediction: dehedged_similarity > raw_similarity, consistently
```

### Prediction 5: Source-Specific Hedge Directions Exist

Different sources may use hedge language with systematic biases. If so, their hedge vectors will differ.

**Test:**
```
For each source S in corpus:
  Collect hedge vectors {dᵢˢ} from S's claims
  Compute source hedge direction hˢ = mean(normalize(dᵢˢ))

Measure: variance of {hˢ} across sources

High variance → sources use hedging differently (σ_source_phrase is large)
Low variance → hedging is universal (can pool across sources)
```

---

## Implications for Epistemic Calibration

If the decomposition hypothesis holds, it dramatically simplifies the calibration problem described in the companion paper.

### From Categorical to Geometric

The hierarchical model treats (source, domain, phrase) as discrete indices:

```
P(true | source, domain, phrase) — a lookup into a sparse table
```

The geometric model treats these as continuous directions:

```
e(claim) = content + α·hedge + β·source_style + γ·domain

confidence = f(α)                    — read from hedge projection
source_calibration = g(source, α)    — learned scaling per source
domain_adjustment = h(domain, α)     — learned offset per domain
```

The advantage: **generalization without pooling**. A new phrase we've never seen still has a projection onto the hedge axis. A new source still produces hedge vectors we can compare to known sources.

### Claim Deduplication

With hedging as a direction, claims can be deduplicated by content:

```
canonical(claim) = project_out(embed(claim), hedge_subspace)
```

"X is definitely true" and "X is probably true" map to the same canonical embedding. Grouping claims by propositional content becomes a nearest-neighbor search in the de-hedged space.

### Confidence Extraction Without Phrase Matching

Traditional approach:
1. Parse claim to extract hedge phrase
2. Look up phrase in calibration table
3. Return associated probability

Geometric approach:
1. Embed claim
2. Project onto hedge axis
3. Map projection magnitude to probability (via learned calibration)

The second approach:
- Handles novel phrasings ("I'd wager that...", "it stands to reason...")
- Captures confidence from tone, not just explicit hedge words
- Degrades gracefully for ambiguous expressions

### Source Calibration as Axis Rotation

If Zi-am-tur's "probably" projects to 0.3 on the hedge axis, and Sophist's "probably" projects to 0.5, this directly encodes their different calibrations.

More subtly: if different sources have slightly different hedge directions (not just magnitudes), we can learn a per-source rotation that aligns their hedge space to a canonical frame. After alignment, projections become comparable across sources.

```
For source S with hedge direction hˢ:
  rotation Rˢ aligns hˢ to canonical hedge axis h*
  calibrated_confidence(claim from S) = (Rˢ · embed(claim)) · h*
```

### Domain as Context-Dependent Calibration

Medical hedging may differ systematically from technical hedging. If so:

```
h_medical ≠ h_technical
```

Claims can be auto-tagged with domain (via embedding similarity to domain centroids), then the appropriate domain-specific hedge axis applied.

Alternatively, domain might manifest as an offset rather than a direction change:

```
confidence_medical = confidence_base + δ_medical
```

The geometric analysis will reveal which structure is present.

---

## Connections to Embedding Model Training

### Why This Structure Might Exist

Modern embedding models are trained with contrastive objectives: push apart semantically different texts, pull together semantically similar ones.

Hedging preserves propositional content while modifying stance. A well-trained model should:
- Keep "X is true" and "X is probably true" relatively close (same content)
- Distinguish them enough to be useful (different meanings)

The natural solution is to encode the difference in a consistent direction — exactly the linear structure we're hypothesizing.

### Hard Negatives and Hedge Sensitivity

You mentioned Gemma's training uses hard negatives to spread vectors and preserve information under quantization. 

"X is definitely true" and "X is probably true" are natural hard negatives — similar enough to require careful distinction, different enough to matter. If the training data includes such pairs (or the model learns from diverse text that contains such variations), the hedge direction emerges.

**Prediction:** Models trained with more sophisticated hard-negative mining will have cleaner hedge directions than models trained with simple random negatives.

### Matryoshka and Hedge Localization

Matryoshka Representation Learning (MRL) organizes dimensions by importance. If hedging is a consistent phenomenon, it might be concentrated in specific dimensions.

**Hypothesis:** The hedge direction in MRL-trained models is concentrated in the first N dimensions (where N << full dimensionality).

**Test:** 
```
Compute hedge axis h using full embeddings
Compute hedge axis h' using truncated embeddings (first 64 dims)
Measure: cosine(h, truncate(h')) and correlation of projections
```

If hedge information is localized in early dimensions, truncated embeddings suffice for confidence extraction — enabling fast inference.

### Fine-Tuning for Epistemic Decomposition

If the structure exists but is noisy, fine-tuning can sharpen it.

**Training objective:**
```
For pairs (C, hedge(C)) with known confidence level:
  
  # Encourage consistent hedge direction
  loss_direction = 1 - cosine(embed(hedge(C)) - embed(C), target_hedge_axis)
  
  # Encourage calibrated magnitude
  loss_magnitude = (projection - target_confidence)²
  
  # Preserve content similarity
  loss_content = 1 - cosine(project_out(embed(hedge(C)), h), 
                            project_out(embed(C), h))
  
  total_loss = loss_direction + λ₁·loss_magnitude + λ₂·loss_content
```

The Mosteller/Vogel data provides supervision for `target_confidence`. The loss_direction term enforces linearity. The loss_content term ensures de-hedging works.

---

## Experimental Path

### Phase 1: Validate the Basic Hypothesis

**Minimal experiment:**
1. Select 50 diverse neutral claims spanning multiple domains
2. Create hedged variants with "probably"
3. Embed with a standard model (e.g., sentence-transformers, Gemma embedding)
4. Compute difference vectors
5. Measure consistency (pairwise cosine)

**Success criterion:** Mean pairwise cosine > 0.6

**Time estimate:** Hours (just embedding and linear algebra)

### Phase 2: Explore the Hedge Subspace

**If Phase 1 succeeds:**
1. Repeat with multiple hedge phrases (probably, possibly, definitely, certainly, might, etc.)
2. Collect all difference vectors across claims and phrases
3. PCA to find the hedge subspace dimensionality
4. Check if Mosteller probability correlates with position in this subspace

**Questions answered:**
- Is hedging 1D or multi-dimensional?
- Do different hedge phrases form interpretable clusters?
- Does geometric position predict calibrated probability?

### Phase 3: Test De-Hedging Utility

**If Phase 2 shows structure:**
1. Create pairs of claims that differ only in hedge phrasing
2. Compute raw similarity and de-hedged similarity
3. Measure improvement in content matching

**Also test:**
- Can de-hedged embeddings find paraphrases better?
- Does de-hedging help claim deduplication in a real corpus?

### Phase 4: Source and Domain Analysis

**With a real corpus:**
1. Extract claims from multiple sources with attribution
2. Compute source-specific hedge vectors
3. Measure cross-source variance
4. Repeat stratified by domain

**Questions answered:**
- Do sources have idiosyncratic hedge directions?
- Is the variance large enough to matter for calibration?
- Do domains have different hedge norms?

### Phase 5: Integration with Outcome-Based Calibration

**The bridge to the first paper:**
1. For claims with known outcomes, compute geometric confidence (hedge projection)
2. Compare to outcome frequency
3. Learn a calibration function: projection → probability
4. Test whether geometric confidence predicts outcomes better than phrase-lookup

If geometric confidence has predictive power, it can serve as a prior in the hierarchical model — or potentially replace the phrase-based approach entirely.

---

## Open Questions

### 1. Linearity Limits

The linear analogy (king - man + woman = queen) is approximate, not exact. The same will be true for hedging. 

**Questions:**
- How much variance is lost by assuming linearity?
- Are there systematic non-linearities (e.g., double hedging: "probably possibly")?
- Does the linear approximation improve or degrade in larger models?

### 2. Hedge vs. Evidential vs. Epistemic

Linguistic theory distinguishes:
- **Epistemic modality**: speaker's confidence ("I think X")
- **Evidential modality**: source of knowledge ("Apparently X", "I heard X")
- **Deontic modality**: obligation/permission ("X should be true")

These are semantically distinct. Are they geometrically distinct?

**Hypothesis:** They occupy different subspaces, possibly overlapping.

**Implication:** The "hedge axis" might need to be decomposed further for precise calibration.

### 3. Negation and Reversal

What is the relationship between:
- "X is likely" → "X is unlikely"
- "X is likely" → "X is probably not"

Is unlikelihood the opposite direction on the same axis? Or a different axis? Or does negation interact with hedging non-linearly?

### 4. Compositional Hedging

How do multiple hedges compose?
- "X is very likely" — intensifier + hedge
- "X is somewhat possible" — attenuator + hedge
- "I suspect X might be true" — multiple hedge markers

**Hypothesis:** Composition is approximately additive in embedding space, but with diminishing returns (sublinear scaling).

### 5. Cross-Model Consistency

Do different embedding models learn similar hedge directions?

**If yes:** The structure is "real" — present in the data, not an artifact of a specific model.

**If no:** The structure is model-dependent — calibration must be model-specific.

### 6. Temporal Stability

Do hedge directions drift as models are updated or as language evolves?

**For practical systems:** If the hedge axis must be recalibrated with each model update, that's a maintenance burden. If it's stable, the calibration can be computed once and reused.

### 7. Multilinguality

Is hedging direction-consistent across languages?

**Hypothesis:** For multilingual embedding models, hedge directions are approximately aligned across languages (since the training objective encourages cross-lingual alignment).

**If true:** Calibration data from English (Mosteller, Vogel) transfers to other languages for free.

---

## The Simplifying Vision

If this research program succeeds, epistemic calibration reduces to:

1. **Embed the claim**
2. **Project onto hedge subspace** → raw confidence score
3. **Apply source-specific scaling** → calibrated confidence
4. **Apply domain adjustment** → context-appropriate confidence
5. **Report as probability interval** → honest uncertainty

No phrase parsing. No lookup tables. No hierarchical pooling for cold-start.

The Mosteller/Vogel data becomes a calibration target for the projection function. Outcome data refines source-specific and domain-specific adjustments. Everything is continuous, differentiable, and geometrically interpretable.

---

## Connections to Conversational Dynamics

The companion notes on conversational dynamics in embedding space (`conversational-dynamics-embeddings.md`) suggest another application:

**Confidence evolution over a conversation:**

If confidence is a direction, we can track how a conversation's stance on a proposition evolves:

```
Turn 1: "X might be true"     → projection α₁
Turn 3: "X seems likely"      → projection α₂
Turn 5: "X is probably true"  → projection α₃
Turn 7: "X is definitely true" → projection α₄
```

The trajectory α₁ → α₂ → α₃ → α₄ traces growing confidence in embedding space. We can detect:
- **Convergence:** Both parties' hedge projections approaching zero (certainty)
- **Divergence:** Hedge projections moving apart (disagreement about confidence)
- **Stagnation:** Hedge projections unchanging despite new information

This connects epistemic calibration (what does a hedge *mean*) to epistemic dynamics (how does confidence *evolve*).

---

## Summary

The hypothesis: **Epistemic hedging is encoded as a linear direction in embedding space**, analogous to gender or tense.

If true, this:
- Explains why hedge language feels like a "modifier" on content
- Provides a geometric method for confidence extraction
- Enables content-based claim deduplication
- Grounds source and domain calibration in measurable geometric variance
- Simplifies the hierarchical model to linear projections and learned scalings

The experimental path is tractable: embed claims, compute differences, measure consistency. The first validation can be done in hours with existing tools.

---

## References

### Word Embeddings and Linear Structure
- Mikolov, T., et al. (2013). "Linguistic Regularities in Continuous Space Word Representations." NAACL-HLT.
- Mikolov, T., et al. (2013). "Distributed Representations of Words and Phrases and their Compositionality." NeurIPS.
- Ethayarajh, K. (2019). "How Contextual are Contextualized Word Representations?" EMNLP. (Explores geometry of contextual embeddings)

### Embedding Model Training
- Muennighoff, N., et al. (2024). "Matryoshka Representation Learning." NeurIPS.
- Lee, C., et al. (2024). "Gecko: Versatile Text Embeddings Distilled from Large Language Models." (Google's embedding training with hard negatives)

### Epistemic Modality in Linguistics
- Nuyts, J. (2001). "Epistemic Modality, Language, and Conceptualization." John Benjamins.
- Palmer, F. R. (2001). "Mood and Modality." Cambridge University Press.

### Empirical Calibration Data
- Mosteller, F., & Youtz, C. (1990). "Quantifying probabilistic expressions." Statistical Science.
- Wallsten, T. S., et al. (1986). "Measuring the vague meanings of probability terms." JEP: General.

### Geometric Methods
- Principal Component Analysis — standard reference for subspace extraction
- Bolukbasi, T., et al. (2016). "Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings." NeurIPS. (Methods for identifying and manipulating semantic directions)

---

**Document Status:** Hypothesis specification with experimental path

**Next Actions:**
1. Run Phase 1 experiment with existing embeddings
2. If successful, proceed to Phase 2 (hedge subspace characterization)
3. Document findings and refine hypotheses
4. Integrate with outcome-based calibration framework
