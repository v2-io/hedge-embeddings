# Experiment Proposal: Causal Measurement Model — Who is Measuring What?

## The Deep Question

We have two independent windows onto "what probability does 'likely' convey?":

1. **Mosteller & Youtz (1990):** Ask 238 science writers "what percentage does 'likely' mean?" → 71.1% (median)
2. **Embedding geometry:** Train a model on billions of tokens of text, then project the difference vector for "likely" onto a supervised axis → ~71% (across models)

These agree at ρ > 0.90. But WHY do they agree? And which is the more fundamental measurement?

The standard framing treats Mosteller as "ground truth" and the embeddings as "predictions to validate." But this framing may be backwards — or at least incomplete. The embedding models were trained on orders of magnitude more language data than Mosteller's 238 respondents ever produced. They capture the distributional reality of how these words are *actually used* across millions of contexts, not how a specific population *reports* they interpret them.

**The question:** Is the embedding geometry a noisy measurement of Mosteller's "ground truth"? Or are both noisy measurements of some deeper latent quantity — the actual communicative function of hedge words in English — and if so, which is the more precise instrument?

## The Causal Structure

### A Structural Causal Model (SCM)

```
                    ┌─────────────┐
                    │  P_true(w)  │  ← The "real" probability semantics
                    │  (latent)   │     of word w in English
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
              ▼            ▼            ▼
     ┌────────────┐ ┌───────────┐ ┌──────────────┐
     │ Language    │ │ Elicited  │ │ Other        │
     │ use in     │ │ judgment  │ │ behavioral   │
     │ text       │ │ (survey)  │ │ signals      │
     └─────┬──────┘ └─────┬─────┘ └──────────────┘
           │              │
     ┌─────┴──────┐       │
     │ Training   │       │
     │ corpus     │       │
     └─────┬──────┘       │
           │              │
     ┌─────┴──────┐       │
     │ Embedding  │       │
     │ model      │       │
     └─────┬──────┘       │
           │              │
     ┌─────┴──────┐ ┌─────┴─────┐
     │ P_embed(w) │ │ P_most(w) │
     │ (our       │ │ (Mosteller│
     │ projection)│ │ median)   │
     └────────────┘ └───────────┘
```

### Noise Sources at Each Stage

**Mosteller path:** P_true → metacognitive elicitation → survey response

| Noise source | Effect | Reducible? |
|---|---|---|
| Population specificity | 238 US science writers in 1990 | Different populations give different numbers |
| Metacognitive access | People report what they THINK they mean, not how they USE words | Fundamental gap between reported and actual semantics |
| Survey framing | "What percentage does X mean to you?" primes numerical thinking | May not reflect natural interpretation |
| Sample size | n=238 per expression | Moderate precision on the median |
| Temporal drift | 1990 usage norms | Language may have shifted |

**Embedding path:** P_true → language use → corpus → model → projection

| Noise source | Effect | Reducible? |
|---|---|---|
| Corpus composition | Web text, books, papers — not representative of all English | Genre/register bias |
| Distributional indirectness | Model learns co-occurrence, not probability | The map is not the territory |
| Training objective | Contrastive learning optimizes for discriminability, not semantics | May distort probability spacing |
| Architecture | Specific pooling, attention patterns | Model-specific geometry |
| Projection method | Ridge regression with λ=0.1 on 8–19 training points | Low-data supervised extraction |

### The Key Insight

The Mosteller path has a **metacognitive bottleneck**: it requires humans to introspect on what words mean and report a number. This is known to be noisy and subject to systematic biases (anchoring, framing, etc.).

The embedding path has a **distributional bottleneck**: it can only capture what's encoded in patterns of word co-occurrence. But it integrates across billions of usage instances.

If the "true" probability semantics of hedge words is defined as **the communicative function these words serve in actual language use** (rather than what people say they mean when asked), then the embedding path may be the more direct measurement — it observes actual usage at massive scale, while Mosteller observes reported introspections at small scale.

## Proposed Analysis

### 1. Hierarchical Measurement Model

Treat Mosteller, Vogel, and the 5 embedding models as **7 noisy instruments** measuring the same latent quantity P_true(w) for each expression w.

```
P_true(w) ~ some prior (e.g., Uniform(0, 100) or weakly informative)

# Survey instruments
P_mosteller(w) | P_true(w) ~ Normal(P_true(w), σ²_mosteller)
P_vogel(w)     | P_true(w) ~ Normal(P_true(w), σ²_vogel)

# Embedding instruments (each model)
P_nomic(w)     | P_true(w) ~ Normal(P_true(w) · α_nomic + β_nomic, σ²_nomic)
P_gemma(w)     | P_true(w) ~ Normal(P_true(w) · α_gemma + β_gemma, σ²_gemma)
P_moe(w)       | P_true(w) ~ Normal(P_true(w) · α_moe   + β_moe,   σ²_moe)
P_mxbai(w)     | P_true(w) ~ Normal(P_true(w) · α_mxbai + β_mxbai, σ²_mxbai)
P_qwen3(w)     | P_true(w) ~ Normal(P_true(w) · α_qwen3 + β_qwen3, σ²_qwen3)
```

The embedding instruments get affine calibration parameters (α, β) because the linear calibration mapping (slope × projection + intercept) may introduce systematic scale/offset differences. The survey instruments are assumed to measure in the "natural" probability scale (though this assumption is testable).

**What this model gives us:**

- **Posterior P_true(w):** Better estimates of what each expression "really means" than any single source, by integrating across all 7 instruments
- **σ² for each instrument:** Which measurements are noisiest? If σ²_mosteller > σ²_mxbai, the embeddings are literally more precise than the psychometric survey
- **α, β for each model:** How well-calibrated is each model? α ≈ 1, β ≈ 0 means the model's probability scale matches the latent scale directly
- **Predictive checks:** For Vogel expressions not in Mosteller, the model predicts what Vogel SHOULD find based on the latent estimates — a genuine prediction

### 2. Instrument Precision Comparison

The critical test: **which instruments have smaller σ²?**

If the embedding models have systematically lower variance than the surveys, it would mean they provide *more precise* measurements of probability semantics. This is plausible because:

- They integrate across vastly more data (billions of tokens vs. hundreds of respondents)
- They're not subject to metacognitive noise
- They capture distributional patterns that humans use but can't easily introspect on

If the surveys have lower variance, it would mean that explicit elicitation captures something that distributional patterns miss — perhaps because metacognitive access provides a more direct window onto probability semantics than statistical co-occurrence.

### 3. Causal Identification via Cross-Instrument Disagreement

The most informative data points are expressions where the instruments **disagree**. Our data already shows a few:

- **"Possible":** Mosteller = 38.5% (but bimodal), qwen3 = 38.6%, nomic = 54.2%
  - Models disagree with each other → model-specific noise
  - The bimodality creates disagreement within Mosteller → survey noise

- **"Impossible":** Mosteller = 0.3%, Vogel = 7.24%
  - Surveys disagree with each other → the "truth" may be different across populations/eras
  - Embedding models project to ~0% → they agree with Mosteller, not Vogel

- **Expressions where LOO errors are large** (e.g., "Always" on mxbai: LOO pred = 67%, true = 99.7%)
  - These reveal where the geometric assumption (linear probability axis) breaks down
  - Not measurement noise per se, but model misspecification

### 4. Interventionist Questions (Harder, Future Work)

Pearl's do-calculus asks: what would happen if we *intervened* on the data-generating process?

- **do(corpus = medical_only):** If we trained an embedding model on only medical text, would "rare" shift from ~7% to ~10% (matching the medical meta-analysis)? This would confirm the causal path: actual usage patterns → embedding geometry → projected probability.

- **do(survey_population = doctors):** The medical meta-analysis already provides this — doctors interpret "rare" differently from the general population. The causal question: does this reflect a different P_true (different communicative conventions in medicine), or a different measurement process (doctors are more precise)?

- **do(temporal_shift):** If we had embeddings from models trained on 1990 vs. 2024 corpora, do the projections shift? Vogel (2022) suggests the survey answers HAVEN'T shifted in 50 years — is the same true for distributional patterns?

## What We Can Do Now (With Current Data)

Without running new experiments, we can fit the hierarchical measurement model using:

- **Mosteller data:** 53 expressions with medians (our training data)
- **Vogel data:** ~20 overlapping expressions with means (independent validation)
- **5 embedding models:** LOO predictions for each expression on each model (from paper_validation.py)

The LOO predictions are the right inputs because they represent what each model predicts for an expression it wasn't trained on — making them genuine "measurements" rather than fitted values.

### Implementation

A Stan or PyMC model with:
- ~40 expressions that have data from at least Mosteller + 1 embedding model
- 7 instrument-specific noise parameters
- 5 affine calibration parameters (for embedding models)
- Latent P_true for each expression

This is a ~50-parameter model with ~200 observations — well-identified and fast to fit.

### Expected Outcomes

**Most likely result:** The embedding models have noise comparable to the survey instruments for well-represented expressions (mid-range probabilities) but higher noise at the extremes (0% and 100% are hard to project precisely due to boundary effects). The posterior P_true values will be a principled "best estimate" for each expression that neither survey nor any single model provides alone.

**Exciting possible result:** The embedding models have *lower* noise than the surveys, especially for expressions with high IQR in Mosteller. This would mean the distributional signal is more precise than explicit elicitation — humans agree more in how they USE these words than in how they REPORT their meanings.

**Also exciting if negative:** If the surveys are consistently more precise, it means metacognitive access to word meaning is a better measurement instrument than distributional statistics. This would have implications for the broader NLP claim that "distributional semantics captures meaning."

## Relevance to the Paper

This analysis could be:

1. **A section in the main paper** — framing the relationship between Mosteller and embeddings as a measurement-theoretic question rather than a validation question. This would elevate the paper from "we replicated psychometric data with embeddings" to "we established embeddings as a measurement instrument for probability semantics with quantified precision."

2. **A companion paper** — the causal/measurement-theoretic analysis is deep enough for its own treatment, connecting to the philosophy of linguistic meaning, measurement theory, and the "what do embeddings know?" literature.

3. **A Discussion section reframing** — even without the formal model, the conceptual point can be made: these are two independent measurements of the same latent quantity, and the agreement (ρ > 0.90) is evidence for the *reality* of the latent quantity, not just validation of one by the other.

## Connection to the First-Principles Calibration Framework

The companion document (`docs/epistemic-claim-calibration-first-principles.md`) already describes a hierarchical Bayesian framework for learning P(true | source, domain, phrase). The measurement model proposed here provides the PRIOR for that framework: instead of using Mosteller medians as fixed priors, use the posterior P_true from the multi-instrument model. This gives better priors (integrated across all evidence sources) with proper uncertainty quantification (the posterior width reflects real measurement uncertainty, not just Mosteller's IQR).

The geometric projection then serves as a LIKELIHOOD FUNCTION within the calibration system: for a new hedged claim, the embedding projection provides a continuous observation that updates the prior toward a calibrated posterior. The measurement model tells us exactly how much to trust this observation (via σ²_model).

## References

- Pearl, J. (2009). *Causality: Models, Reasoning, and Inference*. Cambridge.
- Measurement error models: Carroll, R. J., et al. (2006). *Measurement Error in Nonlinear Models*. Chapman & Hall.
- Multi-trait multi-method: Campbell, D. T., & Fiske, D. W. (1959). "Convergent and discriminant validation by the multitrait-multimethod matrix." *Psychological Bulletin*.
- Item Response Theory (related): The expressions are "items" and the instruments are "raters" — IRT models from psychometrics apply.
- Linguistic meaning as measurement: Lassiter, D. (2017). *Graded Modality*. Oxford. (Formal semantics of gradable epistemic expressions.)
