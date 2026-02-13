# Paper Plan: Epistemic Hedging as Calibrated Linear Structure in Sentence Embedding Space

## Overall Assessment

This is a **genuinely novel and publishable finding**. Literature search confirms no one has specifically investigated whether epistemic hedging manifests as calibrated linear structure in *sentence* embedding space with psychometric ground truth (Mosteller & Youtz 1990). The closest related work is on linear probing of embeddings for other properties, and on LLM verbalized uncertainty — but not on the geometric encoding of hedge semantics by embedding models.

The core result — probability is a linear direction within syntactic types, validated across 5 architecturally diverse models at ρ > 0.90 supervised — is strong and clean. The syntactic-type insight (mixing types confounds the signal) is both practically useful and theoretically interesting.

---

## Strengths for Publication

1. **Clear, testable hypothesis** with crisp positive result
2. **5 architecturally diverse models** (BERT, decoder, MoE, BERT-large, Qwen3) — rules out training artifact
3. **Ground truth from established psychometric data** (Mosteller & Youtz 1990, n=238)
4. **Multiple complementary experiments** building logically
5. **Important negative results** (IQR not in geometry, ensemble doesn't beat modal-only)
6. **Genuinely surprising findings** (syntactic confound, negation as probability-reflection, median-consistency dimensionality flip)
7. **Clear practical implications** (4-axis calibration recipe)

## Weaknesses That Need Addressing

1. **No real-text evaluation** — all experiments use template sentences
2. **No baseline comparisons** — need to compare against naive baselines (random direction, phrase lookup, etc.)
3. **Informal "intuition" values** used for novel-phrase evaluation instead of independent ground truth
4. **No formal cross-validation** on the Mosteller training data (leave-one-out on the ridge regression)
5. **Modal axis training mixes Mosteller data with author estimates** — should clearly separate
6. **No statistical rigor on the key claims** — need bootstrap CIs, correction for multiple comparisons
7. **Reproducibility** — Ollama-only, no standard benchmark framework
8. **No Vogel within-type cross-validation** (was done all-types mixed only)

---

## Paper Structure

### Proposed Title

**"Epistemic Hedging as Calibrated Linear Structure in Sentence Embedding Space"**

Alternative: *"Probability is a Direction: How Sentence Embeddings Encode Calibrated Uncertainty"*

### Target Venue

- **Primary:** arXiv (cs.CL, cs.AI) — immediate visibility
- **Conference targets:** EMNLP 2026, ACL 2026, or NAACL 2026 (computational semantics / representation learning tracks)
- **Alternative:** Computational Linguistics journal (longer format, suits the depth of experiments)

### Sections

#### Abstract (~250 words)
- Hypothesis: epistemic hedging is encoded as calibrated linear structure
- Key finding: within syntactic types, probability is a linear direction (ρ > 0.90 supervised, 5 models)
- The syntactic-type insight (mixing types confounds the signal)
- Modal axis as near-universal fallback (MAE 3.2–8.5%)
- Negative results: IQR not in geometry, ensemble doesn't beat single axis
- Practical implication: phrase-agnostic probability extraction from embeddings

#### 1. Introduction (~1.5 pages)
- The problem: mapping verbal uncertainty to calibrated probability
- The analogy: Mikolov et al.'s linear regularities (king–man+woman=queen) → does this extend to epistemic modifiers in *sentence* embeddings?
- Why this matters: novel phrasings, no phrase lookup tables, geometric confidence extraction
- Preview of key finding and the syntactic-type insight

#### 2. Related Work (~1.5 pages)
- **Linear structure in embeddings:** Mikolov et al. 2013, Ethayarajh 2019 (contextual embeddings), Bolukbasi et al. 2016 (gender direction)
- **Verbal probability calibration:** Mosteller & Youtz 1990, Wallsten et al. 1986, Budescu & Wallsten 1995, Vogel 2022
- **Uncertainty in NLG:** Semantic uncertainty (Kuhn et al. 2023), verbalized uncertainty, LLM confidence calibration
- **Epistemic modality in NLP:** Hedge detection, speculation classification (CoNLL shared task), but NOT geometric calibration
- **Linear probing:** Conneau et al. 2018, Hewitt & Manning 2019 — probing for linguistic properties in representations
- Differentiation: prior work probes for binary/categorical properties; we probe for *continuously calibrated probability*

#### 3. Experimental Setup (~2 pages)
- **Ground truth:** Mosteller & Youtz 1990 dataset (53 expressions, n=238, P25/Median/P75/IQR)
- **Syntactic classification:** 4 types (predicative, frequency adverb, noun phrase, modal adverb) with rationale
- **Models:** 5 architecturally diverse embedding models with table of architectures/dimensions
- **Method:** Difference vector computation, ridge regression axis training (λ=0.1), linear calibration
- **Evaluation metrics:** Spearman ρ, Pearson r, MAE, with bootstrap 95% CIs
- **Baselines:**
  - Random direction (project onto random unit vector)
  - Phrase-lookup (Mosteller median directly, no geometry)
  - Mean-difference direction (unsupervised, like word2vec analogy)
  - PCA axis (unsupervised)

#### 4. Experiment 1: Consistency and the Syntactic Confound (~2 pages)
- Part A: "probably" consistency (cos 0.68) vs. "reportedly" control (cos 0.60)
- Part B: All-types-mixed PCA reveals syntactic type as PC1, not probability
- The fix: within-type analysis
- **Result:** Within types, PC1 captures probability (ρ up to −0.98 unsupervised, > 0.90 supervised)
- Cross-content generalization (axis from "experiment succeeds" works for "treatment effective")
- The "probably direction" puzzle: bare→hedged ≠ within-type probability axis

#### 5. Experiment 2: Generalization to Novel Phrases (~1.5 pages)
- Novel phrases NOT in Mosteller: first-person frames, doubt, anti-hedges, novel phrasings
- Ranking correlation: ρ = 0.75–0.90 across 5 models
- Calibration compression: systematic underestimation of high-probability phrases
- Architectural insight: nomic v1.5 collapses sentence-initial adverbs; MoE/MXBAI/qwen3 don't

#### 6. Experiment 3: The Modal Axis Solution (~1.5 pages)
- Problem: frequency adverbs ≠ modal adverbs (different semantic types, same syntactic slot)
- Solution: 4th axis for modal adverbs
- Result: MAE drops from 30–43% to 3.2–8.5%
- Axis geometry: 4 axes span genuine 4D subspace; predicative↔modal alignment increases with model size
- The practical recipe: modal axis as universal fallback

#### 7. Experiment 4: Negative Result — IQR Not in Geometry (~1 page)
- Prediction: ambiguous phrases (high IQR) should produce less consistent directions
- Result: ρ(consistency, −IQR) ≈ 0 across all 4 models
- Interpretation: models encode ONE meaning per phrase, not a distribution
- Secondary finding: median-consistency correlation flips with dimensionality

#### 8. Experiment 5: Ensemble Approaches (~1 page)
- 6 ensemble strategies tested; none beat modal-only on novel phrases
- Linear ensemble overfits to training distribution
- Logistic regression destroys extremes
- Practical implication: simple > complex

#### 9. Experiment 6: Compound Hedge Composition (~1.5 pages)
- Direction composition: cos 0.82–0.89 (approximately linear)
- Magnitude: sub-linear (ratio ~0.75)
- Negation exception: cos 0.22 on qwen3 — probability-reflecting, not linear
- Conflicting cues: the more extreme component dominates ("I doubt probably" → doubt wins)
- "Hedging is not just a direction — it's a linear algebra"

#### 10. Discussion (~2 pages)
- **What the geometry encodes:** probability level (yes), interpretive precision (no), syntactic frame (dominant)
- **Why the structure exists:** contrastive training naturally separates hedge variants
- **Limitations:**
  - Template sentences only (real-text evaluation needed)
  - Mosteller population may not match modern usage
  - Some training data includes author's probability estimates
  - All models are English-only
- **Connections to:** LLM confidence calibration, epistemic stance detection, de-hedging for claim deduplication
- **Future work:** Real text evaluation, domain shift, multilingual, de-hedging, fine-tuning to sharpen the axis

#### 11. Conclusion (~0.5 pages)
- Compact summary of core finding and practical implications

### Supplementary Material
- Full Mosteller expression rankings per model (the detailed tables from FINDINGS-01)
- Complete novel phrase projection tables
- All axis cosine similarity matrices
- Negation geometry probe results
- Code and data (link to repository)

---

## Pre-Submission Work Required

### Must-do (critical for credibility)

1. **Baselines** — Implement random direction, phrase-lookup, and unsupervised-PCA baselines. Without these, a reviewer can't assess whether the supervised axis is adding value over trivial approaches.

2. **Leave-one-out cross-validation** on the Mosteller ridge regression. Currently the supervised ρ values are in-sample. LOO would show true generalization within the training set.

3. **Bootstrap confidence intervals** on the key correlations (Spearman ρ). "ρ = 0.93" needs a CI to be publishable.

4. **Clean separation of training data** — The modal axis mixes Mosteller-derived probabilities with author estimates. The paper needs to clearly distinguish, and ideally report results with Mosteller-only training.

5. **Vogel cross-validation within type** — The current Vogel validation was done on mixed types. Redo within-type with the supervised axis for a clean train-on-Mosteller/test-on-Vogel result.

6. **Real-text evaluation** (even a small one) — Take 20-30 hedged sentences from real academic abstracts or news articles, have 2-3 annotators assign probability ratings, and test the modal axis on them. This would address the "template-only" limitation.

### Should-do (strengthens the paper)

7. **Formal comparison table** — All 5 models × 4 axes × key metrics in one comprehensive table
8. **Figures** — (a) Scatterplot of projection vs. Mosteller median for the best model/axis combo; (b) PCA visualization showing syntactic confound vs. within-type clean separation; (c) Axis cosine similarity heatmap across models; (d) Compound hedge composition diagram
9. **Matryoshka truncation test** — Does the axis survive dimensional truncation? Already suggested in the hypothesis doc but never tested.
10. **Reproducibility** — Provide scripts that work with HuggingFace sentence-transformers (not just Ollama), so reviewers can reproduce.

### Nice-to-have (if time permits)

11. Domain shift experiment (medical vs. general, using the meta-analysis data already in `docs/`)
12. Multilingual test (if a multilingual embedding model is available)
13. De-hedging demonstration (projecting out the hedge subspace to improve claim similarity)

---

## Estimated Effort

- **Baselines + LOO + bootstrap CIs**: ~1-2 days of scripting
- **Vogel within-type validation**: ~half day
- **Real-text evaluation** (small): ~2-3 days (collection + annotation + testing)
- **Figures**: ~1-2 days
- **Writing**: ~5-7 days for full paper
- **Total**: ~2-3 weeks to submission-ready

---

## Literature References

### Linear Structure in Embeddings
- Mikolov, T., et al. (2013). "Linguistic Regularities in Continuous Space Word Representations." NAACL-HLT.
- Mikolov, T., et al. (2013). "Distributed Representations of Words and Phrases and their Compositionality." NeurIPS.
- Ethayarajh, K. (2019). "How Contextual are Contextualized Word Representations?" EMNLP.
- Bolukbasi, T., et al. (2016). "Man is to Computer Programmer as Woman is to Homemaker?" NeurIPS.

### Verbal Probability Calibration
- Mosteller, F., & Youtz, C. (1990). "Quantifying probabilistic expressions." Statistical Science.
- Wallsten, T. S., et al. (1986). "Measuring the vague meanings of probability terms." JEP: General.
- Budescu, D. V., & Wallsten, T. S. (1995). "Processing linguistic probabilities." Psychology of Learning and Motivation.
- Vogel, T., et al. (2022). Systematic review of verbal probability expressions. [Meta-analysis of 21 studies, 1967-2018]

### Uncertainty in NLG / LLMs
- Kuhn, L., Gal, Y., & Farquhar, S. (2023). "Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in NLG." ICLR.
- Survey of Confidence Estimation and Calibration in LLMs (NAACL 2024)
- "What Verbalized Uncertainty in Language Models is Missing" (2025)
- "An evaluation of estimative uncertainty in large language models" (2026, npj Complexity)

### Linear Probing
- Conneau, A., et al. (2018). "What you can cram into a single $&!#* vector." ACL.
- Hewitt, J., & Manning, C. D. (2019). "A Structural Probe for Finding Syntax in Word Representations." NAACL.

### Embedding Model Training
- Muennighoff, N., et al. (2024). "Matryoshka Representation Learning." NeurIPS.
- Lee, C., et al. (2024). "Gecko: Versatile Text Embeddings Distilled from Large Language Models."

### Epistemic Modality in Linguistics
- Nuyts, J. (2001). "Epistemic Modality, Language, and Conceptualization." John Benjamins.
- Palmer, F. R. (2001). "Mood and Modality." Cambridge University Press.
