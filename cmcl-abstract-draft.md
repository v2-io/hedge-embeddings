# Calibrated Probability as Emergent Linear Structure in Sentence Embeddings

**[Non-archival extended abstract for CMCL 2026]**

---

## Abstract

We investigate whether epistemic hedging — the linguistic expression of uncertainty through words like "probably," "certainly," and "possibly" — manifests as calibrated linear structure in sentence embedding space. Recent work has shown that truth (Marks & Tegmark, 2023) and verbal uncertainty (Ji et al., 2025) are approximately linearly encoded in LLM representations, and that models can be fine-tuned to classify words of estimative probability (Sileo & Moens, 2023). However, no prior work has demonstrated that *pretrained* sentence embeddings encode a *continuously calibrated* probability axis aligned with human psychometric data. Using the Mosteller & Youtz (1990) dataset as ground truth (53 verbal probability expressions, n=238), we find that within syntactic types, probability is the dominant linear direction in five architecturally diverse embedding models (Spearman ρ > 0.90 supervised, 0.70–0.94 leave-one-out). A supervised axis trained on Mosteller data cross-validates against the independent Vogel (2022) meta-analysis at ρ = 0.99. The axis survives 12× dimensional compression and generalizes to novel hedge phrases not in the calibration data (ρ = 0.75–0.94). These results suggest that embedding models learn a geometric encoding of human probability semantics as an emergent property of distributional training.

---

## 1. Introduction

Mikolov et al. (2013) demonstrated that semantic relationships manifest as consistent vector directions in word embedding space. We ask whether this extends to *epistemic modification* in sentence embeddings: does adding "probably" to a claim move its embedding in a consistent direction, and does the magnitude of that movement correlate with the probability the word conveys?

Recent work provides reason to think it might. Marks and Tegmark (2023) show that truth is encoded as a linear direction in LLM activations. Ji et al. (2025) identify a single linear "verbal uncertainty" feature in residual streams that can be causally manipulated to reduce hallucinations. Lin et al. (2022) demonstrate that pretrained GPT-3 embeddings contain linearly decodable epistemic uncertainty. Sileo and Moens (2023) show that language models can be fine-tuned to classify words of estimative probability. However, none of these works extract a *continuously calibrated* probability axis from pretrained embeddings, grounded in human psychometric data, with syntactic-type controls.

We test this using the Mosteller & Youtz (1990) dataset, which provides empirically calibrated probability distributions for 53 verbal expressions (e.g., "likely" → median 71.1%, "possible" → median 38.5%, n=238 science writers). We embed these expressions in controlled sentence templates, extract difference vectors from bare claims, and test whether the resulting geometry encodes a calibrated probability axis.

Our contributions: (1) we show that probability is a continuously calibrated linear direction in pretrained sentence embeddings, not merely a classifiable feature; (2) we identify a critical syntactic-type confound where mixing constructions obscures the probability signal; (3) we validate against an independent meta-analysis (Vogel, 2022) achieving ρ = 0.99; (4) we demonstrate this across five architecturally diverse models, establishing the finding as a property of language rather than a training artifact.

## 2. Method

**Ground truth.** We use the Mosteller & Youtz (1990) dataset: 53 expressions with P25, Median, P75, and IQR from 238 science writers. We classify expressions into four syntactic types — predicative adjective (13 expressions: "certain," "likely," "possible," ...), frequency adverb (19: "always," "often," "rarely," ...), noun phrase (11: "high chance," "low probability," ...), and modal adverb (15: "probably," "certainly," "possibly," ...) — each embedded in a type-appropriate template (e.g., "It is {likely} that the experiment will succeed" for predicatives, "The experiment will {probably} succeed" for modal adverbs).

**Models.** Five architecturally diverse embedding models accessed via Ollama: nomic-embed-text v1.5 (768d, BERT+Matryoshka), embeddinggemma 300M (768d, decoder-derived), nomic-embed-text-v2-moe (768d, Mixture of Experts), mxbai-embed-large (1024d, BERT-large), and qwen3-embedding (4096d, Qwen3).

**Axis extraction.** For each syntactic type, we compute difference vectors (hedged embedding minus bare claim embedding), then train a supervised probability direction via ridge regression (λ=0.1) on the Mosteller medians. We calibrate with a linear mapping from projection to probability.

**Evaluation.** In-sample Spearman ρ, leave-one-out cross-validation (LOO) with bootstrap 95% confidence intervals (10,000 resamples), and cross-validation against the independent Vogel (2022) meta-analysis (21 studies, 1967–2018). We compare against baselines: random direction (mean of 100 random unit vectors), mean-difference direction (unsupervised centroid), and PCA PC1 (unsupervised best component).

## 3. Results

**The syntactic confound.** When all 53 expressions are analyzed together, PCA PC1 captures syntactic type (28.7% variance), not probability. The probability signal appears in PC2 (ρ = −0.68). This confound — that "There is a high probability that X" differs from "X will probably happen" primarily in syntax, not semantics — motivates within-type analysis. This finding echoes the polarity confounds identified in truth-direction work (Bürger et al., 2024).

**Within-type probability axes.** When syntax is held constant, probability becomes the dominant linear direction:

| Type (n) | In-sample ρ range | LOO ρ range | LOO 95% CI (best) |
|---|---|---|---|
| Predicative (13) | 0.90–0.99 | 0.75–0.91 | [0.65, 0.99] |
| Adverbial (19) | 0.92–0.96 | 0.70–0.92 | [0.77, 0.97] |
| Noun phrase (11) | 0.94–0.97 | 0.77–0.94 | [0.68, 1.00] |
| Modal (15) | 0.95–0.98 | 0.67–0.95 | [0.58, 0.98] |

All LOO correlations exceed the random-direction baseline (mean ρ ≈ 0.45) by a wide margin. PCA PC1 (unsupervised) alone achieves ρ up to 0.98 (noun phrases, nomic), demonstrating that probability is often the dominant axis of variation even without supervision.

**Vogel cross-validation.** Axes trained on Mosteller data predict the independent Vogel (2022) meta-analysis:

| Model | Predicative Vogel ρ | Predicative Vogel MAE |
|---|---|---|
| mxbai-embed-large | **0.991** | **5.1%** |
| embeddinggemma | 0.964 | 5.6% |
| qwen3-embedding | 0.909 | 5.6% |
| nomic-embed-text | 0.946 | 7.1% |
| nomic-v2-moe | 0.891 | 6.5% |

The mxbai result (ρ = 0.991, MAE = 5.1%) represents near-perfect transfer from one psychometric dataset to an independent meta-analysis spanning 50 years of research.

**Dimensional truncation.** On the Matryoshka-trained nomic model (Muennighoff et al., 2024), truncating from 768 to 64 dimensions degrades LOO ρ by only 0.027 (worst case), indicating the probability axis is concentrated in early dimensions and extractable with minimal computation.

**Novel phrase generalization.** The trained axes rank novel phrases not in Mosteller (e.g., "I think," "I doubt," "signs point to") with ρ = 0.75–0.94 against intuitive probability ratings across 5 models, demonstrating generalization beyond the calibration vocabulary.

## 4. Discussion

Our results demonstrate that sentence embedding models learn calibrated probability structure as an emergent geometric property — extending the truth-direction findings (Marks & Tegmark, 2023; Yu et al., 2025) from binary factuality to continuously graded epistemic modality.

The structure is **type-specific**: different syntactic constructions have different probability axes, forming a 4-dimensional epistemic subspace (analogous to the multi-dimensional truth cones of Yu et al., 2025). The four axes are moderately correlated (cosine 0.25–0.81), with predicative-modal alignment increasing with model size — suggesting larger models develop a more unified probability representation.

Unlike Sileo and Moens (2023), who show that models can be *fine-tuned* to classify words of estimative probability, we find that calibrated probability structure is *already present* in pretrained representations. And unlike Ji et al. (2025), who identify a verbal uncertainty direction calibrated to model behavior, our axis is calibrated to *human psychometric data* — bridging distributional geometry to cognitive science.

The convergence between human surveys (Mosteller, 1990; Vogel, 2022) and distributional geometry (five embedding models) suggests both measure the same underlying quantity: the communicative probability function of hedge expressions in English. A separate experiment (not detailed here) found that models do *not* encode interpretive precision (IQR; ρ ≈ 0 across 4 models) — they capture a single stable meaning per expression, unlike the bimodal human distribution for ambiguous terms like "possible."

**Limitations.** All experiments use template sentences; real-text evaluation is needed. The novel-phrase evaluation uses author intuitions rather than independent annotations. Cross-linguistic validation with multilingual models remains future work.

---

## References

Bolukbasi, T., Chang, K.-W., Zou, J., Saligrama, V., & Kalai, A. (2016). Man is to computer programmer as woman is to homemaker? Debiasing word embeddings. In *NeurIPS* (pp. 4349–4357).

Budescu, D. V., & Wallsten, T. S. (1995). Processing linguistic probabilities. *Psychology of Learning and Motivation*, 32, 275–318.

Bürger, L., et al. (2024). Truth is universal: Robust detection of lies in LLMs. *arXiv:2407.12831*.

Conneau, A., et al. (2018). What you can cram into a single $&!#* vector. In *Proceedings of ACL* (pp. 2126–2136).

Ji, Z., et al. (2025). Calibrating verbal uncertainty as a linear feature to reduce hallucinations. *arXiv:2501.07929*.

Lassiter, D. (2017). *Graded Modality*. Oxford University Press.

Lin, S. C., Hilton, J., & Evans, O. (2022). Teaching models to express their uncertainty in words. *TMLR*.

Marks, S., & Tegmark, M. (2023). The geometry of truth. *arXiv:2310.06824*.

Mikolov, T., Yih, W., & Zweig, G. (2013). Linguistic regularities in continuous space word representations. In *NAACL-HLT* (pp. 746–751).

Mosteller, F., & Youtz, C. (1990). Quantifying probabilistic expressions. *Statistical Science*, 5(1), 2–34.

Muennighoff, N., et al. (2024). Matryoshka representation learning. In *NeurIPS*.

Sileo, D., & Moens, M.-F. (2023). Probing neural language models for understanding of words of estimative probability. In *Proceedings of *SEM* (pp. 469–476).

Vogel, T., et al. (2022). Systematic review of verbal probability expressions.

Yu, S., et al. (2025). From directions to cones: Exploring multidimensional representations of propositional facts in LLMs. *arXiv:2501.14457*.

Zhou, K., et al. (2023). Navigating the grey area: How expressions of uncertainty and overconfidence affect language models. *arXiv:2302.13439*.
