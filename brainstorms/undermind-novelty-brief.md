# Brief for Undermind: Novelty Framing — Hedge-Embeddings TACL Paper

> Self-contained brief for an external researcher (undermind report generator) to weigh in on whether our intended novelty framing holds up against the literature. Drafted May 2026 by the working session (Joseph + main agent + experiment-runner agent). Findings are verified against `results/exp08_*.txt`, `results/exp09_bge-m3_8lang.txt`, `results/exp10_nomic-v1.5.txt`, `results/exp11_mxbai.txt`, `results/exp11_qwen3.txt`, and `paper_validation.py` outputs in `results/validation_*.txt`.

---

## What we investigate

Whether **epistemic hedging** — verbal expressions of uncertainty like "probably," "certainly," "possibly" — manifests as **calibrated linear structure in pretrained pooled sentence embedding models** (BERT-family encoders, multilingual XLM-RoBERTa, decoder-derived pooled embedding heads). Specifically: can a supervised linear probe extract a continuous probability axis aligned with human psychometric data, and does that axis cross-validate against independent datasets, transfer cross-linguistically, and survive functional-validation tests (concept erasure)?

## Key empirical findings (verified)

**Within-type linear structure.** Within syntactic types — predicative adjective, frequency adverb, noun phrase, modal adverb — probability is the dominant linear direction in **5 architecturally diverse pretrained embedding models**: mxbai-embed-large (BERT-large, 1024d), nomic-embed-text-v1.5 (BERT+Matryoshka, 768d), embeddinggemma 300M (decoder-derived/Gemma, 768d), nomic-v2-moe (Mixture-of-Experts, 768d), qwen3-embedding (decoder-derived/Qwen3, 4096d). **Spearman ρ > 0.90 supervised, 0.67–0.95 leave-one-out** across types and models. Trained on Mosteller & Youtz (1990): 53 verbal probability expressions, n=238 science writers.

**Three-dataset psychometric triangulation.**
- Mosteller (1990) → Vogel (2022) meta-analysis of 21 studies 1967–2018: **ρ = 0.991, MAE = 5.1%** (mxbai-embed-large, predicative).
- Mosteller (1990) → Wintle (2019, n≈924): **ρ = 0.967, MAE = 3.6%**.
- Three independent psychometric datasets, three decades, converging on the same axis.

**Cross-linguistic ranking transfer.** English-trained axis applied zero-shot through bge-m3 (multilingual XLM-RoBERTa, 1024d, 100+ languages) to hedge expressions in **8 typologically diverse languages**: German, French, Spanish, Chinese, Japanese, Korean, Arabic, Hindi. **Mean ranking correlation |ρ| = 0.928**. Calibration MAE varies by language (Japanese modal MAE = 18%, Arabic modal MAE = 3.6%); we treat ranking transfer as the trustworthy claim, calibration as preliminary pending native-speaker psychometric validation.

**Syntactic-type confound (a methodological finding).** When all 53 Mosteller expressions are analyzed together, PCA PC1 captures **syntactic frame** (28.7% variance), not probability. The probability signal lives on PC2. Within-type analysis is essential. This parallels the polarity confound Bürger et al. (2024) report for truth directions in decoder LLMs.

**Functional validation via concept erasure.** Rank-1 projection erasure (e' = e - (e·v_A)v_A) of axis A from type-B embeddings, applied to v_B's calibrated decoder, evaluated against a **cosine-matched random-direction null** (random unit vectors r with |cos(r, v_B)| = |cos(v_A, v_B)|, isolating geometric overlap from functional content).
- **Mxbai-embed-large headline pair (predicative ↔ modal, cos = 0.880):** erasure raises MAE from 6.8 → 25.2pp (z_match = −11.9 against matched-random ρ); reverse direction 4.6 → 29.2pp (z_match = −8.5).
- **Cross-model robust signature: ΔMAE correlates with cos(v_A, v_B) at r = 0.92 (mxbai) and r = 0.86 (qwen3)** across 12 ordered cross-type pairs.
- **Methodological note:** qwen3 (4096d) shows rank-vs-magnitude dissociation — rank-1 erasure preserves Spearman ρ but compresses prediction magnitudes, so MAE is the load-bearing metric for this experiment, ρ z-scores a stricter secondary test that the headline pair clears decisively on mxbai.

**Null hypothesis controls** (`results/exp10_nomic-v1.5.txt`):
- Permutation test (1000 shuffles): **p ≤ 0.005** all syntactic types.
- Pure non-epistemic adjective control: LOO ρ = 0.31 vs. 0.87 for real hedges (mixed-with-leakage set ρ = 0.69 — discussed in Limitations).
- Random Uniform(0,100) labels: real LOO matches in ≤ 6/1000 trials (p ≤ 0.006).

**Robustness extras.** Matryoshka dimensional truncation: **LOO ρ drop = 0.027 at 64 dimensions** on nomic-embed-text-v1.5. Novel-phrase generalization: ρ = 0.75–0.94. Compound-hedge linear composition: cos = 0.82–0.89; negation breaks composition (cos = 0.22) — geometrically appropriate.

**Negative results.** IQR / interpretive-precision is **not** in the geometry (ρ ≈ 0 across 4 models) — embeddings encode a single stable meaning per expression, not the spread of human judgments. Ensemble combinations across the four within-type axes do not beat the single modal axis on novel-phrase generalization.

## Current framing intent

We are framing for **TACL** (rolling, monthly). The intended framing has three load-bearing moves:

1. **Substance-first abstract.** Lead with the empirical convergence story (three datasets, eight languages, five architectures, functional erasure). The paper's importance comes from what the evidence shows, not from a priority claim. Novelty appears as a protective subordinate clause near the end of the abstract or in §1, not as the rhetorical anchor.

2. **Model-class differentiation in §1, not in the headline.** The recent linear-feature interpretability literature has documented analogous structure in **decoder LLM internal states** — residual streams (Ji et al. 2025 EMNLP "Calibrating Verbal Uncertainty as a Linear Feature"; Marks & Tegmark 2023 "The Geometry of Truth"; Bürger et al. 2024 "Truth is Universal"; Yu et al. 2025 COLM "From directions to cones"), unembedding matrices (Valentin et al. 2025 TACL "Frame Representation Hypothesis"). All of these works rely on token-level causal intervention (steering, activation patching) — natural for decoder LLMs because they generate tokens. **Pretrained pooled sentence embedding models** compress entire sentences to single vectors and lack a token-generation pathway; representation-level intervention (concept erasure) is the natural model-class analogue. We position our work as a parallel-territory contribution: same family of finding, different model class, different intervention methodology, calibrated against human psychometric data rather than against model behavior.

3. **Convergent-measurements closing frame.** The convergence of three independent psychometric datasets (50+ years of replication), six embedding models, eight languages, and the functional-erasure result is consistent with distributional embedding geometry and human probability semantics being **convergent measurements of one underlying epistemic dimension** — a measurement-theoretic perspective complementing the standard "embeddings reflect training data" framing.

## Specific novelty claim (current draft)

> *"To our knowledge this is the first demonstration of continuously calibrated probability structure aligned with human psychometric data in pretrained pooled sentence embedding models."*

Each qualifier is meant to do load-bearing work:

- **continuously calibrated** — distinguishes from classification (e.g., Sileo & Moens 2023 fine-tune for words-of-estimative-probability classification, not continuous regression).
- **aligned with human psychometric data** — distinguishes from Ji et al. 2025's calibration to model-internal LLM-as-a-Judge verbal-uncertainty scores aligned with Semantic Uncertainty.
- **pretrained** — distinguishes from fine-tuned approaches (Sileo & Moens; Lin et al. 2022 trained classifiers).
- **pooled sentence embedding models** — distinguishes from decoder LLM internal states (residual streams, unembedding matrices, layer-wise activations).

## Specific questions for undermind

1. **Does the narrowed novelty claim hold up against the literature?** Is there prior work that does the conjunction of (continuous calibration, on pretrained pooled sentence embeddings, aligned with psychometric data, for verbal probability or epistemic hedging) that we are missing or under-positioning? We have already considered Sileo & Moens 2023, Lin et al. 2022, Ji et al. 2025, Valentin et al. 2025, Marks & Tegmark 2023, Bürger et al. 2024, Yu et al. 2025, Park et al. 2024 (LRH formalization), Schockaert 2022 (formal framework for embeddings as epistemic states), Wallsten et al. 1986, Lassiter 2017, Mikolov et al. 2013.

2. **Is the model-class distinction (decoder-internal-state vs. pooled-sentence-embedding) a meaningful contribution from the field's perspective?** We believe it's a real territorial gap because decoder-LLM activation steering does not directly apply to embedding models, but a thorough audit might find adjacent sentence-rep work that already covers this ground (NLI/factuality/stance/hedging probing on pooled outputs).

3. **Are the three named comparators (Ji, Valentin, Marks-Tegmark) the right ones to position against in the abstract?** Should we add or substitute others? In particular: any post-February 2026 work on linear feature extraction for uncertainty/probability/calibration that we should know about?

4. **How does the substance-first framing read for our likely TACL audience** (linear-feature interpretability community + verbal-probability psychometrics + sentence-embedding researchers)? Is the decision to demote the priority claim to a subordinate clause appropriate, or should the priority claim be load-bearing for this audience?

5. **What is the strongest novelty claim the literature actually supports?** Is "first demonstration of continuously calibrated probability structure in pretrained pooled sentence embeddings aligned with human psychometric data" defensible as written, or does it need to narrow further (e.g., add "validated against three independent psychometric datasets" or "with cross-linguistic transfer") — or could it broaden (e.g., drop "pooled" if the contribution holds for any encoder-based representation)?

6. **Reviewer-bait we should pre-empt.** Beyond the obvious objections we've identified — template-only evaluation, author-rated novel phrases, mixed non-epistemic adjective leakage, native-speaker non-validation in cross-linguistic, no causal intervention in the decoder-LLM steering sense — are there literature-level objections specific to this contribution that we are not yet seeing?

## Local artifacts (for verification, if useful)

- `cmcl-abstract-draft.md` — current draft, CMCL 2026 non-archival extended abstract version.
- `paper.md` — TACL working draft (modified Option B framing, in progress).
- `tactical-notes.md` — earlier strategic / venue / decision-category analysis (long, optional).
- `brainstorms/concept-erasure-experiment-design.md` — design rationale for experiment_11.
- `brainstorms/reframed-contribution-options.md` — three abstract framing options the modified Option B choice was made from.
- `comparators/` — full PDFs of Ji 2025, Valentin 2025, Ying 2025 (LaBToM TACL), pre-read.
- `results/` — raw experiment outputs.
- `FINDINGS-01.md` through `FINDINGS-06.md` — internal results write-ups.
- `literature-review-undermind.md` — earlier (February 2026) automated literature review.
