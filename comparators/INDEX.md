# Comparator papers — index

> Working set of comparator PDFs read directly during the May 2026 paper-drafting session. The PDFs themselves are gitignored (we don't have redistribution rights for a public repo). This index lets a future session fetch them locally from the cited links.
>
> **To restore a working `comparators/` directory:** download each paper from the link below into this directory using the filename listed (the agent prompts and `paper.md` references reference these specific filenames).

---

## Linear features in decoder LLM internal states

### `ji2025-verbal-uncertainty-linear-feature-emnlp.pdf`
**Ji, Yu, Koishekenov, Bang, Hartshorn, Schelten, Zhang, Fung & Cancedda (2025)**
*"Calibrating Verbal Uncertainty as a Linear Feature to Reduce Hallucinations"*
EMNLP 2025, pp. 3769–3793. arXiv:2503.14477.
- ACL Anthology: https://aclanthology.org/2025.emnlp-main.187/
- arXiv: https://arxiv.org/abs/2503.14477

Closest direct comparator on the verbal-uncertainty *phenomenon*. Single linear "verbal uncertainty feature" extracted from decoder LLM residual streams via difference-in-means; calibrated to LLM-as-Judge model behavior aligned with Semantic Uncertainty; causal intervention via inference-time scaling of activations along that direction. Three QA datasets (TriviaQA, NQ-Open, PopQA) on three decoder LLMs (Llama-3.1-8B, Mistral-7B, Qwen2.5-7B). The paper's title is dangerously close to ours; differentiation in our §2 must be explicit and early.

### `valentin2025-frame-representation-hypothesis-tacl.pdf`
**Valois, Souza, Shimomoto & Fukui (2025)** [author convention may be "Valentin et al." in some indices — verify lead author from the PDF]
*"Frame Representation Hypothesis: Multi-Token LLM Interpretability and Concept-Guided Text Generation"*
TACL Vol. 13, pp. 1436–1458. doi:10.1162/TACL.a.48.
- ACL Anthology: https://aclanthology.org/2025.tacl-1.65/

Methodological background, not a direct novelty threat. Extends LRH from single-token to multi-token concepts via "frames"; Top-*k* Concept-Guided Decoding for steering on decoder LLM unembedding matrix. Best comparator on geometric-structure-in-representations methodology and on explicit theoretical positioning against Mikolov, Park 2024, Marks-Tegmark, Templeton, Bau, Elhage. The bar for "linear-structure-in-representations" papers right now.

### `lepori2026-modal-difference-vectors-iclr.pdf`
**Lepori, Hu, Dasgupta, Patel, Serre & Pavlick (2026)**
*"Is This Just Fantasy? Language Model Representations Reflect Human Judgments of Event Plausibility"*
ICLR 2026 (published as a conference paper). arXiv:2507.12553v3 dated 2026-04-27.
- arXiv: https://arxiv.org/abs/2507.12553

Strong recent comparator — *post-Feb-2026 publication that the earlier undermind summary mis-dated as a 2025 arXiv*. Modal difference vectors discriminating four scenario-level modal categories (probable / improbable / impossible / inconceivable for event plausibility) extracted from decoder LM activations; validated against expert labels and fine-grained human behavioral categorization. Three coordinates of differentiation from our work: phenomenon (event-plausibility categorization vs. verbal-probability hedge semantics), model class (decoder LM internals vs. frozen pooled sentence embeddings), calibration target (modal-category labels and behavior vs. numerical psychometric medians).

---

## Verbal-probability semantics elicited behaviorally from LMs

### `belem2024-linguistic-uncertainty-perceptions.pdf`
**Belém, Kelly, Steyvers, Singh & Smyth (2024)**
*"Perceptions of Linguistic Uncertainty by Language Models and Humans"*
arXiv:2407.15814v2 dated 2024-11-07. UC Irvine CS + Cog Sci.
- arXiv: https://arxiv.org/abs/2407.15814

Strongest behavioral comparator on the same phenomenon as our work. Prompted-output protocol with 10 LLMs and 94 baseline humans on 14 uncertainty expressions from Wallsten et al. 1986a/2008. Theory-of-mind framing distinguishes speaker's certainty from the participant's own belief about the proposition ("Sonia believes it is unlikely it will rain today" — quantify Sonia's certainty). Findings: 7/10 LLMs map uncertainty to probabilities in human-aligned ways; LLMs systematically more biased by prior knowledge (truth/falsehood of proposition) than humans are. Differentiation: method (prompted-output behavior elicitation vs. frozen-embedding linear axis extraction), model class (conversational LLMs vs. pretrained pooled sentence embeddings).

### `tang2024-estimative-uncertainty-llms.pdf`
**Tang, Shen & Kejriwal (2024)**
*"An Evaluation of Estimative Uncertainty in Large Language Models"*
arXiv:2405.15185v1 dated 2024-05-24. USC ISI. (Undermind cites *npj Complexity* — verify venue before final submission.)
- arXiv: https://arxiv.org/abs/2405.15185

Closest behavioral comparator on cross-linguistic verbal probability. 17 Words of Estimative Probability across GPT-3.5, GPT-4, Llama-2-7B, Llama-2-13B, ERNIE-4. Prompted-output protocol with human survey baseline. Cross-linguistic comparison: English vs. Chinese (with ERNIE-4 prompted in Chinese, GPT-4 prompted in both). Findings: human-LLM alignment for high-certainty WEPs but divergence on 11/12 mid-range WEPs (GPT-3.5/GPT-4); gendered-role prompts increase divergence; CoT prompting fails to improve LLM consistency under varying statistical uncertainty. Their cross-linguistic scope is English+Chinese (behavioral); ours is 8 typologically diverse languages (geometric ranking transfer).

---

## Pretrained sentence-embedding probing for human-aligned graded properties

### `jacobs2022-masked-lms-linguistic-uncertainty-scil.pdf`
**Jacobs, Hubbard & Federmeier (2022)**
*"Masked language models directly encode linguistic uncertainty"*
Proceedings of the Society for Computation in Linguistics (SCiL) 2022, pp. 225–228. (UB Linguistics + UIUC Beckman / Psychology.)
- SCiL Proceedings (held online February 7–9, 2022)

Strongest pooled-embedding precursor: same model class (mean-pooled RoBERTa) with linear probing against human-aligned uncertainty signal. *NB: undermind originally attributed this to "Maslej-Krešňáková et al." with no DOI; the actual lead author is Jacobs.* 282 leave-one-out logistic-regression *binary classifiers* trained to predict strong-vs-weak sentential constraint on the Federmeier et al. 2007 stimulus set, with the critical word masked; classifier output probability then correlated against cloze probability (ρ = 0.43) and cloze entropy (ρ = −0.32). Two-layer differentiation from our work: (i) phenomenon — next-word predictability given preceding context (sentential constraint) vs. verbal-probability hedge semantics ("probably" → 70%); (ii) method — binary classifier with continuous-target correlation vs. continuous regression directly onto psychometric medians yielding a calibrated linear axis. They never extract a calibrated probability axis.

### `chen2019-uncertain-nli.pdf`
**Chen, Jiang, Poliak, Sakaguchi & Van Durme (2019)**
*"Uncertain Natural Language Inference"*
arXiv:1909.03042v2 dated 2020-05-05. ACL 2020. doi:10.18653/v1/2020.acl-main.774. (JHU + Columbia + AI2.)
- arXiv: https://arxiv.org/abs/1909.03042

Refines NLI from categorical labels (entail/contradict/neutral) to direct prediction of subjective probability assessments. Re-labels SNLI subset as u-SNLI: 61,597 MTurk-elicited probability annotations on premise-hypothesis pairs via 10,000-step slider with logistic transformation. Direct fine-tuned scalar regression modeling. Best models approach human performance. Differentiation: phenomenon is NLI inference probability (likelihood that hypothesis follows from premise) rather than verbal probability semantics of hedge expressions; object is sentence-pair entailment rather than single-sentence template; method is fine-tuned scalar regression rather than frozen-embedding linear axis; no general probability axis identified, no psychometric cross-validation.

---

## Adjacent / TACL editorial reference

### `ying2025-labtom-tacl.pdf`
**Ying, Zhi-Xuan, Wong, Mansinghka & Tenenbaum (2025)**
*"Understanding Epistemic Language with a Language-augmented Bayesian Theory of Mind"*
TACL Vol. 13, pp. 613–637.
- ACL Anthology: https://aclanthology.org/2025.tacl-1.30/

TACL 2025 paper on epistemic-language semantics — methodologically very different from ours (Bayesian Theory of Mind cognitive model with grammar-constrained LLM decoding into a formal "epistemic language of thought"; 469 human-rated sentences in maze task). Cite as a TACL-published recent epistemic-language paper for editorial-rigor positioning, not as a methodological comparator. Useful as a model of TACL prose density and behavioral grounding rigor.

---

## Papers cited but PDFs not local

The following papers are cited in `paper.md` but were not downloaded — positioning is sufficient from the undermind audit (`Novelty_memo_for_hedge_embeddings.md`) and general knowledge:

- **Marks & Tegmark (2023)** "The Geometry of Truth," arXiv:2310.06824
- **Bürger et al. (2024)** "Truth is universal," arXiv:2407.12831
- **Yu et al. (2025)** "From Directions to Cones," COLM 2025, arXiv:2505.21800
- **Park et al. (2024)** LRH formalization
- **Sileo & Moens (2023)** "Probing neural language models for understanding of words of estimative probability," *SEM 2023, arXiv:2211.03358
- **Wang et al. (2024)** "Calibrating Expressions of Certainty," arXiv:2410.04315
- **Yang et al. (2026)** "Verbalizing LLM's Higher-order Uncertainty via Imprecise Probabilities"
- **Cohen, Fahn & de Melo (2025)** "Pretrained LLMs Learn Multiple Types of Uncertainty," arXiv:2505.21218
- **Cho, Wu, Costa & Koshiyama (2026)** "The Confidence Manifold"
- **Adarsh, Maistro & Lioma (2026)** "How Context Shapes Truth," arXiv:2601.06599
- **Ahdritz, Qin, Vyas, Barak & Edelman (2024)** "Distinguishing the Knowable from the Unknowable," arXiv:2402.03563
- **Maloney et al. (2024)** — cited within Belém et al. 2024 as closest direct comparator on numerical probability estimates from GPT-4 vs. humans with smaller stimulus set; download if a deeper read is needed.
- **Belrose, Schneider-Joseph, Ravfogel, Cotterell, Raff & Biderman (2023)** LEACE, NeurIPS 2023 — referenced for §4.4 concept-erasure methodology, even though we use rank-1 projection.
- **Schuster, Chen & Degen (2019)** "Harnessing the linguistic signal to predict scalar inferences," ACL 2020
- **Pei & Jurgens (2021)** "Measuring Sentence-Level and Aspect-Level (Un)certainty in Science Communications," EMNLP 2021, arXiv:2109.14776
- **Lee, Artzi, Choi & Zettlemoyer (2015)** "Event Detection and Factuality Assessment," EMNLP 2015
- **Stanovsky, Eckle-Kohler, Puzikov, Dagan & Gurevych (2017)** "Integrating Deep Linguistic Features in Factuality Prediction," ACL 2017
- **Bhatia (2016)** "Vector Space Semantic Models Predict Subjective Probability Judgments for Real-World Events," *Cognitive Science*
- **Shen, Jiang, Liu & Shi (2023)** Sen2Pro, arXiv:2306.02247
- **Mosteller & Youtz (1990)** "Quantifying probabilistic expressions," *Statistical Science* 5(1)
- **Vogel et al. (2022)** systematic review of verbal probability expressions
- **Wintle et al. (2019)** "Verbal probabilities: Very likely to be somewhat more confusing than numbers," *PLoS ONE*
- **Wallsten et al. (1986a, 1986b, 2008)** foundational verbal-probability psychometric studies
- **Lassiter (2017)** *Graded Modality* (OUP)
- **Schockaert (2022)** formal framework for embeddings as epistemic states
- **Mikolov, Yih & Zweig (2013)** linguistic regularities in word embeddings
- **Muennighoff et al. (2024)** Matryoshka representation learning, NeurIPS
- **Chen et al. (2024)** BGE M3-embedding, arXiv:2402.03216
- **Federmeier et al. (2007)** the cloze-completion stimulus set Jacobs 2022 builds on
