# PLAN.md — Master Plan for Paper and Research Program

**Last updated:** 2026-02-14

---

## Project Status at a Glance

```
COMPLETED EXPERIMENTS          PAPER-READY VALIDATION        PROPOSALS (NOT YET RUN)
───────────────────────        ──────────────────────        ───────────────────────
01  Hedge consistency    ✓     Baselines (5 models)    ✓     Possible bimodality
01b Within-type axes     ✓     LOO cross-validation    ✓     Causal measurement model
02  Novel phrases        ✓     Bootstrap 95% CIs       ✓     Corpus grounding
03  Modal axis           ✓     Vogel within-type       ✓     LLM decomposition
04  IQR-consistency      ✓     Mosteller-only modal    ✓     Cross-linguistic
05  Ensemble             ✓     Saved results (7 files) ✓     Self-report truth axis
06  Compound hedges      ✓                                   Fast/slow/authentic
07  Matryoshka truncation✓
```

---

## I. What We Have (Completed Work)

### Core Experiments (Jan 23, 2026)

| # | Script | Finding | Status |
|---|---|---|---|
| 01 | `experiment_01_hedge_direction.py` | "Probably" has consistent direction (cos 0.68); mixed types confound probability signal | Complete, 1 model |
| 01b | `experiment_01b_within_type.py` | **THE KEY FINDING:** Within syntactic types, probability IS the dominant linear axis (ρ > 0.90 supervised, 5 models) | Complete, 5 models |
| 02 | `experiment_02_novel_phrases.py` | Trained axes generalize to novel phrases (ρ = 0.75–0.90); calibration compressed for natural syntax | Complete, 5 models |
| 03 | `experiment_03_modal_axis.py` | 4th axis (modal adverb) solves compression: MAE drops from 30–43% to 3.2–8.5% | Complete, 5 models |
| 04 | `experiment_04_iqr_consistency.py` | **NEGATIVE:** IQR/interpretive precision NOT in geometry; models encode single stable meaning per phrase | Complete, 4 models |
| 05 | `experiment_05_ensemble.py` | **MODEST NEGATIVE:** No ensemble beats modal-only on novel phrases | Complete, 3 models |
| 06 | `experiment_06_compound_hedges.py` | Compound hedges compose ~linearly in direction (cos 0.82–0.89), sub-linearly in magnitude; negation breaks it (cos 0.22) | Complete, 2 models |

Findings documented in `FINDINGS-01.md` through `FINDINGS-06.md`.

### Paper Validation (Feb 13, 2026)

| Component | Script | Result | Status |
|---|---|---|---|
| Baselines | `paper_validation.py` | Supervised ridge (ρ 0.90–0.99) dominates random (ρ ~0.45), mean-diff (ρ ~0.5), PCA PC1 (ρ ~0.7–0.98) | Complete, 5 models |
| Leave-one-out CV | `paper_validation.py` | LOO ρ = 0.70–0.94 across all models and types | Complete, 5 models |
| Bootstrap 95% CIs | `paper_validation.py` | All CIs exclude zero; LOO CIs mostly above 0.3 | Complete, 5 models |
| Vogel cross-validation | `paper_validation.py` | Train Mosteller → test Vogel: **ρ up to 0.991** (mxbai predicative), MAE 5.1% | Complete, 5 models |
| Mosteller-only modal | `paper_validation.py` | 8-expression axis still works; transfers to author-estimated phrases (ρ 0.63–0.96) | Complete, 5 models |
| Matryoshka truncation | `experiment_07_matryoshka_truncation.py` | Axis survives to 64d on MRL model (worst LOO drop = 0.027); mxbai OK at 128d | Complete, 2 models |

Raw outputs saved in `results/`.

### Models Tested

| Model | Dims | Architecture | Used in |
|---|---|---|---|
| nomic-embed-text:v1.5 | 768 | BERT + Matryoshka | All experiments |
| embeddinggemma:300m | 768 | Decoder-derived (Gemma) | Exp 01b–05, validation |
| nomic-embed-text-v2-moe | 768 | Mixture of Experts | Exp 01b–05, validation |
| mxbai-embed-large | 1024 | BERT-large variant | All experiments |
| qwen3-embedding | 4096 | Qwen3 decoder | Exp 01b–05, validation |

---

## II. The Paper (Immediate Goal)

### Target
**arXiv (cs.CL, cs.AI)** → then EMNLP/ACL/NAACL 2026

### Title
"Epistemic Hedging as Calibrated Linear Structure in Sentence Embedding Space"

### What's Ready to Write Now

The empirical paper has all the data it needs:
- Core finding with 5-model replication
- Baselines proving the method adds value over trivial approaches
- LOO cross-validation with bootstrap CIs
- Independent cross-validation (Vogel, ρ = 0.991)
- Matryoshka truncation showing practical applicability
- Negative results (IQR, ensemble) adding depth
- Compound composition and negation analysis

### What Would Strengthen the Paper Before Submission

| Item | Impact | Scope | Depends on |
|---|---|---|---|
| **Figures** (scatterplots, PCA viz, truncation curves) | High — reviewers want visual evidence | 1 session | Nothing |
| **Real-text evaluation** (20–30 hedged sentences from actual documents) | High — addresses "template-only" limitation | 2–3 sessions (collection + annotation + testing) | Native annotators or LLM-assisted rating |
| **"Possible" bimodality probe** | Medium — enriches the narrative | 0.5 session | Nothing |
| **Cross-linguistic pilot** (3 languages, 1 multilingual model) | Medium-high — dramatically strengthens universality claim | 1–2 sessions | Multilingual embedding model in Ollama or HF |
| **LLM decomposition** (ask LLMs for P(claim)) | Medium — adds a third measurement class | 1 session | API access to Claude/GPT-4/Llama |

### Literature-Informed Priorities (from undermind.ai review, Feb 2026)

The automated literature review (`literature-review-undermind.md`) confirms our
work fills a clearly identified gap. It also reveals what reviewers in this space
will expect. The following five items are ordered by impact-per-effort for making
the paper more compelling at top venues (ICLR, TACL, EMNLP):

**1. Causal intervention along the probability axis.**
Ji et al. (2025) and Marks & Tegmark (2023) both do causal interventions —
they manipulate activations along discovered directions and show behavioral
effects. We don't do this. For CMCL this is fine; for ICLR/TACL a reviewer
may ask. Requires open-weights LLM access (can't edit frozen embedding model
activations). Would involve: add/subtract a scaled probability direction in
an LLM's residual stream and show that the model's output hedging changes.
**Scope:** 2–3 sessions. **Resources:** Open-weights LLM (Llama 3, Qwen 2.5),
GPU, interpretability tooling (TransformerLens or hooks).

**2. Wintle et al. (2019) as a THIRD cross-validation source.**
We cross-validate Mosteller → Vogel. Adding Wintle et al. (n≈924, more recent,
different methodology) as a third independent validation source would make the
convergent measurement argument airtight. Just map their expressions to ours
and test projections.
**Scope:** 0.5 session. **Resources:** Download Wintle data (or extract from
paper), match expressions to our templates.

**3. Explicit comparison to Ji et al.'s verbal uncertainty direction.**
Reviewers will ask how our probability axis relates to Ji et al.'s VU direction.
If we can show they're related but distinct (VU = binary confident/uncertain;
ours = continuously calibrated probability), that's a clean differentiation.
If our axis is more informative (predicts Mosteller medians better than a
VU-style probe), even better. This may not require reproducing their exact
method — even a conceptual comparison with citations suffices for a workshop
paper, but an empirical comparison strengthens the full paper.
**Scope:** 1 session (conceptual) to 2 sessions (empirical replication).
**Resources:** For empirical: open-weights LLM, their probe methodology.

**4. Syntactic-type confound as a general contribution to probing methodology.**
The polarity confound identified by Bürger et al. (2024) for truth directions
is analogous to our syntactic confound. We can frame our contribution partly
as: "Here's another confound that matters when probing for semantic properties
in embedding space — syntactic realization type." This generalizes beyond
hedging and increases the paper's relevance to the broader probing/
interpretability community.
**Scope:** 0 additional sessions (framing, not experiments). Just write it up.

**5. Non-template evaluation (reinforced by literature context).**
All truth-direction papers also use controlled templates, so this is somewhat
accepted in the field. But we should acknowledge it as a limitation and, if
possible, show even a small pilot on naturalistic text. For a cognitive modeling
audience (CMCL), naturalistic text is especially valued.
**Scope:** 2–3 sessions if done properly. **Resources:** Corpus of hedged
sentences, annotators or LLM-assisted probability ratings.

### Paper Structure (from `feedback-analysis.md`)

1. Introduction — the hypothesis and why it matters
2. Related Work — linear structure, verbal probability, uncertainty in NLG, probing
3. Experimental Setup — Mosteller data, syntactic types, models, method, baselines
4. Experiments 1–3 — core positive results
5. Experiments 4–6 — negative results, composition, extensions
6. Experiment 7 — truncation (practical applicability)
7. Discussion — what the geometry encodes, limitations, philosophical implications
8. Conclusion

---

## III. Proposals (Not Yet Run)

### A. "Possible" Bimodality Probe
**File:** `EXPERIMENT-PROPOSAL-possible-bimodality.md`

**Question:** When modifiers disambiguate "possible" toward one reading ("barely possible" vs. "entirely possible"), does the projection track the intended reading?

**Why it matters:** Tests whether geometry encodes the full semantic continuum of ambiguous words. Partial evidence exists (Exp 06: "quite possibly" → 59–67%, up from bare 39–45%).

**Scope:** 0.5 session. ~25 phrases, 3 models, straightforward projection analysis.
**Resources:** Just Ollama. No external data needed.
**Relevance to paper:** Enriches the IQR-negative finding with a positive complement. Could be a subsection of the Discussion or a brief Experiment 8.

---

### B. Causal Measurement Model
**File:** `EXPERIMENT-PROPOSAL-causal-measurement-model.md`

**Question:** Are Mosteller surveys and embedding geometry both noisy measurements of the same latent "true probability semantics"? Which instrument is more precise?

**Why it matters:** Reframes the paper from "validation" to "convergent measurement." Could reveal that embeddings are MORE precise than psychometric surveys for some expressions.

**Scope:** 1–2 sessions. Hierarchical Bayesian model (~50 parameters, ~200 observations) in Stan or PyMC. Data already exists (Mosteller + Vogel + 5 model LOO predictions).
**Resources:** Stan/PyMC installation. Familiarity with hierarchical models.
**Relevance to paper:** Could be a section in the Discussion (conceptual framing) or a supplementary analysis (formal model). Or a companion paper.

---

### C. Corpus Grounding
**File:** `EXPERIMENT-PROPOSAL-corpus-llm-philosophy.md` (Section I)

**Question:** What distributional patterns in training corpora give rise to the geometric probability axis? Do high-probability hedge words co-occur with positive/confirmed contexts more than low-probability ones?

**Why it matters:** Closes the causal loop: training text → distributional patterns → embedding geometry → probability calibration.

**Scope:** 2–3 sessions. Requires downloading a large corpus sample (Wikipedia, or a slice of The Pile/C4), extracting ~1000 contexts per hedge expression, computing co-occurrence statistics.
**Resources:** ~50GB disk for corpus. Text processing pipeline.
**Relevance to paper:** Future work section. Or a companion paper on the mechanisms.

---

### D. LLM Decomposition
**File:** `EXPERIMENT-PROPOSAL-corpus-llm-philosophy.md` (Section II)

**Question:** When LLMs explicitly decompose hedged statements into claim + probability, how do their estimates compare to our geometric projections?

**Why it matters:** Creates a third measurement class (explicit LLM reasoning) alongside surveys (metacognitive) and geometry (distributional). Three-way agreement would be strong evidence for a real latent quantity.

**Scope:** 1 session. ~60 statements, 3–4 LLMs, zero-shot prompting, correlation analysis.
**Resources:** API access to Claude, GPT-4, and/or local Llama/Qwen instruct models.
**Relevance to paper:** Could go in the paper as a validation experiment. Quick to implement, high impact.

---

### E. Cross-Linguistic Validation
**File:** `EXPERIMENT-PROPOSAL-cross-linguistic.md`

**Question:** In multilingual embedding models, do hedge expressions from different languages project onto the same probability axis? Is the structure a property of language universally, or just English?

**Why it matters:** The strongest possible evidence for the philosophical claim. If "wahrscheinlich" and "可能" project correctly onto an English-trained axis, the structure is language-universal.

**Scope:** 1–2 sessions for a pilot (3 languages, 1 multilingual model). Full study (10+ languages, calibration per language) would be a separate paper.
**Resources:** Multilingual embedding model (multilingual-e5-large via HuggingFace or Ollama). Native speaker verification for template naturalness (or LLM-assisted).
**Relevance to paper:** A successful pilot could go in the paper as Experiment 8. Even mentioning the prediction as future work strengthens the Discussion.

---

### F. Self-Report Truth Axis
**File:** `EXPERIMENT-PROPOSAL-self-report-truth-axis.md`

**Question:** Do unhedged assertions contain a latent "confidence axis" that correlates with the model's own metacognitive assessment of truth?

**Why it matters:** Extends hedging geometry from *expressed* uncertainty to *latent* confidence. If found, it means embedding geometry encodes epistemic stance even without explicit hedge words.

**Scope:** 2–3 sessions. 50–100 claims across confidence categories, embedding + LLM self-report elicitation, PCA/probe analysis.
**Resources:** An instruct-tuned model from the same family as the embedding model (for self-report). Open-weights model preferred (for hidden state extraction).
**Relevance to paper:** Separate paper or major follow-up. Too large for the current submission.

---

### G. Fast/Slow/Authentic Response Geometry
**File:** `EXPERIMENT-PROPOSAL-fast-slow-authentic.md`

**Question:** When a model responds under different cognitive framings (instinctive, deliberate, authentic), do the embeddings reveal structure correlated with genuine epistemic grounding?

**Why it matters:** Tests whether deliberation and integration have geometric signatures. Connects to interpretability and model phenomenology.

**Scope:** 3–4 sessions. 40–60 questions × 5 phases × multiple models. Requires careful prompt engineering and open-weights model access for hidden state extraction.
**Resources:** Open-weights instruct model (Llama 3.3 70B recommended). GPU for inference. Interpretability tooling (TransformerLens or hooks).
**Relevance to paper:** Separate paper entirely. Connected thematically but methodologically distinct.

---

### H. The Philosophical Argument
**File:** `EXPERIMENT-PROPOSAL-corpus-llm-philosophy.md` (Section III)

**Claim:** Degree of belief is not merely a philosophical *interpretation* of probability — it is a measurable geometric property of natural language. The Bayesian prior has a linguistic grounding.

**Why it matters:** Potentially the most significant intellectual contribution of the entire research program. Connects formal epistemology (de Finetti, Ramsey, Jaynes) to computational linguistics (distributional semantics) via empirical measurement.

**Scope:** Not an experiment — it's an argument. Requires 1–2 sessions of writing, drawing on the empirical results. Strengthened by every additional measurement instrument (LLM decomposition, cross-linguistic, corpus analysis).
**Resources:** Engagement with the epistemology and formal semantics literature (Lassiter, Kratzer, Goodman & Frank).
**Relevance to paper:** Could be the heart of the Discussion section (lighter touch) or developed into a companion paper (full treatment).

---

## IV. Decision Map — What Goes Where

### The Immediate Paper (arXiv)

```
DEFINITELY IN                    PROBABLY IN                   FUTURE WORK MENTION
─────────────                    ───────────                   ────────────────────
Experiments 01b–07               "Possible" bimodality probe   Corpus grounding
Baselines + LOO + CIs           LLM decomposition             Self-report truth axis
Vogel cross-validation          Cross-linguistic pilot         Fast/slow/authentic
Matryoshka truncation           Philosophical argument (lite)  Full causal model
Negative results (IQR, ens.)    Real-text evaluation           Full cross-linguistic
Compound composition                                           Domain shift (medical)
```

### Companion Paper: "Measuring Meaning"
The causal measurement model + philosophical argument + corpus grounding + LLM decomposition + cross-linguistic evidence → a paper about embedding geometry as a measurement instrument for linguistic semantics, with probability as the case study.

### Companion Paper: "Latent Confidence"
Self-report truth axis + fast/slow/authentic → a paper about geometric encoding of model confidence beyond explicit hedging.

---

## V. Recommended Path Forward

### Phase 1: Finish the arXiv paper (target: 3–5 sessions)

| Session | Work |
|---|---|
| **1** | Figures (scatterplots, PCA viz, truncation curves, cross-model comparison). "Possible" bimodality probe (quick experiment, ~25 phrases). |
| **2** | LLM decomposition (ask Claude/GPT-4/Llama for probability ratings on ~60 hedged statements). Cross-linguistic pilot if a multilingual model is readily available. |
| **3** | Write Methods + Results + Figures. This is the bulk of the paper — the data is all in hand. |
| **4** | Write Introduction + Related Work + Discussion. Integrate the philosophical argument at Discussion depth (not full treatment). |
| **5** | Real-text evaluation (if annotators available). Final polish. Supplementary materials. Submit to arXiv. |

**Session** = one focused working block with Claude Code (2–4 hours of interactive work).

### Phase 2: Strengthen and submit to venue (2–3 additional sessions)

- Respond to any early feedback from arXiv
- Run real-text evaluation if not done in Phase 1
- Full cross-linguistic pilot (3 languages)
- Submit to EMNLP/ACL/NAACL

### Phase 3: Companion papers (longer term)

- Causal measurement model paper (needs Stan/PyMC work)
- Self-report truth axis + fast/slow/authentic (needs open-weights model infra)
- Full cross-linguistic study (needs native speaker collaboration)

---

## VI. File Index

### Experiments (completed)
| File | Description |
|---|---|
| `experiment_01_hedge_direction.py` | Part A (consistency) + Part B (mixed-type PCA) |
| `experiment_01b_within_type.py` | Within-type analysis — the core finding |
| `experiment_02_novel_phrases.py` | Novel phrase generalization |
| `experiment_03_modal_axis.py` | Modal adverb axis + subspace analysis |
| `experiment_04_iqr_consistency.py` | IQR-consistency test (negative) |
| `experiment_05_ensemble.py` | Ensemble axis combination (modest negative) |
| `experiment_06_compound_hedges.py` | Compound hedge composition |
| `experiment_07_matryoshka_truncation.py` | Dimensional truncation test |
| `paper_validation.py` | Baselines, LOO, bootstrap CIs, Vogel validation |

### Findings
| File | Description |
|---|---|
| `FINDINGS-01.md` | Within-type probability axes (5 models, full tables) |
| `FINDINGS-02.md` | Novel phrase projection (5 models) |
| `FINDINGS-03.md` | Modal axis solution (5 models, MAE 3.2–8.5%) |
| `FINDINGS-04.md` | IQR-consistency falsified (4 models) |
| `FINDINGS-05.md` | Ensemble: modal-only is near-optimal |
| `FINDINGS-06.md` | Compound composition + negation geometry |

### Results (raw output, version-controlled)
| File | Description |
|---|---|
| `results/validation_nomic-v1.5.txt` | Full validation output for nomic |
| `results/validation_mxbai.txt` | Full validation output for mxbai |
| `results/validation_qwen3.txt` | Full validation output for qwen3 |
| `results/validation_gemma.txt` | Full validation output for gemma |
| `results/validation_moe.txt` | Full validation output for MoE |
| `results/truncation_nomic-v1.5.txt` | Truncation results for nomic (MRL) |
| `results/truncation_mxbai.txt` | Truncation results for mxbai (non-MRL) |

### Proposals (not yet run)
| File | Description | Scope |
|---|---|---|
| `EXPERIMENT-PROPOSAL-possible-bimodality.md` | Disambiguating "possible" with modifiers | 0.5 session |
| `EXPERIMENT-PROPOSAL-causal-measurement-model.md` | Hierarchical Bayesian measurement model | 1–2 sessions |
| `EXPERIMENT-PROPOSAL-corpus-llm-philosophy.md` | Corpus grounding + LLM decomposition + philosophical argument | 1–3 sessions per section |
| `EXPERIMENT-PROPOSAL-cross-linguistic.md` | Cross-linguistic probability geometry | 1–2 sessions (pilot) |
| `EXPERIMENT-PROPOSAL-self-report-truth-axis.md` | Latent confidence in unhedged assertions | 2–3 sessions |
| `EXPERIMENT-PROPOSAL-fast-slow-authentic.md` | Deliberation geometry | 3–4 sessions |

### Planning and context
| File | Description |
|---|---|
| `PLAN.md` | This file — master plan |
| `EXPERIMENT-PLAN.md` | Original experiment plan with results summary |
| `feedback-analysis.md` | Paper structure, venue targets, pre-submission checklist |
| `CLAUDE.md` | Guidance for Claude Code working in this repo |

### Ground truth and theory
| File | Description |
|---|---|
| `docs/mosteller_youtz_1990_full.csv` | Primary ground truth (53 expressions, n=238) |
| `docs/vogel_2022_systematic_review.csv` | Cross-validation data (21 studies) |
| `docs/medical_context_metaanalysis.csv` | Domain shift data |
| `docs/epistemic-geometry-hedging-as-linear-structure.md` | Original hypothesis document |
| `docs/epistemic-claim-calibration-first-principles.md` | Bayesian calibration framework |
| `docs/verbal-probability-calibration.md` | Calibration guide with tier system |

---

## VII. Key Numbers (for quick reference)

### Headline results
- **Supervised within-type ρ:** 0.90–0.99 (in-sample), 0.70–0.94 (LOO)
- **Vogel cross-validation:** ρ = 0.991 (mxbai predicative)
- **Modal axis MAE:** 3.2–8.5% across 5 models
- **Random baseline ρ:** ~0.45 (demonstrates supervised axis adds real value)
- **Novel phrase ranking:** ρ = 0.75–0.94 (5 models)
- **Compound composition:** cos 0.82–0.89 (direction), magnitude ratio ~0.75
- **Truncation:** LOO ρ drops only 0.027 at 64d on MRL model
- **IQR-consistency:** ρ ≈ 0 (clean negative — models don't encode interpretive variance)
- **Ensemble vs. modal-only:** Modal wins (no ensemble improves on novel phrases)
