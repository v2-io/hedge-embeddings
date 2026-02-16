# PLAN.md — Master Plan for Paper and Research Program

**Last updated:** 2026-02-16

---

## Project Status at a Glance

```
COMPLETED EXPERIMENTS          PAPER-READY VALIDATION        PROPOSALS (NOT YET RUN)
───────────────────────        ──────────────────────        ───────────────────────
01  Hedge consistency    ✓     Baselines (5 models)    ✓     Causal measurement model
01b Within-type axes     ✓     LOO cross-validation    ✓     Corpus grounding
02  Novel phrases        ✓     Bootstrap 95% CIs       ✓     LLM decomposition
03  Modal axis           ✓     Vogel within-type       ✓     Self-report truth axis
04  IQR-consistency      ✓     Mosteller-only modal    ✓     Fast/slow/authentic
05  Ensemble             ✓     Saved results (17 files)✓
06  Compound hedges      ✓
07  Matryoshka truncation✓     NEW EXPERIMENTS (Feb 2026)
08  Bimodality + Wintle  ✓     ─────────────────────────
09  Cross-linguistic (8) ✓     Bimodality probe      ✓
10  Null hypothesis      ✓     Wintle 3rd validation ✓
                               8-language transfer   ✓
                               Permutation tests     ✓
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

### New Experiments (Feb 14–16, 2026)

| # | Script | Finding | Status |
|---|---|---|---|
| 08a | `experiment_08_bimodality_wintle.py` | **Bimodality probe:** Modifiers traverse the "possible" ambiguity range. "Barely possible" → 7–13%, "eminently possible" → 63–70%. Spread 20–32pp. ρ = 0.83–0.89 vs. expected. | Complete, 3 models |
| 08b | `experiment_08_bimodality_wintle.py` | **Wintle cross-validation:** Third independent dataset (n≈924). Mosteller → Wintle ρ up to 0.967 (mxbai), MAE 3.5–6.9%. Three-dataset convergence established. | Complete, 3 models |
| 09 | `experiment_09_cross_linguistic.py` | **Cross-linguistic transfer:** English axis → 8 languages zero-shot. Mean |ρ| = 0.928. Korean predicative ρ = 1.000. Hindi modal ρ = 1.000. Arabic modal MAE = 3.6%. Axis alignment cos 0.63–0.94. | Complete, bge-m3 (8 languages) |
| 10 | `experiment_10_null_hypothesis.py` | **Null confirmed:** Permutation test p ≤ 0.005 all types. Non-epistemic adjectives LOO ρ = 0.31 (vs. 0.87 real). Random labels p ≤ 0.006. Signal is genuinely epistemic. | Complete, 1 model |

Raw outputs saved in `results/`.

### Models Tested

| Model | Dims | Architecture | Used in |
|---|---|---|---|
| nomic-embed-text:v1.5 | 768 | BERT + Matryoshka | Exp 01–10, all validation |
| embeddinggemma:300m | 768 | Decoder-derived (Gemma) | Exp 01b–05, 08, validation |
| nomic-embed-text-v2-moe | 768 | Mixture of Experts | Exp 01b–05, validation |
| mxbai-embed-large | 1024 | BERT-large variant | Exp 01–08, validation, truncation |
| qwen3-embedding | 4096 | Qwen3 decoder | Exp 01b–08, validation |
| bge-m3 | 1024 | XLM-RoBERTa (multilingual, 100+ langs) | Exp 09 (cross-linguistic) |

---

## II. The Paper (Immediate Goal)

### Target
**CMCL 2026 non-archival abstract** (deadline: Feb 25, 2026) → then **TACL**
(rolling) or **ICLR 2027** (~Sep 2026) for the full paper.

See `workshops.md` for complete venue analysis.

### Title
"Calibrated Probability as Emergent Linear Structure in Sentence Embeddings"

### What's Ready to Write Now

The empirical paper is comprehensively supported:
- Core finding with 5-model replication + LOO + baselines + bootstrap CIs
- Three-dataset cross-validation (Mosteller → Vogel ρ = 0.991, Mosteller → Wintle ρ = 0.967)
- Cross-linguistic transfer across 8 typologically diverse languages (mean zero-shot |ρ| = 0.928)
- Null hypothesis tests (permutation p ≤ 0.005, non-epistemic control, random label control)
- Bimodality probe showing modifiers traverse the ambiguity spectrum
- Matryoshka truncation showing practical applicability (64d sufficient)
- Negative results (IQR, ensemble) adding theoretical depth
- Compound composition and negation analysis

### What Would Strengthen the Paper Further

| Item | Impact | Scope | Status |
|---|---|---|---|
| **Figures** (scatterplots, PCA viz, truncation curves, cross-lingual heatmap) | High | 1 session | Not started |
| **Real-text evaluation** (20–30 hedged sentences from actual documents) | Medium-high | 2–3 sessions | Not started |
| **LLM decomposition** (ask LLMs for P(claim)) — third measurement class | Medium | 1 session | Not started |
| **Native speaker verification** of cross-linguistic templates | Medium | 1 session + collaborators | Not started |
| **Causal intervention** along probability axis in open-weights LLM | High for ICLR | 2–3 sessions + GPU | Not started |

### Literature-Informed Priorities (from undermind.ai review, Feb 2026)

The automated literature review (`literature-review-undermind.md`) confirms our
work fills a clearly identified gap. No existing paper constructs an epistemic-
hedging axis calibrated to human psychometric probability scales.

Key papers to cite and position against:
- **Ji et al. 2025** — VU direction in residual streams (closest work; ours is
  calibrated to human data, not model behavior)
- **Sileo & Moens 2023** — WEP classification (they fine-tune; we show structure
  is already present in pretrained embeddings)
- **Marks & Tegmark 2023** — truth direction (our probability axis is the graded
  epistemic analog)
- **Yu et al. 2025** — truth cones (our 4-axis subspace parallels their multi-
  dimensional truth structure)
- **Bürger et al. 2024** — cross-model truth robustness (precedent for our
  cross-architecture consistency)
- **Wintle et al. 2019** — psychometric verbal probability (our third validation
  dataset)
- **Zhou et al. 2023** — most-cited foundational paper (72% reference rate)
- **Schockaert 2022** — formal framework for embeddings as epistemic states

Five literature-informed priorities:

1. **Causal intervention** — highest impact for ICLR, requires LLM infra (2–3 sessions)
2. **Wintle cross-validation** — ✓ DONE (ρ up to 0.967)
3. **Comparison to Ji et al.'s VU direction** — conceptual in paper, empirical in follow-up
4. **Syntactic confound as probing contribution** — framing only, no new experiments
5. **Non-template evaluation** — still a limitation to acknowledge

### Framing Guidance (from expert pre-review)

The paper lives or dies by framing. The main reviewer objection will be:
"Semantic directions in embeddings are mature; this is another one." The
counter-argument must be clear:

**We are not classifying.** We are recovering a **quantitative, human-
interpretable scale** with **cross-dataset psychometric validity**. This is
a case study in **calibrated semantic structure**, bridging psychometric
semantics and representation learning, with a measurement perspective:
embeddings and human surveys as convergent instruments measuring the same
latent epistemic probability dimension.

The eight distinctive contributions (no prior work has all of these):

1. **Target phenomenon:** Epistemic hedging as *graded probability*, not
   binary classification of hedged/unhedged or modal/non-modal.
2. **Psychometric calibration:** Quantitative alignment to Mosteller & Youtz
   (1990) with ρ > 0.90, not just "we can recover some ordering."
3. **Three-dataset transfer:** Mosteller → Vogel (ρ = 0.99) AND Mosteller →
   Wintle (ρ = 0.97). Convergence across 3 independent datasets spanning
   1967–2019.
4. **Syntactic confound discovery:** PC1 = syntax, probability on PC2 when
   types are mixed; within-type, probability becomes dominant.
5. **Cross-architecture robustness + compression:** 6 models (including bge-m3),
   axis concentrated in early Matryoshka dimensions.
6. **Cross-linguistic universality:** 8 languages, mean zero-shot |ρ| = 0.928.
   The axis is not an English artifact.
7. **Null hypothesis controls:** Permutation, non-epistemic adjective, and
   random label tests all confirm the signal is genuinely epistemic.
8. **Informative negatives:** IQR not in geometry; ensemble doesn't beat
   single axis; negation breaks linearity.

### Paper Structure (updated)

1. Introduction — the hypothesis and why it matters
2. Related Work — Ji et al., Marks & Tegmark, Sileo & Moens, truth directions,
   verbal probability psychometrics, probing
3. Experimental Setup — Mosteller data, syntactic types, 6 models, method,
   baselines, null hypothesis design
4. Experiment 1: Consistency and the syntactic confound
5. Experiment 2: Within-type probability axes (core result, 5 models, LOO, CIs)
6. Experiment 3: Three-dataset cross-validation (Vogel + Wintle)
7. Experiment 4: Cross-linguistic transfer (8 languages)
8. Experiment 5: Extensions — novel phrases, modal axis, compound composition,
   truncation, bimodality probe
9. Experiment 6: Negative results — IQR, ensemble
10. Discussion — what geometry encodes, null hypothesis, cross-linguistic
    implications, philosophical perspective on degree of belief, limitations
11. Conclusion

---

## III. Proposals (Not Yet Run)

### A. Causal Measurement Model
**File:** `EXPERIMENT-PROPOSAL-causal-measurement-model.md`

**Question:** Are Mosteller surveys and embedding geometry both noisy measurements of the same latent "true probability semantics"? Which instrument is more precise?

**Why it matters:** Reframes the paper from "validation" to "convergent measurement." Could reveal that embeddings are MORE precise than psychometric surveys for some expressions.

**Scope:** 1–2 sessions. Hierarchical Bayesian model (~50 parameters, ~200 observations) in Stan or PyMC. Data already exists (Mosteller + Vogel + Wintle + 5 model LOO predictions).
**Resources:** Stan/PyMC installation. Familiarity with hierarchical models.
**Relevance to paper:** Could be a section in the Discussion (conceptual framing) or a supplementary analysis (formal model). Or a companion paper.

---

### B. Corpus Grounding
**File:** `EXPERIMENT-PROPOSAL-corpus-llm-philosophy.md` (Section I)

**Question:** What distributional patterns in training corpora give rise to the geometric probability axis?

**Scope:** 2–3 sessions. Requires large corpus sample.
**Relevance to paper:** Future work / companion paper.

---

### C. LLM Decomposition
**File:** `EXPERIMENT-PROPOSAL-corpus-llm-philosophy.md` (Section II)

**Question:** When LLMs explicitly decompose hedged statements into claim + probability, how do their estimates compare to our geometric projections?

**Scope:** 1 session. ~60 statements, 3–4 LLMs, zero-shot prompting.
**Relevance to paper:** Could go in the paper as a validation experiment.

---

### D. Self-Report Truth Axis
**File:** `EXPERIMENT-PROPOSAL-self-report-truth-axis.md`

**Scope:** 2–3 sessions. Separate paper.

---

### E. Fast/Slow/Authentic Response Geometry
**File:** `EXPERIMENT-PROPOSAL-fast-slow-authentic.md`

**Scope:** 3–4 sessions. Separate paper entirely.

---

### F. The Philosophical Argument
**File:** `EXPERIMENT-PROPOSAL-corpus-llm-philosophy.md` (Section III)

**Claim:** Degree of belief is a measurable geometric property of natural language.

**Relevance:** Could be the heart of the Discussion section (lighter touch) or a companion paper.

---

## IV. Decision Map — What Goes Where

### The Immediate Paper (arXiv / CMCL / TACL)

```
DEFINITELY IN                    COULD ADD                     FUTURE WORK MENTION
─────────────                    ─────────                     ────────────────────
Experiments 01b–07               LLM decomposition             Corpus grounding
Baselines + LOO + CIs           Real-text evaluation           Self-report truth axis
Vogel + Wintle cross-validation Causal intervention (ICLR)     Fast/slow/authentic
Cross-linguistic (8 languages)  Native speaker verification    Full causal model
Null hypothesis tests                                          Domain shift (medical)
Bimodality probe
Matryoshka truncation
Negative results (IQR, ens.)
Compound composition
Philosophical argument (lite)
```

### Companion Paper: "Measuring Meaning"
Causal measurement model + philosophical argument (full) + corpus grounding +
LLM decomposition + expanded cross-linguistic evidence.

### Companion Paper: "Latent Confidence"
Self-report truth axis + fast/slow/authentic.

---

## V. Recommended Path Forward

### Phase 1: CMCL Abstract + arXiv Preprint (target: 2–3 sessions)

| Session | Work |
|---|---|
| **1** | Figures. Update CMCL abstract with cross-linguistic + null hypothesis results. Convert to LREC LaTeX format. |
| **2** | Write full arXiv paper (Methods + Results + Discussion). All data is in hand. |
| **3** | Introduction + Related Work. Polish. Submit CMCL (Feb 25). Post arXiv preprint. |

### Phase 2: Full Paper Submission (2–3 additional sessions)

- Expand arXiv preprint to full TACL or ICLR submission
- Add LLM decomposition if targeting TACL
- Add causal intervention if targeting ICLR 2027
- Real-text evaluation pilot
- Submit to TACL (rolling, any month) or ICLR 2027 (~Sep 2026)

### Phase 3: Companion papers (longer term)

- Causal measurement model paper
- Self-report truth axis + fast/slow/authentic
- Full cross-linguistic study with native speaker collaboration

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
| `experiment_08_bimodality_wintle.py` | Bimodality probe + Wintle cross-validation |
| `experiment_09_cross_linguistic.py` | Cross-linguistic transfer (8 languages, bge-m3) |
| `experiment_10_null_hypothesis.py` | Permutation + non-epistemic + random label controls |
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
| `results/exp08_nomic-v1.5.txt` | Bimodality + Wintle for nomic |
| `results/exp08_mxbai.txt` | Bimodality + Wintle for mxbai |
| `results/exp08_qwen3.txt` | Bimodality + Wintle for qwen3 |
| `results/exp09_bge-m3.txt` | Cross-linguistic (4 languages, initial run) |
| `results/exp09_bge-m3_8lang.txt` | Cross-linguistic (8 languages, full run) |
| `results/exp10_nomic-v1.5.txt` | Null hypothesis tests |

### Proposals (not yet run)
| File | Description | Scope |
|---|---|---|
| `EXPERIMENT-PROPOSAL-causal-measurement-model.md` | Hierarchical Bayesian measurement model | 1–2 sessions |
| `EXPERIMENT-PROPOSAL-corpus-llm-philosophy.md` | Corpus grounding + LLM decomposition + philosophical argument | 1–3 sessions per section |
| `EXPERIMENT-PROPOSAL-cross-linguistic.md` | Original cross-linguistic proposal (now partially completed) | Pilot done; full study separate paper |
| `EXPERIMENT-PROPOSAL-possible-bimodality.md` | Original bimodality proposal (now completed as Exp 08) | ✓ Done |
| `EXPERIMENT-PROPOSAL-self-report-truth-axis.md` | Latent confidence in unhedged assertions | 2–3 sessions |
| `EXPERIMENT-PROPOSAL-fast-slow-authentic.md` | Deliberation geometry | 3–4 sessions |

### Planning and context
| File | Description |
|---|---|
| `PLAN.md` | This file — master plan |
| `EXPERIMENT-PLAN.md` | Original experiment plan with results summary |
| `feedback-analysis.md` | Paper structure, venue targets, pre-submission checklist |
| `literature-review-undermind.md` | Summary of automated literature review |
| `workshops.md` | Workshop venue analysis and strategy |
| `cmcl-abstract-draft.md` | Draft non-archival abstract for CMCL 2026 |
| `CLAUDE.md` | Guidance for Claude Code working in this repo |

### Ground truth and theory
| File | Description |
|---|---|
| `docs/mosteller_youtz_1990_full.csv` | Primary calibration data (53 expressions, n=238) |
| `docs/vogel_2022_systematic_review.csv` | Cross-validation data (21 studies) |
| `docs/medical_context_metaanalysis.csv` | Domain shift data |
| `docs/epistemic-geometry-hedging-as-linear-structure.md` | Original hypothesis document |
| `docs/epistemic-claim-calibration-first-principles.md` | Bayesian calibration framework |
| `docs/verbal-probability-calibration.md` | Calibration guide with tier system |

---

## VII. Key Numbers (for quick reference)

### Headline results
- **Supervised within-type ρ:** 0.90–0.99 (in-sample), 0.70–0.94 (LOO)
- **Vogel cross-validation:** ρ = 0.991 (mxbai predicative), MAE = 5.1%
- **Wintle cross-validation:** ρ = 0.967 (mxbai predicative), MAE = 3.6%
- **Cross-linguistic transfer:** mean zero-shot |ρ| = 0.928 (8 languages)
- **Null hypothesis:** Permutation p ≤ 0.005 all types; non-epistemic LOO ρ = 0.31
- **Modal axis MAE:** 3.2–8.5% across 5 models
- **Random baseline ρ:** ~0.45 (demonstrates supervised axis adds real value)
- **Novel phrase ranking:** ρ = 0.75–0.94 (5 models)
- **Bimodality spread:** 20–32pp between "barely possible" and "eminently possible"
- **Compound composition:** cos 0.82–0.89 (direction), magnitude ratio ~0.75
- **Truncation:** LOO ρ drops only 0.027 at 64d on MRL model
- **IQR-consistency:** ρ ≈ 0 (clean negative — models don't encode interpretive variance)
- **Ensemble vs. modal-only:** Modal wins (no ensemble improves on novel phrases)

### Cross-linguistic highlights
- Korean predicative: ρ = 1.000 (perfect zero-shot ranking)
- Hindi modal: ρ = 1.000 (perfect zero-shot ranking)
- Chinese modal: ρ = 0.991 (near-perfect, typologically distant)
- Arabic modal: MAE = 3.6% (best cross-linguistic calibration)
- Japanese まさか: 53.5% projected (expected ~5%) — pragmatic loading not captured
- Axis alignment: all language pairs cos > 0.75 (predicative)
- Japanese-Korean cluster: cos = 0.90 (typological affinity reflected)
