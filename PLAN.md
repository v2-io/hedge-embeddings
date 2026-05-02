# PLAN.md — Master Plan for Paper and Research Program

**Last updated:** 2026-05-01 (May 2026 paper-drafting session)

---

## Project Status at a Glance (May 2026)

**Empirical work complete (eleven experiments).** Ten experiments through Feb 2026
plus experiment_11 (concept erasure / functional validation) on May 1, 2026.
All results in `results/`; per-experiment write-ups in `FINDINGS-01.md` through
`FINDINGS-06.md`; concept-erasure design in
`brainstorms/concept-erasure-experiment-design.md`.

**TACL paper draft in progress** (`paper.md` → `paper.tex` → `paper.pdf`):
- Abstract, §1 Introduction, §2 Related Work, §3 Method, §4 Results
  (4.1–4.7), §6 Limitations all in prose form.
- §5 Discussion and §7 Conclusion drafted by an agent during this session
  (May 1, 2026), pending integration.
- Citations: 34 entries in `refs.bib`; in-text rewrite from `(Author, Year)`
  to natbib `\citep{}` / `\citet{}` complete for §1 / §2 / §3 / §4 prose
  subsections / §6.  §4.4, §5, §7 still pending.
- LaTeX pipeline: `convert_to_tex.py` → lualatex → bibtex → lualatex →
  lualatex.  Compiles cleanly to a 20-page PDF (target is ~10 pages — see
  Recommended Path Forward §I.).

**Independent novelty audit complete** (Undermind, May 1, 2026): the conjunction
claim (continuously calibrated verbal probability axis × frozen pretrained
pooled sentence embedding × aligned with human psychometric data × validated
against independent psychometric datasets) is novel as audited.  Memo at
`Novelty_memo_for_hedge_embeddings.md`.

**Connection to ASF (Agentic Systems Framework, `~/src/agentic-systems/`).**
The work on this paper is now understood as measurement infrastructure for
the epistemic states of language-constituted agents — specifically Component
03 (Logogenic Agents) of ASF.  The calibrated probability axis lets a
practitioner read off the linguistic encoding of goal-resolvable observation
ambiguity that Component 03's architectural-bias bound takes as input.
Initial incorporation work has begun in the agentic-systems repo;
the §5 of `paper.md` will fold a one-paragraph reference to this in the
next integration pass.

```
COMPLETED EXPERIMENTS          PAPER-READY VALIDATION        FUTURE-WORK PROPOSALS
───────────────────────        ──────────────────────        ─────────────────────
01  Hedge consistency    ✓     Baselines (5 models)    ✓     Self-report truth axis ★
01b Within-type axes     ✓     LOO cross-validation    ✓     Causal measurement model
02  Novel phrases        ✓     Bootstrap 95% CIs       ✓     Fast/slow/authentic
03  Modal axis           ✓     Vogel cross-validation  ✓     Corpus grounding
04  IQR-consistency      ✓     Wintle cross-validation ✓     LLM decomposition
05  Ensemble             ✓     Bimodality probe        ✓     Cross-model-class
06  Compound hedges      ✓     8-language transfer     ✓       convergence ★★
07  Matryoshka truncation✓     Permutation tests       ✓     Native-speaker
08  Bimodality + Wintle  ✓     Concept-erasure         ✓       cross-linguistic
09  Cross-linguistic (8) ✓     functional validation         psychometrics
10  Null hypothesis      ✓                                   Real-text corpus
11  Concept erasure      ✓                                     evaluation

★  = Highest-impact follow-up: converts TACL paper from "we measured a thing"
     into "we built an instrument and used it for honest-activation discipline."
★★ = Cross-substrate convergence (ours + Ji 2025 decoder + Belém prompted).
     Tests substrate-independence of verbal probability semantics — directly
     parallel to ASF's substrate-independence claim for ELI identity.
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

## V. Recommended Path Forward (May 2026 update)

CMCL Abstract (Feb 25, 2026) was submitted; the path forward now centers
on the TACL paper through trim, supplementary, and citation-verification
passes.

### Phase 1 — TACL paper to submission (current focus)

| Step | Work |
|---|---|
| **1.A** | **Integrate §5 / §7 prose** drafted by agent on 2026-05-01 (currently in agent return; pending insertion into `paper.md`). Add the ASF-bridge §5 ¶4 paragraph framing the calibrated axis as measurement infrastructure for language-constituted agents. Rewrite §5/§7 citations using `refs.bib` keys. |
| **1.B** | **Reconcile §4.4 vs §4.7 cosine inconsistency.** §4.4 reports cos(v_pred, v_modal) = 0.88 (mxbai) / 0.79 (qwen3) post-experiment-11; §4.7 says "0.41–0.81 within a single model on mxbai" from FINDINGS-03's pre-experiment-11 fit. Update §4.7 to "0.41–0.88 on mxbai" (or add a footnote distinguishing the two ridge runs). |
| **1.C** | **Apply Muennighoff → Kusupati citation correction.** The Matryoshka work canonical reference is Kusupati et al. NeurIPS 2022 (arXiv:2205.13147), not Muennighoff 2024. `refs.bib` has the entry; in-text attribution in §3.2, §4.5 needs updating. |
| **1.D** | **Add fig13 reference to §4.4 prose.** The figure now exists (`figures/fig13_erasure_mae_cos.{pdf,png}`); §4.4 has been describing it as "(to make)." |
| **1.E** | **Verify the six citation gaps** the citations agent flagged: Vogel 2022 (author/title/DOI placeholder), Schockaert 2022 (title/venue), Bhatia 2016 (multiple papers in year), Park 2024 (LRH formalization details), Wallsten 2008 (incollection details). |
| **1.F** | **Trim pass — cut ~3–5 pages of body content** to hit the TACL 10-page limit (currently ~13–15 pages once §5/§7 land). Move Table 1 (5 models × 4 types) detail and the §4.4 (e) reviewer-bait subsection to supplementary; tighten §2.4 psychometrics setup; merge L7+L8 limitations; compress §4.5 robustness probes. Collaborative — needs coauthor judgment per cut. |
| **1.G** | **Build supplementary appendix** — full per-model × per-dataset × per-metric tables, all 8-language per-phrase data, Wintle verification trail per model, bootstrap CI tables, concept-erasure full pair table. Delegate to agent once trim is settled. |
| **1.H** | **CJK rendering for §6 L2** — three options documented inline in `convert_to_tex.py` preamble: (a) get fontspec + luaotfload fallback working, (b) switch to xelatex, (c) romanize the Japanese examples. Coauthor will handle. |
| **1.I** | **fig8 colormap fix is done; fig13 generated.** fig12 panel (a) `YlGn_r` flagged as borderline accessibility concern; not fixed pending review. |

### Phase 2 — Companion papers and follow-up work (post-TACL)

**Highest-impact follow-up** (`★` in the status diagram): the **self-report-
truth-axis** experiment (`EXPERIMENT-PROPOSAL-self-report-truth-axis.md`).
Apply the calibrated axis to *unhedged* assertions and correlate with model
metacognitive judgment.  If that works, it is a clinical instrument for
detecting overclaim — directly an honest-activation discipline tool for the
scaffolded-logogenic-agent regime in ASF Component 03.  This is the move
that converts the TACL paper from "we measured a thing" into "we built an
instrument and used it for AI safety / agentic-systems theory."

**Cross-model-class convergence** (`★★`): get the same sentence projected
through (i) our pooled-embedding axis, (ii) Ji 2025's decoder residual-
stream verbal-uncertainty feature, (iii) a generative LLM's elicited
probability via Belém / Tang prompting.  If all three converge, "verbal
probability semantics" is substrate-independent across model classes — a
measurement-theoretic claim parallel to ASF's substrate-independence claim
for ELI identity at $n \geq 10$ ELIs across four model families.  Bigger
lift than the self-report-truth axis but with paper-of-record potential.

**Other follow-ups in priority order:**
- *Causal measurement model* (formalize §5's convergent-measurements close
  as a hierarchical Bayesian model) — companion paper.
- *Fast/slow/authentic* (does geometry distinguish performative from
  genuine assertion? connects to Truth Death detection in ASF Component 04
  / ELI) — companion paper.
- *Corpus grounding* (close the causal loop training-text → distributional
  patterns → embedding geometry → calibration) — would supply the
  strongest evidence the convergent-measurement framing is grounded in
  something real.
- *Native-speaker cross-linguistic psychometric replication* in the eight
  target languages of §4.3 — converts the ranking-transfer claim into a
  calibration-transfer claim, addresses §6 L2.
- *Real-text corpus evaluation* on hedged claims from scientific writing /
  weather forecasts / IPCC reports — addresses §6 L1.

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
