# TACL Tactical Notes — Hedge-Embeddings Submission

*Working document. Captures the venue analysis, decision-category research, comparator paper read, and editorial evaluation of the local embeddings work as it stands going into May 2026. Source-of-truth for the submission decision; updated as the work progresses.*

---

## §1. The venue

TACL is the journal arm of the ACL. It was created as a hybrid: journal-quality peer review, but with conference-style turnaround and ACL-conference visibility. It is the most independent-researcher-friendly top-tier NLP venue currently operating: no APCs, no affiliation barrier, double-blind review, and accepted papers earn a presentation slot at the next ACL/EMNLP/NAACL.[^tacl-about]

### Submission mechanics

| | |
|---|---|
| **Cadence** | Rolling, monthly. Per OPERATA's earlier research: 1st of each month, 11:59pm Honolulu time. The official guidelines page describes the journal as monthly rolling but I did not independently re-verify the exact Honolulu-time wording.[^tacl-submission] |
| **Page limit** | 10 pages for regular submissions (references and appendix beyond).[^tacl-submission] |
| **Format** | TACL LaTeX template (`tacl2021v1-template.tex` + `tacl2021v1.sty`). ACL-family LaTeX, so any ACL-style draft converts cleanly.[^tacl-template][^tacl-format-instructions] |
| **Anonymity** | Double-blind. Submissions must be anonymized.[^tacl-submission] |
| **Review turnaround** | ~45 days mean to first decision (median 44; target "month and a half"). Reviewer "contract" is 21 days; mean review completion ~24 days.[^tacl-2021q1] |
| **APC / fees** | None. Fully open access, no author charges. |
| **Affiliation requirement** | None. Independent researchers explicitly welcome. |
| **Conference benefit** | Accepted papers get a presentation slot at the next ACL/EMNLP/NAACL. Publish in journal *and* present at conference. This is TACL's unique value proposition. |
| **Reviewer pool** | ~50 Action Editors + ~230 standing reviewers, mostly tenured or several-years-post-PhD. Senior reviewer pool means probe/interpretability work is read by the people who have built the field — Marks, Tegmark, Bürger, Sileo, etc. are not optional citations; they are likely on the reviewer pool.[^tacl-2021q1] |

### Decision categories

From the action-editor instructions, four possible decisions on first submission:[^tacl-ae-instructions]

- **(a)** Accept as-is, with minor revisions.
- **(b)** Conditional accept; revisions within two months.
- **(c)** Reject with encouragement to resubmit within 3 months.
- **(d)** Reject with 1-year moratorium on TACL submission.

### Decision distribution (2021Q1 baseline)

- ~6.4% (a) accept-as-is
- ~24.2% (b) conditional accept
- ~48.1% (c) reject-with-encouragement
- ~21.3% (d) reject-with-moratorium[^tacl-2021q1]

A 2018-period snapshot ran 2/17/55/26 — (d) is consistently the second-largest bucket.[^tacl-2018q3]

**Resubmission outcomes are not a safety net.** In 2020, of 48 (c)-resubmissions, ~23% accepted, ~33% conditional accept, ~44% rejected with moratorium.[^tacl-2021q1] So a (c) is roughly a 56/44 coin-flip on whether the eventual outcome is publication or moratorium.

### What triggers (d) versus (c)

The official documentation gives **one explicit criterion** and nothing more:

> *"For original submissions, papers unlikely to meet TACL standards after implementing the mandatory revisions within three months, are rejected outright."*[^tacl-ae-instructions]

> *"When making a (c) decision AEs are asked to construct a list of mandatory revisions that can be addressed within 3 months. If it seems that the paper requires more extensive revisions, then it should be rejected."*[^tacl-ae-instructions]

Plus one explicitly-flagged path from (c) to (d) on resubmission:

> *"the mandatory revisions are evaluated according to their results (e.g., an additional experiment may yield outcomes that contradict the main argument of the paper and then the paper can be rejected)."*[^tacl-ae-instructions]

The boundary appears to be deliberately uncodified editorial judgment. Cotterell's editor-perspective Medium piece[^cotterell] and Reiter's TACL-vs-ACL post[^reiter], both written by people with first-hand TACL experience, do not articulate the boundary either.

**Practical interpretation (inferred, not TACL-attested):** the official line is *"fixable in 3 months by the same authors."* The pragmatic version is **"would the fix change what the paper is, or just polish it?"** Polish → (c). Identity-changing → (d). The pattern across the broader NLP review-guidelines literature:

| Likely → (c) revise-and-resubmit | Likely → (d) moratorium |
|---|---|
| Missing prior art (citation gaps) | Mis-positioned vs. the actual literature lineage |
| Methodology that needs to be explained better | Methodology that's unsound and would invalidate claims |
| Limitations underspecified | Central claim broader than evidence supports, and narrowing it would gut the contribution |
| Figures unclear | Statistics that, if redone correctly, would not show the claimed effect |
| One missing baseline | Missing the *load-bearing* baseline the field expects |
| Underdeveloped discussion | Conceptual confusion at the framing level |
| Cross-linguistic transfer caveated | Cross-linguistic transfer claimed-as-validated when it isn't |

The unifying pattern: **(d) tends to be issued when the fix would change what the paper is, not just polish it.**

---

## §2. The arXiv anonymity-window conflict

This is the single most important venue-level constraint and it deserves explicit attention.

> *No non-anonymous preprint versions of submissions may be posted by the author(s) within the period starting a month before submission to TACL and ending when the paper is no longer under consideration by TACL.* Existing preprints are only allowed if posted **more than a month before** submission, and must be declared to the editors.[^tacl-anonymity]

**Implication for the post-Fellowship arXiv strategy:** the embeddings paper cannot be on arXiv during the TACL window — not in the month before submission, not during the ~45-day first-decision window, not during any subsequent revision cycle. Which means:

- The embeddings paper **cannot be the first arXiv preprint** while it is also the TACL submission.
- The Greenblatt-endorsement first arXiv submission has to be a different paper. The policy-degradation report (target: NeurIPS Position Papers, which permits preprints) is the natural choice and is already what the endorsement-email plan in `~/src/ops/arxiv-endorsement.md` assumes.
- For TACL submission timing: counting backward from "1 month before submission," any arXiv post of the hedge-embeddings paper must happen *more than 30 days before* the chosen TACL deadline, OR not at all until after TACL clears.

**No conflict at the strategic level** — embeddings → TACL, policy-degradation → arXiv first — but the rule is unforgiving and worth re-stating in every submission decision.

---

## §3. The five-threshold OPERATA gate

The submit-vs-roll decision rule that lives in OPERATA §3:[^operata]

1. **Prior-art search depth** — Marks-Tegmark, Ji 2025, Yu 2025, Bürger 2024, Sileo-Moens, Lin 2022, Schockaert 2022 explicitly *positioned*, not just cited.
2. **Figures publication-grade** — color-blind-safe, captioned, PDF-export, taste-passed.
3. **8-page expansion** from CMCL extended abstract — methods + results + supplementary appendix.
4. **Native-speaker caveat phrasing** in Limitations.
5. **One external read-through** — Alan or another peer.

Sanity-check version (from OPERATA): *"if this lands on a TACL referee's desk Monday, am I proud of it, or hoping they're tired?"*

**If any threshold misses, roll cleanly to next monthly deadline.** A roll loses ~30 days of review-clock; a (d) outcome locks out for 12 months. The asymmetry overwhelmingly favors rolling.

The mapping to (c)/(d) risk: thresholds 1 and 4 are (d)-class if mishandled (mis-positioned vs. literature, claimed-not-caveated cross-linguistic transfer); threshold 5 is the only way to externally test for the (d1) "what is being measured" framing risk before reviewers do.

---

## §4. Comparator papers (recent TACL + nearest-neighbor EMNLP)

The pool of TACL papers closely matching this work is genuinely thin — TACL publishes few embedding/probing papers per year, and most adjacent 2024–2026 work has gone to EMNLP/ACL/COLM. Three comparators identified:

### (A) Ying, Zhi-Xuan, Wong, Mansinghka & Tenenbaum (2025)
**"Understanding Epistemic Language with a Language-augmented Bayesian Theory of Mind"** — TACL Vol. 13, pp. 613–637 (25pp).[^ying2025]

The most directly comparable TACL paper on epistemic-language semantics. They translate natural-language epistemic claims (modal language, uncertainty, knowledge, likelihood, false-belief) into a formal language-of-thought via grammar-constrained LLM decoding, then evaluate translations against a generative Bayesian-theory-of-mind model in a maze task with human raters.

**TACL-quality features worth emulating:**
- Bounded behavioral contribution claim — "correlates highly with human judgments" against multimodal LLM and ablated baselines, not an assertion about geometry.
- Behavioral experiment with controlled stimuli grounds every modal-language claim.
- Ablations and competitive baselines (GPT-4o, Gemini Pro, ablated LaBToM) actually run, not gestured at.
- Prior art positioned by *theoretical commitment* (Bayesian theory-of-mind tradition vs. pure LLM probes), not just citation count.

### (B) Valentin et al. (2025)
**"Frame Representation Hypothesis: Multi-Token LLM Interpretability and Concept-Guided Text Generation"** — TACL Vol. 13 (~23pp).[^valentin2025]

The best comparator on geometric-structure-in-representations methodology. Extends the Linear Representation Hypothesis from single-token to multi-token concepts via "frames" (compositional structures of feature directions); uses these for interpretation and steering.

**TACL-quality features worth emulating:**
- Explicit theoretical positioning against Mikolov, Park (LRH formalisation), Marks-Tegmark, Templeton/Bau — not just citation, but explicit gap-statements.
- Contribution decomposed into three claims (representation, intervention, generation), each tested separately.
- Ablations against random projections and baseline architectures.
- **Mechanistic intervention (steering) as the strong evidence** — not just correlation but causal manipulation.

This is the bar for "linear-structure-in-representations" papers right now.

### (C) Boundary case — Ji et al. (2025)
**"Calibrating Verbal Uncertainty as a Linear Feature"** — EMNLP 2025 (not TACL).[^ji2025]

The closest contemporaneous work to this paper's "calibrated linear axis" claim. Single linear feature, causal intervention to reduce hallucinations, behavioral validation. The paper this work has to differentiate against most carefully — and reviewers will know the venue.

---

## §5. The local work — what's actually there (verified)

Verified against `~/src/embeddings/` repo state. The earlier recollection was *mostly* accurate but slightly overstated in places.

**Verified empirical claims:**

- **Mosteller within-type axes:** Supervised ρ across 5 models is **0.90–0.99**, not uniformly "ρ > 0.9." A few values dip to 0.90–0.92 (qwen3 predicative 0.901, qwen3 adverbial 0.916). Some 0.89s on noun-phrase MoE.
- **Vogel cross-validation:** mxbai-embed-large achieves **ρ = 0.991, MAE 5.1%** on predicatives. Real and headline-worthy.
- **Wintle 2019:** Draft cites ρ = 0.97. Repo has `experiment_08_bimodality_wintle.py` and `exp08_*.txt` results, but **the specific model + ρ trail needs verification** before submission.
- **Cross-linguistic 8 languages, mean |ρ| = 0.928:** Confirmed in `results/exp09_bge-m3_8lang.txt`. But per-language story is more textured: modal Spanish ρ=0.852, modal Japanese 0.857, modal Korean 0.841. Japanese modal MAE = 18% — meaningfully degraded. The "expected" values are constructed translations of English Mosteller equivalents, **not native-speaker calibrations**. The cmcl draft acknowledges this in a footnote; for TACL it needs a Limitations *paragraph*.
- **Null hypothesis controls:** Real and well-run. Permutation p≤0.005 across all three types; non-epistemic adjective control LOO ρ = 0.31 vs. 0.87 for real hedges; random-label p≤0.006. Genuine due diligence. (One caveat: FINDINGS-Test-2 flagged "Non-epistemic adjectives show some structure" for the *mixed* set; the *pure* non-epistemic set is clean. A reviewer might pick at this nuance — pre-empt it.)
- **Negative result honestly reported:** FINDINGS-04 (IQR-consistency prediction failed across all 4 models, ρ ≈ 0). Treated correctly. A clean negative result *raises* the credibility of positive ones.
- **10 experiments, 12 figures (PDF + PNG):** Confirmed. Inspected figures (fig1 Vogel crossval, fig4 cross-lingual, fig7 null hypothesis, fig8 PCA confound, fig9 method diagram) are clean and self-contained — proper axes, embedded statistics, publication-style method diagram.

**Methodology depth that the abstract doesn't surface:**

- Four-axis decomposition (Predicative, Adverbial, Noun-phrase, Modal) with explicit subspace geometry (FINDINGS-03).
- Compound-hedge linear composition tests (FINDINGS-06), including the surprising negation non-linearity.
- Clean dimensional-truncation result on Matryoshka.

**One concrete fig issue:** `fig8_pca_confound.png` uses red→green colormap — fails standard color-blind accessibility checks. Switch to viridis or RdBu before submission.

---

## §6. Gate-by-gate evaluation

**Gate 1 — Prior-art positioning.** *Partially met.* The cmcl draft cites all the named works and positions against Sileo-Moens (fine-tuned classifier vs. pretrained calibrated axis) and Ji (calibrated to model behavior vs. calibrated to human psychometrics) explicitly. Yu et al. is woven via "directions to cones" → "4D epistemic subspace." Marks-Tegmark is set up as binary→continuous extension. **Gap:** Schockaert 2022 isn't in the visible draft references; Lin 2022 and Bürger 2024 are cited but contribution-beyond-them claims could be sharper. Compared to *Frame Representation Hypothesis* (which devotes substantial space to sub-claim-level differentiation per cited work), the local positioning is **at workshop-paper depth, not journal depth.** For TACL 8-page expansion this needs to grow into a proper Related Work section (1–1.5 pages) with one sentence on what each cited work does and one on what's different. Right now it lives in an Introduction paragraph.

**Gate 2 — Figures publication-grade.** *Mostly met, one fix.* Figures are clean, captioned, statistics-on-figure, PDF-exported. fig9 method diagram is at TACL accepted-paper level. fig1 Vogel cross-validation is exactly the right "headline figure with the killer number." **Issue: fig8 red→green palette.** Switch to viridis or RdBu. Tasteful but not "exemplary" — comparators have figures that integrate more inset annotation and use consistent palettes paper-wide. Worth one more pass.

**Gate 3 — 8-page expansion.** *Not yet done.* The cmcl draft is ~3 dense pages with tables. Expanding to 8 TACL pages with Methods + Results + Supplementary means: (a) Related Work section (currently absent as such), (b) Methods properly written with template tables, calibration math, cross-validation protocol explicit, (c) per-experiment Results subsections, (d) Discussion engaging the LRH/Yu-cones literature, (e) real Limitations section, (f) Supplementary appendix with full tables, all model comparisons, cross-linguistic per-phrase data. **Raw material exists** — FINDINGS-01 through 06 plus exp07–10 results — but assembling into coherent 8-page narrative with tight prose is multi-day editorial work. Doable in May for June 1 submission.

**Gate 4 — Native-speaker caveat in Limitations.** *Acknowledged but needs upgrading.* Current draft has one sentence noting "expected" values are English-Mosteller equivalents and "ranking may be more reliable than calibration." For TACL, this needs a *paragraph* with: (a) explicit list of 8 languages, (b) statement that translations were not native-speaker-validated, (c) explicit framing as "preliminary evidence consistent with the English-trained axis ranking hedges in other languages, not validated cross-linguistic calibration," (d) acknowledgment that for some languages (especially Japanese modal MAE = 18%), calibration breaks down — and this is *informative*, not a bug. Current framing risks a reviewer reading "transferred zero-shot to 8 languages" in the abstract and feeling baited at Limitations.

**Gate 5 — One external read-through.** *Not evaluable from repo.* `feedback-analysis.md` exists, suggesting some prior feedback loop, but external read-through status unclear. **Most-likely-soft gate.** External read-through is the only way to test for (d1) "what is actually being measured" framing risk before reviewers do.

---

## §7. The (d)-moratorium-risk surface

The failure modes where a TACL reviewer could plausibly argue "the fix changes what the paper is."

### (d1) "What is being measured?" — central scope-of-claim risk

Headline claim: "calibrated probability is linear structure in pretrained sentence embeddings." The experimental design measures: difference vectors between hedged-template and bare-template embeddings, regressed via supervised ridge on Mosteller medians, evaluated with leave-one-out within-type. A skeptical reviewer (Bürger-2024 style, "you found polarity") could argue: *what you've found is that pre-pooled BERT-family encoders preserve enough lexical/positional information that ridge regression on a small ordered vocabulary recovers the ordering — closer to lexical-similarity probing than to "calibrated linear structure of probability."* Ji 2025 does causal intervention on the linear feature; the local work does not.

**The "would change the paper" version of this fix:** add a causal intervention. Take a hedged sentence, project out the probability axis, re-decode (or ask an LLM for a probability estimate from the modified embedding), show the estimate moves predictably. Without it, the paper is doing geometric description. With it, it's at the caliber of Ji and Frame-Representation-Hypothesis. **Most plausible source of (d).**

### (d2) Template-only evaluation

All Mosteller, Vogel, and cross-linguistic results use template sentences ("It is {phrase} that the experiment will succeed"). FINDINGS-02 already shows calibration *compresses* on natural sentences. A (d)-arguing reviewer: *the linear structure is an artifact of holding everything constant except the phrase; this measures word-substitution-in-fixed-context, not "structure in sentence embeddings" in any general sense.*

**Mitigation:** Novel-phrase generalization (FINDINGS-02) and compound-hedge composition (FINDINGS-06) are exactly the right *partial* defenses — natural-language phrasings projected onto template-trained axes still show ρ = 0.84–0.94. **Foreground these in the paper, not in supplementary.** If foregrounded, this risk drops from (d) to (c).

### (d3) Ground-truth weakness for novel claims

"Novel-phrase generalization ρ = 0.75–0.94 against intuitive probability ratings" — *intuitive ratings are author-generated*, per FINDINGS-02. A reviewer: *self-generated intuitions are not psychometric ground truth; this is circular.* Cmcl draft acknowledges this in Limitations. The (d) version: a reviewer wants the novel phrases re-rated by 5+ independent annotators (or via Prolific n=30) before believing the generalization claim.

**Fixable in days, not months** — small annotation study (~$50 on Prolific, ~1 day to set up). Not fixable by deadline-eve, and exactly the kind of objection that pushes a paper from accept-with-revisions to (d).

### (d4) "Calibrated" overclaim

"Calibrated" is loaded technical NLP vocabulary — usually implies reliability-diagram-grade behavior across held-out test distribution. What the paper actually shows is *correlation against psychometric medians on within-type sets of 11–19 phrases*. A strict reviewer could say "this isn't calibration in the technical sense, it's correlation; rename the contribution." This alone triggers (c); combined with (d1) it can trigger (d).

**Soft fix:** In title and abstract, qualify "calibrated linear structure" → "calibrated against psychometric data" or "psychometrically aligned." Technical-rigor crowd will accept the more modest framing.

---

## §8. The (c)-risk surface (fixable in 3 months)

Real issues that competent revision pass addresses without changing the paper's identity:

- Related Work section grown to 1–1.5 pages with explicit gap statements per cited work (Frame Representation Hypothesis is the template).
- Methods section with explicit math and protocol — currently `prob = slope * proj + intercept` lives in scripts, needs to be in §3 with rationale.
- Comprehensive table of all model × all dataset × all metric in supplementary — currently scattered across FINDINGS files.
- Color-blind palette pass (especially fig8).
- Native-speaker caveat upgraded to full Limitations paragraph.
- Re-rate the 26 novel phrases with small independent annotator pool (n=10 Prolific, ~$50, ~1 day). **Single addition turns (d3) from a death-risk into a strength.**
- Wintle ρ = 0.97 verification trail in supplementary — show which model, which subset of phrases.
- Add "we did not do causal intervention" sentence to Limitations and explicitly point to it as future work, citing Ji 2025. **Pre-empts (d1) by naming it.**

---

## §9. Strengths the comparators don't have

The genuine "submit, this is real" side:

- **Three-dataset psychometric triangulation** (Mosteller 1990 + Vogel 2022 + Wintle 2019) is unusually thorough for a probing paper. *Frame Representation Hypothesis* doesn't have this kind of human-data grounding; *LaBToM* (Ying) has rich behavioral data but only one experiment. The 50-year span (Mosteller 1967-era through Vogel 2022) speaks to robustness.
- **Five architecturally diverse models** (BERT-family, decoder-derived, MoE, BERT-large, Qwen3) on the same phenomenon, with size-dependent results that make sense (qwen3 recovers Mosteller exactly on possibly/perhaps; smaller models don't). Cross-architecture replication at this scale is rare.
- **Explicit handling of the syntactic confound** (FINDINGS-01, fig8) is methodologically sophisticated — most probing papers wouldn't have caught that PC1-on-mixed-types captures syntactic frame, not semantics. Showing it explicitly *and* showing within-type analysis fixes it is the kind of thing that wins reviewers.
- **Honest negative result** (IQR-consistency, FINDINGS-04). Frame and LaBToM don't have visible negative-result reporting. Credibility multiplier.
- **Linear composition of compound hedges** (FINDINGS-06) including principled negation exception is a genuinely novel sub-finding.
- **Matryoshka truncation result** — "the structure is concentrated in early dimensions" is a nice practical finding absent from comparators.

---

## §10. Verdict and timeline

### Do not submit May 1 (today)

Three reasons:

1. **8-page expansion (Gate 3) not yet drafted.** CMCL extended abstract is dense and good, but turning it into TACL paper with proper Methods, Related Work, Limitations, and Supplementary takes more than an overnight pass — and submitting a hastily-expanded paper is exactly how (d) outcomes happen.
2. **External read-through (Gate 5) is the gate most likely to catch the (d1) framing risk before reviewers do.** Skipping it is the modal failure mode for independent-researcher submissions.
3. **The novel-phrase intuition-only ground truth (d3)** is fixable in days, not months — but only if started now and not before.

### Submit June 1, conditional on six specific changes

1. Re-rate the 26 novel phrases with n≥10 independent annotators (defangs d3).
2. Either add a small causal-intervention experiment OR explicitly scope the contribution as "geometric description of psychometrically-aligned linear structure," not "calibrated probability extraction" (defangs d1 and d4).
3. Upgrade Limitations on cross-linguistic to full paragraph with explicit "preliminary, not native-speaker validated, ranking trustworthy, calibration not" framing (defangs the obvious cross-linguistic objection).
4. Color-blind-safe figure pass (especially fig8).
5. Full Related Work section with gap statements for Marks-Tegmark, Ji, Yu, Bürger, Sileo-Moens, Lin, Schockaert.
6. One external reader, preferably someone who has reviewed for TACL or ACL.

### If any of those six can't land by June 1 — roll to July 1

The (d) tail this paper faces is dominantly about *framing and scope-of-claim*, not whether the empirical work is real. The empirical work is real — five-model replication, three-dataset triangulation, null-hypothesis controls done right, an honest negative result, figures that already look journal-grade. What it needs is editorial care that prevents reviewers from forming a (d1)-shaped objection from the abstract before they reach the methods.

**The work is above the (d) waterline in empirical substance.** **It is near the (d) waterline in current framing.** The May–June revision pass should close that gap.

### Rolling cleanly is not a setback

For an independent researcher targeting next-cycle conference presentation, May 1 vs. June 1 vs. July 1 is functionally identical: submission → ~45 days first decision → conditional-accept revisions → final acceptance window. Next ACL-family conference is ACL 2027 regardless. The deadline pressure is artificial; the gate is the real question.

---

## §10b. Revisions after direct reading of the comparator PDFs (2026-05-01)

The agent's editorial evaluation in §4–§10 was based on web-summary readings. After downloading and reading all three comparators in full, plus re-checking specific embeddings repo artifacts, several pieces of the analysis need to be sharpened or corrected. The verdict (don't submit May 1, target June 1 with a focused revision pass) does not change. What changes is the (d)-risk shape, the differentiation strategy, and the framing of the contribution.

### What's actually true about the comparators

**On Ying et al. (LaBToM):** The agent presented this as the "most directly comparable TACL paper on epistemic-language semantics," but on direct reading it is **methodologically very different from the embeddings work.** LaBToM is a Bayesian-Theory-of-Mind cognitive model of how humans interpret epistemic language; LLMs appear only as grammar-constrained semantic parsers translating natural language into a formal "epistemic language of thought" (ELoT). The paper validates against 469 human-rated sentences in a maze task. There is no probing of internal representations; there is no embedding geometry. The "TACL quality bar" lessons from this paper are about **behavioral grounding and presentation rigor**, not probing methodology. Joseph already has behavioral grounding via Mosteller/Vogel/Wintle psychometric data — three datasets spanning 50+ years — so he is meeting the *form* of evidence-rigor that LaBToM exemplifies, just from psychometrics rather than fresh participant studies.

**On Valentin et al. (Frame Representation Hypothesis):** The agent's read was substantively correct. FRH does extend the Linear Representation Hypothesis from single-token to multi-token concepts via k-frames, does causal intervention via Top-*k* Concept-Guided Decoding, and does explicit gap-statement positioning against Mikolov, Park 2024 (LRH formalization), Templeton 2024, Bau 2020, and Elhage 2022. The steering experiments demonstrate concrete behavioral effects on token generation (including dual-use jailbreak demonstrations on SafeBench, with a content warning in the paper). One important caveat: **FRH operates on the unembedding matrix of decoder LLMs (Llama 3, Gemma 2, Phi 3), where there is a token-level logit available to steer.** This matters for the (d1) risk analysis below.

**On Ji et al. (VUF):** Also substantively as the agent described. Single linear "verbal uncertainty feature" via difference-in-means in residual stream activations, causal intervention via inference-time scaling of activations along that direction, applied to hallucination detection (combine with Semantic Uncertainty) and mitigation (Mechanistic Uncertainty Calibration, ~30% reduction in confident hallucinations). Three QA datasets (TriviaQA, NQ-Open, PopQA) on three decoder LLMs (Llama-3.1-8B, Mistral-7B, Qwen2.5-7B). The verbal-uncertainty score is computed via **LLM-as-a-Judge** (Llama-3.1-70B), not psychometric data — exactly the differentiation Joseph already names in the cmcl draft.

### The model-class distinction the agent missed (this is the big one)

All three comparators — and indeed the entire recent linear-feature literature (Ji, Valentin, Marks-Tegmark, Bürger, Park, Templeton) — work on **decoder LLM internal states**: residual streams or unembedding matrices, where token-level intervention is natural because the model generates tokens.

**Joseph's work is on sentence embedding models** — mxbai-embed-large, nomic-embed-text, embeddinggemma, nomic-v2-moe, qwen3-embedding, bge-m3. These are encoder/embedding-specialized models that **pool to a single vector per sentence and do not generate tokens**. They are categorically different objects.

This has three large implications:

1. **No prior work has demonstrated calibrated linear probability structure in pretrained sentence embeddings.** The cmcl abstract gestures at this ("no prior work has demonstrated that *pretrained* sentence embeddings encode a *continuously calibrated* probability axis aligned with human psychometric data") but the model-class distinction is not foregrounded as a primary novelty claim. **It should be.** Ji studies decoder residual streams. Valentin studies decoder unembedding matrices. Joseph studies pooled sentence embeddings. These are non-overlapping methodological territories, and Joseph is the first in the third.

2. **The "must add causal intervention" version of (d1) is overstated.** Causal intervention as Ji and Valentin perform it requires a token-generation pathway — scale residual stream activations and watch token output change, or use Top-*k* Concept-Guided Decoding to steer next-token sampling. **Sentence embedding models don't have that pathway.** So a TACL reviewer who demands "do what Ji did" is not just asking for more work — they are asking for a different paper on a different model class. The honest defense is exactly that.

3. **But intervention *analogues* exist for sentence embeddings, and at least one should be in the paper.** Three plausible candidates, ordered by effort:
   - **Concept erasure (cheapest, ~1 day):** project the probability axis OUT of a hedged sentence embedding, then run the modified embedding through a downstream task that should depend on epistemic content (retrieval against a probability-graded corpus, classification of confidence level, similarity to a held-out probability anchor). Show that erasing the axis degrades task performance predictably.
   - **Embedding composition (~1–2 days):** take a neutral claim embedding, add scaled probability-axis projections, show that the resulting embedding behaves in retrieval/classification like real hedged sentences with the corresponding probability. This is the sentence-embedding analogue of "steering."
   - **Cross-model behavioral validation (~3–5 days):** use the sentence-embedding probability projection to PREDICT what a separate generative LLM will assign as probability (or pick as the correct hedge) for a query. Real cross-model behavioral grounding without requiring a token-generation pathway in the embedding model itself.

   Adding even the cheapest of these (concept erasure) substantially defangs (d1). It demonstrates that the linear structure has functional content, not just geometric correlation, even in a model class without classical steering hooks.

### What this changes in the (d)-risk surface

| Original (d)-risk framing (agent) | Refined framing after direct read |
|---|---|
| (d1) "Without causal intervention this is geometric description, not calibrated probability extraction" | (d1) refined: the model-class distinction is a legitimate defense — sentence embeddings don't have token-generation hooks. But a sentence-embedding *intervention analogue* (concept erasure or composition) is reasonable to expect, and including one moves the paper from "geometric description" to "functional probabilistic structure." Still the most plausible (d) source if not addressed, but the fix is smaller than re-running Ji's experiment. |
| (d2) Template-only evaluation | Unchanged. Foreground FINDINGS-02 (novel-phrase generalization) and FINDINGS-06 (compound-hedge composition) prominently in Results, not Supplementary. |
| (d3) Author-generated intuitions as ground truth for novel phrases | Unchanged. n≥10 Prolific annotators, ~$50, ~1 day. This is the single highest-leverage fix. |
| (d4) "Calibrated" overclaim | Unchanged but now sharper: Ji literally calls his work "Calibrating Verbal Uncertainty" and his calibration is to model behavior. Joseph's calibration is to *human psychometric medians*, which is arguably a *stronger* sense of calibrated than Ji's. Title/abstract language can stay close to "calibrated" if the difference from Ji is made explicit in the introduction. |

### Three things the agent overstated

1. **TACL page length.** The agent's "TACL 8-page expansion from CMCL extended abstract" framing is correct as a floor, but **all three comparators run 23–25 pages.** TACL clearly accepts substantially longer papers; the 10-page "regular submission" guideline is a soft framing, not a ceiling. This means the Methods + Related Work + Limitations + Supplementary expansion is more comfortable than "squeeze it into 8 pages" implies. A 12–15 page main paper plus appendix is fully normal at TACL.

2. **The cross-linguistic per-language variance "concern."** The agent worried that "a TACL reviewer who actually opens the supplementary will notice the variance." Re-reading the cmcl draft: the per-language table is **already in the body**, with Japanese modal 0.857, Korean 0.841, Spanish 0.852, Arabic predicative 0.833 all openly displayed. The variance isn't hidden in a supplementary; it's right there. The fix is the Limitations *paragraph* (not "fixing the data"), framing the trustworthy claim (ranking) vs. the less-trustworthy claim (calibration in non-English).

3. **The Wintle ρ=0.97 verification "gap."** The agent flagged that the FINDINGS-on-Wintle file might be missing. In fact `results/exp08_mxbai.txt`, `exp08_nomic-v1.5.txt`, and `exp08_qwen3.txt` exist. The data is there; it just needs to be walked back into the paper's supplementary table with explicit "Wintle 2019, model X, ρ = Y, MAE = Z" lines.

### One thing the agent got right that's worth re-emphasizing

**Fig8 panel (b) does use a red-to-green colormap.** Confirmed by direct visual inspection: the Mosteller Median % is encoded as red (low) → yellow → green (high). This is the canonical color-blind palette failure mode (~8% of men with red-green deficiency). The phrase labels on each point partly mitigate but don't eliminate. Switch to viridis or RdBu before submission. Easy fix; non-optional.

### What this means for the contribution claim

The current cmcl-draft framing positions the work primarily as "calibrated probability is linear structure in pretrained sentence embeddings." That's defensible, but it under-sells the model-class distinction. A sharper framing for the TACL expansion:

> *"The recent linear-feature interpretability literature has established that uncertainty-related properties (Ji et al., 2025), truth (Marks & Tegmark, 2023), and gradable concepts (Valentin et al., 2025) are linearly encoded in **decoder LLM internal states** — residual streams and unembedding matrices, where token-level intervention is natural. We demonstrate that an analogous structure exists in a categorically different model class — pretrained sentence embedding models that pool to single vectors and do not have token-generation pathways — and that in this class the structure is **calibrated against human psychometric data spanning 50+ years**, not against model-internal behavior. This is, to our knowledge, the first such demonstration."*

This frame: (a) acknowledges the prior linear-feature literature precisely, (b) names the model-class gap as the contribution, (c) leverages the Mosteller/Vogel/Wintle triangulation as a *unique* feature rather than just due diligence, (d) preempts the (d1) "what is being measured" critique by naming the methodological territory the paper is in.

### Updated six-item conditional submission list

1. **Concept erasure or composition experiment on the probability axis** — sentence-embedding-native analogue of causal intervention. Defangs (d1) without requiring decoder-style steering. (~1 day)
2. **Re-rate the 26 novel phrases with n≥10 independent annotators** (Prolific, ~$50, ~1 day). Defangs (d3).
3. **Reframe contribution around the model-class distinction** in title, abstract, and introduction. The current framing is correct but under-positioned; sharpen it as the primary novelty claim.
4. **Limitations paragraph on cross-linguistic** — explicit "ranking trustworthy, calibration not, native-speaker validation deferred" framing.
5. **Color-blind palette pass** — fig8 panel (b) is the load-bearing fix; check all figures while you're in there.
6. **Full Related Work section** with explicit gap-statements per work cited. Use Frame Representation Hypothesis §2 as the structural template — one sentence on what each prior work does, one on what's different.

External read-through is still the gate most likely to catch issues that none of these address. Get one TACL/ACL-experienced reader in May.

### Local artifact locations referenced above

- `~/src/embeddings/comparators/ying2025-labtom-tacl.pdf`
- `~/src/embeddings/comparators/valentin2025-frame-representation-hypothesis-tacl.pdf`
- `~/src/embeddings/comparators/ji2025-verbal-uncertainty-linear-feature-emnlp.pdf`
- `~/src/embeddings/results/exp08_mxbai.txt`, `exp08_nomic-v1.5.txt`, `exp08_qwen3.txt` (Wintle results, ready to walk into supplementary)
- `~/src/embeddings/figures/fig8_pca_confound.png` (red→green colormap on panel b confirmed by direct inspection)
- `~/src/embeddings/cmcl-abstract-draft.md` (current draft; differentiation language from Ji is already correct in §4 Discussion)

---

## §11. Files most relevant to this assessment

- `cmcl-abstract-draft.md` — current draft scope; the seed for the 8-page expansion.
- `FINDINGS-01.md` through `FINDINGS-06.md` — substantive evidence record.
- `results/exp09_bge-m3_8lang.txt` — cross-linguistic detail; verify before submission.
- `results/exp10_nomic-v1.5.txt` — null hypothesis evidence trail.
- `figures/fig8_pca_confound.png` — color palette fix needed.
- `figures/fig9_method_diagram.png` — publication-grade exemplar; method-diagram template for other figures.
- `feedback-analysis.md` — prior feedback loop; check whether external read-through has happened.

---

## §12. Resources and links

### TACL official

- TACL journal homepage (MIT Press): https://direct.mit.edu/tacl
- TACL submission guidelines: https://direct.mit.edu/tacl/pages/submission-guidelines
- TACL submissions page (transacl.org): https://transacl.org/index.php/tacl/about/submissions
- TACL about page: https://transacl.org/index.php/tacl/about
- **TACL Action Editor instructions** (the (c)/(d) criterion source): https://transacl.org/index.php/tacl/ae-instructions
- **TACL LaTeX template** (`tacl2021v1-template.tex`): https://transacl.org/tacl-submission-templates/tacl2021v1-template.tex
- TACL formatting instructions PDF: https://transacl.org/tacl-submission-templates/tacl.pdf
- TACL formatting guide (HTML, on arXiv): https://arxiv.org/html/2405.11575v1

### TACL admin reports (decision-distribution data)

- 2021Q1 report: https://www.aclweb.org/adminwiki/index.php/2021Q1_Reports:_TACL_Journal
- 2022Q1 report: https://www.aclweb.org/adminwiki/index.php/2022Q1_Reports:_TACL_Journal
- 2018Q3 report: https://www.aclweb.org/adminwiki/index.php?title=2018Q3_Reports:_TACL_Journal_Editor
- 2024Q3 report (PDF): https://aclweb.org/adminwiki/images/4/40/TACL_2024_Q3_Report.pdf

### Editor-perspective writing on TACL

- Ryan Cotterell, "The Case Against Mandatory Revise and Resubmit": https://medium.com/@ryancotterell/the-case-against-mandatory-revise-and-resubmit-8ab135bc2a39
- Ehud Reiter, "ACL vs TACL Reviewing": https://ehudreiter.com/2023/06/06/acl-vs-tacl-reviewing/
- TACL on Wikipedia: https://en.wikipedia.org/wiki/Transactions_of_the_Association_for_Computational_Linguistics

### ACL ecosystem policies

- ACL Policies for Review and Citation: https://www.aclweb.org/adminwiki/index.php/ACL_Policies_for_Review_and_Citation
- ACL Rolling Review author guidelines: http://aclrollingreview.org/authors

### Comparator papers

| Paper | Local PDF | ACL Anthology |
|---|---|---|
| Ying et al. 2025, "Understanding Epistemic Language with a Language-augmented Bayesian Theory of Mind" (TACL, 25pp) | `comparators/ying2025-labtom-tacl.pdf` | https://aclanthology.org/2025.tacl-1.30/ |
| Valentin et al. 2025, "Frame Representation Hypothesis: Multi-Token LLM Interpretability and Concept-Guided Text Generation" (TACL, 23pp) | `comparators/valentin2025-frame-representation-hypothesis-tacl.pdf` | https://aclanthology.org/2025.tacl-1.65/ |
| Ji et al. 2025, "Calibrating Verbal Uncertainty as a Linear Feature" (EMNLP) | `comparators/ji2025-verbal-uncertainty-linear-feature-emnlp.pdf` | https://aclanthology.org/2025.emnlp-main.187/ |

- TACL accepted papers @ ACL 2025: https://2025.aclweb.org/program/tacl_papers/

**Caveat on the agent's read depth:** The evaluator agent originally read these via web fetch, which often returns summaries rather than full-text. The PDFs above are now downloaded to `~/src/embeddings/comparators/` — *they should be read directly during the May–June revision pass to confirm the comparator characterizations in §4 above and to verify that the (d)-risk analysis in §7 (especially the (d1) "what is being measured" framing) actually matches what these papers do, not what their abstracts suggest.* In particular, Valentin et al. is the primary template for "linear-structure-in-representations" framing the local paper has to differentiate against — re-read it with editor's eye on §3 of their paper (theoretical positioning) and on the steering experiments (the causal-intervention bar).

### Related ops docs

- `~/src/ops/OPERATA.md` — current efforts, contains the 5-threshold gate.
- `~/src/ops/VENUES.md` — Paper 1 venue analysis.
- `~/src/ops/PAPERS.md` — full paper portfolio, Paper 1 entry.
- `~/src/ops/arxiv-endorsement.md` — Greenblatt endorsement working doc; relevant because it deconflicts arXiv-first with TACL anonymity window.

---

## Footnotes / citations

[^tacl-about]: Transactions of the ACL — About: https://transacl.org/index.php/tacl/about

[^tacl-submission]: TACL Submission Guidelines (MIT Press): https://direct.mit.edu/tacl/pages/submission-guidelines and TACL Submissions (transacl.org): https://transacl.org/index.php/tacl/about/submissions

[^tacl-template]: TACL LaTeX template: https://transacl.org/tacl-submission-templates/tacl2021v1-template.tex

[^tacl-format-instructions]: "Formatting Instructions for TACL Submissions" — base files `tacl2021v1-template.tex` & `tacl2021v1.sty`, dated Dec. 15, 2021 (HTML rendering): https://arxiv.org/html/2405.11575v1

[^tacl-2021q1]: TACL 2021Q1 admin report (decision distribution, turnaround, reviewer pool stats): https://www.aclweb.org/adminwiki/index.php/2021Q1_Reports:_TACL_Journal

[^tacl-2018q3]: TACL 2018Q3 editor report (alternate decision distribution snapshot): https://www.aclweb.org/adminwiki/index.php?title=2018Q3_Reports:_TACL_Journal_Editor

[^tacl-ae-instructions]: Instructions and Quick-tips for Action Editors: https://transacl.org/index.php/tacl/ae-instructions

[^tacl-anonymity]: TACL anonymity window policy as documented in the submission guidelines and ACL policies for review: https://direct.mit.edu/tacl/pages/submission-guidelines and https://www.aclweb.org/adminwiki/index.php/ACL_Policies_for_Review_and_Citation

[^cotterell]: Ryan Cotterell, "The Case Against Mandatory Revise and Resubmit": https://medium.com/@ryancotterell/the-case-against-mandatory-revise-and-resubmit-8ab135bc2a39

[^reiter]: Ehud Reiter, "ACL vs TACL Reviewing" (2023-06-06): https://ehudreiter.com/2023/06/06/acl-vs-tacl-reviewing/

[^operata]: `~/src/ops/OPERATA.md` §3 — locked paper-publication decisions, including the TACL five-threshold quality gate.

[^ying2025]: Ying, Zhi-Xuan, Wong, Mansinghka & Tenenbaum (2025), "Understanding Epistemic Language with a Language-augmented Bayesian Theory of Mind," TACL Vol. 13, pp. 613–637: https://aclanthology.org/2025.tacl-1.30/

[^valentin2025]: Valentin et al. (2025), "Frame Representation Hypothesis: Multi-Token LLM Interpretability and Concept-Guided Text Generation," TACL Vol. 13: https://aclanthology.org/2025.tacl-1.65.pdf

[^ji2025]: Ji et al. (2025), "Calibrating Verbal Uncertainty as a Linear Feature," EMNLP 2025: https://aclanthology.org/2025.emnlp-main.187/
