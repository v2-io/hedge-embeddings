# TODO.md — TACL paper (submitted)

## SUBMITTED — 2026-05-02

Submitted to TACL as `11139-epistemic-hedging.pdf` with ~5–6 minutes to spare. Final PDF: 12 pages = 10 body + 2 references; conclusion lands at the bottom of page 10. Submission section: Empirical and Data-Driven Methods. Editor comments: anonymization + reviewer-pairing (linear-feature interpretability ∩ verbal-probability psychometrics) + code-release commitment. Keywords (8): verbal probability, epistemic hedging, pretrained sentence embeddings, linear representation hypothesis, concept erasure, psychometric calibration, cross-linguistic transfer, interpretability.

**Final-push trim (16 → 12 pages, body 14 → 10):**

- §1 four "literature lines" paragraphs → one consolidated paragraph pointing to §2
- §1 contributions list (3 paragraphs) → one (1)–(2)–(3) inline sentence
- §2 paragraphs tightened (decoder-LLM and pooled-embedding lit blocks)
- §3.3 lambda sweep prose → one-sentence summary
- §3.3 linear-calibration paragraph → 2 sentences
- §3.4 evaluation strands → one numbered-inline sentence
- §4.1 confound + within-type paragraphs compressed; figure caption tightened
- §4.2 third paragraph (three observations) → one paragraph
- §4.4 erasure block (three long paragraphs) → three tighter paragraphs preserving both nulls, the cross-pair $r$ signature, the LOO robustness, and the LRH bridge
- §4.5 four-probe paragraph → tighter prose; truncation paragraph compressed
- §4.6 three null-control paragraphs → tighter prose (modal cross-model gradient + multi-$n$ power analysis preserved)
- §4.7 IQR + ensemble paragraphs compressed
- §5 Discussion: one paragraph compressed in place
- §6 L1–L4 each tightened by 30–40%
- §7 Conclusion compressed

**What survived (load-bearing):** Tables 1, 2, 3 unchanged; all six §4 subsections present; both nulls in §4.4; LOO concept-erasure robustness; modal cross-model gradient + multi-$n$ power analysis in §4.6; evaluative non-epistemic control; IQR + ensemble negatives; all four limitations.

Per Joseph's submission-eve guidance: trim was "preserve enough to get past the front door so reviewers know what to ask for." Reviewer-requested expansions land back in the manuscript on revision; the cut material is preserved in `supplement.md` and in the archival sections of this TODO below.

---

## Historical: pre-submission session state — 2026-05-02 (superseded by SUBMITTED block)

**Pre-final-push PDF state:** 16 pages including ~2 pages of references. Body ≈ 14 pages, TACL target 10. ~4 pages still to cut. (See SUBMITTED block above for the final state.)

**Trajectory this session:** 27 (peak post-§4.4 rewrite) → 16 pages, with substantial scientific strengthening alongside the trim (LOO concept-erasure robustness, M10 reruns surfacing the `improbably` outlier, data module eliminating protocol drift).

### All §8 findings are now LANDED in `paper.md`

| Integration | Where it lives |
|---|---|
| §3.3 lambda sweep (two-part stability + asymmetry) | §3.3 prose, ~2 sentences |
| §6 L4 + §4.6 evaluative non-epistemic control | §6 L4 (renumbered from §6) + §4.6 evaluative paragraph |
| §4.4 two-null framing (matched-random + label-permutation) | §4.4 ¶2 leads with both nulls; §5 ¶1 + §6 L3 cross-references aligned |
| §4.4 LOO concept-erasure robustness | §4.4 ¶3 closing sentence; cross-pair r rises to 0.98/0.89 under LOO |
| §4.6 modal-null gradient + §8.1 multi-n power analysis | §4.6 permutation paragraph; §6 L4 (renumbered) develops in full |
| 4-model `experiment_10` chain (gemma/moe/mxbai modal data) | §4.6 cross-model gradient prose |
| M10 outlier finding (`improbably`, `very probably`) | §4.5 truncation paragraph; §4.7 ensemble paragraph; TODO §8.7 + supplement.md preserve original M15 framing |
| Data module (`data/mosteller.py`) + experiment refactors | source-of-truth for item sets; experiments 05/06/07/08/09 + paper_validation import from it; eliminates protocol drift through code |

### Codex Best Remaining Non-Trim Fixes — status

| # | Item | Status |
|---|---|---|
| 1 | Rename "full 53-expression corpus" → "Mosteller-derived analysis set" | DONE (`1a69cd1`) |
| 2 | Item-level provenance table in supplement | NOT DONE — no separate supplementary channel under TACL; item provenance now lives in `data/mosteller.py` instead |
| 3 | Centralize item sets in code with manifest | DONE (`46e356e`/`71f72be`/`c6325f7`) — `data/mosteller.py` with `print_manifest()` |
| 4 | Qualify cross-lingual as translation-subset ranking transfer | DONE (`a562f95`) |
| 5 | Mark M15/modal author-estimated analyses as exploratory | DONE — §3.1 names M10 vs M15; §4.5/§4.7 prose uses M10 numbers |
| 6 | LOO concept-erasure robustness check | DONE (`5a47b7d` script + `3423a9c` prose integration) |
| 7 | Fix stale references and figure caption mismatches | MOSTLY DONE (figure-number stragglers fixed in `5a23a79`); `\bottomrule` bug in TODO §9.1 still open but per Joseph's direction deferred until after trim |

### Remaining trim targets (~4 pages to cut from body)

In rough order of leverage:

- **§4.1 Table 1 + surrounding prose** (~1.5 pages currently). Drop the in-sample column from Table 1 (always >0.90, near-zero information for the reader); compress the prose around the table.
- **§4.4** (still 3 longish paragraphs + LOO sentence). The middle paragraph (headline pair under both nulls) has heavy number density — reportable as a small inline grid or compressed.
- **§4.2** (3 paragraphs after recent compression — could go to 2).
- **§3.1 modal-subsets paragraph** — load-bearing for protocol clarity but could be tighter.
- **§4.5 truncation paragraph** — remains the longest single paragraph in §4.5; could compress while preserving the `improbably` outlier finding.

### Trim-phase planning files (in `tmp/`)

| File | Status |
|---|---|
| `tmp/codex-feedback-01-33.md` | Codex's full audit; integrated into TODO §9 (non-trim items) and inline trim cuts |
| `tmp/rewrite_strategy.md` | Gemini's compression strategy; consulted for trim ordering |
| `tmp/draft_intro_related.md` | Gemini's draft §1+§2 prose; structurally similar to what landed but not used verbatim |
| `tmp/trim-strategy-coauthor-view.md` | My coauthor-view trim doc with positions and open questions |

### Repo cleanup state

- All four agent worktrees removed (`ea13e93` and earlier worktree-removal commit).
- No live experiments running.
- `git status` clean except `tmp/` and `.claude/` (gitignored and project-internal).

---

## §0 — Permutations to consider (scientifically illuminating, not yet greenlit)

Three small experimental permutations that would clarify the modal-axis-fails-permutation-at-n=10 finding. None are in scope for the bugfix track, but each is small and would strengthen the paper's empirical story if there's room.

- [x] **(0.A) Cross-model modal permutation.** Greenlit and run 2026-05-02. Status: data collection in progress (sequential `experiment_10` chain on gemma → moe → mxbai → qwen3). nomic-v1.5 baseline already on file (`results/exp10_nomic-v1.5.txt`). Findings to be added to §8 below as runs land.

- [x] **(0.B) Multi-n power analysis on modal.** Greenlit and run 2026-05-02. Result: `experiment_10b_modal_power_analysis.py` (committed `fe320e0`), output in `results/exp10b_modal_power_analysis.txt`. See §8 below for findings and recommended figure design.

- [x] **(0.C) Label-permutation null on §4.4 ΔMAE.** Greenlit and run 2026-05-02. Result: `experiment_11b_label_permutation.py` (committed `5213a58`), mxbai output in `results/exp11b_label_perm_mxbai.txt`; qwen3 rerun in progress. See §8 below for findings.

- [x] **(0.D — added on Codex feedback) Lambda-sensitivity sweep.** Run 2026-05-02. Result: `experiment_12_lambda_sweep.py` (committed `5cd524a`), output in `results/exp12_lambda_sweep.txt`. Replaces §3.3's "informally verified" hand-wave with quantitative numbers. See §8.

- [x] **(0.E — added on Codex feedback) Evaluative non-epistemic control.** Greenlit 2026-05-02. Result: `experiment_10c_evaluative_control.py` (committed `7203ae2` after canonical-pipeline rewrite that fixed three protocol drifts); output in `results/exp10c_evaluative_control.txt`. Mosteller predicative positive control reproduces canonical 0.874 within 0.001; EVALUATIVE LOO ρ = −0.41 well below the 0.87 ceiling. Integrated into §6 L4 and §4.6 (`db03bbb`) and §1 contribution 7 (`f205407`). See §8.5.

---


Tactical task list for completing the hedge-embeddings TACL submission. Sits below `PLAN.md` (which holds the strategic shape of the program) and above the working-document state in `paper.md`. Tick items as they land. Section ordering reflects dependencies — items in an earlier section unblock items in later sections.

**Last updated:** 2026-05-01 after consolidating feedback from Gemini (strategic + code-audit) and Codex (manuscript audit).

---

## §1 — Verified: TACL page limit is 7–10 content pages

Confirmed against the official formatting instructions PDF (downloaded from `https://transacl.org/tacl-submission-templates/tacl2021v1-submission-formatting-instructions.pdf` on 2026-05-01):

> *"Submissions may consist of seven to ten (7–10) A4 format (not letter) pages of content. The page limit includes any appendices. However, references do not count toward the page limit."*
>
> *"Violation: fewer than seven pages of content or more than ten pages of content, including any appendices."*

**The hard rule for an original (non-resubmission) TACL submission:** body + appendices ≤ 10 pages, references excluded. Comparator papers running 23–25 pages were (b)/(c) resubmissions with explicit extra-page grants from Action Editors, which the formatting PDF lists as the only exception.

Current PDF is 24 pages — we need to cut **~14 pages** to get to 10. The §4 trim is **aggressive**, not modest. Codex's reading was correct; the tactical-notes.md "12–15 is fine" inference was wrong (probably extrapolating from resubmissions).

**Verified sub-question:** the TACL formatting PDF makes no mention of "supplementary materials" or any separate-file channel beyond appendix-within-page-limit. Searched the 5-page formatting instructions for `supplement|additional material|optional material`: zero hits. The only mechanism described is "Appendices, if any, directly follow the text and the references" with "appendices count towards the page limit."

**Planning consequence:** assume the strict case — everything fits in 10 content pages or it doesn't go in the submission. Material we cannot fit either disappears from this paper or moves to a companion paper / future-work direction. The "move to supplementary" inventory below should be treated as "either fold into a tighter main-paper expression, or cut entirely; if a future TACL allowance for separate supplementary turns out to exist, we get the cut material back."

---

## §2 — Submission-blocking manuscript bugs (Codex audit)

Critical: these are required for the PDF to be a coherent submission. None require rerunning any experiment (the new Gemini code-audit confirms math/code/data is bulletproof; the bugs are entirely in the manuscript layer).

- [ ] **§4.4 prose is missing from the compiled PDF.** The §4.4 (a)–(f) substructure is still in blockquoted intent-note form, and `convert_to_tex.py:96` strips blockquotes. The result: §4.4 in the PDF has the heading and fig13 but no body text. Rewrite the (a)–(f) intent block as real prose (3–4 paragraphs), integrating the fig13 reference natively into the text, and following Codex's stylistic recommendation to lead with raw ΔMAE and ΔMAE/matched-null *ratios* rather than the giant z-scores that look reviewer-hostile when matched-control std rounds to ~0.

- [ ] **`[PLACEHOLDER]` text still in compiled paper.tex.** §1 contributions list item #6 has `[PLACEHOLDER: Δρ, MAE]` and `z = [PLACEHOLDER]`. These were left in the abstract and intro contributions list when experiment_11 v2 returned and the integrating session forgot to fill them. Find every `[PLACEHOLDER]` in `paper.md` and fill from `results/exp11_*.txt`.

- [ ] **Modal-set definition unified across artifacts.** Currently mixed:
  - §3.1 prose, §6 limitations, abstract → "n = 10 Mosteller-derived adverbial values"
  - Table 1 in §4.1 → cells labeled `n=10` but caption says LOO on 8 items
  - `paper_validation.py:130` and `results/validation_*.txt` → "Mosteller-only Modal" with 8 items
  - `experiment_11_concept_erasure.py:172` → 10 items (the 8 + `very probably` and `improbably`)
  - **Resolution:** unify on n=10. Rerun `paper_validation.py` with the 10-item set; update Table 1 numbers and caption; verify abstract claim still holds.

- [ ] **LOO ρ range claim "0.67–0.95" vs Table 1's embeddinggemma modal LOO = 0.524.** Update after the modal n=10 rerun. If embeddinggemma still floors LOO < 0.67 on modal, either widen the claimed range, exclude that cell with a stated reason in the caption, or — likeliest — the rerun on the 10-item set will lift it to within range.

- [ ] **"Permutation test across all four syntactic types" claim.** `experiment_10_null_hypothesis.py:209` only tests predicative, adverbial, noun phrase. Modal isn't in the loop. Either (a) add modal permutation/random-label runs (small change to the script), or (b) change "four" to "three" everywhere the claim appears (abstract, §1, §4.6, §6). Option (a) is cleaner if we have the cycles; option (b) is honest and faster.

- [ ] **Fix `$s_0 = $ "The experiment will succeed"` LaTeX rendering.** §3.1's broken inline-math close-then-reopen produces literal `$` signs in the PDF. Rewrite as: *"let $s_0$ denote the bare claim 'The experiment will succeed.'"* (single math span, prose for the string).

- [ ] **CJK rendering for §6 L2.** Joseph's domain — three options documented inline in `convert_to_tex.py` preamble: fontspec + luaotfload fallback, switch to xelatex, or romanize. Out of scope for this list; flagging as a parallel track.

- [ ] **Verify fig8 colormap actually fixed in latest PDF.** The figure agent fixed it earlier; quick check that the regenerated PDF actually shows viridis, not the old RdYlGn. (Should be — but Gemini #1 flagged it as a check-the-box item.)

---

## §3 — Compose-the-bugfix-commit checklist

Sequence matters here:

- [ ] Fix all §2 items above in one editorial pass on `paper.md` and `convert_to_tex.py` (and one targeted rerun of `paper_validation.py` if we go with modal n=10 unification).
- [ ] Regenerate `paper.tex` and `paper.pdf` with `python3 convert_to_tex.py`.
- [ ] Confirm: no `[PLACEHOLDER]` strings, §4.4 has body text, Table 1 modal cells consistent, `$s_0$` renders correctly.
- [ ] Get a clean baseline page count *with all bugs fixed*. That number is the input to §4 (trim).

---

## §4 — Trim pass (aggressive cut: 24 → 10, including appendix)

**Verified target:** 10 content pages including any appendix (§1). Current is 24. Cut: ~14 pages.

**Per Codex's section-level targets** (which fit the 10-page rule):

| Section | Current | Target |
|---|---|---|
| Abstract | ~290 words | 180–230 words |
| §1 Introduction | ~1.5 pp | 1 pp |
| §2 Related Work | ~3 pp | 1.25–1.5 pp |
| §3 Method | ~2 pp | 1.5 pp |
| §4 Results (all subsections) | ~7 pp | 4 pp |
| §5 Discussion | ~1.5 pp | 0.6 pp |
| §6 Limitations | ~2.5 pp | 0.9 pp |
| §7 Conclusion | ~0.6 pp | 0.4 pp |
| Appendix (if used) | — | 0–1 pp |

That sums to ~9–10 main pages, leaving 0–1 page of headroom for appendix. References don't count toward this.

**Cuts I'd propose, in roughly increasing reluctance to make them:**

- [ ] **§5 ¶4 ASF-bridge: 8 sentences → 3.** Drop the speculative applications list (honest-activation discipline / cross-substrate / drift detection) per Codex's "feels broader and less finished" critique. Keep the framing pivot + Schockaert citation. *~0.4 pages.*

- [ ] **§4.5 Robustness: 1.5 → 0.6 pages.** Truncation and bimodality compress to one paragraph each; novel-phrase + compound combine; per-model truncation curves and per-modifier bimodality numbers go to supplementary. *~1.0 page.*

- [ ] **§4.4 Concept erasure: 1.5 → 0.9 pages.** Fold (c) and (c′) together; (e) reviewer-bait answers compress to one paragraph; per-pair details go to supplementary; fig13 stays. *~0.6 pages.*

- [ ] **§2 Related Work: 3 → 1.8 pages.** §2.4 (psychometrics ground-truth setup) repeats §1 context; older-precursor list (Lee, Stanovsky, Bhatia, Shen) compresses to one sentence; Cohen / Cho / Adarsh in §2.1 drop entirely. *~1.2 pages.*

- [ ] **§6 Limitations: 2.5 → 1.5 pages.** Merge L7 + L8 (both small-n psychometric calibration); L4 mixed-non-epistemic compresses to 4–5 sentences; the Ji 2025 Appendix E.2 closing coda → footnote. *~1.0 page.*

- [ ] **§3 Method: 2 → 1.5 pages.** §3.4 evaluation walk-through defers detail to the §4 subsection where each strand actually runs. *~0.5 pages.*

- [ ] **Tables 1 and 2: dense → headline + supplementary.** Move the full 5-model × 4-type grid (Table 1) and the 5-model × 2-dataset grid (Table 2) to supplementary; in-body keep only the headline number per type / per dataset. *~0.5 pages.*

- [ ] **Codex's structural editorial point.** *"The scientific center should be the psychometrically grounded, within-type verbal-probability axis; concept erasure and multilingual transfer should support that center rather than become competing headline claims."* Apply this lens during the cuts above — don't lift §4.3 / §4.4 to co-equal headline status with §4.1 + §4.2.

**Move to supplementary:**
- Per-pair concept-erasure z-score table (§4.4 detail)
- Truncation curves per dimension per model (§4.5 detail)
- Per-modifier bimodality numbers (§4.5 detail)
- 8-language per-phrase ranking data (§4.3 detail)
- Wintle full per-phrase verification trail (§4.2 detail)
- Bootstrap CI tables (§4.1 detail)
- Inter-axis cosine matrices (§4.7 / §5 detail)

---

## §5 — Citation gaps to verify before submission (Codex audit)

Six entries in `refs.bib` need verification — citations agent built them from secondary sources and flagged them. Resolve before submission:

- [ ] **Vogel et al. 2022** — author list / title / DOI placeholder. Verify against `docs/vogel_2022_systematic_review.csv` provenance.
- [ ] **Schockaert 2022** — title / venue best-guess; the agent inferred "A Formal Framework for Embeddings as Epistemic States / JAIR." Verify the exact published title.
- [ ] **Bhatia 2016** — multiple Bhatia papers in that year; current key set as `bhatia2017vector` for *Cognitive Science* venue. Confirm.
- [ ] **Park et al. 2024** (LRH formalization) — title / booktitle best-guess.
- [ ] **Wallsten 2008** — incollection details approximate; verify against Belém et al. 2024's reference list.
- [ ] **Kusupati 2022 (Matryoshka)** — already corrected from Muennighoff 2024 attribution. Double-check the NeurIPS 2022 entry vs. the arXiv version (2205.13147).

---

## §6 — Quality improvements (not blocking submission, but reviewer-friendly)

- [ ] **Annotator study for §4.5 novel phrases.** Both reviews flag this as the highest-ROI pre-submission addition: Prolific n ≥ 10 annotators on the 26 novel phrases (~$50, ~1 day). Converts §4.5 from "author-rated intuition probabilities" to "independent psychometric ground truth." Defangs the (d3) "reject with moratorium" risk. The new Gemini code-audit verdict says we don't need to redo any experiments to publish; but this one would substantially strengthen the paper.

- [ ] **Lead §4.4 concept-erasure with raw ΔMAE + matched-null ratios, not z-scores.** Codex's stylistic point: when matched-control std rounds to ~0 (qwen3 Predicative→Modal at z = +26.88), the z-score looks like a number that overclaims rather than measures. ΔMAE/matched-null mean as a ratio is the cleaner reporting form.

- [ ] **LOO concept erasure (v3 design upgrade).** Currently the §4.4 baseline uses in-sample-fit calibration; the rest of the paper uses LOO. A reviewer will ask why the inconsistency. Either rerun erasure with LOO calibration (~30 min experiment_11 modification + rerun on both models) or explicitly demote §4.4 to "mechanistic diagnostic" rather than generalization claim. The new Gemini audit says the in-sample choice is mathematically honest as written and the paper says so explicitly; but reviewer-eye-test is reviewer-eye-test.

- [ ] **fig12 panel (a) YlGn_r colormap.** Borderline accessibility concern (deuteranopes lose green-end discrimination). Not fixed pending coauthor review. Easy swap to viridis_r or cividis if we want maximum safety.

---

## §8 — Findings from new experiments (need integration into paper)

**Captured here so the agent summaries don't get orphaned before they make it into prose.** The committed `results/exp*.txt` files have the raw numbers; this section captures the interpretive prose, recommended figure designs, surprises, and integration notes from each agent's report-back. Update as new experiments land.

---

### §8.1 — 0.B Multi-n modal power analysis  (`results/exp10b_modal_power_analysis.txt`, commit `fe320e0`)

**Integration status:** NOT YET INTEGRATED into `paper.md`. Pairs with §8.4 (cross-model modal permutation) for the §6 L7 / §4.6 modal-null prose update. Will land in the same integration pass after the `experiment_10` chain completes for mxbai and qwen3.

**Headline:** quantitatively confirms §6 L7 small-n rank-saturation. The p-value for the modal permutation test collapses three orders of magnitude across n=10 → n=14, with the threshold for crossing p < 0.05 between n=10 and n=12.

**Numbers (nomic-embed-text:v1.5, 1000 perms per row, seed 42):**

| n  | Mosteller items | Estimated items | real LOO ρ | perm μ ± σ | perm 95th | %ile  | p-val  |
|---:|----------------:|----------------:|-----------:|-----------:|----------:|------:|-------:|
| 10 | 10              | 0               | +0.6485    | −0.264 ± 0.375 | 0.806 | 79.2  | 0.2080 |
| 12 | 10              | 2               | +0.8561    | −0.208 ± 0.364 | 0.789 | 97.7  | 0.0230 |
| 14 | 10              | 4               | +0.9636    | −0.166 ± 0.343 | 0.736 | 100.0 | <0.001 |
| 17 | 10              | 7               | +0.9469    | −0.151 ± 0.310 | 0.671 | 100.0 | <0.001 |

**Mechanisms (both move together):** real LOO ρ rises (0.65 → 0.96) AND permuted distribution narrows (95th 0.81 → 0.67). At n=14 the real ρ exceeds every one of 1000 permutations.

**Surprise worth noting:** real ρ slightly *decreases* at n=17 (0.96 → 0.95), consistent with author-estimated values introducing label noise (e.g., several "perhaps/maybe/conceivably" items pinned at 38.5). Doesn't undermine the power-analysis reading because the permuted distribution narrows in parallel — but it's evidence that the n=17 row is bounded above by the noise floor of the author-estimated values rather than by intrinsic embedding signal.

**Methodological flag for prose:** the n>10 rows use author-estimated modal values appended to the Mosteller-grounded set. These values are NOT psychometrically calibrated and are explicitly excluded from the load-bearing modal axis training elsewhere. Their use here is a *power analysis* of sample-size variation — testing the structural prediction that small n saturates rank-based metrics — and is NOT a calibration claim on the expanded set. Both the script header and the result file flag this.

**Recommended supplementary figure (two panels, shared x-axis):**

- **Panel (a) — p-value vs. n, log y-axis.** Plot p-value points; horizontal dashed lines at p=0.05 and p=0.01. Use an open-circle marker for n=10 and filled circles for n>10 to flag n=10 as the only Mosteller-calibrated row. Floor the log axis at a value that lets n=14, 17 zeros draw as down-arrows ("p < 1/1000") rather than as actual zeros.
- **Panel (b) — permuted LOO ρ distribution (violin), with real ρ overlaid as star marker per n.** Visualizes the two simultaneous mechanisms: violins narrow with n; real-label star moves up the violin until it sits above every shuffled point.

**Recommended caption (template):** "Modal permutation test power as a function of n. Bars/violins for n>10 include author-estimated modal values appended to the 10-item Mosteller-grounded set; these rows test the §6 L7 structural prediction and do not constitute a calibration claim. The p-value collapses from 0.21 (n=10) to <0.001 (n=14, 17), confirming that §4.6's modal non-significance is a small-sample power artifact rather than evidence of absent signal — independently corroborated by the §4.4 concept-erasure functional validation."

**Where this lands in paper:** §6 L7 prose changes from a verbal claim ("rank-based tests have low power at n=10") to a quantitative one ("threshold for p<0.05 is between n=10 and n=12; small-sample regime is empirically demonstrated"). §4.6 modal-non-significance prose should reference this curve. Suggested as a supplementary figure if room; otherwise a single sentence in §4.6 with the four (n, p) values inline.

**Cross-model robustness:** not run for 0.B; default agent decided the curve was clean enough on nomic-v1.5 alone. If reviewer pressure justifies, a single follow-up run on qwen3-embedding (already foregrounded in §4.4) would be the natural choice — output to `results/exp10b_modal_power_analysis_qwen3.txt`.

---

### §8.2 — 0.C Label-permutation null on §4.4  (`results/exp11b_label_perm_{mxbai,qwen3}.txt`, commits `5213a58` + `633cbbb`)

**Integration status:** IN PROGRESS — §4.4 two-null framing rewrite agent running in background as of 2026-05-02. The current §4.4 in `paper.md` (from commit `687f1ba`, refined by `58af880`) uses only the matched-random null with the ratio framing. The in-flight rewrite incorporates both the label-permutation results below AND the qwen3 11/12 cross-null cross-architecture finding.

**Headline:** the cosine-matched-direction null and the label-permutation null are answering different questions. The label-permutation null has wide std (1.7–4.3 on the headline pair) where the matched-random null's std rounds to ~0 at high cosine, and the predicative ↔ modal headline pair clears it with massive headroom (p = 0.002 both directions). This **directly addresses Codex's z-score-inflation worry far more cleanly than the §4.4 prose-level ratio reframing alone could**.

**Per-pair table (mxbai-embed-large, K=500 permutations):**

| A→B | cos | real ΔMAE | label-perm null μ±σ | percentile | p-val | matched-random verdict | label-perm verdict |
|---|---:|---:|---:|---:|---:|---|---|
| **Pred→Modal** | **0.88** | **+18.37** | **+1.22 ± 1.71** | **100.0** | **0.002** | functional | **functional** |
| **Modal→Pred** | **0.88** | **+24.62** | **+3.27 ± 4.28** | **100.0** | **0.002** | functional | **functional** |
| Pred→Adv  | 0.42 | +9.10  | +0.73 ± 1.35 | 99.6  | 0.006 | functional | functional |
| Pred→NP   | 0.47 | +5.44  | +0.53 ± 0.73 | 100.0 | 0.002 | functional | functional |
| Adv→NP    | 0.41 | +3.40  | +0.42 ± 0.59 | 99.8  | 0.004 | functional | functional |
| Adv→Pred  | 0.42 | +3.36  | +0.60 ± 0.79 | 99.0  | 0.012 | trend      | trend |
| Modal→Adv | 0.44 | +10.32 | +1.69 ± 2.34 | 98.6  | 0.016 | functional | trend |
| Modal→NP  | 0.34 | +3.50  | +0.86 ± 1.09 | 96.0  | 0.042 | functional | trend |
| NP→Adv    | 0.41 | +4.88  | +0.72 ± 1.52 | 97.0  | 0.032 | functional | trend |
| Adv→Modal | 0.44 | +2.69  | +0.49 ± 0.70 | 98.8  | 0.014 | functional | trend |
| NP→Pred   | 0.47 | +3.89  | +0.88 ± 1.61 | 93.4  | 0.068 | —          | — |
| NP→Modal  | 0.34 | +1.36  | +0.58 ± 1.10 | 87.4  | 0.128 | trend      | — |

**Cross-null agreement:** 5/12 functional under both nulls (the strongest cases), 4/12 only under matched-random, **0/12 only under label-perm**, 3/12 under neither. Label-perm is the more stringent null and **never disagrees directionally** with matched-random — when they disagree, label-perm is just the more conservative test.

**Methodological note:** the label-permutation null and the matched-random null answer different questions. Matched-random asks: "is this *direction* more damaging than a random direction with the same overlap with v_B?" Label-permutation asks: "is the *probability content* of v_A doing the work, or just the geometric/structural properties that survive label shuffling?" Both are valid and complementary; pairs functional under both are the strongest functional-validation cases.

**Where this lands in paper:** the §4.4 rewrite landed before 0.C ran, so the prose currently uses only the matched-random null with the ratio framing. Integration options:

1. **Minimal:** add 1–2 sentences to §4.4 ¶2 noting that the headline pair clears a label-permutation null at p = 0.002 with wide null variance — addresses Codex's z-score-inflation worry directly.
2. **Stronger:** rewrite §4.4 ¶2 to lead with both nulls. Frame as "two complementary nulls: matched-random for geometry, label-permutation for content; both clear on the headline pair." Costs ~3 lines net beyond minimal option.
3. **Strongest:** drop the matched-random null to a parenthetical and lead with label-permutation. Probably overcorrection — matched-random is the standard control for this kind of test, and demoting it would invite a different reviewer objection.

Recommendation: option 2. The two-nulls framing is genuinely the strongest version of the §4.4 result.

**qwen3 result (DONE, `results/exp11b_label_perm_qwen3.txt`, ready to commit):** Output flushed when script completed. **11/12 ordered pairs are functional under BOTH nulls on qwen3** (vs. 5/12 on mxbai) — substantially stronger cross-null agreement than the mxbai run. Headline pair clears p=0.002 both directions: Predicative→Modal real ΔMAE +17.36 vs. null μ±σ = +1.28±1.87; Modal→Predicative real ΔMAE +20.49 vs. null +2.67±3.88. Only Modal→Adverbial is functional only under matched-random; **0/12 functional only under label-perm**, **0/12 functional under neither**.

**Cross-architecture summary:** matched-random and label-permutation nulls agree on the headline predicative ↔ modal pair on both models (p=0.002 in all four model × direction combinations under label-perm). The two complementary nulls + two architectures + clean cross-null per-pair agreement is now a much tighter §4.4 result than the §4.4 prose rewrite captured (which used only matched-random with the ratio framing).

**Updated integration recommendation:** option 2 from earlier ("rewrite §4.4 ¶2 to lead with both nulls") is reinforced. The qwen3 numbers make the both-nulls framing the obvious choice: it's not just mxbai's headline pair clearing two different nulls; it's qwen3's 11/12 pairs all clearing both nulls. That's a paper-level structural finding, not a per-pair sanity check.

---

### §8.3 — 0.D Lambda-sensitivity sweep  (`results/exp12_lambda_sweep.txt`, commit `5cd524a`)

**Integration status:** LANDED in `paper.md` §3.3 at commits `ca04668` (initial integration) + `c0ddb01` (cell-count fix from "sixteen" → "eight"). Current §3.3 prose carries the two-part stability + asymmetry claim with the modal-axis sensitivity observation cross-referenced to §6 L7.

**Headline:** ridge λ choice is robust across four decades. Worst-case LOO ρ drop relative to the paper's λ=0.1 reference is ≤ 0.072 across all (model × type × λ) cells of the swept grid (nomic-v1.5 + mxbai-embed-large × four within-type axes × five λ values).

**Numbers (LOO ρ per λ):**

**nomic-embed-text:v1.5 (768d):**

| Type | λ=0.001 | λ=0.01 | λ=0.1 (ref) | λ=1 | λ=10 |
|---|---:|---:|---:|---:|---:|
| Predicative      (n=13) | 0.946 | 0.901 | **0.874** | 0.835 | 0.802 |
| Frequency adverb (n=19) | 0.956 | 0.944 | **0.923** | 0.867 | 0.856 |
| Noun phrase      (n=11) | 0.900 | 0.918 | **0.936** | 0.936 | 0.936 |
| Modal            (n=10) | 0.867 | 0.770 | **0.648** | 0.673 | 0.636 |

**mxbai-embed-large (1024d):**

| Type | λ=0.001 | λ=0.01 | λ=0.1 (ref) | λ=1 | λ=10 |
|---|---:|---:|---:|---:|---:|
| Predicative      (n=13) | 0.951 | 0.940 | **0.912** | 0.930 | 0.902 |
| Frequency adverb (n=19) | 0.851 | 0.926 | **0.905** | 0.879 | 0.840 |
| Noun phrase      (n=11) | 0.827 | 0.855 | **0.855** | 0.855 | 0.836 |
| Modal            (n=10) | 0.903 | 0.903 | **0.806** | 0.833 | 0.869 |

**Surprises worth flagging in §3.3:**

- **λ=0.1 is intentionally on the conservative side, not at the optimum.** On 6 of 8 (model × type) cells, λ=0.001 or λ=0.01 yields a *higher* LOO ρ than λ=0.1. A reviewer probing "why λ=0.1?" needs an honest answer: chosen for fairness/comparability across cells, not optimized.

- **mxbai modal is sensitive in the opposite direction.** LOO ρ is 0.903 at λ=0.001 vs. 0.806 at λ=0.1 (a +0.097 gain by *decreasing* λ 100×), recovering to 0.869 at λ=10. The reference λ=0.1 sits at the *minimum* across the sweep for this cell. Same pattern, weaker, on nomic modal. Modal at n=10 is the cell where regularizer choice does the most rank shuffling — consistent with §6 L7's small-n saturation framing.

- **Noun phrase on nomic shows the opposite pattern** — LOO ρ rises monotonically with λ from 0.001 to 1 (then plateaus). For that one cell, the paper's λ=0.1 is at or above the LOO optimum.

**Where this lands in paper — and what NOT to say:** §3.3 currently reads "informally verified across λ ∈ {0.001, 0.01, 0.1, 1, 10}" near line 125. The replacement must NOT summarize the result as just "stable" — that is true but understates what the data says, and Codex flagged this directly. The honest two-part claim is:

  1. **Stability:** LOO Spearman ρ stays within 0.072 of the λ=0.1 reference across the full four-decade sweep on both models × four within-type axes (worst case: nomic predicative, 0.874 at λ=0.1 → 0.802 at λ=10).

  2. **Asymmetry:** On 6 of 8 (model × type) cells, *smaller* λ would yield a higher LOO ρ than λ=0.1. The strongest such cell is mxbai modal (LOO ρ 0.806 at λ=0.1 → 0.903 at λ=0.001 → 0.869 at λ=10; reference sits at the minimum). Modal at n=10 is also the cell where regularizer choice does the most rank shuffling, consistent with the §6 L7 small-n saturation framing.

The honest framing for §3.3 prose is therefore: *the reference λ=0.1 was chosen for cross-cell comparability rather than per-cell optimum, and the sweep confirms the choice is robust within ±0.07 LOO ρ across the order of magnitude tested while flagging that several cells (notably mxbai modal, n=10) would improve with smaller λ in a per-cell tuning regime.* Cross-reference to §6 L7 on the modal sensitivity.

If the trim budget allows, this could be a single 2–3 sentence paragraph addition to §3.3. If not, fold the asymmetry into a footnote or §6 L7 forward-reference — but do NOT drop it; a reviewer who sweeps λ themselves will find the asymmetry and ask why we didn't disclose it.

---

### §8.4 — 0.A Cross-model modal permutation  (4 models complete; qwen3 deliberately not run)

**Integration status:** READY for integration. 4-of-5 models complete (`results/exp10_{nomic-v1.5,gemma,moe,mxbai}.txt`). qwen3 portion of the chain was killed after 17 minutes with only 18 lines of output (still in the first permutation test of the first axis); see "qwen3 tactical decision" below.

**Modal-permutation gradient across the four completed models:**

| Model | Modal LOO ρ | Real-rank percentile | Modal perm p | Random-label p | Verdict |
|---|---:|---:|---:|---:|---|
| nomic-embed-text:v1.5 | 0.6485 | 79.2  | 0.2080 | 0.207 | fails |
| embeddinggemma:300m   | 0.5879 | 78.2  | 0.2180 | 0.198 | fails |
| mxbai-embed-large     | 0.8061 | 94.6  | **0.0540** | 0.066 | **near-threshold** |
| nomic-v2-moe          | 0.9152 | 99.6  | 0.0040 | 0.005 | **passes** |

**Pattern:** modal permutation outcome tracks modal LOO ρ across the four models in a smooth gradient — 0.59 / 0.65 fail, 0.81 near-threshold, 0.92 clear pass. This is exactly the §6 L7 small-n rank-saturation prediction made visible across architectures, with mxbai sitting as a natural transition point right at the boundary. Same direction on both the permutation test (Test 1) and the random-label control (Test 3) — see the per-model files.

**Larger-n axes (predicative, frequency adverb, noun phrase) — also captured for §4.6 integration:**

All three of these axes pass cleanly across all four completed models with p ≤ 0.029 on permutation and corresponding random-label tests. This is Codex's "cleanest reviewer-facing evidence" — the within-type calibration is not a one-model finding. Specific p-values per Codex's read of the result files:

- gemma: predicative p=0.005, adverbial p<0.001, noun phrase p=0.028
- moe:   predicative p=0.009, adverbial p<0.001, noun phrase p=0.025
- mxbai: predicative p=0.001, adverbial p<0.001, noun phrase p=0.029

(Plus nomic-v1.5 from prior data, also passing all three at p ≤ 0.005.)

**qwen3 tactical decision (2026-05-02):**

Killed the qwen3 chain after weighing: data already informative without it (4-model gradient is clean), runtime cost very high (primal ridge at d=4096 with 53,000+ ridge solves per axis × 4 axes × 3 tests; first-axis permutation test alone was on track to take hours; total chain estimated 6–10 hours), and the qualitative finding ("modal permutation outcome tracks modal LOO ρ; small-n + model-sensitive") doesn't change with one more data point landing somewhere between mxbai's 0.81 LOO and moe's 0.92 LOO. The §4.6 prose framing — "small-n and model-sensitive, consistent with the §6 L7 rank-saturation regime" — is supported by the 4-model gradient as it stands. The honest tradeoff is closure-over-completeness: we chose to land integration tonight rather than wait hours for one more data point.

**Recovery path if qwen3 is ever needed (e.g., reviewer revision):** rewrite `train_axis` in dual form. Current implementation solves a 4096×4096 primal system; the dual form solves an n×n system (n ≤ 19), which would make the whole chain finish in minutes rather than hours at d=4096. Cheap follow-up; not blocking.

**Careful prose framing for §4.6 integration (incorporating Codex's hedging directive):** Do not say "the modal failure is a power artifact." Say: *"the modal permutation result is small-n and model-sensitive; nomic-v1.5 and gemma fail at modal LOO 0.65 and 0.59 respectively, mxbai sits at the 0.054-threshold boundary at LOO 0.81, and moe passes cleanly at 0.004 with LOO 0.92 — the failure pattern is consistent with the small-n rank-test regime developed in §6 L7 and the §4.6→§8.1 power-analysis curve."* The "consistent with" hedging is doing real work: we have evidence compatible with the L7 framing across four architectures, not evidence that proves it.

**Integration target in paper:** §4.6 modal paragraph rewrite + small forward-reference into §6 L7. Pair with §8.1 (the multi-n power analysis on nomic-v1.5) as the two evidence streams jointly supporting the small-n + model-sensitive interpretation. Probably 2–3 sentences in §4.6 plus possibly one row added to a table; if §8.1's two-panel supplementary figure makes the cut in trim, the §4.6 prose can reference it.

**Status as of 2026-05-02:** sequential `experiment_10` chain on gemma → moe → mxbai → qwen3 (PID 54675, `python -u` for line-buffered progress). gemma and moe complete (results in `results/exp10_gemma.txt` and `results/exp10_moe.txt`); mxbai and qwen3 still running.

**Modal-axis findings so far** (cross-referenced with Codex's read of the result files):

| Model | Modal LOO ρ | Modal perm percentile | Modal perm p | Result |
|---|---:|---:|---:|---|
| nomic-embed-text:v1.5 | 0.6485 | 79.2  | 0.2080 | fails |
| embeddinggemma:300m   | 0.5879 | ~78.2 | 0.2180 | fails |
| nomic-v2-moe          | 0.9152 | 99.6  | **0.0040** | **passes** |
| mxbai-embed-large     | 0.806  | —     | (running) | pending |
| qwen3-embedding       | 0.855  | —     | (running) | pending |

**Hypothesis tracking so far:** prediction was *modal passes on the three high-modal-LOO models (moe 0.92, mxbai 0.81, qwen3 0.86) and fails on the two low-modal-LOO models (nomic 0.65, gemma 0.59)*. Two confirms (gemma fails as predicted, moe passes as predicted), three pending (nomic was already known to fail, mxbai and qwen3 still running). The two failures are exactly the two lowest-modal-LOO models; the one pass so far is from a high-modal-LOO model. Pattern is on track.

**Larger-n axes (predicative, adverbial, noun phrase) on the two completed models reinforce the main calibrated-axis claim:**

- gemma: predicative p=0.005, adverbial p<0.001, noun phrase p=0.028
- moe:   predicative p=0.009, adverbial p<0.001, noun phrase p=0.025

This is Codex's "cleanest reviewer-facing evidence" for the within-type calibration claim — it's not just nomic-v1.5; the larger-n axes pass null tests on gemma and moe too, with predicative/adverbial well below p=0.01.

**Caveat (Codex caught this):** `results/exp10_moe.txt` line 88 stops at the random-label test header — modal permutation is usable, but the random-label section for moe is incomplete. Verify when chain completes. (May be a writing-buffer issue with the chain still in flight.)

**Careful prose framing (Codex's recommended hedging — incorporate into §4.6):** Do not say "the modal failure is a power artifact." That's overcommitted. Say: *"the modal permutation result is small-n and model-sensitive; nomic-v1.5 and gemma fail (modal LOO 0.65 and 0.59 respectively, both small-n at n=10), while nomic-v2-moe (modal LOO 0.92) passes cleanly at p=0.004; the failure pattern is consistent with the small-n rank-test regime developed in §6 L7, and pairs with the §4.6→§8.1 power-analysis curve showing p<0.05 is reached with 2 author-estimated additions to the Mosteller-grounded set."* Note: the "consistent with" is doing real work — we have evidence that's compatible with the L7 framing, not evidence that proves it.

**Integration target in paper:** §4.6 modal paragraph rewrite. Honest framing: model-sensitivity story (with the gemma+moe contrast as the empirical content) plus the §8.1 power-analysis curve as the structural-prediction confirmation. Probably 2–3 sentences in §4.6 plus possibly one row added to a table. Once mxbai and qwen3 land, the table either confirms (3/5 passes) or further nuances the framing.

---

### §8.5 — 0.E Evaluative non-epistemic control  (LANDED)

**Integration status:** LANDED in `paper.md` §6 L4 + §4.6 at commit `db03bbb`; §1 contributions list aligned at `f205407`. The original `PURE_NON_EPISTEMIC` set is dropped from the main text per Codex's recommendation; the EVALUATIVE control replaces it as the load-bearing non-epistemic null. The MIXED set stays unchanged as the epistemic-leakage probe.

**Goal:** replace the original `PURE_NON_EPISTEMIC` set in `experiment_10_null_hypothesis.py` (lines 143–157) — flagged by Codex for grammatical mismatch — with a grammatically clean evaluative set whose semantics are uniformly non-probabilistic.

**Proposed adjective set (13 items, all *that*-complement-grammatical, scrambled-arbitrary labels in the 20–80 range so they do NOT track natural evaluative valence):**

```python
EVALUATIVE = {
    "tragic": 60, "regrettable": 35, "unfortunate": 75, "disappointing": 20,
    "concerning": 50, "troubling": 80, "interesting": 30, "noteworthy": 65,
    "fitting": 25, "encouraging": 70, "satisfying": 40, "fortunate": 55,
    "wonderful": 45,
}
```

**Status as of 2026-05-02:** First version of `experiment_10c_evaluative_control.py` was written with three protocol drifts vs. canonical (BARE_CLAIM trailing period, phrases not lowercased, `train_axis` missing `/std()` on `medians_c`) and produced an impossible Mosteller predicative LOO ρ = −0.022. Codex caught two of the three drifts in independent code review against `paper_validation.py`. Rewritten to import canonical functions directly from `paper_validation.py` rather than maintain a parallel implementation; positive-control gate added to abort if Mosteller LOO ρ < 0.80. Committed `7203ae2`.

**Results (nomic-embed-text:v1.5):**

| Set | LOO ρ | Canonical reference | Match? |
|---|---:|---:|---|
| Mosteller predicative (positive ceiling) | **+0.8736** | 0.874  (paper Table 1) | ✓ within 0.001 |
| Original PURE (ungrammatical) | **+0.3058** | 0.31   (paper §4.6) | ✓ within 0.005 |
| Original MIXED (epistemic leakage probe) | **+0.6905** | 0.69   (paper §4.6) | ✓ within 0.005 |
| **EVALUATIVE (new clean control)** | **−0.4121** | — | new |

All three pre-existing reference points reproduce within 0.005 — the rewritten pipeline is byte-identical to canonical. The EVALUATIVE result is trustworthy.

**Reading of the EVALUATIVE result:** −0.41 is well below the 0.87 Mosteller ceiling and below the +0.31 original-PURE baseline; the protocol does not fit arbitrary labels on grammatical *that*-complement evaluative adjectives. The negative sign is interpretable: the embedding has a real evaluative axis (the new set spans "tragic" → "wonderful"), ridge regression detects this direction from the diff vectors, and the protocol's LOO predictions therefore track natural valence — which the deliberately-scrambled arbitrary labels disagree with. So the negative sign is feature, not bug; it's evidence the embedding has multiple semantic axes (the paper already knows this from §4.1's PCA-PC1 = syntactic frame finding).

**Integration target — §6 L4 prose update (specific edit):** Replace the current PURE_NON_EPISTEMIC framing ("LOO ρ = 0.31 against author-assigned arbitrary labels, well within the random-label null distribution") with prose grounded in the new clean control: *"A grammatically clean control of thirteen evaluative *that*-complement adjectives ('tragic', 'regrettable', 'unfortunate', through 'wonderful'), placed in the same predicative template and trained against deliberately-scrambled arbitrary labels, returns LOO ρ = −0.41 — well below the 0.87 Mosteller ceiling. The negative sign reflects the protocol detecting the embedding's evaluative axis and producing predictions that track natural valence rather than the arbitrary labels we trained on, evidence that the protocol does not trivially fit any-labels-to-any-grammatical-adjectives in this template."* Keep the MIXED (`NON_EPISTEMIC_ADJECTIVES`) set unchanged as the epistemic-leakage probe; LOO ρ = 0.69 there is interpreted as the geometry's epistemic sensitivity to leakage in items like "expected" / "obvious" / "documented."

The original PURE set (LOO ρ = 0.31) can either be dropped from the paper entirely or kept as a transparency note. Codex's preference is to drop it — the new EVALUATIVE control is cleaner, and reporting both invites the reviewer question "why two non-epistemic controls?". Recommendation: drop original PURE from main text; EVALUATIVE replaces it as the load-bearing non-epistemic control.

---

### §8.6 — Codex's revised integration plan (after seeing 0.B + 0.C-mxbai + 0.D)

**Status:** Codex confirmed the framing in `tmp/codex-feedback-01-33.md` (his most recent read after seeing all the new findings). Five-point plan below was verified item-by-item against the new evidence. The four-claims restructure (§4.6 in this file) and contribution-list compression (8→3-4) are trim-phase work, deferred until empirical-evidence integration is complete.

After seeing the new findings, Codex revised his earlier read in a favorable but bounded direction. Captured here so the recommendations don't get lost.

**Headline shift in his read:**

- Lambda sweep substantially fixes the regularization concern, but supports "qualitatively robust" not "λ-insensitive" — say so explicitly.
- Label-permutation null is the strongest single new result. He upgrades his prior skepticism of the §4.4 erasure section. Recommends "two complementary nulls" framing for §4.4.
- Modal-null story is no longer a hole; reframes as "small-n + model-sensitive" with the gemma/moe contrast and the §8.1 power-analysis curve as supporting structural evidence.
- Larger-n predicative/adverbial/noun-phrase nulls on gemma and moe are the cleanest reviewer-facing evidence among the new runs.

**What does NOT change in his read:** the paper still needs to narrow hard. The new evidence makes a *tighter* paper possible, not a *bigger* one. Concept erasure stays a mechanistic diagnostic, not a co-equal headline claim — particularly while qwen3 label-perm is incomplete and the erasure analysis still uses in-sample-calibrated axes rather than LOO. The center stays where it belongs: psychometrically grounded within-type calibration.

**His five-point integration plan when we move into the trim/integration phase:**

1. **§3.3 lambda sweep** — replace the "informally verified" hand-wave with a precise 2-sentence robustness/caveat statement (per §8.3 above; do NOT say "insensitive" without the asymmetry clause).
2. **§4.4 concept erasure** — rewrite around the two-null framing. Report only the headline predicative↔modal pair plus the 5/12 both-null summary. Frame as "matched-random tests whether this *direction* damages more than geometrically matched directions; label-permutation tests whether the *probability content* of the direction is load-bearing." Strongest claim is pairs that pass both nulls.
3. **§4.6 modal null** — revise the modal paragraph: nomic-v1.5 and gemma fail at small n; moe passes cleanly; pairs with §8.1 multi-n power analysis as a small-sample-power interpretation. Use the careful "consistent with" hedging from §8.4.
4. **§6 L4 evaluative control** — keep out of the paper until §8.5 is unblocked.
5. **Center of mass** — keep §4.1 (within-type calibration) and §4.2 (Mosteller→Vogel→Wintle triangulation) as the center; concept erasure / multilingual / robustness sit as supporting evidence, not as competing headlines. (This is the same structural editorial point Codex made in his earlier read, reaffirmed.)

These five do NOT replace §4 trim of TODO.md (they don't get us from 25 pages to 10 by themselves). They're the specific prose deltas the new experimental evidence justifies, on top of whatever section-level cuts we make in the trim pass.

---

### §8.7 — M10 modal-axis outliers (`improbably`, `very probably`); M9 as a revision-pass option

**Finding (2026-05-02, surfaced by data-module agent's M10 reruns; commit `c6325f7` for results, prose integration in commits `463ab33` for §4.5/§4.7).** Under canonical M10 the modal axis on nomic-embed-text v1.5 has full-dimensional LOO ρ = 0.648 — substantially below the M15 number (0.948) that earlier exploratory runs reported. Two M10 items account for the gap: `improbably` (true 12.5%, predicted 50–58% across all five embedding models tested) and `very probably` (similar pattern, smaller magnitude). The mechanism is morphological-derivative drift: the adverbial form `improbably` is often used as an intensifier ("improbably handsome") rather than as a probability hedge, so its embedding sits closer to 50% than to its source-adjective median of 12.5%. This is a genuine linguistic finding, not a code bug, and is now surfaced honestly in §4.5 truncation prose.

Truncation in fact mitigates the distortion (M10 modal LOO ρ on nomic rises from 0.648 at full d to 0.806 at 64d — the 64d projection apparently drops dimensions where `improbably`'s intensifier-loaded geometry lives).

**Why M10 stays canonical for first submission.** M10's calibration grounding is the load-bearing virtue: every M10 item has a probability value derived from Mosteller's 1990 psychometric survey. M15's 7 author-estimated additions (`conceivably`, `definitely`, `perhaps`, `maybe`, `presumably`, `undoubtedly`, `arguably`) trade calibration grounding for sample-size headroom. The paper's central novelty claim — *"calibrated to human psychometric medians from \citet{mosteller1990quantifying}"* — survives M10 cleanly and is compromised by M15. Codex's preserve-negative-results principle applies: the `improbably` outlier finding makes the paper more credible, not weaker.

**M9-Mosteller-only-clean as a revision-pass option (NOT for first submission).** If a reviewer presses on the modal LOO ρ = 0.648 number being lower than the larger-n axes, the principled remediation is to drop `improbably` (and possibly `very probably`) from M10 on the grounds that the adverb form has demonstrable semantic drift from the source adjective. Call this M9 (or M8 if both items go). This move would:

- *Preserve* the calibration-grounding virtue: every remaining item is still Mosteller-grounded.
- *Increase* the modal LOO ρ substantially (the agent's data suggests this is largely the cause of the drop from 0.948 M15 to 0.648 M10).
- *Be defensible on linguistic grounds*: we'd be excluding items for documented semantic-drift reasons, not because they hurt the numbers. The exclusion criterion is principled (adverb-form drift from source adjective); the resulting set is still psychometrically grounded.
- *Require rerunning* the modal-axis-using experiments (§4.1 Table 1 modal column, §4.4 concept erasure, §4.6 null tests, §4.5 truncation, §4.7 ensemble) on the new M9 set. The data module already supports this cleanly via the `--modal=` flag pattern; a `MODAL_M9` constant in `data/mosteller.py` plus a `--modal=m9` flag on each experiment is the implementation.

**For first submission:** keep canonical M10 with the `improbably` outlier surfaced honestly in §4.5 prose. The M9 path is in reserve for revision response if a reviewer specifically asks why the modal LOO ρ is lower than the larger-n axes. Acting on it pre-emptively would look like cherry-picking; it's defensible only as a *response* to a critique that names the specific concern.

**Cross-references in current paper.md:**
- §4.5 truncation paragraph names `improbably` and `very probably` as systematic outliers and gives the morphological-derivative-drift explanation.
- §4.7 ensemble paragraph notes the model-dependent ensemble-vs-modal pattern that the M10 outlier partially explains.
- §6 L4 (small-n + modal axis) does not currently mention the outlier finding specifically — would be a natural place to add a clause if §4.5 prose feels insufficient.

---

## §9 — Manuscript-quality items captured from Codex 01-33 read (post-trim phase)

Per Joseph (2026-05-02): defer rendering / manuscript-quality issues until after the trim phase. Captured here so they don't get lost. Categorized as non-trim items extracted from `tmp/codex-feedback-01-33.md`; trim/restructuring items in that file are deliberately not duplicated here.

### §9.1 — `\bottomrule` placement bug in Table 1 (paper.tex:344)

Table 1 in `paper.md` has a header row, a "sup. / LOO" sub-header row, then the body rows. `convert_to_tex.py` is mishandling the table conversion: the generated `paper.tex` shows `\bottomrule` after `\midrule` but **before** the body rows (paper.tex line ~344, between the model header line and the data rows). Booktabs `\bottomrule` belongs at the end of the table, after the last data row. Visual symptom in the rendered PDF: a horizontal rule appears between the column header and the first data row where there shouldn't be one.

**Fix path:** investigate `convert_to_tex.py`'s table-handling logic. The "sup. / LOO" row probably parses as something other than a body row (maybe interpreted as a separator?). Fix the converter, regenerate, verify Table 1 has only `\toprule` ... `\midrule` ... data rows ... `\bottomrule`.

### §9.2 — Working-doc residue still in `paper.md` source

Even though `convert_to_tex.py` strips most blockquotes (which is why the §4.4 intent block stayed invisible until I rewrote it), submission source should not depend on "strip-this-later" behavior. Specific residue:

- Line 5 — opening **Working draft, May 2026** blockquote (project provenance / placeholder description).
- Line 332 — `## References` section's blockquoted **Intent** placeholder with the full audited citation lists (linear-feature decoder line, behavioral psychometrics line, embedding-probing line, methodology, data, theoretical, embedding models). Useful internally; not for submission.
- Line 352 — `## Supplementary` section's blockquoted **Intent** placeholder describing what the supplementary would contain.
- Line 356 — `## Open dependencies` section, with both done items (concept-erasure v2, Undermind audit) and outstanding TODO items.

**Fix path:** delete these sections from `paper.md` (or move them to `brainstorms/` if any of the content is still load-bearing). The actual references appear in the compiled PDF via natbib + `refs.bib`, not via this Markdown section.

### §9.3 — 53 vs 52 Mosteller-count discrepancy

The abstract (line 11) and §1 (line 25) and §2.4 (line 67) and §3.1 (line 89) all say "53-expression Mosteller dataset" / "53 verbal expressions". The PCA results in §3.1 line 93 and §4.1 line 169 say "PCA on the 52 mixed-type Mosteller difference vectors." Type counts in §3.1 sum: predicative 13 + frequency adverb 19 + noun phrase 11 + modal 10 = 53.

So the 52-vs-53 gap is one item excluded from the mixed-type PCA but retained in the within-type analyses. Likely candidates: an item that's in two type partitions (and excluded from the mixed-PCA to avoid double-counting), or one that fails the PCA construction (e.g., empty / NaN diff vector). The data and FINDINGS files would say which.

**Fix path:** identify the excluded item and either (a) name it in §3.1 in one parenthetical clause (e.g., *"PCA on the 52 mixed-type difference vectors — modal {expression} appears in both predicative and modal types and is dropped from the mixed-type PCA to avoid duplication — recovers PC1 as a syntactic-frame direction..."*), or (b) if no exclusion is intended, fix the count to 53.

### §9.4 — Codex 01-33 trim/restructuring directives (deferred to trim phase, captured for reference)

The file `tmp/codex-feedback-01-33.md` contains a comprehensive trim/restructuring plan: collapse 8 contributions to 3–4, target a 9-page main paper, demote concept erasure / multilingual / robustness to support claims, drop the "language-constituted agents" §5 paragraph for TACL, merge L7+L8 in §6, etc. These are NOT captured here; the file is the source of truth. When the trim phase begins, that file is the integration plan.

---

## §7 — After submission (companion-paper directions)

These are *not* on the critical path for the TACL submission. They're flagged here so they don't get lost when this TODO closes out. Full discussion in `PLAN.md` §V Phase 2.

- Self-report-truth-axis experiment (highest-impact follow-up; converts the paper from "we measured a thing" to "we built an instrument used for honest-activation discipline").
- Cross-model-class convergence (paper-of-record potential; same sentence projected through our pooled-embedding axis, Ji 2025's decoder feature, Belém/Tang prompted LLM elicitation).
- Causal measurement model (formalize §5's convergent-instruments close as a hierarchical Bayesian model).
- Fast/slow/authentic.
- Native-speaker cross-linguistic psychometric replication.
- Real-text corpus evaluation.
- §03.II scaffolded-logogenic agent application (use the calibrated axis as input to the architectural-bias bound — bridges to the agentic-systems repo's active work).

---

## Notes from the three audits

- **Gemini #1 (strategic)** said skip the trim. Overruled by Codex's reading of the actual TACL formatting PDF, pending the §1 verification.
- **Codex** found the bugs in §2 and recommended the trim in §4. This list incorporates both.
- **Gemini #2 (code/math/data audit)** found math is bulletproof, no data leakage, results match logs precisely. Verdict: we don't need to rerun experiments to publish; we just need to fix the manuscript.

The bugs in §2 are submission-blocking but mechanical — the new Gemini verdict means there's no upstream methodology to relitigate. Most of the work between us and a clean submission is editorial. That's a good place to be.
