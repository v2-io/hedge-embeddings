# TODO.md — get the TACL paper to submission

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
