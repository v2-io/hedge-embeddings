# §4.4 prose rewrite — brief for the next session

> **Working brief for whoever picks up the §4.4 prose rewrite.** Captures the editorial decisions, recommended structure, numbers to use, and reviewer-bait to preserve, so the rewrite can land cleanly without re-deriving everything from the three external reviews. Drafted 2026-05-02 by the prior session before context handoff.

---

## What's actually broken

§4.4 in `paper.md` is currently a blockquoted intent block from lines (≈238–262) with sub-sections (a)–(f). `convert_to_tex.py:96` strips blockquotes, so the entire body of §4.4 is *missing from the compiled PDF* — only the section heading `### 4.4 Functional validation via concept erasure` and the fig13 caption survive. This is the single biggest remaining submission-blocking bug. The math is bulletproof per the third Gemini code-audit; this is purely a manuscript-layer problem.

## What §4.4 has to deliver

The empirical content, in priority order:

1. **Cross-model robust signature: Pearson r(cos(v_A, v_B), ΔMAE_real) = +0.92 (mxbai) / +0.86 (qwen3)** across all 12 ordered cross-type pairs. This is the headline empirical claim — it's robust across two architectures and gives the §4.4 result a single number per model that summarizes the whole table.

2. **Headline pair predicative ↔ modal both directions on both models.** Specific numbers:
   - mxbai-embed-large, Predicative → Modal: ΔMAE 6.79 → 25.16 (+18.37 pp); cos = 0.880; matched-random ΔMAE mean +18.06 ± 0.16; ratio ≈ 1.02; z_match_MAE = +2.02.
   - mxbai-embed-large, Modal → Predicative: ΔMAE 4.58 → 29.20 (+24.62 pp); matched-random mean +22.01 ± 0.32; ratio ≈ 1.12; z_match_MAE = +8.18.
   - qwen3-embedding, Predicative → Modal: ΔMAE 5.17 → 22.53 (+17.36 pp); z_match_MAE = +26.88.
   - qwen3-embedding, Modal → Predicative: ΔMAE 5.47 → 25.96 (+20.49 pp); z_match_MAE = +18.35.

3. **The ρ-vs-MAE dissociation framing.** ρ saturates at small n (in-sample fit ≥ 0.95 across all four axes for both models on the n=10–19 sets), so the v1 ρ-only view of erasure looked dissociated across models (5/12 mxbai vs. 3/12 qwen3 functional, zero overlap). MAE has headroom; the MAE view recovers cross-model agreement (9/12 ordered pairs MAE-functional in both models simultaneously). MAE is the load-bearing metric.

4. **Methodological framing.** §4.4 is the embedding-class structural analogue of decoder-LLM activation steering — same structural role, different mechanism. We frame it as analogue-of, not substitute-for (§6 L5).

## Codex's editorial recommendations to honor

These are the load-bearing reframing decisions Codex's audit asked for, beyond just turning blockquote into prose:

- **Lead with raw ΔMAE and ΔMAE / matched-null *ratio*, not z-scores.** The z-scores look reviewer-hostile when matched-null std rounds to ~0 (qwen3 has +26.88, looks like overclaim rather than evidence). Effect-size ratios are more interpretable. The mxbai Predicative→Modal *ratio* of ΔMAE_real / ΔMAE_matched-null-mean = 18.37 / 18.06 ≈ 1.02 reads as "real-axis erasure produces 2% more MAE damage than matched-random" — that's the honest framing, and it's the same finding the z = +2.02 reports, just less inflated-looking.
- **Compress the (a)–(f) substructure.** Fold (c) and (c′) together. (e) reviewer-bait answers → one paragraph at most. Per-pair full table → supplementary if the page budget permits one, otherwise drop.
- **Target length: ~0.9 pages including fig13.** The current intent block in markdown is verbose; the prose version should be substantially tighter while preserving the empirical numbers and the methodological framing.
- **Integrate the fig13 reference natively into the prose,** not as a "Headline figure (to make)" intent note. The figure now exists (`figures/fig13_erasure_mae_cos.{pdf,png}`).

## Recommended prose structure (~3 paragraphs)

**¶1 — Cross-model headline.** Open with the cross-model ΔMAE-cos correlation (r ≈ 0.9 both models) and the structural claim it supports: erasing a probability-aligned axis from cross-type embeddings degrades calibration in proportion to the inter-axis cosine alignment, and this scaling holds across two architecturally distinct models. Reference fig13 here. ~5 sentences.

**¶2 — Headline pair details and the matched-random control.** Predicative↔modal both directions both models. Lead with ΔMAE (18–25pp on the headline pair) and the matched-null ratio framing (real-axis vs. matched-random ΔMAE). Mention z-scores once, briefly, with the caveat that small matched-null std on some pairs makes raw ΔMAE-ratio more interpretable. Acknowledge §6 L6's over-conservative-at-high-cos point — the matched-random control reproduces ~98% of real-axis MAE damage at cos = 0.88, so the +2 threshold is genuinely stringent. ~6 sentences.

**¶3 — Why MAE is the load-bearing metric and what the cross-pair pattern shows.** ρ saturation at small n (cf. §6 L7). v1 ρ-only view looked cross-model-dissociated; MAE view recovers cross-model agreement (9/12 pairs MAE-functional in both models). The cross-pair r ≈ 0.9 in both models is the structural pattern; per-pair functional verdicts on individual low-cosine pairs are below the experiment's resolving power and are reported in supplementary only. ~5 sentences. Close on the §3.4 → §4.4 → §5 thread: this is the embedding-class structural analogue of decoder-LLM causal intervention, validated via representation-level erasure under a calibration-aware downstream linear decoder, with the methodological-framing detail in §5 ¶1.

## What §4.4 should NOT do

- Re-define the experimental protocol (already in §3.4).
- Restate the §6 limitations on cosine-matched-control over-conservatism (it's in L6) or small-n saturation (L7) — reference them, don't re-litigate.
- Get into the §5 measurement-theoretic interpretation (that's §5's job).
- Put per-pair tables inline. Either move to a small supplementary (if room) or drop entirely.
- Fight the in-sample-vs-LOO calibration question. §6 L6/L7 acknowledge it; the v3 LOO-concept-erasure rerun is flagged as future work in §6 and §V Phase 2 of PLAN.md. The §4.4 prose just needs to acknowledge the in-sample choice once, briefly, with cross-reference to §6.
- Lead with the ASF measurement-infrastructure framing — that lives in §5 ¶4. §4.4 is empirical results, not interpretation.

## Cross-references to preserve

- `§3.4` — concept erasure protocol definition.
- `§3.1` — modal n=10 Mosteller-grounded set (the canonical 10-item set used for the modal axis in this experiment).
- `§4.1` Table 1 — within-type LOO ρ values (gives readers the saturation context).
- `§5 ¶2` — four-axis subspace geometry that the §4.4 ΔMAE-cos correlation is the functional confirmation of.
- `§5 ¶1` — methodological framing of §4.4 as embedding-class analogue of decoder-LLM causal intervention.
- `§6 L5` — no-causal-intervention-in-decoder-steering-sense limitation.
- `§6 L6` — matched-random control over-conservative at high cos.
- `§6 L7` — small-n rank-saturation regime.
- `fig13` — the cross-pair scatter (now exists at `figures/fig13_erasure_mae_cos.{pdf,png}`; not "to make" anymore).

## Source data files

All in `results/`:
- `exp11_mxbai.txt` — has the full 12-pair table with ρ and MAE columns plus all the matched-random distributions.
- `exp11_qwen3.txt` — same for qwen3-embedding.
- The structure of these files is documented in `experiment_11_concept_erasure.py` lines 197–432; both files were regenerated cleanly in the v2 run with MAE distribution stats added.

## Numbers to double-check before writing

The MAE / cos / z-score values quoted above are from the integrated state in `paper.md` §4.4 intent block. They were verified against `results/exp11_*.txt` by the v2 experiment-runner agent (see commits `a60b99b` and `6040540` for the audit trail). The third Gemini code-audit also independently verified the §4.4 numbers match the result-file logs precisely. These should not need re-extraction.

## Build verification after rewrite

```bash
python3 convert_to_tex.py
```

Should emit `lualatex: OK (paper.pdf built)`. Then verify:

```bash
grep -c "PLACEHOLDER" paper.md          # expect 2 (working-doc references only, not actual placeholders)
grep -c "intent" paper.tex              # expect 0 in §4.4 region
pdftotext paper.pdf - | grep -A2 "4.4 Functional"  # should show real prose, not just heading + figure
```

Also page count should drop a bit from the current 25 (§4.4 intent block compressed to ~0.9 pages of prose net-trims relative to the intent block's verbose bullet structure).

## Editorial sensibility notes

- Codex's structural editorial point applies here: *"The scientific center should be the psychometrically grounded, within-type verbal-probability axis; concept erasure and multilingual transfer should support that center rather than become competing headline claims."* Don't lift §4.4 to co-equal headline status with §4.1 + §4.2; keep it as functional-validation support for the central within-type result.
- Prose register: TACL / journal-formal, matching §1, §2, §3, §6 in the current paper.md. Avoid the bullet-list / sub-headed-paragraph style of the intent block; standard prose paragraphs only.
- Citation conventions: use \citep{key} / \citet{key} per the natbib rewrite already done elsewhere in the paper. Relevant keys for §4.4: `ji2025calibrating`, `marks2023geometry`, `valois2025frame`, `lepori2026fantasy`. Don't introduce new citations.

## Coordination with other in-flight work

None right now. The two agents that ran during the previous session (modal-set unification on `paper_validation.py`, modal-extension of `experiment_10_null_hypothesis.py`) are both done; their outputs are committed. The §4.4 rewrite is independent; it doesn't depend on any pending work.

CJK rendering for §6 L2 is parallel and on the coauthor's track; doesn't affect §4.4.

## After §4.4 lands

Next pieces in TODO.md priority order:
1. Trim pass to 10-page TACL limit (collaborative — coauthor proposes specific cuts, agent executes). PLAN.md §V Phase 1.F has the section-level targets.
2. The three permutations in TODO.md §0 are scientifically illuminating but optional; decide whether they fit in the page budget.
3. Six citation gaps from `refs.bib` to verify before submission (TODO.md §5).
4. CJK rendering (coauthor track).

## Recommended approach

Ideally a fresh session reading: this brief → current §4.4 intent block in `paper.md` → Codex's audit (in conversation history if available, or summary in `TODO.md` §2) → the four `results/exp11_*.txt` files → fig13 caption already in `paper.md`. Then write the rewrite as a single Edit replacing the entire blockquote intent block with three paragraphs of prose plus the existing fig13 reference. Recompile, verify page count drops, commit.
