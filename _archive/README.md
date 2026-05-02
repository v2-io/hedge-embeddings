# _archive — historical and superseded materials

Files moved here have been completed, superseded, or are otherwise deprecated as
of the current state of work but are retained for traceability and context.
Don't link to them from active documents; link to the superseding artifact instead.

## Contents

### Experiment proposals — completed and folded into the paper

- **`EXPERIMENT-PROPOSAL-cross-linguistic.md`** (Feb 2026) — completed as
  `experiment_09_cross_linguistic.py`; results in `results/exp09_bge-m3_8lang.txt`;
  reported in `paper.md` §4.3 (Cross-linguistic ranking transfer) and Limitations
  L2.
- **`EXPERIMENT-PROPOSAL-possible-bimodality.md`** (Feb 2026) — completed as
  `experiment_08_bimodality_wintle.py` (the Wintle cross-validation was added in
  the same script); results in `results/exp08_*.txt`; reported in `paper.md` §4.5
  (Bimodal-modifier "possible" probe) and §4.2 (Wintle cross-validation).

### Submitted / superseded write-ups

- **`cmcl-abstract-draft.md`** (Feb 2026) — non-archival CMCL 2026 extended
  abstract; submitted by the Feb 25, 2026 deadline. Superseded by `paper.md`,
  the TACL-targeted full paper draft.
- **`literature-review-undermind.md`** (Feb 2026) — earlier Undermind literature
  review used to scope CMCL framing. Superseded by
  `Novelty_memo_for_hedge_embeddings.md` (Undermind audit, May 2026), which
  applied to the TACL-targeted manuscript and includes the post-Feb-2026 work
  (Lepori 2026 ICLR, Cohen 2025, Cho 2026, Adarsh 2026, Yang 2026).

## Active analogues

The following remain at the repo root and are *not* archived because they name
future work that the paper either references directly or that is on the planned
follow-up list:

- `EXPERIMENT-PROPOSAL-self-report-truth-axis.md` — highest-impact follow-up
  (clinical instrument for honest-activation discipline; see PLAN.md §V Phase 2).
- `EXPERIMENT-PROPOSAL-causal-measurement-model.md` — companion-paper material
  (formalizes §5's convergent-measurements close as a hierarchical Bayesian model).
- `EXPERIMENT-PROPOSAL-fast-slow-authentic.md` — companion-paper material
  (Truth Death detection / ASF Component 04 connection).
- `EXPERIMENT-PROPOSAL-corpus-llm-philosophy.md` — three sub-experiments (corpus
  grounding, LLM decomposition, philosophical argument), all future work.

## Conventions for using `_archive/`

Move material here when it is clearly completed, superseded, or no longer
actively used. Add a short note above naming what it was and what supersedes it.
Do not delete from history (`git mv` preserves the trail). When in doubt, leave
the file at the repo root and add a note rather than archiving prematurely.
