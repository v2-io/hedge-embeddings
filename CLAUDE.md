# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Research codebase demonstrating that epistemic hedging (uncertainty language like "probably", "certainly", "possibly") manifests as calibrated linear structure in sentence embedding space. The structure is calibrated to human psychometric data (Mosteller & Youtz 1990), cross-validated against two independent datasets (Vogel 2022, Wintle 2019), and transfers zero-shot across 8 typologically diverse languages.

**Status:** 10 experiments complete. Preparing paper for submission (CMCL 2026 workshop abstract + TACL or ICLR 2027 full paper).

## Running Experiments

Requires Python 3.13+, numpy, scipy, scikit-learn, pandas, requests. No package manager—deps are installed globally.

**Ollama must be running locally** (`ollama serve`) with embedding models pre-pulled before running any script.

```bash
# Run with default model (nomic-embed-text:v1.5)
python3 experiment_01_hedge_direction.py

# Run with a specific model
python3 experiment_01b_within_type.py mxbai-embed-large

# Cross-linguistic experiment uses bge-m3 by default
python3 experiment_09_cross_linguistic.py
```

Every script takes an optional model name as the sole CLI argument. All output goes to stdout; scripts never write files or mutate state. Save results by redirecting: `python3 script.py model > results/output.txt 2>&1`

There is no test framework. Each experiment script IS the test.

## Architecture

**Self-contained experiment scripts.** Each `experiment_*.py` deliberately duplicates helper functions (`embed_texts`, `normalize`, `train_axis`) rather than importing from a shared module. This is an intentional design choice for independence.

**Common pattern in every script:**
1. Define `MODEL` from `sys.argv[1]` or default
2. Call Ollama's `/api/embed` endpoint at `http://localhost:11434/api/embed`
3. Hardcode Mosteller training expressions with median probabilities inline
4. Compute difference vectors (hedged embedding − bare claim embedding)
5. Train probability axis via ridge regression (`np.linalg.solve`, λ=0.1)
6. Project and convert to probability via linear calibration
7. Print structured results

**Ground truth:** `docs/mosteller_youtz_1990_full.csv` — 53 verbal probability expressions from Mosteller & Youtz 1990 (n=238).

**Cross-validation:** `docs/vogel_2022_systematic_review.csv` (21 studies, 1967–2018). Wintle et al. 2019 (n≈924) values are hardcoded in experiment_08.

## Key Documents

- **PLAN.md** — Master plan with project status, decisions, path forward
- **EXPERIMENT-PLAN.md** — Original experiment plan with results summary
- **FINDINGS-01.md through FINDINGS-06.md** — Detailed results per experiment
- **feedback-analysis.md** — Paper structure and venue analysis
- **literature-review-undermind.md** — Automated literature review confirming novelty
- **cmcl-abstract-draft.md** — Draft CMCL 2026 non-archival abstract
- **workshops.md** — Workshop venue analysis
- **results/** — Version-controlled raw output (17 files)
- **docs/** — Theoretical foundations, calibration data, reference CSVs

## Key Findings

- Probability IS linear structure in embedding space (ρ > 0.90 supervised, 0.70–0.94 LOO, 6 models)
- **Type-specific** axes: predicative, frequency, noun phrase, modal. Mixing types confounds syntax with semantics
- **Three-dataset cross-validation:** Mosteller → Vogel ρ = 0.991; Mosteller → Wintle ρ = 0.967
- **Cross-linguistic:** English axis → 8 languages zero-shot, mean |ρ| = 0.928 (Korean pred ρ = 1.000, Hindi modal ρ = 1.000)
- **Null hypothesis confirmed:** Permutation p ≤ 0.005; non-epistemic adjectives LOO ρ = 0.31 vs. 0.87 real
- **Bimodality probe:** Modifiers traverse "possible" ambiguity range (spread 20–32pp)
- **Truncation:** Axis survives to 64d on MRL model (LOO ρ drop = 0.027)
- Modal axis is near-universal fallback (MAE 3.2–8.5%)
- IQR/interpretive precision NOT in geometry (informative negative)
- Compound hedges compose ~linearly; negation breaks it (probability-reflecting)
- Ensemble doesn't beat single modal axis

## Models Tested

| Model | Dims | Notes |
|---|---|---|
| nomic-embed-text:v1.5 | 768 | Default, BERT+Matryoshka |
| embeddinggemma:300m | 768 | Decoder-derived (Gemma) |
| nomic-embed-text-v2-moe | 768 | Mixture of Experts |
| mxbai-embed-large | 1024 | BERT-large variant |
| qwen3-embedding | 4096 | Qwen3 architecture |
| bge-m3 | 1024 | XLM-RoBERTa, multilingual (100+ langs), used for cross-linguistic |
