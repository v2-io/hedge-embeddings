# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Research codebase investigating whether epistemic hedging (uncertainty language like "probably", "certainly", "possibly") manifests as consistent linear structure in sentence embedding space, and whether that structure is calibrated to empirically measured probability.

## Running Experiments

Requires Python 3.13+, numpy, scipy, scikit-learn, pandas, requests. No package manager—deps are installed globally.

**Ollama must be running locally** (`ollama serve`) with embedding models pre-pulled before running any script.

```bash
# Run with default model (nomic-embed-text:v1.5)
python3 experiment_01_hedge_direction.py

# Run with a specific model
python3 experiment_01b_within_type.py mxbai-embed-large
```

Every script takes an optional model name as the sole CLI argument. All output goes to stdout; scripts never write files or mutate state.

There is no test framework. Each experiment script IS the test—it trains axes, runs probes, and prints results.

## Architecture

**Self-contained experiment scripts.** Each `experiment_*.py` deliberately duplicates helper functions (`embed_texts`, `normalize`, `train_axis`) rather than importing from a shared module. This is an intentional design choice for independence. The one exception: `experiment_01b` imports `CLAIM_TRIPLES` from `experiment_01_hedge_direction`.

**Common pattern in every script:**
1. Define `MODEL` from `sys.argv[1]` or default `nomic-embed-text:v1.5`
2. Call Ollama's `/api/embed` endpoint at `http://localhost:11434/api/embed`
3. Hardcode Mosteller training expressions with median probabilities inline
4. Compute difference vectors (hedged embedding − bare claim embedding)
5. Train probability axis via ridge regression (`np.linalg.solve`, λ=0.1)
6. Project novel phrases and convert to probability via linear calibration
7. Print structured results

**Ground truth:** `docs/mosteller_youtz_1990_full.csv` — 53 verbal probability expressions from Mosteller & Youtz 1990 (n=238), providing P25, Median, P75, IQR.

## Key Documents

- **EXPERIMENT-PLAN.md** — Master plan with full results summary and remaining frontiers
- **FINDINGS-01.md through FINDINGS-06.md** — Detailed results and analysis per experiment
- **docs/** — Theoretical foundations, calibration data, and reference CSVs (not generated output)

## Key Findings to Preserve

- Probability IS linear structure in embedding space (validated across 5 architecturally different models, ρ > 0.90 supervised)
- The structure is **type-specific**: predicative adjectives, frequency adverbs, noun phrases, and modal adverbs each have their own axis. Mixing types confounds syntax with semantics.
- Modal axis is the near-universal fallback (MAE 3.2–8.5%)
- IQR/interpretive precision is NOT in the geometry (informative negative result)
- Negation breaks linear composition — it acts as probability-reflecting, not linear
- Compound hedges compose approximately linearly in direction, sub-linearly in magnitude

## Models Tested

| Model | Dims | Notes |
|---|---|---|
| nomic-embed-text:v1.5 | 768 | Default, BERT+Matryoshka |
| embeddinggemma:300m | 768 | Decoder-derived (Gemma) |
| nomic-embed-text-v2-moe | 768 | Mixture of Experts |
| mxbai-embed-large | 1024 | BERT-large variant |
| qwen3-embedding | 4096 | Qwen3 architecture |
