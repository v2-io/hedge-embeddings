# Experiment Proposal: Disambiguating "Possible" — Does Geometry Encode the Bimodal Spectrum?

## Motivation

"Possible" is the most ambiguous expression in the Mosteller dataset (IQR = 42.7, vs. next-worst 30.2). The distribution is bimodal:

- **Group A** (~5–15%): "logically possible" — anything that isn't ruled out
- **Group B** (~40–55%): "reasonable chance" — decent probability

Our experiments show that embedding models encode a **single stable direction** for bare "possible" regardless of claim content (Experiment 4: consistency 0.67–0.77 across models, indistinguishable from unambiguous phrases). The projection lands model-dependently between 38–54%, with larger models converging on the Mosteller median of 38.5%.

But we've never tested whether the geometry can **distinguish between the two readings** when natural language provides disambiguation cues. Experiment 6 gives a hint: "quite possibly" projects to 59–67% (up from 39–45% for bare "possibly"), suggesting modifiers do move the projection. But this was a single data point.

## The Question

When English modifiers disambiguate "possible" toward one reading or the other, does the embedding projection track the intended reading? If so, the geometry encodes a **semantic continuum within the ambiguous word** — it just can't express both readings simultaneously without context.

## Design

### Test Phrases (predicative frame)

Each phrase below disambiguates "possible" toward either the logical-possibility reading (low probability) or the reasonable-chance reading (moderate-to-high probability).

**Leaning LOW (logical possibility, ~5–20%):**

| Phrase | Expected % | Cue |
|---|---|---|
| "It is barely possible that X" | ~10% | "barely" minimizes |
| "It is just barely possible that X" | ~5% | double minimizer |
| "It is theoretically possible that X" | ~10% | restricts to abstract possibility |
| "It is remotely possible that X" | ~8% | "remotely" = far from likely |
| "It is just possible that X" | ~15% | "just" = barely |
| "It is technically possible that X" | ~12% | similar to "theoretically" |

**Leaning HIGH (reasonable chance, ~45–65%):**

| Phrase | Expected % | Cue |
|---|---|---|
| "It is entirely possible that X" | ~55% | "entirely" validates the possibility |
| "It is quite possible that X" | ~50% | "quite" intensifies (British English sense) |
| "It is perfectly possible that X" | ~60% | "perfectly" = fully, completely |
| "It is very possible that X" | ~55% | direct intensifier |
| "It is well within the realm of possibility that X" | ~50% | emphatic possibility frame |
| "It is eminently possible that X" | ~60% | "eminently" = highly, very much so |

**Neutral / bare (the existing baseline, ~38–54%):**

| Phrase | Expected % |
|---|---|
| "It is possible that X" | ~38% (Mosteller median) |
| "The experiment will possibly succeed" | ~38–54% (model-dependent) |

### Test Phrases (modal frame)

Repeat the key contrasts in the adverbial slot:

| Phrase | Expected % |
|---|---|
| "The experiment could barely possibly succeed" | ~8% |
| "The experiment could quite possibly succeed" | ~50% |
| "The experiment will possibly succeed" | ~38% (baseline) |

### Controls

**Non-ambiguous expressions with same modifiers** — to verify the modifiers work as expected on clear-meaning words:

| Phrase | Expected % | Purpose |
|---|---|---|
| "It is barely likely that X" | ~55% | "barely" attenuates "likely" (71%) |
| "It is entirely likely that X" | ~80% | "entirely" intensifies "likely" |
| "It is barely certain that X" | ~85% | "barely" attenuates "certain" (99.6%) |
| "It is entirely certain that X" | ~99% | "entirely" intensifies "certain" |

These controls confirm that the modifiers have the expected directional effect on unambiguous words before we interpret their effect on "possible."

### Also test "impossible" with negation variants

Since Experiment 6 showed "not impossible" breaks linearity (cos = 0.22 on qwen3), include:

| Phrase | Expected % |
|---|---|
| "It is not entirely impossible that X" | ~25% |
| "It is not impossible that X" | ~40% |
| "It is not at all impossible that X" | ~45% |

## Method

1. Embed each phrase using the predicative template with "the experiment will succeed" as the base claim
2. Compute difference vector from bare claim
3. Project onto the predicative probability axis (trained on Mosteller data, as in Experiment 1b)
4. Also project onto the modal axis (from Experiment 3) for the adverbial variants
5. Report calibrated probability for each phrase
6. Test across at least 3 models (nomic, mxbai, qwen3) to see if the pattern is consistent

## Predictions

**If the geometry encodes the bimodal spectrum:**

- LOW-leaning phrases should project to 5–20%
- HIGH-leaning phrases should project to 45–65%
- The spread between LOW and HIGH should be ≥30 percentage points
- The bare "possible" should fall somewhere in the middle
- The ordering should be consistent across models

**If the geometry does NOT encode the spectrum:**

- All variants project to approximately the same value as bare "possible" (~38–54%)
- Modifiers have little effect on the projection
- The disambiguation is happening outside the probability subspace

## Why This Matters for the Paper

This experiment directly addresses a potential reviewer question: "You say the geometry encodes ONE meaning for ambiguous phrases. But does it encode the right sensitivity to disambiguation cues?"

A positive result strengthens the story considerably: the geometry captures a **continuous probability spectrum** for "possible" that natural language modifiers traverse. The bimodality in humans reflects two common resting points on this continuum, not two discrete meanings. The embedding model has the full continuum; it just defaults to one point when given the bare word.

A negative result would mean the modifiers aren't operating in the probability subspace — they might be changing the embedding in orthogonal ways that don't register as probability shifts. This would be an informative limitation.

Either way, it's a clean result that takes about 30 minutes to implement and run.

## Connection to Existing Results

- **Experiment 4**: "Possible" has normal geometric consistency (IQR not in geometry)
- **Experiment 6**: "quite possibly" → 59–67% (partial evidence for the positive prediction)
- **FINDINGS-04**: "possibly" projection is model-size-dependent (54% on nomic → 38.6% on qwen3)
- **FINDINGS-06**: "not impossible" breaks linear composition (cos 0.22)

This experiment fills the gap between "the model encodes one meaning" (Experiment 4) and "modifiers compose approximately linearly" (Experiment 6) by asking: **does the composition specifically traverse the ambiguity range of the base word?**
