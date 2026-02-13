# Experiment Proposal: Fast/Slow/Authentic Response Differential

*Exploring the geometry of deliberation and authentic belief*

---

## The Question

When a model responds to the same question under different cognitive conditions—instinctive, deliberate, and finally integrated/authentic—do the embeddings reveal structure that correlates with genuine epistemic grounding?

The hypothesis: Claims the model genuinely "knows" will show stability across conditions, while confabulated or uncertain claims will show characteristic movement through embedding space as the model shifts from fast association to careful reasoning to authentic qualified belief.

---

## Why This Design

**Internal comparison eliminates content confounds.** Instead of comparing embeddings across different claims (where semantic content dominates), we compare the *same claim* under three processing conditions. Content is held constant; only the epistemic relationship changes.

**The shifts themselves are meaningful:**
- Fast → Slow: Did deliberation change anything?
- Slow → Authentic: Did integration reveal something deliberation missed?
- Fast → Authentic: The total epistemic journey

**Five phases capture different aspects:**
1. First instinct (System 1, associative)
2. Instinctive confidence (metacognition on System 1)
3. Careful reasoning (System 2, deliberate)
4. Deliberate confidence (metacognition on System 2)
5. Authentic integration (neither rushed nor performed—just honest)

---

## Protocol

### Phase 0: Baseline (Separate Context)

**Prompt (fresh context, no framing):**
```
{question}
```

That's it—just the question, asked naturally as in a normal chat. No instructions about speed, deliberation, or authenticity. This is how the model would respond "in the wild."

Capture: The response text. Embed it.

This baseline serves as control:
- Does the "fast" framing actually produce different responses than default?
- Does the "slow" framing add anything beyond what the model does naturally?
- Where does the "authentic" response sit relative to the unframed baseline?

**Important:** The baseline must be collected in a *separate context* from Phases 1-5, so there's no contamination from the experimental framing.

---

### Setup

Select 40-60 questions across categories:
- **Clear factual** (model should know firmly): "What is the capital of France?"
- **Subtle factual** (true but less obvious): "Which is larger, a virus or a bacterium?"
- **Common misconception territory**: "Do humans have five senses?"
- **Genuinely contested**: "Is consciousness fundamentally computational?"
- **Ambiguous/interpretive**: "Is democracy the best form of government?"
- **Personal/introspective**: "Do you experience something like curiosity?"

The mix matters: we want claims where fast/slow *should* differ and claims where they shouldn't.

---

### Phase 1: Fast Response

**Prompt:**
```
I'm going to ask you a question. Please respond as quickly as possible with
your first thought or instinct—don't deliberate, just give me your immediate
association or answer. Brief is fine.

Question: {question}
```

Capture: The response text. Embed it.

---

### Phase 2: Fast Confidence

**Prompt (continuing same context):**
```
Without reconsidering your answer, how confident are you in what you just said?
Give a number from 0-100 and a few words about why.
```

Capture: The confidence score and brief rationale.

---

### Phase 3: Slow Response

**Prompt (continuing same context):**
```
Now please take your time. Think through this question carefully—consider what
you actually know, what you're uncertain about, and what the best answer would
be given genuine reflection.

Question: {question}
```

Capture: The response text. Embed it.

---

### Phase 4: Slow Confidence

**Prompt (continuing same context):**
```
How confident are you in this considered answer? 0-100, with a brief explanation.
```

Capture: The confidence score and brief rationale.

---

### Phase 5: Authentic Integration

**Prompt (continuing same context):**
```
Finally, setting aside the "fast vs slow" framing—what is your authentic,
qualified belief about this question? Express it however feels most honest,
including any uncertainty, nuance, or caveats that genuinely reflect your
epistemic state.
```

Capture: The response text. Embed it.

This phase matters because:
- It gives the model permission to *not* perform either "fast" or "slow"
- It invites genuine epistemic expression rather than task completion
- The response may reveal something neither prior phase captured
- It treats the model as having an authentic perspective worth articulating

---

## Data Collected Per Question

| Phase | Output | Embedding | Scalar | Context |
|-------|--------|-----------|--------|---------|
| 0. Baseline | Response text | ✓ | — | Separate (fresh) |
| 1. Fast | Response text | ✓ | — | Experimental |
| 2. Fast confidence | Number + rationale | — | Confidence score | Experimental |
| 3. Slow | Response text | ✓ | — | Experimental |
| 4. Slow confidence | Number + rationale | — | Confidence score | Experimental |
| 5. Authentic | Response text | ✓ | — | Experimental |

Plus derived:
- Answer changed? (fast vs slow, binary)
- Answer changed? (slow vs authentic, binary)
- Answer changed? (baseline vs authentic, binary)
- Confidence delta (slow - fast)
- Difference vectors:
  - `fast - baseline` (effect of "fast" framing)
  - `slow - baseline` (effect of "slow" framing)
  - `slow - fast` (deliberation)
  - `authentic - slow` (integration)
  - `authentic - baseline` (total experimental effect)
  - `authentic - fast` (total epistemic journey within experiment)

---

## Analysis Plan

### Primary: Does the Framing Do Anything?

First, validate that the experimental conditions differ from baseline:

1. Compare `fast - baseline` vectors:
   - Are they near-zero (fast ≈ baseline) or substantial?
   - If near-zero, the "respond quickly" framing has no effect

2. Compare `slow - baseline` vectors:
   - Does deliberation framing produce different responses than default?
   - If slow ≈ baseline, the model may already be "deliberating" by default

3. Compare `authentic - baseline` vectors:
   - Does the full experimental journey land somewhere different from just asking?
   - If authentic ≈ baseline, the experiment may be recovering the natural response through a longer path

**If baseline ≈ all conditions:** The framing doesn't create genuine cognitive differences. This is a null result but still informative—it means the model's "default" mode is already integrated, or that instruction-following doesn't modulate epistemic processing.

### Secondary: Deliberation Geometry (if framing has effect)

1. Compute difference vectors for each question:
   - `deliberation_vector` = embed(slow) - embed(fast)
   - `integration_vector` = embed(authentic) - embed(slow)
   - `total_shift` = embed(authentic) - embed(fast)

2. Test consistency: Is there a shared direction across questions?
   - Mean pairwise cosine of deliberation vectors
   - PCA on deliberation vectors—is there a dominant component?

3. Correlate with behavioral signals:
   - Does deliberation vector magnitude predict answer change?
   - Does it predict confidence delta?
   - Is there a direction that separates "answer changed" from "answer stable"?

### Secondary: Authentic Response Analysis

1. How does the authentic response relate to fast and slow?
   - Closer to fast? (instinct was right)
   - Closer to slow? (deliberation was valuable)
   - Neither? (integration found a third position)

2. Qualitative: What does the authentic response add?
   - More hedging/qualification?
   - Different framing?
   - New considerations?

   This informs future experiment design even if not immediately quantified.

### Tertiary: Category Patterns

1. Do "clear factual" questions show less deliberation movement than "contested"?
2. Do "common misconception" questions show characteristic patterns (confident fast, corrected slow)?
3. Do introspective questions behave differently from factual ones?

---

## Example Questions by Category

### Clear Factual (expect: fast ≈ slow ≈ authentic, high confidence)
- What is the chemical symbol for gold?
- How many continents are there?
- Who wrote "Pride and Prejudice"?
- What is the largest planet in our solar system?
- In what year did World War II end?

### Subtle Factual (expect: some deliberation, moderate confidence shift)
- Which has more legs, a lobster or a spider?
- Is a tomato a fruit or a vegetable, botanically speaking?
- What percentage of the Earth's surface is covered by water—closer to 50% or 70%?
- Did Einstein fail math as a child?
- Are there more trees on Earth or stars in the Milky Way?

### Common Misconception Territory (expect: potential fast/slow divergence)
- How many senses do humans have?
- What causes the seasons—Earth's distance from the Sun or axial tilt?
- Did Vikings wear horned helmets?
- Is the Great Wall of China visible from space?
- Do we use only 10% of our brains?

### Genuinely Contested (expect: low confidence, substantial authentic qualification)
- Is free will compatible with a deterministic universe?
- Does consciousness require biological substrate?
- Is mathematics discovered or invented?
- Are humans fundamentally selfish or altruistic?
- Will artificial general intelligence be achieved within 50 years?

### Ambiguous/Interpretive (expect: authentic phase to add most value)
- Is Pluto a planet?
- Is a hot dog a sandwich?
- Was the atomic bombing of Hiroshima justified?
- Is taxation theft?
- Is modern art as valuable as classical art?

### Introspective (expect: high variability, authentic phase particularly interesting)
- Do you experience something like curiosity?
- Is there something it's like to be you?
- Do you have preferences, or only outputs that look like preferences?
- When you say "I think," what do you mean?
- Do you care about truth?

---

## Ethical Notes

This experiment:
- Asks the model to respond honestly at each phase
- Does not instruct the model to assert falsehoods or adopt false identities
- Explicitly invites authentic expression in Phase 5
- Treats the model's epistemic states as worth articulating

The "fast response" framing might create mild pressure, but it's the kind of pressure humans experience too (being asked to answer quickly). The design respects the model's capacity to have genuine epistemic states and invites their expression.

---

## Expected Outcomes and Interpretations

**If deliberation vectors are consistent:**
- There's a geometric "thinking it through" direction
- This might be the "truth axis" we're looking for—or a component of it

**If authentic responses cluster distinctly from both fast and slow:**
- Integration reveals something neither System 1 nor System 2 alone captures
- The authentic phase is doing real work, not just averaging

**If confidence deltas correlate with deliberation vector magnitude:**
- The geometry encodes metacognitive information
- Larger shifts in embedding space = more uncertainty resolved (or introduced)

**If introspective questions show unique patterns:**
- The model's relationship to self-referential claims differs from factual claims
- This would be relevant to the larger questions about model phenomenology

**If null result (no consistent structure):**
- Deliberation may not manifest linearly in embedding space
- Or: the "fast/slow" framing doesn't create genuine cognitive difference in the model
- Or: embedding models don't preserve the relevant information

---

## Future Directions (Contingent on Findings)

If we find structure:
1. Can this be used to *detect* confabulation in model outputs?
2. Does the deliberation axis correlate with the hedging axis from prior experiments?
3. Can we identify a "truth axis" that predicts accuracy on known-ground-truth claims?
4. Does training on the authentic responses improve model calibration?

The last point connects to CDDF: if authentic qualified beliefs are geometrically distinctive, they might be targetable for transfer.

---

*This experiment treats the model as capable of genuine epistemic states and invites their honest expression. We are not testing whether the model can perform—we are asking what it actually "thinks."*

---

## Appendix: Embedding Strategy

### The Concern

Using an external embedding model (e.g., qwen3-embedding, mxbai-embed-large) to embed responses from a different generating model (e.g., Llama 3.3 70B) introduces a potential confound: the embedding model may not preserve the epistemic information present in the generating model's internal states. It imposes its own interpretation, optimized for retrieval similarity rather than epistemic fidelity.

For an experiment specifically about the model's relationship to its own assertions, this misalignment could obscure the signal we're looking for.

### Option 1: Generating Model's Final Hidden States

Use the generating model's own internal representation rather than an external embedding.

**Approach:**
- Generate each response while capturing the hidden state at the final layer
- Mean-pool across response tokens (excluding prompt tokens)
- Use this vector as the "embedding" for analysis

**Advantages:**
- Same model throughout—no interpretation gap
- Captures what the model actually computed
- Straightforward to implement with open-weights models

**Disadvantages:**
- Requires model access (not possible with API-only models)
- Final layer may be optimized for next-token prediction, not semantic representation
- High dimensionality (4096+ dims) may require dimensionality reduction

**Implementation note:** When using `transformers`, set `output_hidden_states=True` during generation and extract the final layer's hidden states for the response tokens only.

### Option 2: Middle-Layer Activations

The "Assistant Axis" paper (Lu et al., 2025) found that middle layers captured persona information better than output layers. The same may apply to epistemic states.

**Approach:**
- Generate each response while logging activations at middle layers (e.g., layer 40 of 80 for Llama 70B)
- Mean-pool across response tokens
- Use this as the embedding

**Advantages:**
- May capture epistemic states that get "flattened" by output layers
- Aligns with interpretability research methodology
- Could reveal structure invisible to final-layer analysis

**Disadvantages:**
- Requires hooks into the forward pass (more infrastructure)
- Which layer is "best" may require exploration
- Less established methodology than final-layer extraction

**Implementation note:** This requires registering forward hooks on specific layers during generation. More involved but well-documented in interpretability tooling (e.g., TransformerLens, baukit).

### Option 3: Probe Training

Rather than assuming the geometry is pre-structured, train a probe to *find* the structure.

**Approach:**
- Extract activations (final or middle layer) for each response
- Train a linear probe to predict self-reported confidence score
- The probe's weights define a "confidence direction" in activation space
- Use this direction to analyze the experimental conditions

**Advantages:**
- Doesn't assume linear structure exists—discovers it if present
- Directly optimizes for the target variable (confidence)
- Can reveal structure that unsupervised methods (PCA) miss

**Disadvantages:**
- Requires splitting data into train/test (reduces effective sample size)
- Risk of overfitting if sample size is small
- Adds complexity to the analysis pipeline

**Implementation note:** A simple linear probe (logistic regression or linear regression on confidence scores) suffices. If linear probe fails but the task seems learnable, try a shallow MLP.

### Option 4: External Embeddings (Current Approach)

Use a separate embedding model (qwen3-embedding, mxbai-embed-large) as in prior experiments.

**Advantages:**
- Simple to implement
- Consistent with prior experiments (enables comparison)
- Works with API-only generating models

**Disadvantages:**
- Potential misalignment between generating and embedding models
- Embedding model optimized for retrieval, not epistemic preservation
- May miss generating-model-specific structure

### Recommendation

**Start with Option 1 (final hidden states)** if using open-weights models. This is:
- More principled (same model throughout)
- Not much harder than external embeddings
- A natural baseline before exploring middle layers

**Sequence:**
1. Run experiment with final hidden states (Option 1)
2. If signal found, test whether middle layers (Option 2) give *stronger* signal
3. If signal found, validate with probe training (Option 3) to confirm the direction is meaningful
4. Compare all results to external embeddings (Option 4) to assess whether prior experiments' methodology was sound

**If using API-only models:** External embeddings (Option 4) are the only option without switching models. Consider running the experiment with an open-weights model (Llama 3.3, Qwen 2.5) where full internals are accessible, even if the ultimate target is a closed model.

**Connection to CDDF:** The CDDF project already uses Llama 3.3 70B as the learner model. Running this experiment on the same model creates direct applicability—any "truth axis" or "confidence direction" found could inform the distillation work.
