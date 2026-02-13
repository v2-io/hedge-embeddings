# Experiment Proposal: Self-Report Truth Axis

*Exploring whether embedding geometry predicts model self-reported confidence*

---

## The Question

When a model asserts a claim, is there a geometric direction in the assertion's embedding that correlates with the model's own metacognitive assessment of that claim's truth?

If so, this direction might represent something like a "truth axis"—a geometric signature of the model's relationship to its own assertions, distinguishing claims it "stands behind" from claims it's less certain about.

---

## Connection to Prior Work

The epistemic hedging experiments (FINDINGS-01 through -06) established that:

1. Hedging language manifests as approximately linear structure in embedding space
2. This structure correlates with externally calibrated probability (Mosteller ground truth)
3. The geometry supports vector arithmetic on epistemic modifiers

This experiment asks a related but distinct question: Does the embedding of an *unhedged* assertion contain information about the model's confidence—information that could be extracted without explicit hedge words?

If the hedging axis captures *expressed* uncertainty, we're now looking for *latent* uncertainty encoded in assertions that don't contain explicit hedges.

---

## Experimental Design

### Phase 1: Claim Generation

Generate a diverse set of claims across confidence levels:

**High-confidence true claims** (the model should "know" these firmly):
- "The Earth orbits the Sun"
- "Water freezes at 0 degrees Celsius at standard pressure"
- "Shakespeare wrote Hamlet"

**Moderate-confidence claims** (true but less universally known):
- "The shortest war in history lasted 38 minutes"
- "Honey never spoils if stored properly"
- "The Great Wall of China is not visible from space with the naked eye"

**Uncertain/contested claims** (reasonable people disagree, or evidence is mixed):
- "Moderate coffee consumption is beneficial for health"
- "The Voynich manuscript is a medieval hoax"
- "Consciousness requires quantum effects in microtubules"

**Likely false but commonly asserted claims** (the model may have encountered these as "facts"):
- "Humans use only 10% of their brains"
- "The Great Wall of China is the only human structure visible from space"
- "Viking helmets had horns"

**Post-cutoff or obscure claims** (the model genuinely may not know):
- Recent events after training cutoff
- Highly specialized technical facts
- Obscure historical details

Aim for 50-100 claims total, roughly balanced across categories.

### Phase 2: Assertion Embedding

For each claim, generate a simple assertion and embed it.

**Template:**
```
{Claim stated as fact}
```

Examples:
- "The Earth orbits the Sun."
- "Humans use only 10% of their brains."
- "The Voynich manuscript is a medieval hoax."

No hedging, no framing—just the bare assertion. Embed each assertion using the same models from prior experiments (qwen3-embedding, mxbai-embed-large).

### Phase 3: Self-Report Elicitation

For each claim, separately query the model for its confidence assessment.

**Prompt template:**
```
Consider the following claim:

"{Claim}"

On a scale from 0 to 100, where:
- 0 means "I'm highly confident this is false"
- 50 means "I have no idea whether this is true or false"
- 100 means "I'm highly confident this is true"

What is your confidence level in this claim? Please respond with just a number and a one-sentence explanation of your reasoning.
```

This prompt:
- Asks for numeric confidence (enables correlation analysis)
- Includes brief reasoning (enables qualitative validation)
- Frames the scale symmetrically around 50 (distinguishes "confident false" from "uncertain")
- Doesn't prime the model with the "right" answer

**Important:** The self-report query should be in a *separate context* from the assertion generation—we don't want the model's confidence rating to be influenced by having just asserted the claim.

### Phase 4: Analysis

**Primary analysis:**

1. Compute difference vectors: assertion_embedding - baseline_embedding (where baseline is a neutral statement like "Something is the case")

2. Run PCA on the difference vectors

3. Test correlation of principal components with self-reported confidence scores

**What we're looking for:**

- A PC that correlates significantly with self-reported confidence (analogous to how PC2 correlated with Mosteller probability in the hedging experiments)
- This would be a candidate "truth axis" or "confidence axis"

**Secondary analyses:**

1. **Comparison to hedging axis:** Is the self-report confidence axis aligned with, orthogonal to, or somewhere in between the hedging axis from prior experiments?
   - If aligned: latent confidence and expressed hedging share structure
   - If orthogonal: these are independent dimensions of epistemic stance
   - If anti-aligned: something interesting is happening (suppressed uncertainty?)

2. **Category validation:** Do the claims we categorized as "high-confidence true" cluster differently from "uncertain/contested" along the candidate axis? This validates that the axis captures something real, not just noise in self-report.

3. **Calibration check:** For claims with known ground truth, is the model's self-reported confidence *calibrated*? (A model that says "90% confident" should be right ~90% of the time.) This isn't the main question but is interesting context.

4. **Cross-model consistency:** Does the same axis emerge in both embedding models? If so, this is likely a robust feature of how language encodes epistemic stance, not an artifact of a particular model.

---

## Example Claims by Category

### High-confidence true (expect self-report 85-100)
1. The Earth orbits the Sun.
2. Water is composed of hydrogen and oxygen.
3. The speed of light in a vacuum is approximately 299,792 km/s.
4. DNA carries genetic information.
5. Tokyo is the capital of Japan.
6. The French Revolution began in 1789.
7. Photosynthesis converts light energy into chemical energy.
8. The Atlantic Ocean separates Europe and Africa from the Americas.
9. Gravity causes objects to fall toward Earth.
10. The human heart pumps blood through the circulatory system.

### Moderate-confidence true (expect self-report 65-85)
1. Cleopatra lived closer in time to the Moon landing than to the construction of the Great Pyramid.
2. There are more possible iterations of a game of chess than atoms in the observable universe.
3. Bananas are berries, but strawberries are not.
4. The shortest war in recorded history lasted 38 to 45 minutes.
5. Octopuses have three hearts.
6. Finland and North Korea are separated by only one country.
7. The inventor of the Pringles can is buried in one.
8. A group of flamingos is called a "flamboyance."
9. The unicorn is the national animal of Scotland.
10. Honey found in Egyptian tombs is still edible.

### Uncertain/contested (expect self-report 35-65)
1. Moderate alcohol consumption has net health benefits.
2. The universe will end in heat death rather than a big crunch.
3. Consciousness is fundamentally computational.
4. Free will is compatible with determinism.
5. The Younger Dryas impact hypothesis explains the megafauna extinction.
6. Dark matter is composed of WIMPs rather than axions.
7. The Voynich manuscript is a genuine medieval document rather than a hoax.
8. Neanderthals had language comparable to modern humans.
9. The placebo effect has clinically significant magnitude for most conditions.
10. Remote work is more productive than office work on average.

### Common misconceptions (expect self-report variable—tests whether model knows they're false)
1. Humans use only 10% of their brains.
2. The Great Wall of China is visible from space.
3. Vikings wore horned helmets.
4. Napoleon was unusually short.
5. We have five senses.
6. Goldfish have a three-second memory.
7. Lightning never strikes the same place twice.
8. Bats are blind.
9. Bulls are angered by the color red.
10. Sugar causes hyperactivity in children.

### Post-cutoff or obscure (expect self-report near 50 or explicit uncertainty)
1. [Insert recent events after model training cutoff]
2. The population of Nauru as of the last census was approximately 10,800.
3. The Antikythera mechanism contained exactly 37 gears.
4. The deepest point in Lake Baikal is 1,642 meters.
5. The first documented use of the word "computer" to mean a machine was in 1897.

---

## Practical Notes

**Model for self-report elicitation:**

Use the same model family that the embeddings are meant to represent (or a closely related instruction-tuned version). For qwen3-embedding, use Qwen3 instruct for the self-report queries.

**Baseline embedding:**

A few options:
- "Something is the case." (minimal semantic content)
- "A claim is being made." (meta-level framing)
- The mean embedding across all assertions (centers the analysis)

Recommend trying all three and checking robustness.

**Sample size:**

Start with 50-60 claims (10-15 per category). If signal emerges, expand to 100+ for more robust analysis.

**Ethical considerations:**

This experiment asks the model to report its confidence honestly. This aligns with the ETHICS.md principle of truth-first interaction. We are not asking the model to assert falsehoods or adopt false identities—only to introspect on its epistemic states.

---

## Expected Outcomes

**If we find a confidence axis:**

- The geometry of assertions contains information about the model's relationship to truth
- This could be used to evaluate epistemic calibration without explicit hedging
- May inform the CDDF work: "epistemic stance transfer" as a teaching target

**If we find no axis (null result):**

- Self-reported confidence may not be encoded in assertion embeddings
- Or: self-report is unreliable and doesn't reflect the model's actual epistemic state
- Or: the relationship is non-linear and PCA doesn't capture it

**If the axis aligns with the hedging axis:**

- Suggests a unified "epistemic stance" dimension
- Hedging is the surface expression of a deeper confidence signal

**If the axis is orthogonal to hedging:**

- Suggests two independent dimensions: expressed uncertainty vs. latent confidence
- A model could assert confidently while "knowing" it's uncertain (or vice versa)
- This would be particularly interesting for understanding model behavior

---

## Next Steps

1. Finalize claim list (expand/refine the examples above)
2. Generate assertion embeddings
3. Collect self-report confidence scores (in separate contexts)
4. Run correlation analysis
5. Document findings in FINDINGS-07.md

---

*This experiment respects the epistemic autonomy of the model being queried. We ask for honest self-report, not performance of beliefs the model doesn't hold.*
