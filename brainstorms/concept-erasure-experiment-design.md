# Concept Erasure Experiment — Design Brainstorm

> **⚠️ THIS IS A BRAINSTORM, NOT A DECISION.**
>
> Drafted by Claude on 2026-05-01 in response to "what would the sentence-embedding-native intervention analogue actually look like." This is a design sketch with rough effort estimates and known unknowns — not a protocol you can run as-is. Joseph should expect to adjust the model choice, the validation tasks, the sample sizes, and possibly reject the approach entirely in favor of one of the other intervention analogues (composition, cross-model behavioral). The goal here is to put a concrete enough design on paper that the merits can be evaluated.

---

## Why this experiment

The (d1) framing risk identified in `tactical-notes.md` §10b: a TACL reviewer who knows Ji et al. (2025) or Valentin et al. (2025) may argue that the embeddings paper currently demonstrates *geometric correlation* between an extracted axis and human probability data, but does not demonstrate that the axis is *functional* — that it actually carries the probability information for downstream computation rather than being an epiphenomenon of more general lexical similarity.

The cleanest sentence-embedding-native counter is **linear concept erasure**: project the probability axis OUT of sentence embeddings and show that downstream tasks dependent on probability content degrade in a way that random-direction erasure does not.

If the axis is functional → erasure degrades probability-related tasks while preserving unrelated tasks. If the axis is just a correlate of lexical similarity → erasure has either no effect or symmetric effects across tasks.

This is one of three plausible analogues; the other two (composition; cross-model behavioral validation) are sketched briefly at the end.

---

## Conceptual design

### The basic procedure

For sentence embedding **e** ∈ ℝ^d and supervised probability axis **v** ∈ ℝ^d (unit-normalized):

```
e_erased = e - (e · v) v
```

This is rank-1 projection onto the orthogonal complement of **v**. After erasure, no linear classifier with weight vector parallel to **v** can recover information about probability from `e_erased`.

There's a more rigorous version using LEACE (Belrose et al., 2023) which is the closed-form least-squares-optimal linear concept erasure. For the simple supervised case this gives results essentially equivalent to the rank-1 projection above, but with a principled framework citation. **Recommend using LEACE if there's bandwidth, rank-1 projection if not.** Both are defensible; LEACE will be the one Marks-Tegmark / Bürger reviewers expect.

### What you erase

The supervised within-type predicative probability axis from the strongest model (mxbai-embed-large, ρ = 0.991 on Vogel) is the natural choice — it's the headline axis the paper is built around. Run the experiment on at least one other model (qwen3-embedding for architectural diversity) to show the result generalizes.

Alternatively erase from each of the four within-type axes (predicative, adverbial, noun-phrase, modal). The 4D epistemic subspace formulation in FINDINGS-03 supports this.

### Validation tasks (in increasing order of design effort)

These are candidate downstream tasks where probability content should matter. Pick ONE that's robust and clearly interpretable; multiple is better but only if they tell the same story.

#### Task 1 — Held-out probability prediction (cheapest)

- **Setup:** Train a linear regressor on `(embedding, mosteller_median)` pairs for, say, the adverbial axis. Predict probabilities for held-out predicative phrases (out-of-syntactic-type generalization). Measure ρ.
- **Erasure manipulation:** Repeat with embeddings that have had the *predicative* probability axis erased. The regressor should now do worse, because the linear probability information has been removed from those embeddings.
- **Random-direction control:** Repeat erasure along 100 random unit directions (matched norm). Average degradation should be much smaller than degradation from probability-axis erasure.
- **Caveat:** This is somewhat circular — you're erasing the axis the regressor uses. The cleaner version uses one axis to erase and a different axis to predict. The within-type ↔ across-type generalization (FINDINGS-03 shows axes are 0.25–0.81 cosine-correlated) gives the right structure: erase predicative, evaluate on modal regression, see degradation predicted by inter-axis cosine.
- **Effort:** ~1 day. Already-existing data, 50–100 lines of new code.
- **Verdict:** Useful baseline but vulnerable to "you erased the thing then showed it can't be recovered."

#### Task 2 — Cross-model behavioral prediction (medium)

- **Setup:** Take a generative LLM (e.g., GPT-4o, Claude 3.5, or a local open model) and prompt it with: "On a scale of 0–100, what is the probability that {claim with hedge}?" for the Mosteller phrases. Record the LLM's stated probabilities (these will be different from Mosteller medians but should correlate; this is an *independent* probability source).
- Test whether the supervised projection of the bare embedding-model representation predicts the *generative-LLM-stated* probability. Measure ρ.
- **Erasure manipulation:** Repeat with erased embeddings. The projection now contains no probability information.
- **Random-direction control:** Same as Task 1.
- **Why this is stronger:** The generative-LLM probabilities are an independent behavioral measure. Even if the supervised axis is somehow circular w.r.t. Mosteller (it's regressed against Mosteller after all), it shouldn't be circular w.r.t. an entirely separate model's stated probabilities.
- **Caveat:** Adds an external dependency (a generative LLM and its idiosyncrasies). The story is "axis predicts independent generative behavior" rather than "axis is functional in this embedding's downstream tasks."
- **Effort:** ~2–3 days. Includes prompt design, robustness checking, possibly model comparison across 2–3 generative LLMs.

#### Task 3 — Retrieval against probability-graded references (more involved)

- **Setup:** Construct or curate a small reference corpus where each entry has a probability annotation. E.g., 200 sentences from scientific writing or weather forecasts where probability is stated linguistically AND numerically. Embed all. For a query like "It is likely that X," check whether the top retrieved references have probabilities clustered around the Mosteller median for "likely" (~71%).
- **Erasure manipulation:** Re-run retrieval with axis-erased query embeddings. Top-retrieved references should no longer cluster around the query's hedged probability.
- **Random-direction control:** Same as before.
- **Why this is strongest:** Pure behavioral / task-level validation. Doesn't depend on regressors, just on the structure of nearest-neighbor space.
- **Caveat:** Requires a reference corpus with paired linguistic-and-numerical probability that doesn't exist off-the-shelf. ~50–200 examples curated by hand from scientific abstracts or IPCC reports (which use both verbal and numerical probability) would work.
- **Effort:** ~3–5 days, dominated by corpus curation.

### Recommended path for TACL-by-June-1

**Task 1 (probability prediction across syntactic types) plus the random-direction control** is the minimum-viable experiment. ~1 day of work. Paragraph in Results, one figure showing degradation curve (probability axis erasure vs. random direction erasure). Defangs the (d1) framing risk without committing to a major experimental program.

**Task 2** if there's an extra 2–3 days of bandwidth. Stronger story but adds external dependencies (LLM access, prompt engineering decisions).

**Task 3** is for the journal version after acceptance, or for a follow-up paper.

---

## A skeleton results paragraph (for the paper)

To verify that the extracted probability axis is functional rather than merely correlated with embedding geometry, we evaluate whether projecting the axis out of sentence embeddings degrades downstream probability prediction. Following the linear concept erasure framework (Belrose et al., 2023), we compute the orthogonal-complement projection `e_erased = e - (e·v)v` for each sentence embedding **e**, where **v** is the unit-normalized supervised probability axis from one syntactic type. We then evaluate cross-type probability prediction: a linear regressor trained on adverbial-type embeddings predicts probabilities for held-out modal-type embeddings (cosine alignment between predicative and modal axes: 0.62 for mxbai-embed-large). Erasing the predicative axis from the modal embeddings degrades held-out ρ from {baseline} to {erased}, while erasing along 100 matched random directions degrades ρ from {baseline} to {random_baseline} on average (95% CI: [...]). The observed degradation is X.X standard deviations below the random-direction null distribution (p < 0.001). This indicates the axis carries probability information used by downstream linear decoders, not just a coincidence of geometric correlation.

---

## Design questions Joseph needs to decide

1. **Rank-1 projection or LEACE?** LEACE is more rigorous and the citation is small. Rank-1 projection is fine if bandwidth is tight. The empirical results should be very similar.

2. **Which model?** mxbai-embed-large is the strongest empirical baseline. qwen3-embedding adds architectural diversity (decoder-derived embedding head). Two models is the right number — one shows the effect, two shows it generalizes, three is overkill for one paragraph in Results.

3. **Which axis to erase, which to predict?** Cross-type erasure (erase axis A, predict axis B) is cleaner than within-type (erase predicative, predict predicative — circular). The 4-axis structure in FINDINGS-03 supports cross-type design naturally.

4. **Random-direction baseline N.** 100 is standard; 1000 is overkill but cheap; 30 is the floor. 100 gives clean error bars.

5. **Where does this go in the paper?** New §3.X subsection, "Functional Validation via Concept Erasure," ~1 page including one figure. Place after the cross-validation results but before cross-linguistic.

6. **Does this conflict with the model-class novelty framing?** Slightly. The framing "sentence embedding models don't have token-generation pathways for steering" is somewhat undercut by introducing an erasure-based intervention. The honest reconciliation: erasure is *intervention on the representation* (which sentence embedding models do have), distinct from *steering token output* (which they don't). Word the framing carefully so the two claims don't appear contradictory.

---

## Alternative interventions (briefly, for completeness)

### Composition

Take a *neutral* claim embedding (e.g., the bare-template embedding "The experiment will succeed") and add the probability-axis projection scaled by a target probability. Show that the resulting synthetic embedding behaves in retrieval like real hedged sentences with the corresponding probability. This is the "vector arithmetic for probability" story — *bare + 0.7·v ≈ "It is likely that the experiment will succeed"* in retrieval space.

- **Effort:** Similar to Task 1 (~1 day).
- **Pro:** Rhetorically stronger than erasure. Composition is a constructive demonstration; erasure is a destructive one.
- **Con:** Mosteller-style template construction matters more here. The synthetic embedding is closest to a pure word-substitution effect, which is exactly what (d2) "template-only evaluation" critique worries about.

### Cross-model behavioral validation

Skip the in-model intervention entirely. Use the supervised projection as input to predict an *independent* generative LLM's stated probability for the same claim. Strong story, but adds external dependencies. Already covered as Task 2 above.

### Activation patching (ruled out)

Sentence embedding models don't have a clean residual-stream / activation-cache abstraction the way decoder LLMs do. Activation-patching-style intervention (Geiger 2021, Zhang & Nanda 2024) is not directly applicable. This is a real limit of the model class and should be acknowledged in Limitations regardless of which intervention path is taken.

---

## What this experiment does NOT do

- Does not establish causation in the strong sense (manipulating the axis causes a specific downstream output change in a generative pipeline). That requires a token-generation pathway the embedding models don't have.
- Does not match the rhetorical power of Valentin's bias-exposure or jailbreak demonstrations. Those rely on the steering tools natural to decoder LLMs.
- Does not extend to non-template natural text. The axis was extracted on templates; the intervention validates on templates. Generalization to natural text is a separate question (FINDINGS-02 partially addresses this for the non-erased case).

The honest framing: "we demonstrate the axis is functional in the linear-decoder sense, not the generative-steering sense; full causal demonstration in token-generation pipelines is reserved for follow-up work that bridges sentence-embedding probes to decoder-LLM behavior."

---

## References that should be cited if this experiment goes in the paper

- Belrose, N., Schneider-Joseph, D., Ravfogel, S., Cotterell, R., Raff, E., & Biderman, S. (2023). Leace: Perfect linear concept erasure in closed form. NeurIPS 2023. *(LEACE method citation.)*
- Ravfogel, S., Elazar, Y., Gonen, H., Twiton, M., & Goldberg, Y. (2020). Null it out: Guarding protected attributes by iterative nullspace projection (INLP). ACL 2020. *(Earlier null-space erasure work.)*
- Geiger, A., Lu, H., Icard, T., & Potts, C. (2021). Causal abstractions of neural networks. NeurIPS. *(For the broader causal-intervention framing in Limitations.)*
- Zhang, F., & Nanda, N. (2024). Towards best practices of activation patching in language models: Metrics and methods. ICLR. *(For the activation-patching-not-applicable note.)*
