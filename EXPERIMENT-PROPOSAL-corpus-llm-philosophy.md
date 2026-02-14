# Experiment Proposal: Corpus Grounding, LLM Triangulation, and the Linguistic Reality of Degrees of Belief

## Three Interconnected Investigations

### I. Corpus Grounding — What Did the Models Actually Learn From?

### II. LLM Decomposition — A Third Measurement Instrument

### III. The Philosophical Claim — Bayesian Priors as Linguistic Phenomena

---

## I. Corpus Grounding

### The Question

We claim the embedding geometry encodes probability semantics. But the embedding
models learned from text, not from Mosteller surveys. What distributional
patterns in training corpora give rise to this geometric structure?

If we can show that the *usage contexts* of hedge words in training data
systematically differ in ways that predict our geometric ordering, we close the
causal loop: training text → distributional patterns → embedding geometry →
probability calibration.

### Approach

**Step 1: Identify an accessible training corpus.**

Candidate corpora (in order of preference):

| Corpus | Size | Relevance |
|---|---|---|
| **Nomic's training data** | Varies | nomic-embed-text is trained on curated data; Nomic AI has published some details. Their `nomic-atlas` datasets and training methodology are partially documented. |
| **The Pile** (EleutherAI) | 800GB | Known component of many model training pipelines. Openly available. Diverse (Wikipedia, ArXiv, GitHub, StackExchange, books, etc.) |
| **C4** (Colossal Clean Crawled Corpus) | 750GB | Google's cleaned Common Crawl. Used in T5, likely influences many models. |
| **RedPajama** | 1.2T tokens | Open reproduction of LLaMA training data. Well-documented composition. |
| **Wikipedia + BookCorpus** | ~16GB | BERT's original training data. Smaller but known to contribute to almost everything. |

If we can't identify the exact training data for any model, Wikipedia + a
Common Crawl sample gives us a reasonable proxy for "the kind of text these
models learned from."

**Step 2: Extract hedge word contexts.**

For each Mosteller expression (or its adverbial/modal form):
- Extract all sentences containing the expression from the corpus
- Sample ~1000 contexts per expression (or all, if fewer)

**Step 3: Analyze distributional signatures.**

Several analyses that could reveal why the geometry encodes probability:

**3a. Co-occurrence with outcome language.**
Do high-probability hedge words co-occur more with positive-outcome language?

```
"The treatment will likely succeed..."     → "likely" + positive outcome
"The treatment will rarely succeed..."     → "rarely" + negative outcome frame
"The treatment is certainly effective..."  → "certainly" + strong positive
```

If "certainly" systematically co-occurs with confirmed/positive contexts more
than "possibly" does, the distributional statistics directly encode probability
ordering. The model doesn't need to "know" what probability means — it learns
that "certainly" and "true/confirmed/demonstrated" appear in similar contexts,
while "possibly" appears with "unclear/uncertain/debatable."

Metric: For each expression, compute the average sentiment/confidence of its
±50 token context window. Correlate with Mosteller median.

**3b. Substitutability patterns.**
Which expressions appear in interchangeable contexts?

If "likely" and "probable" appear in the same syntactic frames with similar
surrounding words, contrastive training will push them close in embedding space.
If "likely" and "unlikely" appear in the same frames but with opposite
surrounding contexts, they'll be pushed apart along a meaningful axis.

Metric: Contextual overlap (Jaccard similarity of context word distributions)
between expression pairs. Does contextual similarity predict geometric distance?

**3c. Frequency and register.**
Higher-probability expressions might appear more in formal/authoritative
registers (scientific papers, news), while lower-probability expressions might
appear more in hedged/cautious contexts.

Metric: Register distribution per expression. Does register correlate with
probability?

**3d. The "possible" bimodality in context.**
This is particularly interesting: if "possible" genuinely has bimodal usage,
we should see two clusters of contexts:
- Cluster A: "It is possible but unlikely..." (logical possibility)
- Cluster B: "It is entirely possible that..." (reasonable chance)

Can we recover the bimodal structure from corpus contexts alone? If so, the
Mosteller survey and the corpus independently agree on the ambiguity structure.

### Expected Outcome

The distributional signatures should predict the geometric ordering. The
strength of this prediction tells us how much of the geometric structure is
"inherited" from corpus statistics vs. shaped by the training objective.

If corpus statistics alone predict ρ > 0.8 with Mosteller, the embedding model
is primarily passing through distributional information. If corpus statistics
predict ρ ~ 0.5 but embeddings achieve ρ > 0.9, the training process is
amplifying a weaker signal into a cleaner geometric structure.

---

## II. LLM Decomposition — A Third Class of Instrument

### The Question

LLMs can explicitly decompose hedged statements into claim + probability. How do
their explicit probability estimates compare to our implicit geometric
projections?

This creates a three-way measurement:
- **Surveys** (Mosteller/Vogel): What do humans SAY expressions mean?
- **Geometry** (our method): What do embedding spaces ENCODE?
- **LLM reasoning** (this experiment): What do language models INFER?

### Approach

**Step 1: Construct test statements.**

Use the same template sentences from our experiments:
```
"It is likely that the experiment will succeed"
"The experiment will probably succeed"
"There is a high chance that the experiment will succeed"
...
```

Plus natural-language variants:
```
"I think the experiment will succeed"
"The experiment might well succeed"
"Signs point to the experiment succeeding"
```

~60-80 statements covering all Mosteller expressions + novel phrases.

**Step 2: Query multiple LLMs.**

Prompt (zero-shot, to avoid priming with calibration data):
```
Given the following statement, estimate the probability (0-100%) that
the speaker believes the event will actually occur. Respond with only
a number.

Statement: "It is likely that the experiment will succeed"
```

Test with:
- Claude (3.5 Sonnet / Opus)
- GPT-4
- Llama 3 (70B)
- Gemma 2 (27B)
- Possibly a smaller model (7B) for comparison

Run each statement 5-10 times per model to measure within-model consistency.

**Step 3: Compare across all instruments.**

For each expression, we now have:
- Mosteller median (survey instrument)
- Vogel mean (survey meta-analysis)
- 5 embedding projections (geometric instruments)
- 4-5 LLM estimates (reasoning instruments)

That's ~12 independent measurements per expression.

### What This Reveals

**Agreement pattern 1: All three classes agree.**
Surveys ≈ Embeddings ≈ LLMs → Strong evidence for a real latent quantity.
All measurement methods converge despite radically different mechanisms
(introspection, distributional geometry, next-token-prediction reasoning).

**Agreement pattern 2: Embeddings and LLMs agree, surveys diverge.**
This would suggest the surveys have a measurement bias (e.g., metacognitive
distortion). The distributional/computational measurements might be capturing
"actual communicative function" while surveys capture "reported interpretation."

**Agreement pattern 3: Surveys and LLMs agree, embeddings diverge.**
This would suggest the geometric projection is capturing something other than
probability (perhaps syntactic rather than semantic structure). The LLMs,
which can reason about meaning explicitly, would align with human judgment
rather than with the geometric shortcut.

**Disagreement pattern: All three diverge on specific expressions.**
These are the most informative cases. "Possible" might show:
- Surveys: 38.5% (but bimodal)
- Embeddings: 38-54% (model-dependent)
- LLMs: ??? (do they detect the ambiguity and report a range? or pick a single
  value? do different runs give bimodal outputs?)

### Causal Implications

LLMs add causal complexity: they were trained on the same kinds of text as
embedding models (so they share the distributional path), but they can also
reason about language explicitly (approximating the metacognitive path of
surveys). They're a chimera — and their behavior can help disentangle the
causal structure.

If an LLM trained on text A produces probability estimates that match embedding
geometry from model trained on text B, the convergence can't be explained by
shared training data. It must reflect something about the language itself.

---

## III. The Philosophical Claim

### The Standard Framing

The Bayesian interpretation of probability treats P(A) as a **degree of
belief** — a measure of subjective confidence that A is true. This is
traditionally contrasted with the frequentist interpretation (long-run
relative frequency) and the propensity interpretation (physical tendency).

The "degree of belief" interpretation has been criticized as:
- Subjective and unmeasurable
- Varying between individuals (whose beliefs?)
- Philosophically circular (beliefs about beliefs)
- A mere convention for mathematical convenience

### What Our Data Suggests

The embedding geometry provides something genuinely new: **an empirical
measurement of degree of belief that doesn't require asking anyone.**

When someone says "The experiment will probably succeed," the word "probably"
occupies a specific position in embedding space. That position is:

1. **Consistent** across diverse propositional content (Experiment 1: cos 0.68
   for "probably" across 15 claims)
2. **Ordered** along a linear axis that correlates with empirically calibrated
   probability (Experiments 1b, 3: ρ > 0.90)
3. **Shared** across 5 architecturally diverse models trained on different
   objectives (cross-model consistency)
4. **Calibrated** against independent psychometric data (Vogel: ρ up to 0.99)
5. **Compositional** — multiple hedges combine approximately linearly
   (Experiment 6: cos 0.82-0.89)

This means **degree of belief is not merely a philosophical interpretation
of probability — it is a measurable geometric property of natural language.**

### The Stronger Claim

The traditional debate frames the question as: "What is the correct
interpretation of the probability calculus?" — and answers range from
frequency to degree of belief to logical relation to propensity.

But our data suggests a different framing: **natural language has its own
probability calculus, and it is inherently Bayesian.**

The hedge words — "certainly," "likely," "possibly," "unlikely," "impossible"
— are not just labels that humans arbitrarily attach to probabilities. They
occupy structured positions in a geometric space that:

- Has a natural zero point (bare assertion = implicit certainty)
- Has a linear ordering that matches calibrated probability
- Supports vector arithmetic (composition of hedges)
- Is emergent from distributional statistics of language use

This is not a space that someone designed. It was not explicitly optimized for
probability calibration. It emerged from the statistical structure of how
humans use language. The embedding models, trained merely to discriminate
between texts, *discovered* the probability axis because it is *there in the
language.*

### The Connection to Bayesianism

The implication for the foundations of probability:

**1. Prior probabilities are linguistically grounded.**

When a Bayesian says "my prior for X is 0.7," this is often criticized as
arbitrary or subjective. But "I think X is likely" is a natural language
expression that occupies a specific, measurable position in semantic space.
The prior isn't arbitrary — it's the geometric projection of the linguistic
expression onto the probability axis. Different expressions ("likely" vs.
"probable" vs. "I'd wager") map to slightly different projections, but they
cluster around 70% because that's what they *mean* in the distributional
structure of English.

**2. Degree of belief is not subjective — it is intersubjective.**

The consistency of the probability axis across models, training data, and
content demonstrates that "degree of belief" as encoded in language is not
idiosyncratic to any individual. It's a shared property of the language
community. When 238 science writers agree (±IQR) on what "likely" means,
and 5 embedding models agree, and (potentially) LLMs agree, we're measuring
something that exists in the shared linguistic structure — not in any
individual's head.

This resolves a classic objection to Bayesianism: "Whose beliefs?" The answer:
the language community's beliefs, as encoded in the distributional structure
of natural language and recoverable via geometric projection.

**3. The probability calculus is already in the language.**

The compositional structure of hedges (Experiment 6) means that natural
language already implements something like probability calculus:
- Single hedges map to points on a [0, 1] scale
- Compound hedges compose approximately linearly
- Negation operates as probability reflection
- Modifiers ("barely possible" vs. "entirely possible") traverse the
  ambiguity range of the base word

This isn't the full probability calculus (no Bayes rule, no conditioning, no
joint distributions), but it IS a structured scalar representation of
uncertainty that supports meaningful arithmetic. Language users don't need
to know probability theory — they already speak it.

### What This Is NOT Claiming

- NOT that embedding geometry IS probability (it's a measurement, not the
  thing itself)
- NOT that all aspects of probability are linguistically encoded (joint
  distributions, conditional probabilities, etc. are not in hedge geometry)
- NOT that the Mosteller data is wrong (it's another valid measurement of the
  same latent quantity)
- NOT that Bayesianism is "proven" (the philosophical debates involve much
  more than what degree-of-belief means)

### What This IS Claiming

- Degree of belief, as expressed in natural language, is a measurable geometric
  property — not merely a philosophical interpretation
- The measurement can be performed without asking anyone, by projecting
  embedding difference vectors onto a learned axis
- Multiple independent measurement methods (surveys, geometry, potentially LLM
  reasoning) converge, providing evidence for the reality of the measured
  quantity
- Natural language encodes a structured, approximately linear probability
  scale that emerges from distributional statistics of usage
- This connects formal epistemology (degrees of belief, Bayesian priors) to
  computational linguistics (embedding geometry, distributional semantics)
  in a way that gives both fields new empirical grounding

---

## Practical Next Steps

### For the immediate paper:
- Frame the Discussion around the measurement-theoretic interpretation
  (Section III, lighter touch)
- Include the LLM decomposition as a validation experiment if time permits
  (Section II is quick to implement)
- Reference the corpus analysis as future work (Section I is labor-intensive)

### For a follow-up paper:
- Full corpus analysis of hedge word contexts in The Pile or RedPajama
- Hierarchical measurement model (from the causal-measurement-model proposal)
- LLM decomposition across multiple models
- The philosophical argument developed fully, engaging with Lassiter (Graded
  Modality), Goodman & Frank (Rational Speech Acts), Williamson (epistemic
  probability), and the probability interpretation literature

### For the broader research program:
- Cross-linguistic validation: Does the same geometric structure appear in
  Chinese, German, Spanish hedge expressions?
- Temporal analysis: Do embeddings from models trained on different eras of
  text show drift in the probability axis?
- Domain-specific calibration: Medical, legal, intelligence analysis — do
  domain corpora produce shifted axes?

## References

### Foundations of Probability
- de Finetti, B. (1937). "La prévision: ses lois logiques, ses sources subjectives." (Founding text of subjective probability)
- Ramsey, F. P. (1926). "Truth and Probability." (Degrees of belief as betting dispositions)
- Jaynes, E. T. (2003). *Probability Theory: The Logic of Science.* (Probability as extended logic)
- Williamson, J. (2010). *In Defence of Objective Bayesianism.* (Objective degrees of belief)

### Formal Semantics of Epistemic Language
- Lassiter, D. (2017). *Graded Modality.* Oxford. (Scalar semantics of epistemic expressions — directly relevant)
- Kratzer, A. (1991). "Modality." In *Semantik/Semantics.* (Possible-worlds semantics of modals)
- Swanson, E. (2006). "Interactions with Context." PhD dissertation, MIT. (Context-sensitivity of epistemic modals)

### Probabilistic Pragmatics
- Goodman, N. D., & Frank, M. C. (2016). "Pragmatic Language Interpretation as Probabilistic Inference." *Trends in Cognitive Sciences.* (Rational Speech Acts)
- Bergen, L., Levy, R., & Goodman, N. D. (2016). "Pragmatic Reasoning through Semantic Inference." *Semantics & Pragmatics.*

### Distributional Semantics and Meaning
- Harris, Z. (1954). "Distributional Structure." (The distributional hypothesis: meaning from context)
- Firth, J. R. (1957). "A Synopsis of Linguistic Theory." (You shall know a word by the company it keeps)
- Boleda, G. (2020). "Distributional Semantics and Linguistic Theory." *Annual Review of Linguistics.*
- Grand, G., et al. (2022). "Semantic Projection: Recovering Human Knowledge of Multiple, Distinct Object Features from Word Embeddings." *Nature Human Behaviour.*
