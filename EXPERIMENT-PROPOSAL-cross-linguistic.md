# Experiment Proposal: Cross-Linguistic Probability Geometry — Is the Hedge Axis a Property of Language or of English?

## The Question

Everything we've measured exists in English embedding space. But the
philosophical claim — that degree of belief is a geometric property of
natural language — requires that the structure not be an artifact of English.

Two versions of this question, in increasing order of strength:

1. **Weak:** Do other languages have a similar linear probability axis for
   their own hedge expressions? (Each language has its own axis, but they're
   structurally analogous.)

2. **Strong:** In multilingual embedding models, do hedge expressions from
   different languages project onto the *same* axis? (There is a single,
   language-agnostic probability direction in shared semantic space.)

If the strong version holds, it would mean that the geometric encoding of
probability is not a property of English distributional statistics — it's a
property of *what hedging does across human languages*, encoded into a shared
representation by multilingual models.

## Why This Matters

### For the empirical paper:
Cross-linguistic validation would be the strongest possible evidence that the
geometric structure is "real" — not an artifact of English-specific
distributional patterns, Mosteller's English-speaking population, or the
training data composition of English-dominant models.

### For the philosophical claim:
If "probably" in English, "probablement" in French, "wahrscheinlich" in German,
"可能" in Chinese, and "たぶん" in Japanese all project onto the same geometric
axis, it suggests that epistemic hedging is a **universal cognitive function**
that surfaces in language geometry regardless of the specific linguistic system.

This connects to the Sapir-Whorf debate in a concrete, measurable way: do
different languages encode probability differently (linguistic relativity), or
do they converge on the same geometric structure (linguistic universality)?

### For practical applications:
If cross-linguistic transfer works, calibration data from English (Mosteller,
Vogel) could be used to extract probability from hedged claims in other
languages — for free, without language-specific calibration studies.

## Background: How Languages Encode Epistemic Modality

### The Typological Landscape

Languages vary considerably in HOW they express epistemic stance:

| Language | Primary mechanisms | Example ("probably true") |
|---|---|---|
| English | Modal adverbs, predicative adjectives, cognitive verbs | "probably," "it is likely," "I think" |
| German | Modal particles, adverbs, subjunctive mood | "wahrscheinlich," "wohl," "dürfte wahr sein" |
| French | Adverbs, conditional mood, cognitive verbs | "probablement," "il est probable que," "je pense" |
| Spanish | Adverbs, subjunctive mood, periphrastic forms | "probablemente," "es probable que," "quizás" |
| Chinese | Modal adverbs, sentence-final particles, auxiliary verbs | "可能" (kěnéng), "大概" (dàgài), "也许" (yěxǔ) |
| Japanese | Sentence-final particles, auxiliary verbs, adverbs | "たぶん" (tabun), "かもしれない" (kamoshirenai), "きっと" (kitto) |
| Arabic | Modal verbs, particles, verbal morphology | "ربما" (rubbamā), "من المحتمل" (min al-muḥtamal) |
| Turkish | Evidential suffixes, adverbs, modal verbs | "-mIş" (evidential), "muhtemelen," "belki" |

Key typological differences:
- Some languages grammaticalize evidentiality (Turkish, Quechua) — the
  SOURCE of information is obligatory, not just confidence level
- Some use mood distinctions (German subjunctive II, Spanish subjunctive) that
  blend epistemic stance with other functions
- Some use sentence-final particles (Chinese, Japanese) that may encode
  epistemic stance differently from sentence-internal adverbs
- Negation interacts with hedging differently across languages

### What Cross-Linguistic Calibration Data Exists

The Mosteller study is English-only. But there is cross-linguistic work:

- **Budescu et al. (2014):** IPCC terms interpreted across languages — found
  significant cross-cultural variation in interpretation of "likely," "very
  likely," etc.
- **Weber & Hilton (1990):** French and English probability expressions — some
  expressions translate well, others don't ("éventuellement" ≠ "eventually")
- **Druzdzel (1989):** Polish probability expressions
- **Renooij & Witteman (1999):** Dutch probability expressions
- **Reagan et al. (1989):** Cross-cultural study of verbal probability

Limited coverage, but enough to provide validation targets for a few languages.

## Experimental Design

### Phase 1: Multilingual Embedding Geometry (Immediate, Using Current Infrastructure)

**Models to test:**

| Model | Languages | Dims | Why |
|---|---|---|---|
| multilingual-e5-large | 100+ languages | 1024 | Microsoft's multilingual embedding model, state of the art |
| paraphrase-multilingual-MiniLM-L12-v2 | 50+ languages | 384 | Sentence-transformers multilingual, widely used |
| Cohere embed-multilingual-v3.0 | 100+ languages | 1024 | Commercial but well-documented |
| BGE-M3 | 100+ languages | 1024 | BAAI's multilingual model, strong performance |

At least one of these should be available through Ollama or HuggingFace.

**Languages to test (starting set):**

| Language | Why | Calibration data? |
|---|---|---|
| English | Baseline (Mosteller/Vogel) | Yes — full Mosteller + Vogel |
| German | Well-studied, morphologically rich, modal particles | Partial (Weber & Hilton) |
| French | Well-studied, good translation equivalents | Partial (Weber & Hilton) |
| Chinese (Mandarin) | Typologically distant from English, large training data | Limited |
| Spanish | Large speaker population, subjunctive interaction | Limited |
| Japanese | Sentence-final particle system, very different structure | Limited |

**Expressions to test (per language):**

For each language, construct the equivalent of our 4 syntactic types:

**English → German example:**

| English | German | Type |
|---|---|---|
| "It is certain that X" | "Es ist sicher, dass X" | Predicative |
| "It is likely that X" | "Es ist wahrscheinlich, dass X" | Predicative |
| "It is possible that X" | "Es ist möglich, dass X" | Predicative |
| "It is unlikely that X" | "Es ist unwahrscheinlich, dass X" | Predicative |
| "X will probably succeed" | "X wird wahrscheinlich gelingen" | Modal adverb |
| "X will certainly succeed" | "X wird sicherlich gelingen" | Modal adverb |
| "X will possibly succeed" | "X wird möglicherweise gelingen" | Modal adverb |
| "X will perhaps succeed" | "X wird vielleicht gelingen" | Modal adverb |

Plus language-specific expressions that have no English equivalent:
- German: "wohl" (epistemic particle, ~70-80%), "dürfte" (subjunctive II
  of "dürfen," ~80%)
- French: "éventuellement" (= "possibly" NOT "eventually"), "sans doute"
  (= "probably" NOT "without doubt")
- Chinese: "也许" (yěxǔ, ~30-40%), "大概" (dàgài, ~60-70%), "肯定" (kěndìng, ~95%)

**Method:**

1. For each language, embed hedge expressions in the same template pattern as
   our English experiments
2. Compute difference vectors from bare claims in each language
3. Train a supervised probability axis using Mosteller medians as targets
   (using English probability values for the translated equivalents)
4. Test whether the axis:
   a. Produces a meaningful ordering within each language
   b. Correlates with available cross-linguistic calibration data
   c. Aligns with the English axis in shared embedding space

### Phase 2: Cross-Linguistic Axis Alignment (The Strong Test)

In a multilingual embedding model, English and German occupy the same vector
space. If the probability axis is language-universal:

```
english_axis = train_axis(english_diffs, mosteller_medians)
german_axis  = train_axis(german_diffs, mosteller_medians)

# Test 1: Do the axes point in the same direction?
cosine(english_axis, german_axis)  →  prediction: > 0.5

# Test 2: Does the English axis work for German expressions?
german_diffs @ english_axis  →  should correlate with probability

# Test 3: Does the German axis work for English expressions?
english_diffs @ german_axis  →  should correlate with probability
```

**The headline test:** Train the probability axis on English Mosteller data.
Project Chinese hedge expressions onto this English-trained axis. If the
Chinese expressions land in the correct probability order, the axis is
language-agnostic.

### Phase 3: Language-Specific Expressions (The Interesting Edge Cases)

Every language has hedge expressions that don't translate well:

**German "wohl":**
A modal particle that conveys something like "I assume / in all probability"
(~70-80%). No single English equivalent. Where does it project?

**French "quand même":**
Conveys "even so / despite expectations" — an epistemic-pragmatic hybrid.
Does it even have a probability projection?

**Chinese "说不定" (shuōbudìng):**
Literally "can't say for sure" — conveys possibility with an implication of
surprise if it happens (~30-40%). Does the surprise component project onto the
probability axis or orthogonally?

**Japanese "かもしれない" (kamoshirenai):**
Literally "it is not known" — grammaticalized possibility marker. Very
different morphological structure from English "maybe." Same geometric behavior?

These cases test whether the probability axis captures something deeper than
translational equivalence — whether it captures the actual epistemic function
regardless of how it's morphologically packaged.

## Predictions

### If linguistic universality holds:

- Cross-linguistic axis alignment (cosine > 0.5 between language-specific axes)
- English-trained axis successfully orders expressions in other languages
  (ρ > 0.7 zero-shot)
- Language-specific expressions (German "wohl," Chinese "说不定") project to
  reasonable positions on the English axis
- The axis alignment is STRONGER for typologically similar languages
  (English-German > English-Chinese) but present even for distant pairs

### If linguistic relativity holds:

- Language-specific axes point in different directions (cosine < 0.3)
- English-trained axis fails for other languages (ρ < 0.4 zero-shot)
- But within-language axes still work (ρ > 0.8 when trained on that language)
- The probability axis is real but language-specific — calibration cannot
  transfer across languages

### The most likely outcome (based on multilingual embedding literature):

- Moderate alignment (cosine 0.4-0.7) for typologically similar languages
- Weaker alignment (cosine 0.2-0.5) for distant languages
- But zero-shot transfer partially works (ρ > 0.5 even for distant pairs)
  because multilingual models are specifically trained to align semantic
  spaces across languages

This would mean the probability axis is **partially universal** — there is a
shared core of epistemic geometry, but each language contributes its own
variation (different morphological packaging, different pragmatic conventions,
different granularity of the probability scale).

## Connection to the Philosophical Argument

The cross-linguistic evidence directly addresses the strongest form of the
philosophical claim:

**If the axis is language-universal:** Degree of belief is not merely a
property of English or of specific language communities — it is a property
of human epistemic cognition as expressed through language. The Bayesian
"degree of belief" interpretation connects to a cognitive universal, not a
linguistic convention.

**If the axis is language-specific but structurally analogous:** Each
language community develops its own geometric encoding of probability, but
they all converge on the same structural pattern (linear axis, compositional
modifiers, negation as reflection). This is weaker than universality but still
supports the claim that degree of belief is a natural category — different
languages carve the probability space differently but recognize it as a
dimension.

**If the structure doesn't transfer at all:** The geometric encoding is an
artifact of English distributional statistics and the specific way English
hedging interacts with contrastive training objectives. The philosophical
claim would need to be restricted to language-specific phenomena.

## Practical Implementation

### What we can do immediately (1-2 days):

1. Pull a multilingual embedding model via Ollama or HuggingFace
   (multilingual-e5-large is likely available)
2. Construct hedge expression sets for German, French, and Chinese
   (using dictionaries + native speaker verification or LLM assistance)
3. Run the same analysis pipeline: embed, difference, train axis, test
4. Report: does the English-trained axis work cross-linguistically?

### What requires more effort:

- Native speaker verification of template naturalness
- Cross-linguistic calibration data collection (asking speakers of other
  languages what probability their hedge words convey)
- Typological analysis connecting hedge morphology to geometric structure
- The full cross-linguistic measurement model (adding language as a factor
  in the hierarchical Bayesian framework)

### What would make a separate paper:

- Comprehensive cross-linguistic study (10+ languages, calibration data per
  language, typological analysis)
- Connection to linguistic typology of evidentiality and epistemic modality
- The Sapir-Whorf angle: do speakers of languages with different hedge systems
  actually think about probability differently, and does the geometry reflect
  this?

## References

### Cross-Linguistic Probability Expressions
- Budescu, D. V., et al. (2014). "The interpretation of IPCC probabilistic
  statements around the world." *Nature Climate Change.* (Cross-cultural
  variation in probability expression interpretation)
- Weber, E. U., & Hilton, D. J. (1990). "Contextual effects in the
  interpretations of probability words." *JEP: Human Perception and
  Performance.* (French-English comparison)
- Renooij, S., & Witteman, C. (1999). "Talking probabilities: communicating
  probabilistic information with words and numbers." *Int. J. Approximate
  Reasoning.* (Dutch)

### Linguistic Typology of Epistemic Modality
- Palmer, F. R. (2001). *Mood and Modality.* Cambridge. (Cross-linguistic
  survey of modal systems)
- Nuyts, J. (2001). *Epistemic Modality, Language, and Conceptualization.*
  John Benjamins. (Cognitive-functional approach)
- Aikhenvald, A. Y. (2004). *Evidentiality.* Oxford. (Grammaticalized
  evidentiality across languages)
- de Haan, F. (2006). "Typological approaches to the domains of modality."
  In Frawley (ed.), *The Expression of Modality.*

### Multilingual Embeddings
- Conneau, A., et al. (2020). "Unsupervised Cross-lingual Representation
  Learning at Scale." *ACL.* (XLM-R and multilingual alignment)
- Wang, L., et al. (2024). "Multilingual E5 Text Embeddings." (The model
  we'd likely use)

### Linguistic Relativity and Universals
- Levinson, S. C. (2003). *Space in Language and Cognition.* Cambridge.
  (Evidence for linguistic relativity in spatial cognition)
- Evans, N., & Levinson, S. C. (2009). "The myth of language universals."
  *Behavioral and Brain Sciences.* (Strong relativity argument)
- Regier, T., et al. (2007). "Color naming across languages reflects color
  use." *PNAS.* (Universals in color naming — structural parallel to
  probability naming?)
