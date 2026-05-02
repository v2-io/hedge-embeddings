# Reframed Contribution — Brainstorm

> **⚠️ THIS IS A BRAINSTORM, NOT A DECISION.**
>
> Drafted by Claude on 2026-05-01 in response to "what would the model-class-distinction reframing actually look like in the abstract and introduction." These are *raw options*, not recommendations. Joseph should expect to rewrite, blend, reject, or invert these. The cmcl-abstract-draft.md framing is also defensible — the question is what to amplify in the TACL expansion, and that is Joseph's call.
>
> Three alternatives below at different intensities. None of them have been read against the actual draft prose. None of them have been word-counted. None of them have been checked against TACL's specific abstract conventions. They are starting material.

---

## What this is responding to

After direct reading of the three comparator papers (Ying TACL 2025, Valentin TACL 2025, Ji EMNLP 2025), the most underplayed contribution claim in the current cmcl abstract is the **model-class distinction**: the recent linear-feature literature works on *decoder LLM* internal states (residual streams, unembedding matrices) where token-level intervention is natural, while this work establishes analogous structure on *sentence embedding* models (mxbai, nomic, qwen3-embedding, bge-m3) which pool to single vectors and have no token-generation pathway. No prior work in the linear-feature corpus has shown calibrated continuous epistemic structure on this model class.

The three alternatives below differ in how much they lean into this framing.

---

## Option A — Light edit (closest to current draft)

*Smallest delta. Adds one sentence of model-class positioning, sharpens "calibrated" against Ji.*

> We investigate whether epistemic hedging — the linguistic expression of uncertainty through words like "probably," "certainly," and "possibly" — manifests as calibrated linear structure in pretrained sentence embedding space. Recent linear-feature work has established that uncertainty (Ji et al., 2025), truth (Marks & Tegmark, 2023), and gradable concepts (Valentin et al., 2025) are linearly encoded in *decoder language model* internal states, where token-level intervention is natural. We ask the parallel question for sentence embedding models — encoder/pooled architectures that compress text to single vectors — and demonstrate that within syntactic types, probability is the dominant linear direction in five architecturally diverse embedding models (Spearman ρ > 0.90 supervised). The axis cross-validates against two independent psychometric datasets spanning 50 years (Mosteller & Youtz, 1990: ρ_LOO 0.67–0.95; Vogel, 2022: ρ = 0.99; Wintle, 2019: ρ = 0.97), transfers zero-shot to 8 typologically diverse languages (mean |ρ| = 0.93), survives 12× dimensional compression, and passes permutation null hypothesis tests (p ≤ 0.005). Unlike Ji et al., whose linear feature is calibrated to model-internal verbal-uncertainty judgments, our axis is calibrated to *human psychometric data*. These results suggest that pretrained sentence embedding models learn a cross-linguistic geometric encoding of human probability semantics as an emergent property of distributional training.

**Tradeoffs:** Preserves the existing structure. Adds the model-class distinction in one sentence and the Ji-vs-human-psychometric distinction in one sentence. Does not commit to "first work to do X" claim. Safest if you're worried about overclaiming.

---

## Option B — Foreground the model-class gap

*Moderate rewrite. Leads with what this paper does that no prior work has done. Pre-empts the (d1) "what is being measured" critique by naming the methodological territory explicitly.*

> Recent interpretability research has established that uncertainty (Ji et al., 2025), truth (Marks & Tegmark, 2023), and gradable concepts (Valentin et al., 2025) are linearly encoded in decoder language models — in residual stream activations or unembedding matrices, where token-level causal intervention is natural. Whether analogous structure exists in *sentence embedding models* — encoder/pooled architectures that compress entire sentences to single vectors and do not have a token-generation pathway — remains open. We show that it does, and that in this model class the structure is calibrated against human psychometric data rather than model behavior. Using the Mosteller & Youtz (1990) dataset of 53 verbal probability expressions (n=238 science writers), we find that within syntactic types, probability is the dominant linear direction in five architecturally diverse pretrained embedding models, with Spearman ρ > 0.90 supervised and 0.67–0.95 leave-one-out. The axis cross-validates against two independent psychometric datasets spanning 50 years (Vogel 2022: ρ = 0.99; Wintle 2019: ρ = 0.97), transfers zero-shot to 8 typologically diverse languages (mean |ρ| = 0.93), survives 12× dimensional compression, and passes permutation null tests (p ≤ 0.005). [If concept-erasure experiment is included: We further demonstrate the axis is functional, not merely descriptive: erasing it from sentence embeddings degrades downstream probability-prediction performance by X%, while a matched random-direction erasure does not.] To our knowledge this is the first demonstration of calibrated continuous epistemic structure in pretrained sentence embeddings.

**Tradeoffs:** Strongest framing of the contribution. Commits to "first to our knowledge" — defensible after a thorough prior-art pass on sentence-embedding probing (must include Sileo & Moens 2023 differentiation, since they probe sentence-level epistemic *classification* but not *calibration*). The bracketed sentence assumes the concept-erasure experiment lands; remove if it doesn't. The opening sentence requires the reader to know the linear-feature literature, which TACL reviewers will, but a non-NLP reader (e.g., Synthese referee on a parallel submission) will need more setup.

---

## Option C — Lead with the cognitive-science bridge

*More ambitious rewrite. Frames the contribution as a bridge between distributional geometry and human probability semantics, with the model-class distinction supporting the bridge claim. Closer in spirit to Ying et al.'s framing of LaBToM.*

> Human languages encode probability through dense lexicons of epistemic expressions ("probably," "certain," "high chance") whose meanings have been mapped psychometrically since the 1960s, with median calibrations remarkably stable across 50 years of replication (Mosteller & Youtz 1990; Vogel 2022; Wintle 2019). Whether modern distributional language models recover this structure has been studied at the *behavior* level (Lin et al., 2022; Sileo & Moens, 2023) and recently in *decoder LLM internal states* (Ji et al., 2025). We ask the question one model class earlier in the pipeline: do pretrained *sentence embedding* models — pooled encoder architectures with no token-generation pathway — already encode a calibrated probability axis aligned with human psychometric data? They do. Within syntactic types, probability is the dominant linear direction in five architecturally diverse models (Spearman ρ > 0.90 supervised, 0.67–0.95 leave-one-out), the axis cross-validates against two independent datasets spanning 50 years (ρ = 0.99, 0.97), transfers zero-shot to 8 typologically diverse languages (mean |ρ| = 0.93), and passes permutation null tests (p ≤ 0.005). The convergence of three independent psychometric datasets, six embedding models, and eight languages on the same continuous axis suggests these are independent measurements of one underlying quantity: the communicative probability function that human languages have evolved to express.

**Tradeoffs:** Most elevated framing; reads more like a Synthese / *Cognitive Science* paper than a probing paper. Strongest for the philosophical/cog-sci audience (Synthese, *Topics in Cognitive Science*). Slight risk for TACL reviewers who want "what's the technical contribution" upfront — they may have to wait until the second half of the abstract. Use this version if you also want to repurpose it for paper 9 (epistemic humility) Synthese submission.

---

## Introduction-paragraph option (separate from abstract)

*This is for the opening paragraph of §1, not the abstract. Designed to install the model-class distinction as the framing before the empirical contributions arrive.*

> Mikolov et al. (2013) demonstrated that semantic relationships manifest as consistent vector directions in word embedding space — *king − man + woman ≈ queen*. The Linear Representation Hypothesis (LRH) generalizes this observation: many human-meaningful concepts correspond to linear directions in language model representations (Park et al., 2023). The recent interpretability literature has substantially extended this hypothesis at the *decoder LLM* level. Marks & Tegmark (2023) identify a truth direction in residual streams. Bürger et al. (2024) refine truth representations to handle polarity confounds. Yu et al. (2025) generalize one-dimensional truth directions to multidimensional cones. Ji et al. (2025) identify a linear "verbal uncertainty" feature that can be causally manipulated to reduce hallucinations. Valentin et al. (2025) extend single-token concept directions to multi-token "frames" via the Frame Representation Hypothesis. All of these works share a methodological commitment: they study internal states of *decoder LLMs* — residual streams, unembedding matrices, layer-wise activations — where token-level intervention is natural because the model generates tokens.
>
> A parallel question, less examined in the linear-feature literature, concerns *sentence embedding models* — encoder or pooled architectures (BERT-family, multilingual XLM-R, decoder-derived embedding heads) that compress entire sentences to single vectors for retrieval and similarity tasks. These models do not have a token-generation pathway, do not expose a residual stream in the standard sense, and resist direct causal-intervention methods designed for decoder-LLM internals. Whether they encode the kind of calibrated continuous structure that the recent linear-feature literature has documented in decoder LLMs is unclear from prior work.
>
> We address this gap for one specific phenomenon: epistemic hedging.

**Tradeoffs:** Honestly positions the entire prior-art body and names the gap concretely. Requires a corresponding Related Work section that goes deeper on each cited work — but that section is needed regardless (Gate 1). Reads as a "journal-style" introduction, which fits TACL well.

---

## Things to be careful about regardless of option chosen

1. **Don't overclaim "first."** Sileo & Moens (2023) probed sentence-level epistemic *classification* on pretrained models. Lin et al. (2022) probed sentence-level uncertainty extraction in GPT-3 embeddings. The "first" claim must be specifically about *continuously calibrated probability structure aligned with psychometric data* — not "first to study epistemic content in sentence embeddings." Do a prior-art pass before settling on any "first to our knowledge" language.

2. **The model-class distinction is real but not absolute.** Some sentence embedding models are decoder-derived (qwen3-embedding, embeddinggemma) — they share weights or ancestry with decoder LLMs. The interesting claim is about the *pooled output representation*, which is functionally an encoder regardless of training history. Phrase carefully: "pooled sentence representations" rather than "encoder-only models" avoids the architectural-history quibble.

3. **"Calibrated" vs. Ji.** Ji's title is "Calibrating Verbal Uncertainty as a Linear Feature." Joseph's calibration target is human psychometric medians; Ji's calibration target is LLM-as-a-Judge verbal-uncertainty scores aligned with Semantic Uncertainty. Both are legitimate uses of "calibrated." If you keep "calibrated" in the title or abstract, the differentiation has to be explicit and early — ideally in the first paragraph of §1, not buried in §4 Discussion.

4. **The 50-year framing is rhetorically strong but factually loose.** Mosteller & Youtz 1990 is the consolidating paper; the underlying data goes back further (Lichtenstein & Newman 1967, Beyth-Marom 1982, Wallsten et al. 1986). Vogel 2022 is a meta-analysis spanning 1967–2018, which is where the "50-year span" comes from. If you keep that framing, attribute the span correctly to "the meta-analyzed literature 1967–2018" rather than to any single paper.

5. **Cross-linguistic claim scope.** The mean |ρ| = 0.928 is real. The variance across languages is also real (Japanese modal 0.857, Korean modal 0.841, Spanish modal 0.852). Whichever abstract option is chosen, the cross-linguistic sentence should claim *ranking transfer* unambiguously, not *calibration transfer*. "Ranks hedge expressions correctly across 8 typologically diverse languages" is defensible. "Transfers zero-shot to 8 typologically diverse languages with mean |ρ| = 0.93" is also defensible if the |ρ| number is qualified as ranking correlation, not calibration accuracy.
