# Novelty memo for hedge embeddings

##### [**Undermind**](https://undermind.ai)

---


## Table of Contents

- [Key finding](#key-finding)
- [Bottom line for the novelty claim](#bottom-line-for-the-novelty-claim)
- [Direct competitors and near misses](#direct-competitors-and-near-misses)
- [What each pressure point forces you to say](#what-each-pressure-point-forces-you-to-say)
  - [Continuous human targets already existed](#continuous-human-targets-already-existed)
  - [Human verbal probability psychometrics already existed](#human-verbal-probability-psychometrics-already-existed)
  - [Linear uncertainty geometry already existed](#linear-uncertainty-geometry-already-existed)
  - [The conjunction still looks open](#the-conjunction-still-looks-open)
- [Is the model class distinction meaningful](#is-the-model-class-distinction-meaningful)
- [Which comparators belong in the abstract](#which-comparators-belong-in-the-abstract)
- [Post February 2026 check](#post-february-2026-check)
- [Recommended novelty wording](#recommended-novelty-wording)
  - [Best abstract level wording](#best-abstract-level-wording)
  - [Best introduction level priority claim](#best-introduction-level-priority-claim)
  - [If you want a slightly softer but still assertive version](#if-you-want-a-slightly-softer-but-still-assertive-version)
  - [Wording I would avoid](#wording-i-would-avoid)
- [Reviewer bait to pre-empt](#reviewer-bait-to-pre-empt)
  - [The strongest literature level objection](#the-strongest-literature-level-objection)
  - [The second strongest objection](#the-second-strongest-objection)
  - [The third strongest objection](#the-third-strongest-objection)
  - [Additional points worth addressing in the paper](#additional-points-worth-addressing-in-the-paper)
- [Overall judgment](#overall-judgment)
- [References](#references)

## Key finding

No paper I read demonstrates the full conjunction at issue: a continuously calibrated verbal probability axis extracted from frozen pretrained pooled sentence embeddings, aligned with human psychometric judgments, and validated beyond the fitting set. The closest papers each cover only part of that space. The strongest pressure points are \[Che19\], \[Mas22\], \[Sil22\], \[Bel24\], \[Tan24\], \[Ji25\], and \[Lep25\].

The main update from the full reads is that two papers deserve more weight than the search summary alone suggests. \[Mas22\] is a real embedding space comparator because it probes mean pooled RoBERTa sentence representations against human cloze based uncertainty. \[Lep25\] is a real human aligned geometry comparator because it links decoder hidden state directions to graded human plausibility and modal judgments. Neither one appears to break the narrowed claim, but both should be pre-empted in related work.

## Bottom line for the novelty claim

The literature appears to support a strong claim if it stays explicit about the object of study: verbal probability semantics in frozen pooled sentence embeddings, not uncertainty in language models in general. Without that verbal probability qualifier, the claim becomes more vulnerable to \[Mas22\] and the broader human aligned uncertainty literature. With it, the claim still looks intact.

A defensible strongest form is:

- Frozen pretrained pooled sentence embeddings encode a continuously calibrated verbal probability axis aligned with human psychometric judgments.
- No prior paper located in this audit appears to show that conjunction.

If the paper wants an explicit priority sentence, the most defensible version is narrower than the current draft:

- This work is the first demonstration of a continuously calibrated verbal probability axis in frozen pretrained pooled sentence embeddings, aligned with human psychometric judgments and validated against independent psychometric datasets.

That wording is stronger than the current draft in one useful way. It replaces the broader phrase “probability structure” with “verbal probability axis,” which better excludes the strongest near misses.

## Direct competitors and near misses

| Paper | What overlaps | Why it does not appear to break the claim | Verdict |
|:---|:---|:---|:---|
| \[Che19\] | Continuous human scalar judgments for sentence meaning. BERT `[CLS]` used in regression. | Fine tuned NLI model, task specific sentence pair setup, no frozen geometry claim, no linear axis, no independent psychometric transfer. | Strong near miss |
| \[Mas22\] | Mean pooled RoBERTa sentence embeddings, linear probe, human uncertainty grounding through cloze probability and entropy. | Next word predictability and sentential constraint, not verbal probability expressions or hedge semantics, binary strong vs weak probe target, no cross dataset psychometric calibration. | Strong near miss |
| \[Sil22\] | Human numerical meanings for verbal probability expressions. | Behavioral prompting and fine tuning, categorical choice tasks, no embedding geometry, no linear axis. | Strong near miss |
| \[Bel24\] | Human psychometric mapping from uncertainty phrases to numbers. | Prompted number generation in decoder LLMs, no pooled embeddings, no geometry. | Strong near miss |
| \[Tan24\] | Human verbal probability calibration and multilingual behavioral evaluation. | Prompt based numeric elicitation, no embedding analysis, no learned axis. | Strong near miss |
| \[Ji25\] | Linear verbal uncertainty feature in pretrained representations. | Decoder residual streams, LLM as judge labels, causal steering during generation, calibration to model behavior rather than human psychometric verbal probability. | Strong comparator outside model class |
| \[Lep25\] | Linear geometry aligned with graded human modal and plausibility judgments. | Decoder final token states, modal categories rather than hedge lexicon, no pooled sentence embeddings, no verbal probability psychometric datasets. | Strong comparator outside model class |
| \[Bha16\] | Pretrained distributional vectors predict human probability judgments. | Event likelihood from keyword associations, not sentence embeddings, not verbal probability semantics, no psychometric transfer design. | Older precursor |
| \[Pei21\] | Continuous certainty regression from pretrained transformers. | Fine tuned certainty prediction, rhetorical certainty in science communication, no frozen geometry or psychometric verbal probability axis. | Adjacent background |
| \[Lee15\], \[Sta17\] | Scalar factuality prediction from human judgments and cross dataset scaling. | Structured task models, no pretrained pooled embeddings, no geometry claim. | Adjacent background |
| \[Sch19c\] | Human scalar pragmatic ratings predicted from sentence encoders. | Task trained encoder, pragmatic inference strength rather than probability semantics, no geometry claim. | Adjacent background |
| \[She23b\] | Probabilistic sentence embeddings on top of pretrained models. | Models distributional embeddings for model and data uncertainty, not human psychometric semantics, not a linear probability axis. | Likely reviewer confusion only |
| \[Wan24b\] | Calibration of certainty phrases against human data. | Phrase level calibration via distributions and transport, no embeddings, no geometry. | Likely reviewer confusion only |

## What each pressure point forces you to say

### Continuous human targets already existed

That part is not novel on its own. \[Che19\] shows scalar human probability judgments in sentence pair inference. \[Lee15\], \[Sta17\], \[Pei21\], and \[Sch19c\] show continuous human rated certainty, factuality, or inference strength in sentence level NLP settings. The paper should not imply that continuous human scalar supervision for epistemic meaning begins here.

### Human verbal probability psychometrics already existed

That part is also not novel on its own. \[Sil22\], \[Bel24\], \[Tan24\], and \[Wan24b\] all treat verbal uncertainty expressions as human interpretable quantitative objects, though they do so behaviorally rather than geometrically.

### Linear uncertainty geometry already existed

That part is also not novel on its own. Decoder state papers including \[Mar23\], \[Ji25\], \[Ahd24\], \[Coh25\], \[Yu25\], \[Lep25\], \[Cho26\], and \[Ada26\] show that truth, uncertainty, correctness, or modality can be linearly or low dimensionally accessed in pretrained LLM internals.

### The conjunction still looks open

What still appears missing is the conjunction of:

- frozen pretrained pooled sentence embeddings
- a linear axis rather than just a predictive head
- verbal probability or epistemic hedge semantics rather than generic certainty or token predictability
- alignment to human psychometric judgments
- validation beyond the fitting set, ideally against independent psychometric studies

That is the real novelty territory supported by this audit.

## Is the model class distinction meaningful

Yes, but it needs to be argued as a parallel methodological territory, not as a wholly separate universe. The decoder state literature studies token localized residual stream representations during generation and validates them with steering, patching, or other causal interventions \[Mar23, Ji25, Yu25, Cho26, Ada26\]. Your target models instead expose fixed sentence vectors meant for retrieval, comparison, and semantic encoding. In that setting, concept erasure and fixed space decoding are the natural analogue.

This distinction will read as meaningful to a TACL audience if the paper also concedes two things plainly.

- The broader family resemblance is real. The work belongs in the same geometric interpretability arc as \[Mar23\] and \[Ji25\].
- Sentence representation work is not empty background. \[Mas22\] and \[She23b\] show that embedding spaces themselves have already been studied as uncertainty bearing objects.

A good framing sentence for the introduction is:

- Prior work has shown human aligned uncertainty behavior in prompted LMs and linear uncertainty features in decoder activations, but not a psychometrically calibrated verbal probability axis in frozen pooled sentence embeddings.

## Which comparators belong in the abstract

The current trio should change a little.

**Keep** \[Ji25\]. It is the closest recent paper on verbal uncertainty as a linear feature.

**Keep** \[Mar23\] if the abstract wants one broad anchor from linear feature interpretability.

**Add** one psychometric verbal probability paper, preferably \[Bel24\] or \[Tan24\]. Those papers make clear that human numerical interpretation of hedge expressions is established, while also making clear that the existing evidence is behavioral rather than geometric.

**Move out of the abstract** \[Val24\]. It is methodological background on decoder concepts and unembedding space, not a close novelty threat.

**Consider adding in related work** \[Mas22\] and \[Lep25\]. The former is the strongest embedding space near miss. The latter is the strongest recent human aligned decoder geometry near miss.

A compact comparator set is therefore:

| Role | Best fit |
|:---|:---|
| Linear feature in decoder states | \[Ji25\] |
| Broad geometry of truth and factuality | \[Mar23\] |
| Human verbal probability psychometrics in LMs | \[Bel24\] or \[Tan24\] |
| Strong embedding space near miss for related work | \[Mas22\] |
| Strong human aligned decoder geometry near miss for related work | \[Lep25\] |

## Post February 2026 check

I did not find a post February 2026 paper that breaks the claim. The most relevant recent items I found continue the decoder internal line rather than moving into pooled sentence embeddings.

- \[Cho26\] studies low dimensional correctness geometry in decoder residual states with causal steering, but uses factual benchmarks rather than human psychometric verbal probability data.
- \[Ada26\] studies how context changes decoder truth vectors. It is related work for truth geometry, not a direct novelty threat.
- \[Yan26\] studies prompt based elicitation of higher order uncertainty through imprecise probabilities, not embedding geometry.

So the post February 2026 literature changes the background pressure, not the verdict.

## Recommended novelty wording

### Best abstract level wording

We show that frozen pretrained pooled sentence embeddings encode a continuously calibrated verbal probability axis aligned with human psychometric judgments.

That sentence is strong, specific, and avoids making the priority claim carry the whole abstract.

### Best introduction level priority claim

This work is the first demonstration of a continuously calibrated verbal probability axis in frozen pretrained pooled sentence embeddings, aligned with human psychometric judgments and validated against independent psychometric datasets.

### If you want a slightly softer but still assertive version

Prior work has not shown a psychometrically calibrated verbal probability axis in frozen pooled sentence embeddings.

### Wording I would avoid

- first demonstration of calibrated probability structure in pretrained embeddings
- first demonstration of continuous probability in sentence representations
- first human aligned uncertainty direction in language model representations

Those versions are too easy to attack using \[Mas22\], \[Che19\], \[Lep25\], or the broader decoder geometry line.

## Reviewer bait to pre-empt

### The strongest literature level objection

A reviewer may say that \[Mas22\] already found human uncertainty structure in pooled RoBERTa sentence embeddings. That objection should be answered directly. The distinction is that \[Mas22\] concerns sentential constraint for next word prediction, operationalized through cloze probability and entropy, not the psychometric semantics of verbal probability expressions.

### The second strongest objection

A reviewer may say that \[Che19\] already did continuous human probability regression with BERT sentence representations. The reply is that \[Che19\] is a fine tuned NLI regression model over sentence pairs, not an analysis of frozen pooled embedding geometry, and it does not identify a general verbal probability axis.

### The third strongest objection

A reviewer may say that \[Ji25\] and \[Lep25\] already showed linear uncertainty or modal structure aligned with graded judgments. The reply is that both papers work in decoder internal states, not pooled embedding models, and neither is calibrated against verbal probability psychometrics.

### Additional points worth addressing in the paper

- State early that the contribution is not continuous scalar supervision per se.
- State early that the contribution is not linear feature interpretability per se.
- State early that cross linguistic ranking transfer is stronger than cross linguistic calibration unless supported by native speaker psychometrics.
- Cite \[Sil22\], \[Bel24\], and \[Tan24\] when introducing verbal probability semantics, so reviewers cannot say the paper ignored that line.
- Cite \[Mas22\] when introducing embedding geometry and uncertainty, so reviewers cannot say the paper ignored the one serious pooled embedding precursor.

## Overall judgment

The substance first framing is the right choice. The evidence is strong enough that the paper does not need to lead with a fragile priority claim. The best use of novelty language is as a precise subordinate clause that names the conjunction clearly. On the present evidence, the conjunction still looks novel.

---

## References

\[Che19\] T. Chen, Z. Jiang, K. Sakaguchi, and B. V. Durme, “Uncertain Natural Language Inference,” *Annual Meeting of the Association for Computational Linguistics*, pp. 8772–8779, Sep. 2019, doi: [10.18653/v1/2020.acl-main.774](https://doi.org/10.18653/v1/2020.acl-main.774).

\[Mas22\] “Masked language models directly encode linguistic uncertainty,” 2022.

\[Sil22\] D. Sileo and M. Moens, “Probing neural language models for understanding of words of estimative probability,” *STARSEM*, pp. 469–476, Nov. 2022, doi: [10.48550/arXiv.2211.03358](https://doi.org/10.48550/arXiv.2211.03358).

\[Bel24\] C. G. Belém, M. Kelly, M. Steyvers, S. Singh, and P. Smyth, “Perceptions of Linguistic Uncertainty by Language Models and Humans,” *ArXiv*, vol. abs/2407.15814, Jul. 2024, doi: [10.48550/arXiv.2407.15814](https://doi.org/10.48550/arXiv.2407.15814).

\[Tan24\] Z. Tang, K. Shen, and M. Kejriwal, “An evaluation of estimative uncertainty in large language models,” *npj Complexity*, vol. 3, May 2024, doi: [10.1038/s44260-026-00070-6](https://doi.org/10.1038/s44260-026-00070-6).

\[Ji25\] Z. Ji *et al.*, “Calibrating Verbal Uncertainty as a Linear Feature to Reduce Hallucinations,” *Conference on Empirical Methods in Natural Language Processing*, pp. 3769–3793, Mar. 2025, doi: [10.48550/arXiv.2503.14477](https://doi.org/10.48550/arXiv.2503.14477).

\[Lep25\] M. A. Lepori, J. Hu, I. Dasgupta, R. Patel, T. Serre, and E. Pavlick, “Is This Just Fantasy? Language Model Representations Reflect Human Judgments of Event Plausibility,” *ArXiv*, vol. abs/2507.12553, Jul. 2025, doi: [10.48550/arXiv.2507.12553](https://doi.org/10.48550/arXiv.2507.12553).

\[Bha16\] S. Bhatia, “Vector Space Semantic Models Predict Subjective Probability Judgments for Real-World Events,” *Cognitive Science*, 2016.

\[Pei21\] J. Pei and D. Jurgens, “Measuring Sentence-Level and Aspect-Level (Un)certainty in Science Communications,” *ArXiv*, vol. abs/2109.14776, Sep. 2021, doi: [10.18653/v1/2021.emnlp-main.784](https://doi.org/10.18653/v1/2021.emnlp-main.784).

\[Lee15\] K. Lee, Y. Artzi, Y. Choi, and L. Zettlemoyer, “Event Detection and Factuality Assessment with Non-Expert Supervision,” *Conference on Empirical Methods in Natural Language Processing*, pp. 1643–1648, Sep. 2015, doi: [10.18653/v1/D15-1189](https://doi.org/10.18653/v1/D15-1189).

\[Sta17\] G. Stanovsky, J. Eckle-Kohler, Y. Puzikov, I. Dagan, and I. Gurevych, “Integrating Deep Linguistic Features in Factuality Prediction over Unified Datasets,” *Annual Meeting of the Association for Computational Linguistics*, pp. 352–357, Jul. 2017, doi: [10.18653/v1/P17-2056](https://doi.org/10.18653/v1/P17-2056).

\[Sch19c\] S. Schuster, Y. Chen, and J. Degen, “Harnessing the linguistic signal to predict scalar inferences,” *Annual Meeting of the Association for Computational Linguistics*, pp. 5387–5403, Oct. 2019, doi: [10.18653/v1/2020.acl-main.479](https://doi.org/10.18653/v1/2020.acl-main.479).

\[She23b\] L. Shen, H. Jiang, L. Liu, and S. Shi, “Sen2Pro: A Probabilistic Perspective to Sentence Embedding from Pre-trained Language Model,” *ArXiv*, vol. abs/2306.02247, Jun. 2023, doi: [10.48550/arXiv.2306.02247](https://doi.org/10.48550/arXiv.2306.02247).

\[Wan24b\] P. Wang *et al.*, “Calibrating Expressions of Certainty,” *ArXiv*, vol. abs/2410.04315, Oct. 2024, doi: [10.48550/arXiv.2410.04315](https://doi.org/10.48550/arXiv.2410.04315).

\[Mar23\] S. Marks and M. Tegmark, “The Geometry of Truth: Emergent Linear Structure in Large Language Model Representations of True/False Datasets,” *ArXiv*, vol. abs/2310.06824, Oct. 2023, doi: [10.48550/arXiv.2310.06824](https://doi.org/10.48550/arXiv.2310.06824).

\[Ahd24\] G. Ahdritz, T. Qin, N. Vyas, B. Barak, and B. L. Edelman, “Distinguishing the Knowable from the Unknowable with Language Models,” *ArXiv*, vol. abs/2402.03563, Feb. 2024, doi: [10.48550/arXiv.2402.03563](https://doi.org/10.48550/arXiv.2402.03563).

\[Coh25\] R. Cohen, O. Fahn, and G. de Melo, “Pretrained LLMs Learn Multiple Types of Uncertainty,” *ArXiv*, vol. abs/2505.21218, May 2025, doi: [10.48550/arXiv.2505.21218](https://doi.org/10.48550/arXiv.2505.21218).

\[Yu25\] S. Yu *et al.*, “From Directions to Cones: Exploring Multidimensional Representations of Propositional Facts in LLMs,” *ArXiv*, vol. abs/2505.21800, May 2025, doi: [10.48550/arXiv.2505.21800](https://doi.org/10.48550/arXiv.2505.21800).

\[Cho26\] S. Cho, Z. Wu, K. Costa, and A. S. Koshiyama, “The Confidence Manifold: Geometric Structure of Correctness Representations in Language Models,” Feb. 08, 2026.

\[Ada26\] S. Adarsh, M. Maistro, and C. Lioma, “How Context Shapes Truth: Geometric Transformations of Statement-level Truth Representations in LLMs,” *ArXiv*, vol. abs/2601.06599, Jan. 2026, doi: [10.48550/arXiv.2601.06599](https://doi.org/10.48550/arXiv.2601.06599).

\[Val24\] P. Valois, L. Souza, E. K. Shimomoto, and K. Fukui, “Frame Representation Hypothesis: Multi-Token LLM Interpretability and Concept-Guided Text Generation,” *Trans. Assoc. Comput. Linguistics*, vol. 13, pp. 1436–1458, Dec. 2024, doi: [10.1162/TACL.a.48](https://doi.org/10.1162/TACL.a.48).

\[Yan26\] A. Yang, K. Muandet, M. Caprio, S. L. Chau, and M. Adachi, “Verbalizing LLM’s Higher-order Uncertainty via Imprecise Probabilities,” Mar. 11, 2026.
