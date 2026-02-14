# Literature Review: Embedding-based Representations of Epistemic Strength

**Source:** undermind.ai automated literature search, February 2026
**Query focus:** Embedding-based representations of epistemic strength calibrated to human probability judgments

## Key Finding

**No existing paper does what we do.** The review identifies our work as filling a
"well-defined gap" at the intersection of three strands:

1. Verbal uncertainty as a linear direction in LLM activations (Ji et al. 2025)
2. Model epistemic uncertainty linearly decodable from embeddings (Lin et al. 2022)
3. Human psychometrics for verbal probability phrases (Mosteller & Youtz 1990, Wintle et al. 2019)

## Critical Papers to Cite

### Closest prior work

| Paper | Relation to our work |
|---|---|
| **Ji et al. 2025** — "Calibrating Verbal Uncertainty as a Linear Feature" | Finds a 1D VU direction in residual streams. No lexical inventory, no syntactic types, no human calibration, no cross-model. We do all four. |
| **Sileo & Moens 2023** — "Probing neural language models for WEPs" (*SEM) | Binary NLI classification of WEPs, not continuous calibration. Uses fine-tuning; we show the structure is already present in pretrained embeddings. |
| **Lin et al. 2022** — "Teaching models to express uncertainty in words" (TMLR) | Linear probe on GPT-3 embeddings predicts correctness. No hedge expressions, no human psychometric calibration. |

### Truth-direction precedents (methodological)

| Paper | Relation to our work |
|---|---|
| **Marks & Tegmark 2023** — "Geometry of Truth" | Truth as linear direction. They test "likely" briefly — it doesn't align with truth direction. Consistent with our finding that probability axis ≠ truth axis. |
| **Yu et al. 2025** — "From Directions to Cones" | Multi-dimensional truth structure. Our 4-axis subspace parallels their truth cones. |
| **Bürger et al. 2024** — "Truth is Universal" | Cross-model truth subspaces. Precedent for our cross-model consistency. |

### Psychometric anchors

| Paper | Relation to our work |
|---|---|
| **Wintle et al. 2019** — verbal probability psychometrics (n≈924) | Another calibration dataset. Could be a third cross-validation source. |
| **Wallsten et al. 1986** — "Measuring vague meanings of probability terms" | Classic calibration work. Foundation of the field. |

### Theoretical

| Paper | Relation to our work |
|---|---|
| **Schockaert 2022** — "Embeddings as Epistemic States" | Formal foundation for our philosophical argument about degree of belief as geometric property. |
| **Zhou et al. 2023** — "Navigating the Grey Area" | Most-cited foundational paper (72% reference rate). Must cite. |

## Five Gaps Our Work Fills (per the review)

1. **No direct calibration from embedding geometry to human probability scales** — we do this
2. **No systematic study of lexical and syntactic variation** — our 4-type analysis is the key contribution
3. **No disentangling epistemic strength from polarity, truth, affect** — our syntactic confound finding addresses this
4. **No cross-model robustness for epistemic hedging** — we test 5 architecturally diverse models
5. **No linking of lexical vs sentential epistemic scales** — our difference-vector method bridges this

## Full Review

The complete undermind.ai review with all tables, analysis, and adjacent work is
preserved separately. Key sections include comparative tables for:
- Target property and relation to epistemic hedging (Table 1)
- Representational level and geometric assumptions (Table 2)
- Calibration and human-related evaluation (Table 3)
- Lexical/syntactic epistemic expressions (Table 4)
- Cross-model robustness (Table 5)
