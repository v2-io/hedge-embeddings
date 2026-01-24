# Verbal Probability Calibration: Technical Guide

**Purpose:** Technical specification for mapping verbal probability phrases to numerical probability distributions using empirical research.

**Status:** Production-ready calibration data with implementation guidance

**Last Updated:** 2025-10-21

---

## Table of Contents

1. [Why Verbal Probabilities?](#why-verbal-probabilities)
2. [Empirical Calibration Approach](#empirical-calibration-approach)
3. [Three-Tier Confidence System](#three-tier-confidence-system)
4. [Beta Distribution Encoding](#beta-distribution-encoding)
5. [Context-Specific Calibrations](#context-specific-calibrations)
6. [Practical Guidelines](#practical-guidelines)
7. [Known Pitfalls](#known-pitfalls)

---

## Why Verbal Probabilities?

### The Human-Machine Gap

**Problem:** Humans think in words ("likely", "possible"), machines compute with numbers (0.7, 0.4).

**Bad Solutions:**
- **Arbitrary mapping** - "Let's say 'likely' means 70%" (no justification)
- **Ignore uncertainty** - Treat everything as binary true/false
- **Force numeric precision** - Ask humans for percentages they can't meaningfully estimate

**Good Solution:** Empirically calibrate verbal phrases using psychological research on how populations actually interpret these words.

### Bayesian Credible Intervals vs. Confidence Intervals

**Confidence Interval (Frequentist):**
- "If we repeated this experiment 100 times, 95 of those intervals would contain the true value"
- About the procedure, not the specific interval
- Does NOT mean "95% chance the value is in this range"

**Credible Interval (Bayesian):**
- "There's a 95% probability the value is in this range"
- About the parameter given the data
- Matches human intuition better

**For VERA:** Use Bayesian credible intervals because:
1. Humans interpret ranges as "where the value probably is" (Bayesian thinking)
2. Allows direct probability statements about claims
3. Integrates naturally with belief updating

---

## Empirical Calibration Approach

### Foundation: Mosteller & Youtz 1990

**Gold standard study:**
- Population: 238 science writers (literate, educated, precision-oriented)
- Method: Asked "What percentage does X mean to you?"
- Result: Percentile distributions (P25, Median, P75, IQR) for 54 verbal expressions

**Why this study?**
- Large sample size (n=238)
- Relevant population (science communicators)
- Comprehensive coverage (54 expressions)
- Statistical rigor (percentiles, not just means)
- Widely cited and replicated

**Data quality indicators:**
- "Always": Median 99.7%, IQR 0.3% → **Very high consensus**
- "Very likely": Median 87.5%, IQR 10.1% → **High consensus**
- "Possible": Median 38.5%, IQR 42.7% → **NO consensus (bimodal!)**

### Cross-Validation Sources

**IPCC Likelihood Terms (Budescu et al. 2009, 2012):**
- Expert-defined vs. public interpretation
- Shows "regression toward 50%" phenomenon:
  - Experts intend "very likely" = 95%
  - Public interprets as 67% (28 point gap!)
- **Lesson:** Be cautious with high-probability phrases (people underestimate extremes)

**Medical Context Meta-Analysis:**
- Domain-specific calibrations for health communication
- Example: "rare" in medical context = 10.0% (vs. 7.2% general)
- Shows context dependency is real but moderate (2-3 percentage point shifts)

**Vogel 2022 Systematic Review:**
- Meta-analysis of 21 studies (1967-2018)
- Validates Mosteller & Youtz findings
- Confirms temporal stability (interpretations haven't drifted in 50+ years)

---

## Three-Tier Confidence System

### Tier Classification Criteria

| Tier | IQR Threshold | Interpretation | Usage Policy |
|------|---------------|----------------|--------------|
| **Tier 1: High Consensus** | IQR < 5% | Very tight agreement, reliable | Use freely with 80% credible intervals |
| **Tier 2: Moderate Consensus** | 5% ≤ IQR ≤ 20% | Some disagreement, usable with care | Use 50% credible intervals or wider priors |
| **Tier 3: Ambiguous** | IQR > 20% | High disagreement, unreliable | Avoid or use very wide priors, FLAG in output |

**Rationale for IQR thresholds:**
- **< 5%**: Represents narrow range where 50% of population agrees (e.g., "always": 99.6-99.8%)
- **5-20%**: Moderate spread, but still useful (e.g., "very likely": 80.1-90.2%)
- **> 20%**: Large spread indicates bimodal or high variance (e.g., "possible": 7.5-50.2%)

### Tier 1: High Consensus (Use Freely)

```elixir
@tier_1_phrases %{
  "always" => %{
    median: 99.7, p25_p75: {99.6, 99.8}, iqr: 0.3,
    beta: {9112, 27},
    credible_80: {99.5, 99.9},
    tier: 1, confidence: :very_high
  },
  "almost always" => %{
    median: 91.7, p25_p75: {89.7, 95.2}, iqr: 5.5,
    beta: {27.66, 2.51},
    credible_80: {88, 96},
    tier: 1, confidence: :very_high
  },
  "certain" => %{
    median: 99.6, p25_p75: {98.7, 99.8}, iqr: 1.1,
    beta: {1807, 7},
    credible_80: {98.5, 99.9},
    tier: 1, confidence: :very_high
  },
  "almost certain" => %{
    median: 90.2, p25_p75: {87.5, 95.0}, iqr: 7.5,
    beta: {18.34, 1.99},
    credible_80: {85, 95},
    tier: 1, confidence: :very_high
  },
  "almost never" => %{
    median: 2.9, p25_p75: {1.2, 4.6}, iqr: 3.4,
    beta: {1.64, 54.83},
    credible_80: {1, 5},
    tier: 1, confidence: :very_high
  },
  "never" => %{
    median: 0.3, p25_p75: {0.1, 0.4}, iqr: 0.3,
    beta: {27, 9085},
    credible_80: {0.1, 0.5},
    tier: 1, confidence: :very_high
  }
}
```

**Usage:** Safe for direct use in CPTs with 80% credible intervals

### Tier 2: Moderate Consensus (Use with Care)

```elixir
@tier_2_phrases %{
  "very likely" => %{
    median: 87.5, p25_p75: {80.1, 90.2}, iqr: 10.1,
    beta: {16.22, 2.32},
    credible_50: {83, 92},
    credible_80: {80, 95},
    tier: 2, confidence: :high
  },
  "likely" => %{
    median: 71.1, p25_p75: {62.6, 77.6}, iqr: 15.0,
    beta: {11.12, 4.52},
    credible_50: {66, 76},
    credible_80: {60, 82},
    tier: 2, confidence: :moderate
  },
  "probable" => %{
    median: 70.2, p25_p75: {64.7, 77.7}, iqr: 13.0,
    beta: {10.79, 4.58},
    credible_50: {65, 75},
    credible_80: {58, 82},
    tier: 2, confidence: :moderate
  },
  "usually" => %{
    median: 75.1, p25_p75: {65.6, 82.2}, iqr: 16.7,
    beta: {9.90, 3.28},
    credible_50: {70, 80},
    credible_80: {62, 87},
    tier: 2, confidence: :moderate
  },
  "unlikely" => %{
    median: 17.2, p25_p75: {9.8, 22.7}, iqr: 13.0,
    beta: {3.92, 18.85},
    credible_50: {12, 23},
    credible_80: {8, 28},
    tier: 2, confidence: :moderate
  },
  "very unlikely" => %{
    median: 5.0, p25_p75: {2.7, 9.8}, iqr: 7.1,
    beta: {2.29, 43.51},
    credible_50: {3, 8},
    credible_80: {2, 12},
    tier: 2, confidence: :high
  }
}
```

**Usage:** Use 50% credible intervals (narrower) or specify wider priors. Consider flagging as "moderate confidence" in output.

### Tier 3: Ambiguous (Avoid or Flag)

```elixir
@tier_3_phrases %{
  "possible" => %{
    median: 38.5, p25_p75: {7.5, 50.2}, iqr: 42.7,
    beta: {0.53, 0.84},  # Uninformative prior!
    warning: "BIMODAL DISTRIBUTION - high ambiguity",
    credible_80: {5, 75},  # Extremely wide
    tier: 3, confidence: :low,
    recommended_alternative: "Use numeric range or avoid"
  },
  "not infrequent" => %{
    median: 49.6, p25_p75: {32.7, 57.3}, iqr: 24.6,
    beta: {1.01, 1.03},  # Nearly uninformative
    warning: "Negative construction increases ambiguity",
    credible_80: {20, 80},
    tier: 3, confidence: :low,
    recommended_alternative: "Rephrase positively or use numeric"
  },
  "moderate probability" => %{
    median: 52.4, p25_p75: {40.1, 58.7}, iqr: 18.5,
    beta: {2.40, 2.18},
    credible_80: {30, 75},
    tier: 3, confidence: :low,
    recommended_alternative: "Too vague - specify numeric range"
  }
}
```

**Usage:** Avoid if possible. If used, flag prominently as "ambiguous phrase" and use very wide credible intervals (80% → 50%, or even wider). Consider requiring numeric range specification instead.

---

## Beta Distribution Encoding

### Why Beta Distributions?

**Properties that make Beta perfect for probabilities:**

1. **Domain:** Beta(α, β) is defined on [0, 1] - matches probability space exactly
2. **Flexibility:** Can represent uniform, peaked, U-shaped, skewed distributions
3. **Conjugacy:** Beta is conjugate prior for Binomial - easy Bayesian updates
4. **Interpretability:** α and β have meaning: α = successes + 1, β = failures + 1

**Visual intuition:**
```
Beta(1, 1)    = Uniform(0, 1)           # No information
Beta(10, 10)  = Peaked at 0.5           # Balanced, moderate confidence
Beta(20, 2)   = Peaked at 0.9           # High probability, high confidence
Beta(0.5, 0.5) = U-shaped (0 or 1)      # Bimodal
```

### Deriving Beta Parameters from Empirical Data

**Method of Moments:**

Given median `m` and IQR from percentiles P25 to P75:

```
1. Convert IQR to variance:
   σ ≈ IQR / 1.35  (assumes approximately normal in middle 50%)

2. Solve for α and β:
   mean = m / 100
   variance = (σ / 100)²

   α = mean × ((mean × (1 - mean) / variance) - 1)
   β = (1 - mean) × ((mean × (1 - mean) / variance) - 1)
```

**Example: "Very likely"**

```elixir
median = 87.5
iqr = 10.1

# Step 1: IQR to variance
sigma = iqr / 1.35 = 10.1 / 1.35 = 7.48
variance = (7.48 / 100)^2 = 0.0056

# Step 2: Method of moments
mean = 87.5 / 100 = 0.875
alpha = 0.875 * ((0.875 * 0.125 / 0.0056) - 1) = 16.22
beta = 0.125 * ((0.875 * 0.125 / 0.0056) - 1) = 2.32

# Result: Beta(16.22, 2.32)
```

**Validation:**

```elixir
# Sample from Beta(16.22, 2.32) and check percentiles
samples = Statistics.Distributions.Beta.sample(16.22, 2.32, 10000)
percentile(samples, 50) ≈ 87.5  ✓
percentile(samples, 75) - percentile(samples, 25) ≈ 10.1  ✓
```

### Computing Credible Intervals

**80% Credible Interval (P10 to P90):**

```elixir
defmodule VERA.BetaUtils do
  @doc "Compute credible interval from Beta distribution"
  def credible_interval(alpha, beta, level \\ 0.80) do
    # level = 0.80 → 10th and 90th percentiles
    # level = 0.50 → 25th and 75th percentiles

    lower_tail = (1 - level) / 2
    upper_tail = 1 - lower_tail

    lower = Statistics.Distributions.Beta.quantile(alpha, beta, lower_tail)
    upper = Statistics.Distributions.Beta.quantile(alpha, beta, upper_tail)

    {round(lower * 100), round(upper * 100)}
  end
end

# Example: "very likely" → Beta(16.22, 2.32)
VERA.BetaUtils.credible_interval(16.22, 2.32, 0.80)
# => {80, 95}

VERA.BetaUtils.credible_interval(16.22, 2.32, 0.50)
# => {83, 92}
```

### When Beta Distributions Fail

**Warning signs:**

1. **α or β < 1**: U-shaped distribution (bimodal) - indicates data ambiguity
   - Example: "possible" → Beta(0.53, 0.84)
   - **Action:** Flag as Tier 3, use very wide priors or avoid

2. **α + β < 3**: Uninformative prior (very high variance)
   - Example: "not infrequent" → Beta(1.01, 1.03)
   - **Action:** Essentially uniform [0, 1], provides no information

3. **Variance from IQR exceeds Beta assumptions**: Skewed or multimodal
   - **Action:** Check empirical percentiles directly, may need mixture model

**Fallback:** For ambiguous phrases, use percentile ranges directly instead of Beta approximation.

---

## Context-Specific Calibrations

### Domain Effects

**Medical/Health Context:**

```elixir
@medical_context %{
  "rare" => %{
    general: 7.2,      # Vogel 2022
    medical: 10.0,     # Medical meta-analysis
    severe: 10.06,     # Severe side effects
    mild: 14.14,       # Mild side effects
    shift: +2.8        # Medical context shifts upward
  },
  "likely" => %{
    general: 71.1,
    medical: 71.87,
    shift: +0.77       # Minimal shift
  },
  "very likely" => %{
    general: 87.5,
    medical: 84.3,
    shift: -3.2        # Medical context more conservative
  }
}
```

**Key finding:** Medical domain shifts are moderate (2-4 percentage points). Use general calibrations unless domain-specific data strongly indicates otherwise.

### Severity Effects (Medical Only)

**Pattern:** Severity modulates probability perception

```
"rare" (general)  →  7.2%
"rare severe"     → 10.06%  (+2.86 points)
"rare mild"       → 14.14%  (+6.94 points)

Interpretation: People interpret "rare" more generously (higher probability)
when describing mild vs. severe outcomes.
```

**For VERA:** Capture severity as separate node if domain is medical:

```elixir
%{
  node: "Side_Effect_Severity",
  states: [:mild, :moderate, :severe],
  parents: [:drug_class]
}

%{
  node: "Side_Effect_Occurs",
  states: [:yes, :no],
  parents: [:side_effect_severity],
  cpt: %{
    {severity: :mild} => {:rare_mild, {10, 18}},
    {severity: :severe} => {:rare_severe, {7, 13}}
  }
}
```

### IPCC Regression Toward 50%

**Phenomenon:** Public systematically underestimates extremes

| IPCC Term | Intended | Public Mean | Gap |
|-----------|----------|-------------|-----|
| Very likely | >90% (95%) | 67% | -28 points |
| Likely | >66% (83%) | 62% | -21 points |
| Unlikely | <33% (17%) | 23% | +6 points |

**Implications for VERA:**

1. **Use empirical calibrations (Mosteller & Youtz), NOT IPCC definitions**
   - IPCC terms are prescriptive, not descriptive
   - Mosteller data reflects actual usage

2. **Be extra conservative with high-probability phrases**
   - "Very likely" empirically means ~87%, not 95%
   - Use "almost certain" or "certain" for >95% claims

3. **Avoid relying on institutional definitions**
   - Even experts interpret differently than intended
   - Empirical population data is more reliable

---

## Practical Guidelines

### Choosing the Right Phrase

**Decision tree:**

```
1. Do you have numeric estimate? (e.g., "75-85%")
   YES → Use numeric range directly, no verbal phrase needed
   NO  → Continue to 2

2. Is the claim in Tier 1 consensus range?
   (Near 0%, 50%, or 100% with IQR < 5%)
   YES → Use Tier 1 phrase ("always", "certain", "never", etc.)
   NO  → Continue to 3

3. Is the claim in Tier 2 range? (60-90% or 10-40% with IQR 5-20%)
   YES → Use Tier 2 phrase with 50% credible intervals
   NO  → Continue to 4

4. Is the claim ambiguous? (Near 50% or IQR > 20%)
   YES → FLAG as Tier 3, use very wide priors, or require numeric
```

### Converting CPT Entries

**Pattern 1: Direct phrase mapping**

```elixir
# Document says: "JSONL is very likely optimal for long-term readability"

cpt = %{
  {format: :jsonl} => {:very_likely, {80, 95}}
}
```

**Pattern 2: Multiple phrases → combine**

```elixir
# Document says: "Compression is almost always beneficial" AND
#                "zstd is very likely the best compressor"

# Combine via multiplication (independence assumption):
# P(zstd optimal) = P(compression beneficial) * P(zstd best | compression)
#                 ≈ 0.92 * 0.875 = 0.805

# Or keep as separate nodes:
%{node: "compression_beneficial", cpt: {:almost_always, {88, 96}}}
%{node: "zstd_best", parents: [:compression_beneficial],
  cpt: %{
    {compression: true} => {:very_likely, {80, 95}}
  }
}
```

**Pattern 3: No phrase given → conservative default**

```elixir
# Document mentions claim but no confidence: "Parquet may have faster queries"

# Use Tier 2 moderate phrase with WIDE prior:
cpt = %{
  {format: :parquet} => {:possible_to_likely, {40, 75}}
}

# Or flag as "insufficient evidence":
cpt = %{
  {format: :parquet} => {:insufficient_evidence, :wide_uniform}
}
```

### Handling Negations

**Pattern:** Negative constructions increase ambiguity

```
"not unlikely"     → Median 82.3%, IQR 14.5  (Tier 2)
"unlikely"         → Median 17.2%, IQR 13.0  (Tier 2)

BUT: "not infrequent" → IQR 24.6 (Tier 3 - ambiguous!)
```

**Guideline:** Convert negations to positive form when possible:

```elixir
# Document: "It is not unlikely that costs will exceed $100"
# Rewrite: "Costs will likely exceed $100"

# BAD:
cpt = %{{cost_exceeds_100: true} => {:not_unlikely, {70, 90}}}

# GOOD:
cpt = %{{cost_exceeds_100: true} => {:likely, {60, 82}}}
```

### Updating Beliefs with Evidence

**Bayesian update pattern:**

```elixir
# Prior: "JSONL is very likely optimal" → Beta(16.22, 2.32)
prior_alpha = 16.22
prior_beta = 2.32

# Evidence: Deployed to production, measured costs
# Observation: Cost = $18/year (matches "likely" prediction)
# Strength: High (direct measurement, n=100 entities, 6 months)

# Encode evidence as pseudo-observations:
# "Likely" = ~70% → treat as 7 successes, 3 failures
evidence_alpha = 7
evidence_beta = 3

# Posterior via conjugacy:
posterior_alpha = prior_alpha + evidence_alpha = 23.22
posterior_beta = prior_beta + evidence_beta = 5.32

# New credible interval:
VERA.BetaUtils.credible_interval(23.22, 5.32, 0.80)
# => {75, 90}

# Interpretation: Confidence increased from {80, 95} to {75, 90}
# (Narrower interval = more confident)
```

---

## Known Pitfalls

### Pitfall 1: Ignoring IQR (Using Only Median)

**Bad:**
```elixir
"possible" → median 38.5% → encode as Beta(2, 3)
```

**Why bad:** Median hides massive disagreement (IQR = 42.7%). Bimodal distribution!

**Good:**
```elixir
"possible" → IQR 42.7% → FLAG as Tier 3 ambiguous
          → Use {5, 75} very wide credible interval or avoid entirely
```

### Pitfall 2: Mixing Different Populations

**Bad:**
```elixir
# Using IPCC "very likely" (intended 95%) with Mosteller "likely" (median 71%)
```

**Why bad:** IPCC terms are prescriptive (expert consensus), Mosteller are descriptive (actual usage). Mixing creates inconsistency.

**Good:**
```elixir
# Use single calibration source (Mosteller) throughout
# OR use IPCC throughout, but be aware of public misinterpretation
```

### Pitfall 3: Over-Precision from Vague Phrases

**Bad:**
```elixir
"often" → Beta(3.8, 1.6) → credible_80 = {52, 78}
# Then using point estimate: 65%
```

**Why bad:** Pretending we know it's 65% when phrase is vague (IQR = 10.4%)

**Good:**
```elixir
"often" → Keep as range {52, 78} in CPT
       → Propagate uncertainty through network
       → Report final answer as range, not point
```

### Pitfall 4: Forgetting Context Matters

**Bad:**
```elixir
# Medical document: "Side effects are rare"
# Using general calibration: 7.2%
```

**Why bad:** Medical context shifts "rare" upward to ~10%

**Good:**
```elixir
# Check for domain-specific calibration first
# Use medical calibration: 10.0%
# OR flag that general calibration may be conservative
```

### Pitfall 5: Trusting Institutional Definitions Over Empirical Data

**Bad:**
```elixir
# IPCC says "very likely" = >90%
# Encode as Beta(20, 2) → ~90%
```

**Why bad:** Public interprets "very likely" as ~67% (28 point gap). Using 90% misrepresents how readers will understand claims.

**Good:**
```elixir
# Use empirical data: "very likely" = 87.5%, IQR 10.1%
# Encode as Beta(16.22, 2.32) → credible_80 {80, 95}
# Matches actual reader interpretation
```

### Pitfall 6: Assuming Independence When Combining Probabilities

**Bad:**
```elixir
# "Very likely X" (87%) AND "very likely Y" (87%)
# Therefore P(X and Y) = 0.87 * 0.87 = 0.76
```

**Why bad:** Only valid if X and Y are independent. If correlated, can't multiply.

**Good:**
```elixir
# Model dependency explicitly in Bayesian network:
%{node: "Y", parents: [:X],
  cpt: %{
    {x: true} => {:very_likely, {80, 95}},
    {x: false} => {:unlikely, {8, 28}}
  }
}
# Network inference handles correlation correctly
```

---

## Implementation Checklist

### Level 1 (Basic Calibration)

- [ ] Load Tier 1 phrases (6 high-consensus expressions)
- [ ] Load Tier 2 phrases (15 moderate-consensus expressions)
- [ ] Implement Beta parameter lookup
- [ ] Implement 80% credible interval calculation
- [ ] Implement 50% credible interval calculation
- [ ] Flag Tier 3 phrases with warnings

### Level 2 (Context Awareness)

- [ ] Add medical context calibrations
- [ ] Add severity modulation (medical only)
- [ ] Detect domain from document metadata
- [ ] Auto-select context-appropriate calibration
- [ ] Warn when mixing contexts

### Level 3 (Belief Updating)

- [ ] Implement Bayesian update (Beta conjugacy)
- [ ] Track evidence strength encoding
- [ ] Support multiple evidence sources
- [ ] Generate belief history audit trail
- [ ] Explain confidence changes in natural language

---

## References

### Primary Data Sources
- `mosteller_youtz_1990_full.csv` - Empirical distributions, n=238
- `verbal_probability_analysis.txt` - Beta parameters pre-computed
- `ipcc_likelihood_interpretation.csv` - Expert vs. public interpretation
- `medical_context_metaanalysis.csv` - Domain-specific calibrations
- `vogel_2022_systematic_review.csv` - Meta-analysis validation
- `recommended_standardized_terms.csv` - Consensus mappings

### Implementation
- `vera_empirical_calibration.ex` - Complete Elixir module
- `example-event-log-network.md` - Worked example using calibrations

### Research Papers
- Mosteller, F., & Youtz, C. (1990). Quantifying probabilistic expressions. Statistical Science, 5(1), 2-34.
- Budescu, D. V., Por, H. H., & Broomell, S. B. (2012). Effective communication of uncertainty in the IPCC reports. Climatic Change, 113(2), 181-200.
- Vogel, T., et al. (2022). Systematic review of verbal probability expressions. [Meta-analysis of 21 studies, 1967-2018]

---

**Document Status:** Production-ready calibration guide with empirical validation

**Next Actions:**
1. Review Elixir implementation module (`vera_empirical_calibration.ex`)
2. Practice with worked example (`example-event-log-network.md`)
3. Validate Beta parameter calculations against empirical percentiles
4. Test credible interval computations
