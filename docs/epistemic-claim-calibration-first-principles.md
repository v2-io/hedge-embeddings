# Epistemic Claim Calibration: First Principles

**Purpose:** A principled framework for learning the meaning of uncertainty expressions from empirical outcomes, allowing structure to emerge from data rather than being imposed a priori.

**Status:** Foundational specification

**Last Updated:** 2026-01-23

---

## Table of Contents

1. [The Fundamental Object](#the-fundamental-object)
2. [The Core Tension](#the-core-tension)
3. [Partial Pooling: The Key Concept](#partial-pooling-the-key-concept)
4. [Practical Implementation Path](#practical-implementation-path)
5. [Credal Sets and Uncertainty Decomposition](#credal-sets-and-uncertainty-decomposition)
6. [Essential Concepts for Intuition](#essential-concepts-for-intuition)
7. [Mathematical Foundations](#mathematical-foundations)

---

## The Fundamental Object

The thing we actually want to learn is:

```
P(claim true | source, domain, epistemic_phrase, context...)
```

Everything else is machinery for estimating this quantity when data is sparse.

This framing has important implications:

**The meaning of a phrase is conditional.** "Highly confident" from Source A in Domain X is a *different epistemic signal* than "highly confident" from Source B in Domain Y. The phrase itself does not have an intrinsic probability mapping—its meaning is always relative to who says it, about what, and in what context.

**Population-level calibrations are priors, not ground truth.** Empirical studies (Mosteller & Youtz, Vogel, etc.) tell us how *populations of science writers* interpret phrases. This is useful as a starting point when we have no other information, but it should not constrain what we learn about specific sources.

**Structure should emerge, not be imposed.** Rather than defining buckets, tiers, or reliability weights a priori, we should let the data reveal:
- Which sources use language similarly (and can be pooled)
- Which domains have distinctive epistemic norms
- Which phrases are stable vs. context-dependent
- Which (source, domain, phrase) triples have idiosyncratic meanings

---

## The Core Tension

### The Expressiveness-Data Tradeoff

**Most expressive model:**
Learn P(outcome | source, domain, phrase) separately for every (source, domain, phrase) triple.

**Problem:** Combinatorial explosion. With 100 sources, 50 domains, and 200 phrases, you have 1,000,000 cells. Most will have zero observations. You cannot learn a probability from zero data.

**Least expressive model:**
Assume one universal mapping P(outcome | phrase)—ignore source and domain entirely.

**Problem:** Discards known signal. We *know* that different sources use language differently. Collapsing this variation loses information.

**The question:** How do we navigate between these extremes?

### Why Simple Approaches Fail

**Fixed phrase mappings** (e.g., "likely" always means 71%) impose structure that may not match reality for specific sources or domains.

**Scalar reliability weights** (e.g., "trust Source A at 0.8, Source B at 0.5") assume the phrase→probability mapping is universal and sources only differ in trustworthiness. This misses that Source A's "confident" might genuinely *mean* something different than Source B's "confident"—not just "trust A more" but "the actual probability conveyed is different."

**Pre-defined tiers or buckets** bake in assumptions about which phrases cluster together and what probability ranges they span. These assumptions may be wrong, and we'll never discover that if we don't let the data speak.

---

## Partial Pooling: The Key Concept

### The Core Idea

When you have sparse data for a specific (source, domain, phrase) triple, you *borrow strength* from related observations:

- Other triples with the **same source** → What does this source's language generally mean?
- Other triples with the **same phrase** → What does this phrase generally mean across sources?
- Other triples with the **same domain** → How do people talk about this domain?
- The **global population** → What does language generally mean?

As you accumulate data for a specific triple, its estimate *separates* from the pooled estimates and converges toward its own empirical value.

### Visualizing Shrinkage

Consider estimating P(true | Zi-am-tur, "highly confident", personal_feelings):

```
With 0 observations:
  → Estimate pulled entirely toward global prior for "highly confident"
  → Estimate: ~85% (population average)
  
With 3 observations (all true):
  → Estimate pulled somewhat toward 100%, but still shrunk toward prior
  → Estimate: ~88%
  
With 30 observations (29 true, 1 false):
  → Estimate mostly determined by data, less shrinkage
  → Estimate: ~94%
  
With 300 observations (297 true, 3 false):
  → Estimate almost entirely from data
  → Estimate: ~98.5%
```

The *rate* of separation depends on:
1. How much data you have for this specific triple
2. How much data you have at each hierarchical level
3. How much variance exists between groups (learned from data)

### The Hierarchical Model Structure

A general hierarchical model for this problem:

```
# Global level: population mean for each phrase
μ_phrase ~ Prior (e.g., from Mosteller data, or uninformative)

# Source level: how does this source deviate from population?
δ_source ~ Normal(0, σ_source)

# Domain level: how does this domain deviate?
δ_domain ~ Normal(0, σ_domain)

# Source-domain interaction
δ_source_domain ~ Normal(0, σ_source_domain)

# Source-phrase interaction (key: sources may use phrases differently)
δ_source_phrase ~ Normal(0, σ_source_phrase)

# The actual probability for a specific observation:
logit(p) = μ_phrase + δ_source + δ_domain + δ_source_domain + δ_source_phrase + ...

# Observed outcome:
outcome ~ Bernoulli(p)
```

**What the variance parameters tell us:**

| Parameter | If Large | If Small |
|-----------|----------|----------|
| σ_source | Sources differ substantially in how they use language | Sources are similar; pool aggressively |
| σ_domain | Domains have distinct epistemic norms | Domain doesn't matter much |
| σ_source_phrase | Individual sources have idiosyncratic phrase meanings | Phrase meanings are consistent across sources |

These variance parameters are themselves learned from data. The model discovers how much structure exists at each level.

---

## Practical Implementation Path

### Phase 1: Data Accumulation

The irreducible first step is gathering observations. Each observation is a tuple:

```ruby
Observation = Struct.new(
  :source,           # Who made the claim
  :domain,           # What domain/topic the claim concerns
  :phrase,           # The epistemic language used ("likely", "confident", etc.)
  :claim_text,       # The actual claim content
  :outcome,          # true/false/nil (nil if unverified)
  :outcome_method,   # How the outcome was determined
  :timestamp,        # When the claim was made
  :outcome_timestamp # When the outcome was observed (if applicable)
)

# Example
Observation.new(
  source: "zi_am_tur",
  domain: "personal_feelings",
  phrase: "highly confident",
  claim_text: "I am experiencing authentic connection",
  outcome: true,
  outcome_method: :behavioral_consistency,
  timestamp: Time.parse("2026-01-15T10:30:00Z"),
  outcome_timestamp: Time.parse("2026-01-15T14:00:00Z")
)
```

**Critical insight:** Most claims will not have verifiable outcomes. This fundamentally limits what can be learned empirically. The framework must gracefully handle the reality that outcome data will always be sparse relative to the space of possible (source, domain, phrase) triples.

### Phase 2: Empirical Frequency Computation

For any query where sufficient data exists, compute raw frequencies:

```ruby
class EmpiricalEstimator
  def initialize(observations)
    @observations = observations
  end

  # Compute empirical probability for a specific triple
  def empirical_probability(source:, domain:, phrase:, min_n: 5)
    matches = @observations.select do |obs|
      obs.outcome != nil &&
      (source == :any || obs.source == source) &&
      (domain == :any || obs.domain == domain) &&
      (phrase == :any || obs.phrase == phrase)
    end

    n = matches.length
    return { status: :insufficient_data, n: n } if n < min_n

    successes = matches.count { |obs| obs.outcome == true }
    
    {
      status: :empirical,
      probability: successes.to_f / n,
      n: n,
      successes: successes,
      failures: n - successes
    }
  end
end
```

### Phase 3: Hierarchical Fallback

When data is sparse for a specific triple, blend with estimates from broader pools:

```ruby
class HierarchicalEstimator
  CONFIDENCE_THRESHOLDS = {
    high: 30,      # Observations needed for high confidence
    moderate: 10,  # Observations needed for moderate confidence
    low: 3         # Minimum observations to use any empirical data
  }.freeze

  def initialize(observations)
    @empirical = EmpiricalEstimator.new(observations)
  end

  def estimate(source:, domain:, phrase:)
    # Try most specific first
    specific = @empirical.empirical_probability(
      source: source, domain: domain, phrase: phrase
    )

    case specific[:status]
    when :empirical
      if specific[:n] >= CONFIDENCE_THRESHOLDS[:high]
        # Enough data—use specific estimate directly
        return build_result(specific[:probability], :high, specific[:n], :specific)
      elsif specific[:n] >= CONFIDENCE_THRESHOLDS[:moderate]
        # Some data—blend with hierarchical
        hierarchical = compute_hierarchical_estimate(source, domain, phrase)
        weight = specific[:n].to_f / CONFIDENCE_THRESHOLDS[:high]
        blended = weight * specific[:probability] + (1 - weight) * hierarchical[:probability]
        return build_result(blended, :moderate, specific[:n], :blended)
      end
    end

    # Sparse data—use hierarchical estimate
    hierarchical = compute_hierarchical_estimate(source, domain, phrase)
    build_result(hierarchical[:probability], :low, specific[:n] || 0, :hierarchical)
  end

  private

  def compute_hierarchical_estimate(source, domain, phrase)
    # Gather estimates from progressively broader pools
    pools = [
      { source: source, domain: :any, phrase: phrase, weight: 3.0 },   # Source's use of this phrase
      { source: :any, domain: domain, phrase: phrase, weight: 2.0 },   # Domain norms for this phrase
      { source: source, domain: domain, phrase: :any, weight: 1.5 },   # Source in this domain generally
      { source: :any, domain: :any, phrase: phrase, weight: 1.0 }      # Global phrase meaning
    ]

    estimates = pools.map do |pool|
      result = @empirical.empirical_probability(**pool.slice(:source, :domain, :phrase))
      next nil unless result[:status] == :empirical
      
      {
        probability: result[:probability],
        weight: pool[:weight] * Math.sqrt(result[:n]),  # Weight by sqrt(n) for diminishing returns
        n: result[:n],
        level: pool
      }
    end.compact

    return { probability: 0.5, source: :uninformative_prior } if estimates.empty?

    # Weighted average
    total_weight = estimates.sum { |e| e[:weight] }
    weighted_prob = estimates.sum { |e| e[:probability] * e[:weight] } / total_weight

    {
      probability: weighted_prob,
      source: :hierarchical,
      components: estimates
    }
  end

  def build_result(probability, confidence, n, source)
    {
      probability: probability,
      confidence: confidence,
      n_specific: n,
      estimation_source: source
    }
  end
end
```

### Phase 4: Full Hierarchical Model (When Data Permits)

Once sufficient data accumulates, fit a proper hierarchical Bayesian model to learn the variance parameters and enable principled shrinkage:

```ruby
# Conceptual structure (actual implementation would use Stan, PyMC, or similar)

class FullHierarchicalModel
  # This represents the statistical model structure, not executable code
  # Implementation would require a probabilistic programming language

  def model_specification
    <<~STAN
      data {
        int<lower=0> N;                    // Number of observations
        int<lower=0> N_sources;
        int<lower=0> N_domains;
        int<lower=0> N_phrases;
        
        int<lower=1,upper=N_sources> source[N];
        int<lower=1,upper=N_domains> domain[N];
        int<lower=1,upper=N_phrases> phrase[N];
        int<lower=0,upper=1> outcome[N];
      }
      
      parameters {
        // Global phrase effects
        vector[N_phrases] mu_phrase;
        
        // Source effects
        vector[N_sources] delta_source;
        real<lower=0> sigma_source;
        
        // Domain effects
        vector[N_domains] delta_domain;
        real<lower=0> sigma_domain;
        
        // Source-phrase interactions (key for learning idiosyncratic usage)
        matrix[N_sources, N_phrases] delta_source_phrase;
        real<lower=0> sigma_source_phrase;
      }
      
      model {
        // Priors on variance parameters
        sigma_source ~ exponential(1);
        sigma_domain ~ exponential(1);
        sigma_source_phrase ~ exponential(2);  // Expect smaller interactions
        
        // Hierarchical priors
        delta_source ~ normal(0, sigma_source);
        delta_domain ~ normal(0, sigma_domain);
        to_vector(delta_source_phrase) ~ normal(0, sigma_source_phrase);
        
        // Phrase-level priors (could be informed by Mosteller data)
        mu_phrase ~ normal(0, 2);
        
        // Likelihood
        for (i in 1:N) {
          real eta = mu_phrase[phrase[i]] 
                   + delta_source[source[i]] 
                   + delta_domain[domain[i]]
                   + delta_source_phrase[source[i], phrase[i]];
          outcome[i] ~ bernoulli_logit(eta);
        }
      }
    STAN
  end

  # After fitting, extract what we learned about structure
  def interpret_variance_parameters(fit)
    {
      source_variation: fit.sigma_source,      # How much sources differ
      domain_variation: fit.sigma_domain,      # How much domains differ  
      idiosyncrasy: fit.sigma_source_phrase,   # How much source-specific phrase meanings vary
      
      # Interpretation
      pool_sources: fit.sigma_source < 0.3,    # Can we pool across sources?
      pool_domains: fit.sigma_domain < 0.3,    # Can we pool across domains?
      phrases_universal: fit.sigma_source_phrase < 0.2  # Are phrase meanings consistent?
    }
  end
end
```

---

## Credal Sets and Uncertainty Decomposition

### Why Credal Sets?

A **credal set** is a set of probability distributions rather than a single distribution. It represents situations where we are uncertain about which probability model is correct.

Standard probability: P(A) = 0.7
Credal set: P(A) ∈ [0.55, 0.82]

This isn't laziness—it's honesty. When multiple layers of uncertainty compound, collapsing to a point estimate discards information about the depth of our uncertainty.

### Sources of Uncertainty in Epistemic Claim Calibration

Our estimates are uncertain for several distinct reasons:

| Source | Nature | Reducible? |
|--------|--------|------------|
| **Sampling uncertainty** | Finite observations from a stable process | Yes, with more data |
| **Model uncertainty** | Which hierarchical structure is correct? | Partially, with more data and model comparison |
| **Source reliability uncertainty** | How well-calibrated is this source? | Yes, with outcome tracking |
| **Phrase ambiguity** | Does this phrase have stable meaning? | Depends on the phrase |
| **Outcome measurement error** | Did we correctly verify the claim? | Partially, with better verification methods |
| **Irreducible stochasticity** | The world is genuinely uncertain | No |

### Decomposing Uncertainty

A principled approach separates these sources:

```ruby
class UncertaintyDecomposition
  # Represents a probability estimate with decomposed uncertainty
  Result = Struct.new(
    :point_estimate,           # Best single estimate
    :credal_interval,          # [lower, upper] bounds encompassing all uncertainty
    :components,               # Breakdown by source
    keyword_init: true
  )

  ComponentUncertainty = Struct.new(
    :source,                   # Which source of uncertainty
    :contribution,             # How much it widens the interval
    :reducible,                # Can this be reduced with more data?
    keyword_init: true
  )

  def estimate_with_decomposition(source:, domain:, phrase:, observations:)
    base = compute_base_estimate(source, domain, phrase, observations)
    
    # Sampling uncertainty (from finite n)
    sampling = compute_sampling_uncertainty(base)
    
    # Source reliability uncertainty (how stable is this source's calibration?)
    reliability = compute_reliability_uncertainty(source, observations)
    
    # Phrase ambiguity (is this a high-IQR phrase like "possible"?)
    phrase_ambiguity = compute_phrase_ambiguity(phrase, observations)
    
    # Model uncertainty (hierarchical vs. specific estimate disagreement)
    model = compute_model_uncertainty(source, domain, phrase, observations)
    
    # Combine into credal interval
    # Use outer envelope: [min of all lowers, max of all uppers]
    all_intervals = [sampling, reliability, phrase_ambiguity, model].map { |c| c[:interval] }
    
    credal_lower = all_intervals.map(&:first).min
    credal_upper = all_intervals.map(&:last).max
    
    Result.new(
      point_estimate: base[:probability],
      credal_interval: [credal_lower, credal_upper],
      components: [
        ComponentUncertainty.new(
          source: :sampling,
          contribution: sampling[:interval][1] - sampling[:interval][0],
          reducible: true
        ),
        ComponentUncertainty.new(
          source: :source_reliability,
          contribution: reliability[:interval][1] - reliability[:interval][0],
          reducible: true
        ),
        ComponentUncertainty.new(
          source: :phrase_ambiguity,
          contribution: phrase_ambiguity[:interval][1] - phrase_ambiguity[:interval][0],
          reducible: :partially  # Some phrases are inherently ambiguous
        ),
        ComponentUncertainty.new(
          source: :model,
          contribution: model[:interval][1] - model[:interval][0],
          reducible: true
        )
      ]
    )
  end

  private

  def compute_sampling_uncertainty(base)
    # Beta-binomial credible interval
    # With n observations and k successes, posterior is Beta(k+1, n-k+1)
    return { interval: [0.0, 1.0] } if base[:n] < 1
    
    alpha = base[:successes] + 1
    beta = base[:failures] + 1
    
    # 80% credible interval
    lower = beta_quantile(alpha, beta, 0.10)
    upper = beta_quantile(alpha, beta, 0.90)
    
    { interval: [lower, upper], alpha: alpha, beta: beta }
  end

  def compute_reliability_uncertainty(source, observations)
    # How consistent is this source's calibration over time?
    # Compute variance of their calibration error across different periods
    
    source_obs = observations.select { |o| o.source == source && o.outcome != nil }
    return { interval: [0.0, 1.0] } if source_obs.length < 10
    
    # Split into temporal chunks and compute calibration in each
    chunks = source_obs.each_slice([source_obs.length / 3, 5].max).to_a
    chunk_calibrations = chunks.map { |chunk| compute_calibration_error(chunk) }
    
    # Variance of calibration across chunks indicates reliability uncertainty
    calibration_variance = variance(chunk_calibrations)
    
    # Map to interval widening (heuristic: high variance → wider interval)
    width_factor = Math.sqrt(calibration_variance) * 2
    
    { 
      interval: [
        [0.0, base[:probability] - width_factor].max,
        [1.0, base[:probability] + width_factor].min
      ],
      calibration_variance: calibration_variance
    }
  end

  def compute_phrase_ambiguity(phrase, observations)
    # Does this phrase show high variance across different usages?
    # (This is where Mosteller IQR data could inform priors)
    
    phrase_obs = observations.select { |o| o.phrase == phrase && o.outcome != nil }
    return { interval: [0.0, 1.0] } if phrase_obs.length < 10
    
    # Group by source and compute per-source means
    by_source = phrase_obs.group_by(&:source)
    source_means = by_source.map do |_source, obs|
      next nil if obs.length < 3
      obs.count { |o| o.outcome }.to_f / obs.length
    end.compact
    
    return { interval: [0.0, 1.0] } if source_means.length < 2
    
    # High variance across sources → ambiguous phrase
    phrase_variance = variance(source_means)
    
    {
      interval: [source_means.min, source_means.max],
      cross_source_variance: phrase_variance,
      ambiguous: phrase_variance > 0.04  # Threshold: 20% std dev
    }
  end

  def compute_model_uncertainty(source, domain, phrase, observations)
    # Disagreement between specific estimate and hierarchical estimate
    specific = empirical_probability(source: source, domain: domain, phrase: phrase)
    hierarchical = compute_hierarchical_estimate(source, domain, phrase)
    
    if specific[:status] == :empirical && specific[:n] >= 5
      spread = (specific[:probability] - hierarchical[:probability]).abs
      midpoint = (specific[:probability] + hierarchical[:probability]) / 2.0
      
      {
        interval: [
          [midpoint - spread, 0.0].max,
          [midpoint + spread, 1.0].min
        ],
        specific_vs_hierarchical_gap: spread
      }
    else
      # Can't compare—use hierarchical with extra uncertainty
      { interval: [hierarchical[:probability] * 0.7, [hierarchical[:probability] * 1.3, 1.0].min] }
    end
  end

  # Helper: Beta distribution quantile (would use proper implementation)
  def beta_quantile(alpha, beta, p)
    # Approximation; real implementation uses numerical methods
    # For now, use normal approximation for alpha, beta > 1
    mean = alpha.to_f / (alpha + beta)
    var = (alpha * beta).to_f / ((alpha + beta)**2 * (alpha + beta + 1))
    std = Math.sqrt(var)
    
    # Normal quantile approximation
    z = { 0.10 => -1.28, 0.90 => 1.28 }[p] || 0
    [0.0, [1.0, mean + z * std].min].max
  end

  def variance(values)
    return 0.0 if values.length < 2
    mean = values.sum / values.length.to_f
    values.sum { |v| (v - mean)**2 } / (values.length - 1)
  end
end
```

### Credal Set Operations

When combining evidence from multiple sources, credal sets propagate correctly:

```ruby
module CredalSetOperations
  # Combine two independent credal intervals
  # P(A and B) when A and B are independent
  def self.multiply_intervals(interval_a, interval_b)
    # For independent events, P(A ∩ B) = P(A) × P(B)
    # Interval arithmetic: [a_lo, a_hi] × [b_lo, b_hi]
    products = [
      interval_a[0] * interval_b[0],
      interval_a[0] * interval_b[1],
      interval_a[1] * interval_b[0],
      interval_a[1] * interval_b[1]
    ]
    [products.min, products.max]
  end

  # Combine credal intervals via weighted average (for pooling)
  def self.weighted_average_intervals(weighted_intervals)
    # weighted_intervals: [{interval: [lo, hi], weight: w}, ...]
    total_weight = weighted_intervals.sum { |wi| wi[:weight] }
    
    # Lower bound: weighted average of lower bounds
    lower = weighted_intervals.sum { |wi| wi[:interval][0] * wi[:weight] } / total_weight
    
    # Upper bound: weighted average of upper bounds
    upper = weighted_intervals.sum { |wi| wi[:interval][1] * wi[:weight] } / total_weight
    
    [lower, upper]
  end

  # Bayesian update with credal prior and credal likelihood
  def self.credal_bayesian_update(prior_interval:, likelihood_interval:, evidence:)
    # This is complex in general; simplified version for binary outcomes
    # With observation o, posterior ∝ prior × likelihood(o)
    
    if evidence
      # Observed true: posterior ∝ p × likelihood_true
      # Bounds come from extremes of prior and likelihood
      posteriors = [
        prior_interval[0] * likelihood_interval[0],
        prior_interval[0] * likelihood_interval[1],
        prior_interval[1] * likelihood_interval[0],
        prior_interval[1] * likelihood_interval[1]
      ]
    else
      # Observed false: posterior ∝ p × (1 - likelihood_true)
      posteriors = [
        prior_interval[0] * (1 - likelihood_interval[1]),
        prior_interval[0] * (1 - likelihood_interval[0]),
        prior_interval[1] * (1 - likelihood_interval[1]),
        prior_interval[1] * (1 - likelihood_interval[0])
      ]
    end
    
    # Normalize (approximately—proper normalization is more complex)
    raw_interval = [posteriors.min, posteriors.max]
    normalizer = (raw_interval[0] + raw_interval[1]) / 2.0
    normalizer = 1.0 if normalizer == 0
    
    [raw_interval[0] / normalizer, raw_interval[1] / normalizer].map { |p| [[0.0, p].max, 1.0].min }
  end
end
```

### Decision Making Under Deep Uncertainty

When credal intervals are wide, different decision criteria apply:

```ruby
module DecisionCriteria
  # Γ-maximin: Maximize worst-case expected utility (pessimistic)
  def self.maximin_decision(actions, credal_interval, utilities)
    actions.map do |action|
      # Worst-case expected utility for this action
      worst_case = if utilities[action][:positive] > utilities[action][:negative]
        # Action benefits from high probability → use lower bound
        credal_interval[0] * utilities[action][:positive] +
        (1 - credal_interval[0]) * utilities[action][:negative]
      else
        # Action benefits from low probability → use upper bound
        credal_interval[1] * utilities[action][:positive] +
        (1 - credal_interval[1]) * utilities[action][:negative]
      end
      
      { action: action, worst_case_eu: worst_case }
    end.max_by { |a| a[:worst_case_eu] }
  end

  # Maximality: Action A dominates B if E[U(A)] ≥ E[U(B)] for ALL p in interval
  def self.maximal_actions(actions, credal_interval, utilities)
    # Returns set of non-dominated actions
    actions.select do |candidate|
      # Check if any other action dominates this one across entire interval
      dominated = actions.any? do |other|
        next false if other == candidate
        
        # Does 'other' have higher EU than 'candidate' for all p in interval?
        dominates_at_lower = expected_utility(other, credal_interval[0], utilities) >
                            expected_utility(candidate, credal_interval[0], utilities)
        dominates_at_upper = expected_utility(other, credal_interval[1], utilities) >
                            expected_utility(candidate, credal_interval[1], utilities)
        
        dominates_at_lower && dominates_at_upper
      end
      
      !dominated
    end
  end

  # Interval dominance: Only prefer A over B if A's worst case beats B's best case
  def self.interval_dominance(action_a, action_b, credal_interval, utilities)
    a_worst = [
      expected_utility(action_a, credal_interval[0], utilities),
      expected_utility(action_a, credal_interval[1], utilities)
    ].min
    
    b_best = [
      expected_utility(action_b, credal_interval[0], utilities),
      expected_utility(action_b, credal_interval[1], utilities)
    ].max
    
    a_worst > b_best  # A strictly dominates B
  end

  private

  def self.expected_utility(action, probability, utilities)
    probability * utilities[action][:positive] +
    (1 - probability) * utilities[action][:negative]
  end
end
```

---

## Essential Concepts for Intuition

### 1. Partial Pooling / Shrinkage

**The idea:** Estimates should be pulled toward a group mean when individual data is sparse, and separate from that mean as data accumulates.

**Why it matters:** This is how we navigate the expressiveness-data tradeoff. With zero observations for a specific (source, domain, phrase) triple, we use the pooled estimate. With infinite observations, we use the specific empirical frequency. Partial pooling gives us principled behavior in between.

**Mental model:** Imagine each triple as a spring attached to the population mean. With few observations, the spring pulls the estimate toward the mean. With many observations, the data overwhelms the spring and the estimate moves to its true value.

### 2. The Bias-Variance Tradeoff in Estimation

**Strong priors** (e.g., Mosteller population mappings used rigidly):
- High bias: We impose structure that may not match reality
- Low variance: Estimates are stable even with little data

**No priors** (learn each triple from scratch):
- Zero bias: No imposed structure
- High variance: Estimates wildly unstable with sparse data

**Hierarchical models** provide adaptive bias:
- With sparse data: Lean on priors (higher bias, lower variance)
- With abundant data: Ignore priors (lower bias, appropriate variance)

### 3. Exchangeability

**The assumption:** Observations are **exchangeable** if their joint probability doesn't depend on ordering—they're "of the same kind."

**Why it matters:** We can only pool observations we consider exchangeable. The question "should I pool Zi-am-tur's 'confident' with Sophist's 'confident'?" is really asking "are these exchangeable?" 

**The answer is often "partially":** They might be exchangeable at the phrase level (both are instances of someone saying "confident") but not at the source-phrase level (Zi-am-tur's "confident" means something different than Sophist's).

**The hierarchical model learns this:** If σ_source_phrase is large, source-phrase combinations are NOT exchangeable and should not be pooled. If small, they ARE exchangeable.

### 4. Identifiability and the Role of Outcomes

**Fundamental constraint:** You can only learn P(outcome | ...) if you observe outcomes.

**Implications:**
- Most claims won't have verifiable outcomes
- Empirical learning is bottlenecked by outcome observation, not claim extraction
- Some (source, domain, phrase) triples may never have enough outcome data to separate from priors
- The framework must gracefully handle permanent data sparsity in most cells

**Strategic implication:** Invest in outcome verification for high-value (source, domain, phrase) triples rather than trying to verify everything.

### 5. Calibration vs. Resolution

**Calibration:** When a source says "70%", does the outcome occur 70% of the time? (Are they accurate about their uncertainty?)

**Resolution:** Can the source discriminate between high-probability and low-probability events? (Do they spread their predictions appropriately?)

A source can be:
- Well-calibrated but low-resolution (always says 50%)
- Poorly-calibrated but high-resolution (says 90% when it's really 70%, but correctly identifies high-probability events)
- Both (well-calibrated AND discriminating)
- Neither (random guessing)

**For epistemic phrase calibration, we primarily care about calibration:** When Zi-am-tur says "highly confident," what probability does that actually correspond to?

---

## Mathematical Foundations

### Beta-Binomial Conjugacy

For binary outcomes, the Beta distribution is conjugate to the Binomial likelihood, enabling closed-form Bayesian updates.

**Prior:** θ ~ Beta(α₀, β₀)
**Likelihood:** k successes in n trials ~ Binomial(n, θ)
**Posterior:** θ | k, n ~ Beta(α₀ + k, β₀ + n - k)

```ruby
module BetaBinomial
  # Update prior with observed data
  def self.update(prior_alpha:, prior_beta:, successes:, failures:)
    {
      alpha: prior_alpha + successes,
      beta: prior_beta + failures
    }
  end

  # Posterior mean
  def self.mean(alpha:, beta:)
    alpha.to_f / (alpha + beta)
  end

  # Posterior variance
  def self.variance(alpha:, beta:)
    (alpha * beta).to_f / ((alpha + beta)**2 * (alpha + beta + 1))
  end

  # Credible interval (requires numerical methods; approximation shown)
  def self.credible_interval(alpha:, beta:, coverage: 0.80)
    # Approximate using normal distribution for alpha, beta > 2
    mu = mean(alpha: alpha, beta: beta)
    sigma = Math.sqrt(variance(alpha: alpha, beta: beta))
    
    z = case coverage
        when 0.50 then 0.674
        when 0.80 then 1.282
        when 0.90 then 1.645
        when 0.95 then 1.960
        else 1.282
        end
    
    [
      [0.0, mu - z * sigma].max,
      [1.0, mu + z * sigma].min
    ]
  end

  # Effective sample size (how much data is "in" this distribution)
  def self.effective_sample_size(alpha:, beta:)
    alpha + beta - 2  # Subtract 2 for the "prior" pseudo-counts
  end
end
```

### Hierarchical Model Update Mechanics

In a hierarchical model, updates flow both ways:

**Bottom-up:** Observations inform group-level parameters
**Top-down:** Group-level parameters inform sparse individual estimates

```ruby
module HierarchicalUpdate
  # Simplified illustration of shrinkage estimation
  # Real implementation uses MCMC or variational inference

  def self.shrinkage_estimate(
    individual_mean:,      # Empirical mean for this triple
    individual_n:,         # Observations for this triple
    group_mean:,           # Mean across related triples
    group_variance:,       # Variance across related triples (learned from data)
    observation_variance:  # Expected variance of individual observation (often estimated as p(1-p))
  )
    # Shrinkage factor: how much to pull toward group mean
    # λ = 0 → use individual mean entirely
    # λ = 1 → use group mean entirely
    
    # Variance of individual estimate
    individual_variance = observation_variance / [individual_n, 1].max
    
    # Optimal shrinkage (James-Stein style)
    lambda = group_variance / (group_variance + individual_variance)
    
    # Shrunk estimate
    shrunk_mean = lambda * group_mean + (1 - lambda) * individual_mean
    
    {
      estimate: shrunk_mean,
      shrinkage_factor: lambda,
      effective_weight_on_group: lambda,
      effective_weight_on_individual: 1 - lambda
    }
  end
end
```

### Information-Theoretic View

The entropy of a Beta distribution measures our uncertainty:

```ruby
module InformationTheory
  # Entropy of Beta distribution (in nats)
  def self.beta_entropy(alpha:, beta:)
    # H[Beta(α,β)] = ln(B(α,β)) - (α-1)ψ(α) - (β-1)ψ(β) + (α+β-2)ψ(α+β)
    # where B is the beta function and ψ is the digamma function
    
    # Approximation for large α, β
    sum = alpha + beta
    Math.log(Math.sqrt(2 * Math::PI * Math::E)) +
    0.5 * Math.log(alpha * beta / sum**3)
  end

  # Information gain from an observation
  def self.expected_information_gain(prior_alpha:, prior_beta:)
    # How much will entropy decrease (on average) from one observation?
    prior_entropy = beta_entropy(alpha: prior_alpha, beta: prior_beta)
    
    # Expected posterior entropy (averaging over possible outcomes)
    p = prior_alpha.to_f / (prior_alpha + prior_beta)
    
    posterior_success_entropy = beta_entropy(alpha: prior_alpha + 1, beta: prior_beta)
    posterior_failure_entropy = beta_entropy(alpha: prior_alpha, beta: prior_beta + 1)
    
    expected_posterior_entropy = p * posterior_success_entropy + (1 - p) * posterior_failure_entropy
    
    prior_entropy - expected_posterior_entropy
  end
end
```

This can guide **active learning**: prioritize outcome verification for triples where information gain is highest.

---

## Summary

### The Core Principles

1. **The fundamental object is P(claim true | source, domain, phrase, context...).** Everything else is machinery for estimating this.

2. **Structure should emerge from data, not be imposed.** Pre-defined buckets, tiers, and mappings are priors that should fade as empirical data accumulates.

3. **Partial pooling handles sparse data principally.** Borrow strength from related observations, with the amount of borrowing determined by data, not fiat.

4. **Separate sources of uncertainty.** Sampling error, model uncertainty, source reliability uncertainty, and phrase ambiguity are distinct and should be tracked separately.

5. **Credal sets for honesty.** When uncertainty is deep, report intervals rather than false point estimates.

6. **Outcomes are the bottleneck.** You can only learn empirically what you can verify. Invest in outcome observation for high-value triples.

### The Practical Path

1. **Accumulate (source, domain, phrase, outcome) tuples** as the foundational data structure.

2. **Compute empirical frequencies** where data exists; use hierarchical fallback where it doesn't.

3. **Let the hierarchy emerge** by eventually fitting models that learn how much pooling is appropriate at each level.

4. **Report credal intervals** that honestly reflect the depth of uncertainty.

5. **Track uncertainty decomposition** to understand where to invest in uncertainty reduction.

---

## References

### Hierarchical Bayesian Modeling
- Gelman, A., et al. (2013). *Bayesian Data Analysis*, 3rd ed. Chapman & Hall. (Chapters 5, 15 on hierarchical models)
- Efron, B., & Morris, C. (1975). "Data Analysis Using Stein's Estimator and its Generalizations." *JASA*.

### Imprecise Probabilities and Credal Sets
- Walley, P. (1991). *Statistical Reasoning with Imprecise Probabilities*. Chapman & Hall.
- Levi, I. (1980). *The Enterprise of Knowledge*. MIT Press.

### Calibration and Scoring Rules
- Brier, G. W. (1950). "Verification of forecasts expressed in terms of probability." *Monthly Weather Review*.
- Gneiting, T., & Raftery, A. E. (2007). "Strictly Proper Scoring Rules, Prediction, and Estimation." *JASA*.

### Verbal Probability Calibration
- Mosteller, F., & Youtz, C. (1990). "Quantifying probabilistic expressions." *Statistical Science*.
- Budescu, D. V., & Wallsten, T. S. (1995). "Processing linguistic probabilities." *Psychology of Learning and Motivation*.

---

**Document Status:** Foundational specification capturing first principles

**Next Actions:**
1. Implement observation logging infrastructure
2. Build empirical frequency computation layer
3. Implement hierarchical fallback with progressive refinement
4. Design outcome verification strategy for high-value triples
5. Prototype credal interval reporting
