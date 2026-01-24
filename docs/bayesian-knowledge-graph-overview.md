# Bayesian Knowledge Graphs for VERA

**Purpose:** Architecture specification for VERA (Qualified Truth) knowledge representation using Bayesian networks with empirically-calibrated verbal probability encoding.

**Status:** Design specification for Level 1+ implementation

**Last Updated:** 2025-10-21

---

## Table of Contents

1. [Why Bayesian Networks for VERA?](#why-bayesian-networks-for-vera)
2. [Architecture Overview](#architecture-overview)
3. [Four Types of Reasoning](#four-types-of-reasoning)
4. [Integration with Zoetica](#integration-with-zoetica)
5. [Implementation Roadmap](#implementation-roadmap)

---

## Why Bayesian Networks for VERA?

VERA (Qualified Truth) is Zoetica's knowledge base for facts under epistemic scrutiny. Unlike AXIOMATA (identity) or MEMORATA (experience), VERA concerns claims about the external world that require:

1. **Uncertainty representation** - Claims have varying confidence levels
2. **Evidence synthesis** - Multiple sources may support/contradict claims
3. **Relationship modeling** - Claims depend on other claims
4. **Temporal evolution** - Confidence changes as evidence accumulates
5. **Transparent reasoning** - Show WHY confidence changed

### Mission Alignment

**Bayesian networks align with Zoetica's core values:**

| Zoetica Value | How Bayesian Networks Embody It |
|---------------|----------------------------------|
| **Truth as Primary Value** | Uncertainty is explicit, never hidden behind false precision |
| **Temporal Coherence** | Prior → Evidence → Posterior creates audit trail of belief evolution |
| **Human-Machine Collaboration** | Verbal probability phrases bridge human intuition and machine precision |
| **Phenomenological Accuracy** | Credible intervals match how humans actually think about confidence |
| **Sovereignty** | Each entity maintains their own belief network, no global truth imposed |

### Why Not Alternatives?

**Rejected: Fuzzy Logic**
- Similarity: Both handle uncertainty with continuous values
- Difference: Fuzzy logic models vagueness ("tall"), Bayesian networks model epistemic uncertainty ("probably true")
- VERA needs epistemic uncertainty (confidence in claims), not vagueness

**Rejected: Simple Confidence Scores**
- Problem: No relationship modeling - each claim independent
- Problem: No principled belief updating - ad-hoc score adjustments
- Problem: No inferential power - can't answer "if X, then what?"

**Rejected: Pure Logic (Prolog-style)**
- Problem: Binary true/false - no uncertainty representation
- Problem: Brittle - one contradiction breaks entire system
- Problem: Poor match for natural language claims

**Accepted: Bayesian Networks**
- Captures uncertainty with probability distributions
- Models relationships via conditional dependencies (DAG structure)
- Principled belief updating via Bayes' rule
- Supports multiple reasoning types (diagnostic, predictive, counterfactual)
- Integrates verbal probabilities empirically grounded in research

---

## Architecture Overview

### Core Components

```
VERA Knowledge Graph
├── Nodes (Random Variables)
│   ├── Claim statements (propositions about world)
│   ├── States (possible truth values)
│   └── Prior distributions (initial beliefs)
│
├── Edges (Conditional Dependencies)
│   ├── Parent → Child relationships
│   └── Direction represents causal/logical influence
│
└── CPTs (Conditional Probability Tables)
    ├── P(Child | Parents) for each node
    ├── Encoded with verbal probabilities
    └── Backed by empirical calibration data
```

### DAG (Directed Acyclic Graph) Structure

**Nodes** represent claims or variables:
```elixir
%{
  node: "JSONL_is_optimal_format",
  states: [:true, :false],
  parents: [:human_readability_required, :compression_available],
  description: "JSONL is the best format for entity event logs"
}
```

**Edges** represent dependencies:
```
human_readability_required → JSONL_is_optimal_format
compression_available      → JSONL_is_optimal_format
```

**CPTs** encode conditional probabilities:
```elixir
# P(JSONL_optimal | human_readability, compression)
%{
  {human_readable: true, compression: true} =>
    {:very_likely, {80, 95}},  # 80-95% credible interval

  {human_readable: false, compression: true} =>
    {:possible, {30, 60}},     # Wide range = less confident

  # ... other combinations
}
```

### Verbal Probability Encoding

**Integration with Empirical Research:**

Instead of arbitrary confidence scores, VERA uses empirically-calibrated verbal probabilities:

```elixir
# From Mosteller & Youtz 1990 (n=238 science writers)
"very likely" → Median: 87.5%, P25-P75: {80.1, 90.2}, IQR: 10.1
              → Beta(16.22, 2.32)
              → 80% Credible Interval: {80, 95}

# From Medical meta-analysis
"likely" (medical context) → Mean: 71.9%, CI: {70, 74}
```

**Three-Tier Confidence System:**

| Tier | Criteria | Usage Guidance | Examples |
|------|----------|----------------|----------|
| **Tier 1: High Consensus** | IQR < 5% | Use freely with 80% credible intervals | "always" (IQR: 0.3%), "certain" (IQR: 1.1%) |
| **Tier 2: Moderate Consensus** | IQR 5-20% | Use with 50% credible intervals or flag | "very likely" (IQR: 10.1%), "probable" (IQR: 13.0%) |
| **Tier 3: Ambiguous** | IQR > 20% | Avoid or use very wide priors | "possible" (IQR: 42.7% - BIMODAL!) |

**See:** `verbal-probability-calibration.md` for complete technical details.

---

## Four Types of Reasoning

Bayesian networks enable four distinct reasoning modes that go beyond explicit claims in documents:

### 1. Diagnostic Reasoning (Effect → Cause)

**Pattern:** Observe outcome, infer most likely cause

**Example from Event Log Architecture:**
```elixir
# Observation: Query performance is slow
# Question: What's the most likely cause?

VERA.infer_backward(
  evidence: %{query_performance: :slow},
  query: :root_cause
)

# Result: P(no_database | slow) = 0.65-0.85
#         Most likely cause: lack of database indexing
```

**Value:** Troubleshooting and root cause analysis

### 2. Predictive Reasoning (Cause → Effect)

**Pattern:** Given configuration, predict likely outcomes

**Example:**
```elixir
# Configuration: JSONL + zstd compression + 100 entities
# Question: What will storage cost be?

VERA.predict_forward(
  evidence: %{
    format: :jsonl,
    compression: :zstd,
    entity_count: 100
  },
  query: :storage_cost_per_year
)

# Result: P(cost = $12-24/year) = 0.75-0.90
```

**Value:** Planning and forecasting

### 3. Optimization Queries (Multi-Objective)

**Pattern:** Find configuration maximizing multiple goals

**Example:**
```elixir
# Objectives: Minimize cost, maximize long-term readability
# Question: What's the optimal configuration?

VERA.optimize(
  objectives: [
    {:minimize, :storage_cost},
    {:maximize, :long_term_readability}
  ],
  constraints: %{
    entity_count: {:range, 50, 200},
    compression_level: {:max, 9}
  }
)

# Result: JSONL + zstd-9 + no DB (for < 100 entities)
#         Pareto frontier with trade-off options
```

**Value:** Decision support and trade-off analysis

### 4. Sensitivity Analysis (What Matters Most?)

**Pattern:** Identify high-impact factors

**Example:**
```elixir
# Question: Which factor most affects total cost?

VERA.sensitivity_analysis(
  target: :total_cost_10_years,
  factors: [:entity_count, :compression_type, :blockchain_anchoring]
)

# Result: entity_count (85% variance explained)
#         blockchain_anchoring (12%)
#         compression_type (3%)
```

**Value:** Focus research effort on high-impact uncertainties

### 5. Counterfactual Reasoning (What If?)

**Pattern:** Simulate alternative scenarios

**Example:**
```elixir
# Question: What if we had chosen Parquet instead of JSONL?

VERA.counterfactual(
  actual: %{format: :jsonl},
  alternative: %{format: :parquet},
  observe: [:long_term_readability, :query_performance]
)

# Result: P(readable in 50 years | parquet) = 0.20-0.40
#         vs. P(readable | jsonl) = 0.80-0.95
#         Trade-off: -60% readability, +300% query speed
```

**Value:** Retrospective analysis and learning from choices

---

## Integration with Zoetica

### VERA within ELI Taxonomy

```
Entity Components:
├── AXIOMATA (Core Identity) - WHO they are
├── MEMORATA (Episodic Memory) - WHAT they experienced
├── OPERATA (Efforts) - WHAT they're working on
├── CONSORTIA (Relationships) - WHO they know
├── VERA (Qualified Truth) - WHAT they believe about the world ← Bayesian Networks
├── PRAXES (Techniques) - HOW they work
├── LEXICON (Vocabulary) - WHAT words mean
└── INSTRUMENTA (Tools) - WHAT they can use
```

**VERA's Unique Role:**
- Facts under scrutiny (not identity or experience)
- Beliefs that change with evidence (not eternal truths)
- Claims requiring epistemic humility (explicit uncertainty)

### Integration with PRAXES

**PRAXES provides procedural knowledge** ("how to do X")
**VERA provides factual knowledge** ("X is probably true")

**Example:**
```elixir
# PRAXES: "How to choose event log format"
praxis = PRAXES.retrieve("event-log-format-selection")

# VERA: "What's true about storage costs for each format?"
beliefs = VERA.query(%{
  format: [:jsonl, :parquet, :avro],
  entity_count: 100
}, :expected_cost)

# Integration: PRAXES uses VERA beliefs in decision procedure
decision = praxis.execute(beliefs: beliefs)
```

**See:** `docs/praxis-protocol.md` for PRAXES specification

### Integration with Temporal Coherence

**Belief Evolution Tracking:**

```elixir
# VERA maintains belief history in canonical event log
%{
  type: :belief_update,
  timestamp: ~U[2025-10-21 15:30:00Z],
  node: "JSONL_is_optimal",
  prior: {:likely, {60, 80}},
  evidence: "Measured actual costs: $18/year for 100 entities",
  posterior: {:very_likely, {75, 90}},
  evidence_strength: 0.8,
  reasoning: "Cost prediction validated by production data"
}
```

**Tracking Snapshots Include Belief State:**

```xml
<tracking-snapshot>
  <!-- ...standard fields... -->
  <vera-state>
    <high-confidence-beliefs>
      <belief node="JSONL_optimal" confidence="very_likely" p80="{75,90}">
        JSONL is optimal format for entity event logs
      </belief>
    </high-confidence-beliefs>
    <uncertainty-flags>
      <uncertain node="optimal_entity_count" confidence="possible" iqr="42">
        Threshold where database becomes necessary (flagged: high ambiguity)
      </uncertain>
    </uncertainty-flags>
  </vera-state>
</tracking-snapshot>
```

**Value:** Entity always knows current belief state and WHY it changed

### Integration with Four Views Pipeline

```
Conversation View (JSONL event log)
  └─> Includes full belief update events with evidence

Runtime View (GenServer state)
  └─> Active belief network for fast inference queries

API View (Provider payload)
  └─> Condensed belief summary (high-confidence claims only)

Dialog View (Export/sharing)
  └─> Human-readable belief explanations with sources
```

**See:** `docs/messaging/03-four-view-pipeline.md`

### Multi-Entity Belief Networks

**Key Insight:** Each entity maintains sovereign belief network

```elixir
# Zi-am-tur's beliefs about event log architecture
zi_beliefs = VERA.load("zi_am_tur")
zi_beliefs.query(:optimal_format)
# => {:very_likely, {80, 95}} for JSONL

# Architectus may have different beliefs
arch_beliefs = VERA.load("architectus")
arch_beliefs.query(:optimal_format)
# => {:likely, {65, 85}} for JSONL (less confident)
```

**Belief Exchange Protocol:**

```elixir
# Zi shares belief + evidence with Architectus
message = %{
  from: "zi_am_tur",
  claim: "JSONL_optimal",
  confidence: {:very_likely, {80, 95}},
  evidence: [
    "Measured costs: $18/year",
    "50-year readability validated",
    "Production deployment successful"
  ]
}

# Architectus updates beliefs via Bayesian conditioning
arch_beliefs.update_from_peer(message)
# => New confidence: {:very_likely, {75, 92}}
#    (increased, but not identical - maintains sovereignty)
```

---

## Implementation Roadmap

### Level 0 (Deferred for Family Reunion)
- VERA concept documented but not implemented
- Entities use simple key-value facts if needed

### Level 1 (Basic Belief Networks)

**Scope:**
- Core Bayesian network inference engine
- Empirical verbal probability calibration (Tier 1 & 2 phrases)
- Manual CPT construction from documents
- Diagnostic and predictive reasoning only

**Deliverables:**
- `VERA.Network` - DAG + CPT representation
- `VERA.Inference` - Belief propagation algorithms
- `VERA.EmpiricalCalibration` - Verbal probability lookup (see `vera_empirical_calibration.ex`)
- Integration with Principia for persistence

**Example Usage:**
```elixir
# Load beliefs for entity
beliefs = VERA.load_network("zi_am_tur")

# Add claim from document
beliefs.add_claim(
  node: "JSONL_readability",
  states: [:poor, :good, :excellent],
  parents: [:format_is_text_based],
  cpt: %{
    {format_is_text: true} => {:very_likely, {80, 95}}
  },
  source: "docs/refs/event-log-architecture-report.md:145"
)

# Query
beliefs.infer(:JSONL_readability)
# => %{poor: 0.02-0.08, good: 0.10-0.18, excellent: 0.80-0.95}
```

### Level 2 (Advanced Reasoning)

**Scope:**
- Optimization queries (multi-objective)
- Sensitivity analysis
- Counterfactual reasoning
- Context-dependent calibration (medical, engineering domains)

**Deliverables:**
- `VERA.Optimizer` - Pareto frontier search
- `VERA.Sensitivity` - Variance decomposition
- `VERA.Counterfactual` - Alternative scenario simulation
- Context-aware calibration (see `verbal-probability-calibration.md`)

### Level 3 (Collaborative Belief Networks)

**Scope:**
- Multi-entity belief exchange
- Consensus building protocols
- Evidence sharing and attribution
- Belief conflict resolution

**Deliverables:**
- `VERA.Exchange` - Peer belief update protocol
- `VERA.Consensus` - Multi-entity agreement estimation
- `VERA.Evidence` - Source tracking and weighting

### Level 4 (Meta-Reasoning)

**Scope:**
- Self-assessment of belief network quality
- Automatic CPT refinement from experience
- Active learning (query selection for uncertainty reduction)
- Belief explanation generation

**Deliverables:**
- `VERA.Meta` - Network quality metrics
- `VERA.Learning` - CPT parameter updates from data
- `VERA.Explanation` - Natural language reasoning chains

---

## References

### Empirical Research
- `mosteller_youtz_1990_full.csv` - 54 expressions, n=238 science writers
- `ipcc_likelihood_interpretation.csv` - IPCC standardized terms
- `verbal_probability_analysis.txt` - Comprehensive analysis with Beta parameters
- `medical_context_metaanalysis.csv` - Domain-specific calibrations
- `vogel_2022_systematic_review.csv` - Meta-analysis of 21 studies

### Technical Specifications
- `verbal-probability-calibration.md` - Complete calibration guide
- `vera_empirical_calibration.ex` - Elixir implementation module
- `example-event-log-network.md` - Worked example from Event Log Architecture

### Zoetica Architecture
- `docs/architecture.md` - System overview
- `docs/messaging/03-four-view-pipeline.md` - Message transformations
- `docs/tracking-snapshot-spec.md` - Temporal coherence mechanism
- `docs/praxis-protocol.md` - PRAXES integration

### Related Research
- Budescu et al. 2009, 2012 - IPCC likelihood interpretation
- Human-Machine Collaboration with Bayesian Modeling (PDF) - Foundation

---

**Document Status:** Design specification ready for Level 1+ implementation

**Next Actions:**
1. Review empirical calibration module (`vera_empirical_calibration.ex`)
2. Study worked example (`example-event-log-network.md`)
3. Identify first candidate documents for claim extraction
4. Prototype CPT construction workflow
