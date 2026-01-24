# Saliency-Weighted Embedding Retrieval

How to combine document importance ("saliency") with semantic similarity in
embedding-based retrieval systems, particularly with pgvector.

## Bayesian Grounding

The principled way to combine saliency with semantic similarity comes directly
from Bayes' rule. We want to rank documents by:

```
P(relevant | query, doc) ∝ P(query | doc, relevant) · P(relevant | doc)
```

Where:

- **P(relevant | doc)** is the saliency — "how likely is this document to be
  relevant *before* seeing any query?" This is query-independent prior
  knowledge (source authority, user-marked importance, recency, etc.).

- **P(query | doc, relevant)** is the semantic similarity — "given this
  document is relevant, how well does it match this specific query?" This is
  what cosine similarity approximates.

The posterior (what we actually want to rank by) is their product:

```
score = cosine(q, d) · saliency(d)
```

This multiplicative form is not an arbitrary heuristic — it's the
probabilistically correct way to combine a query-independent prior with a
query-dependent likelihood.

---

## Approach 1: The Magnitude Trick

**Idea:** Encode saliency in the vector's magnitude. Use inner product (not
cosine) for retrieval, so magnitude naturally contributes to the score.

### The Math

For a unit query vector **q** and a document embedding **d** stored with
magnitude equal to its saliency factor s:

```
q · d = ||q|| · ||d|| · cos(θ) = 1 · s · cos(θ) = s · cos(q, d̂)
```

The inner product decomposes into "how relevant" × "how important" — exactly
the Bayesian posterior.

### Implementation (pgvector)

```sql
CREATE TABLE chunks (
    id SERIAL PRIMARY KEY,
    content TEXT,
    embedding vector(768),  -- stored UN-normalized, magnitude = saliency
    saliency FLOAT NOT NULL DEFAULT 1.0  -- kept for reference/updates
);

-- Index for inner product search:
CREATE INDEX ON chunks USING hnsw (embedding vector_ip_ops);

-- At insert time, scale the unit embedding by saliency:
INSERT INTO chunks (content, embedding, saliency)
VALUES (
    'important text',
    (unit_embedding * sqrt(saliency))::vector,  -- sqrt dampens the effect
    saliency_value
);

-- Query (query vector should be unit-normalized):
SELECT content,
       (embedding <#> query_embedding) * -1 AS score
FROM chunks
ORDER BY embedding <#> query_embedding
LIMIT 10;
```

Note: pgvector's `<#>` operator returns *negative* inner product (for ASC
ordering compatibility), hence the `* -1` for human-readable scores.

### Saliency Range

The saliency range determines how much importance can override relevance:

- **[0.8, 1.2]** — gentle nudge, saliency is a tiebreaker
- **[0.5, 2.0]** — moderate influence
- **[0.1, 10.0]** — saliency can dominate (usually too aggressive)

Using `sqrt(saliency)` instead of raw saliency compresses the range further
(a saliency of 4.0 becomes a magnitude of 2.0).

### Tradeoffs

| Pro | Con |
|-----|-----|
| Fully index-accelerated (HNSW) | Can't easily separate cosine from saliency at query time |
| Single vector, no post-processing | Must reindex if saliency changes |
| Mathematically clean | Wildly varying magnitudes can degrade HNSW recall |
| No extra storage | Loses pure cosine distance if needed later |

---

## Approach 2: Separate Column with Post-Hoc Reranking

**Idea:** Store saliency as a separate float column. Retrieve candidates by
pure cosine (index-accelerated), then rerank with the combined score.

### Implementation (pgvector)

```sql
CREATE TABLE chunks (
    id SERIAL PRIMARY KEY,
    content TEXT,
    embedding vector(768),  -- unit-normalized
    saliency FLOAT NOT NULL DEFAULT 1.0
);

-- Index for cosine search:
CREATE INDEX ON chunks USING hnsw (embedding vector_cosine_ops);

-- Retrieve with combined scoring:
-- Over-fetch candidates, rerank with saliency
SELECT content,
       (1 - (embedding <=> query_embedding)) * power(saliency, gamma) AS score
FROM (
    -- First: get top-K candidates by cosine alone (index-accelerated)
    SELECT *
    FROM chunks
    ORDER BY embedding <=> query_embedding
    LIMIT 100  -- over-fetch
) candidates
ORDER BY score DESC
LIMIT 10;
```

### Dampening Functions

Raw multiplication can let saliency dominate. Common dampening strategies:

| Function | Formula | When to use |
|----------|---------|-------------|
| Power | `saliency^γ` (γ < 1) | When saliency varies widely |
| Logarithmic | `log(1 + saliency)` | When saliency is count-based (citations, views) |
| Z-score nudge | `1 + α·(saliency - μ)/σ` | When saliency should be a tiebreaker only |
| Sigmoid | `1 / (1 + exp(-saliency))` | When saliency is unbounded |

The γ exponent is the most practical single knob:

- γ = 0: saliency ignored (pure cosine)
- γ = 0.5: square-root dampening (gentle boost)
- γ = 1.0: linear (the Bayesian default)
- γ > 1: saliency dominates

### Tradeoffs

| Pro | Con |
|-----|-----|
| Saliency updates are cheap (no reindexing) | Cannot use index for the combined score |
| Tunable γ at query time | Requires over-fetching + reranking |
| Preserves pure cosine for other uses | Two-pass retrieval adds complexity |
| Simple to understand and debug | Over-fetch size K must be large enough |

---

## Approach 3: The Extra Dimension Trick

**Idea:** Append saliency as an additional embedding dimension. The geometry
naturally encodes both semantic content and importance, and the index handles
it natively.

### The Math

Augment document and query vectors:

```
d_aug = [d · √(1 - w²),  w · saliency]     (769 dims)
q_aug = [q · √(1 - w²),  w · 1.0     ]     (769 dims)
```

where `w` ∈ [0, 1] controls the saliency-vs-semantics tradeoff.

The dot product of augmented vectors:

```
q_aug · d_aug = (1 - w²)·cos(q, d) + w²·saliency
```

This is a weighted linear combination of cosine similarity and saliency,
baked directly into the vector geometry.

### Implementation

```python
import numpy as np

def augment_document(embedding: np.ndarray, saliency: float, weight: float = 0.3):
    """Augment document embedding with saliency dimension.

    Args:
        embedding: Unit-normalized embedding vector.
        saliency: Importance score in [0, 1].
        weight: How much saliency influences the final score (0 = pure
                semantics, 1 = pure saliency).
    """
    e_scaled = embedding * np.sqrt(1 - weight**2)
    return np.append(e_scaled, weight * saliency)

def augment_query(embedding: np.ndarray, weight: float = 0.3):
    """Augment query — always seeks maximum saliency."""
    e_scaled = embedding * np.sqrt(1 - weight**2)
    return np.append(e_scaled, weight * 1.0)
```

```sql
-- Store 769-dimensional augmented vectors:
CREATE TABLE chunks (
    id SERIAL PRIMARY KEY,
    content TEXT,
    embedding vector(769)  -- 768 semantic + 1 saliency
);

CREATE INDEX ON chunks USING hnsw (embedding vector_cosine_ops);
```

### Tradeoffs

| Pro | Con |
|-----|-----|
| Single index, fully accelerated | Weight is fixed at index time |
| Mathematically clean decomposition | Saliency changes require reindexing |
| No post-processing | Adds a dimension (trivial cost) |
| Tunable weight parameter | Cannot adjust weight per-query |

---

## Approach 4: Reciprocal Rank Fusion (RRF)

**Idea:** Retrieve candidates by both signals independently, fuse the ranked
lists using RRF. Robust when the two signals have incompatible scales.

### Implementation

```sql
WITH semantic AS (
    SELECT id, content,
           rank() OVER (ORDER BY embedding <=> query_embedding) AS rank_s
    FROM chunks
    ORDER BY embedding <=> query_embedding
    LIMIT 100
),
importance AS (
    SELECT id, content,
           rank() OVER (ORDER BY saliency DESC) AS rank_i
    FROM chunks
    WHERE (embedding <=> query_embedding) < 0.7  -- rough relevance filter
    ORDER BY saliency DESC
    LIMIT 100
)
SELECT COALESCE(s.id, i.id) AS id,
       COALESCE(s.content, i.content) AS content,
       COALESCE(1.0 / (60 + s.rank_s), 0) +
       COALESCE(1.0 / (60 + i.rank_i), 0) AS rrf_score
FROM semantic s
FULL OUTER JOIN importance i ON s.id = i.id
ORDER BY rrf_score DESC
LIMIT 10;
```

The constant 60 is standard (from the original RRF paper). It controls how
quickly rank-based scores decay.

### Tradeoffs

| Pro | Con |
|-----|-----|
| No scale normalization needed | Two retrieval passes |
| Robust to score distribution mismatches | More complex query |
| Easy to add more signals | Loses score granularity |

---

## Which Approach to Use

| Situation | Recommended approach |
|-----------|---------------------|
| Saliency is stable, set at ingest time | **Magnitude trick** (Approach 1) |
| Saliency changes frequently | **Separate column** (Approach 2) |
| Need tunable weight at query time | **Separate column** with adjustable γ |
| Fixed weight, want single fast index | **Extra dimension** (Approach 3) |
| Multiple heterogeneous signals | **RRF** (Approach 4) |

---

## Types of Saliency

"Important" means different things in different contexts:

| Type | Meaning | Update frequency | Suggested dampening |
|------|---------|-----------------|---------------------|
| Source authority | Trusted source | Static | None (fixed multiplier) |
| User-marked | "Star" or "Important!" flag | On user action | None (binary or small range) |
| Recency | Newer = more relevant | Periodic decay | Exponential decay function |
| Citation count | Frequently referenced | Incremental | `log(1 + count)` |
| Retrieval frequency | Often retrieved | Incremental | `log(1 + freq)` |
| Semantic centrality | Representative of topic | At ingest | None (already normalized) |

---

## The Anti-Pattern: Don't Modify Embedding Direction

Never encode saliency by adding a "saliency vector" to shift the embedding's
direction, or by any operation that rotates the embedding in its semantic
space. The model was trained to encode meaning in specific directions —
pushing the vector elsewhere for non-semantic reasons corrupts the learned
geometry.

Saliency should only affect:
- **Magnitude** (Approach 1)
- **Extra dimensions** (Approach 3)
- **A separate signal** (Approaches 2 and 4)

The semantic embedding's *direction* must remain untouched.
