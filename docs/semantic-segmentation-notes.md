# Hierarchical Semantic Segmentation: Design Notes

Working notes from a brainstorming session on multi-resolution text
segmentation using embedding models, signal processing analogies, and
noise-assisted robustness. These ideas are at the "strong hypothesis" stage —
theoretically grounded and internally consistent, but not yet empirically
validated as a unified system.

---

## Core Idea

Text documents have intrinsic semantic structure at multiple scales —
sentences, paragraphs, sections, chapters. Rather than imposing fixed-size
chunks (the typical RAG approach), we detect the natural boundaries
empirically, bottom-up, at progressively coarser scales. The result is a tree
of chunks, each with its own embedding, searchable at the granularity that
best matches a given query.

The key insight: this is a **change-point detection problem** on a
**multivariate signal** (the sequence of embedding vectors over document
position), and decades of signal processing tools apply directly.

---

## The Signal Processing Frame

### The Embedding Trajectory

A document, when embedded chunk-by-chunk, traces a path through R^D (where D
= embedding dimensionality, e.g. 768). Within a stable topic, this path
meanders locally — small, continuous changes. At a topic boundary, it jumps
to a distant region.

The **coherence signal** is the 1D projection of this trajectory's smoothness:

```
C(p) = cos(e_p, e_{p-1})
```

High C = smooth continuation. Low C = potential boundary. This is the signal
we analyze.

### Change-Point Detection

A topic boundary is where C(p) drops significantly below its local baseline.
The detector compares the current coherence to a running window:

```
boundary_score(p) = mean(C(p-n)...C(p-1)) - C(p)
                    \_____________________/   \___/
                     local baseline            current value
```

This is a CUSUM-like anomaly detector. The lookback window (n ≈ 5) provides
the local norm — what "normal coherence" looks like in this region of the
document.

### Curvature Interpretation

The embedding path's curvature at position p is:

```
curvature(p) ≈ ||e_{p+1} - 2·e_p + e_{p-1}||    (second difference)
```

Fine topic shifts = high curvature = high "frequency" of semantic change.
Section boundaries = sustained directional change. The document's overall
theme = the DC component (mean of all embeddings).

---

## Syntactic Cues as Bayesian Priors

Raw embedding coherence alone can produce false boundaries (mid-sentence
coherence dips from unusual vocabulary) or miss real boundaries (smooth topic
transitions between paragraphs). Syntactic cues provide a prior probability of
boundary independent of the semantic signal.

Framing:

```
P(boundary | position p) ∝ P(semantic_drop | boundary) · P(boundary | syntax)
                            \_________________________/   \________________/
                                    likelihood                  prior
```

The prior encodes structural formatting knowledge:

| Syntactic cue | P(boundary | syntax) |
|---------------|---------------------|
| Markdown heading | ~0.9 |
| Double newline (paragraph break) | ~0.8 |
| List item start | ~0.4 |
| Sentence end (period + space) | ~0.3 |
| Single newline | ~0.2 |
| Mid-sentence | ~0.05 |

These priors prevent the algorithm from breaking mid-sentence (even with a
coherence dip) while making it easier to break at structural boundaries (even
with a modest dip). The combination of semantic likelihood and syntactic prior
produces more natural segmentation than either alone.

---

## Multi-Pass Hierarchical Segmentation

### Why Bottom-Up

Top-down approaches (like TreeSeg's divisive clustering) impose structure from
global statistics. Bottom-up approaches discover structure empirically from
local signals, which is:

1. More robust to documents with irregular structure
2. More natural for streaming/ingestion (no need to see the whole document
   first)
3. Analogous to how signal processing decomposes signals (EMD extracts the
   highest-frequency mode first, then works down)

### The Algorithm

```
Pass 0: Micro-chunking
  Split on syntactic boundaries (sentence-end, double newline)
  into atomic units of ~30-100 tokens.

Pass 1: Fine boundary detection
  Embed micro-chunks with a cheap, fast model.
  Run coherence-drop detector with syntactic priors.
  → Produces paragraph-level segments.

Pass 2: Medium boundary detection
  Re-embed each paragraph-level segment as a whole.
  Run the same detector on these new embeddings.
  → Produces section-level segments.

Pass 3+: Coarse boundary detection
  Re-embed each section-level segment.
  Repeat until no more boundaries are found or a max depth is reached.
  → Produces chapter/part-level segments.
```

The critical step is **re-embedding at each level**. The embeddings of whole
paragraphs capture different information than embeddings of individual
sentences — they represent the paragraph's *overall theme*, enabling detection
of thematic shifts that aren't visible at the sentence level.

### Result: A Tree

```
Level 0 (sentences):     [s1][s2][s3][s4][s5][s6][s7][s8][s9]...
Level 1 (paragraphs):    [===para 1===][===para 2===][==para 3==]...
Level 2 (sections):      [=======section A=========][===sec B===]...
Level 3 (chapters):      [===============chapter 1================]...
```

Each node has its own embedding. All levels can be stored and searched,
enabling adaptive granularity matching: specific queries match fine-grained
chunks, broad queries match coarse ones.

---

## EMD Analogy: Why This Is Modal Decomposition

### Empirical Mode Decomposition (EMD)

EMD (Huang et al., 1998) decomposes a signal into Intrinsic Mode Functions
(IMFs) — oscillatory components at different scales — without imposing a fixed
basis. The decomposition is data-driven: the modes are whatever the signal
actually contains, not what a predefined basis expects.

### The Mapping

| EMD concept | Text segmentation analog |
|-------------|--------------------------|
| Input signal | Embedding trajectory through R^D |
| Highest-frequency IMF | Sentence-level topic variation |
| Lower-frequency IMFs | Paragraph, section-level thematic structure |
| Residual | Document's overall theme (the "DC component") |
| Extrema / envelopes | Local coherence peaks and valleys |
| Sifting (extract mode) | Boundary detection + subtraction of fine structure |
| Zero crossings | Topic shift positions |

### Multivariate EMD (MEMD)

Since embeddings are vectors (not scalars), the full formalism requires MEMD
(Rehman & Mandic, 2010), which handles multi-channel signals by:

1. Projecting the multivariate signal onto uniformly-sampled directions on the
   unit hypersphere
2. Computing 1D envelopes along each projection
3. Averaging envelopes to get the multivariate local mean
4. Sifting to extract each IMF

For embedding sequences, each IMF would be a semantic oscillation mode —
fine-grained topic shifts at the highest frequency, section-level themes at
lower frequencies.

### Practical Consideration

Full MEMD in 768 dimensions is computationally expensive (requires many
projection directions). The PCA reduction discussed below makes this
tractable, or the simpler bottom-up boundary detection approximates the IMF
extraction without the full MEMD machinery.

---

## CEEMDAN-Inspired Noise Injection for Robustness

### The Mode Mixing Problem

Plain EMD (or plain boundary detection) can suffer from **mode mixing**: a
single run might confuse stylistic variation for topic shifts. Verbose
phrasing, parenthetical asides, unusual vocabulary choices — all create small
coherence dips that aren't genuine topic boundaries.

### The CEEMDAN Solution (Torres et al., 2011)

Complete Ensemble EMD with Adaptive Noise addresses mode mixing by:

1. Adding scaled Gaussian noise to the signal
2. Running the decomposition N times (N ≈ 10-500)
3. Averaging the resulting modes across all runs
4. The noise cancels in the average; only robust modes survive

The noise amplitude is scaled to the current residual:

```
noise_amplitude = ε · std(current_residual)     (ε ≈ 0.1-0.4)
```

This means noise is injected at each sifting stage, proportional to what
remains — progressively smaller as modes are extracted.

### Application to Boundary Detection

Instead of running boundary detection once on the raw coherence signal, run it
N times on noise-perturbed versions of the embeddings, then **vote**:

```python
def ceemdan_boundaries(embeddings, n_ensemble=10, noise_fraction=0.2,
                       lookback=5, threshold_sigma=1.5, syntactic_priors=None):
    """
    CEEMDAN-inspired robust boundary detection.

    Run boundary detection N times on noise-perturbed embeddings.
    Boundaries that survive across perturbations are genuine topic shifts;
    those triggered by stylistic quirks wash out in the vote.
    """
    n_chunks, d = embeddings.shape
    boundary_votes = np.zeros(n_chunks)
    noise_scale = noise_fraction * embeddings.std()

    for run in range(n_ensemble):
        # Perturb embeddings with Gaussian noise
        noisy = embeddings + np.random.randn(n_chunks, d) * noise_scale
        noisy = noisy / np.linalg.norm(noisy, axis=1, keepdims=True)

        # Detect boundaries on this noisy version
        boundaries = detect_boundaries(
            noisy, lookback=lookback,
            threshold_sigma=threshold_sigma,
            syntactic_weights=syntactic_priors
        )

        for b in boundaries:
            boundary_votes[b] += 1

    # Boundaries that win majority vote
    robust_boundaries = np.where(boundary_votes >= n_ensemble * 0.5)[0]
    confidences = boundary_votes[robust_boundaries] / n_ensemble

    return robust_boundaries, confidences
```

### Hierarchical Noise Scaling

At each level of the hierarchy, the noise amplitude adapts — analogous to
CEEMDAN's noise being proportional to the current residual:

```python
level_noise = base_noise * current_embeddings.std() * (decay ** level)
```

Finer levels get more noise (more stylistic variation to overcome); coarser
levels get less (section boundaries are unambiguous).

### Why This Works: What Noise Cancels vs. What Survives

**Washes out (mode mixing artifacts):**
- Stylistic variation: verbose vs. terse phrasing of the same topic
- Lexical surface patterns: proper nouns creating spurious coherence
- Embedding artifacts: systematic biases in the model

**Survives (genuine structure):**
- Semantic discontinuities: actual content shifts where embedding
  *directions* change fundamentally
- Topic boundaries that are robust because the underlying meaning is
  different, regardless of how it's phrased

The mechanism: genuine boundaries produce coherence drops in every noisy run
(the underlying embedding directions are fundamentally different, and
perturbation in redundant dimensions can't change that). Spurious boundaries
produce drops only in some runs (the "boundary" was an artifact of one
particular arrangement of noise in the redundant dims).

### The Temperature/LLM Analogy

An LLM at temperature > 0 generates different token sequences each run, but
the semantic content is approximately invariant. Embedding those 10 different
phrasings gives vectors that cluster tightly around the semantic invariant.

Noise injection into embeddings does the same thing: each perturbed embedding
is a "different way of saying the same thing." The average is the semantic
invariant — what survives across all possible phrasings. Boundaries detected
on this average are boundaries in *meaning*, not boundaries in *phrasing*.

This is also the principle behind **Monte Carlo Dropout**: run inference N
times with random dropout, average the embeddings. The mean is robust; the
variance is uncertainty.

### Computational Cost

The noise injection is essentially free:
- Embeddings are computed once (the expensive step)
- Adding noise + re-normalizing: O(n_chunks × d) per run
- Boundary detection: O(n_chunks × lookback) per run
- Total overhead: ~10× the boundary detection step, which is negligible
  compared to embedding

---

## PCA and Intrinsic Dimensionality

### Key Findings from the Literature (2024-2025)

The embedding space is massively over-parameterized relative to the
information it actually encodes:

| Finding | Source |
|---------|--------|
| Intrinsic dimensionality of embeddings: ~10-30 | Ueda & Yokoi, 2024 |
| ~98% redundancy ratio in large models | arXiv:2503.02142 |
| PCA to 50% of dims: no significant perf loss | LREC 2024 |
| PCA from 3072 → 110: usable for RAG | PCA-RAG, 2025 |
| Randomly removing 50% dims: <10% retrieval drop | 2024 multi-task study |
| PCA sometimes *improves* over original dims | LREC 2024 |

The 768 dimensions decompose roughly as:
- ~20-50 dims: genuine semantic signal for a given corpus
- ~50-100 dims: potentially useful but noisy features
- ~500+ dims: redundancy / training artifacts

### PCA as Corpus-Specific Dimensionality Reduction

For a specific corpus, the effective space is even smaller. All your
documents share domain vocabulary, common concepts, structural patterns —
these reduce the intrinsic dimensionality further.

```python
from sklearn.decomposition import PCA

# Embed a representative sample of your corpus
all_embeddings = np.array([embed(chunk) for chunk in corpus])

# Find intrinsic dimensionality
pca = PCA(n_components=768)
pca.fit(all_embeddings)
cumvar = np.cumsum(pca.explained_variance_ratio_)
n_dims_95 = np.argmax(cumvar >= 0.95) + 1  # often 30-80

# Reduce
pca_model = PCA(n_components=n_dims_95)
reduced = pca_model.fit_transform(all_embeddings)
```

### PCA + EMD: The Natural Combination

In PCA-reduced space:
1. Each principal component is potentially interpretable as a topic/theme axis
2. Topic shifts show up as step functions on specific PC axes
3. MEMD becomes computationally tractable (~30-50 channels vs. 768)
4. The noise dimensions (which obscure structure in full space) are gone

You can literally plot the PC time series and *see* the topic structure:
```
PC1 over document position: major topic axis
PC2 over document position: secondary thematic dimension
PC3 over document position: register/formality shifts
...
```

Step transitions on PC1 = major topic shifts. Oscillations on PC3 =
alternation between abstract and concrete passages.

### Matryoshka Models as Learned PCA

Models trained with Matryoshka Representation Learning (EmbeddingGemma, Nomic
v2, Jina v3) organize their dimensions by importance during training. The
first N dimensions contain the most information — this is essentially learned
PCA, without needing a separate fitting step.

For boundary detection, truncating to 64-128 Matryoshka dimensions is
equivalent to PCA reduction but requires no corpus-specific fitting:

```python
embedding_full = embed(chunk)          # 768 dims
embedding_fast = embedding_full[:64]   # first 64 dims (most important)
embedding_fast /= np.linalg.norm(embedding_fast)  # re-normalize
```

This gives a cheap, low-dimensional signal for boundary detection that's still
semantically meaningful.

### The Deep Insight: Redundancy as a Resource

The redundant dimensions serve different purposes in different contexts:

- **For retrieval:** Redundancy is waste. PCA/truncation removes it → faster,
  tighter, sometimes even better (denoised).
- **For CEEMDAN-style segmentation:** Redundancy is a *resource*. Noise
  injection *into* the redundant dimensions enables ensemble averaging that
  converges on the true semantic signal.

These are two sides of the same coin. The redundant dimensions are like a
thermal bath in statistical mechanics — they provide the degrees of freedom
needed for ergodic exploration of the solution space.

---

## Choice of Models for the Pipeline

The pipeline naturally separates into a cheap detection phase and a final
embedding phase:

### Boundary Detection (cheap, fast, many calls)

| Model | Params | Speed | Suitability |
|-------|--------|-------|-------------|
| all-minilm | 23M | Very fast | Ideal — coherence detection doesn't need semantic precision |
| EmbeddingGemma (int4, truncated to 64-128 dims) | 308M | Fast | Good — Matryoshka truncation makes it efficient |

For boundary detection, you don't need semantically precise embeddings. You
just need embeddings where same-topic chunks produce high similarity and
different-topic chunks produce low similarity. Even a small model achieves
this — the *relative* coherence signal is what matters, not the absolute
embedding quality.

### Final Segment Embeddings (quality, stored in DB)

| Model | Params | Quality | Context |
|-------|--------|---------|---------|
| EmbeddingGemma | 308M | Excellent for size | 2048 tokens |
| Qwen3-Embedding-4B | 4B | State-of-art | 8192+ tokens |
| Jina v3 | 570M | Very good, task LoRAs | 8192 tokens (CC BY-NC!) |

These are computed once per segment and stored. Quality matters because these
are what get compared against queries at retrieval time.

---

## Retrieval on the Tree Structure

The tree enables **adaptive granularity matching**:

- **Short, specific query** ("What year was the Eiffel Tower built?")
  → matches at sentence/paragraph level (sharp, focused embeddings)

- **Broad conceptual query** ("Explain French architectural history")
  → matches at section/chapter level (summary-like embeddings)

All levels are stored in the vector DB. Options at query time:

1. **Search all levels simultaneously** — cosine similarity naturally selects
   the right granularity (specific queries match small chunks better; broad
   queries match large chunks better).

2. **Query specificity routing** — estimate query specificity (how far the
   query embedding is from the corpus centroid, or the query's information
   density) and route to the appropriate tree level.

3. **Hierarchical expansion** — find the best match at the finest level, then
   return its parent context from the tree for richer answering.

---

## Connections to Existing Work

| System/Paper | Relationship to this approach |
|--------------|-------------------------------|
| TextTiling (Hearst, 1997) | The original: cosine-based boundary detection with sliding windows, but BoW not embeddings |
| BertSeg / LLM-TextTiling | Modern TextTiling with BERT embeddings, but single-pass, no hierarchy |
| TreeSeg (2024) | Hierarchical segmentation, but top-down (divisive), not bottom-up |
| Late Chunking (Jina, 2024) | Complementary: better embeddings for segments once boundaries are found |
| CEEMDAN (Torres et al., 2011) | The noise-assisted decomposition framework we adapt |
| MEMD (Rehman & Mandic, 2010) | The multivariate extension relevant to high-dim embeddings |
| PCA-RAG (2025) | Validates PCA reduction for RAG; we use it for tractable boundary detection |
| ruptures (Python library) | Implements window-based change-point detection on 1D signals; could be used directly on the coherence signal |

---

## Open Questions

1. **Optimal noise amplitude ε for embedding perturbation.** CEEMDAN uses
   ε ≈ 0.1-0.4 for 1D signals. For 768-dim embeddings, the right scale might
   be different — empirical calibration needed.

2. **Boundary position uncertainty.** When votes are spread across adjacent
   positions (e.g., positions 42, 43, 44 each get 30% of votes), should we
   pick the mode, or should the boundary itself be "fuzzy"?

3. **PCA vs. Matryoshka truncation.** For a given corpus, which gives better
   coherence signals — corpus-specific PCA, or model-trained Matryoshka
   truncation? Probably depends on how domain-specific the corpus is.

4. **Number of ensemble runs.** CEEMDAN literature suggests N=100-500 for high
   quality. For our cheap boundary detection, N=10 might suffice since we're
   voting on discrete boundary positions, not averaging continuous modes.

5. **Inter-level consistency.** Should a level-2 boundary always coincide with
   a level-1 boundary? (Probably yes — a section break should always be a
   paragraph break too.) How to enforce this constraint?

6. **Late chunking synergy.** If using Jina v3 (8192 tokens), we could run
   late chunking within each section: feed the full section through the
   transformer, then apply our detected paragraph boundaries to the
   token-level hidden states. This gives paragraph embeddings that carry
   section-level context.

7. **Evaluation.** How to measure whether this produces "better" chunks than
   fixed-size or naive semantic chunking? Presumably: retrieval accuracy on a
   benchmark with mixed-granularity queries. But no standard benchmark exists
   for hierarchical chunk retrieval specifically.

---

## References

### Signal Processing & EMD
- Huang et al. (1998). The empirical mode decomposition and the Hilbert
  spectrum for nonlinear and non-stationary time series analysis.
- Wu & Huang (2005). Ensemble Empirical Mode Decomposition: a Noise-Assisted
  Data Analysis Method.
- Torres, Colominas, Schlotthauer & Flandrin (2011). A Complete Ensemble
  Empirical Mode Decomposition with Adaptive Noise. ICASSP.
- Rehman & Mandic (2010). Multivariate Empirical Mode Decomposition.
  Proceedings of the Royal Society A.
- PyEMD library: https://pyemd.readthedocs.io/en/latest/ceemdan.html

### Text Segmentation
- Hearst (1997). TextTiling: Segmenting Text into Multi-paragraph Subtopic
  Passages.
- Solbiati et al. (2021). BertSeg: BERT-based TextTiling.
- Sterner et al. (2024). TreeSeg: Hierarchical Topic Segmentation of Large
  Transcripts. arXiv:2407.12028.
- Survey: Recent Trends in Linear Text Segmentation (Nov 2024).
  arXiv:2411.16613.
- saeedabc/llm-text-tiling (GitHub): Extended TextTiling with LLM embeddings.

### Embedding Dimensionality & PCA
- PCA-RAG (April 2025). arXiv:2504.08386.
- Evaluating Unsupervised Dimensionality Reduction for Sentence Embeddings
  (LREC 2024). arXiv:2403.14001.
- Measuring Intrinsic Dimension of Token Embeddings (2025).
  arXiv:2503.02142.
- Compressing LLMs with PCA Without Performance Loss (2025).
  arXiv:2508.04307.

### Embedding Models Referenced
- EmbeddingGemma (Google, Sep 2025). arXiv:2509.20354.
  https://ollama.com/library/embeddinggemma
- Jina Embeddings v3 (Sep 2024). arXiv:2409.10173.
- Late Chunking (Jina, 2024). arXiv:2409.04701.
  https://github.com/jina-ai/late-chunking
- all-MiniLM-L6-v2 (sentence-transformers).
  https://ollama.com/library/all-minilm

### Noise Robustness & Ensembles
- Uncertainty-driven Embedding Convolution (2025). arXiv:2507.20718.
- Mean Embeddings with Test-Time Augmentation (MeTTA). arXiv:2106.08038.
- Monte Carlo Dropout for uncertainty estimation (Gal & Ghahramani, 2016).

### Tools
- ruptures (Python): Change-point detection.
  https://centre-borelli.github.io/ruptures-docs/
- Jina mlx-retrieval: Embedding training on Apple Silicon.
  https://github.com/jina-ai/mlx-retrieval
- pgvector: Vector similarity search for PostgreSQL.
  https://github.com/pgvector/pgvector
