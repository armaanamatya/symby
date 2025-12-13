# Implementation Plan: SOTA Graph + Vector Search for AI Research Impact Observatory

## Executive Summary

Build a **hybrid semantic-graph search system** using 2024-2025 SOTA methods to analyze 3.2M+ scientific papers. This replaces shallow keyword matching with embedding-based semantic understanding and citation-free graph construction.

**Key Innovation**: Build a similarity graph FROM embeddings (no external API needed), then run GPU-accelerated graph analytics to trace ML influence across scientific disciplines.

---

## Approach Comparison: Pros & Cons

### Approach 1: Pure Keyword/Regex Matching (Current Implementation)

| Pros | Cons |
|------|------|
| ✅ Fast, no GPU needed | ❌ "transformer" matches electrical transformers |
| ✅ Interpretable (know exactly what matched) | ❌ Misses paraphrases ("neural net" ≠ "deep learning") |
| ✅ No external dependencies | ❌ No semantic understanding |
| ✅ Works offline | ❌ High false positive/negative rate |
| ✅ Zero preprocessing time | ❌ Can't answer "papers similar to X" |

**Best for**: Quick prototyping, baseline comparison
**Not for**: Production semantic search, influence tracing

---

### Approach 2: Pure Dense Vector Search (Embeddings Only)

| Pros | Cons |
|------|------|
| ✅ Semantic understanding | ❌ Misses exact keyword matches |
| ✅ Finds paraphrases automatically | ❌ 45+ hours to embed 3.2M papers |
| ✅ "Papers like this" queries | ❌ ~15GB storage for embeddings |
| ✅ Language-agnostic | ❌ Can't trace influence paths |
| ✅ SOTA accuracy on retrieval benchmarks | ❌ No graph structure = no "how did X spread" |
| ✅ GPU-accelerated (cuVS) | ❌ Black box (hard to explain why results match) |

**Best for**: Finding similar papers, semantic search
**Not for**: Influence tracing, adoption dynamics, path queries

---

### Approach 3: Pure Citation Graph (External API)

| Pros | Cons |
|------|------|
| ✅ Real citation relationships | ❌ Requires Semantic Scholar API |
| ✅ Can trace actual influence paths | ❌ 3s/request = 111 days for 3.2M papers |
| ✅ PageRank = real influence measure | ❌ Only captures explicit citations |
| ✅ Community detection meaningful | ❌ Misses implicit influence (no citation but same method) |
| ✅ Ground truth for academic influence | ❌ API rate limits, failures, incomplete data |

**Best for**: If you have pre-existing citation data
**Not for**: Hackathon timeline, papers without citation data

---

### Approach 4: Pure Sparse Vector Search (BM25)

| Pros | Cons |
|------|------|
| ✅ Exact keyword matching | ❌ No semantic understanding |
| ✅ Fast indexing | ❌ "ML" won't match "machine learning" |
| ✅ Interpretable (TF-IDF scores) | ❌ Sensitive to vocabulary mismatch |
| ✅ Works well for rare terms | ❌ Misses conceptual similarity |
| ✅ Low memory footprint | ❌ Can't do "papers like X" |

**Best for**: Known-item search, rare term lookup
**Not for**: Exploratory search, concept-based queries

---

### Approach 5: Hybrid Vector Search (Dense + Sparse + Reranking)

| Pros | Cons |
|------|------|
| ✅ Best of both semantic + keyword | ❌ More complex pipeline |
| ✅ SOTA on retrieval benchmarks (BEIR +9 pts) | ❌ Two indexes to maintain |
| ✅ Handles both "transformer" and paraphrases | ❌ Tuning alpha (dense vs sparse weight) |
| ✅ Reranking improves precision | ❌ Reranking adds latency (~100ms) |
| ✅ Robust to query type variation | ❌ Still no graph structure |

**Best for**: Production search systems
**Not for**: Influence tracing (still no paths)

---

### Approach 6: Semantic Similarity Graph (Recommended)

| Pros | Cons |
|------|------|
| ✅ No external API needed | ❌ Edges are inferred, not ground truth |
| ✅ Captures implicit influence | ❌ Similarity threshold is arbitrary |
| ✅ GPU-accelerated (cuGraph) | ❌ 160M edges = significant storage |
| ✅ Can trace influence paths | ❌ Paths may not reflect real citations |
| ✅ Community detection = research clusters | ❌ Graph construction takes hours |
| ✅ Cross-field edges automatically discovered | ❌ Edge semantics less clear than citations |
| ✅ Works with YOUR data (no API) | ❌ May have spurious edges (false similarity) |
| ✅ "How did X influence Y" queries | ❌ PageRank meaning differs from citation graph |

**Best for**: This hackathon (no API, need paths, have GPU)
**Limitations**: Influence is "semantic proximity" not "actual citation"

---

### Approach 7: Full Hybrid (Vector + Semantic Graph) — RECOMMENDED

| Pros | Cons |
|------|------|
| ✅ Semantic search for "find papers" | ❌ Most complex implementation |
| ✅ Graph traversal for "trace influence" | ❌ Longest development time |
| ✅ Best accuracy (hybrid search) | ❌ Most storage (~20GB total) |
| ✅ Answers ALL query types | ❌ Multiple systems to maintain |
| ✅ Maximum hackathon score potential | ❌ More failure points |
| ✅ Strong "Spark Story" (cuVS + cuGraph) | ❌ Requires embedding generation (45h) |
| ✅ Novel approach (not just keyword matching) | ❌ Harder to debug/explain |

**Best for**: Winning the hackathon
**Trade-off**: Complexity vs. capability

---

### Decision Matrix

| Capability | Keywords | Dense | Sparse | Hybrid | Citation Graph | Semantic Graph | Full Hybrid |
|------------|----------|-------|--------|--------|----------------|----------------|-------------|
| "Find papers about X" | ⚠️ | ✅ | ⚠️ | ✅ | ❌ | ⚠️ | ✅ |
| "Papers similar to Y" | ❌ | ✅ | ❌ | ✅ | ❌ | ⚠️ | ✅ |
| "How did A influence B" | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ |
| "Which fields adopted X" | ⚠️ | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ |
| "Bridge papers between fields" | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ |
| Works without external API | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ |
| Hackathon Obj 1 (ML Impact) | ⚠️ | ⚠️ | ⚠️ | ⚠️ | ✅ | ✅ | ✅ |
| Hackathon Obj 2 (Adoption) | ⚠️ | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ |
| Hackathon Obj 3 (Quality) | ⚠️ | ⚠️ | ⚠️ | ⚠️ | ⚠️ | ⚠️ | ✅ |
| Hackathon Obj 4 (Discovery Paths) | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ |
| Development time | 1h | 50h | 5h | 55h | 111 days | 60h | 70h |
| NVIDIA Stack usage | ❌ | ✅ | ❌ | ✅ | ⚠️ | ✅ | ✅ |

Legend: ✅ = Strong | ⚠️ = Partial | ❌ = Cannot do

---

### Recommendation for Hackathon

**Primary: Full Hybrid (Approach 7)**
- Highest capability
- Best "Spark Story" (cuVS + cuGraph)
- Covers all 4 objectives

**Fallback: Semantic Graph Only (Approach 6)**
- If embedding time is too long
- Skip hybrid search, use graph-based retrieval
- Still covers objectives 1, 2, 4

**Minimum Viable: Enhanced Keywords + Stats (Approach 1 improved)**
- If everything fails
- Add word boundaries, better patterns
- Focus on objectives 1-3 via statistical analysis

---

## Hardware & Constraints

| Resource | Specs | Use Case |
|----------|-------|----------|
| DGX Spark | 128GB unified memory, Arm CPU, NVIDIA GPU | Inference, demo, search |
| 3x A100 GPUs | 80GB each | Embedding generation, training |
| Dataset | 3.2M papers, ~84GB uncompressed | Full corpus analysis |

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           DATA LAYER                                     │
│  s2orc_data/ (3.2M papers, 242 files, 14 fields, 2007-2022)            │
└────────────────────────────────┬────────────────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       EMBEDDING LAYER                                    │
│  BGE-M3 (8K context) → Dense (1024d) + Sparse (lexical) + ColBERT      │
│  GPU: A100 cluster for batch embedding                                  │
└────────────────────────────────┬────────────────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        INDEX LAYER                                       │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐     │
│  │ cuVS/FAISS      │    │ BM25/Sparse     │    │ Semantic Graph  │     │
│  │ (Dense Index)   │    │ (Lexical Index) │    │ (cuGraph)       │     │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘     │
└────────────────────────────────┬────────────────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        QUERY LAYER                                       │
│  Hybrid Search (RRF) + Graph Traversal + Re-ranking                     │
│  Natural Language Queries → Influence Paths, Adoption Analysis          │
└────────────────────────────────┬────────────────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      APPLICATION LAYER                                   │
│  Streamlit Dashboard | FastAPI Backend | 4 Hackathon Objectives         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Data Preparation & Preprocessing

### 1.1 Data Audit & Validation
- [ ] **Verify dataset integrity**
  - Count total papers across all 242 files
  - Validate JSON structure consistency
  - Check for missing/null fields (id, text, year, metadata)
  - Log field distribution and year coverage

- [ ] **Create data manifest**
  ```python
  # Output: data_manifest.json
  {
    "total_papers": 3200000,
    "fields": {"Biology": 477169, "Medicine": 890702, ...},
    "year_range": [2007, 2022],
    "avg_text_length": 15000,
    "files": [{"path": "...", "papers": 1000, "size_mb": 50}, ...]
  }
  ```

### 1.2 Text Preprocessing Pipeline
- [ ] **Implement robust text cleaner** (`src/data/preprocessor.py`)
  - Remove boilerplate (acknowledgments, references section)
  - Normalize whitespace, fix encoding issues
  - Extract structured sections (Abstract, Methods, Results) if detectable
  - Truncate to 8192 tokens (BGE-M3 limit) with smart truncation:
    - Priority: Title > Abstract > Introduction > Methods > Results

- [ ] **Create chunking strategy for long papers**
  - Option A: Single embedding per paper (truncated)
  - Option B: Multi-chunk with overlap (512 token chunks, 64 overlap)
  - **Recommendation**: Option A for graph construction, Option B for detailed retrieval

### 1.3 Metadata Enrichment
- [ ] **Enhance ML classification** (replace keyword matching)
  - Keep existing keyword classifier as baseline/fallback
  - Add confidence scores to classifications
  - Extract mentioned techniques, frameworks, datasets

- [ ] **Build technique taxonomy**
  ```python
  TECHNIQUE_HIERARCHY = {
    "deep_learning": {
      "architectures": ["transformer", "cnn", "rnn", "gan", "vae", "diffusion"],
      "methods": ["attention", "convolution", "recurrence", "adversarial"],
      "landmarks": ["bert", "gpt", "resnet", "alphafold", "stable_diffusion"]
    },
    ...
  }
  ```

---

## Phase 2: Embedding Generation (A100 Cluster)

### 2.1 Model Selection & Setup
- [ ] **Install BGE-M3**
  ```bash
  pip install FlagEmbedding
  # Or for faster inference:
  pip install sentence-transformers
  ```

- [ ] **Model configuration**
  ```python
  # src/embeddings/config.py
  EMBEDDING_CONFIG = {
    "model_name": "BAAI/bge-m3",
    "max_length": 8192,
    "batch_size": 32,  # Adjust based on A100 memory
    "use_fp16": True,
    "normalize_embeddings": True,
    "return_dense": True,
    "return_sparse": True,
    "return_colbert_vecs": False,  # Too memory-intensive for 3.2M papers
  }
  ```

### 2.2 Batch Embedding Pipeline
- [ ] **Implement distributed embedding** (`src/embeddings/batch_embed.py`)
  ```python
  # Pseudocode for multi-GPU embedding
  def embed_corpus(papers_df, num_gpus=3):
      # Split papers across GPUs
      chunks = np.array_split(papers_df, num_gpus)

      # Parallel embedding with ProcessPoolExecutor
      with ProcessPoolExecutor(max_workers=num_gpus) as executor:
          futures = [
              executor.submit(embed_chunk, chunk, gpu_id)
              for gpu_id, chunk in enumerate(chunks)
          ]

      # Collect results
      embeddings = concat([f.result() for f in futures])
      return embeddings
  ```

- [ ] **Checkpointing & resume**
  - Save embeddings every 100K papers
  - Store in memory-mapped format (numpy .npy or HDF5)
  - Track progress in `embedding_progress.json`

- [ ] **Output format**
  ```
  embeddings/
  ├── dense/
  │   ├── chunk_000.npy    # Shape: (100000, 1024)
  │   ├── chunk_001.npy
  │   └── ...
  ├── sparse/
  │   ├── chunk_000.npz    # Sparse matrices
  │   └── ...
  ├── metadata/
  │   ├── paper_ids.json   # Mapping: index → paper_id
  │   └── manifest.json
  └── progress.json
  ```

### 2.3 Estimated Resources
| Metric | Estimate |
|--------|----------|
| Papers | 3,200,000 |
| Avg tokens/paper | ~4,000 |
| Embedding time/paper | ~50ms (batched) |
| Total GPU hours | ~45 hours (3x A100) |
| Storage (dense) | ~13GB (3.2M × 1024 × 4 bytes) |
| Storage (sparse) | ~5GB (variable) |

---

## Phase 3: Vector Index Construction (DGX Spark)

### 3.1 Dense Vector Index (cuVS/FAISS)
- [ ] **Install cuVS** (RAPIDS)
  ```bash
  conda install -c rapidsai -c conda-forge -c nvidia \
      cuvs=24.10 faiss-gpu=1.7.4
  ```

- [ ] **Build CAGRA index** (`src/index/dense_index.py`)
  ```python
  import cuvs
  from cuvs.neighbors import cagra

  def build_cagra_index(embeddings, params=None):
      """
      CAGRA: GPU-native graph index
      - 21x faster build than CPU HNSW
      - Converts to HNSW for CPU inference if needed
      """
      if params is None:
          params = cagra.IndexParams(
              intermediate_graph_degree=64,
              graph_degree=32,
              build_algo="nn_descent"
          )

      index = cagra.build(params, embeddings)
      return index
  ```

- [ ] **Alternative: FAISS IVF-PQ** (if cuVS unavailable)
  ```python
  import faiss

  # IVF4096,PQ64 - good balance of speed/accuracy
  index = faiss.index_factory(1024, "IVF4096,PQ64", faiss.METRIC_INNER_PRODUCT)

  # Train on subset
  index.train(embeddings[:100000])

  # Add all vectors
  index.add(embeddings)
  ```

### 3.2 Sparse Vector Index (BM25)
- [ ] **Implement BM25 index** (`src/index/sparse_index.py`)
  ```python
  from rank_bm25 import BM25Okapi
  # Or use BGE-M3's sparse output directly

  def build_sparse_index(sparse_vectors, paper_ids):
      """
      Build inverted index from BGE-M3 lexical weights
      """
      # BGE-M3 outputs: {token_id: weight, ...} per document
      # Convert to searchable inverted index
      ...
  ```

- [ ] **Alternative: Elasticsearch/OpenSearch** (if scaling issues)
  - Containerized deployment
  - Native BM25 + vector search hybrid

### 3.3 Index Persistence
- [ ] **Save/load mechanisms**
  ```python
  # Save
  cagra.save(index, "indexes/cagra_3.2m.bin")
  faiss.write_index(index, "indexes/faiss_ivfpq.bin")

  # Load (on DGX Spark)
  index = cagra.load("indexes/cagra_3.2m.bin")
  ```

---

## Phase 4: Semantic Graph Construction

### 4.1 Graph Design
- [ ] **Define node schema**
  ```python
  @dataclass
  class PaperNode:
      id: str
      title: str
      year: int
      field: str
      is_ml_paper: bool
      ml_score: float
      techniques: List[str]
      embedding_idx: int  # Pointer to dense vector
  ```

- [ ] **Define edge types**
  ```python
  EDGE_TYPES = {
      "semantic_similar": {
          "description": "Cosine similarity > threshold",
          "weight": "similarity_score",
          "directed": False
      },
      "temporal_influence": {
          "description": "Similar + later publication",
          "weight": "similarity_score",
          "directed": True  # A → B means A influenced B
      },
      "cross_field_bridge": {
          "description": "Similar + different fields",
          "weight": "similarity_score",
          "directed": True
      },
      "technique_adoption": {
          "description": "Both use same ML technique",
          "weight": "technique_overlap_score",
          "directed": False
      }
  }
  ```

### 4.2 Edge Construction Algorithm
- [ ] **Implement efficient edge builder** (`src/graph/edge_builder.py`)
  ```python
  def build_semantic_edges(
      embeddings,
      metadata,
      similarity_threshold=0.75,
      cross_field_threshold=0.70,
      max_edges_per_node=50,
      batch_size=10000
  ):
      """
      Build edges from embedding similarity.

      Strategy:
      1. For each paper, find top-K similar papers via ANN search
      2. Filter by threshold and constraints
      3. Add temporal direction (older → newer)
      4. Boost cross-field edges (more interesting)
      """
      edges = []

      for batch_start in range(0, len(embeddings), batch_size):
          batch = embeddings[batch_start:batch_start + batch_size]

          # ANN search for top-K neighbors
          distances, indices = index.search(batch, k=max_edges_per_node * 2)

          for i, (dists, idxs) in enumerate(zip(distances, indices)):
              source_idx = batch_start + i
              source_meta = metadata[source_idx]

              for dist, target_idx in zip(dists, idxs):
                  if target_idx == source_idx:
                      continue

                  similarity = 1 - dist  # Convert distance to similarity
                  target_meta = metadata[target_idx]

                  # Apply filters
                  edge = create_edge_if_valid(
                      source_idx, target_idx,
                      source_meta, target_meta,
                      similarity,
                      similarity_threshold,
                      cross_field_threshold
                  )
                  if edge:
                      edges.append(edge)

      return edges

  def create_edge_if_valid(src, tgt, src_meta, tgt_meta, sim, thresh, cross_thresh):
      """Filter and create directed edge"""
      cross_field = src_meta['field'] != tgt_meta['field']
      effective_threshold = cross_thresh if cross_field else thresh

      if sim < effective_threshold:
          return None

      # Direction: older → newer (influence direction)
      if src_meta['year'] > tgt_meta['year']:
          src, tgt = tgt, src
          src_meta, tgt_meta = tgt_meta, src_meta

      return {
          'source': src,
          'target': tgt,
          'weight': sim,
          'type': 'cross_field_bridge' if cross_field else 'temporal_influence',
          'year_gap': tgt_meta['year'] - src_meta['year']
      }
  ```

### 4.3 Graph Storage & Loading
- [ ] **cuGraph format** (GPU-accelerated)
  ```python
  import cudf
  import cugraph

  def build_cugraph(edges, nodes):
      # Edge DataFrame
      edge_df = cudf.DataFrame({
          'src': [e['source'] for e in edges],
          'dst': [e['target'] for e in edges],
          'weight': [e['weight'] for e in edges]
      })

      # Build graph
      G = cugraph.Graph(directed=True)
      G.from_cudf_edgelist(edge_df, source='src', destination='dst', edge_attr='weight')

      return G
  ```

- [ ] **Persistence**
  ```python
  # Save edges
  edge_df.to_parquet("graph/edges.parquet")

  # Save node attributes
  node_df.to_parquet("graph/nodes.parquet")

  # Load on DGX Spark
  edge_df = cudf.read_parquet("graph/edges.parquet")
  G = cugraph.Graph(directed=True)
  G.from_cudf_edgelist(edge_df, ...)
  ```

### 4.4 Expected Graph Statistics
| Metric | Estimate |
|--------|----------|
| Nodes | 3,200,000 |
| Edges (50 per node avg) | ~160,000,000 |
| Cross-field edges | ~20,000,000 (12.5%) |
| Storage (parquet) | ~5GB |

---

## Phase 5: Graph Analytics (cuGraph)

### 5.1 Core Algorithms
- [ ] **PageRank** (influence scoring)
  ```python
  def compute_pagerank(G, alpha=0.85, max_iter=100):
      pr = cugraph.pagerank(G, alpha=alpha, max_iter=max_iter)
      return pr  # DataFrame: vertex, pagerank
  ```

- [ ] **Community Detection** (research clusters)
  ```python
  def detect_communities(G, resolution=1.0):
      # Louvain for community detection
      parts, modularity = cugraph.louvain(G, resolution=resolution)
      return parts  # DataFrame: vertex, partition
  ```

- [ ] **Betweenness Centrality** (bridge papers)
  ```python
  def find_bridge_papers(G, k=1000):
      # Sample-based for large graphs
      bc = cugraph.betweenness_centrality(G, k=k)
      return bc.nlargest(100, 'betweenness_centrality')
  ```

- [ ] **Shortest Paths** (influence chains)
  ```python
  def trace_influence_path(G, source_id, target_id):
      # BFS-based shortest path
      distances, predecessors = cugraph.sssp(G, source_id)
      # Reconstruct path
      path = reconstruct_path(predecessors, source_id, target_id)
      return path
  ```

### 5.2 Custom Analytics for Hackathon Objectives

- [ ] **Objective 1: ML Impact Quantification**
  ```python
  def compute_ml_influence_score(G, ml_paper_ids, target_field):
      """
      Measure how much ML papers influence a target field.

      Score = Σ (PageRank of ML paper × edge weight to field papers)
      """
      ml_pagerank = pagerank_df[pagerank_df['vertex'].isin(ml_paper_ids)]

      # Get edges from ML papers to target field
      field_paper_ids = get_papers_by_field(target_field)
      influence_edges = edge_df[
          (edge_df['src'].isin(ml_paper_ids)) &
          (edge_df['dst'].isin(field_paper_ids))
      ]

      # Weighted influence score
      influence_score = (
          influence_edges
          .merge(ml_pagerank, left_on='src', right_on='vertex')
          .assign(weighted_influence=lambda x: x['pagerank'] * x['weight'])
          ['weighted_influence'].sum()
      )

      return influence_score
  ```

- [ ] **Objective 2: Adoption Dynamics**
  ```python
  def compute_adoption_flow(G, technique, year_range):
      """
      Track how a technique spreads across fields over time.

      Returns: DataFrame with year, field, adoption_count, cumulative_influence
      """
      # Find seed papers (early papers with technique)
      seed_papers = get_technique_papers(technique, year_range[0])

      # BFS from seeds, tracking field transitions
      adoption_timeline = []
      visited = set(seed_papers)
      current_frontier = seed_papers

      for year in range(year_range[0], year_range[1] + 1):
          # Get papers influenced this year
          next_frontier = get_neighbors_in_year(G, current_frontier, year)
          next_frontier -= visited

          # Count by field
          field_counts = count_by_field(next_frontier)
          for field, count in field_counts.items():
              adoption_timeline.append({
                  'year': year,
                  'field': field,
                  'technique': technique,
                  'new_adoptions': count,
                  'frontier_size': len(next_frontier)
              })

          visited |= next_frontier
          current_frontier = next_frontier

      return pd.DataFrame(adoption_timeline)
  ```

- [ ] **Objective 4: Discovery Paths**
  ```python
  def trace_discovery_impact(G, landmark_method, target_domain):
      """
      Trace path from ML method → domain application → impact.

      Example: transformer → drug_discovery → clinical_trial
      """
      # Find landmark papers
      landmark_papers = find_landmark_papers(landmark_method)

      # Find target domain papers
      target_papers = get_papers_by_field(target_domain)

      # Find shortest paths
      paths = []
      for src in landmark_papers[:10]:  # Top 10 landmarks
          for tgt in target_papers[:100]:  # Sample targets
              path = trace_influence_path(G, src, tgt)
              if path and len(path) <= 5:  # Max 5 hops
                  paths.append({
                      'source': src,
                      'target': tgt,
                      'path': path,
                      'length': len(path),
                      'fields_traversed': get_fields_in_path(path)
                  })

      return paths
  ```

---

## Phase 6: Hybrid Search System

### 6.1 Query Processing
- [ ] **Implement query encoder** (`src/search/query.py`)
  ```python
  class HybridQueryEncoder:
      def __init__(self, model_name="BAAI/bge-m3"):
          self.model = BGEM3FlagModel(model_name, use_fp16=True)

      def encode(self, query: str):
          output = self.model.encode(
              [query],
              return_dense=True,
              return_sparse=True
          )
          return {
              'dense': output['dense_vecs'][0],
              'sparse': output['lexical_weights'][0]
          }
  ```

### 6.2 Hybrid Retrieval
- [ ] **Implement RRF fusion** (`src/search/hybrid.py`)
  ```python
  def hybrid_search(query, top_k=100, alpha=0.5):
      """
      Combine dense and sparse search with Reciprocal Rank Fusion.

      Args:
          query: Natural language query
          top_k: Number of results
          alpha: Weight for dense vs sparse (0.5 = equal)
      """
      q_emb = query_encoder.encode(query)

      # Dense search (semantic)
      dense_scores, dense_ids = dense_index.search(q_emb['dense'], top_k * 2)

      # Sparse search (lexical)
      sparse_results = sparse_index.search(q_emb['sparse'], top_k * 2)

      # Reciprocal Rank Fusion
      fused = reciprocal_rank_fusion(
          [dense_ids, sparse_results['ids']],
          [dense_scores, sparse_results['scores']],
          k=60  # RRF parameter
      )

      return fused[:top_k]

  def reciprocal_rank_fusion(id_lists, score_lists, k=60):
      """
      RRF score = Σ 1/(k + rank_i)
      """
      scores = defaultdict(float)

      for ids, _ in zip(id_lists, score_lists):
          for rank, doc_id in enumerate(ids):
              scores[doc_id] += 1.0 / (k + rank + 1)

      sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
      return sorted_results
  ```

### 6.3 Re-ranking Stage
- [ ] **Implement cross-encoder reranker** (`src/search/rerank.py`)
  ```python
  from FlagEmbedding import FlagReranker

  class Reranker:
      def __init__(self, model_name="BAAI/bge-reranker-v2-m3"):
          self.reranker = FlagReranker(model_name, use_fp16=True)

      def rerank(self, query: str, candidates: List[dict], top_k=10):
          """
          Re-rank candidates with cross-encoder.
          Much more accurate but slower.
          """
          pairs = [[query, c['text']] for c in candidates]
          scores = self.reranker.compute_score(pairs)

          # Sort by reranker score
          ranked = sorted(
              zip(candidates, scores),
              key=lambda x: x[1],
              reverse=True
          )

          return [c for c, s in ranked[:top_k]]
  ```

### 6.4 Graph-Augmented Search
- [ ] **Implement influence-aware search** (`src/search/graph_search.py`)
  ```python
  class GraphAugmentedSearch:
      def __init__(self, hybrid_searcher, graph, pagerank_df):
          self.searcher = hybrid_searcher
          self.G = graph
          self.pagerank = pagerank_df

      def search_with_influence(self, query, top_k=20):
          """
          Search + boost by PageRank influence.
          """
          # Get hybrid search results
          candidates = self.searcher.search(query, top_k=top_k * 5)

          # Boost by PageRank
          boosted = []
          for doc_id, search_score in candidates:
              pr = self.pagerank.get(doc_id, 0.0)
              boosted_score = search_score * (1 + pr * 10)  # Boost factor
              boosted.append((doc_id, boosted_score))

          return sorted(boosted, key=lambda x: x[1], reverse=True)[:top_k]

      def search_influence_chain(self, query_source, query_target, max_hops=3):
          """
          Find papers connecting two concepts via influence paths.

          Example: "How did transformers influence drug discovery?"
          """
          # Find source papers (transformers)
          source_papers = self.searcher.search(query_source, top_k=20)

          # Find target papers (drug discovery)
          target_papers = self.searcher.search(query_target, top_k=20)

          # Find shortest paths in graph
          paths = []
          for src_id, _ in source_papers[:5]:
              for tgt_id, _ in target_papers[:10]:
                  path = find_path(self.G, src_id, tgt_id, max_hops)
                  if path:
                      paths.append({
                          'source': src_id,
                          'target': tgt_id,
                          'path': path,
                          'path_papers': [get_paper(p) for p in path]
                      })

          return sorted(paths, key=lambda x: len(x['path']))
  ```

---

## Phase 7: Natural Language Query Interface (GraphRAG)

### 7.1 Query Understanding
- [ ] **Implement intent classifier** (`src/rag/query_parser.py`)
  ```python
  QUERY_INTENTS = {
      'influence': r'how did (.+) influence (.+)',
      'adoption': r'which fields adopted (.+)',
      'path': r'path from (.+) to (.+)',
      'top': r'top (.+) papers in (.+)',
      'compare': r'compare (.+) (and|vs) (.+)',
      'trend': r'trend of (.+) in (.+)',
      'impact': r'impact of (.+) on (.+)'
  }

  def parse_query(query: str) -> dict:
      """Parse natural language query into structured intent."""
      query_lower = query.lower()

      for intent, pattern in QUERY_INTENTS.items():
          match = re.search(pattern, query_lower)
          if match:
              return {
                  'intent': intent,
                  'entities': match.groups(),
                  'raw_query': query
              }

      return {'intent': 'search', 'entities': [], 'raw_query': query}
  ```

### 7.2 Query Routing
- [ ] **Implement query router** (`src/rag/router.py`)
  ```python
  class QueryRouter:
      def __init__(self, hybrid_search, graph_search, analytics):
          self.hybrid = hybrid_search
          self.graph = graph_search
          self.analytics = analytics

      def route(self, parsed_query: dict):
          intent = parsed_query['intent']
          entities = parsed_query['entities']

          if intent == 'influence':
              return self.graph.search_influence_chain(entities[0], entities[1])

          elif intent == 'adoption':
              technique = entities[0]
              return self.analytics.compute_adoption_flow(technique)

          elif intent == 'path':
              return self.graph.find_shortest_paths(entities[0], entities[1])

          elif intent == 'top':
              field = entities[1] if len(entities) > 1 else None
              return self.hybrid.search_with_influence(entities[0], field=field)

          elif intent == 'compare':
              return self.analytics.compare_techniques(entities[0], entities[2])

          else:
              # Default: hybrid search
              return self.hybrid.search(parsed_query['raw_query'])
  ```

### 7.3 Response Generation
- [ ] **Implement response formatter** (`src/rag/response.py`)
  ```python
  def format_influence_response(paths, source, target):
      """Format influence path results for display."""
      if not paths:
          return f"No direct influence path found between {source} and {target}."

      response = f"## Influence Paths: {source} → {target}\n\n"
      response += f"Found {len(paths)} paths:\n\n"

      for i, path in enumerate(paths[:5]):
          response += f"### Path {i+1} ({len(path['path'])} hops)\n"
          for j, paper in enumerate(path['path_papers']):
              arrow = "→ " if j > 0 else ""
              response += f"{arrow}**{paper['title'][:60]}** ({paper['year']}, {paper['field']})\n"
          response += "\n"

      return response
  ```

---

## Phase 8: Dashboard & API Integration

### 8.1 Streamlit Dashboard Updates
- [ ] **Add semantic search tab** (`app.py`)
  ```python
  # New tab: Semantic Search
  with tabs[4]:
      st.header("Semantic Search & Influence Tracing")

      query = st.text_input("Enter your question:")

      if query:
          parsed = query_parser.parse(query)
          st.write(f"**Intent**: {parsed['intent']}")

          results = query_router.route(parsed)

          if parsed['intent'] == 'influence':
              display_influence_paths(results)
          elif parsed['intent'] == 'adoption':
              display_adoption_chart(results)
          else:
              display_search_results(results)
  ```

- [ ] **Add influence visualization**
  ```python
  def display_influence_paths(paths):
      """Visualize influence paths as network graph."""
      import plotly.graph_objects as go

      # Build network visualization
      fig = create_network_graph(paths)
      st.plotly_chart(fig)

      # Show path details
      for path in paths[:5]:
          with st.expander(f"Path: {path['source_title'][:30]}... → {path['target_title'][:30]}..."):
              for paper in path['path_papers']:
                  st.write(f"- **{paper['title']}** ({paper['year']}, {paper['field']})")
  ```

### 8.2 FastAPI Endpoints
- [ ] **Add search endpoints** (`api.py`)
  ```python
  @app.post("/api/search")
  async def semantic_search(query: str, top_k: int = 20):
      results = hybrid_search.search(query, top_k)
      return {"results": results}

  @app.post("/api/influence")
  async def trace_influence(source: str, target: str):
      paths = graph_search.search_influence_chain(source, target)
      return {"paths": paths}

  @app.get("/api/adoption/{technique}")
  async def get_adoption_flow(technique: str):
      flow = analytics.compute_adoption_flow(technique)
      return {"adoption_flow": flow.to_dict('records')}

  @app.get("/api/graph/stats")
  async def get_graph_stats():
      return {
          "nodes": len(G.nodes()),
          "edges": len(G.edges()),
          "communities": len(set(communities.values())),
          "top_pagerank": pagerank.nlargest(10, 'pagerank').to_dict('records')
      }
  ```

---

## Phase 9: Testing & Validation

### 9.1 Embedding Quality Tests
- [ ] **Similarity sanity checks**
  ```python
  def test_embedding_quality():
      # Known similar papers should have high similarity
      transformer_papers = ["Attention Is All You Need", "BERT: Pre-training"]
      drug_papers = ["Drug Discovery with ML", "Pharmaceutical AI"]

      # Same domain = high similarity
      assert cosine_sim(embed(transformer_papers[0]), embed(transformer_papers[1])) > 0.8

      # Different domain = lower similarity
      assert cosine_sim(embed(transformer_papers[0]), embed(drug_papers[0])) < 0.7
  ```

### 9.2 Graph Quality Tests
- [ ] **Edge validity checks**
  ```python
  def test_graph_quality():
      # Cross-field edges should exist
      cs_papers = get_papers_by_field("Computer Science")
      bio_papers = get_papers_by_field("Biology")
      cross_edges = get_edges_between(cs_papers, bio_papers)
      assert len(cross_edges) > 1000, "Too few cross-field edges"

      # Temporal direction should be correct
      for edge in sample(edges, 100):
          src_year = get_paper(edge['source'])['year']
          tgt_year = get_paper(edge['target'])['year']
          assert src_year <= tgt_year, "Edge direction wrong"
  ```

### 9.3 Search Quality Tests
- [ ] **Retrieval accuracy**
  ```python
  def test_search_quality():
      # Query about transformers should return transformer papers
      results = hybrid_search("transformer attention mechanism NLP")
      titles = [r['title'].lower() for r in results[:10]]
      assert any('transformer' in t or 'attention' in t for t in titles)

      # Influence query should return paths
      paths = graph_search.search_influence_chain("deep learning", "drug discovery")
      assert len(paths) > 0, "No influence paths found"
  ```

---

## Phase 10: Optimization & DGX Spark Deployment

### 10.1 Memory Optimization
- [ ] **Quantize embeddings for inference**
  ```python
  # FP16 quantization (50% memory reduction)
  embeddings_fp16 = embeddings.astype(np.float16)

  # INT8 quantization (75% memory reduction, slight accuracy loss)
  from scipy.cluster.vq import kmeans, vq
  codebook, _ = kmeans(embeddings, 256)
  quantized, _ = vq(embeddings, codebook)
  ```

- [ ] **Memory-mapped loading**
  ```python
  # Load embeddings as memory-mapped (doesn't load all to RAM)
  embeddings = np.load("embeddings/dense.npy", mmap_mode='r')
  ```

### 10.2 Batch Processing
- [ ] **Batch queries for throughput**
  ```python
  async def batch_search(queries: List[str], batch_size=32):
      results = []
      for i in range(0, len(queries), batch_size):
          batch = queries[i:i+batch_size]
          batch_embeddings = encoder.encode(batch)
          batch_results = index.search(batch_embeddings, k=20)
          results.extend(batch_results)
      return results
  ```

### 10.3 Caching
- [ ] **Result caching**
  ```python
  from functools import lru_cache
  import hashlib

  @lru_cache(maxsize=10000)
  def cached_search(query_hash: str):
      # Actual search implementation
      ...

  def search(query: str):
      query_hash = hashlib.md5(query.encode()).hexdigest()
      return cached_search(query_hash)
  ```

---

## File Structure (Final)

```
ai_research/
├── src/
│   ├── data/
│   │   ├── loader.py           # Existing - data loading
│   │   └── preprocessor.py     # Enhanced - text preprocessing
│   │
│   ├── embeddings/             # NEW
│   │   ├── __init__.py
│   │   ├── config.py           # Embedding model config
│   │   ├── batch_embed.py      # Multi-GPU embedding pipeline
│   │   └── encoder.py          # Query encoder
│   │
│   ├── index/                  # NEW
│   │   ├── __init__.py
│   │   ├── dense_index.py      # cuVS/FAISS index
│   │   ├── sparse_index.py     # BM25 index
│   │   └── hybrid_index.py     # Combined index
│   │
│   ├── graph/
│   │   ├── citation_graph.py   # Existing - S2 API (optional)
│   │   ├── graph_analysis.py   # Existing - cuGraph analytics
│   │   ├── edge_builder.py     # NEW - build edges from embeddings
│   │   └── graph_rag.py        # Enhanced - query interface
│   │
│   ├── search/                 # NEW
│   │   ├── __init__.py
│   │   ├── query.py            # Query encoding
│   │   ├── hybrid.py           # Hybrid search + RRF
│   │   ├── rerank.py           # Cross-encoder reranking
│   │   └── graph_search.py     # Graph-augmented search
│   │
│   ├── rag/                    # NEW
│   │   ├── __init__.py
│   │   ├── query_parser.py     # Intent classification
│   │   ├── router.py           # Query routing
│   │   └── response.py         # Response formatting
│   │
│   └── metrics/
│       └── impact_metrics.py   # Existing - hackathon metrics
│
├── embeddings/                 # Generated embeddings
│   ├── dense/
│   ├── sparse/
│   └── metadata/
│
├── indexes/                    # Vector indexes
│   ├── cagra_3.2m.bin
│   └── sparse_bm25.pkl
│
├── graph/                      # Graph data
│   ├── edges.parquet
│   └── nodes.parquet
│
├── app.py                      # Streamlit dashboard
├── api.py                      # FastAPI backend
├── requirements.txt            # Updated dependencies
└── plan.md                     # This file
```

---

## Dependencies (Updated requirements.txt)

```
# Core
pandas>=2.0.0
numpy>=1.24.0
pyarrow>=14.0.0

# Embeddings
FlagEmbedding>=1.2.0
sentence-transformers>=2.2.0
torch>=2.0.0

# Vector Search
faiss-gpu>=1.7.4
# cuvs>=24.10  # Install via conda

# Graph
networkx>=3.2
# cugraph>=24.10  # Install via conda

# Search
rank-bm25>=0.2.2

# Web
streamlit>=1.29.0
fastapi>=0.104.0
uvicorn>=0.24.0
plotly>=5.18.0

# Utilities
tqdm>=4.66.0
orjson>=3.9.0
scipy>=1.11.0
```

---

## Timeline Estimate

| Phase | Task | Duration |
|-------|------|----------|
| 1 | Data Preparation | 2-3 hours |
| 2 | Embedding Generation | 45 hours (GPU) |
| 3 | Index Construction | 2-3 hours |
| 4 | Graph Construction | 4-6 hours |
| 5 | Graph Analytics | 2-3 hours |
| 6 | Hybrid Search | 3-4 hours |
| 7 | GraphRAG Interface | 3-4 hours |
| 8 | Dashboard Integration | 2-3 hours |
| 9 | Testing | 2-3 hours |
| 10 | Optimization | 2-3 hours |
| **Total** | | **~70 hours** |

**Critical Path**: Embedding generation (45h) is the bottleneck. Start this first, parallelize other work.

---

## DGX Spark Story

> "We embed 3.2 million scientific papers using BGE-M3's 8K context window, generating both dense semantic vectors and sparse lexical representations. These embeddings are indexed using NVIDIA cuVS CAGRA—a GPU-native graph index that builds 21x faster than CPU alternatives. From the dense vectors, we construct a 160-million-edge semantic similarity graph and analyze it with cuGraph to compute PageRank influence scores, detect research communities, and trace how ML techniques propagate across scientific disciplines. The entire system—embeddings, indexes, and graph—fits in the DGX Spark's 128GB unified memory, enabling real-time influence queries like 'How did transformers impact drug discovery?' with sub-second response times."

---

## Success Criteria

- [ ] 3.2M papers embedded with BGE-M3
- [ ] Hybrid search returning relevant results in <100ms
- [ ] Semantic graph with >100M edges built from embeddings
- [ ] Influence paths traceable between any two papers/concepts
- [ ] All 4 hackathon objectives enhanced with graph-based insights
- [ ] Dashboard demo running on DGX Spark

---

# DETAILED IMPLEMENTATION GUIDE

## Phase 1: Data Preparation — Step by Step

### Step 1.1: Audit Dataset Structure
```bash
# Run from ai_research directory
python3 -c "
from pathlib import Path
import json

data_dir = Path('s2orc_data')
manifest = {
    'total_files': 0,
    'total_papers': 0,
    'fields': {},
    'years': set(),
    'files': []
}

for subdir in sorted(data_dir.iterdir()):
    if not subdir.is_dir():
        continue
    field = subdir.name.split(',')[0]
    manifest['fields'][field] = manifest['fields'].get(field, 0)

    for split in ['train', 'val', 'test']:
        split_dir = subdir / split
        if not split_dir.exists():
            continue
        for f in split_dir.glob('*.json*'):
            manifest['total_files'] += 1
            # Count lines (papers) in file
            with open(f, 'r') as fp:
                count = sum(1 for _ in fp)
            manifest['total_papers'] += count
            manifest['fields'][field] += count
            manifest['files'].append({
                'path': str(f),
                'papers': count,
                'field': field
            })

manifest['years'] = list(manifest['years'])
print(json.dumps(manifest, indent=2))
with open('data_manifest.json', 'w') as f:
    json.dump(manifest, f, indent=2)
"
```

### Step 1.2: Create Text Preprocessor
```python
# src/data/preprocessor.py
import re
from typing import Optional, Tuple

class TextPreprocessor:
    """Clean and prepare paper text for embedding."""

    # Sections to remove (low signal)
    REMOVE_SECTIONS = [
        r'(?i)acknowledgment.*?(?=\n[A-Z]|\Z)',
        r'(?i)references.*?(?=\Z)',
        r'(?i)bibliography.*?(?=\Z)',
        r'(?i)funding.*?(?=\n[A-Z]|\Z)',
    ]

    # Max tokens for BGE-M3
    MAX_TOKENS = 8192
    CHARS_PER_TOKEN = 4  # Approximate
    MAX_CHARS = MAX_TOKENS * CHARS_PER_TOKEN

    @classmethod
    def clean(cls, text: str) -> str:
        """Clean paper text."""
        if not text:
            return ""

        # Remove boilerplate sections
        for pattern in cls.REMOVE_SECTIONS:
            text = re.sub(pattern, '', text, flags=re.DOTALL)

        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()

        return text

    @classmethod
    def truncate_smart(cls, text: str, max_chars: int = None) -> str:
        """
        Smart truncation prioritizing important sections.
        Priority: Title > Abstract > Introduction > Methods > Results
        """
        if max_chars is None:
            max_chars = cls.MAX_CHARS

        if len(text) <= max_chars:
            return text

        # Try to find abstract
        abstract_match = re.search(
            r'(?i)(abstract|summary)[:\s]*(.+?)(?=\n\s*\n|\n[A-Z][a-z]+:)',
            text,
            re.DOTALL
        )

        # Try to find introduction
        intro_match = re.search(
            r'(?i)(introduction|background)[:\s]*(.+?)(?=\n\s*\n[A-Z])',
            text,
            re.DOTALL
        )

        # Build prioritized text
        parts = []

        # Title (first line)
        title = text.split('\n')[0][:500]
        parts.append(title)
        remaining = max_chars - len(title)

        # Abstract
        if abstract_match and remaining > 0:
            abstract = abstract_match.group(2)[:remaining//2]
            parts.append(f"\nAbstract: {abstract}")
            remaining -= len(abstract)

        # Introduction
        if intro_match and remaining > 0:
            intro = intro_match.group(2)[:remaining//2]
            parts.append(f"\nIntroduction: {intro}")
            remaining -= len(intro)

        # Fill with remaining text
        if remaining > 0:
            start_idx = len('\n'.join(parts))
            parts.append(text[start_idx:start_idx + remaining])

        return ''.join(parts)

    @classmethod
    def extract_title(cls, text: str) -> str:
        """Extract paper title (first line)."""
        if not text:
            return ""
        return text.split('\n')[0][:300]

    @classmethod
    def process(cls, text: str) -> Tuple[str, str]:
        """
        Full preprocessing pipeline.
        Returns: (processed_text, title)
        """
        cleaned = cls.clean(text)
        truncated = cls.truncate_smart(cleaned)
        title = cls.extract_title(cleaned)
        return truncated, title
```

### Step 1.3: Create Paper Iterator with Preprocessing
```python
# src/data/paper_iterator.py
import json
from pathlib import Path
from typing import Iterator, Dict, Optional
from .preprocessor import TextPreprocessor

class PaperIterator:
    """Memory-efficient iterator over all papers."""

    def __init__(self, data_dir: str = "s2orc_data"):
        self.data_dir = Path(data_dir)
        self.preprocessor = TextPreprocessor()

    def __iter__(self) -> Iterator[Dict]:
        """Iterate over all papers with preprocessing."""
        for subdir in sorted(self.data_dir.iterdir()):
            if not subdir.is_dir() or ',' not in subdir.name:
                continue

            field = subdir.name.split(',')[0]

            for split in ['train', 'val', 'test']:
                split_dir = subdir / split
                if not split_dir.exists():
                    continue

                for filepath in split_dir.glob('*.json*'):
                    yield from self._read_file(filepath, field, split)

    def _read_file(self, filepath: Path, field: str, split: str) -> Iterator[Dict]:
        """Read and preprocess papers from a single file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                try:
                    paper = json.loads(line)

                    # Preprocess text
                    text = paper.get('text', '')
                    processed_text, title = self.preprocessor.process(text)

                    # Extract metadata
                    metadata = paper.get('metadata', {})
                    year = metadata.get('year', 0)
                    if not year:
                        created = paper.get('created', '')
                        if created and len(created) >= 4:
                            try:
                                year = int(created[:4])
                            except:
                                year = 0

                    yield {
                        'id': str(paper.get('id', f"{filepath.stem}_{line_num}")),
                        'text': processed_text,
                        'title': title,
                        'year': year,
                        'field': field,
                        'split': split,
                        's2_fields': metadata.get('s2fieldsofstudy', []) or [],
                        'ext_fields': metadata.get('extfieldsofstudy', []) or [],
                        'text_length': len(processed_text),
                        'source_file': str(filepath)
                    }
                except json.JSONDecodeError:
                    continue
                except Exception as e:
                    continue

    def count(self) -> int:
        """Count total papers."""
        return sum(1 for _ in self)
```

### Step 1.4: Validate Data Quality
```python
# scripts/validate_data.py
from src.data.paper_iterator import PaperIterator
from collections import defaultdict
import json

def validate_dataset():
    iterator = PaperIterator("s2orc_data")

    stats = {
        'total': 0,
        'missing_text': 0,
        'missing_year': 0,
        'by_field': defaultdict(int),
        'by_year': defaultdict(int),
        'text_lengths': [],
    }

    for paper in iterator:
        stats['total'] += 1

        if not paper['text']:
            stats['missing_text'] += 1
        if not paper['year']:
            stats['missing_year'] += 1

        stats['by_field'][paper['field']] += 1
        stats['by_year'][paper['year']] += 1
        stats['text_lengths'].append(paper['text_length'])

        if stats['total'] % 100000 == 0:
            print(f"Processed {stats['total']:,} papers...")

    # Compute statistics
    stats['avg_text_length'] = sum(stats['text_lengths']) / len(stats['text_lengths'])
    stats['text_lengths'] = None  # Don't save all lengths

    print(f"\n=== Dataset Validation ===")
    print(f"Total papers: {stats['total']:,}")
    print(f"Missing text: {stats['missing_text']:,} ({stats['missing_text']/stats['total']:.2%})")
    print(f"Missing year: {stats['missing_year']:,} ({stats['missing_year']/stats['total']:.2%})")
    print(f"Avg text length: {stats['avg_text_length']:.0f} chars")
    print(f"\nBy field:")
    for field, count in sorted(stats['by_field'].items(), key=lambda x: -x[1]):
        print(f"  {field}: {count:,}")

    with open('data_validation.json', 'w') as f:
        json.dump(dict(stats), f, indent=2, default=str)

if __name__ == "__main__":
    validate_dataset()
```

---

## Phase 2: Embedding Generation — Step by Step

### Step 2.1: Install Dependencies
```bash
# On A100 cluster
pip install FlagEmbedding torch transformers accelerate

# Verify GPU
python3 -c "import torch; print(f'GPUs: {torch.cuda.device_count()}')"
```

### Step 2.2: Create Embedding Configuration
```python
# src/embeddings/config.py
from dataclasses import dataclass
from typing import Optional

@dataclass
class EmbeddingConfig:
    model_name: str = "BAAI/bge-m3"
    max_length: int = 8192
    batch_size: int = 32
    use_fp16: bool = True
    normalize_embeddings: bool = True
    return_dense: bool = True
    return_sparse: bool = True
    return_colbert: bool = False  # Too memory-intensive

    # Output paths
    output_dir: str = "embeddings"
    checkpoint_every: int = 100000  # Save every N papers

    # Multi-GPU settings
    num_gpus: int = 3

    @property
    def dense_dim(self) -> int:
        return 1024  # BGE-M3 dimension
```

### Step 2.3: Create Batch Embedding Pipeline
```python
# src/embeddings/batch_embed.py
import os
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Generator
from tqdm import tqdm
import torch
from FlagEmbedding import BGEM3FlagModel

from .config import EmbeddingConfig
from src.data.paper_iterator import PaperIterator

class BatchEmbedder:
    """Batch embedding pipeline for large-scale paper corpus."""

    def __init__(self, config: EmbeddingConfig, gpu_id: int = 0):
        self.config = config
        self.gpu_id = gpu_id

        # Set GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        # Load model
        print(f"[GPU {gpu_id}] Loading model {config.model_name}...")
        self.model = BGEM3FlagModel(
            config.model_name,
            use_fp16=config.use_fp16,
            device=f'cuda:0'
        )

        # Setup output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / 'dense').mkdir(exist_ok=True)
        (self.output_dir / 'sparse').mkdir(exist_ok=True)
        (self.output_dir / 'metadata').mkdir(exist_ok=True)

    def embed_batch(self, texts: List[str]) -> Dict[str, np.ndarray]:
        """Embed a batch of texts."""
        output = self.model.encode(
            texts,
            batch_size=self.config.batch_size,
            max_length=self.config.max_length,
            return_dense=self.config.return_dense,
            return_sparse=self.config.return_sparse,
            return_colbert_vecs=self.config.return_colbert
        )

        return {
            'dense': np.array(output['dense_vecs']),
            'sparse': output.get('lexical_weights', None)
        }

    def process_corpus(
        self,
        paper_iterator: Generator[Dict, None, None],
        start_idx: int = 0,
        end_idx: int = None
    ):
        """Process entire corpus with checkpointing."""

        # Progress tracking
        progress_file = self.output_dir / f'progress_gpu{self.gpu_id}.json'
        if progress_file.exists():
            with open(progress_file) as f:
                progress = json.load(f)
            start_idx = progress.get('last_idx', 0)
            print(f"[GPU {self.gpu_id}] Resuming from index {start_idx}")

        # Buffers
        dense_buffer = []
        sparse_buffer = []
        id_buffer = []
        metadata_buffer = []

        chunk_idx = start_idx // self.config.checkpoint_every

        for idx, paper in enumerate(tqdm(paper_iterator, desc=f"GPU {self.gpu_id}")):
            if idx < start_idx:
                continue
            if end_idx and idx >= end_idx:
                break

            # Collect batch
            id_buffer.append(paper['id'])
            metadata_buffer.append({
                'id': paper['id'],
                'title': paper['title'],
                'year': paper['year'],
                'field': paper['field'],
                'text_length': paper['text_length']
            })

            # When batch is full, embed
            if len(id_buffer) >= self.config.batch_size:
                # Get texts for this batch
                texts = [p['text'] for p in metadata_buffer[-self.config.batch_size:]]

                # Embed
                embeddings = self.embed_batch(texts)
                dense_buffer.append(embeddings['dense'])
                if embeddings['sparse']:
                    sparse_buffer.extend(embeddings['sparse'])

            # Checkpoint
            if len(id_buffer) >= self.config.checkpoint_every:
                self._save_checkpoint(
                    chunk_idx, dense_buffer, sparse_buffer,
                    id_buffer, metadata_buffer
                )

                # Clear buffers
                dense_buffer = []
                sparse_buffer = []
                id_buffer = []
                metadata_buffer = []
                chunk_idx += 1

                # Save progress
                with open(progress_file, 'w') as f:
                    json.dump({'last_idx': idx + 1}, f)

        # Save final chunk
        if id_buffer:
            self._save_checkpoint(
                chunk_idx, dense_buffer, sparse_buffer,
                id_buffer, metadata_buffer
            )

    def _save_checkpoint(
        self,
        chunk_idx: int,
        dense_buffer: List[np.ndarray],
        sparse_buffer: List,
        id_buffer: List[str],
        metadata_buffer: List[Dict]
    ):
        """Save checkpoint to disk."""
        print(f"[GPU {self.gpu_id}] Saving chunk {chunk_idx}...")

        # Dense embeddings
        if dense_buffer:
            dense_array = np.vstack(dense_buffer)
            np.save(
                self.output_dir / 'dense' / f'chunk_{chunk_idx:04d}.npy',
                dense_array.astype(np.float16)  # Save as FP16
            )

        # Sparse embeddings (as pickle for now)
        if sparse_buffer:
            import pickle
            with open(self.output_dir / 'sparse' / f'chunk_{chunk_idx:04d}.pkl', 'wb') as f:
                pickle.dump(sparse_buffer, f)

        # Metadata
        with open(self.output_dir / 'metadata' / f'chunk_{chunk_idx:04d}.json', 'w') as f:
            json.dump({
                'paper_ids': id_buffer,
                'papers': metadata_buffer
            }, f)


def run_multi_gpu_embedding(num_gpus: int = 3):
    """Run embedding across multiple GPUs."""
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp

    # Count total papers
    iterator = PaperIterator("s2orc_data")
    total = sum(1 for _ in iterator)
    print(f"Total papers: {total:,}")

    # Split across GPUs
    papers_per_gpu = total // num_gpus

    def run_gpu(gpu_id: int):
        config = EmbeddingConfig(num_gpus=num_gpus)
        embedder = BatchEmbedder(config, gpu_id=gpu_id)
        iterator = PaperIterator("s2orc_data")

        start_idx = gpu_id * papers_per_gpu
        end_idx = start_idx + papers_per_gpu if gpu_id < num_gpus - 1 else None

        embedder.process_corpus(iterator, start_idx, end_idx)

    # Launch processes
    mp.set_start_method('spawn', force=True)
    with ProcessPoolExecutor(max_workers=num_gpus) as executor:
        futures = [executor.submit(run_gpu, i) for i in range(num_gpus)]
        for f in futures:
            f.result()

    print("Embedding complete!")


if __name__ == "__main__":
    run_multi_gpu_embedding(num_gpus=3)
```

### Step 2.4: Merge Embedding Chunks
```python
# scripts/merge_embeddings.py
import numpy as np
from pathlib import Path
import json

def merge_embeddings(embedding_dir: str = "embeddings"):
    """Merge chunked embeddings into single files."""
    emb_dir = Path(embedding_dir)

    # Merge dense embeddings
    print("Merging dense embeddings...")
    dense_chunks = sorted((emb_dir / 'dense').glob('chunk_*.npy'))
    dense_arrays = [np.load(f) for f in dense_chunks]
    dense_merged = np.vstack(dense_arrays)
    np.save(emb_dir / 'dense_all.npy', dense_merged)
    print(f"Dense shape: {dense_merged.shape}")

    # Merge metadata
    print("Merging metadata...")
    meta_chunks = sorted((emb_dir / 'metadata').glob('chunk_*.json'))
    all_ids = []
    all_meta = []
    for f in meta_chunks:
        with open(f) as fp:
            data = json.load(fp)
            all_ids.extend(data['paper_ids'])
            all_meta.extend(data['papers'])

    # Create ID mapping
    id_to_idx = {pid: idx for idx, pid in enumerate(all_ids)}
    with open(emb_dir / 'id_to_idx.json', 'w') as f:
        json.dump(id_to_idx, f)

    with open(emb_dir / 'metadata_all.json', 'w') as f:
        json.dump(all_meta, f)

    print(f"Total papers: {len(all_ids)}")
    print("Merge complete!")

if __name__ == "__main__":
    merge_embeddings()
```

---

## Phase 3: Vector Index Construction — Step by Step

### Step 3.1: Build FAISS Index
```python
# src/index/dense_index.py
import numpy as np
import faiss
from pathlib import Path
from typing import Tuple, List
import json

class DenseIndex:
    """GPU-accelerated dense vector index using FAISS."""

    def __init__(self, dim: int = 1024):
        self.dim = dim
        self.index = None
        self.id_map = None  # idx -> paper_id

    def build(
        self,
        embeddings: np.ndarray,
        paper_ids: List[str],
        use_gpu: bool = True,
        index_type: str = "IVF4096,PQ64"
    ):
        """
        Build index from embeddings.

        Args:
            embeddings: (N, dim) array
            paper_ids: List of paper IDs matching embeddings
            use_gpu: Use GPU acceleration
            index_type: FAISS index type
        """
        print(f"Building index for {len(embeddings):,} vectors...")

        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)

        # Create index
        self.index = faiss.index_factory(
            self.dim, index_type, faiss.METRIC_INNER_PRODUCT
        )

        # Train on subset
        train_size = min(500000, len(embeddings))
        train_idx = np.random.choice(len(embeddings), train_size, replace=False)
        print(f"Training on {train_size:,} vectors...")
        self.index.train(embeddings[train_idx])

        # Add all vectors
        if use_gpu:
            print("Moving to GPU...")
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)

        print("Adding vectors...")
        self.index.add(embeddings)

        # Store ID mapping
        self.id_map = {i: pid for i, pid in enumerate(paper_ids)}

        print(f"Index built: {self.index.ntotal:,} vectors")

    def search(
        self,
        query_vectors: np.ndarray,
        k: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for similar vectors.

        Returns:
            scores: (n_queries, k) similarity scores
            indices: (n_queries, k) vector indices
        """
        faiss.normalize_L2(query_vectors)
        scores, indices = self.index.search(query_vectors, k)
        return scores, indices

    def get_paper_ids(self, indices: np.ndarray) -> List[List[str]]:
        """Convert indices to paper IDs."""
        return [[self.id_map[i] for i in row] for row in indices]

    def save(self, path: str):
        """Save index to disk."""
        path = Path(path)

        # Move to CPU if on GPU
        if hasattr(self.index, 'index'):
            cpu_index = faiss.index_gpu_to_cpu(self.index)
        else:
            cpu_index = self.index

        faiss.write_index(cpu_index, str(path / 'faiss_index.bin'))

        with open(path / 'id_map.json', 'w') as f:
            json.dump(self.id_map, f)

        print(f"Index saved to {path}")

    def load(self, path: str, use_gpu: bool = True):
        """Load index from disk."""
        path = Path(path)

        self.index = faiss.read_index(str(path / 'faiss_index.bin'))

        if use_gpu:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)

        with open(path / 'id_map.json') as f:
            self.id_map = {int(k): v for k, v in json.load(f).items()}

        print(f"Index loaded: {self.index.ntotal:,} vectors")


# Build script
def build_dense_index():
    """Build dense index from embeddings."""
    from pathlib import Path

    print("Loading embeddings...")
    embeddings = np.load("embeddings/dense_all.npy").astype(np.float32)

    with open("embeddings/id_to_idx.json") as f:
        id_to_idx = json.load(f)

    # Invert mapping
    paper_ids = [None] * len(id_to_idx)
    for pid, idx in id_to_idx.items():
        paper_ids[idx] = pid

    # Build index
    index = DenseIndex(dim=embeddings.shape[1])
    index.build(embeddings, paper_ids, use_gpu=True)

    # Save
    Path("indexes").mkdir(exist_ok=True)
    index.save("indexes")

    # Test search
    print("\nTesting search...")
    scores, indices = index.search(embeddings[:5], k=10)
    print(f"Top result scores: {scores[0][:5]}")

if __name__ == "__main__":
    build_dense_index()
```

### Step 3.2: Build Sparse Index (BM25)
```python
# src/index/sparse_index.py
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict
import math

class SparseIndex:
    """BM25-based sparse index using BGE-M3 lexical weights."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.inverted_index = defaultdict(list)  # token -> [(doc_id, weight), ...]
        self.doc_lengths = {}
        self.avg_doc_length = 0
        self.num_docs = 0
        self.paper_ids = []

    def build(self, sparse_vectors: List[Dict], paper_ids: List[str]):
        """
        Build inverted index from BGE-M3 sparse outputs.

        Args:
            sparse_vectors: List of {token_id: weight} dicts
            paper_ids: List of paper IDs
        """
        print(f"Building sparse index for {len(sparse_vectors):,} documents...")

        self.paper_ids = paper_ids
        self.num_docs = len(paper_ids)

        total_length = 0
        for doc_idx, (sparse_vec, pid) in enumerate(zip(sparse_vectors, paper_ids)):
            if sparse_vec is None:
                continue

            doc_length = sum(sparse_vec.values())
            self.doc_lengths[doc_idx] = doc_length
            total_length += doc_length

            for token_id, weight in sparse_vec.items():
                self.inverted_index[token_id].append((doc_idx, weight))

            if doc_idx % 100000 == 0:
                print(f"  Processed {doc_idx:,} documents...")

        self.avg_doc_length = total_length / max(self.num_docs, 1)
        print(f"Index built: {len(self.inverted_index):,} unique tokens")

    def search(
        self,
        query_sparse: Dict,
        k: int = 100
    ) -> Tuple[List[str], List[float]]:
        """
        Search using BM25 scoring.

        Args:
            query_sparse: {token_id: weight} from BGE-M3
            k: Number of results

        Returns:
            paper_ids: Top-k paper IDs
            scores: Corresponding scores
        """
        scores = defaultdict(float)

        for token_id, query_weight in query_sparse.items():
            if token_id not in self.inverted_index:
                continue

            posting_list = self.inverted_index[token_id]
            idf = math.log((self.num_docs - len(posting_list) + 0.5) / (len(posting_list) + 0.5) + 1)

            for doc_idx, doc_weight in posting_list:
                doc_length = self.doc_lengths.get(doc_idx, self.avg_doc_length)

                # BM25 formula
                numerator = doc_weight * (self.k1 + 1)
                denominator = doc_weight + self.k1 * (1 - self.b + self.b * doc_length / self.avg_doc_length)

                scores[doc_idx] += idf * (numerator / denominator) * query_weight

        # Sort by score
        sorted_results = sorted(scores.items(), key=lambda x: -x[1])[:k]

        result_ids = [self.paper_ids[idx] for idx, _ in sorted_results]
        result_scores = [score for _, score in sorted_results]

        return result_ids, result_scores

    def save(self, path: str):
        """Save index to disk."""
        path = Path(path)
        with open(path / 'sparse_index.pkl', 'wb') as f:
            pickle.dump({
                'inverted_index': dict(self.inverted_index),
                'doc_lengths': self.doc_lengths,
                'avg_doc_length': self.avg_doc_length,
                'num_docs': self.num_docs,
                'paper_ids': self.paper_ids,
                'k1': self.k1,
                'b': self.b
            }, f)
        print(f"Sparse index saved to {path}")

    def load(self, path: str):
        """Load index from disk."""
        path = Path(path)
        with open(path / 'sparse_index.pkl', 'rb') as f:
            data = pickle.load(f)

        self.inverted_index = defaultdict(list, data['inverted_index'])
        self.doc_lengths = data['doc_lengths']
        self.avg_doc_length = data['avg_doc_length']
        self.num_docs = data['num_docs']
        self.paper_ids = data['paper_ids']
        self.k1 = data['k1']
        self.b = data['b']

        print(f"Sparse index loaded: {self.num_docs:,} docs, {len(self.inverted_index):,} tokens")
```

---

## Phase 4: Semantic Graph Construction — Step by Step

### Step 4.1: Build Edges from Similarity
```python
# src/graph/edge_builder.py
import numpy as np
from typing import List, Dict, Tuple, Generator
from pathlib import Path
import json
from tqdm import tqdm

class SemanticGraphBuilder:
    """Build semantic similarity graph from embeddings."""

    def __init__(
        self,
        dense_index,
        metadata: List[Dict],
        similarity_threshold: float = 0.75,
        cross_field_threshold: float = 0.70,
        max_edges_per_node: int = 50
    ):
        self.index = dense_index
        self.metadata = metadata
        self.sim_threshold = similarity_threshold
        self.cross_threshold = cross_field_threshold
        self.max_edges = max_edges_per_node

        # Build metadata lookup
        self.meta_lookup = {m['id']: m for m in metadata}

    def build_edges(
        self,
        embeddings: np.ndarray,
        batch_size: int = 10000
    ) -> Generator[Dict, None, None]:
        """
        Build edges by finding similar papers.

        Yields edge dicts: {source, target, weight, type, year_gap}
        """
        n_papers = len(embeddings)
        print(f"Building edges for {n_papers:,} papers...")

        for batch_start in tqdm(range(0, n_papers, batch_size)):
            batch_end = min(batch_start + batch_size, n_papers)
            batch = embeddings[batch_start:batch_end]

            # Find top-K similar for each paper in batch
            scores, indices = self.index.search(batch, k=self.max_edges * 2)

            for i, (score_row, idx_row) in enumerate(zip(scores, indices)):
                source_idx = batch_start + i
                source_meta = self.metadata[source_idx]

                edges_added = 0
                for score, target_idx in zip(score_row, idx_row):
                    if target_idx == source_idx:
                        continue
                    if edges_added >= self.max_edges:
                        break

                    target_meta = self.metadata[target_idx]

                    # Check if edge is valid
                    edge = self._create_edge(
                        source_idx, target_idx,
                        source_meta, target_meta,
                        float(score)
                    )

                    if edge:
                        edges_added += 1
                        yield edge

    def _create_edge(
        self,
        src_idx: int,
        tgt_idx: int,
        src_meta: Dict,
        tgt_meta: Dict,
        similarity: float
    ) -> Dict:
        """Create edge if valid, with temporal direction."""

        # Determine edge type
        cross_field = src_meta['field'] != tgt_meta['field']
        threshold = self.cross_threshold if cross_field else self.sim_threshold

        if similarity < threshold:
            return None

        # Ensure temporal direction: older → newer
        src_year = src_meta.get('year', 0)
        tgt_year = tgt_meta.get('year', 0)

        if src_year > tgt_year:
            # Swap direction
            src_idx, tgt_idx = tgt_idx, src_idx
            src_meta, tgt_meta = tgt_meta, src_meta
            src_year, tgt_year = tgt_year, src_year

        edge_type = 'cross_field_bridge' if cross_field else 'temporal_influence'

        return {
            'source': src_idx,
            'target': tgt_idx,
            'source_id': src_meta['id'],
            'target_id': tgt_meta['id'],
            'weight': similarity,
            'type': edge_type,
            'year_gap': tgt_year - src_year,
            'source_field': src_meta['field'],
            'target_field': tgt_meta['field']
        }

    def save_edges(self, edges: List[Dict], output_path: str):
        """Save edges to parquet."""
        import pandas as pd

        df = pd.DataFrame(edges)
        df.to_parquet(output_path, index=False)
        print(f"Saved {len(df):,} edges to {output_path}")

        # Statistics
        print(f"\nEdge Statistics:")
        print(f"  Total edges: {len(df):,}")
        print(f"  Cross-field: {(df['type'] == 'cross_field_bridge').sum():,}")
        print(f"  Avg weight: {df['weight'].mean():.3f}")
        print(f"  Avg year gap: {df['year_gap'].mean():.1f} years")


def build_semantic_graph():
    """Main script to build semantic graph."""
    from src.index.dense_index import DenseIndex
    import json

    # Load components
    print("Loading embeddings...")
    embeddings = np.load("embeddings/dense_all.npy").astype(np.float32)

    print("Loading metadata...")
    with open("embeddings/metadata_all.json") as f:
        metadata = json.load(f)

    print("Loading index...")
    index = DenseIndex()
    index.load("indexes", use_gpu=True)

    # Build graph
    builder = SemanticGraphBuilder(
        dense_index=index,
        metadata=metadata,
        similarity_threshold=0.75,
        cross_field_threshold=0.70,
        max_edges_per_node=50
    )

    # Collect edges
    edges = list(builder.build_edges(embeddings))

    # Save
    Path("graph").mkdir(exist_ok=True)
    builder.save_edges(edges, "graph/edges.parquet")

    # Save nodes
    import pandas as pd
    nodes_df = pd.DataFrame(metadata)
    nodes_df.to_parquet("graph/nodes.parquet", index=False)
    print(f"Saved {len(nodes_df):,} nodes to graph/nodes.parquet")


if __name__ == "__main__":
    build_semantic_graph()
```

---

## Phase 5-10: See Existing Sections Above

The remaining phases (5-10) are already detailed in the plan above with code snippets.

---

# HOW FULL HYBRID MEETS THE 4 HACKATHON OBJECTIVES

## Objective 1: Quantify ML Impact

### What the Hackathon Wants:
- Attribution scoring: What % of breakthrough comes from ML vs. domain insight?
- Acceleration metrics: Did ML speed discovery by months/years?
- Efficiency measures: Cost per discovery

### How Full Hybrid Delivers:

#### 1.1 Attribution Scoring (Graph-Based)
```
Traditional Approach (Keywords):
  - Count keyword matches → "20% papers mention ML"
  - Problem: Can't measure ACTUAL influence

Full Hybrid Approach (Graph + Embeddings):
  - PageRank on semantic graph → Papers ranked by influence
  - ML Attribution = Σ(PageRank of ML papers with edges to field) / Σ(PageRank of all papers in field)
  - Captures ACTUAL methodological influence, not just mentions
```

**Implementation**:
```python
def compute_ml_attribution(graph, field):
    """
    ML Attribution Score = How much do ML papers influence this field?

    Uses: Graph edges (semantic influence) + PageRank (importance weighting)
    """
    # Get all papers in field
    field_papers = get_papers_by_field(field)

    # Get incoming edges from ML papers
    ml_papers = get_ml_papers()
    influence_edges = graph.get_edges(
        source_filter=ml_papers,
        target_filter=field_papers
    )

    # Weight by PageRank
    ml_pagerank = pagerank[ml_papers]
    weighted_influence = sum(
        edge.weight * ml_pagerank[edge.source]
        for edge in influence_edges
    )

    # Normalize by total field influence
    total_influence = sum(pagerank[field_papers])

    return weighted_influence / total_influence
```

**Why This Is Better**:
| Metric | Keyword Approach | Full Hybrid |
|--------|------------------|-------------|
| Measures | Keyword frequency | Semantic influence flow |
| Captures implicit influence | ❌ | ✅ |
| Weights by paper importance | ❌ | ✅ (PageRank) |
| Handles paraphrases | ❌ | ✅ (embeddings) |

#### 1.2 Acceleration Metrics
```python
def compute_acceleration(graph, field, ml_technique):
    """
    Did ML speed up discovery?

    Compare: Time between breakthrough papers BEFORE vs AFTER ML adoption
    """
    # Find influential papers (top PageRank)
    influential = get_top_pagerank_papers(field, top_k=100)

    # Split by ML adoption
    pre_ml = [p for p in influential if p.year < ml_adoption_year(field)]
    post_ml = [p for p in influential if p.year >= ml_adoption_year(field)]

    # Measure time between breakthroughs
    pre_ml_gap = avg_year_gap_between_breakthroughs(pre_ml)
    post_ml_gap = avg_year_gap_between_breakthroughs(post_ml)

    acceleration_factor = pre_ml_gap / post_ml_gap
    return acceleration_factor  # >1 means ML sped things up
```

#### 1.3 Efficiency Measures
```python
def compute_cost_efficiency(graph, field):
    """
    Cost per discovery = Compute mentions / Influential papers produced
    """
    field_papers = get_papers_by_field(field)

    # Compute cost proxy (GPU/HPC mentions in text)
    avg_compute_cost = mean([
        compute_cost_score(paper.text)
        for paper in field_papers if paper.is_ml
    ])

    # Impact = Number of high-influence papers
    high_influence = len([
        p for p in field_papers
        if pagerank[p] > percentile_90(pagerank)
    ])

    cost_efficiency = high_influence / (avg_compute_cost + 0.1)
    return cost_efficiency
```

---

## Objective 2: Visualize Adoption Dynamics

### What the Hackathon Wants:
- Interactive dashboards showing S-curves
- Citation flows from ML methods to domain papers
- Animated temporal evolution (2016-2024)

### How Full Hybrid Delivers:

#### 2.1 S-Curves from Graph Traversal
```
Traditional Approach:
  - Count papers with "transformer" keyword by year
  - Problem: Misses papers using transformers without saying "transformer"

Full Hybrid Approach:
  - Start from seed papers (original transformer papers)
  - BFS through semantic graph, tracking year
  - Each hop = one "generation" of adoption
  - Result: True adoption cascade, not keyword counts
```

**Implementation**:
```python
def compute_adoption_cascade(graph, technique, start_year):
    """
    Track adoption via graph traversal, not keyword counting.

    Returns: {year: {field: count}} showing true spread
    """
    # Find seed papers
    seeds = find_technique_seed_papers(technique, start_year)

    # BFS with year tracking
    visited = set(seeds)
    frontier = seeds
    adoption_by_year_field = defaultdict(lambda: defaultdict(int))

    # Initialize with seeds
    for paper in seeds:
        adoption_by_year_field[paper.year][paper.field] += 1

    while frontier:
        next_frontier = set()

        for paper_id in frontier:
            # Get influenced papers (graph neighbors published later)
            neighbors = graph.get_neighbors(paper_id)

            for neighbor in neighbors:
                if neighbor.id in visited:
                    continue
                if neighbor.year <= graph.get_node(paper_id).year:
                    continue  # Must be published after

                visited.add(neighbor.id)
                next_frontier.add(neighbor.id)
                adoption_by_year_field[neighbor.year][neighbor.field] += 1

        frontier = next_frontier

    return adoption_by_year_field
```

#### 2.2 Citation Flow Visualization
```python
def visualize_adoption_flow(adoption_data, technique):
    """
    Sankey diagram: ML Origin → Intermediate Fields → Target Fields
    """
    import plotly.graph_objects as go

    # Build flow data
    sources, targets, values = [], [], []

    for year, field_counts in sorted(adoption_data.items()):
        for field, count in field_counts.items():
            sources.append(f"{technique} ({year})")
            targets.append(field)
            values.append(count)

    fig = go.Figure(go.Sankey(
        node=dict(label=list(set(sources + targets))),
        link=dict(source=sources, target=targets, value=values)
    ))

    return fig
```

#### 2.3 Animated Temporal Evolution
```python
def create_adoption_animation(adoption_data):
    """
    Animated bar chart race showing field adoption over time.
    """
    import plotly.express as px

    # Convert to DataFrame for animation
    records = []
    cumulative = defaultdict(int)

    for year in sorted(adoption_data.keys()):
        for field, count in adoption_data[year].items():
            cumulative[field] += count
            records.append({
                'year': year,
                'field': field,
                'cumulative_papers': cumulative[field]
            })

    df = pd.DataFrame(records)

    fig = px.bar(
        df,
        x='cumulative_papers',
        y='field',
        animation_frame='year',
        orientation='h',
        title='ML Adoption by Field Over Time'
    )

    return fig
```

---

## Objective 3: Analyze Quality Trade-offs

### What the Hackathon Wants:
- Cross-reference ML adoption with reproducibility rates
- Retraction pattern detection
- Code/data availability correlations

### How Full Hybrid Delivers:

#### 3.1 Reproducibility Analysis (Embedding-Enhanced)
```
Traditional Approach:
  - Check if "github" or "code available" in text
  - Problem: Misses variations, false positives

Full Hybrid Approach:
  - Embed reproducibility concept: "code available github reproducible"
  - Find semantically similar papers
  - Higher recall, captures paraphrases like "implementation released"
```

**Implementation**:
```python
def compute_reproducibility_correlation(papers_df, embeddings, index):
    """
    Correlate ML adoption with reproducibility using semantic search.
    """
    # Embed reproducibility concepts
    repro_queries = [
        "code available github implementation released",
        "data available dataset published",
        "reproducible results replication package"
    ]

    query_embeddings = embed_queries(repro_queries)

    # For each paper, compute "reproducibility similarity"
    repro_scores = []
    for i, paper in enumerate(papers_df.itertuples()):
        paper_emb = embeddings[i:i+1]

        # Similarity to reproducibility concepts
        sims = cosine_similarity(paper_emb, query_embeddings)
        repro_score = sims.max()
        repro_scores.append(repro_score)

    papers_df['repro_score_semantic'] = repro_scores

    # Correlate with ML adoption
    ml_papers = papers_df[papers_df['is_ml_paper']]
    non_ml_papers = papers_df[~papers_df['is_ml_paper']]

    return {
        'ml_avg_repro': ml_papers['repro_score_semantic'].mean(),
        'non_ml_avg_repro': non_ml_papers['repro_score_semantic'].mean(),
        'correlation': papers_df['ml_score'].corr(papers_df['repro_score_semantic'])
    }
```

#### 3.2 Retraction Pattern Detection
```python
def detect_retraction_patterns(graph, papers_df):
    """
    Use graph to find if retracted papers form clusters.

    Hypothesis: Problematic research may cluster (same methods, same lab)
    """
    # Find papers with retraction signals
    retracted = papers_df[papers_df['text'].str.contains(
        'retracted|withdrawn|correction', case=False, na=False
    )]

    # Check if they cluster in graph
    retracted_ids = set(retracted['id'])

    # Get communities
    communities = detect_communities(graph)

    # Find communities with high retraction rate
    risky_communities = []
    for comm_id, members in communities.items():
        retracted_in_comm = len(set(members) & retracted_ids)
        rate = retracted_in_comm / len(members)

        if rate > 0.05:  # >5% retraction rate
            risky_communities.append({
                'community': comm_id,
                'size': len(members),
                'retraction_rate': rate,
                'sample_papers': members[:5]
            })

    return risky_communities
```

#### 3.3 Hype Risk Score
```python
def compute_hype_risk(field, adoption_data, quality_data):
    """
    Hype Risk = High adoption velocity + Low reproducibility

    Uses: Graph (adoption spread) + Semantic (quality detection)
    """
    # Adoption velocity from graph traversal
    adoption_velocity = compute_adoption_velocity(adoption_data, field)

    # Reproducibility from semantic analysis
    repro_rate = quality_data[field]['repro_score_semantic']
    code_rate = quality_data[field]['code_availability_rate']

    # Hype risk formula
    hype_risk = (adoption_velocity / (repro_rate + code_rate + 0.1))

    return min(hype_risk, 1.0)
```

---

## Objective 4: Trace Discovery Impact

### What the Hackathon Wants:
- High-profile case studies (AlphaFold, COVID drug discovery)
- Journey tracking: ML method → domain application → real-world impact

### How Full Hybrid Delivers:

This is where Full Hybrid SHINES. Pure keyword search CANNOT do this.

#### 4.1 Influence Path Tracing
```
Query: "How did AlphaFold impact drug discovery?"

Keyword Approach: ❌
  - Find papers with "AlphaFold" AND "drug discovery"
  - Returns: Direct mentions only
  - Misses: Intermediate papers that bridged the gap

Full Hybrid Approach: ✅
  - Vector search → Find AlphaFold papers (seeds)
  - Vector search → Find drug discovery papers (targets)
  - Graph shortest path → Find papers connecting them
  - Result: Full influence chain with intermediate steps
```

**Implementation**:
```python
def trace_discovery_impact(
    graph,
    index,
    source_concept: str,  # "AlphaFold protein structure prediction"
    target_concept: str,  # "drug discovery pharmaceutical screening"
    max_hops: int = 5
) -> List[Dict]:
    """
    Trace influence path from ML method to real-world application.

    This is THE key differentiator of Full Hybrid.
    """
    # Step 1: Find source papers via semantic search
    source_emb = embed(source_concept)
    source_scores, source_ids = index.search(source_emb, k=20)
    source_papers = [id for id, score in zip(source_ids[0], source_scores[0]) if score > 0.7]

    # Step 2: Find target papers via semantic search
    target_emb = embed(target_concept)
    target_scores, target_ids = index.search(target_emb, k=50)
    target_papers = [id for id, score in zip(target_ids[0], target_scores[0]) if score > 0.7]

    # Step 3: Find shortest paths in graph
    paths = []
    for src in source_papers[:5]:
        for tgt in target_papers[:20]:
            path = graph.shortest_path(src, tgt)

            if path and len(path) <= max_hops:
                # Get full paper info for each node in path
                path_papers = [get_paper_details(p) for p in path]

                paths.append({
                    'source': source_concept,
                    'target': target_concept,
                    'path_length': len(path),
                    'path_ids': path,
                    'path_papers': path_papers,
                    'fields_traversed': [p['field'] for p in path_papers],
                    'year_span': path_papers[-1]['year'] - path_papers[0]['year']
                })

    # Sort by path length (shorter = more direct influence)
    return sorted(paths, key=lambda x: x['path_length'])
```

#### 4.2 Case Study: AlphaFold → Drug Discovery
```python
def alphafold_case_study(graph, index):
    """
    Trace AlphaFold's impact on drug discovery.
    """
    paths = trace_discovery_impact(
        graph, index,
        source_concept="AlphaFold protein structure deep learning",
        target_concept="drug discovery pharmaceutical compound screening"
    )

    # Analyze paths
    result = {
        'total_paths_found': len(paths),
        'shortest_path_length': paths[0]['path_length'] if paths else None,
        'avg_year_span': np.mean([p['year_span'] for p in paths]),
        'intermediate_fields': Counter([
            field for p in paths
            for field in p['fields_traversed'][1:-1]  # Exclude source/target
        ]),
        'example_paths': paths[:5]
    }

    # Visualize
    print(f"\n=== AlphaFold → Drug Discovery ===")
    print(f"Found {result['total_paths_found']} influence paths")
    print(f"Shortest path: {result['shortest_path_length']} hops")
    print(f"Intermediate fields: {result['intermediate_fields']}")

    print(f"\nExample path:")
    for i, paper in enumerate(result['example_paths'][0]['path_papers']):
        arrow = "→ " if i > 0 else ""
        print(f"  {arrow}{paper['title'][:50]}... ({paper['year']}, {paper['field']})")

    return result
```

#### 4.3 Journey Visualization
```python
def visualize_discovery_journey(paths):
    """
    Network visualization of influence paths.
    """
    import networkx as nx
    import plotly.graph_objects as go

    G = nx.DiGraph()

    # Add all papers from all paths
    for path in paths[:10]:
        for i, paper in enumerate(path['path_papers']):
            G.add_node(
                paper['id'],
                title=paper['title'][:30],
                year=paper['year'],
                field=paper['field']
            )

            if i > 0:
                prev_paper = path['path_papers'][i-1]
                G.add_edge(prev_paper['id'], paper['id'])

    # Layout
    pos = nx.spring_layout(G)

    # Create Plotly figure
    edge_trace = go.Scatter(...)
    node_trace = go.Scatter(...)

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(title="Discovery Impact Journey")

    return fig
```

---

## Summary: Full Hybrid Advantage by Objective

| Objective | Keyword Only | Vector Only | Graph Only | **Full Hybrid** |
|-----------|--------------|-------------|------------|-----------------|
| **1. ML Impact** | Keyword counts | Similarity clusters | Citation PageRank | **Semantic influence + PageRank weighting** |
| **2. Adoption Dynamics** | Keyword frequency over time | Can't track spread | Citation cascades | **Semantic cascade through graph BFS** |
| **3. Quality Trade-offs** | "github" matching | Semantic similarity to "reproducible" | Community-based clustering | **Semantic quality + graph clustering** |
| **4. Discovery Paths** | ❌ Impossible | Can find similar papers, no paths | Can trace paths, can't find start/end | **Vector finds nodes, graph traces paths** |

**Bottom Line**: Objective 4 (Discovery Paths) is ONLY possible with Full Hybrid. This is the "wow factor" for the hackathon.
