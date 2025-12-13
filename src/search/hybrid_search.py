"""
Hybrid Search Engine for Full Hybrid Semantic-Graph Search

Combines:
1. Dense retrieval (semantic similarity via FAISS)
2. Sparse retrieval (keyword matching via BM25)
3. Graph retrieval (related papers via semantic graph)

Uses Reciprocal Rank Fusion (RRF) to combine results
and optional cross-encoder reranking for precision.
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass, field
import json
import time

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


@dataclass
class SearchResult:
    """A single search result with metadata."""
    paper_id: str
    score: float
    title: str = ""
    year: int = 0
    source_field: str = ""
    source: str = ""  # 'dense', 'sparse', 'graph', 'hybrid'
    rank_dense: int = -1
    rank_sparse: int = -1
    rank_graph: int = -1
    graph_hops: int = 0  # Distance from seed papers in graph
    snippet: str = ""  # Relevant text snippet
    metadata: Dict = field(default_factory=dict)


class HybridSearchEngine:
    """
    Hybrid search engine combining dense, sparse, and graph retrieval.

    Uses Reciprocal Rank Fusion (RRF) to combine rankings from:
    - Dense retrieval: Semantic similarity via BGE-M3 embeddings
    - Sparse retrieval: Keyword matching via lexical weights
    - Graph retrieval: Related papers via semantic graph traversal
    """

    def __init__(
        self,
        dense_weight: float = 0.5,
        sparse_weight: float = 0.3,
        graph_weight: float = 0.2,
        rrf_k: int = 60,
        use_reranking: bool = True,
        rerank_top_k: int = 50,
    ):
        """
        Initialize hybrid search engine.

        Args:
            dense_weight: Weight for dense retrieval in RRF
            sparse_weight: Weight for sparse retrieval in RRF
            graph_weight: Weight for graph retrieval in RRF
            rrf_k: RRF constant (higher = more emphasis on top ranks)
            use_reranking: Whether to use cross-encoder reranking
            rerank_top_k: Number of results to rerank
        """
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.graph_weight = graph_weight
        self.rrf_k = rrf_k
        self.use_reranking = use_reranking
        self.rerank_top_k = rerank_top_k

        # Components (set via set_* methods)
        self.vector_index = None
        self.sparse_index = None
        self.semantic_graph = None
        self.embedder = None
        self.reranker = None
        self.paper_metadata: Dict[str, Dict] = {}

    def set_vector_index(self, index):
        """Set the dense vector index."""
        self.vector_index = index

    def set_sparse_index(self, index):
        """Set the sparse (lexical) index."""
        self.sparse_index = index

    def set_semantic_graph(self, graph):
        """Set the semantic similarity graph."""
        self.semantic_graph = graph

    def set_embedder(self, embedder):
        """Set the embedding model for query encoding."""
        self.embedder = embedder

    def set_paper_metadata(self, metadata: Dict[str, Dict]):
        """Set paper metadata for result enrichment."""
        self.paper_metadata = metadata

    def set_reranker(self, reranker):
        """Set cross-encoder reranker."""
        self.reranker = reranker

    def _encode_query(self, query: str) -> Tuple[np.ndarray, Dict]:
        """Encode query to dense and sparse representations."""
        if self.embedder is None:
            raise ValueError("Embedder not set. Call set_embedder() first.")

        result = self.embedder.embed_texts([query])
        dense = result['dense'][0]
        sparse = result.get('sparse', [{}])[0] if 'sparse' in result else {}

        return dense, sparse

    def _dense_search(
        self,
        query_dense: np.ndarray,
        k: int,
        filter_ids: Optional[List[str]] = None
    ) -> List[Tuple[str, float, int]]:
        """Perform dense retrieval."""
        if self.vector_index is None:
            return []

        scores, result_ids = self.vector_index.search(
            query_dense.reshape(1, -1),
            k=k,
            filter_ids=filter_ids
        )

        results = []
        for rank, (pid, score) in enumerate(zip(result_ids[0], scores[0])):
            results.append((pid, float(score), rank + 1))

        return results

    def _sparse_search(
        self,
        query_sparse: Dict[str, float],
        k: int,
        filter_ids: Optional[List[str]] = None
    ) -> List[Tuple[str, float, int]]:
        """Perform sparse (lexical) retrieval."""
        if self.sparse_index is None or not query_sparse:
            return []

        scores, result_ids = self.sparse_index.search(
            query_sparse,
            k=k,
            filter_ids=filter_ids
        )

        results = []
        for rank, (pid, score) in enumerate(zip(result_ids, scores)):
            results.append((pid, float(score), rank + 1))

        return results

    def _graph_search(
        self,
        seed_ids: List[str],
        k: int,
        max_hops: int = 2
    ) -> List[Tuple[str, float, int, int]]:
        """Perform graph-based retrieval from seed papers."""
        if self.semantic_graph is None or not seed_ids:
            return []

        # Get subgraph around seed papers
        visited = set(seed_ids)
        results = []
        current_frontier = set(seed_ids)
        hop_scores = {pid: 1.0 for pid in seed_ids}  # Seed papers get max score

        for hop in range(1, max_hops + 1):
            next_frontier = set()
            decay = 0.7 ** hop  # Score decay with distance

            for paper_id in current_frontier:
                neighbors = self.semantic_graph.get_neighbors(paper_id, k=10)
                for neighbor_id, edge_weight in neighbors:
                    if neighbor_id not in visited:
                        visited.add(neighbor_id)
                        next_frontier.add(neighbor_id)
                        # Score combines edge weight and distance decay
                        hop_scores[neighbor_id] = edge_weight * decay

            current_frontier = next_frontier

        # Sort by score and return
        sorted_results = sorted(
            [(pid, score) for pid, score in hop_scores.items() if pid not in seed_ids],
            key=lambda x: -x[1]
        )[:k]

        return [
            (pid, score, rank + 1, self._get_hop_distance(seed_ids, pid))
            for rank, (pid, score) in enumerate(sorted_results)
        ]

    def _get_hop_distance(self, seed_ids: List[str], target_id: str) -> int:
        """Get minimum hop distance from any seed to target."""
        if self.semantic_graph is None:
            return 0

        min_hops = float('inf')
        for seed_id in seed_ids:
            path = self.semantic_graph.find_path(seed_id, target_id, max_length=5)
            if path:
                min_hops = min(min_hops, len(path) - 1)

        return min_hops if min_hops != float('inf') else 0

    def _rrf_fusion(
        self,
        dense_results: List[Tuple[str, float, int]],
        sparse_results: List[Tuple[str, float, int]],
        graph_results: List[Tuple[str, float, int, int]],
        k: int
    ) -> List[SearchResult]:
        """
        Combine results using Reciprocal Rank Fusion (RRF).

        RRF score = sum over all rankings of: weight / (rrf_k + rank)
        """
        scores = {}  # paper_id -> RRF score
        rank_info = {}  # paper_id -> rank info

        # Process dense results
        for pid, score, rank in dense_results:
            rrf_score = self.dense_weight / (self.rrf_k + rank)
            scores[pid] = scores.get(pid, 0) + rrf_score
            if pid not in rank_info:
                rank_info[pid] = {'dense': -1, 'sparse': -1, 'graph': -1, 'hops': 0}
            rank_info[pid]['dense'] = rank

        # Process sparse results
        for pid, score, rank in sparse_results:
            rrf_score = self.sparse_weight / (self.rrf_k + rank)
            scores[pid] = scores.get(pid, 0) + rrf_score
            if pid not in rank_info:
                rank_info[pid] = {'dense': -1, 'sparse': -1, 'graph': -1, 'hops': 0}
            rank_info[pid]['sparse'] = rank

        # Process graph results
        for item in graph_results:
            pid, score, rank, hops = item
            rrf_score = self.graph_weight / (self.rrf_k + rank)
            scores[pid] = scores.get(pid, 0) + rrf_score
            if pid not in rank_info:
                rank_info[pid] = {'dense': -1, 'sparse': -1, 'graph': -1, 'hops': 0}
            rank_info[pid]['graph'] = rank
            rank_info[pid]['hops'] = hops

        # Sort by RRF score
        sorted_results = sorted(scores.items(), key=lambda x: -x[1])[:k]

        # Build SearchResult objects
        results = []
        for pid, score in sorted_results:
            info = rank_info.get(pid, {})
            metadata = self.paper_metadata.get(pid, {})

            result = SearchResult(
                paper_id=pid,
                score=score,
                title=metadata.get('title', ''),
                year=metadata.get('year', 0),
                source_field=metadata.get('source_field', ''),
                source='hybrid',
                rank_dense=info.get('dense', -1),
                rank_sparse=info.get('sparse', -1),
                rank_graph=info.get('graph', -1),
                graph_hops=info.get('hops', 0),
                metadata=metadata
            )
            results.append(result)

        return results

    def _rerank(
        self,
        query: str,
        results: List[SearchResult]
    ) -> List[SearchResult]:
        """Rerank results using cross-encoder."""
        if self.reranker is None or not results:
            return results

        # Get texts for reranking
        pairs = []
        for result in results[:self.rerank_top_k]:
            text = result.title
            if result.metadata.get('embedding_text'):
                text = result.metadata['embedding_text'][:1000]
            pairs.append((query, text))

        # Score with cross-encoder
        rerank_scores = self.reranker.score(pairs)

        # Update scores and re-sort
        for i, score in enumerate(rerank_scores):
            if i < len(results):
                results[i].score = score

        results = sorted(results, key=lambda x: -x.score)
        return results

    def search(
        self,
        query: str,
        k: int = 10,
        dense_k: int = 100,
        sparse_k: int = 100,
        graph_k: int = 50,
        seed_paper_ids: Optional[List[str]] = None,
        filter_fields: Optional[List[str]] = None,
        filter_years: Optional[Tuple[int, int]] = None,
        use_graph: bool = True,
    ) -> List[SearchResult]:
        """
        Perform hybrid search.

        Args:
            query: Search query text
            k: Number of final results to return
            dense_k: Number of dense retrieval results
            sparse_k: Number of sparse retrieval results
            graph_k: Number of graph retrieval results
            seed_paper_ids: Seed papers for graph search (if None, uses dense results)
            filter_fields: Only return papers from these fields
            filter_years: Only return papers from (min_year, max_year)
            use_graph: Whether to include graph-based retrieval

        Returns:
            List of SearchResult objects
        """
        start_time = time.time()

        # Encode query
        query_dense, query_sparse = self._encode_query(query)

        # Apply filters
        filter_ids = None
        if filter_fields or filter_years:
            filter_ids = self._apply_filters(filter_fields, filter_years)

        # Dense retrieval
        dense_results = self._dense_search(query_dense, dense_k, filter_ids)

        # Sparse retrieval
        sparse_results = self._sparse_search(query_sparse, sparse_k, filter_ids)

        # Graph retrieval
        graph_results = []
        if use_graph and self.semantic_graph:
            # Use top dense results as seeds if not provided
            if not seed_paper_ids:
                seed_paper_ids = [r[0] for r in dense_results[:5]]
            graph_results = self._graph_search(seed_paper_ids, graph_k)

        # RRF Fusion
        results = self._rrf_fusion(dense_results, sparse_results, graph_results, k=self.rerank_top_k)

        # Reranking
        if self.use_reranking and self.reranker:
            results = self._rerank(query, results)

        # Final top-k
        results = results[:k]

        # Add timing info
        search_time = time.time() - start_time
        for result in results:
            result.metadata['search_time_ms'] = int(search_time * 1000)

        return results

    def _apply_filters(
        self,
        filter_fields: Optional[List[str]],
        filter_years: Optional[Tuple[int, int]]
    ) -> List[str]:
        """Get paper IDs matching filters."""
        valid_ids = []

        for pid, metadata in self.paper_metadata.items():
            # Field filter
            if filter_fields:
                paper_field = metadata.get('source_field', '')
                if paper_field not in filter_fields:
                    continue

            # Year filter
            if filter_years:
                year = metadata.get('year', 0)
                if not (filter_years[0] <= year <= filter_years[1]):
                    continue

            valid_ids.append(pid)

        return valid_ids if valid_ids else None

    def search_similar(
        self,
        paper_id: str,
        k: int = 10
    ) -> List[SearchResult]:
        """Find papers similar to a given paper."""
        if self.vector_index is None:
            return []

        # Get paper's embedding
        embedding = self.vector_index.dense_index.get(paper_id)
        if embedding is None:
            return []

        # Search
        scores, result_ids = self.vector_index.search(
            embedding.reshape(1, -1),
            k=k + 1  # +1 to exclude self
        )

        results = []
        for score, pid in zip(scores[0], result_ids[0]):
            if pid == paper_id:
                continue
            metadata = self.paper_metadata.get(pid, {})
            results.append(SearchResult(
                paper_id=pid,
                score=float(score),
                title=metadata.get('title', ''),
                year=metadata.get('year', 0),
                source_field=metadata.get('source_field', ''),
                source='dense',
                metadata=metadata
            ))

        return results[:k]

    def explain_result(self, query: str, result: SearchResult) -> Dict:
        """Explain why a result was returned."""
        explanation = {
            'paper_id': result.paper_id,
            'final_score': result.score,
            'contributions': {}
        }

        # Dense contribution
        if result.rank_dense > 0:
            dense_rrf = self.dense_weight / (self.rrf_k + result.rank_dense)
            explanation['contributions']['dense'] = {
                'rank': result.rank_dense,
                'rrf_score': dense_rrf,
                'description': f'Ranked #{result.rank_dense} by semantic similarity'
            }

        # Sparse contribution
        if result.rank_sparse > 0:
            sparse_rrf = self.sparse_weight / (self.rrf_k + result.rank_sparse)
            explanation['contributions']['sparse'] = {
                'rank': result.rank_sparse,
                'rrf_score': sparse_rrf,
                'description': f'Ranked #{result.rank_sparse} by keyword matching'
            }

        # Graph contribution
        if result.rank_graph > 0:
            graph_rrf = self.graph_weight / (self.rrf_k + result.rank_graph)
            explanation['contributions']['graph'] = {
                'rank': result.rank_graph,
                'rrf_score': graph_rrf,
                'hops': result.graph_hops,
                'description': f'Ranked #{result.rank_graph} via graph ({result.graph_hops} hops)'
            }

        return explanation

    def save_config(self, path: str):
        """Save search engine configuration."""
        config = {
            'dense_weight': self.dense_weight,
            'sparse_weight': self.sparse_weight,
            'graph_weight': self.graph_weight,
            'rrf_k': self.rrf_k,
            'use_reranking': self.use_reranking,
            'rerank_top_k': self.rerank_top_k,
        }

        with open(path, 'w') as f:
            json.dump(config, f, indent=2)

    def load_config(self, path: str):
        """Load search engine configuration."""
        with open(path, 'r') as f:
            config = json.load(f)

        self.dense_weight = config.get('dense_weight', self.dense_weight)
        self.sparse_weight = config.get('sparse_weight', self.sparse_weight)
        self.graph_weight = config.get('graph_weight', self.graph_weight)
        self.rrf_k = config.get('rrf_k', self.rrf_k)
        self.use_reranking = config.get('use_reranking', self.use_reranking)
        self.rerank_top_k = config.get('rerank_top_k', self.rerank_top_k)


class CrossEncoderReranker:
    """
    Cross-encoder reranker for improving search precision.

    Uses a cross-encoder model to score query-document pairs
    for more accurate relevance ranking.
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        """
        Initialize reranker.

        Args:
            model_name: HuggingFace model name for cross-encoder
        """
        self.model_name = model_name
        self.model = None
        self._initialized = False

    def initialize(self):
        """Lazy initialization of the model."""
        if self._initialized:
            return

        try:
            from FlagEmbedding import FlagReranker
            self.model = FlagReranker(self.model_name, use_fp16=True)
            self._initialized = True
            print(f"[Reranker] Loaded {self.model_name}")
        except ImportError:
            print("[Reranker] FlagEmbedding not available. Install with: pip install FlagEmbedding")

    def score(self, pairs: List[Tuple[str, str]]) -> List[float]:
        """
        Score query-document pairs.

        Args:
            pairs: List of (query, document) tuples

        Returns:
            List of relevance scores
        """
        self.initialize()

        if self.model is None:
            return [0.0] * len(pairs)

        scores = self.model.compute_score(pairs)

        # Ensure list output
        if isinstance(scores, (int, float)):
            scores = [scores]

        return list(scores)


if __name__ == "__main__":
    print("\n=== Hybrid Search Engine Test ===\n")

    # Create dummy search engine
    engine = HybridSearchEngine(
        dense_weight=0.5,
        sparse_weight=0.3,
        graph_weight=0.2
    )

    # Test RRF fusion
    dense_results = [
        ("paper_1", 0.95, 1),
        ("paper_2", 0.90, 2),
        ("paper_3", 0.85, 3),
    ]
    sparse_results = [
        ("paper_2", 0.80, 1),
        ("paper_4", 0.75, 2),
        ("paper_1", 0.70, 3),
    ]
    graph_results = [
        ("paper_3", 0.60, 1, 1),
        ("paper_5", 0.55, 2, 2),
    ]

    # Set dummy metadata
    engine.paper_metadata = {
        f"paper_{i}": {'title': f'Paper {i}', 'year': 2020 + i, 'source_field': 'CS'}
        for i in range(1, 6)
    }

    results = engine._rrf_fusion(dense_results, sparse_results, graph_results, k=5)

    print("RRF Fusion Results:")
    for result in results:
        print(f"  {result.paper_id}: score={result.score:.4f}, "
              f"dense_rank={result.rank_dense}, sparse_rank={result.rank_sparse}, "
              f"graph_rank={result.rank_graph}")

    print("\nExplanation for top result:")
    explanation = engine.explain_result("test query", results[0])
    print(json.dumps(explanation, indent=2))
