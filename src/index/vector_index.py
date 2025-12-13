"""
Vector Index for Full Hybrid Semantic-Graph Search

Supports:
- FAISS IVF-PQ for efficient approximate nearest neighbor search
- GPU acceleration with FAISS-GPU or cuVS
- Incremental index building
- Hybrid dense+sparse retrieval
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
import json
import pickle
import time

# Try to import FAISS
try:
    import faiss
    FAISS_AVAILABLE = True
    # Check for GPU support
    FAISS_GPU = faiss.get_num_gpus() > 0
    print(f"[VectorIndex] FAISS available, GPU: {FAISS_GPU} ({faiss.get_num_gpus()} GPUs)")
except ImportError:
    FAISS_AVAILABLE = False
    FAISS_GPU = False
    print("[VectorIndex] FAISS not installed. Install with: pip install faiss-cpu or faiss-gpu")

# Try cuVS for DGX Spark (RAPIDS)
try:
    from cuvs.neighbors import cagra, ivf_flat
    import cupy as cp
    CUVS_AVAILABLE = True
    print("[VectorIndex] cuVS (RAPIDS) available")
except ImportError:
    CUVS_AVAILABLE = False


class VectorIndex:
    """
    FAISS-based vector index for dense retrieval.

    Uses IVF-PQ (Inverted File with Product Quantization) for
    memory-efficient approximate nearest neighbor search on millions of vectors.
    """

    def __init__(
        self,
        dim: int = 1024,
        index_type: str = "IVF4096,PQ64",  # Good balance for 3M+ vectors
        metric: str = "cosine",
        use_gpu: bool = True,
        nprobe: int = 64,  # Number of clusters to search
    ):
        """
        Initialize vector index.

        Args:
            dim: Embedding dimension
            index_type: FAISS index type string
            metric: 'cosine' or 'l2'
            use_gpu: Use GPU acceleration if available
            nprobe: Number of clusters to search (higher = more accurate, slower)
        """
        self.dim = dim
        self.index_type = index_type
        self.metric = metric
        self.use_gpu = use_gpu and (FAISS_GPU or CUVS_AVAILABLE)
        self.nprobe = nprobe

        self.index = None
        self.is_trained = False
        self.paper_ids: List[str] = []
        self.paper_id_to_idx: Dict[str, int] = {}

    def build(
        self,
        embeddings: np.ndarray,
        paper_ids: List[str],
        train_size: int = 100000
    ):
        """
        Build the vector index.

        Args:
            embeddings: Dense embeddings (N, D)
            paper_ids: Corresponding paper IDs
            train_size: Number of samples for training IVF
        """
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS not available. Install with: pip install faiss-cpu")

        n, d = embeddings.shape
        assert d == self.dim, f"Dimension mismatch: expected {self.dim}, got {d}"

        print(f"[VectorIndex] Building index for {n:,} vectors...")
        start = time.time()

        # Normalize for cosine similarity
        if self.metric == "cosine":
            embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9)

        # Create index
        if "IVF" in self.index_type:
            # For IVF indexes, we need to train first
            quantizer = faiss.IndexFlatIP(d) if self.metric == "cosine" else faiss.IndexFlatL2(d)

            # Parse index type
            if "PQ" in self.index_type:
                # IVF with Product Quantization
                parts = self.index_type.split(",")
                nlist = int(parts[0].replace("IVF", ""))
                m = int(parts[1].replace("PQ", ""))
                self.index = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8)
            else:
                # IVF Flat
                nlist = int(self.index_type.replace("IVF", "").replace("Flat", ""))
                self.index = faiss.IndexIVFFlat(quantizer, d, nlist)

            # Train on subset
            train_data = embeddings[:min(train_size, n)]
            print(f"[VectorIndex] Training on {len(train_data):,} samples...")
            self.index.train(train_data.astype(np.float32))
            self.is_trained = True

        else:
            # Flat index (exact search)
            if self.metric == "cosine":
                self.index = faiss.IndexFlatIP(d)
            else:
                self.index = faiss.IndexFlatL2(d)

        # Add vectors
        print(f"[VectorIndex] Adding {n:,} vectors...")
        self.index.add(embeddings.astype(np.float32))

        # Set search parameters
        if hasattr(self.index, 'nprobe'):
            self.index.nprobe = self.nprobe

        # Store paper IDs
        self.paper_ids = list(paper_ids)
        self.paper_id_to_idx = {pid: idx for idx, pid in enumerate(self.paper_ids)}

        # Move to GPU if available
        if self.use_gpu and FAISS_GPU:
            print("[VectorIndex] Moving index to GPU...")
            self.index = faiss.index_cpu_to_all_gpus(self.index)

        print(f"[VectorIndex] Index built in {time.time() - start:.1f}s")

    def search(
        self,
        query_embeddings: np.ndarray,
        k: int = 10,
        filter_ids: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, List[List[str]]]:
        """
        Search for nearest neighbors.

        Args:
            query_embeddings: Query vectors (Q, D)
            k: Number of neighbors to return
            filter_ids: Only return these paper IDs (post-filtering)

        Returns:
            Tuple of (scores, paper_ids) where:
            - scores: (Q, k) similarity scores
            - paper_ids: List of k paper ID lists per query
        """
        if self.index is None:
            raise ValueError("Index not built. Call build() first.")

        # Ensure 2D
        if query_embeddings.ndim == 1:
            query_embeddings = query_embeddings.reshape(1, -1)

        # Normalize for cosine
        if self.metric == "cosine":
            query_embeddings = query_embeddings / (np.linalg.norm(query_embeddings, axis=1, keepdims=True) + 1e-9)

        # Search
        scores, indices = self.index.search(query_embeddings.astype(np.float32), k)

        # Convert indices to paper IDs
        result_ids = []
        for idx_row in indices:
            ids = []
            for idx in idx_row:
                if 0 <= idx < len(self.paper_ids):
                    pid = self.paper_ids[idx]
                    if filter_ids is None or pid in filter_ids:
                        ids.append(pid)
            result_ids.append(ids)

        return scores, result_ids

    def save(self, path: str):
        """Save index to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        if self.use_gpu and FAISS_GPU:
            # Move to CPU for saving
            cpu_index = faiss.index_gpu_to_cpu(self.index)
            faiss.write_index(cpu_index, str(path / "index.faiss"))
        else:
            faiss.write_index(self.index, str(path / "index.faiss"))

        # Save metadata
        metadata = {
            'dim': self.dim,
            'index_type': self.index_type,
            'metric': self.metric,
            'nprobe': self.nprobe,
            'is_trained': self.is_trained,
            'count': len(self.paper_ids),
        }
        with open(path / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        # Save paper IDs
        with open(path / "paper_ids.json", 'w') as f:
            json.dump(self.paper_ids, f)

        print(f"[VectorIndex] Saved to {path}")

    def load(self, path: str):
        """Load index from disk."""
        path = Path(path)

        if not (path / "index.faiss").exists():
            raise FileNotFoundError(f"Index not found at {path}")

        # Load metadata
        with open(path / "metadata.json", 'r') as f:
            metadata = json.load(f)

        self.dim = metadata['dim']
        self.index_type = metadata['index_type']
        self.metric = metadata['metric']
        self.nprobe = metadata['nprobe']
        self.is_trained = metadata['is_trained']

        # Load FAISS index
        self.index = faiss.read_index(str(path / "index.faiss"))

        if hasattr(self.index, 'nprobe'):
            self.index.nprobe = self.nprobe

        # Move to GPU if available
        if self.use_gpu and FAISS_GPU:
            self.index = faiss.index_cpu_to_all_gpus(self.index)

        # Load paper IDs
        with open(path / "paper_ids.json", 'r') as f:
            self.paper_ids = json.load(f)
        self.paper_id_to_idx = {pid: idx for idx, pid in enumerate(self.paper_ids)}

        print(f"[VectorIndex] Loaded {len(self.paper_ids):,} vectors from {path}")

    def __len__(self) -> int:
        return len(self.paper_ids)


class SparseIndex:
    """
    Sparse (lexical) index for BM25-like retrieval.

    Uses inverted index structure for efficient keyword matching.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Initialize sparse index with BM25 parameters.

        Args:
            k1: Term frequency saturation parameter
            b: Length normalization parameter
        """
        self.k1 = k1
        self.b = b

        # Inverted index: token -> [(paper_id, weight), ...]
        self.inverted_index: Dict[str, List[Tuple[str, float]]] = {}
        self.paper_ids: List[str] = []
        self.doc_lengths: Dict[str, float] = {}
        self.avg_doc_length: float = 0.0

    def build(
        self,
        paper_ids: List[str],
        sparse_embeddings: List[Dict[str, float]]
    ):
        """
        Build sparse index from BGE-M3 sparse embeddings.

        Args:
            paper_ids: Paper IDs
            sparse_embeddings: List of token->weight dicts from BGE-M3
        """
        print(f"[SparseIndex] Building index for {len(paper_ids):,} documents...")
        start = time.time()

        self.paper_ids = list(paper_ids)
        self.inverted_index = {}

        # Build inverted index
        for pid, sparse in zip(paper_ids, sparse_embeddings):
            doc_length = sum(sparse.values())
            self.doc_lengths[pid] = doc_length

            for token, weight in sparse.items():
                if token not in self.inverted_index:
                    self.inverted_index[token] = []
                self.inverted_index[token].append((pid, weight))

        # Compute average document length
        if self.doc_lengths:
            self.avg_doc_length = sum(self.doc_lengths.values()) / len(self.doc_lengths)

        print(f"[SparseIndex] Built in {time.time() - start:.1f}s")
        print(f"[SparseIndex] Vocabulary size: {len(self.inverted_index):,}")

    def search(
        self,
        query_sparse: Dict[str, float],
        k: int = 10,
        filter_ids: Optional[List[str]] = None
    ) -> Tuple[List[float], List[str]]:
        """
        Search using BM25-like scoring.

        Args:
            query_sparse: Query sparse embedding (token->weight dict)
            k: Number of results to return
            filter_ids: Only return these paper IDs

        Returns:
            Tuple of (scores, paper_ids)
        """
        # Collect candidate documents
        candidates = {}  # paper_id -> score
        N = len(self.paper_ids)

        for token, query_weight in query_sparse.items():
            if token not in self.inverted_index:
                continue

            postings = self.inverted_index[token]
            df = len(postings)
            idf = np.log((N - df + 0.5) / (df + 0.5) + 1)

            for pid, doc_weight in postings:
                if filter_ids and pid not in filter_ids:
                    continue

                doc_len = self.doc_lengths.get(pid, self.avg_doc_length)
                len_norm = 1 - self.b + self.b * (doc_len / self.avg_doc_length)

                # BM25-like score
                score = idf * (doc_weight * (self.k1 + 1)) / (doc_weight + self.k1 * len_norm)
                score *= query_weight  # Weight by query term importance

                if pid not in candidates:
                    candidates[pid] = 0
                candidates[pid] += score

        # Sort by score
        sorted_results = sorted(candidates.items(), key=lambda x: -x[1])[:k]

        if not sorted_results:
            return [], []

        scores, paper_ids = zip(*sorted_results)
        return list(scores), list(paper_ids)

    def save(self, path: str):
        """Save sparse index to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        with open(path / "sparse_index.pkl", 'wb') as f:
            pickle.dump({
                'inverted_index': self.inverted_index,
                'paper_ids': self.paper_ids,
                'doc_lengths': self.doc_lengths,
                'avg_doc_length': self.avg_doc_length,
                'k1': self.k1,
                'b': self.b,
            }, f)

        print(f"[SparseIndex] Saved to {path}")

    def load(self, path: str):
        """Load sparse index from disk."""
        path = Path(path)

        with open(path / "sparse_index.pkl", 'rb') as f:
            data = pickle.load(f)

        self.inverted_index = data['inverted_index']
        self.paper_ids = data['paper_ids']
        self.doc_lengths = data['doc_lengths']
        self.avg_doc_length = data['avg_doc_length']
        self.k1 = data['k1']
        self.b = data['b']

        print(f"[SparseIndex] Loaded {len(self.paper_ids):,} documents from {path}")


class HybridIndex:
    """
    Hybrid dense + sparse index with RRF fusion.

    Combines FAISS dense retrieval with sparse lexical retrieval
    using Reciprocal Rank Fusion for robust search.
    """

    def __init__(
        self,
        dim: int = 1024,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
        rrf_k: int = 60,  # RRF parameter
    ):
        """
        Initialize hybrid index.

        Args:
            dim: Embedding dimension
            dense_weight: Weight for dense retrieval
            sparse_weight: Weight for sparse retrieval
            rrf_k: RRF constant (higher = more emphasis on top results)
        """
        self.dim = dim
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.rrf_k = rrf_k

        self.dense_index = VectorIndex(dim=dim)
        self.sparse_index = SparseIndex()

    def build(
        self,
        embeddings: np.ndarray,
        paper_ids: List[str],
        sparse_embeddings: Optional[List[Dict]] = None
    ):
        """
        Build both dense and sparse indexes.

        Args:
            embeddings: Dense embeddings
            paper_ids: Paper IDs
            sparse_embeddings: Sparse embeddings (optional)
        """
        # Build dense index
        self.dense_index.build(embeddings, paper_ids)

        # Build sparse index if available
        if sparse_embeddings:
            self.sparse_index.build(paper_ids, sparse_embeddings)

    def search(
        self,
        query_dense: np.ndarray,
        query_sparse: Optional[Dict[str, float]] = None,
        k: int = 10,
        dense_k: int = 100,  # Retrieve more for fusion
        sparse_k: int = 100,
        filter_ids: Optional[List[str]] = None
    ) -> Tuple[List[float], List[str]]:
        """
        Hybrid search with RRF fusion.

        Args:
            query_dense: Dense query embedding
            query_sparse: Sparse query embedding (optional)
            k: Final number of results
            dense_k: Dense retrieval count
            sparse_k: Sparse retrieval count
            filter_ids: Only return these paper IDs

        Returns:
            Tuple of (fused_scores, paper_ids)
        """
        # Dense retrieval
        dense_scores, dense_ids = self.dense_index.search(
            query_dense, k=dense_k, filter_ids=filter_ids
        )
        dense_ids = dense_ids[0]  # Single query

        # Sparse retrieval (if available)
        if query_sparse and self.sparse_index.paper_ids:
            sparse_scores, sparse_ids = self.sparse_index.search(
                query_sparse, k=sparse_k, filter_ids=filter_ids
            )
        else:
            sparse_ids = []

        # RRF Fusion
        rrf_scores = {}

        # Add dense contributions
        for rank, pid in enumerate(dense_ids):
            score = self.dense_weight / (self.rrf_k + rank + 1)
            rrf_scores[pid] = rrf_scores.get(pid, 0) + score

        # Add sparse contributions
        for rank, pid in enumerate(sparse_ids):
            score = self.sparse_weight / (self.rrf_k + rank + 1)
            rrf_scores[pid] = rrf_scores.get(pid, 0) + score

        # Sort by fused score
        sorted_results = sorted(rrf_scores.items(), key=lambda x: -x[1])[:k]

        if not sorted_results:
            return [], []

        scores, paper_ids = zip(*sorted_results)
        return list(scores), list(paper_ids)

    def save(self, path: str):
        """Save hybrid index to disk."""
        path = Path(path)
        self.dense_index.save(str(path / "dense"))
        self.sparse_index.save(str(path / "sparse"))

        metadata = {
            'dense_weight': self.dense_weight,
            'sparse_weight': self.sparse_weight,
            'rrf_k': self.rrf_k,
        }
        with open(path / "hybrid_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

    def load(self, path: str):
        """Load hybrid index from disk."""
        path = Path(path)
        self.dense_index.load(str(path / "dense"))

        if (path / "sparse" / "sparse_index.pkl").exists():
            self.sparse_index.load(str(path / "sparse"))

        with open(path / "hybrid_metadata.json", 'r') as f:
            metadata = json.load(f)
        self.dense_weight = metadata['dense_weight']
        self.sparse_weight = metadata['sparse_weight']
        self.rrf_k = metadata['rrf_k']


if __name__ == "__main__":
    # Test vector index
    print("\n=== Vector Index Test ===\n")

    if FAISS_AVAILABLE:
        # Create dummy data
        n = 1000
        dim = 1024
        embeddings = np.random.randn(n, dim).astype(np.float32)
        paper_ids = [f"paper_{i}" for i in range(n)]

        # Build index
        index = VectorIndex(dim=dim, index_type="IVF100,PQ32")
        index.build(embeddings, paper_ids)

        # Search
        query = np.random.randn(1, dim).astype(np.float32)
        scores, results = index.search(query, k=5)

        print(f"Query results (top 5):")
        for score, pid in zip(scores[0], results[0]):
            print(f"  {pid}: {score:.4f}")

        # Test save/load
        index.save("test_index")
        index2 = VectorIndex(dim=dim)
        index2.load("test_index")
        print(f"\nLoaded index size: {len(index2)}")

        # Clean up
        import shutil
        shutil.rmtree("test_index", ignore_errors=True)
    else:
        print("FAISS not available. Install with: pip install faiss-cpu")

    print("\n=== Hybrid Index Test ===\n")

    if FAISS_AVAILABLE:
        # Create dummy data
        embeddings = np.random.randn(100, 1024).astype(np.float32)
        paper_ids = [f"paper_{i}" for i in range(100)]
        sparse_embeddings = [
            {f"term_{j}": np.random.rand() for j in range(10)}
            for _ in range(100)
        ]

        # Build hybrid index
        hybrid = HybridIndex(dim=1024)
        hybrid.build(embeddings, paper_ids, sparse_embeddings)

        # Search
        query_dense = np.random.randn(1024).astype(np.float32)
        query_sparse = {"term_0": 0.5, "term_1": 0.3}
        scores, results = hybrid.search(query_dense, query_sparse, k=5)

        print(f"Hybrid search results:")
        for score, pid in zip(scores, results):
            print(f"  {pid}: {score:.4f}")
