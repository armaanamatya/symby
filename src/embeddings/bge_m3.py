"""
BGE-M3 Embedding Generator for Full Hybrid Semantic-Graph Search

BGE-M3 provides three types of embeddings:
1. Dense: 1024-dim vectors for semantic similarity
2. Sparse: Lexical (BM25-like) for keyword matching
3. ColBERT: Multi-vector for fine-grained matching

Features:
- Multi-GPU support for DGX Spark
- Batch processing with dynamic batching
- Checkpoint/resume support
- Memory-efficient storage with memory-mapped arrays
"""

import os
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Generator
import pickle
import json
from collections import defaultdict
import time

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    def tqdm(x, **kwargs):
        return x

# Check for GPU
try:
    import torch
    TORCH_AVAILABLE = True
    GPU_COUNT = torch.cuda.device_count()
    print(f"[BGE-M3] PyTorch available, {GPU_COUNT} GPU(s) detected")
except ImportError:
    TORCH_AVAILABLE = False
    GPU_COUNT = 0
    print("[BGE-M3] PyTorch not available, using CPU")

# Try to import FlagEmbedding for BGE-M3
try:
    from FlagEmbedding import BGEM3FlagModel
    BGE_AVAILABLE = True
    print("[BGE-M3] FlagEmbedding library available")
except ImportError:
    BGE_AVAILABLE = False
    print("[BGE-M3] FlagEmbedding not installed. Install with: pip install FlagEmbedding")


class BGEEmbedder:
    """
    BGE-M3 embedding generator with multi-GPU support.

    Generates dense, sparse, and optionally ColBERT embeddings
    for semantic search and graph construction.
    """

    # Model configuration
    MODEL_NAME = "BAAI/bge-m3"
    DENSE_DIM = 1024
    MAX_LENGTH = 8192  # BGE-M3 supports 8K tokens

    def __init__(
        self,
        device: str = None,
        use_fp16: bool = True,
        batch_size: int = 8,
        max_length: int = 8192,
        return_sparse: bool = True,
        return_colbert: bool = False,  # ColBERT uses more memory
    ):
        """
        Initialize BGE-M3 embedder.

        Args:
            device: Device to use ('cuda', 'cuda:0', 'cpu', or None for auto)
            use_fp16: Use half precision for faster inference
            batch_size: Batch size for embedding (tune for GPU memory)
            max_length: Maximum token length
            return_sparse: Return sparse embeddings
            return_colbert: Return ColBERT embeddings (memory intensive)
        """
        self.use_fp16 = use_fp16
        self.batch_size = batch_size
        self.max_length = min(max_length, self.MAX_LENGTH)
        self.return_sparse = return_sparse
        self.return_colbert = return_colbert

        # Determine device
        if device is None:
            self.device = 'cuda' if TORCH_AVAILABLE and torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        self.model = None
        self._initialized = False

    def initialize(self):
        """Lazy initialization of the model."""
        if self._initialized:
            return

        if not BGE_AVAILABLE:
            raise ImportError(
                "FlagEmbedding library not installed. "
                "Install with: pip install FlagEmbedding"
            )

        print(f"[BGE-M3] Loading model on {self.device}...")
        start = time.time()

        self.model = BGEM3FlagModel(
            self.MODEL_NAME,
            use_fp16=self.use_fp16,
            device=self.device
        )

        print(f"[BGE-M3] Model loaded in {time.time() - start:.1f}s")
        self._initialized = True

    def embed_texts(
        self,
        texts: List[str],
        show_progress: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of texts to embed
            show_progress: Show progress bar

        Returns:
            Dict with 'dense', 'sparse' (optional), 'colbert' (optional) arrays
        """
        self.initialize()

        if not texts:
            return {'dense': np.array([])}

        # Process in batches
        all_dense = []
        all_sparse = [] if self.return_sparse else None
        all_colbert = [] if self.return_colbert else None

        iterator = range(0, len(texts), self.batch_size)
        if show_progress and TQDM_AVAILABLE:
            iterator = tqdm(iterator, desc="Embedding", unit="batch")

        for i in iterator:
            batch_texts = texts[i:i + self.batch_size]

            # Generate embeddings
            output = self.model.encode(
                batch_texts,
                batch_size=len(batch_texts),
                max_length=self.max_length,
                return_dense=True,
                return_sparse=self.return_sparse,
                return_colbert_vecs=self.return_colbert,
            )

            # Collect results
            all_dense.append(output['dense_vecs'])

            if self.return_sparse and 'lexical_weights' in output:
                all_sparse.append(output['lexical_weights'])

            if self.return_colbert and 'colbert_vecs' in output:
                all_colbert.append(output['colbert_vecs'])

        # Stack results
        result = {
            'dense': np.vstack(all_dense) if all_dense else np.array([])
        }

        if all_sparse:
            result['sparse'] = all_sparse  # Keep as list of dicts

        if all_colbert:
            result['colbert'] = all_colbert  # Keep as list of arrays

        return result

    def embed_single(self, text: str) -> Dict[str, np.ndarray]:
        """Embed a single text."""
        return self.embed_texts([text])

    def compute_similarity(
        self,
        query_embedding: np.ndarray,
        doc_embeddings: np.ndarray,
        metric: str = 'cosine'
    ) -> np.ndarray:
        """
        Compute similarity between query and documents.

        Args:
            query_embedding: Query embedding (1, D) or (D,)
            doc_embeddings: Document embeddings (N, D)
            metric: 'cosine' or 'dot'

        Returns:
            Similarity scores (N,)
        """
        query = query_embedding.reshape(1, -1)

        if metric == 'cosine':
            # Normalize
            query_norm = query / (np.linalg.norm(query, axis=1, keepdims=True) + 1e-9)
            doc_norm = doc_embeddings / (np.linalg.norm(doc_embeddings, axis=1, keepdims=True) + 1e-9)
            return (query_norm @ doc_norm.T).flatten()
        else:
            return (query @ doc_embeddings.T).flatten()


class EmbeddingStore:
    """
    Persistent storage for embeddings with memory-mapped arrays.

    Supports incremental building and efficient retrieval.
    """

    def __init__(
        self,
        store_dir: str = "embeddings_store",
        dense_dim: int = 1024
    ):
        """
        Initialize embedding store.

        Args:
            store_dir: Directory for storing embeddings
            dense_dim: Dimension of dense embeddings
        """
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.dense_dim = dense_dim

        # Paths
        self.dense_path = self.store_dir / "dense_embeddings.npy"
        self.sparse_path = self.store_dir / "sparse_embeddings.pkl"
        self.ids_path = self.store_dir / "paper_ids.json"
        self.metadata_path = self.store_dir / "metadata.json"

        # State
        self.paper_ids: List[str] = []
        self.paper_id_to_idx: Dict[str, int] = {}
        self.dense_mmap: Optional[np.memmap] = None
        self.sparse_embeddings: List[Dict] = []

        # Load existing store if available
        self._load_state()

    def _load_state(self):
        """Load existing state from disk."""
        if self.ids_path.exists():
            with open(self.ids_path, 'r') as f:
                self.paper_ids = json.load(f)
            self.paper_id_to_idx = {pid: idx for idx, pid in enumerate(self.paper_ids)}
            print(f"[EmbeddingStore] Loaded {len(self.paper_ids):,} paper IDs")

        if self.sparse_path.exists():
            with open(self.sparse_path, 'rb') as f:
                self.sparse_embeddings = pickle.load(f)
            print(f"[EmbeddingStore] Loaded {len(self.sparse_embeddings):,} sparse embeddings")

        if self.dense_path.exists():
            self.dense_mmap = np.load(self.dense_path, mmap_mode='r')
            print(f"[EmbeddingStore] Dense embeddings shape: {self.dense_mmap.shape}")

    def _save_state(self):
        """Save state to disk."""
        with open(self.ids_path, 'w') as f:
            json.dump(self.paper_ids, f)

        with open(self.sparse_path, 'wb') as f:
            pickle.dump(self.sparse_embeddings, f)

        metadata = {
            'count': len(self.paper_ids),
            'dense_dim': self.dense_dim,
            'has_sparse': len(self.sparse_embeddings) > 0,
        }
        with open(self.metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    def add_embeddings(
        self,
        paper_ids: List[str],
        dense_embeddings: np.ndarray,
        sparse_embeddings: Optional[List[Dict]] = None
    ):
        """
        Add embeddings to the store.

        Args:
            paper_ids: List of paper IDs
            dense_embeddings: Dense embeddings (N, D)
            sparse_embeddings: Optional sparse embeddings
        """
        if len(paper_ids) != dense_embeddings.shape[0]:
            raise ValueError("Mismatch between paper_ids and embeddings count")

        # Check for duplicates
        new_ids = []
        new_dense = []
        new_sparse = []

        for i, pid in enumerate(paper_ids):
            if pid not in self.paper_id_to_idx:
                new_ids.append(pid)
                new_dense.append(dense_embeddings[i])
                if sparse_embeddings:
                    new_sparse.append(sparse_embeddings[i])

        if not new_ids:
            return

        # Update paper IDs
        start_idx = len(self.paper_ids)
        for pid in new_ids:
            self.paper_id_to_idx[pid] = len(self.paper_ids)
            self.paper_ids.append(pid)

        # Stack new dense embeddings
        new_dense_array = np.array(new_dense)

        # Append to file
        if self.dense_path.exists():
            existing = np.load(self.dense_path)
            combined = np.vstack([existing, new_dense_array])
        else:
            combined = new_dense_array

        np.save(self.dense_path, combined)
        self.dense_mmap = np.load(self.dense_path, mmap_mode='r')

        # Add sparse embeddings
        if new_sparse:
            self.sparse_embeddings.extend(new_sparse)

        # Save state
        self._save_state()

        print(f"[EmbeddingStore] Added {len(new_ids)} embeddings, total: {len(self.paper_ids):,}")

    def get_dense_embedding(self, paper_id: str) -> Optional[np.ndarray]:
        """Get dense embedding for a paper ID."""
        if paper_id not in self.paper_id_to_idx:
            return None
        idx = self.paper_id_to_idx[paper_id]
        return self.dense_mmap[idx].copy()

    def get_dense_embeddings_batch(self, paper_ids: List[str]) -> np.ndarray:
        """Get dense embeddings for multiple paper IDs."""
        indices = [self.paper_id_to_idx[pid] for pid in paper_ids if pid in self.paper_id_to_idx]
        if not indices:
            return np.array([])
        return self.dense_mmap[indices].copy()

    def get_all_dense_embeddings(self) -> np.ndarray:
        """Get all dense embeddings."""
        if self.dense_mmap is None:
            return np.array([])
        return self.dense_mmap[:]

    def get_sparse_embedding(self, paper_id: str) -> Optional[Dict]:
        """Get sparse embedding for a paper ID."""
        if paper_id not in self.paper_id_to_idx:
            return None
        idx = self.paper_id_to_idx[paper_id]
        if idx < len(self.sparse_embeddings):
            return self.sparse_embeddings[idx]
        return None

    def __len__(self) -> int:
        return len(self.paper_ids)

    def __contains__(self, paper_id: str) -> bool:
        return paper_id in self.paper_id_to_idx


class EmbeddingPipeline:
    """
    End-to-end pipeline for generating and storing embeddings.

    Integrates PaperIterator -> BGEEmbedder -> EmbeddingStore
    """

    def __init__(
        self,
        data_dir: str = "s2orc_data",
        store_dir: str = "embeddings_store",
        checkpoint_dir: str = "checkpoints",
        batch_size: int = 8,
        use_fp16: bool = True,
        return_sparse: bool = True,
    ):
        """
        Initialize the embedding pipeline.

        Args:
            data_dir: Path to s2orc_data directory
            store_dir: Path to store embeddings
            checkpoint_dir: Path for checkpoints
            batch_size: Batch size for embedding generation
            use_fp16: Use half precision
            return_sparse: Generate sparse embeddings
        """
        self.data_dir = data_dir
        self.store_dir = store_dir
        self.checkpoint_dir = checkpoint_dir
        self.batch_size = batch_size

        # Initialize components (lazy)
        self.embedder = BGEEmbedder(
            batch_size=batch_size,
            use_fp16=use_fp16,
            return_sparse=return_sparse,
        )
        self.store = EmbeddingStore(store_dir)

    def run(
        self,
        max_papers: Optional[int] = None,
        fields_filter: Optional[List[str]] = None,
        checkpoint_every: int = 1000,
        resume: bool = True
    ):
        """
        Run the embedding pipeline.

        Args:
            max_papers: Maximum papers to process
            fields_filter: Only process these fields
            checkpoint_every: Save every N papers
            resume: Resume from checkpoint
        """
        # Import here to avoid circular imports
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from data.paper_iterator import PaperIterator

        iterator = PaperIterator(
            data_dir=self.data_dir,
            batch_size=self.batch_size * 4,  # Larger batches for preprocessing
            checkpoint_dir=self.checkpoint_dir,
            preprocess=True
        )

        processed = 0
        skipped = 0

        print(f"\n[Pipeline] Starting embedding generation")
        print(f"[Pipeline] Store contains {len(self.store):,} embeddings")

        for batch in iterator.iterate(
            fields_filter=fields_filter,
            max_papers=max_papers,
            resume=resume,
            checkpoint_every=checkpoint_every
        ):
            # Filter out already processed papers
            new_papers = [p for p in batch if p['paper_id'] not in self.store]
            skipped += len(batch) - len(new_papers)

            if not new_papers:
                continue

            # Extract texts and IDs
            paper_ids = [p['paper_id'] for p in new_papers]
            texts = [p['embedding_text'] for p in new_papers]

            # Generate embeddings
            embeddings = self.embedder.embed_texts(texts)

            # Store embeddings
            self.store.add_embeddings(
                paper_ids=paper_ids,
                dense_embeddings=embeddings['dense'],
                sparse_embeddings=embeddings.get('sparse')
            )

            processed += len(new_papers)

            if processed % checkpoint_every == 0:
                print(f"[Pipeline] Processed: {processed:,}, Skipped: {skipped:,}, "
                      f"Total stored: {len(self.store):,}")

        print(f"\n[Pipeline] Complete!")
        print(f"[Pipeline] Processed: {processed:,}")
        print(f"[Pipeline] Skipped (already stored): {skipped:,}")
        print(f"[Pipeline] Total embeddings: {len(self.store):,}")


if __name__ == "__main__":
    # Test embedding generation
    print("\n=== BGE-M3 Embedding Test ===\n")

    # Test with sample texts
    sample_texts = [
        "Deep learning has revolutionized natural language processing with transformer architectures.",
        "AlphaFold uses attention mechanisms to predict protein structures with high accuracy.",
        "Climate models increasingly incorporate machine learning for weather prediction.",
    ]

    if BGE_AVAILABLE:
        embedder = BGEEmbedder(batch_size=2, return_sparse=True)
        embeddings = embedder.embed_texts(sample_texts, show_progress=True)

        print(f"\nDense embeddings shape: {embeddings['dense'].shape}")
        print(f"Embedding dimension: {embeddings['dense'].shape[1]}")

        if 'sparse' in embeddings:
            print(f"Sparse embeddings count: {len(embeddings['sparse'])}")
            print(f"Sample sparse keys: {list(embeddings['sparse'][0].keys())[:5]}")

        # Test similarity
        print("\nSimilarity test:")
        sims = embedder.compute_similarity(
            embeddings['dense'][0],
            embeddings['dense']
        )
        for i, sim in enumerate(sims):
            print(f"  Text 0 vs Text {i}: {sim:.4f}")
    else:
        print("BGE-M3 not available. Install with: pip install FlagEmbedding")

    # Test store
    print("\n=== Embedding Store Test ===\n")
    store = EmbeddingStore("test_embeddings")

    # Add dummy embeddings
    dummy_ids = ["paper_1", "paper_2", "paper_3"]
    dummy_embeddings = np.random.randn(3, 1024).astype(np.float32)
    store.add_embeddings(dummy_ids, dummy_embeddings)

    print(f"Store size: {len(store)}")
    print(f"Contains paper_1: {'paper_1' in store}")

    retrieved = store.get_dense_embedding("paper_1")
    print(f"Retrieved embedding shape: {retrieved.shape}")

    # Clean up test
    import shutil
    shutil.rmtree("test_embeddings", ignore_errors=True)
    print("\nTest cleanup complete.")
