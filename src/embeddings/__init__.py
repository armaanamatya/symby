"""
Embedding Generation Module for Full Hybrid Semantic-Graph Search

Uses BGE-M3 for:
- Dense embeddings (1024-dim)
- Sparse embeddings (lexical)
- ColBERT multi-vector embeddings

Optimized for DGX Spark with multi-GPU support.
"""

from .bge_m3 import BGEEmbedder, EmbeddingStore

__all__ = ['BGEEmbedder', 'EmbeddingStore']
