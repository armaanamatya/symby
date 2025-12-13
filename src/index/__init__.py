"""
Vector Index Module for Full Hybrid Semantic-Graph Search

Supports:
- FAISS for CPU/GPU vector search
- cuVS for NVIDIA GPU-accelerated search (DGX Spark)
- Hybrid sparse+dense indexing
"""

from .vector_index import VectorIndex, HybridIndex

__all__ = ['VectorIndex', 'HybridIndex']
