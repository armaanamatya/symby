"""
Semantic Graph Construction for Full Hybrid Semantic-Graph Search

Builds graph edges from embedding similarity instead of citation data.
This enables:
- Discovering implicit connections (related methods, similar findings)
- Cross-domain knowledge linking
- Temporal influence patterns
- Method diffusion tracking

Supports:
- NetworkX for CPU
- cuGraph for GPU-accelerated graph operations (DGX Spark)
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Generator, Set
from collections import defaultdict
import json
import pickle
import time

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    print("[SemanticGraph] NetworkX not available")

# Try cuGraph for GPU acceleration
try:
    import cugraph
    import cudf
    CUGRAPH_AVAILABLE = True
    print("[SemanticGraph] cuGraph (RAPIDS) available for GPU acceleration")
except ImportError:
    CUGRAPH_AVAILABLE = False

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    def tqdm(x, **kwargs):
        return x


class SemanticGraphBuilder:
    """
    Build semantic similarity graph from embeddings.

    Creates edges between papers based on embedding similarity,
    enabling graph-based discovery of related research.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.75,
        max_neighbors: int = 20,
        use_gpu: bool = True,
        edge_batch_size: int = 10000,
    ):
        """
        Initialize graph builder.

        Args:
            similarity_threshold: Minimum cosine similarity for edge creation
            max_neighbors: Maximum edges per node
            use_gpu: Use GPU acceleration (cuGraph) if available
            edge_batch_size: Batch size for edge creation
        """
        self.similarity_threshold = similarity_threshold
        self.max_neighbors = max_neighbors
        self.use_gpu = use_gpu and CUGRAPH_AVAILABLE
        self.edge_batch_size = edge_batch_size

        # Graph storage
        self.edges: List[Tuple[str, str, float]] = []  # (source, target, weight)
        self.node_metadata: Dict[str, Dict] = {}
        self.graph = None

    def build_from_index(
        self,
        vector_index,  # VectorIndex instance
        paper_metadata: Dict[str, Dict] = None,
        batch_size: int = 1000,
        show_progress: bool = True
    ):
        """
        Build graph edges by querying the vector index for each paper.

        Args:
            vector_index: VectorIndex with embeddings
            paper_metadata: Optional metadata for nodes
            batch_size: Papers to process per batch
            show_progress: Show progress bar
        """
        print(f"[SemanticGraph] Building edges for {len(vector_index):,} papers...")
        print(f"[SemanticGraph] Threshold: {self.similarity_threshold}, Max neighbors: {self.max_neighbors}")
        start = time.time()

        paper_ids = vector_index.paper_ids
        n_papers = len(paper_ids)

        # Store metadata
        if paper_metadata:
            self.node_metadata = paper_metadata

        # Process in batches
        edge_count = 0
        seen_edges: Set[Tuple[str, str]] = set()

        iterator = range(0, n_papers, batch_size)
        if show_progress and TQDM_AVAILABLE:
            iterator = tqdm(iterator, desc="Building edges", unit="batch")

        for batch_start in iterator:
            batch_end = min(batch_start + batch_size, n_papers)
            batch_ids = paper_ids[batch_start:batch_end]

            # Get embeddings for this batch
            batch_embeddings = vector_index.dense_index.dense_mmap[batch_start:batch_end] \
                if hasattr(vector_index, 'dense_index') else None

            if batch_embeddings is None:
                # Try to get from store
                from ..embeddings import EmbeddingStore
                continue

            # Query for similar papers
            scores, neighbor_ids = vector_index.search(
                batch_embeddings,
                k=self.max_neighbors + 1  # +1 to exclude self
            )

            # Create edges
            for i, (source_id, score_row, id_row) in enumerate(zip(batch_ids, scores, neighbor_ids)):
                for score, target_id in zip(score_row, id_row):
                    # Skip self-loops
                    if target_id == source_id:
                        continue

                    # Check threshold
                    if score < self.similarity_threshold:
                        continue

                    # Avoid duplicate edges (undirected graph)
                    edge_key = tuple(sorted([source_id, target_id]))
                    if edge_key in seen_edges:
                        continue

                    seen_edges.add(edge_key)
                    self.edges.append((source_id, target_id, float(score)))
                    edge_count += 1

        print(f"[SemanticGraph] Created {edge_count:,} edges in {time.time() - start:.1f}s")
        print(f"[SemanticGraph] Avg edges per node: {edge_count * 2 / n_papers:.1f}")

    def build_from_embeddings(
        self,
        embeddings: np.ndarray,
        paper_ids: List[str],
        paper_metadata: Dict[str, Dict] = None,
        batch_size: int = 500,
        show_progress: bool = True
    ):
        """
        Build graph directly from embeddings using batch similarity computation.

        More memory efficient for smaller datasets.

        Args:
            embeddings: Dense embeddings (N, D)
            paper_ids: Paper IDs
            paper_metadata: Optional metadata
            batch_size: Papers per batch
            show_progress: Show progress bar
        """
        print(f"[SemanticGraph] Building edges for {len(paper_ids):,} papers...")
        start = time.time()

        n = len(paper_ids)

        # Normalize embeddings for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / (norms + 1e-9)

        # Store metadata
        if paper_metadata:
            self.node_metadata = paper_metadata

        edge_count = 0
        seen_edges: Set[Tuple[str, str]] = set()

        iterator = range(0, n, batch_size)
        if show_progress and TQDM_AVAILABLE:
            iterator = tqdm(iterator, desc="Computing similarities", unit="batch")

        for batch_start in iterator:
            batch_end = min(batch_start + batch_size, n)
            batch = normalized[batch_start:batch_end]

            # Compute similarities with all papers
            similarities = batch @ normalized.T  # (batch_size, n)

            # Find edges above threshold
            for i, row in enumerate(similarities):
                source_idx = batch_start + i
                source_id = paper_ids[source_idx]

                # Get top-k neighbors above threshold
                top_indices = np.argsort(row)[::-1][:self.max_neighbors + 1]

                for target_idx in top_indices:
                    if target_idx == source_idx:
                        continue

                    score = row[target_idx]
                    if score < self.similarity_threshold:
                        break

                    target_id = paper_ids[target_idx]

                    # Avoid duplicates
                    edge_key = tuple(sorted([source_id, target_id]))
                    if edge_key in seen_edges:
                        continue

                    seen_edges.add(edge_key)
                    self.edges.append((source_id, target_id, float(score)))
                    edge_count += 1

        print(f"[SemanticGraph] Created {edge_count:,} edges in {time.time() - start:.1f}s")

    def to_networkx(self) -> 'nx.Graph':
        """Convert to NetworkX graph."""
        if not NETWORKX_AVAILABLE:
            raise ImportError("NetworkX not available")

        G = nx.Graph()

        # Add nodes with metadata
        for paper_id, metadata in self.node_metadata.items():
            G.add_node(paper_id, **metadata)

        # Add edges
        for source, target, weight in self.edges:
            G.add_edge(source, target, weight=weight, similarity=weight)

        self.graph = G
        print(f"[SemanticGraph] NetworkX graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
        return G

    def to_cugraph(self):
        """Convert to cuGraph for GPU-accelerated analytics."""
        if not CUGRAPH_AVAILABLE:
            raise ImportError("cuGraph not available. Install RAPIDS.")

        # Create edge DataFrame
        sources = [e[0] for e in self.edges]
        targets = [e[1] for e in self.edges]
        weights = [e[2] for e in self.edges]

        edge_df = cudf.DataFrame({
            'source': sources,
            'target': targets,
            'weight': weights
        })

        # Create graph
        G = cugraph.Graph()
        G.from_cudf_edgelist(edge_df, source='source', destination='target', edge_attr='weight')

        self.graph = G
        print(f"[SemanticGraph] cuGraph: {G.number_of_vertices():,} nodes, {G.number_of_edges():,} edges")
        return G

    def get_neighbors(self, paper_id: str, k: int = 10) -> List[Tuple[str, float]]:
        """Get k most similar neighbors for a paper."""
        if self.graph is None:
            self.to_networkx()

        if not self.graph.has_node(paper_id):
            return []

        neighbors = []
        for neighbor in self.graph.neighbors(paper_id):
            weight = self.graph[paper_id][neighbor].get('weight', 0)
            neighbors.append((neighbor, weight))

        return sorted(neighbors, key=lambda x: -x[1])[:k]

    def find_path(
        self,
        source_id: str,
        target_id: str,
        max_length: int = 5
    ) -> Optional[List[str]]:
        """Find shortest path between two papers."""
        if self.graph is None:
            self.to_networkx()

        if not self.graph.has_node(source_id) or not self.graph.has_node(target_id):
            return None

        try:
            path = nx.shortest_path(self.graph, source_id, target_id)
            if len(path) <= max_length:
                return path
        except nx.NetworkXNoPath:
            pass

        return None

    def get_subgraph(
        self,
        seed_ids: List[str],
        hops: int = 2,
        max_nodes: int = 100
    ) -> 'nx.Graph':
        """Extract subgraph around seed papers."""
        if self.graph is None:
            self.to_networkx()

        nodes = set(seed_ids)
        frontier = set(seed_ids)

        for _ in range(hops):
            if len(nodes) >= max_nodes:
                break

            new_frontier = set()
            for node in frontier:
                if node in self.graph:
                    neighbors = list(self.graph.neighbors(node))
                    # Sort by edge weight and take top
                    neighbors = sorted(
                        neighbors,
                        key=lambda n: self.graph[node][n].get('weight', 0),
                        reverse=True
                    )[:10]
                    new_frontier.update(neighbors)

            nodes.update(new_frontier)
            frontier = new_frontier - (nodes - new_frontier)

            if len(nodes) >= max_nodes:
                nodes = set(list(nodes)[:max_nodes])
                break

        return self.graph.subgraph(nodes).copy()

    def save(self, path: str):
        """Save graph to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save edges
        with open(path / "edges.pkl", 'wb') as f:
            pickle.dump(self.edges, f)

        # Save node metadata
        with open(path / "node_metadata.json", 'w') as f:
            json.dump(self.node_metadata, f)

        # Save config
        config = {
            'similarity_threshold': self.similarity_threshold,
            'max_neighbors': self.max_neighbors,
            'edge_count': len(self.edges),
            'node_count': len(self.node_metadata),
        }
        with open(path / "config.json", 'w') as f:
            json.dump(config, f, indent=2)

        print(f"[SemanticGraph] Saved to {path}")

    def load(self, path: str):
        """Load graph from disk."""
        path = Path(path)

        with open(path / "edges.pkl", 'rb') as f:
            self.edges = pickle.load(f)

        if (path / "node_metadata.json").exists():
            with open(path / "node_metadata.json", 'r') as f:
                self.node_metadata = json.load(f)

        with open(path / "config.json", 'r') as f:
            config = json.load(f)
            self.similarity_threshold = config['similarity_threshold']
            self.max_neighbors = config['max_neighbors']

        print(f"[SemanticGraph] Loaded {len(self.edges):,} edges from {path}")

    def get_stats(self) -> Dict:
        """Get graph statistics."""
        if self.graph is None:
            self.to_networkx()

        G = self.graph
        stats = {
            'nodes': G.number_of_nodes(),
            'edges': G.number_of_edges(),
            'density': nx.density(G),
            'avg_degree': sum(dict(G.degree()).values()) / G.number_of_nodes() if G.number_of_nodes() > 0 else 0,
        }

        # Get connected components
        if G.number_of_nodes() > 0:
            components = list(nx.connected_components(G))
            stats['connected_components'] = len(components)
            stats['largest_component_size'] = max(len(c) for c in components)

        return stats


class TemporalSemanticGraph(SemanticGraphBuilder):
    """
    Semantic graph with temporal awareness.

    Tracks how research topics evolve and spread over time.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.temporal_edges: Dict[int, List[Tuple[str, str, float]]] = defaultdict(list)

    def build_temporal(
        self,
        embeddings: np.ndarray,
        paper_ids: List[str],
        paper_years: Dict[str, int],
        **kwargs
    ):
        """Build graph with temporal information."""
        # Build base graph
        self.build_from_embeddings(embeddings, paper_ids, **kwargs)

        # Organize edges by year
        for source, target, weight in self.edges:
            source_year = paper_years.get(source, 0)
            target_year = paper_years.get(target, 0)

            # Track influence direction (earlier -> later)
            if source_year and target_year and source_year < target_year:
                self.temporal_edges[source_year].append((source, target, weight))
            elif target_year and source_year and target_year < source_year:
                self.temporal_edges[target_year].append((target, source, weight))

    def get_influence_over_time(self, paper_id: str) -> Dict[int, int]:
        """Track how a paper influenced later work over years."""
        influence = defaultdict(int)

        for year, edges in self.temporal_edges.items():
            for source, target, _ in edges:
                if source == paper_id:
                    influence[year] += 1

        return dict(sorted(influence.items()))

    def get_adoption_curve(self, technique_ids: List[str]) -> Dict[int, float]:
        """Track adoption of a technique (set of papers) over time."""
        adoption = defaultdict(int)
        total_edges = defaultdict(int)

        for year, edges in self.temporal_edges.items():
            for source, target, _ in edges:
                total_edges[year] += 1
                if source in technique_ids or target in technique_ids:
                    adoption[year] += 1

        # Normalize
        curve = {}
        for year in sorted(total_edges.keys()):
            if total_edges[year] > 0:
                curve[year] = adoption[year] / total_edges[year]

        return curve


if __name__ == "__main__":
    # Test semantic graph
    print("\n=== Semantic Graph Test ===\n")

    if NETWORKX_AVAILABLE:
        # Create dummy data
        n = 100
        dim = 128
        embeddings = np.random.randn(n, dim).astype(np.float32)
        paper_ids = [f"paper_{i}" for i in range(n)]

        # Build graph
        builder = SemanticGraphBuilder(
            similarity_threshold=0.0,  # Low for random data
            max_neighbors=5
        )
        builder.build_from_embeddings(embeddings, paper_ids)

        # Convert to NetworkX
        G = builder.to_networkx()

        # Get stats
        stats = builder.get_stats()
        print(f"Graph stats: {stats}")

        # Test neighbor lookup
        neighbors = builder.get_neighbors("paper_0", k=5)
        print(f"\nNeighbors of paper_0:")
        for pid, weight in neighbors:
            print(f"  {pid}: {weight:.4f}")

        # Test subgraph
        subgraph = builder.get_subgraph(["paper_0", "paper_1"], hops=2)
        print(f"\nSubgraph: {subgraph.number_of_nodes()} nodes, {subgraph.number_of_edges()} edges")
    else:
        print("NetworkX not available")
