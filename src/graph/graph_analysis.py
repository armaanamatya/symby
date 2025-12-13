"""
Graph Analysis Engine
GPU-accelerated graph analytics using cuGraph with NetworkX fallback
Implements: PageRank, Community Detection, Path Analysis, Citation Flow
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict
from dataclasses import dataclass

# Try GPU graph library first
try:
    import cugraph
    import cudf
    GPU_GRAPH = True
    print("cuGraph available for GPU-accelerated analysis")
except ImportError:
    GPU_GRAPH = False

try:
    import networkx as nx
    from networkx.algorithms import community as nx_community
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    print("Warning: NetworkX not available")

try:
    import pandas as pd
except ImportError:
    import pandas as pd


@dataclass
class InfluenceScore:
    """Influence metrics for a paper"""
    paper_id: str
    pagerank: float
    in_degree: int  # citations received
    out_degree: int  # references made
    betweenness: float
    community_id: int
    influence_rank: int


@dataclass
class CitationFlow:
    """Citation flow between fields"""
    source_field: str
    target_field: str
    flow_count: int
    flow_strength: float  # normalized
    key_papers: List[str]


class GraphAnalyzer:
    """
    GPU-accelerated graph analysis for citation networks.

    Provides:
    - PageRank influence scoring
    - Community detection (Louvain/Leiden)
    - Citation path tracing
    - Field-to-field flow analysis
    - Temporal diffusion analysis
    """

    def __init__(self, nodes: Dict, edges: List[Tuple[str, str]]):
        """
        Initialize with graph data.

        Args:
            nodes: Dict mapping paper_id -> {title, year, fields, citation_count, ...}
            edges: List of (citing_paper_id, cited_paper_id) tuples
        """
        self.nodes = nodes
        self.edges = edges
        self.node_ids = list(nodes.keys())

        # Create ID mappings for numeric operations
        self.id_to_idx = {pid: i for i, pid in enumerate(self.node_ids)}
        self.idx_to_id = {i: pid for i, pid in enumerate(self.node_ids)}

        # Build graph representation
        self._build_graph()

        # Cache for computed metrics
        self._pagerank_cache = None
        self._community_cache = None
        self._betweenness_cache = None

    def _build_graph(self):
        """Build graph in appropriate format"""
        # Filter edges to only include known nodes
        valid_edges = [(s, t) for s, t in self.edges
                       if s in self.id_to_idx and t in self.id_to_idx]

        if GPU_GRAPH:
            self._build_cugraph(valid_edges)
        elif NETWORKX_AVAILABLE:
            self._build_networkx(valid_edges)
        else:
            raise ImportError("No graph library available")

    def _build_cugraph(self, edges: List[Tuple]):
        """Build cuGraph representation"""
        if not edges:
            self.G = None
            return

        # Convert to numeric edge list
        src = [self.id_to_idx[e[0]] for e in edges]
        dst = [self.id_to_idx[e[1]] for e in edges]

        edge_df = cudf.DataFrame({'src': src, 'dst': dst})

        self.G = cugraph.Graph(directed=True)
        self.G.from_cudf_edgelist(edge_df, source='src', destination='dst')

    def _build_networkx(self, edges: List[Tuple]):
        """Build NetworkX representation"""
        self.G = nx.DiGraph()

        # Add nodes with attributes
        for pid, attrs in self.nodes.items():
            self.G.add_node(pid, **attrs)

        # Add edges
        self.G.add_edges_from(edges)

    def compute_pagerank(self, alpha: float = 0.85, max_iter: int = 100) -> Dict[str, float]:
        """
        Compute PageRank influence scores.

        Higher PageRank = more influential paper (cited by influential papers)
        """
        if self._pagerank_cache is not None:
            return self._pagerank_cache

        if self.G is None or len(self.edges) == 0:
            return {pid: 0.0 for pid in self.node_ids}

        if GPU_GRAPH:
            pr_df = cugraph.pagerank(self.G, alpha=alpha, max_iter=max_iter)
            # Map back to paper IDs
            result = {}
            for _, row in pr_df.to_pandas().iterrows():
                pid = self.idx_to_id.get(int(row['vertex']))
                if pid:
                    result[pid] = float(row['pagerank'])
        else:
            result = nx.pagerank(self.G, alpha=alpha, max_iter=max_iter)

        self._pagerank_cache = result
        return result

    def compute_communities(self, resolution: float = 1.0) -> Dict[str, int]:
        """
        Detect research communities using Louvain algorithm.

        Returns mapping of paper_id -> community_id
        """
        if self._community_cache is not None:
            return self._community_cache

        if self.G is None or len(self.edges) == 0:
            return {pid: 0 for pid in self.node_ids}

        if GPU_GRAPH:
            # cuGraph Louvain (works on undirected)
            G_undirected = self.G.to_undirected()
            parts, modularity = cugraph.louvain(G_undirected, resolution=resolution)

            result = {}
            for _, row in parts.to_pandas().iterrows():
                pid = self.idx_to_id.get(int(row['vertex']))
                if pid:
                    result[pid] = int(row['partition'])
        else:
            # NetworkX Louvain
            G_undirected = self.G.to_undirected()
            communities = nx_community.louvain_communities(G_undirected, resolution=resolution)
            result = {}
            for comm_id, comm in enumerate(communities):
                for pid in comm:
                    result[pid] = comm_id

        self._community_cache = result
        return result

    def compute_betweenness(self, k: int = 100) -> Dict[str, float]:
        """
        Compute betweenness centrality (papers that bridge communities).

        Uses sampling (k) for large graphs.
        """
        if self._betweenness_cache is not None:
            return self._betweenness_cache

        if self.G is None or len(self.edges) == 0:
            return {pid: 0.0 for pid in self.node_ids}

        if GPU_GRAPH:
            bc_df = cugraph.betweenness_centrality(self.G, k=min(k, len(self.node_ids)))
            result = {}
            for _, row in bc_df.to_pandas().iterrows():
                pid = self.idx_to_id.get(int(row['vertex']))
                if pid:
                    result[pid] = float(row['betweenness_centrality'])
        else:
            # Sample for large graphs
            if len(self.G) > 1000:
                result = nx.betweenness_centrality(self.G, k=min(k, len(self.G)))
            else:
                result = nx.betweenness_centrality(self.G)

        self._betweenness_cache = result
        return result

    def get_influence_scores(self) -> List[InfluenceScore]:
        """Get comprehensive influence scores for all papers"""
        pagerank = self.compute_pagerank()
        communities = self.compute_communities()
        betweenness = self.compute_betweenness()

        # Compute degrees
        in_degrees = defaultdict(int)
        out_degrees = defaultdict(int)
        for src, dst in self.edges:
            out_degrees[src] += 1
            in_degrees[dst] += 1

        # Build scores
        scores = []
        for pid in self.node_ids:
            scores.append(InfluenceScore(
                paper_id=pid,
                pagerank=pagerank.get(pid, 0.0),
                in_degree=in_degrees.get(pid, 0),
                out_degree=out_degrees.get(pid, 0),
                betweenness=betweenness.get(pid, 0.0),
                community_id=communities.get(pid, 0),
                influence_rank=0  # Will be set after sorting
            ))

        # Rank by PageRank
        scores.sort(key=lambda x: x.pagerank, reverse=True)
        for i, score in enumerate(scores):
            score.influence_rank = i + 1

        return scores

    def find_shortest_path(self, source_id: str, target_id: str) -> Optional[List[str]]:
        """Find shortest citation path between two papers"""
        if source_id not in self.id_to_idx or target_id not in self.id_to_idx:
            return None

        if GPU_GRAPH:
            src_idx = self.id_to_idx[source_id]
            tgt_idx = self.id_to_idx[target_id]

            try:
                distances = cugraph.sssp(self.G, src_idx)
                # Reconstruct path (cuGraph doesn't return path directly)
                # Fall back to NetworkX for path reconstruction
                if NETWORKX_AVAILABLE:
                    return self._nx_shortest_path(source_id, target_id)
            except:
                return None
        elif NETWORKX_AVAILABLE:
            return self._nx_shortest_path(source_id, target_id)

        return None

    def _nx_shortest_path(self, source_id: str, target_id: str) -> Optional[List[str]]:
        """NetworkX shortest path"""
        try:
            return nx.shortest_path(self.G, source_id, target_id)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def trace_influence_chain(self, paper_id: str, max_hops: int = 3) -> Dict:
        """
        Trace how a paper's influence spreads through citations.

        Returns papers influenced at each hop level.
        """
        if paper_id not in self.id_to_idx:
            return {'error': 'Paper not found'}

        # Build citing index
        citing_index = defaultdict(set)
        for src, dst in self.edges:
            citing_index[dst].add(src)  # papers that cite dst

        result = {
            'source': paper_id,
            'hops': []
        }

        current_level = {paper_id}
        visited = {paper_id}

        for hop in range(max_hops):
            next_level = set()
            for pid in current_level:
                citers = citing_index.get(pid, set())
                for citer in citers:
                    if citer not in visited:
                        next_level.add(citer)
                        visited.add(citer)

            if not next_level:
                break

            hop_info = {
                'hop': hop + 1,
                'paper_count': len(next_level),
                'papers': list(next_level)[:20],  # Limit for display
                'fields': self._get_field_distribution(next_level)
            }
            result['hops'].append(hop_info)
            current_level = next_level

        result['total_influenced'] = len(visited) - 1
        return result

    def _get_field_distribution(self, paper_ids: Set[str]) -> Dict[str, int]:
        """Get field distribution for a set of papers"""
        field_counts = defaultdict(int)
        for pid in paper_ids:
            node = self.nodes.get(pid, {})
            fields = node.get('fields', []) or []
            for field in fields:
                field_counts[field] += 1
        return dict(field_counts)

    def compute_field_citation_flow(self) -> pd.DataFrame:
        """
        Compute citation flow matrix between fields.

        Shows how often papers in field A cite papers in field B.
        """
        flow = defaultdict(lambda: defaultdict(int))

        for src, dst in self.edges:
            src_node = self.nodes.get(src, {})
            dst_node = self.nodes.get(dst, {})

            src_fields = src_node.get('fields', []) or []
            dst_fields = dst_node.get('fields', []) or []

            for sf in src_fields:
                for df in dst_fields:
                    flow[sf][df] += 1

        # Convert to DataFrame
        all_fields = set(flow.keys())
        for f in flow.values():
            all_fields.update(f.keys())

        all_fields = sorted(all_fields)
        matrix = pd.DataFrame(0, index=all_fields, columns=all_fields)

        for sf, targets in flow.items():
            for df, count in targets.items():
                matrix.loc[sf, df] = count

        return matrix

    def compute_temporal_diffusion(self, source_paper_id: str) -> pd.DataFrame:
        """
        Track how a paper's influence spreads over time.

        Returns DataFrame with year, field, citation_count.
        """
        if source_paper_id not in self.id_to_idx:
            return pd.DataFrame()

        # Build citing index
        citing_index = defaultdict(set)
        for src, dst in self.edges:
            citing_index[dst].add(src)

        # Get all papers that cite the source (directly or indirectly)
        all_citers = set()
        current = {source_paper_id}

        for _ in range(5):  # Max 5 hops
            next_level = set()
            for pid in current:
                next_level.update(citing_index.get(pid, set()))
            if not next_level - all_citers:
                break
            all_citers.update(next_level)
            current = next_level

        # Aggregate by year and field
        records = []
        for pid in all_citers:
            node = self.nodes.get(pid, {})
            year = node.get('year')
            fields = node.get('fields', []) or []

            for field in fields:
                records.append({
                    'year': year,
                    'field': field,
                    'paper_id': pid
                })

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)
        summary = df.groupby(['year', 'field']).size().reset_index(name='count')
        return summary.sort_values(['year', 'count'], ascending=[True, False])

    def identify_bridge_papers(self, top_n: int = 20) -> List[Dict]:
        """
        Identify papers that bridge different research communities.

        High betweenness + connects multiple fields = bridge paper.
        """
        betweenness = self.compute_betweenness()
        communities = self.compute_communities()

        # Get papers sorted by betweenness
        sorted_papers = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)

        bridges = []
        for pid, bc in sorted_papers[:top_n * 2]:  # Get extra in case some don't qualify
            node = self.nodes.get(pid, {})
            fields = node.get('fields', []) or []

            # Check if it connects multiple communities
            neighbor_communities = set()

            for src, dst in self.edges:
                if src == pid:
                    neighbor_communities.add(communities.get(dst, -1))
                elif dst == pid:
                    neighbor_communities.add(communities.get(src, -1))

            if len(neighbor_communities) >= 2 or len(fields) >= 2:
                bridges.append({
                    'paper_id': pid,
                    'title': node.get('title', '')[:100],
                    'year': node.get('year'),
                    'betweenness': bc,
                    'fields': fields,
                    'communities_connected': len(neighbor_communities),
                })

            if len(bridges) >= top_n:
                break

        return bridges

    def compute_ml_influence_score(self, ml_paper_ids: Set[str]) -> Dict:
        """
        Compute how much ML papers influence each field.

        ML Influence = PageRank contribution from ML papers to field.
        """
        pagerank = self.compute_pagerank()

        # Build citation index
        citing_index = defaultdict(set)
        for src, dst in self.edges:
            citing_index[dst].add(src)

        field_influence = defaultdict(float)
        field_citation_count = defaultdict(int)

        for pid in ml_paper_ids:
            pr = pagerank.get(pid, 0)

            # Find papers citing this ML paper
            citers = citing_index.get(pid, set())
            for citer in citers:
                citer_node = self.nodes.get(citer, {})
                fields = citer_node.get('fields', []) or []

                for field in fields:
                    field_influence[field] += pr
                    field_citation_count[field] += 1

        # Normalize
        max_influence = max(field_influence.values()) if field_influence else 1

        result = {}
        for field in field_influence:
            result[field] = {
                'raw_influence': field_influence[field],
                'normalized_influence': field_influence[field] / max_influence,
                'citation_count': field_citation_count[field],
            }

        return result

    def get_summary_stats(self) -> Dict:
        """Get summary statistics of the graph"""
        communities = self.compute_communities()
        num_communities = len(set(communities.values())) if communities else 0

        in_degrees = defaultdict(int)
        out_degrees = defaultdict(int)
        for src, dst in self.edges:
            out_degrees[src] += 1
            in_degrees[dst] += 1

        return {
            'num_nodes': len(self.nodes),
            'num_edges': len(self.edges),
            'num_communities': num_communities,
            'avg_in_degree': np.mean(list(in_degrees.values())) if in_degrees else 0,
            'avg_out_degree': np.mean(list(out_degrees.values())) if out_degrees else 0,
            'max_in_degree': max(in_degrees.values()) if in_degrees else 0,
            'density': len(self.edges) / (len(self.nodes) ** 2) if self.nodes else 0,
            'gpu_accelerated': GPU_GRAPH,
        }


if __name__ == "__main__":
    # Test with sample data
    nodes = {
        'p1': {'title': 'ML Paper', 'year': 2018, 'fields': ['Computer Science']},
        'p2': {'title': 'Bio Paper', 'year': 2019, 'fields': ['Biology']},
        'p3': {'title': 'Applied Paper', 'year': 2020, 'fields': ['Biology', 'Computer Science']},
        'p4': {'title': 'Another Paper', 'year': 2020, 'fields': ['Chemistry']},
    }
    edges = [('p2', 'p1'), ('p3', 'p1'), ('p3', 'p2'), ('p4', 'p3')]

    analyzer = GraphAnalyzer(nodes, edges)

    print("PageRank:", analyzer.compute_pagerank())
    print("Communities:", analyzer.compute_communities())
    print("Summary:", analyzer.get_summary_stats())
    print("Field Flow:")
    print(analyzer.compute_field_citation_flow())
