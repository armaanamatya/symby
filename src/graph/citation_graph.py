"""
Citation Graph Builder
Fetches citation data from Semantic Scholar API and builds graph structure
"""

import json
import time
import requests
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import pickle

try:
    import pandas as pd
except ImportError:
    import pandas as pd

# Try GPU graph library
try:
    import cugraph
    import cudf
    GPU_GRAPH = True
    print("Using cuGraph for GPU-accelerated graph analysis")
except ImportError:
    GPU_GRAPH = False
    print("cuGraph not available, using NetworkX")

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False


class SemanticScholarClient:
    """Client for Semantic Scholar API"""

    BASE_URL = "https://api.semanticscholar.org/graph/v1"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.session = requests.Session()
        if api_key:
            self.session.headers['x-api-key'] = api_key
        self.request_count = 0
        self.last_request_time = 0

    def _rate_limit(self):
        """Respect rate limits: 100 requests/5 min without key, 1 req/sec"""
        now = time.time()
        if self.api_key:
            # With API key: 10 requests/second
            min_interval = 0.1
        else:
            # Without key: ~1 request per 3 seconds to be safe
            min_interval = 3.0

        elapsed = now - self.last_request_time
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self.last_request_time = time.time()

    def get_paper(self, paper_id: str, fields: List[str] = None) -> Optional[Dict]:
        """Fetch paper details from S2 API"""
        if fields is None:
            fields = ['title', 'year', 'citationCount', 'referenceCount',
                     'fieldsOfStudy', 'citations.paperId', 'references.paperId']

        self._rate_limit()

        url = f"{self.BASE_URL}/paper/CorpusId:{paper_id}"
        params = {'fields': ','.join(fields)}

        try:
            response = self.session.get(url, params=params, timeout=10)
            self.request_count += 1

            if response.status_code == 200:
                return response.json()
            elif response.status_code == 404:
                return None
            elif response.status_code == 429:
                print(f"Rate limited, waiting 60s...")
                time.sleep(60)
                return self.get_paper(paper_id, fields)
            else:
                print(f"Error {response.status_code} for paper {paper_id}")
                return None
        except Exception as e:
            print(f"Exception fetching {paper_id}: {e}")
            return None

    def get_papers_batch(self, paper_ids: List[str], fields: List[str] = None) -> List[Dict]:
        """Fetch multiple papers (POST endpoint)"""
        if fields is None:
            fields = ['title', 'year', 'citationCount', 'referenceCount',
                     'fieldsOfStudy', 'citations.paperId', 'references.paperId']

        self._rate_limit()

        url = f"{self.BASE_URL}/paper/batch"
        params = {'fields': ','.join(fields)}

        # Format IDs
        ids = [f"CorpusId:{pid}" for pid in paper_ids]

        try:
            response = self.session.post(url, params=params, json={'ids': ids}, timeout=30)
            self.request_count += 1

            if response.status_code == 200:
                return response.json()
            else:
                print(f"Batch error {response.status_code}")
                return []
        except Exception as e:
            print(f"Batch exception: {e}")
            return []


class CitationGraphBuilder:
    """Build and analyze citation graph"""

    def __init__(self, cache_dir: str = "graph_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        self.client = SemanticScholarClient()

        # Graph storage
        self.nodes = {}  # paper_id -> {title, year, field, is_ml, citation_count, ...}
        self.edges = []  # [(citing_id, cited_id), ...]
        self.citing_index = defaultdict(set)  # paper -> papers that cite it
        self.reference_index = defaultdict(set)  # paper -> papers it cites

        # Load cache if exists
        self._load_cache()

    def _load_cache(self):
        """Load cached graph data"""
        cache_file = self.cache_dir / "graph_data.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
                    self.nodes = data.get('nodes', {})
                    self.edges = data.get('edges', [])
                    self._rebuild_indices()
                print(f"Loaded cache: {len(self.nodes)} nodes, {len(self.edges)} edges")
            except Exception as e:
                print(f"Cache load error: {e}")

    def _save_cache(self):
        """Save graph data to cache"""
        cache_file = self.cache_dir / "graph_data.pkl"
        with open(cache_file, 'wb') as f:
            pickle.dump({
                'nodes': self.nodes,
                'edges': self.edges
            }, f)

    def _rebuild_indices(self):
        """Rebuild citation indices from edges"""
        self.citing_index.clear()
        self.reference_index.clear()
        for citing, cited in self.edges:
            self.citing_index[cited].add(citing)
            self.reference_index[citing].add(cited)

    def fetch_paper_citations(self, paper_id: str, depth: int = 1) -> Dict:
        """Fetch a paper and its citations/references"""
        if paper_id in self.nodes:
            return self.nodes[paper_id]

        data = self.client.get_paper(paper_id)
        if not data:
            return None

        # Store node
        self.nodes[paper_id] = {
            'id': paper_id,
            'title': data.get('title', ''),
            'year': data.get('year'),
            'citation_count': data.get('citationCount', 0),
            'reference_count': data.get('referenceCount', 0),
            'fields': data.get('fieldsOfStudy', []),
        }

        # Store edges - citations (papers that cite this one)
        citations = data.get('citations', []) or []
        for citing_paper in citations:
            if citing_paper and citing_paper.get('paperId'):
                citing_id = citing_paper['paperId']
                edge = (citing_id, paper_id)
                if edge not in self.edges:
                    self.edges.append(edge)
                    self.citing_index[paper_id].add(citing_id)

        # Store edges - references (papers this one cites)
        references = data.get('references', []) or []
        for ref_paper in references:
            if ref_paper and ref_paper.get('paperId'):
                ref_id = ref_paper['paperId']
                edge = (paper_id, ref_id)
                if edge not in self.edges:
                    self.edges.append(edge)
                    self.reference_index[paper_id].add(ref_id)

        return self.nodes[paper_id]

    def build_graph_from_papers(self, paper_ids: List[str],
                                 max_papers: int = 1000,
                                 save_every: int = 50) -> None:
        """Build citation graph from a list of paper IDs"""
        print(f"Building graph from {len(paper_ids)} papers (max {max_papers})...")

        fetched = 0
        for i, pid in enumerate(paper_ids[:max_papers]):
            if pid in self.nodes:
                continue

            result = self.fetch_paper_citations(str(pid))
            if result:
                fetched += 1

            if fetched % save_every == 0 and fetched > 0:
                self._save_cache()
                print(f"  Fetched {fetched} papers, {len(self.edges)} edges")

            if fetched >= max_papers:
                break

        self._save_cache()
        print(f"Done. Total: {len(self.nodes)} nodes, {len(self.edges)} edges")

    def to_networkx(self):
        """Convert to NetworkX graph"""
        if not NETWORKX_AVAILABLE:
            raise ImportError("NetworkX not available")

        G = nx.DiGraph()

        # Add nodes with attributes
        for pid, attrs in self.nodes.items():
            G.add_node(pid, **attrs)

        # Add edges
        G.add_edges_from(self.edges)

        return G

    def compute_pagerank(self, alpha: float = 0.85) -> Dict[str, float]:
        """Compute PageRank influence scores"""
        if GPU_GRAPH:
            # Use cuGraph
            edge_df = cudf.DataFrame(self.edges, columns=['src', 'dst'])
            G = cugraph.Graph(directed=True)
            G.from_cudf_edgelist(edge_df, source='src', destination='dst')
            pr = cugraph.pagerank(G, alpha=alpha)
            return dict(zip(pr['vertex'].to_pandas(), pr['pagerank'].to_pandas()))
        elif NETWORKX_AVAILABLE:
            G = self.to_networkx()
            return nx.pagerank(G, alpha=alpha)
        else:
            raise ImportError("No graph library available")

    def find_citation_path(self, source_id: str, target_id: str,
                           max_depth: int = 5) -> Optional[List[str]]:
        """Find citation path between two papers (BFS)"""
        if source_id == target_id:
            return [source_id]

        visited = {source_id}
        queue = [(source_id, [source_id])]

        while queue:
            current, path = queue.pop(0)

            if len(path) > max_depth:
                continue

            # Check references (papers this one cites)
            for ref_id in self.reference_index.get(current, []):
                if ref_id == target_id:
                    return path + [ref_id]
                if ref_id not in visited:
                    visited.add(ref_id)
                    queue.append((ref_id, path + [ref_id]))

        return None

    def get_citation_flow(self, source_field: str, target_field: str) -> List[Tuple]:
        """Find papers where source_field cites target_field"""
        flows = []

        for citing_id, cited_id in self.edges:
            citing_node = self.nodes.get(citing_id, {})
            cited_node = self.nodes.get(cited_id, {})

            citing_fields = citing_node.get('fields', []) or []
            cited_fields = cited_node.get('fields', []) or []

            if source_field in citing_fields and target_field in cited_fields:
                flows.append((citing_id, cited_id, citing_node, cited_node))

        return flows

    def compute_field_citation_matrix(self) -> pd.DataFrame:
        """Compute how often each field cites each other field"""
        field_pairs = defaultdict(int)

        for citing_id, cited_id in self.edges:
            citing_fields = self.nodes.get(citing_id, {}).get('fields', []) or []
            cited_fields = self.nodes.get(cited_id, {}).get('fields', []) or []

            for cf in citing_fields:
                for rf in cited_fields:
                    field_pairs[(cf, rf)] += 1

        # Convert to matrix
        all_fields = set()
        for (cf, rf) in field_pairs:
            all_fields.add(cf)
            all_fields.add(rf)

        matrix = pd.DataFrame(0, index=sorted(all_fields), columns=sorted(all_fields))
        for (cf, rf), count in field_pairs.items():
            matrix.loc[cf, rf] = count

        return matrix

    def get_influence_by_year(self, paper_id: str) -> Dict[int, int]:
        """Get citation count by year for a paper"""
        year_counts = defaultdict(int)

        for citing_id in self.citing_index.get(paper_id, []):
            citing_year = self.nodes.get(citing_id, {}).get('year')
            if citing_year:
                year_counts[citing_year] += 1

        return dict(year_counts)

    def get_stats(self) -> Dict:
        """Get graph statistics"""
        if not self.nodes:
            return {'nodes': 0, 'edges': 0}

        years = [n.get('year') for n in self.nodes.values() if n.get('year')]
        citation_counts = [n.get('citation_count', 0) for n in self.nodes.values()]

        return {
            'nodes': len(self.nodes),
            'edges': len(self.edges),
            'year_range': (min(years), max(years)) if years else (None, None),
            'avg_citations': sum(citation_counts) / len(citation_counts) if citation_counts else 0,
            'max_citations': max(citation_counts) if citation_counts else 0,
        }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '/home/abheekp/ai_research/src')
    from data.loader import DataLoader

    # Load some paper IDs
    loader = DataLoader("/home/abheekp/ai_research/s2orc_data")
    df = loader.load_sample(20)
    paper_ids = df['id'].tolist()

    print(f"Sample paper IDs: {paper_ids[:5]}")

    # Build citation graph
    builder = CitationGraphBuilder("/home/abheekp/ai_research/graph_cache")

    # Fetch a few papers to test
    print("\nFetching sample papers...")
    builder.build_graph_from_papers(paper_ids, max_papers=5)

    print(f"\nGraph stats: {builder.get_stats()}")
