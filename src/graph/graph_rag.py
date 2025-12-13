"""
GraphRAG Query Engine
Natural language queries over citation knowledge graph

Enables queries like:
- "How did transformers influence drug discovery?"
- "Which fields adopted AlphaFold fastest?"
- "What's the path from GPT to climate science?"
"""

import json
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

try:
    import pandas as pd
except ImportError:
    import pandas as pd


@dataclass
class QueryResult:
    """Result from a GraphRAG query"""
    query: str
    query_type: str
    answer: str
    evidence: List[Dict]
    confidence: float
    graph_stats: Dict


class GraphRAGEngine:
    """
    Graph-based Retrieval Augmented Generation Engine.

    Combines graph traversal with LLM for natural language queries
    about scientific influence and ML adoption.
    """

    # Query patterns for classification
    QUERY_PATTERNS = {
        'influence': [
            r'how did (.+) influence (.+)',
            r'impact of (.+) on (.+)',
            r'(.+) affect (.+)',
        ],
        'adoption': [
            r'which fields adopted (.+)',
            r'who uses (.+)',
            r'adoption of (.+)',
        ],
        'path': [
            r'path from (.+) to (.+)',
            r'connection between (.+) and (.+)',
            r'how is (.+) connected to (.+)',
        ],
        'comparison': [
            r'compare (.+) and (.+)',
            r'(.+) vs (.+)',
            r'difference between (.+) and (.+)',
        ],
        'top': [
            r'top (.+) papers',
            r'most influential (.+)',
            r'best (.+)',
            r'leading (.+)',
        ],
        'trend': [
            r'trend of (.+)',
            r'how has (.+) changed',
            r'growth of (.+)',
        ],
    }

    # Known ML methods/techniques for entity recognition
    ML_ENTITIES = {
        'transformer': ['transformer', 'attention mechanism', 'self-attention'],
        'bert': ['bert', 'bidirectional encoder'],
        'gpt': ['gpt', 'generative pre-trained', 'gpt-3', 'gpt-4', 'chatgpt'],
        'alphafold': ['alphafold', 'protein structure prediction'],
        'diffusion': ['diffusion model', 'stable diffusion', 'ddpm'],
        'cnn': ['cnn', 'convolutional neural network', 'resnet', 'convnet'],
        'gan': ['gan', 'generative adversarial'],
        'reinforcement_learning': ['reinforcement learning', 'rl', 'q-learning'],
        'graph_neural': ['graph neural network', 'gnn'],
    }

    # Known fields
    FIELD_ENTITIES = {
        'drug_discovery': ['drug discovery', 'pharmaceutical', 'drug design'],
        'biology': ['biology', 'biological', 'genomics', 'proteomics'],
        'medicine': ['medicine', 'medical', 'clinical', 'healthcare'],
        'physics': ['physics', 'quantum', 'particle'],
        'chemistry': ['chemistry', 'chemical', 'molecular'],
        'materials': ['materials science', 'materials'],
        'climate': ['climate', 'environmental', 'weather'],
        'neuroscience': ['neuroscience', 'brain', 'cognitive'],
        'computer_science': ['computer science', 'cs', 'computing'],
    }

    def __init__(self, graph_analyzer, nodes: Dict, llm_client=None):
        """
        Initialize GraphRAG engine.

        Args:
            graph_analyzer: GraphAnalyzer instance
            nodes: Dict of paper nodes with metadata
            llm_client: Optional LLM client for answer generation
        """
        self.analyzer = graph_analyzer
        self.nodes = nodes
        self.llm = llm_client

        # Build indices
        self._build_indices()

    def _build_indices(self):
        """Build search indices for efficient retrieval"""
        self.field_index = defaultdict(list)
        self.year_index = defaultdict(list)
        self.title_index = {}

        for pid, node in self.nodes.items():
            # Field index
            fields = node.get('fields', []) or []
            for field in fields:
                self.field_index[field.lower()].append(pid)

            # Year index
            year = node.get('year')
            if year:
                self.year_index[year].append(pid)

            # Title index (for keyword search)
            title = node.get('title', '').lower()
            self.title_index[pid] = title

    def _classify_query(self, query: str) -> Tuple[str, Dict]:
        """Classify query type and extract entities"""
        query_lower = query.lower()

        for query_type, patterns in self.QUERY_PATTERNS.items():
            for pattern in patterns:
                match = re.search(pattern, query_lower)
                if match:
                    return query_type, {'groups': match.groups()}

        return 'general', {}

    def _extract_entities(self, query: str) -> Dict[str, List[str]]:
        """Extract ML methods and fields from query"""
        query_lower = query.lower()
        entities = {'methods': [], 'fields': []}

        for method, keywords in self.ML_ENTITIES.items():
            for kw in keywords:
                if kw in query_lower:
                    entities['methods'].append(method)
                    break

        for field, keywords in self.FIELD_ENTITIES.items():
            for kw in keywords:
                if kw in query_lower:
                    entities['fields'].append(field)
                    break

        return entities

    def _find_papers_by_method(self, method: str) -> List[str]:
        """Find papers related to an ML method"""
        keywords = self.ML_ENTITIES.get(method, [method])
        matching_papers = []

        for pid, title in self.title_index.items():
            for kw in keywords:
                if kw in title:
                    matching_papers.append(pid)
                    break

        return matching_papers

    def _find_papers_by_field(self, field: str) -> List[str]:
        """Find papers in a specific field"""
        # Try exact match first
        for stored_field, papers in self.field_index.items():
            if field.lower() in stored_field.lower():
                return papers

        # Try keyword match
        field_keywords = self.FIELD_ENTITIES.get(field, [field])
        matching = []
        for stored_field, papers in self.field_index.items():
            for kw in field_keywords:
                if kw in stored_field.lower():
                    matching.extend(papers)

        return list(set(matching))

    def query_influence(self, source: str, target: str) -> QueryResult:
        """
        Answer: How did [source] influence [target]?

        Uses citation flow analysis.
        """
        # Find source papers (e.g., transformer papers)
        source_papers = self._find_papers_by_method(source)
        if not source_papers:
            source_papers = self._find_papers_by_field(source)

        # Find target papers (e.g., drug discovery papers)
        target_papers = self._find_papers_by_field(target)
        if not target_papers:
            target_papers = self._find_papers_by_method(target)

        # Compute citation flow
        flow_count = 0
        evidence = []

        # Check direct citations
        for src_pid in source_papers:
            influence = self.analyzer.trace_influence_chain(src_pid, max_hops=2)
            for hop in influence.get('hops', []):
                hop_papers = set(hop.get('papers', []))
                intersection = hop_papers & set(target_papers)
                flow_count += len(intersection)

                for pid in list(intersection)[:3]:
                    node = self.nodes.get(pid, {})
                    evidence.append({
                        'paper_id': pid,
                        'title': node.get('title', '')[:100],
                        'year': node.get('year'),
                        'hop': hop['hop'],
                    })

        # Generate answer
        if flow_count > 0:
            answer = f"{source.title()} has influenced {target} through {flow_count} citation connections. "
            answer += f"Found {len(evidence)} key papers bridging these areas."
            confidence = min(0.9, 0.5 + (flow_count / 100))
        else:
            answer = f"Limited direct citation evidence found between {source} and {target} in the current dataset."
            confidence = 0.3

        return QueryResult(
            query=f"How did {source} influence {target}?",
            query_type='influence',
            answer=answer,
            evidence=evidence[:10],
            confidence=confidence,
            graph_stats={'source_papers': len(source_papers), 'target_papers': len(target_papers)}
        )

    def query_adoption(self, method: str) -> QueryResult:
        """
        Answer: Which fields adopted [method]?

        Uses field citation flow analysis.
        """
        method_papers = self._find_papers_by_method(method)

        if not method_papers:
            return QueryResult(
                query=f"Which fields adopted {method}?",
                query_type='adoption',
                answer=f"No papers found mentioning {method}.",
                evidence=[],
                confidence=0.2,
                graph_stats={}
            )

        # Track which fields cite these method papers
        field_adoption = defaultdict(lambda: {'count': 0, 'years': [], 'papers': []})

        # Get citing papers for each method paper
        pagerank = self.analyzer.compute_pagerank()

        for method_pid in method_papers:
            influence = self.analyzer.trace_influence_chain(method_pid, max_hops=2)

            for hop in influence.get('hops', []):
                for pid in hop.get('papers', []):
                    node = self.nodes.get(pid, {})
                    fields = node.get('fields', []) or []
                    year = node.get('year')

                    for field in fields:
                        field_adoption[field]['count'] += 1
                        if year:
                            field_adoption[field]['years'].append(year)
                        field_adoption[field]['papers'].append(pid)

        # Sort by adoption count
        sorted_fields = sorted(field_adoption.items(), key=lambda x: x[1]['count'], reverse=True)

        evidence = []
        for field, data in sorted_fields[:10]:
            first_year = min(data['years']) if data['years'] else None
            evidence.append({
                'field': field,
                'adoption_count': data['count'],
                'first_adoption_year': first_year,
                'sample_papers': data['papers'][:3]
            })

        if sorted_fields:
            top_fields = [f[0] for f in sorted_fields[:5]]
            answer = f"{method.title()} has been adopted by {len(sorted_fields)} fields. "
            answer += f"Top adopters: {', '.join(top_fields)}."
            confidence = 0.8
        else:
            answer = f"No adoption data found for {method}."
            confidence = 0.3

        return QueryResult(
            query=f"Which fields adopted {method}?",
            query_type='adoption',
            answer=answer,
            evidence=evidence,
            confidence=confidence,
            graph_stats={'method_papers': len(method_papers), 'fields_found': len(sorted_fields)}
        )

    def query_path(self, source: str, target: str) -> QueryResult:
        """
        Answer: What's the path from [source] to [target]?

        Uses shortest path algorithm.
        """
        source_papers = self._find_papers_by_method(source)
        if not source_papers:
            source_papers = self._find_papers_by_field(source)

        target_papers = self._find_papers_by_field(target)
        if not target_papers:
            target_papers = self._find_papers_by_method(target)

        # Find shortest path between any source and target paper
        shortest_path = None
        for src_pid in source_papers[:20]:  # Limit search
            for tgt_pid in target_papers[:20]:
                path = self.analyzer.find_shortest_path(src_pid, tgt_pid)
                if path:
                    if shortest_path is None or len(path) < len(shortest_path):
                        shortest_path = path

        evidence = []
        if shortest_path:
            for pid in shortest_path:
                node = self.nodes.get(pid, {})
                evidence.append({
                    'paper_id': pid,
                    'title': node.get('title', '')[:100],
                    'year': node.get('year'),
                    'fields': node.get('fields', []),
                })

            answer = f"Found a path of {len(shortest_path)} papers connecting {source} to {target}."
            confidence = 0.85
        else:
            answer = f"No direct citation path found between {source} and {target}."
            confidence = 0.4

        return QueryResult(
            query=f"Path from {source} to {target}",
            query_type='path',
            answer=answer,
            evidence=evidence,
            confidence=confidence,
            graph_stats={'path_length': len(shortest_path) if shortest_path else 0}
        )

    def query_top_papers(self, topic: str, n: int = 10) -> QueryResult:
        """
        Answer: What are the top papers in [topic]?

        Uses PageRank for influence scoring.
        """
        topic_papers = self._find_papers_by_method(topic)
        if not topic_papers:
            topic_papers = self._find_papers_by_field(topic)

        if not topic_papers:
            return QueryResult(
                query=f"Top papers in {topic}",
                query_type='top',
                answer=f"No papers found for {topic}.",
                evidence=[],
                confidence=0.2,
                graph_stats={}
            )

        pagerank = self.analyzer.compute_pagerank()

        # Score and sort
        scored_papers = [(pid, pagerank.get(pid, 0)) for pid in topic_papers]
        scored_papers.sort(key=lambda x: x[1], reverse=True)

        evidence = []
        for pid, score in scored_papers[:n]:
            node = self.nodes.get(pid, {})
            evidence.append({
                'paper_id': pid,
                'title': node.get('title', '')[:100],
                'year': node.get('year'),
                'pagerank_score': round(score, 6),
                'citation_count': node.get('citation_count', 0),
            })

        answer = f"Found {len(topic_papers)} papers related to {topic}. Top {min(n, len(evidence))} by influence:"
        confidence = 0.8

        return QueryResult(
            query=f"Top papers in {topic}",
            query_type='top',
            answer=answer,
            evidence=evidence,
            confidence=confidence,
            graph_stats={'total_papers': len(topic_papers)}
        )

    def query(self, query_text: str) -> QueryResult:
        """
        Main query interface. Classifies and routes to appropriate handler.
        """
        query_type, match_info = self._classify_query(query_text)
        entities = self._extract_entities(query_text)

        groups = match_info.get('groups', ())

        if query_type == 'influence' and len(groups) >= 2:
            return self.query_influence(groups[0], groups[1])

        elif query_type == 'adoption' and len(groups) >= 1:
            return self.query_adoption(groups[0])

        elif query_type == 'path' and len(groups) >= 2:
            return self.query_path(groups[0], groups[1])

        elif query_type == 'top' and len(groups) >= 1:
            return self.query_top_papers(groups[0])

        elif entities['methods']:
            # Default: adoption query for first method
            return self.query_adoption(entities['methods'][0])

        elif entities['fields']:
            # Default: top papers in field
            return self.query_top_papers(entities['fields'][0])

        else:
            return QueryResult(
                query=query_text,
                query_type='unknown',
                answer="I couldn't understand that query. Try asking about influence, adoption, or top papers.",
                evidence=[],
                confidence=0.1,
                graph_stats={}
            )

    def get_suggested_queries(self) -> List[str]:
        """Get example queries that work well with current data"""
        return [
            "How did transformers influence drug discovery?",
            "Which fields adopted AlphaFold fastest?",
            "What's the path from deep learning to climate science?",
            "Top papers in machine learning",
            "Compare transformer and CNN adoption",
            "Impact of GPT on biology",
        ]


if __name__ == "__main__":
    # Test with mock data
    from graph_analysis import GraphAnalyzer

    nodes = {
        'ml1': {'title': 'Attention Is All You Need', 'year': 2017, 'fields': ['Computer Science']},
        'ml2': {'title': 'BERT: Pre-training', 'year': 2018, 'fields': ['Computer Science']},
        'bio1': {'title': 'Drug Discovery with Deep Learning', 'year': 2020, 'fields': ['Biology', 'Medicine']},
        'bio2': {'title': 'Protein Prediction', 'year': 2021, 'fields': ['Biology']},
    }
    edges = [('bio1', 'ml1'), ('bio2', 'ml1'), ('bio2', 'ml2'), ('bio1', 'ml2')]

    analyzer = GraphAnalyzer(nodes, edges)
    rag = GraphRAGEngine(analyzer, nodes)

    print("\n=== GraphRAG Test ===")

    queries = [
        "How did transformer influence biology?",
        "Which fields adopted bert?",
        "Top papers in drug discovery",
    ]

    for q in queries:
        print(f"\nQuery: {q}")
        result = rag.query(q)
        print(f"Answer: {result.answer}")
        print(f"Confidence: {result.confidence}")
        print(f"Evidence count: {len(result.evidence)}")
