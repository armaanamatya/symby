"""
Full Hybrid GraphRAG Interface

Integrates all components for the AI Research Impact Observatory:
- Hybrid Search Engine (dense + sparse + graph retrieval)
- Graph Analytics (PageRank, communities, paths)
- Query parsing and routing
- Objective-specific answer generation

Meets the 4 Hackathon Objectives:
1. Quantify ML Impact
2. Visualize Adoption Dynamics
3. Analyze Quality Trade-offs
4. Trace Discovery Impact
"""

import json
import re
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np

try:
    import pandas as pd
except ImportError:
    import pandas as pd


class HackathonObjective(Enum):
    """The four hackathon objectives."""
    QUANTIFY_IMPACT = "quantify_impact"
    ADOPTION_DYNAMICS = "adoption_dynamics"
    QUALITY_TRADEOFFS = "quality_tradeoffs"
    DISCOVERY_IMPACT = "discovery_impact"


@dataclass
class RAGResponse:
    """Response from the GraphRAG interface."""
    query: str
    query_type: str
    objective: Optional[HackathonObjective]
    answer: str
    evidence: List[Dict]
    visualizations: List[Dict]  # Data for charts/graphs
    metrics: Dict[str, Any]
    confidence: float
    search_results: List[Dict] = field(default_factory=list)


class QueryParser:
    """Parse natural language queries to structured format."""

    OBJECTIVE_PATTERNS = {
        HackathonObjective.QUANTIFY_IMPACT: [
            r'how much (.+) contributes?',
            r'quantif(y|ication).+impact',
            r'measur(e|ing).+impact',
            r'ml attribution',
            r'breakthrough score',
        ],
        HackathonObjective.ADOPTION_DYNAMICS: [
            r'adoption.+(curve|pattern|rate)',
            r'spread of (.+)',
            r'diffusion.+(.+)',
            r'which fields (use|adopt)',
            r's-curve',
            r'temporal.+evolution',
        ],
        HackathonObjective.QUALITY_TRADEOFFS: [
            r'reproducib(le|ility)',
            r'retraction',
            r'code.+available',
            r'quality.+tradeoff',
            r'reliable',
        ],
        HackathonObjective.DISCOVERY_IMPACT: [
            r'path from (.+) to (.+)',
            r'trace.+impact',
            r'journey from',
            r'how did (.+) lead to',
            r'alphafold|covid|breakthrough',
        ],
    }

    QUERY_TYPES = {
        'influence': [r'how did (.+) influence (.+)', r'impact of (.+) on (.+)'],
        'adoption': [r'which fields (use|adopt) (.+)', r'adoption of (.+)'],
        'comparison': [r'compare (.+) (and|vs|with) (.+)'],
        'top': [r'(top|most influential|best) (.+) papers?'],
        'trend': [r'trend of (.+)', r'growth of (.+)'],
        'path': [r'path from (.+) to (.+)', r'connection between'],
        'quantify': [r'quantif(y|ication)', r'measur(e|ing)'],
        'quality': [r'reproducib', r'retraction', r'code available'],
    }

    @classmethod
    def parse(cls, query: str) -> Dict:
        """Parse query into structured format."""
        query_lower = query.lower()

        result = {
            'raw_query': query,
            'objective': None,
            'query_type': 'general',
            'entities': cls._extract_entities(query_lower),
            'filters': cls._extract_filters(query_lower),
        }

        # Detect objective
        for objective, patterns in cls.OBJECTIVE_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    result['objective'] = objective
                    break
            if result['objective']:
                break

        # Detect query type
        for qtype, patterns in cls.QUERY_TYPES.items():
            for pattern in patterns:
                match = re.search(pattern, query_lower)
                if match:
                    result['query_type'] = qtype
                    result['match_groups'] = match.groups()
                    break
            if result['query_type'] != 'general':
                break

        return result

    @staticmethod
    def _extract_entities(query: str) -> Dict[str, List[str]]:
        """Extract ML methods and fields from query."""
        entities = {'methods': [], 'fields': [], 'papers': []}

        # ML methods
        methods = {
            'transformer': ['transformer', 'attention'],
            'bert': ['bert'],
            'gpt': ['gpt', 'chatgpt', 'large language model', 'llm'],
            'alphafold': ['alphafold', 'protein prediction'],
            'diffusion': ['diffusion', 'stable diffusion'],
            'cnn': ['cnn', 'convolutional'],
            'gan': ['gan', 'generative adversarial'],
            'reinforcement': ['reinforcement learning', 'rl', 'q-learning'],
            'gnn': ['graph neural', 'gnn'],
        }

        for method, keywords in methods.items():
            if any(kw in query for kw in keywords):
                entities['methods'].append(method)

        # Fields
        fields = {
            'drug_discovery': ['drug discovery', 'pharmaceutical'],
            'biology': ['biology', 'genomics', 'proteomics'],
            'medicine': ['medicine', 'medical', 'clinical'],
            'physics': ['physics', 'quantum'],
            'chemistry': ['chemistry', 'molecular'],
            'materials': ['materials science'],
            'climate': ['climate', 'weather'],
            'neuroscience': ['neuroscience', 'brain'],
        }

        for field, keywords in fields.items():
            if any(kw in query for kw in keywords):
                entities['fields'].append(field)

        return entities

    @staticmethod
    def _extract_filters(query: str) -> Dict:
        """Extract filters like year ranges."""
        filters = {}

        # Year ranges
        year_match = re.search(r'(\d{4})\s*-\s*(\d{4})', query)
        if year_match:
            filters['year_range'] = (int(year_match.group(1)), int(year_match.group(2)))

        # Single year
        year_match = re.search(r'in (\d{4})', query)
        if year_match:
            year = int(year_match.group(1))
            filters['year_range'] = (year, year)

        return filters


class FullHybridRAG:
    """
    Full Hybrid GraphRAG Interface.

    Combines all components to answer natural language queries
    about ML's impact on scientific research.
    """

    def __init__(
        self,
        search_engine=None,
        graph_analyzer=None,
        semantic_graph=None,
        paper_metadata: Dict[str, Dict] = None,
    ):
        """
        Initialize FullHybridRAG.

        Args:
            search_engine: HybridSearchEngine instance
            graph_analyzer: GraphAnalyzer instance
            semantic_graph: SemanticGraphBuilder instance
            paper_metadata: Dict mapping paper_id to metadata
        """
        self.search_engine = search_engine
        self.graph_analyzer = graph_analyzer
        self.semantic_graph = semantic_graph
        self.paper_metadata = paper_metadata or {}

        self.parser = QueryParser()

    def query(self, query_text: str, k: int = 10) -> RAGResponse:
        """
        Main query interface.

        Args:
            query_text: Natural language query
            k: Number of results to return

        Returns:
            RAGResponse with answer, evidence, and visualizations
        """
        # Parse query
        parsed = self.parser.parse(query_text)

        # Route to appropriate handler based on objective and query type
        if parsed['objective'] == HackathonObjective.QUANTIFY_IMPACT:
            return self._handle_quantify_impact(parsed, k)
        elif parsed['objective'] == HackathonObjective.ADOPTION_DYNAMICS:
            return self._handle_adoption_dynamics(parsed, k)
        elif parsed['objective'] == HackathonObjective.QUALITY_TRADEOFFS:
            return self._handle_quality_tradeoffs(parsed, k)
        elif parsed['objective'] == HackathonObjective.DISCOVERY_IMPACT:
            return self._handle_discovery_impact(parsed, k)
        else:
            # Default hybrid search
            return self._handle_general_query(parsed, k)

    def _handle_quantify_impact(self, parsed: Dict, k: int) -> RAGResponse:
        """Handle queries about quantifying ML impact."""
        entities = parsed['entities']
        methods = entities.get('methods', [])
        fields = entities.get('fields', [])

        answer_parts = []
        evidence = []
        visualizations = []
        metrics = {}

        # Get ML papers and their influence
        if self.graph_analyzer:
            pagerank = self.graph_analyzer.compute_pagerank()

            # Find ML papers
            ml_papers = []
            for pid, meta in self.paper_metadata.items():
                if meta.get('is_ml_paper', False):
                    ml_papers.append(pid)

            if ml_papers:
                # Compute ML influence by field
                ml_influence = self.graph_analyzer.compute_ml_influence_score(set(ml_papers))

                metrics['ml_paper_count'] = len(ml_papers)
                metrics['ml_influence_by_field'] = ml_influence

                # Top ML papers
                top_ml = sorted(
                    [(pid, pagerank.get(pid, 0)) for pid in ml_papers],
                    key=lambda x: -x[1]
                )[:10]

                for pid, score in top_ml:
                    meta = self.paper_metadata.get(pid, {})
                    evidence.append({
                        'paper_id': pid,
                        'title': meta.get('title', ''),
                        'year': meta.get('year'),
                        'pagerank': score,
                        'field': meta.get('source_field', ''),
                    })

                # Visualization data
                visualizations.append({
                    'type': 'bar_chart',
                    'title': 'ML Influence by Field',
                    'data': {
                        field: data.get('normalized_influence', 0)
                        for field, data in ml_influence.items()
                    }
                })

                answer_parts.append(f"Found {len(ml_papers):,} ML papers in the dataset.")
                answer_parts.append(f"Top influenced fields: {', '.join(list(ml_influence.keys())[:5])}")

        answer = ' '.join(answer_parts) if answer_parts else "Unable to quantify ML impact with current data."

        return RAGResponse(
            query=parsed['raw_query'],
            query_type='quantify_impact',
            objective=HackathonObjective.QUANTIFY_IMPACT,
            answer=answer,
            evidence=evidence,
            visualizations=visualizations,
            metrics=metrics,
            confidence=0.8 if evidence else 0.3,
        )

    def _handle_adoption_dynamics(self, parsed: Dict, k: int) -> RAGResponse:
        """Handle queries about adoption patterns and dynamics."""
        entities = parsed['entities']
        methods = entities.get('methods', ['machine learning'])

        answer_parts = []
        evidence = []
        visualizations = []
        metrics = {}

        # Compute adoption over time
        if self.paper_metadata:
            year_adoption = defaultdict(lambda: {'total': 0, 'ml': 0})

            for pid, meta in self.paper_metadata.items():
                year = meta.get('year')
                if year and 2010 <= year <= 2024:
                    year_adoption[year]['total'] += 1
                    if meta.get('is_ml_paper', False):
                        year_adoption[year]['ml'] += 1

            # Compute adoption rates
            adoption_curve = {}
            for year in sorted(year_adoption.keys()):
                data = year_adoption[year]
                if data['total'] > 0:
                    adoption_curve[year] = data['ml'] / data['total']

            metrics['adoption_by_year'] = adoption_curve
            metrics['total_papers'] = sum(d['total'] for d in year_adoption.values())
            metrics['ml_papers'] = sum(d['ml'] for d in year_adoption.values())

            # Visualization
            visualizations.append({
                'type': 'line_chart',
                'title': 'ML Adoption Rate Over Time',
                'x_label': 'Year',
                'y_label': 'Adoption Rate',
                'data': adoption_curve
            })

            # Field-specific adoption
            field_adoption = defaultdict(lambda: {'total': 0, 'ml': 0})
            for pid, meta in self.paper_metadata.items():
                field = meta.get('source_field', 'Unknown')
                field_adoption[field]['total'] += 1
                if meta.get('is_ml_paper', False):
                    field_adoption[field]['ml'] += 1

            field_rates = {
                f: d['ml'] / d['total'] if d['total'] > 0 else 0
                for f, d in field_adoption.items()
            }
            metrics['adoption_by_field'] = field_rates

            visualizations.append({
                'type': 'bar_chart',
                'title': 'ML Adoption by Field',
                'data': dict(sorted(field_rates.items(), key=lambda x: -x[1])[:10])
            })

            answer_parts.append(f"ML adoption has grown from {adoption_curve.get(2016, 0):.1%} (2016) to {adoption_curve.get(2022, 0):.1%} (2022).")
            top_fields = sorted(field_rates.items(), key=lambda x: -x[1])[:3]
            answer_parts.append(f"Highest adoption fields: {', '.join(f[0] for f in top_fields)}")

        answer = ' '.join(answer_parts) if answer_parts else "Unable to analyze adoption dynamics."

        return RAGResponse(
            query=parsed['raw_query'],
            query_type='adoption_dynamics',
            objective=HackathonObjective.ADOPTION_DYNAMICS,
            answer=answer,
            evidence=evidence,
            visualizations=visualizations,
            metrics=metrics,
            confidence=0.85 if visualizations else 0.3,
        )

    def _handle_quality_tradeoffs(self, parsed: Dict, k: int) -> RAGResponse:
        """Handle queries about reproducibility and quality trade-offs."""
        answer_parts = []
        evidence = []
        visualizations = []
        metrics = {}

        if self.paper_metadata:
            # Analyze reproducibility indicators
            repro_stats = {
                'ml_with_code': 0,
                'ml_without_code': 0,
                'nonml_with_code': 0,
                'nonml_without_code': 0,
            }

            for pid, meta in self.paper_metadata.items():
                is_ml = meta.get('is_ml_paper', False)
                has_code = meta.get('has_code', False)

                if is_ml:
                    if has_code:
                        repro_stats['ml_with_code'] += 1
                    else:
                        repro_stats['ml_without_code'] += 1
                else:
                    if has_code:
                        repro_stats['nonml_with_code'] += 1
                    else:
                        repro_stats['nonml_without_code'] += 1

            # Compute rates
            ml_total = repro_stats['ml_with_code'] + repro_stats['ml_without_code']
            nonml_total = repro_stats['nonml_with_code'] + repro_stats['nonml_without_code']

            ml_code_rate = repro_stats['ml_with_code'] / ml_total if ml_total > 0 else 0
            nonml_code_rate = repro_stats['nonml_with_code'] / nonml_total if nonml_total > 0 else 0

            metrics['ml_code_availability'] = ml_code_rate
            metrics['nonml_code_availability'] = nonml_code_rate
            metrics['repro_stats'] = repro_stats

            visualizations.append({
                'type': 'bar_chart',
                'title': 'Code Availability: ML vs Non-ML Papers',
                'data': {
                    'ML Papers': ml_code_rate,
                    'Non-ML Papers': nonml_code_rate,
                }
            })

            answer_parts.append(f"ML papers have {ml_code_rate:.1%} code availability vs {nonml_code_rate:.1%} for non-ML papers.")

            if ml_code_rate > nonml_code_rate:
                answer_parts.append("ML research shows BETTER reproducibility practices.")
            else:
                answer_parts.append("ML research shows room for improvement in reproducibility.")

        answer = ' '.join(answer_parts) if answer_parts else "Unable to analyze quality trade-offs."

        return RAGResponse(
            query=parsed['raw_query'],
            query_type='quality_tradeoffs',
            objective=HackathonObjective.QUALITY_TRADEOFFS,
            answer=answer,
            evidence=evidence,
            visualizations=visualizations,
            metrics=metrics,
            confidence=0.8 if metrics else 0.3,
        )

    def _handle_discovery_impact(self, parsed: Dict, k: int) -> RAGResponse:
        """Handle queries about tracing discovery impact paths."""
        entities = parsed['entities']
        methods = entities.get('methods', [])
        fields = entities.get('fields', [])

        answer_parts = []
        evidence = []
        visualizations = []
        metrics = {}

        # If we have specific methods/fields, trace paths
        if self.graph_analyzer and self.semantic_graph:
            if methods and fields:
                # Find path from method to field
                method = methods[0]
                field = fields[0]

                # Get relevant papers
                method_papers = self._find_papers_by_keyword(method)[:10]
                field_papers = self._find_papers_by_field(field)[:10]

                # Find paths
                paths_found = []
                for mp in method_papers:
                    for fp in field_papers:
                        path = self.semantic_graph.find_path(mp, fp)
                        if path and len(path) <= 5:
                            paths_found.append(path)

                if paths_found:
                    # Show shortest path
                    shortest = min(paths_found, key=len)
                    for pid in shortest:
                        meta = self.paper_metadata.get(pid, {})
                        evidence.append({
                            'paper_id': pid,
                            'title': meta.get('title', ''),
                            'year': meta.get('year'),
                            'field': meta.get('source_field', ''),
                        })

                    answer_parts.append(f"Found {len(paths_found)} paths from {method} to {field}.")
                    answer_parts.append(f"Shortest path has {len(shortest)} papers.")

                    # Path visualization
                    visualizations.append({
                        'type': 'path',
                        'title': f'Discovery Path: {method} → {field}',
                        'nodes': [
                            {'id': pid, 'label': self.paper_metadata.get(pid, {}).get('title', '')[:50]}
                            for pid in shortest
                        ]
                    })

                    metrics['paths_found'] = len(paths_found)
                    metrics['shortest_path_length'] = len(shortest)

        answer = ' '.join(answer_parts) if answer_parts else "Unable to trace discovery impact path."

        return RAGResponse(
            query=parsed['raw_query'],
            query_type='discovery_impact',
            objective=HackathonObjective.DISCOVERY_IMPACT,
            answer=answer,
            evidence=evidence,
            visualizations=visualizations,
            metrics=metrics,
            confidence=0.75 if evidence else 0.3,
        )

    def _handle_general_query(self, parsed: Dict, k: int) -> RAGResponse:
        """Handle general queries with hybrid search."""
        search_results = []

        if self.search_engine:
            results = self.search_engine.search(
                parsed['raw_query'],
                k=k,
                filter_years=parsed['filters'].get('year_range'),
            )

            search_results = [
                {
                    'paper_id': r.paper_id,
                    'title': r.title,
                    'year': r.year,
                    'score': r.score,
                    'source': r.source,
                }
                for r in results
            ]

        answer = f"Found {len(search_results)} relevant papers."
        if search_results:
            answer += f" Top result: {search_results[0]['title'][:100]}"

        return RAGResponse(
            query=parsed['raw_query'],
            query_type='general',
            objective=None,
            answer=answer,
            evidence=search_results[:5],
            visualizations=[],
            metrics={'result_count': len(search_results)},
            confidence=0.7 if search_results else 0.3,
            search_results=search_results,
        )

    def _find_papers_by_keyword(self, keyword: str) -> List[str]:
        """Find papers containing keyword in title."""
        matching = []
        keyword_lower = keyword.lower()
        for pid, meta in self.paper_metadata.items():
            title = meta.get('title', '').lower()
            if keyword_lower in title:
                matching.append(pid)
        return matching

    def _find_papers_by_field(self, field: str) -> List[str]:
        """Find papers in a specific field."""
        matching = []
        field_lower = field.lower()
        for pid, meta in self.paper_metadata.items():
            paper_field = meta.get('source_field', '').lower()
            if field_lower in paper_field:
                matching.append(pid)
        return matching

    def get_suggested_queries(self) -> List[Dict[str, str]]:
        """Get example queries for each objective."""
        return [
            {
                'objective': 'Quantify ML Impact',
                'query': 'How much has machine learning contributed to drug discovery breakthroughs?',
            },
            {
                'objective': 'Adoption Dynamics',
                'query': 'Show the adoption curve of transformers across scientific fields',
            },
            {
                'objective': 'Quality Trade-offs',
                'query': 'Is ML-enabled research more or less reproducible than traditional methods?',
            },
            {
                'objective': 'Discovery Impact',
                'query': 'Trace the path from AlphaFold to clinical applications',
            },
            {
                'objective': 'General',
                'query': 'Top papers on graph neural networks in materials science',
            },
        ]

    def get_dashboard_data(self) -> Dict:
        """Get aggregated data for dashboard visualizations."""
        data = {
            'overview': {},
            'adoption': {},
            'influence': {},
            'quality': {},
        }

        if self.paper_metadata:
            total = len(self.paper_metadata)
            ml_count = sum(1 for m in self.paper_metadata.values() if m.get('is_ml_paper'))

            data['overview'] = {
                'total_papers': total,
                'ml_papers': ml_count,
                'ml_percentage': ml_count / total if total > 0 else 0,
            }

            # Year distribution
            year_counts = defaultdict(int)
            for meta in self.paper_metadata.values():
                year = meta.get('year')
                if year and 2010 <= year <= 2024:
                    year_counts[year] += 1

            data['overview']['papers_by_year'] = dict(sorted(year_counts.items()))

            # Field distribution
            field_counts = defaultdict(int)
            for meta in self.paper_metadata.values():
                field = meta.get('source_field', 'Unknown')
                field_counts[field] += 1

            data['overview']['papers_by_field'] = dict(sorted(field_counts.items(), key=lambda x: -x[1]))

        return data


if __name__ == "__main__":
    print("\n=== FullHybridRAG Test ===\n")

    # Create with mock data
    paper_metadata = {
        f'p{i}': {
            'title': f'Paper {i}',
            'year': 2018 + (i % 7),
            'source_field': ['Biology', 'Physics', 'Chemistry', 'CS'][i % 4],
            'is_ml_paper': i % 3 == 0,
            'has_code': i % 2 == 0,
        }
        for i in range(100)
    }

    rag = FullHybridRAG(paper_metadata=paper_metadata)

    # Test queries
    queries = [
        "How much has ML contributed to biology?",
        "Show adoption dynamics of deep learning",
        "Is ML research reproducible?",
        "Trace path from transformer to drug discovery",
        "Top papers in machine learning",
    ]

    for query in queries:
        print(f"\nQuery: {query}")
        result = rag.query(query)
        print(f"Type: {result.query_type}")
        print(f"Objective: {result.objective}")
        print(f"Answer: {result.answer[:200]}...")
        print(f"Confidence: {result.confidence}")
        print(f"Visualizations: {len(result.visualizations)}")
