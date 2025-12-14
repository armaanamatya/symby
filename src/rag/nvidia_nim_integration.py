"""
NVIDIA NIM Integration for Scientific Impact Analysis
Integrates with existing BGE-M3 + FAISS hybrid search pipeline
"""

import os
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import json
from openai import OpenAI
import numpy as np


@dataclass
class AttributionScore:
    """ML attribution analysis result."""
    paper_id: str
    attribution_score: float  # 0-1, how much credit goes to ML
    ml_methods: List[str]
    domain_contributions: List[str]
    evidence: List[str]
    confidence: float


@dataclass
class AccelerationMetric:
    """Discovery acceleration analysis."""
    paper_id: str
    baseline_timeline: str  # e.g., "5-10 years without ML"
    actual_timeline: str   # e.g., "6 months with ML"
    speedup_factor: float  # e.g., 10x
    evidence: List[str]
    breakthrough_type: str  # "incremental", "transformational", "revolutionary"


class NVIDIANIMClient:
    """
    NVIDIA NIM client for scientific impact analysis.

    Supports multiple models:
    - Llama 3.1 70B: Deep reasoning, attribution analysis
    - Mixtral 8x22B: Fast classification, high throughput
    - Nemotron 70B: Structured outputs, JSON responses
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://integrate.api.nvidia.com/v1",
        default_model: str = "meta/llama-3.1-70b-instruct"
    ):
        """
        Initialize NVIDIA NIM client.

        Args:
            api_key: NVIDIA API key (or set NVIDIA_API_KEY env var)
            base_url: NIM API endpoint
            default_model: Default model to use
        """
        self.api_key = api_key or os.environ.get('NVIDIA_API_KEY')
        if not self.api_key:
            raise ValueError(
                "NVIDIA API key required. Set NVIDIA_API_KEY env var or pass api_key parameter.\n"
                "Get API key from: https://build.nvidia.com/explore/discover"
            )

        self.client = OpenAI(
            base_url=base_url,
            api_key=self.api_key
        )
        self.default_model = default_model

    def analyze_attribution(
        self,
        paper: Dict,
        cited_papers: List[Dict],
        model: Optional[str] = None
    ) -> AttributionScore:
        """
        Analyze ML attribution in a scientific paper.

        Determines what percentage of the breakthrough comes from:
        - ML techniques/methods
        - Domain-specific insights
        - Combination of both

        Args:
            paper: Paper metadata with 'title', 'abstract', 'text' fields
            cited_papers: List of cited papers for context
            model: Model to use (defaults to Llama 70B)

        Returns:
            AttributionScore with ML vs domain credit breakdown
        """
        model = model or self.default_model

        # Build context
        paper_text = f"""
Title: {paper.get('title', '')}
Year: {paper.get('year', '')}
Abstract: {paper.get('abstract', '')}
Full Text Excerpt: {paper.get('text', '')[:2000]}
"""

        citations_text = "\n".join([
            f"- {cite.get('title', '')} ({cite.get('year', '')})"
            for cite in cited_papers[:10]
        ])

        # Attribution analysis prompt
        system_prompt = """You are an expert in analyzing machine learning's role in scientific breakthroughs.

Your task is to quantify the attribution between ML methods and domain-specific insights.

Output a JSON object with:
{
  "attribution_score": 0.0-1.0,  // 0 = pure domain science, 1 = pure ML innovation
  "ml_methods": ["method1", "method2"],  // ML techniques used
  "domain_contributions": ["insight1", "insight2"],  // Domain-specific contributions
  "evidence": ["quote1", "quote2"],  // Supporting evidence from paper
  "confidence": 0.0-1.0,  // Your confidence in this assessment
  "reasoning": "explanation of your analysis"
}

Attribution score guidelines:
- 0.0-0.2: ML is minor tool, breakthrough is domain insight
- 0.3-0.5: ML and domain equally important
- 0.6-0.8: ML is key enabler, domain provides context
- 0.9-1.0: Breakthrough is primarily ML innovation"""

        user_prompt = f"""Analyze this paper's ML attribution:

{paper_text}

Key citations:
{citations_text}

Provide your attribution analysis in JSON format."""

        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,
                max_tokens=1500,
                response_format={"type": "json_object"}  # Structured output
            )

            result = json.loads(response.choices[0].message.content)

            return AttributionScore(
                paper_id=paper.get('paper_id', ''),
                attribution_score=result.get('attribution_score', 0.5),
                ml_methods=result.get('ml_methods', []),
                domain_contributions=result.get('domain_contributions', []),
                evidence=result.get('evidence', []),
                confidence=result.get('confidence', 0.0)
            )

        except Exception as e:
            print(f"[NIM] Attribution analysis failed: {e}")
            return AttributionScore(
                paper_id=paper.get('paper_id', ''),
                attribution_score=0.5,
                ml_methods=[],
                domain_contributions=[],
                evidence=[],
                confidence=0.0
            )

    def analyze_acceleration(
        self,
        paper: Dict,
        baseline_papers: List[Dict],
        model: Optional[str] = None
    ) -> AccelerationMetric:
        """
        Analyze discovery acceleration from ML adoption.

        Compares:
        - Historical timeline without ML
        - Actual timeline with ML
        - Speedup factor

        Args:
            paper: Target paper
            baseline_papers: Historical papers on same problem (pre-ML)
            model: Model to use

        Returns:
            AccelerationMetric with speedup analysis
        """
        model = model or self.default_model

        system_prompt = """You are an expert in analyzing how machine learning accelerates scientific discovery.

Given a paper and historical baseline, estimate:
1. How long this discovery would have taken without ML
2. How long it actually took with ML
3. The speedup factor
4. Whether this is incremental, transformational, or revolutionary

Output JSON:
{
  "baseline_timeline": "estimated time without ML (e.g., '5-10 years')",
  "actual_timeline": "actual time with ML (e.g., '6 months')",
  "speedup_factor": 10.0,  // numeric speedup
  "breakthrough_type": "transformational",  // incremental/transformational/revolutionary
  "evidence": ["supporting quote 1", "quote 2"],
  "confidence": 0.8
}"""

        paper_text = f"""
Paper: {paper.get('title', '')}
Year: {paper.get('year', '')}
Abstract: {paper.get('abstract', '')}
"""

        baseline_text = "\n".join([
            f"Historical paper ({bp.get('year', '')}): {bp.get('title', '')}"
            for bp in baseline_papers[:5]
        ])

        user_prompt = f"""Analyze acceleration:

Target paper:
{paper_text}

Historical baseline (pre-ML approaches):
{baseline_text}

Estimate the speedup from ML adoption."""

        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,
                max_tokens=1000,
                response_format={"type": "json_object"}
            )

            result = json.loads(response.choices[0].message.content)

            return AccelerationMetric(
                paper_id=paper.get('paper_id', ''),
                baseline_timeline=result.get('baseline_timeline', 'unknown'),
                actual_timeline=result.get('actual_timeline', 'unknown'),
                speedup_factor=result.get('speedup_factor', 1.0),
                evidence=result.get('evidence', []),
                breakthrough_type=result.get('breakthrough_type', 'incremental')
            )

        except Exception as e:
            print(f"[NIM] Acceleration analysis failed: {e}")
            return AccelerationMetric(
                paper_id=paper.get('paper_id', ''),
                baseline_timeline='unknown',
                actual_timeline='unknown',
                speedup_factor=1.0,
                evidence=[],
                breakthrough_type='incremental'
            )

    def classify_ml_adoption(
        self,
        papers: List[Dict],
        model: str = "mistralai/mixtral-8x22b-instruct-v0.1"
    ) -> List[Dict]:
        """
        Fast classification of ML adoption level (0-5).

        Uses Mixtral for high-throughput classification.

        Args:
            papers: List of papers to classify
            model: Fast model for classification (defaults to Mixtral)

        Returns:
            List of dicts with paper_id and adoption_level
        """
        results = []

        for paper in papers:
            prompt = f"""Classify ML adoption level (0-5):
0 = No ML
1 = Mentions ML in related work
2 = Uses existing ML tool
3 = Adapts ML for domain
4 = Novel ML method for domain
5 = Fundamental ML innovation

Paper: {paper.get('title', '')}
Abstract: {paper.get('abstract', '')[:500]}

Output only the number (0-5)."""

            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=10
                )

                level = int(response.choices[0].message.content.strip()[0])
                results.append({
                    'paper_id': paper.get('paper_id', ''),
                    'adoption_level': level
                })

            except Exception as e:
                print(f"[NIM] Classification failed for {paper.get('paper_id', '')}: {e}")
                results.append({
                    'paper_id': paper.get('paper_id', ''),
                    'adoption_level': 0
                })

        return results

    def answer_query_with_rag(
        self,
        query: str,
        retrieved_papers: List[Dict],
        model: Optional[str] = None
    ) -> str:
        """
        Answer user query with RAG (Retrieval-Augmented Generation).

        Args:
            query: User question
            retrieved_papers: Papers from vector search + graph expansion
            model: Model to use

        Returns:
            Natural language answer with citations
        """
        model = model or self.default_model

        # Build context
        context = "\n\n".join([
            f"""Paper ID: {p.get('paper_id', '')}
Title: {p.get('title', '')}
Year: {p.get('year', '')}
Field: {p.get('source_field', '')}
Abstract: {p.get('abstract', '')}
Citations: {p.get('citation_count', 0)}"""
            for p in retrieved_papers[:10]
        ])

        system_prompt = """You are an expert research assistant analyzing ML's impact on science.

Answer user questions based ONLY on the provided context papers.
Always cite papers using [Paper ID: xxx] format.
Provide quantitative evidence when available (citation counts, years, etc.).
If you cannot answer from the context, say so."""

        user_prompt = f"""Context papers:
{context}

Question: {query}

Provide a detailed answer with citations."""

        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=2000
            )

            return response.choices[0].message.content

        except Exception as e:
            return f"Error generating answer: {e}"


class EnhancedRAGPipeline:
    """
    Enhanced RAG pipeline integrating:
    - Existing hybrid search (BGE-M3 + FAISS)
    - cuGraph citation network
    - NVIDIA NIM for reasoning
    """

    def __init__(
        self,
        hybrid_search_engine,  # Your existing HybridSearchEngine
        citation_graph=None,  # cuGraph or NetworkX graph
        nvidia_api_key: Optional[str] = None
    ):
        """
        Initialize enhanced RAG pipeline.

        Args:
            hybrid_search_engine: Your existing HybridSearchEngine instance
            citation_graph: Citation network (cuGraph or NetworkX)
            nvidia_api_key: NVIDIA API key
        """
        self.search_engine = hybrid_search_engine
        self.citation_graph = citation_graph
        self.nim_client = NVIDIANIMClient(api_key=nvidia_api_key)

    def query(
        self,
        query: str,
        k: int = 10,
        include_citation_context: bool = True,
        analyze_attribution: bool = False
    ) -> Dict:
        """
        Enhanced query with NIM reasoning.

        Args:
            query: User question
            k: Number of results
            include_citation_context: Expand with citation graph
            analyze_attribution: Run attribution analysis

        Returns:
            Dict with answer, sources, and optional analysis
        """
        # 1. Hybrid search (your existing pipeline)
        results = self.search_engine.search(query, k=k)

        # 2. Citation graph expansion (if available)
        if include_citation_context and self.citation_graph:
            top_paper_ids = [r.paper_id for r in results[:3]]
            # Expand with citations (implement based on your graph structure)
            # expanded_ids = self._expand_via_citations(top_paper_ids)
            # results += self.search_engine.search_by_ids(expanded_ids)
            pass

        # 3. NIM reasoning
        papers = [r.metadata for r in results]
        answer = self.nim_client.answer_query_with_rag(
            query=query,
            retrieved_papers=papers
        )

        # 4. Optional attribution analysis
        attributions = []
        if analyze_attribution and papers:
            top_paper = papers[0]
            cited = papers[1:6]  # Use other results as citation context
            attribution = self.nim_client.analyze_attribution(
                paper=top_paper,
                cited_papers=cited
            )
            attributions.append(attribution)

        return {
            'query': query,
            'answer': answer,
            'sources': papers[:5],
            'attributions': attributions,
            'total_papers_analyzed': len(papers)
        }


# Example usage
if __name__ == "__main__":
    print("\n=== NVIDIA NIM Integration Test ===\n")

    # Initialize client
    nim = NVIDIANIMClient()

    # Test with sample paper
    sample_paper = {
        'paper_id': 'alphafold_2020',
        'title': 'Highly accurate protein structure prediction with AlphaFold',
        'year': 2020,
        'abstract': 'Proteins are essential to life, and understanding their structure is crucial...',
        'text': 'We present AlphaFold, a deep learning approach that predicts protein structure...'
    }

    sample_citations = [
        {'title': 'Deep learning methods in protein structure prediction', 'year': 2019},
        {'title': 'Attention mechanisms for sequence modeling', 'year': 2017},
    ]

    # Test attribution analysis
    print("Testing attribution analysis...")
    attribution = nim.analyze_attribution(
        paper=sample_paper,
        cited_papers=sample_citations
    )
    print(f"Attribution Score: {attribution.attribution_score:.2f}")
    print(f"ML Methods: {attribution.ml_methods}")
    print(f"Evidence: {attribution.evidence[:2]}")

    # Test RAG query
    print("\nTesting RAG query...")
    answer = nim.answer_query_with_rag(
        query="How did AlphaFold impact drug discovery?",
        retrieved_papers=[sample_paper]
    )
    print(f"Answer: {answer[:200]}...")

    print("\n✅ Integration test complete!")
