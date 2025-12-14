"""
Enhanced RAG Pipeline with NVIDIA Best Practices

Based on NVIDIA's official examples:
1. txt2kg: Knowledge graph extraction from papers
2. rag-workbench: Agentic RAG with hallucination detection
3. nim-llm: Optimized NIM integration

Uses Llama 3.3 70B (latest model)
"""

import os
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from openai import OpenAI
import json


@dataclass
class KnowledgeTriple:
    """Subject-Predicate-Object triple from paper."""
    subject: str
    predicate: str
    object: str
    confidence: float
    source_paper_id: str


@dataclass
class RAGResponse:
    """RAG response with evaluation."""
    answer: str
    is_relevant: bool
    has_hallucination: bool
    confidence: float
    sources: List[str]
    iteration_count: int


class NVIDIAEnhancedRAG:
    """
    Enhanced RAG system with NVIDIA best practices.

    Features:
    - Knowledge graph extraction (txt2kg pattern)
    - Agentic evaluation with hallucination detection (rag-workbench pattern)
    - Llama 3.3 70B for reasoning
    - Iterative refinement until high-quality answer
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize enhanced RAG system.

        Args:
            api_key: NVIDIA API key (or set NVIDIA_API_KEY env var)

        SECURITY WARNING: Never hardcode API keys!
        Always use environment variables or secret management.
        """
        self.api_key = api_key or os.environ.get('NVIDIA_API_KEY')

        if not self.api_key:
            raise ValueError(
                "NVIDIA API key required. Set NVIDIA_API_KEY environment variable.\n"
                "Get key from: https://build.nvidia.com"
            )

        # IMPORTANT: Validate API key format
        if not self.api_key.startswith("nvapi-"):
            raise ValueError("Invalid NVIDIA API key format. Should start with 'nvapi-'")

        self.client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=self.api_key
        )

        # Use latest Llama 3.3 70B (more capable than 3.1)
        self.model = "meta/llama-3.3-70b-instruct"

        # Agentic parameters
        self.max_iterations = 3
        self.relevance_threshold = 0.7
        self.hallucination_threshold = 0.3

    def extract_knowledge_triples(
        self,
        paper: Dict,
        temperature: float = 0.1
    ) -> List[KnowledgeTriple]:
        """
        Extract knowledge graph triples from paper (txt2kg pattern).

        Extracts Subject-Predicate-Object relationships like:
        - (AlphaFold, uses, Transformer Architecture)
        - (Protein Folding, accelerated_by, Deep Learning)

        Args:
            paper: Paper with 'title', 'abstract', 'text' fields
            temperature: Lower for more consistent extraction

        Returns:
            List of KnowledgeTriple objects
        """

        system_prompt = """You are SymbyAI's knowledge graph extraction specialist. Your task is to extract precise Subject-Predicate-Object triples that capture the essential relationships in scientific papers.

EXTRACTION GUIDELINES:

1. ENTITY TYPES TO EXTRACT:
   - Methods/Algorithms (e.g., "AlphaFold", "BERT", "Gradient Descent")
   - Technologies/Frameworks (e.g., "PyTorch", "Transformer Architecture")
   - Scientific Concepts (e.g., "Protein Folding", "Language Modeling")
   - Datasets (e.g., "ImageNet", "PDB")
   - Metrics (e.g., "Accuracy", "RMSD")
   - Applications (e.g., "Drug Discovery", "Machine Translation")

2. PREDICATE TYPES TO USE:
   - Technical: "uses", "implements", "extends", "combines_with", "trained_on"
   - Comparative: "outperforms", "improves_upon", "compared_to"
   - Causal: "enables", "accelerates", "solves", "addresses"
   - Compositional: "consists_of", "part_of", "instance_of"

3. CONFIDENCE SCORING:
   - 0.9-1.0: Relationship explicitly stated
   - 0.7-0.9: Relationship strongly implied
   - 0.5-0.7: Relationship reasonably inferred
   - <0.5: Don't include (too speculative)

OUTPUT FORMAT - Return JSON with "triples" array:
{
  "triples": [
    {"subject": "...", "predicate": "...", "object": "...", "confidence": 0.95},
    ...
  ]
}

Extract 5-10 high-confidence triples that capture the paper's core contributions and methodology."""

        paper_text = f"""
Title: {paper.get('title', '')}
Abstract: {paper.get('abstract', '')}
Full Text: {paper.get('text', '')[:3000]}
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": paper_text}
                ],
                temperature=temperature,
                top_p=0.7,
                max_tokens=1024,
                response_format={"type": "json_object"}
            )

            result = json.loads(response.choices[0].message.content)
            triples = result.get('triples', [])

            return [
                KnowledgeTriple(
                    subject=t['subject'],
                    predicate=t['predicate'],
                    object=t['object'],
                    confidence=t.get('confidence', 0.8),
                    source_paper_id=paper.get('paper_id', '')
                )
                for t in triples
            ]

        except Exception as e:
            print(f"[KG Extraction] Failed: {e}")
            return []

    def evaluate_response(
        self,
        query: str,
        answer: str,
        sources: List[Dict]
    ) -> Tuple[bool, bool, float]:
        """
        Evaluate RAG response for relevancy and hallucination (rag-workbench pattern).

        Returns:
            (is_relevant, has_hallucination, confidence)
        """

        source_text = "\n".join([
            f"Paper: {s.get('title', '')}\nAbstract: {s.get('abstract', '')[:200]}"
            for s in sources[:3]
        ])

        eval_prompt = f"""Perform rigorous evaluation of this RAG (Retrieval-Augmented Generation) response.

=== USER QUERY ===
{query}

=== GENERATED ANSWER ===
{answer}

=== SOURCE DOCUMENTS ===
{source_text}

=== EVALUATION CRITERIA ===

1. RELEVANCY ASSESSMENT:
   - Does the answer directly address the user's question?
   - Is the response on-topic and useful?
   - Are all parts of a multi-part question addressed?

2. HALLUCINATION DETECTION:
   - Does EVERY factual claim in the answer have support in the sources?
   - Are there any invented statistics, dates, names, or technical details?
   - Does the answer make claims that go beyond what sources state?
   - Check: specific numbers, proper nouns, technical claims

3. QUALITY INDICATORS:
   - Citation usage: Does answer reference specific papers?
   - Completeness: Is the answer thorough given available sources?
   - Accuracy: Are source details correctly represented?

=== REQUIRED OUTPUT ===
Return JSON:
{{
  "is_relevant": <true if answer addresses the query, false otherwise>,
  "has_hallucination": <true if ANY unsupported claims found, false if all claims backed by sources>,
  "confidence": <0.0-1.0 overall quality score>,
  "hallucination_examples": ["<specific unsupported claim 1>", "<claim 2>"] or [],
  "missing_aspects": ["<query aspect not addressed>"] or [],
  "reasoning": "<2-3 sentence explanation of evaluation>"
}}

Be strict about hallucination - if uncertain whether a claim is supported, mark has_hallucination as true."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": eval_prompt}],
                temperature=0.1,
                max_tokens=512,
                response_format={"type": "json_object"}
            )

            result = json.loads(response.choices[0].message.content)

            return (
                result.get('is_relevant', False),
                result.get('has_hallucination', True),
                result.get('confidence', 0.0)
            )

        except Exception as e:
            print(f"[Evaluation] Failed: {e}")
            return False, True, 0.0

    def generate_answer(
        self,
        query: str,
        sources: List[Dict],
        previous_answer: Optional[str] = None,
        feedback: Optional[str] = None
    ) -> str:
        """
        Generate answer with optional refinement from previous iteration.

        Args:
            query: User question
            sources: Retrieved papers
            previous_answer: Answer from previous iteration (if refining)
            feedback: Evaluation feedback for refinement

        Returns:
            Generated answer
        """

        # Build context from sources
        context = "\n\n".join([
            f"""Paper ID: {s.get('paper_id', '')}
Title: {s.get('title', '')}
Year: {s.get('year', '')}
Field: {s.get('source_field', '')}
Abstract: {s.get('abstract', '')}
Citations: {s.get('citation_count', 0)}"""
            for s in sources[:10]
        ])

        system_prompt = """You are SymbyAI's research intelligence assistant, specializing in analyzing ML's transformative impact across scientific disciplines.

CORE PRINCIPLES:

1. SOURCE FIDELITY:
   - Base ALL factual claims on the provided source documents
   - ALWAYS cite using format: [Paper ID: xxx] or [Title, Year]
   - If sources don't contain the answer, explicitly state: "The provided sources do not contain sufficient information to answer this question."

2. EVIDENCE STANDARDS:
   - Include quantitative evidence when available (citation counts, performance metrics, dates)
   - Distinguish between what papers claim vs. what they demonstrate
   - Note the recency and relevance of cited evidence

3. ANALYTICAL DEPTH:
   - Synthesize information across multiple sources when applicable
   - Identify trends, patterns, and contradictions in the literature
   - Highlight methodological considerations and limitations

4. RESPONSE QUALITY:
   - Structure answers clearly with logical flow
   - Lead with the most relevant and well-supported information
   - Be concise but comprehensive"""

        user_prompt = f"""Context Papers:
{context}

Question: {query}"""

        # Add refinement instructions if this is an iteration
        if previous_answer and feedback:
            user_prompt += f"""

PREVIOUS ATTEMPT:
{previous_answer}

EVALUATION FEEDBACK:
{feedback}

Please refine your answer to address the feedback while maintaining citation rigor."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,
                top_p=0.7,
                max_tokens=2048,
                stream=False
            )

            return response.choices[0].message.content

        except Exception as e:
            return f"Error generating answer: {e}"

    def query_with_evaluation(
        self,
        query: str,
        sources: List[Dict],
        max_iterations: Optional[int] = None
    ) -> RAGResponse:
        """
        Agentic RAG with iterative evaluation and refinement.

        Workflow (rag-workbench pattern):
        1. Generate answer
        2. Evaluate for relevancy and hallucination
        3. If quality insufficient, refine with feedback
        4. Iterate until high quality or max iterations

        Args:
            query: User question
            sources: Retrieved papers from hybrid search
            max_iterations: Max refinement iterations (default: 3)

        Returns:
            RAGResponse with final answer and evaluation
        """

        max_iter = max_iterations or self.max_iterations

        answer = None
        iteration = 0

        for iteration in range(1, max_iter + 1):
            print(f"[Agentic RAG] Iteration {iteration}/{max_iter}")

            # Generate answer (or refine if iteration > 1)
            if iteration == 1:
                answer = self.generate_answer(query, sources)
            else:
                feedback = f"Iteration {iteration}: Previous answer had issues. "
                if not is_relevant:
                    feedback += "Make answer more relevant to the query. "
                if has_hallucination:
                    feedback += "Remove claims not supported by sources. "
                if confidence < self.relevance_threshold:
                    feedback += "Improve citation rigor. "

                answer = self.generate_answer(query, sources, answer, feedback)

            # Evaluate
            is_relevant, has_hallucination, confidence = self.evaluate_response(
                query, answer, sources
            )

            print(f"  Relevant: {is_relevant}, Hallucination: {has_hallucination}, Confidence: {confidence:.2f}")

            # Check if quality is sufficient
            if (is_relevant and
                not has_hallucination and
                confidence >= self.relevance_threshold):
                print(f"[Agentic RAG] High-quality answer achieved at iteration {iteration}")
                break

        return RAGResponse(
            answer=answer,
            is_relevant=is_relevant,
            has_hallucination=has_hallucination,
            confidence=confidence,
            sources=[s.get('paper_id', '') for s in sources[:5]],
            iteration_count=iteration
        )

    def analyze_ml_impact_chain(
        self,
        seed_paper: Dict,
        citing_papers: List[Dict]
    ) -> Dict:
        """
        Trace ML impact through citation chain.

        Combines knowledge graph extraction with impact analysis.

        Args:
            seed_paper: Original ML method paper (e.g., AlphaFold)
            citing_papers: Papers that cite the seed paper

        Returns:
            Dict with impact analysis and knowledge graph
        """

        # Extract knowledge from seed paper
        seed_triples = self.extract_knowledge_triples(seed_paper)

        # Extract knowledge from citing papers
        downstream_triples = []
        for paper in citing_papers[:10]:
            triples = self.extract_knowledge_triples(paper)
            downstream_triples.extend(triples)

        # Analyze impact propagation
        analysis_prompt = f"""Analyze the scientific impact propagation of this ML method through citation networks.

=== SEED PAPER (Origin) ===
Title: {seed_paper.get('title', '')}
Year: {seed_paper.get('year', '')}

=== EXTRACTED KNOWLEDGE FROM SEED ===
{json.dumps([{{'subject': t.subject, 'predicate': t.predicate, 'object': t.object}} for t in seed_triples], indent=2)}

=== CITATION METRICS ===
Total citing papers analyzed: {len(citing_papers)}

=== KNOWLEDGE FROM DOWNSTREAM (CITING) PAPERS ===
{json.dumps([{{'subject': t.subject, 'predicate': t.predicate, 'object': t.object}} for t in downstream_triples[:20]], indent=2)}

=== ANALYSIS REQUIRED ===

Provide a comprehensive impact analysis:

1. DISCIPLINARY SPREAD:
   - Which scientific fields adopted this method?
   - What adaptations were made for different domains?
   - Identify any cross-pollination patterns

2. APPLICATION EVOLUTION:
   - What novel applications emerged from the original method?
   - How did the method's use cases expand over time?
   - Any unexpected or creative applications?

3. IMPACT CLASSIFICATION:
   - Incremental: Minor improvements to existing approaches
   - Significant: Meaningful advances enabling new capabilities
   - Transformational: Paradigm-shifting impact across fields
   - Revolutionary: Fundamental change in how science is conducted

4. EVIDENCE SYNTHESIS:
   - Key indicators supporting impact assessment
   - Notable limitations or gaps in adoption
   - Future trajectory predictions

=== OUTPUT FORMAT ===
Return JSON:
{{
    "disciplinary_spread": {{
        "primary_fields": ["<field1>", "<field2>"],
        "secondary_fields": ["<field3>"],
        "adaptation_patterns": "<description>"
    }},
    "applications": {{
        "original": "<original application>",
        "emerged": ["<new app 1>", "<new app 2>"],
        "unexpected": ["<surprising application>"]
    }},
    "impact_classification": "<incremental|significant|transformational|revolutionary>",
    "impact_evidence": ["<evidence point 1>", "<evidence point 2>"],
    "limitations": ["<limitation 1>"],
    "future_trajectory": "<prediction based on trends>"
}}"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": analysis_prompt}],
                temperature=0.2,
                max_tokens=2048,
                response_format={"type": "json_object"}
            )

            impact_analysis = json.loads(response.choices[0].message.content)

            return {
                'seed_paper_id': seed_paper.get('paper_id', ''),
                'seed_triples': seed_triples,
                'downstream_triples': downstream_triples,
                'impact_analysis': impact_analysis,
                'citation_count': len(citing_papers)
            }

        except Exception as e:
            return {
                'error': str(e),
                'seed_triples': seed_triples,
                'downstream_triples': downstream_triples
            }


class NVIDIAKnowledgeGraphBuilder:
    """
    Build citation + semantic knowledge graph (txt2kg pattern).

    Combines:
    - Citation relationships (paper A cites paper B)
    - Semantic relationships (paper A uses technique from paper B)
    - ML adoption flow (AlphaFold → drug discovery → clinical trials)
    """

    def __init__(self, rag_client: NVIDIAEnhancedRAG):
        self.rag = rag_client
        self.triples = []

    def extract_from_corpus(
        self,
        papers: List[Dict],
        max_papers: Optional[int] = None
    ):
        """
        Extract knowledge triples from entire corpus.

        Args:
            papers: List of papers
            max_papers: Limit extraction (for testing)
        """

        papers_to_process = papers[:max_papers] if max_papers else papers

        print(f"[KG Builder] Extracting triples from {len(papers_to_process)} papers...")

        for i, paper in enumerate(papers_to_process):
            if i % 100 == 0:
                print(f"  Progress: {i}/{len(papers_to_process)}")

            triples = self.rag.extract_knowledge_triples(paper)
            self.triples.extend(triples)

        print(f"[KG Builder] Extracted {len(self.triples)} triples")

    def export_to_graph_format(self, output_path: str):
        """Export triples for graph visualization or ArangoDB."""

        graph_data = {
            'nodes': [],
            'edges': []
        }

        # Build nodes and edges
        entities = set()
        for triple in self.triples:
            entities.add(triple.subject)
            entities.add(triple.object)

            graph_data['edges'].append({
                'source': triple.subject,
                'target': triple.object,
                'relation': triple.predicate,
                'confidence': triple.confidence,
                'source_paper': triple.source_paper_id
            })

        graph_data['nodes'] = [{'id': e, 'label': e} for e in entities]

        with open(output_path, 'w') as f:
            json.dump(graph_data, f, indent=2)

        print(f"[KG Builder] Exported graph to {output_path}")
        print(f"  Nodes: {len(graph_data['nodes'])}")
        print(f"  Edges: {len(graph_data['edges'])}")


# Example usage
if __name__ == "__main__":
    print("\n=== NVIDIA Enhanced RAG Test ===\n")

    # SECURITY: Use environment variable, not hardcoded key!
    # export NVIDIA_API_KEY="nvapi-xxxxx"

    if not os.environ.get('NVIDIA_API_KEY'):
        print("⚠️  Set NVIDIA_API_KEY environment variable first!")
        print("   export NVIDIA_API_KEY='nvapi-xxxxx'")
        exit(1)

    # Initialize
    rag = NVIDIAEnhancedRAG()

    # Test 1: Knowledge Graph Extraction
    print("Test 1: Knowledge Triple Extraction\n")

    sample_paper = {
        'paper_id': 'alphafold_2020',
        'title': 'Highly accurate protein structure prediction with AlphaFold',
        'abstract': 'Proteins are essential to life, and understanding their structure is crucial...',
        'text': 'We present AlphaFold, which uses deep learning to predict protein structure from sequence...'
    }

    triples = rag.extract_knowledge_triples(sample_paper)
    print(f"Extracted {len(triples)} triples:")
    for t in triples[:5]:
        print(f"  ({t.subject}) --[{t.predicate}]--> ({t.object}) [confidence: {t.confidence}]")

    # Test 2: Agentic RAG with Evaluation
    print("\n\nTest 2: Agentic RAG with Evaluation\n")

    sample_sources = [sample_paper]

    response = rag.query_with_evaluation(
        query="How does AlphaFold predict protein structures?",
        sources=sample_sources,
        max_iterations=3
    )

    print(f"Answer (after {response.iteration_count} iterations):")
    print(f"  {response.answer[:200]}...")
    print(f"\nEvaluation:")
    print(f"  Relevant: {response.is_relevant}")
    print(f"  Hallucination: {response.has_hallucination}")
    print(f"  Confidence: {response.confidence:.2f}")

    print("\n✅ Enhanced RAG test complete!")
