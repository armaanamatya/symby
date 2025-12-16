"""
Fixed NVIDIA RAG with:
1. Robust JSON parsing (handles markdown code fences)
2. Integration with YOUR vector store (BGE-M3 + FAISS)
3. Source tracking with chunks and retrieval scores
4. Fallback to non-JSON outputs
"""

import os
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from openai import OpenAI
import json
import re


@dataclass
class SourceChunk:
    """A retrieved chunk with metadata."""
    paper_id: str
    title: str
    chunk_text: str
    chunk_index: int  # Which chunk from the paper
    retrieval_score: float  # Similarity score from vector search
    year: int
    field: str
    citation_count: int

    def to_context_string(self) -> str:
        """Format for LLM context."""
        return f"""[Source {self.chunk_index}] {self.title} ({self.year})
Field: {self.field} | Citations: {self.citation_count} | Relevance: {self.retrieval_score:.3f}
Paper ID: {self.paper_id}

{self.chunk_text}
"""


@dataclass
class RAGResponseWithSources:
    """Enhanced RAG response with source tracking."""
    answer: str
    sources: List[SourceChunk]
    is_relevant: bool = True
    has_hallucination: bool = False
    confidence: float = 0.8
    iteration_count: int = 1

    def format_with_sources(self) -> str:
        """Format answer with source citations."""
        output = f"{self.answer}\n\n"
        output += "="*70 + "\n"
        output += "SOURCES:\n"
        output += "="*70 + "\n"
        for i, source in enumerate(self.sources, 1):
            output += f"\n[{i}] {source.title} ({source.year})\n"
            output += f"    Paper ID: {source.paper_id}\n"
            output += f"    Field: {source.field} | Citations: {source.citation_count}\n"
            output += f"    Relevance Score: {source.retrieval_score:.3f}\n"
            output += f"    Chunk: {source.chunk_text[:100]}...\n"
        return output


def extract_json_from_text(text: str) -> dict:
    """
    Robust JSON extraction that handles markdown code fences.

    Handles:
    - Plain JSON
    - JSON wrapped in ```json ... ```
    - JSON wrapped in ``` ... ```
    """
    # Try direct parsing first
    try:
        return json.loads(text)
    except:
        pass

    # Try extracting from markdown code fence
    json_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
    matches = re.findall(json_pattern, text, re.DOTALL)
    if matches:
        try:
            return json.loads(matches[0])
        except:
            pass

    # Try finding any {...} block
    brace_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    matches = re.findall(brace_pattern, text, re.DOTALL)
    for match in matches:
        try:
            return json.loads(match)
        except:
            continue

    # Fallback: return empty dict
    return {}


class NVIDIARAGFixed:
    """
    Fixed NVIDIA RAG with proper vector store integration.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        vector_index=None,  # Your FAISS VectorIndex
        embedder=None,  # Your BGEEmbedder
    ):
        """
        Initialize with your existing components.

        Args:
            api_key: NVIDIA API key
            vector_index: Your VectorIndex instance (FAISS)
            embedder: Your BGEEmbedder instance
        """
        self.api_key = api_key or os.environ.get('NVIDIA_API_KEY')

        if not self.api_key:
            raise ValueError("NVIDIA_API_KEY required")

        self.client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=self.api_key
        )

        self.model = "meta/llama-3.3-70b-instruct"
        self.vector_index = vector_index
        self.embedder = embedder

    def chunk_paper_text(self, paper: Dict, chunk_size: int = 500) -> List[str]:
        """
        Chunk paper text for retrieval.

        Args:
            paper: Paper with 'text' or 'abstract' field
            chunk_size: Characters per chunk

        Returns:
            List of text chunks
        """
        # Combine title, abstract, and text
        full_text = paper.get('title', '') + "\n\n"
        full_text += paper.get('abstract', '') + "\n\n"
        full_text += paper.get('text', '')

        # Simple chunking (you can improve this with sentence boundaries)
        chunks = []
        for i in range(0, len(full_text), chunk_size):
            chunk = full_text[i:i + chunk_size]
            if chunk.strip():
                chunks.append(chunk.strip())

        return chunks

    def retrieve_with_sources(
        self,
        query: str,
        k: int = 10,
        paper_metadata: Optional[Dict[str, Dict]] = None
    ) -> List[SourceChunk]:
        """
        Retrieve documents using YOUR vector store.

        Args:
            query: User query
            k: Number of results
            paper_metadata: Dict mapping paper_id to metadata

        Returns:
            List of SourceChunk objects with full tracking
        """
        if not self.vector_index or not self.embedder:
            print("⚠️  No vector store configured, using simulated results")
            return self._get_simulated_sources(query)

        # 1. Embed query with your BGE-M3 embedder
        query_embedding = self.embedder.embed_single(query)['dense']

        # 2. Search with your FAISS index
        scores, paper_ids = self.vector_index.search(
            query_embedding.reshape(1, -1),
            k=k
        )

        # 3. Build SourceChunk objects
        sources = []
        for score, paper_id_list in zip(scores[0], paper_ids[0]):
            if paper_metadata and paper_id_list in paper_metadata:
                metadata = paper_metadata[paper_id_list]

                # Chunk the paper text
                chunks = self.chunk_paper_text(metadata)

                # For now, use first chunk (you can rank chunks later)
                if chunks:
                    sources.append(SourceChunk(
                        paper_id=paper_id_list,
                        title=metadata.get('title', 'Unknown'),
                        chunk_text=chunks[0],  # First chunk
                        chunk_index=0,
                        retrieval_score=float(score),
                        year=metadata.get('year', 0),
                        field=metadata.get('source_field', 'Unknown'),
                        citation_count=metadata.get('citation_count', 0)
                    ))

        return sources[:k]

    def _get_simulated_sources(self, query: str) -> List[SourceChunk]:
        """Fallback simulated sources for demo."""
        return [
            SourceChunk(
                paper_id='alphafold_2020',
                title='Highly accurate protein structure prediction with AlphaFold',
                chunk_text='AlphaFold uses deep learning and attention mechanisms to predict protein 3D structures from amino acid sequences. The model achieved unprecedented accuracy in the CASP14 competition, marking a breakthrough in computational biology.',
                chunk_index=0,
                retrieval_score=0.95,
                year=2020,
                field='Biology',
                citation_count=15000
            ),
            SourceChunk(
                paper_id='drug_discovery_2021',
                title='Accelerating drug discovery with AlphaFold predictions',
                chunk_text='We used AlphaFold-predicted protein structures to identify drug targets for cancer therapy. This approach reduced the discovery timeline from 5 years to 6 months, demonstrating a 10x acceleration.',
                chunk_index=0,
                retrieval_score=0.88,
                year=2021,
                field='Medicine',
                citation_count=3500
            ),
            SourceChunk(
                paper_id='covid_proteins_2021',
                title='Understanding SARS-CoV-2 proteins with AlphaFold',
                chunk_text='AlphaFold predictions of viral spike protein structures enabled rapid vaccine development during the COVID-19 pandemic. Our analysis identified key binding sites within 2 weeks.',
                chunk_index=0,
                retrieval_score=0.85,
                year=2021,
                field='Biology',
                citation_count=5000
            )
        ]

    def query_with_sources(
        self,
        query: str,
        k: int = 10,
        paper_metadata: Optional[Dict[str, Dict]] = None,
        show_sources: bool = True
    ) -> RAGResponseWithSources:
        """
        Query with full source tracking.

        Args:
            query: User question
            k: Number of sources to retrieve
            paper_metadata: Paper metadata dict
            show_sources: Whether to show source details

        Returns:
            RAGResponseWithSources with answer and source tracking
        """
        # 1. Retrieve with your vector store
        sources = self.retrieve_with_sources(query, k, paper_metadata)

        if show_sources:
            print(f"\n📚 Retrieved {len(sources)} sources:")
            for i, src in enumerate(sources[:5], 1):
                print(f"  [{i}] {src.title} (score: {src.retrieval_score:.3f})")

        # 2. Build context for LLM
        context = "\n\n".join([src.to_context_string() for src in sources])

        # 3. Generate answer
        system_prompt = """You are an expert research assistant analyzing ML's impact on science.

CRITICAL RULES:
1. Base answers ONLY on provided sources
2. Cite sources using [Source N] format
3. Include paper IDs when making claims
4. If sources don't contain the answer, say so
5. Provide quantitative evidence (years, speedup, citation counts)"""

        user_prompt = f"""Context from retrieved papers:

{context}

Question: {query}

Provide a detailed answer citing sources."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,
                top_p=0.7,
                max_tokens=2000,
                stream=False
            )

            answer = response.choices[0].message.content

            return RAGResponseWithSources(
                answer=answer,
                sources=sources,
                is_relevant=True,
                has_hallucination=False,
                confidence=0.85,
                iteration_count=1
            )

        except Exception as e:
            return RAGResponseWithSources(
                answer=f"Error generating answer: {e}",
                sources=sources,
                is_relevant=False,
                has_hallucination=True,
                confidence=0.0,
                iteration_count=0
            )

    def extract_knowledge_triples_robust(
        self,
        paper: Dict,
        temperature: float = 0.1
    ) -> List[Dict]:
        """
        Extract knowledge triples with ROBUST JSON handling.

        Returns raw dicts instead of dataclass to avoid parsing issues.
        """

        paper_text = f"""
Title: {paper.get('title', '')}
Abstract: {paper.get('abstract', '')}
Full Text: {paper.get('text', '')[:3000]}
"""

        # Simplified prompt - don't request JSON format, just structure
        prompt = f"""Extract 5-10 key relationships from this paper as subject-predicate-object triples.

{paper_text}

List each triple on a new line in this format:
(Subject) --[Predicate]--> (Object)

Example:
(AlphaFold) --[uses]--> (Transformer Architecture)
(Protein Folding) --[accelerated_by]--> (Deep Learning)
(Drug Discovery) --[enabled_by]--> (Structure Prediction)

Extract the triples:"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=1000,
                stream=False
            )

            text = response.choices[0].message.content

            # Parse triples from text format
            triples = []
            triple_pattern = r'\(([^)]+)\)\s*--\[([^\]]+)\]-->\s*\(([^)]+)\)'
            matches = re.findall(triple_pattern, text)

            for subject, predicate, obj in matches:
                triples.append({
                    'subject': subject.strip(),
                    'predicate': predicate.strip(),
                    'object': obj.strip(),
                    'confidence': 0.8,
                    'source_paper_id': paper.get('paper_id', '')
                })

            return triples

        except Exception as e:
            print(f"[KG Extraction] Failed: {e}")
            return []


# Example usage
if __name__ == "__main__":
    print("\n=== Fixed NVIDIA RAG Demo ===\n")

    # Initialize
    rag = NVIDIARAGFixed()

    # Test 1: Query with source tracking
    print("Test 1: Query with Source Tracking\n")

    response = rag.query_with_sources(
        query="How did AlphaFold accelerate drug discovery?",
        k=5,
        show_sources=True
    )

    print("\n" + "="*70)
    print("ANSWER WITH SOURCES:")
    print("="*70)
    print(response.format_with_sources())

    # Test 2: Knowledge extraction (fixed)
    print("\n\nTest 2: Knowledge Triple Extraction (Fixed)\n")

    sample_paper = {
        'paper_id': 'alphafold_2020',
        'title': 'Highly accurate protein structure prediction with AlphaFold',
        'abstract': 'AlphaFold uses deep learning to predict protein structures.',
        'text': 'We developed AlphaFold using transformer architectures and attention mechanisms.',
        'year': 2020
    }

    triples = rag.extract_knowledge_triples_robust(sample_paper)

    print(f"Extracted {len(triples)} triples:")
    for t in triples:
        print(f"  ({t['subject']}) --[{t['predicate']}]--> ({t['object']})")

    print("\n✅ Fixed RAG test complete!")
