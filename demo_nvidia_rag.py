#!/usr/bin/env python3
"""
Complete NVIDIA RAG Demo
Demonstrates integration of:
1. NVIDIA NIM (Llama 3.3 70B)
2. Knowledge graph extraction
3. Agentic RAG with evaluation
4. Your existing hybrid search (simulated)
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from src.rag.nvidia_enhanced_rag import NVIDIAEnhancedRAG, NVIDIAKnowledgeGraphBuilder
from src.utils.logger import setup_logging
import json


def demo_1_basic_completion():
    """Demo 1: Basic Llama 3.3 70B completion."""
    print("\n" + "="*70)
    print("DEMO 1: Basic Llama 3.3 70B Completion")
    print("="*70)

    from openai import OpenAI

    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=os.environ['NVIDIA_API_KEY']
    )

    response = client.chat.completions.create(
        model="meta/llama-3.3-70b-instruct",
        messages=[{
            "role": "user",
            "content": "In one sentence, explain how transformers revolutionized NLP."
        }],
        temperature=0.2,
        top_p=0.7,
        max_tokens=200
    )

    print(f"\nQuery: How did transformers revolutionize NLP?")
    print(f"\nAnswer: {response.choices[0].message.content}")
    print("\n✅ Basic completion successful!")


def demo_2_knowledge_extraction():
    """Demo 2: Extract knowledge triples from AlphaFold paper."""
    print("\n" + "="*70)
    print("DEMO 2: Knowledge Graph Extraction (txt2kg pattern)")
    print("="*70)

    rag = NVIDIAEnhancedRAG()

    # Sample paper about AlphaFold
    alphafold_paper = {
        'paper_id': 'alphafold_2020',
        'title': 'Highly accurate protein structure prediction with AlphaFold',
        'abstract': '''
        Proteins are essential to life, and understanding their structure is crucial for
        drug discovery and understanding diseases. We present AlphaFold, a deep learning
        system that predicts protein 3D structures from amino acid sequences with unprecedented
        accuracy. AlphaFold uses attention mechanisms and transformer architectures, trained
        on a large database of known protein structures. Our method achieved first place in
        the CASP14 competition, marking a breakthrough in computational biology.
        ''',
        'text': '''
        We developed AlphaFold using deep neural networks with attention mechanisms.
        The model processes amino acid sequences and predicts inter-residue distances and angles.
        Training was performed on 170,000 protein structures from the Protein Data Bank.
        AlphaFold's architecture incorporates transformer blocks for sequence processing
        and geometric constraints for structure refinement.
        ''',
        'year': 2020,
        'source_field': 'Biology'
    }

    print("\nExtracting knowledge triples from AlphaFold paper...")

    triples = rag.extract_knowledge_triples(alphafold_paper)

    print(f"\n✅ Extracted {len(triples)} knowledge triples:\n")
    for i, t in enumerate(triples, 1):
        print(f"{i}. ({t.subject}) --[{t.predicate}]--> ({t.object})")
        print(f"   Confidence: {t.confidence:.2f}\n")


def demo_3_agentic_rag():
    """Demo 3: Agentic RAG with evaluation and refinement."""
    print("\n" + "="*70)
    print("DEMO 3: Agentic RAG with Evaluation (rag-workbench pattern)")
    print("="*70)

    rag = NVIDIAEnhancedRAG()

    # Simulate search results from your hybrid search
    search_results = [
        {
            'paper_id': 'alphafold_2020',
            'title': 'Highly accurate protein structure prediction with AlphaFold',
            'abstract': 'AlphaFold uses deep learning to predict protein structures with high accuracy.',
            'year': 2020,
            'source_field': 'Biology',
            'citation_count': 15000
        },
        {
            'paper_id': 'drug_discovery_2021',
            'title': 'Accelerating drug discovery with AlphaFold predictions',
            'abstract': 'We used AlphaFold-predicted structures to identify drug targets, reducing discovery time from years to months.',
            'year': 2021,
            'source_field': 'Medicine',
            'citation_count': 3500
        },
        {
            'paper_id': 'covid_proteins_2021',
            'title': 'Understanding SARS-CoV-2 proteins with AlphaFold',
            'abstract': 'AlphaFold predictions of viral proteins enabled rapid vaccine development.',
            'year': 2021,
            'source_field': 'Biology',
            'citation_count': 5000
        }
    ]

    query = "How did AlphaFold accelerate drug discovery and COVID-19 research?"

    print(f"\nQuery: {query}")
    print(f"Retrieved: {len(search_results)} papers from hybrid search")
    print("\nRunning agentic RAG (will iterate until high quality)...\n")

    response = rag.query_with_evaluation(
        query=query,
        sources=search_results,
        max_iterations=3
    )

    print("\n" + "-"*70)
    print("ANSWER:")
    print("-"*70)
    print(response.answer)

    print("\n" + "-"*70)
    print("EVALUATION:")
    print("-"*70)
    print(f"Relevant: {'✅ Yes' if response.is_relevant else '❌ No'}")
    print(f"Hallucination: {'❌ Detected' if response.has_hallucination else '✅ None detected'}")
    print(f"Confidence: {response.confidence:.2f}")
    print(f"Iterations: {response.iteration_count}")
    print(f"Sources: {', '.join(response.sources)}")

    print("\n✅ Agentic RAG complete!")


def demo_4_impact_chain():
    """Demo 4: Trace ML impact through citation chain."""
    print("\n" + "="*70)
    print("DEMO 4: ML Impact Chain Analysis")
    print("="*70)

    rag = NVIDIAEnhancedRAG()

    seed_paper = {
        'paper_id': 'transformer_2017',
        'title': 'Attention is All You Need',
        'abstract': 'We propose the Transformer, a model architecture eschewing recurrence and instead relying entirely on attention mechanisms.',
        'text': 'The Transformer uses multi-head self-attention and position-wise feedforward networks...',
        'year': 2017,
        'source_field': 'Computer Science'
    }

    citing_papers = [
        {
            'paper_id': 'bert_2018',
            'title': 'BERT: Pre-training of Deep Bidirectional Transformers',
            'abstract': 'BERT uses transformer encoders for language understanding tasks.',
            'year': 2018
        },
        {
            'paper_id': 'gpt3_2020',
            'title': 'Language Models are Few-Shot Learners',
            'abstract': 'GPT-3 scales transformers to 175B parameters for few-shot learning.',
            'year': 2020
        },
        {
            'paper_id': 'alphafold_2020',
            'title': 'Highly accurate protein structure prediction with AlphaFold',
            'abstract': 'AlphaFold adapts transformer attention for protein structure.',
            'year': 2020
        }
    ]

    print("\nAnalyzing how Transformers propagated through science...")
    print(f"Seed: {seed_paper['title']} (2017)")
    print(f"Downstream papers: {len(citing_papers)}")

    impact_analysis = rag.analyze_ml_impact_chain(
        seed_paper=seed_paper,
        citing_papers=citing_papers
    )

    print("\n✅ Impact Analysis Complete!")
    print(f"\nSeed paper triples: {len(impact_analysis.get('seed_triples', []))}")
    print(f"Downstream triples: {len(impact_analysis.get('downstream_triples', []))}")

    if 'impact_analysis' in impact_analysis:
        print("\nImpact Summary:")
        print(json.dumps(impact_analysis['impact_analysis'], indent=2)[:500] + "...")


def demo_5_integration_with_existing():
    """Demo 5: How to integrate with your existing hybrid search."""
    print("\n" + "="*70)
    print("DEMO 5: Integration with Your Existing Code")
    print("="*70)

    print("""
Your existing pipeline:
1. ✅ BGE-M3 embeddings
2. ✅ FAISS vector index
3. ✅ Hybrid search (dense + sparse + graph)

Enhanced with NVIDIA NIM:
4. 🆕 NIM for reasoning & analysis
5. 🆕 Knowledge graph extraction
6. 🆕 Agentic evaluation

Integration is simple:

```python
# Your existing search
from src.search.hybrid_search import HybridSearchEngine

search_engine = HybridSearchEngine()
results = search_engine.search("Your query", k=10)

# Add NVIDIA NIM reasoning
from src.rag.nvidia_enhanced_rag import NVIDIAEnhancedRAG

rag = NVIDIAEnhancedRAG()
enhanced_response = rag.query_with_evaluation(
    query="Your query",
    sources=[r.metadata for r in results],
    max_iterations=3
)

# Display
print(enhanced_response.answer)
print(f"Quality: {enhanced_response.confidence:.2f}")
```

That's it! 5 lines to add AI reasoning to your search.
""")


def main():
    """Run all demos with logging."""

    # Setup logging - captures all output to logs/demo_nvidia_rag_TIMESTAMP.log
    with setup_logging("demo_nvidia_rag"):
        print("\n" + "="*70)
        print("NVIDIA ENHANCED RAG - COMPLETE DEMO")
        print("="*70)
        print("\nThis demo showcases all NVIDIA patterns integrated:")
        print("1. Basic Llama 3.3 70B completion")
        print("2. Knowledge graph extraction (txt2kg)")
        print("3. Agentic RAG with evaluation (rag-workbench)")
        print("4. ML impact chain analysis")
        print("5. Integration with your existing code")

        if not os.environ.get('NVIDIA_API_KEY'):
            print("\n❌ NVIDIA_API_KEY not set!")
            print("   Run: source venv/bin/activate")
            print("   .env file should be loaded automatically")
            return

        print(f"\n✅ API Key loaded: {os.environ['NVIDIA_API_KEY'][:15]}...")

        input("\nPress Enter to start demos...")

        try:
            demo_1_basic_completion()
            input("\nPress Enter for next demo...")

            demo_2_knowledge_extraction()
            input("\nPress Enter for next demo...")

            demo_3_agentic_rag()
            input("\nPress Enter for next demo...")

            demo_4_impact_chain()
            input("\nPress Enter for next demo...")

            demo_5_integration_with_existing()

        except KeyboardInterrupt:
            print("\n\n⏸️  Demo interrupted")
            return
        except Exception as e:
            print("\n\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return

        print("\n" + "="*70)
        print("🎉 ALL DEMOS COMPLETE!")
        print("="*70)
    print("\nYou've seen:")
    print("✅ Llama 3.3 70B reasoning")
    print("✅ Knowledge graph extraction")
    print("✅ Agentic RAG with quality control")
    print("✅ ML impact tracing")
    print("✅ Integration patterns")

    print("\nNext steps:")
    print("1. Integrate into app.py (see INTEGRATION_GUIDE.md)")
    print("2. Process your 81K papers dataset")
    print("3. Build dashboard visualizations")
    print("4. Demo at hackathon!")

    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
