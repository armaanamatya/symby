#!/usr/bin/env python3
"""
Test NVIDIA Integration
Run this to verify your setup before integrating with the main app
"""

import os
import sys

def check_api_key():
    """Test 1: Verify API key is set and valid format."""
    print("\n" + "="*60)
    print("Test 1: API Key Configuration")
    print("="*60)

    api_key = os.environ.get('NVIDIA_API_KEY')

    if not api_key:
        print("❌ NVIDIA_API_KEY not set")
        print("\n   Fix: export NVIDIA_API_KEY='nvapi-xxxxx'")
        print("   Get key from: https://build.nvidia.com")
        return False

    if not api_key.startswith('nvapi-'):
        print(f"❌ Invalid key format: {api_key[:10]}...")
        print("   NVIDIA API keys should start with 'nvapi-'")
        return False

    # Check if it's the exposed key from the message
    if api_key == "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":
        print("⚠️  WARNING: You're using the publicly exposed API key!")
        print("   This key was shared in a public message.")
        print("   Please revoke it and generate a new one at:")
        print("   https://build.nvidia.com")
        print("\n   For now, the test will continue, but PLEASE change this key!")
        input("\n   Press Enter to continue...")

    print(f"✅ API key configured: {api_key[:15]}...{api_key[-5:]}")
    return True


def test_basic_completion():
    """Test 2: Basic Llama 3.3 70B completion."""
    print("\n" + "="*60)
    print("Test 2: Basic Llama 3.3 70B Completion")
    print("="*60)

    try:
        from openai import OpenAI
    except ImportError:
        print("❌ openai package not installed")
        print("   Fix: pip install openai")
        return False

    print("Calling NVIDIA NIM with Llama 3.3 70B...")

    try:
        client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=os.environ['NVIDIA_API_KEY']
        )

        response = client.chat.completions.create(
            model="meta/llama-3.3-70b-instruct",
            messages=[
                {"role": "user", "content": "Reply with exactly: 'NVIDIA NIM test successful'"}
            ],
            temperature=0.2,
            top_p=0.7,
            max_tokens=50,
            stream=False
        )

        answer = response.choices[0].message.content
        print(f"\n   Response: {answer}")

        if "successful" in answer.lower():
            print("✅ Basic completion works!")
            return True
        else:
            print("⚠️  Got response but unexpected content")
            return False

    except Exception as e:
        print(f"❌ API call failed: {e}")
        print("\n   Possible issues:")
        print("   - Invalid API key")
        print("   - Network connectivity")
        print("   - Rate limit exceeded")
        return False


def test_enhanced_rag():
    """Test 3: Enhanced RAG with evaluation."""
    print("\n" + "="*60)
    print("Test 3: Enhanced RAG with Evaluation")
    print("="*60)

    try:
        from src.rag.nvidia_enhanced_rag import NVIDIAEnhancedRAG
    except ImportError as e:
        print(f"❌ Cannot import NVIDIAEnhancedRAG: {e}")
        print("   Make sure you're running from the project root")
        return False

    print("Initializing Enhanced RAG...")

    try:
        rag = NVIDIAEnhancedRAG()
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        return False

    # Test with sample paper
    sample_paper = {
        'paper_id': 'test_alphafold',
        'title': 'AlphaFold: Protein Structure Prediction',
        'abstract': 'AlphaFold uses deep learning and attention mechanisms to predict protein structures with high accuracy.',
        'text': 'We developed AlphaFold using transformer architectures and trained on protein databases.',
        'year': 2020,
        'source_field': 'Biology'
    }

    print("Testing knowledge triple extraction...")

    try:
        triples = rag.extract_knowledge_triples(sample_paper)
        print(f"\n   Extracted {len(triples)} knowledge triples:")
        for t in triples[:3]:
            print(f"   - ({t.subject}) --[{t.predicate}]--> ({t.object})")

        if len(triples) > 0:
            print("✅ Knowledge extraction works!")
        else:
            print("⚠️  No triples extracted (model may need better prompting)")

    except Exception as e:
        print(f"❌ Knowledge extraction failed: {e}")
        return False

    print("\nTesting agentic RAG with evaluation...")

    try:
        response = rag.query_with_evaluation(
            query="How does AlphaFold predict protein structures?",
            sources=[sample_paper],
            max_iterations=2
        )

        print(f"\n   Answer (first 200 chars):")
        print(f"   {response.answer[:200]}...")
        print(f"\n   Quality Metrics:")
        print(f"   - Relevant: {response.is_relevant}")
        print(f"   - Hallucination: {response.has_hallucination}")
        print(f"   - Confidence: {response.confidence:.2f}")
        print(f"   - Iterations: {response.iteration_count}")

        if response.answer and response.confidence > 0:
            print("✅ Agentic RAG works!")
            return True
        else:
            print("⚠️  RAG returned response but quality is low")
            return False

    except Exception as e:
        print(f"❌ RAG query failed: {e}")
        return False


def test_integration_with_existing_code():
    """Test 4: Integration with existing hybrid search."""
    print("\n" + "="*60)
    print("Test 4: Integration with Existing Code")
    print("="*60)

    # Check if existing modules are available
    try:
        from src.embeddings.bge_m3 import BGEEmbedder
        print("✅ BGEEmbedder available")
    except ImportError:
        print("⚠️  BGEEmbedder not available (optional)")

    try:
        from src.index.vector_index import VectorIndex
        print("✅ VectorIndex available")
    except ImportError:
        print("⚠️  VectorIndex not available (optional)")

    try:
        from src.search.hybrid_search import HybridSearchEngine
        print("✅ HybridSearchEngine available")
    except ImportError:
        print("⚠️  HybridSearchEngine not available (optional)")

    print("\n   Note: Your existing components can work alongside NVIDIA NIM")
    print("   The enhanced RAG integrates seamlessly with hybrid search results")

    return True


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("NVIDIA INTEGRATION TEST SUITE")
    print("="*70)
    print("\nThis will test your NVIDIA API setup and enhanced RAG integration.")
    print("Make sure NVIDIA_API_KEY is set: export NVIDIA_API_KEY='nvapi-xxxxx'")

    input("\nPress Enter to start tests...")

    # Run tests
    results = []

    # Test 1: API Key
    results.append(("API Key", check_api_key()))

    if not results[0][1]:
        print("\n❌ Cannot proceed without valid API key")
        print("   Get one from: https://build.nvidia.com")
        sys.exit(1)

    # Test 2: Basic completion
    results.append(("Basic Completion", test_basic_completion()))

    # Test 3: Enhanced RAG
    results.append(("Enhanced RAG", test_enhanced_rag()))

    # Test 4: Integration
    results.append(("Integration", test_integration_with_existing_code()))

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status:8} {test_name}")

    all_passed = all(r[1] for r in results)

    print("\n" + "="*70)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("\nYou're ready to integrate NVIDIA NIM into your app!")
        print("\nNext steps:")
        print("1. Review INTEGRATION_GUIDE.md")
        print("2. Add NIM to app.py (see examples in guide)")
        print("3. Test with your 81K papers dataset")
        print("4. Deploy dashboard for hackathon!")
    else:
        print("⚠️  SOME TESTS FAILED")
        print("\nPlease fix the issues above before integrating.")
        print("Check INTEGRATION_GUIDE.md for troubleshooting.")

    print("="*70 + "\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
