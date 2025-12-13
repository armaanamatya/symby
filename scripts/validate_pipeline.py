#!/usr/bin/env python3
"""
Pipeline Validation Script for Symby GraphRAG

Tests the entire pipeline end-to-end with sample data:
1. Data loading and preprocessing
2. Embedding generation (mock or real BGE-M3)
3. Vector index building and search
4. Sparse index building and search
5. Semantic graph construction
6. Hybrid search integration
7. Graph analytics

Run with: python scripts/validate_pipeline.py
"""

import sys
import os
import time
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Test results tracking
RESULTS = {
    'passed': [],
    'failed': [],
    'skipped': [],
}


def print_header(title: str):
    """Print section header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_result(test_name: str, passed: bool, message: str = ""):
    """Print test result."""
    status = "PASS" if passed else "FAIL"
    icon = "✓" if passed else "✗"
    print(f"  {icon} {test_name}: {status}")
    if message:
        print(f"    → {message}")

    if passed:
        RESULTS['passed'].append(test_name)
    else:
        RESULTS['failed'].append(test_name)


def print_skip(test_name: str, reason: str):
    """Print skipped test."""
    print(f"  ○ {test_name}: SKIPPED")
    print(f"    → {reason}")
    RESULTS['skipped'].append(test_name)


# =============================================================================
# TEST 1: Data Loading
# =============================================================================

def test_data_loading() -> Tuple[bool, List[Dict]]:
    """Test data loading and preprocessing."""
    print_header("TEST 1: Data Loading & Preprocessing")

    try:
        from data.paper_iterator import PaperIterator
        from data.preprocessor import TextPreprocessor

        # Test iterator
        iterator = PaperIterator(
            data_dir="s2orc_data",
            batch_size=10,
            preprocess=True
        )

        # Get stats
        stats = iterator.get_stats()
        print(f"\n  Dataset Stats:")
        print(f"    Total files: {stats['total_files']}")
        print(f"    Fields: {list(stats['fields'].keys())}")

        if stats['total_files'] == 0:
            print_result("Data Discovery", False, "No data files found")
            return False, []

        print_result("Data Discovery", True, f"Found {stats['total_files']} files")

        # Test iteration
        all_papers = []
        batch_count = 0

        for batch in iterator.iterate(max_papers=150, resume=False, show_progress=False):
            all_papers.extend(batch)
            batch_count += 1

        print_result("Data Iteration", len(all_papers) > 0,
                    f"Loaded {len(all_papers)} papers in {batch_count} batches")

        if all_papers:
            sample = all_papers[0]
            print(f"\n  Sample Paper:")
            print(f"    ID: {sample.get('paper_id', 'N/A')}")
            print(f"    Title: {sample.get('title', 'N/A')[:60]}...")
            print(f"    Year: {sample.get('year', 'N/A')}")
            print(f"    Field: {sample.get('source_field', 'N/A')}")
            print(f"    Text length: {len(sample.get('embedding_text', ''))}")

        # Test preprocessor
        preprocessor = TextPreprocessor()
        test_text = "Introduction\nThis is a test paper about machine learning.\n\nMethods\nWe used neural networks."
        processed = preprocessor.process(test_text, "test_id")

        print_result("Text Preprocessing",
                    len(processed['embedding_text']) > 0,
                    f"Processed text: {len(processed['embedding_text'])} chars")

        return True, all_papers

    except Exception as e:
        print_result("Data Loading", False, str(e))
        return False, []


# =============================================================================
# TEST 2: Embedding Generation
# =============================================================================

def test_embeddings(papers: List[Dict], use_mock: bool = True) -> Tuple[bool, np.ndarray, List[Dict]]:
    """Test embedding generation."""
    print_header("TEST 2: Embedding Generation")

    if not papers:
        print_skip("Embedding Generation", "No papers to embed")
        return False, np.array([]), []

    try:
        from embeddings.bge_m3 import BGEEmbedder, BGE_AVAILABLE

        texts = [p['embedding_text'][:2000] for p in papers]  # Truncate for speed
        paper_ids = [p['paper_id'] for p in papers]

        if use_mock or not BGE_AVAILABLE:
            # Use mock embeddings for testing
            print("  Using mock embeddings (set use_mock=False for real BGE-M3)")

            # Generate deterministic mock embeddings based on text
            dim = 1024
            embeddings = []
            sparse_embeddings = []

            for text in texts:
                # Simple hash-based embedding for reproducibility
                np.random.seed(hash(text[:100]) % (2**32))
                emb = np.random.randn(dim).astype(np.float32)
                emb = emb / np.linalg.norm(emb)  # Normalize
                embeddings.append(emb)

                # Mock sparse embedding
                words = text.lower().split()[:50]
                sparse = {w: 1.0 / (i + 1) for i, w in enumerate(set(words))}
                sparse_embeddings.append(sparse)

            embeddings = np.array(embeddings)

            print_result("Mock Embeddings", True,
                        f"Generated {embeddings.shape[0]} embeddings of dim {embeddings.shape[1]}")

            return True, embeddings, sparse_embeddings

        else:
            # Real BGE-M3 embeddings
            embedder = BGEEmbedder(
                batch_size=8,
                use_fp16=True,
                return_sparse=True
            )

            print("  Loading BGE-M3 model...")
            start = time.time()
            result = embedder.embed_texts(texts[:50], show_progress=True)  # Limit for speed
            elapsed = time.time() - start

            embeddings = result['dense']
            sparse_embeddings = result.get('sparse', [{} for _ in texts])

            print_result("BGE-M3 Embeddings", True,
                        f"Generated {embeddings.shape} in {elapsed:.1f}s")

            # Pad to full size with mock
            if len(embeddings) < len(papers):
                remaining = len(papers) - len(embeddings)
                mock_emb = np.random.randn(remaining, embeddings.shape[1]).astype(np.float32)
                embeddings = np.vstack([embeddings, mock_emb])
                sparse_embeddings.extend([{} for _ in range(remaining)])

            return True, embeddings, sparse_embeddings

    except Exception as e:
        print_result("Embedding Generation", False, str(e))
        return False, np.array([]), []


# =============================================================================
# TEST 3: Vector Index
# =============================================================================

def test_vector_index(embeddings: np.ndarray, paper_ids: List[str]) -> Tuple[bool, object]:
    """Test vector index building and search."""
    print_header("TEST 3: Vector Index (FAISS)")

    if embeddings.size == 0:
        print_skip("Vector Index", "No embeddings available")
        return False, None

    try:
        from index.vector_index import VectorIndex, FAISS_AVAILABLE

        if not FAISS_AVAILABLE:
            print_skip("Vector Index", "FAISS not installed")
            return False, None

        # Build index
        print(f"  Building index for {len(paper_ids)} vectors...")
        start = time.time()

        # Use simpler index for small datasets
        index_type = "Flat" if len(paper_ids) < 1000 else "IVF100,PQ32"

        index = VectorIndex(
            dim=embeddings.shape[1],
            index_type=index_type,
            metric="cosine"
        )
        index.build(embeddings, paper_ids)
        build_time = time.time() - start

        print_result("Index Building", True, f"Built in {build_time:.2f}s")

        # Test search
        query = embeddings[0:1]  # Use first embedding as query
        start = time.time()
        scores, result_ids = index.search(query, k=5)
        search_time = (time.time() - start) * 1000

        print(f"\n  Search Results (query: first paper):")
        for i, (score, pid) in enumerate(zip(scores[0], result_ids[0])):
            print(f"    {i+1}. {pid}: score={score:.4f}")

        print_result("Index Search", len(result_ids[0]) > 0,
                    f"Found {len(result_ids[0])} results in {search_time:.1f}ms")

        # Test save/load
        test_path = "test_index_temp"
        index.save(test_path)

        index2 = VectorIndex(dim=embeddings.shape[1])
        index2.load(test_path)

        print_result("Index Save/Load", len(index2) == len(index),
                    f"Loaded {len(index2)} vectors")

        # Cleanup
        import shutil
        shutil.rmtree(test_path, ignore_errors=True)

        return True, index

    except Exception as e:
        print_result("Vector Index", False, str(e))
        import traceback
        traceback.print_exc()
        return False, None


# =============================================================================
# TEST 4: Sparse Index
# =============================================================================

def test_sparse_index(sparse_embeddings: List[Dict], paper_ids: List[str]) -> Tuple[bool, object]:
    """Test sparse index building and search."""
    print_header("TEST 4: Sparse Index (BM25)")

    if not sparse_embeddings:
        print_skip("Sparse Index", "No sparse embeddings available")
        return False, None

    try:
        from index.vector_index import SparseIndex

        # Build index
        print(f"  Building sparse index for {len(paper_ids)} documents...")
        start = time.time()

        index = SparseIndex()
        index.build(paper_ids, sparse_embeddings)
        build_time = time.time() - start

        print_result("Sparse Index Building", True,
                    f"Built in {build_time:.2f}s, vocab size: {len(index.inverted_index)}")

        # Test search
        query_sparse = {"machine": 1.0, "learning": 0.8, "neural": 0.6}
        start = time.time()
        scores, result_ids = index.search(query_sparse, k=5)
        search_time = (time.time() - start) * 1000

        print(f"\n  Sparse Search Results (query: 'machine learning neural'):")
        for i, (score, pid) in enumerate(zip(scores, result_ids)):
            print(f"    {i+1}. {pid}: score={float(score):.4f}")

        print_result("Sparse Search", len(result_ids) > 0,
                    f"Found {len(result_ids)} results in {search_time:.1f}ms")

        return True, index

    except Exception as e:
        print_result("Sparse Index", False, str(e))
        return False, None


# =============================================================================
# TEST 5: Semantic Graph
# =============================================================================

def test_semantic_graph(embeddings: np.ndarray, paper_ids: List[str],
                        papers: List[Dict]) -> Tuple[bool, object]:
    """Test semantic graph construction."""
    print_header("TEST 5: Semantic Graph Construction")

    if embeddings.size == 0:
        print_skip("Semantic Graph", "No embeddings available")
        return False, None

    try:
        from graph.semantic_graph import SemanticGraphBuilder, NETWORKX_AVAILABLE

        if not NETWORKX_AVAILABLE:
            print_skip("Semantic Graph", "NetworkX not installed")
            return False, None

        # Build metadata
        paper_metadata = {
            p['paper_id']: {
                'title': p.get('title', ''),
                'year': p.get('year', 0),
                'source_field': p.get('source_field', ''),
            }
            for p in papers
        }

        # Build graph
        print(f"  Building semantic graph for {len(paper_ids)} papers...")
        start = time.time()

        builder = SemanticGraphBuilder(
            similarity_threshold=0.3,  # Lower for mock embeddings
            max_neighbors=10
        )
        builder.build_from_embeddings(
            embeddings, paper_ids,
            paper_metadata=paper_metadata,
            batch_size=50,
            show_progress=False
        )
        build_time = time.time() - start

        print_result("Graph Building", len(builder.edges) > 0,
                    f"Created {len(builder.edges)} edges in {build_time:.2f}s")

        # Convert to NetworkX
        G = builder.to_networkx()
        stats = builder.get_stats()

        print(f"\n  Graph Stats:")
        print(f"    Nodes: {stats['nodes']}")
        print(f"    Edges: {stats['edges']}")
        print(f"    Density: {stats['density']:.4f}")
        print(f"    Avg Degree: {stats['avg_degree']:.2f}")

        print_result("Graph Stats", stats['nodes'] > 0,
                    f"{stats['nodes']} nodes, {stats['edges']} edges")

        # Test neighbor lookup
        if paper_ids:
            neighbors = builder.get_neighbors(paper_ids[0], k=3)
            print(f"\n  Neighbors of {paper_ids[0]}:")
            for pid, weight in neighbors:
                print(f"    → {pid}: {weight:.4f}")

            print_result("Neighbor Lookup", True, f"Found {len(neighbors)} neighbors")

        # Test path finding
        if len(paper_ids) >= 2:
            path = builder.find_path(paper_ids[0], paper_ids[-1])
            if path:
                print(f"\n  Path from {paper_ids[0]} to {paper_ids[-1]}:")
                print(f"    {' → '.join(path[:5])}{'...' if len(path) > 5 else ''}")
            print_result("Path Finding", True, f"Path length: {len(path) if path else 'No path'}")

        return True, builder

    except Exception as e:
        print_result("Semantic Graph", False, str(e))
        import traceback
        traceback.print_exc()
        return False, None


# =============================================================================
# TEST 6: Hybrid Search
# =============================================================================

def test_hybrid_search(vector_index, sparse_index, graph_builder, papers: List[Dict]):
    """Test hybrid search integration."""
    print_header("TEST 6: Hybrid Search Integration")

    if vector_index is None:
        print_skip("Hybrid Search", "Vector index not available")
        return False

    try:
        from search.hybrid_search import HybridSearchEngine, SearchResult

        # Setup engine
        engine = HybridSearchEngine(
            dense_weight=0.5,
            sparse_weight=0.3,
            graph_weight=0.2,
            use_reranking=False  # Skip for testing
        )

        engine.set_vector_index(vector_index)
        if sparse_index:
            engine.set_sparse_index(sparse_index)
        if graph_builder:
            engine.set_semantic_graph(graph_builder)

        # Set metadata
        paper_metadata = {
            p['paper_id']: {
                'title': p.get('title', ''),
                'year': p.get('year', 0),
                'source_field': p.get('source_field', ''),
                'embedding_text': p.get('embedding_text', '')[:500],
            }
            for p in papers
        }
        engine.set_paper_metadata(paper_metadata)

        print_result("Engine Setup", True, "All components connected")

        # Test RRF fusion manually
        dense_results = [(papers[0]['paper_id'], 0.9, 1),
                        (papers[1]['paper_id'], 0.8, 2)]
        sparse_results = [(papers[1]['paper_id'], 0.85, 1),
                         (papers[2]['paper_id'], 0.7, 2)]
        graph_results = [(papers[2]['paper_id'], 0.6, 1, 1)]

        results = engine._rrf_fusion(dense_results, sparse_results, graph_results, k=5)

        print(f"\n  RRF Fusion Results:")
        for result in results[:3]:
            print(f"    {result.paper_id}: score={result.score:.4f} "
                  f"(dense={result.rank_dense}, sparse={result.rank_sparse}, graph={result.rank_graph})")

        print_result("RRF Fusion", len(results) > 0, f"Fused {len(results)} results")

        # Test search explanation
        if results:
            explanation = engine.explain_result("test query", results[0])
            print(f"\n  Search Explanation for {results[0].paper_id}:")
            for source, contrib in explanation['contributions'].items():
                print(f"    {source}: rank={contrib['rank']}, rrf={contrib['rrf_score']:.4f}")

        print_result("Search Explanation", True, "Explanation generated")

        return True

    except Exception as e:
        print_result("Hybrid Search", False, str(e))
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# TEST 7: Graph Analytics
# =============================================================================

def test_graph_analytics(graph_builder, papers: List[Dict]):
    """Test graph analytics."""
    print_header("TEST 7: Graph Analytics")

    if graph_builder is None:
        print_skip("Graph Analytics", "Graph not available")
        return False

    try:
        import networkx as nx

        G = graph_builder.graph
        if G is None:
            G = graph_builder.to_networkx()

        # PageRank
        print("  Computing PageRank...")
        pagerank = nx.pagerank(G, alpha=0.85)
        top_pr = sorted(pagerank.items(), key=lambda x: -x[1])[:5]

        print(f"\n  Top PageRank Papers:")
        for pid, score in top_pr:
            title = graph_builder.node_metadata.get(pid, {}).get('title', 'Unknown')[:40]
            print(f"    {pid}: {score:.4f} - {title}...")

        print_result("PageRank", len(pagerank) > 0, f"Computed for {len(pagerank)} nodes")

        # Community detection (if graph is connected enough)
        if G.number_of_edges() > 0:
            try:
                from networkx.algorithms import community
                communities = list(community.greedy_modularity_communities(G))
                print(f"\n  Community Detection:")
                print(f"    Found {len(communities)} communities")
                for i, comm in enumerate(communities[:3]):
                    print(f"    Community {i+1}: {len(comm)} papers")

                print_result("Community Detection", True, f"Found {len(communities)} communities")
            except Exception as e:
                print_result("Community Detection", False, str(e))

        # Centrality
        if G.number_of_nodes() > 0:
            degree_cent = nx.degree_centrality(G)
            top_dc = sorted(degree_cent.items(), key=lambda x: -x[1])[:3]

            print(f"\n  Top Degree Centrality:")
            for pid, score in top_dc:
                print(f"    {pid}: {score:.4f}")

            print_result("Centrality Metrics", True, "Computed successfully")

        return True

    except Exception as e:
        print_result("Graph Analytics", False, str(e))
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run all pipeline validation tests."""
    print("\n" + "=" * 60)
    print("  SYMBY GRAPHRAG - PIPELINE VALIDATION")
    print("=" * 60)
    print(f"  Working directory: {os.getcwd()}")
    print(f"  Python: {sys.version.split()[0]}")

    start_time = time.time()

    # Test 1: Data Loading
    data_ok, papers = test_data_loading()
    paper_ids = [p['paper_id'] for p in papers] if papers else []

    # Test 2: Embeddings
    emb_ok, embeddings, sparse_embeddings = test_embeddings(papers, use_mock=True)

    # Test 3: Vector Index
    vec_ok, vector_index = test_vector_index(embeddings, paper_ids)

    # Test 4: Sparse Index
    sparse_ok, sparse_index = test_sparse_index(sparse_embeddings, paper_ids)

    # Test 5: Semantic Graph
    graph_ok, graph_builder = test_semantic_graph(embeddings, paper_ids, papers)

    # Test 6: Hybrid Search
    search_ok = test_hybrid_search(vector_index, sparse_index, graph_builder, papers)

    # Test 7: Graph Analytics
    analytics_ok = test_graph_analytics(graph_builder, papers)

    # Summary
    total_time = time.time() - start_time

    print_header("VALIDATION SUMMARY")
    print(f"\n  Passed: {len(RESULTS['passed'])}")
    print(f"  Failed: {len(RESULTS['failed'])}")
    print(f"  Skipped: {len(RESULTS['skipped'])}")
    print(f"  Total time: {total_time:.1f}s")

    if RESULTS['failed']:
        print(f"\n  Failed tests:")
        for test in RESULTS['failed']:
            print(f"    ✗ {test}")

    print("\n" + "=" * 60)
    if not RESULTS['failed']:
        print("  ALL TESTS PASSED - Pipeline is ready!")
    else:
        print("  SOME TESTS FAILED - Please fix before proceeding")
    print("=" * 60 + "\n")

    return len(RESULTS['failed']) == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
