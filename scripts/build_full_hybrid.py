#!/usr/bin/env python3
"""
Full Hybrid System Builder

Orchestrates the complete pipeline for the AI Research Impact Observatory:
1. Data preparation and validation
2. Embedding generation with BGE-M3
3. Vector index construction (FAISS)
4. Semantic graph construction
5. Hybrid search system setup
6. GraphRAG interface initialization

Run this script to build the entire system from scratch or resume from checkpoints.
"""

import sys
import os
import argparse
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def print_banner():
    """Print startup banner."""
    print("""
    ╔═══════════════════════════════════════════════════════════════════╗
    ║       AI Research Impact Observatory - Full Hybrid Builder        ║
    ║              DGX Spark Frontier Hackathon | Symby AI              ║
    ╠═══════════════════════════════════════════════════════════════════╣
    ║  Building:                                                        ║
    ║    • BGE-M3 Embeddings (Dense + Sparse)                          ║
    ║    • FAISS Vector Index (IVF-PQ)                                 ║
    ║    • Semantic Similarity Graph                                    ║
    ║    • Hybrid Search Engine (RRF Fusion)                           ║
    ║    • GraphRAG Query Interface                                     ║
    ╚═══════════════════════════════════════════════════════════════════╝
    """)


def phase1_validate_data(data_dir: str) -> bool:
    """Phase 1: Validate data files."""
    print("\n" + "=" * 60)
    print("PHASE 1: DATA VALIDATION")
    print("=" * 60)

    from data.paper_iterator import PaperIterator

    iterator = PaperIterator(data_dir, batch_size=100)
    stats = iterator.get_stats()

    print(f"  Total files: {stats['total_files']}")
    print(f"  Fields: {stats['fields']}")
    print(f"  Splits: {stats['splits']}")

    if stats['total_files'] == 0:
        print("  ERROR: No data files found!")
        return False

    print("  ✓ Data validation passed")
    return True


def phase2_generate_embeddings(
    data_dir: str,
    store_dir: str,
    max_papers: int = None,
    batch_size: int = 8,
    resume: bool = True
):
    """Phase 2: Generate embeddings using BGE-M3."""
    print("\n" + "=" * 60)
    print("PHASE 2: EMBEDDING GENERATION")
    print("=" * 60)

    try:
        from embeddings.bge_m3 import EmbeddingPipeline

        pipeline = EmbeddingPipeline(
            data_dir=data_dir,
            store_dir=store_dir,
            batch_size=batch_size,
            use_fp16=True,
            return_sparse=True,
        )

        pipeline.run(
            max_papers=max_papers,
            resume=resume,
            checkpoint_every=1000,
        )

        print(f"  ✓ Generated embeddings for {len(pipeline.store)} papers")
        return pipeline.store

    except ImportError as e:
        print(f"  WARNING: BGE-M3 not available ({e})")
        print("  Install with: pip install FlagEmbedding")
        return None


def phase3_build_vector_index(
    store_dir: str,
    index_dir: str,
    index_type: str = "IVF4096,PQ64"
):
    """Phase 3: Build FAISS vector index."""
    print("\n" + "=" * 60)
    print("PHASE 3: VECTOR INDEX CONSTRUCTION")
    print("=" * 60)

    from embeddings.bge_m3 import EmbeddingStore
    from index.vector_index import VectorIndex, SparseIndex, HybridIndex

    # Load embeddings
    store = EmbeddingStore(store_dir)

    if len(store) == 0:
        print("  ERROR: No embeddings found in store!")
        return None

    print(f"  Loaded {len(store)} embeddings")

    # Get all embeddings
    embeddings = store.get_all_dense_embeddings()
    paper_ids = store.paper_ids

    # Build dense index
    print(f"  Building FAISS index ({index_type})...")
    dense_index = VectorIndex(
        dim=embeddings.shape[1],
        index_type=index_type,
        metric="cosine"
    )
    dense_index.build(embeddings, paper_ids)

    # Build sparse index if available
    sparse_index = None
    if store.sparse_embeddings:
        print("  Building sparse index...")
        sparse_index = SparseIndex()
        sparse_index.build(paper_ids, store.sparse_embeddings)

    # Build hybrid index
    print("  Building hybrid index...")
    hybrid = HybridIndex(dim=embeddings.shape[1])
    hybrid.dense_index = dense_index
    if sparse_index:
        hybrid.sparse_index = sparse_index

    # Save
    hybrid.save(index_dir)
    print(f"  ✓ Index saved to {index_dir}")

    return hybrid


def phase4_build_semantic_graph(
    store_dir: str,
    graph_dir: str,
    index_dir: str,
    similarity_threshold: float = 0.75,
    max_neighbors: int = 20
):
    """Phase 4: Build semantic similarity graph."""
    print("\n" + "=" * 60)
    print("PHASE 4: SEMANTIC GRAPH CONSTRUCTION")
    print("=" * 60)

    from embeddings.bge_m3 import EmbeddingStore
    from graph.semantic_graph import SemanticGraphBuilder
    import numpy as np

    # Load embeddings
    store = EmbeddingStore(store_dir)
    embeddings = store.get_all_dense_embeddings()
    paper_ids = store.paper_ids

    if len(embeddings) == 0:
        print("  ERROR: No embeddings found!")
        return None

    print(f"  Building graph for {len(paper_ids)} papers...")
    print(f"  Similarity threshold: {similarity_threshold}")
    print(f"  Max neighbors: {max_neighbors}")

    builder = SemanticGraphBuilder(
        similarity_threshold=similarity_threshold,
        max_neighbors=max_neighbors
    )

    builder.build_from_embeddings(
        embeddings,
        paper_ids,
        batch_size=500,
        show_progress=True
    )

    # Get stats
    G = builder.to_networkx()
    stats = builder.get_stats()

    print(f"  Nodes: {stats['nodes']:,}")
    print(f"  Edges: {stats['edges']:,}")
    print(f"  Avg degree: {stats['avg_degree']:.1f}")
    print(f"  Components: {stats.get('connected_components', 'N/A')}")

    # Save
    builder.save(graph_dir)
    print(f"  ✓ Graph saved to {graph_dir}")

    return builder


def phase5_setup_search_engine(
    index_dir: str,
    graph_dir: str,
    data_dir: str
) -> dict:
    """Phase 5: Setup hybrid search engine."""
    print("\n" + "=" * 60)
    print("PHASE 5: SEARCH ENGINE SETUP")
    print("=" * 60)

    from index.vector_index import HybridIndex
    from graph.semantic_graph import SemanticGraphBuilder
    from search.hybrid_search import HybridSearchEngine
    from data.loader import DataLoader

    # Load components
    print("  Loading vector index...")
    hybrid_index = HybridIndex()
    hybrid_index.load(index_dir)

    print("  Loading semantic graph...")
    graph_builder = SemanticGraphBuilder()
    graph_builder.load(graph_dir)

    print("  Loading paper metadata...")
    loader = DataLoader(data_dir)
    # Load a sample to get metadata structure
    df = loader.load_sample(10000)
    paper_metadata = {}
    for _, row in df.iterrows():
        paper_metadata[row['id']] = row.to_dict()

    # Setup search engine
    print("  Configuring search engine...")
    search_engine = HybridSearchEngine(
        dense_weight=0.5,
        sparse_weight=0.3,
        graph_weight=0.2,
    )

    search_engine.set_vector_index(hybrid_index)
    search_engine.set_semantic_graph(graph_builder)
    search_engine.set_paper_metadata(paper_metadata)

    print("  ✓ Search engine configured")

    return {
        'search_engine': search_engine,
        'hybrid_index': hybrid_index,
        'semantic_graph': graph_builder,
        'paper_metadata': paper_metadata,
    }


def phase6_setup_graphrag(
    search_engine,
    graph_builder,
    paper_metadata: dict
):
    """Phase 6: Setup GraphRAG interface."""
    print("\n" + "=" * 60)
    print("PHASE 6: GRAPHRAG INTERFACE SETUP")
    print("=" * 60)

    from rag.graphrag_interface import FullHybridRAG
    from graph.graph_analysis import GraphAnalyzer

    # Create graph analyzer from semantic graph
    print("  Setting up graph analyzer...")
    nodes = {pid: meta for pid, meta in paper_metadata.items()}
    edges = [(e[0], e[1]) for e in graph_builder.edges]
    graph_analyzer = GraphAnalyzer(nodes, edges)

    # Setup RAG
    print("  Configuring GraphRAG...")
    rag = FullHybridRAG(
        search_engine=search_engine,
        graph_analyzer=graph_analyzer,
        semantic_graph=graph_builder,
        paper_metadata=paper_metadata,
    )

    # Test query
    print("\n  Testing with sample query...")
    result = rag.query("How has machine learning impacted drug discovery?")
    print(f"    Query type: {result.query_type}")
    print(f"    Answer: {result.answer[:200]}...")
    print(f"    Confidence: {result.confidence}")

    print("  ✓ GraphRAG interface ready")

    return rag


def main():
    parser = argparse.ArgumentParser(
        description="Build Full Hybrid Semantic-Graph Search System"
    )
    parser.add_argument(
        "--data-dir",
        default="s2orc_data",
        help="Path to S2ORC data directory"
    )
    parser.add_argument(
        "--output-dir",
        default="full_hybrid_output",
        help="Directory for output files"
    )
    parser.add_argument(
        "--max-papers",
        type=int,
        default=None,
        help="Maximum papers to process (None = all)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for embedding generation"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Resume from checkpoints"
    )
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Skip embedding generation (use existing)"
    )
    parser.add_argument(
        "--skip-index",
        action="store_true",
        help="Skip index building (use existing)"
    )
    parser.add_argument(
        "--skip-graph",
        action="store_true",
        help="Skip graph building (use existing)"
    )

    args = parser.parse_args()

    print_banner()
    start_time = time.time()

    # Setup directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    store_dir = str(output_dir / "embeddings")
    index_dir = str(output_dir / "index")
    graph_dir = str(output_dir / "graph")

    # Phase 1: Validate data
    if not phase1_validate_data(args.data_dir):
        sys.exit(1)

    # Phase 2: Generate embeddings
    if not args.skip_embeddings:
        store = phase2_generate_embeddings(
            args.data_dir,
            store_dir,
            args.max_papers,
            args.batch_size,
            args.resume
        )
        if store is None:
            print("  Skipping embedding-dependent phases")
            sys.exit(0)

    # Phase 3: Build vector index
    if not args.skip_index:
        hybrid_index = phase3_build_vector_index(
            store_dir,
            index_dir
        )

    # Phase 4: Build semantic graph
    if not args.skip_graph:
        graph_builder = phase4_build_semantic_graph(
            store_dir,
            graph_dir,
            index_dir
        )

    # Phase 5: Setup search engine
    components = phase5_setup_search_engine(
        index_dir,
        graph_dir,
        args.data_dir
    )

    # Phase 6: Setup GraphRAG
    rag = phase6_setup_graphrag(
        components['search_engine'],
        components['semantic_graph'],
        components['paper_metadata']
    )

    # Summary
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("BUILD COMPLETE")
    print("=" * 60)
    print(f"  Total time: {elapsed / 60:.1f} minutes")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Embeddings: {store_dir}")
    print(f"  Index: {index_dir}")
    print(f"  Graph: {graph_dir}")
    print()
    print("  To run the dashboard with the new system:")
    print(f"    streamlit run app.py")
    print()


if __name__ == "__main__":
    main()
