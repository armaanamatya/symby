#!/usr/bin/env python3
"""
Data Validation Script for Full Hybrid Semantic-Graph Search System

Validates:
- Data file integrity
- Text preprocessing quality
- Section extraction accuracy
- Dataset statistics
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.paper_iterator import PaperIterator
from data.preprocessor import TextPreprocessor
from collections import defaultdict
import json


def validate_preprocessing(sample_size: int = 1000):
    """Validate text preprocessing on sample papers."""
    print("=" * 60)
    print("VALIDATING TEXT PREPROCESSING")
    print("=" * 60)

    iterator = PaperIterator(
        data_dir="/home/abheekp/ai_research/s2orc_data",
        batch_size=100,
        preprocess=True
    )

    stats = {
        'total': 0,
        'with_abstract': 0,
        'with_methods': 0,
        'with_results': 0,
        'avg_text_length': 0,
        'avg_embedding_length': 0,
        'years': defaultdict(int),
        'fields': defaultdict(int),
        'section_counts': defaultdict(int),
    }

    print(f"\nProcessing {sample_size} papers...")

    for batch in iterator.iterate(max_papers=sample_size, resume=False, show_progress=True):
        for paper in batch:
            stats['total'] += 1

            # Count sections
            sections = paper.get('sections_found', [])
            for section in sections:
                stats['section_counts'][section] += 1

            if 'abstract' in sections:
                stats['with_abstract'] += 1
            if 'methods' in sections:
                stats['with_methods'] += 1
            if 'results' in sections:
                stats['with_results'] += 1

            stats['avg_text_length'] += paper.get('text_length', 0)
            stats['avg_embedding_length'] += len(paper.get('embedding_text', ''))

            year = paper.get('year', 0)
            if year > 2000:
                stats['years'][year] += 1

            stats['fields'][paper.get('source_field', 'unknown')] += 1

    # Compute averages
    if stats['total'] > 0:
        stats['avg_text_length'] /= stats['total']
        stats['avg_embedding_length'] /= stats['total']

    # Print results
    print("\n" + "=" * 40)
    print("VALIDATION RESULTS")
    print("=" * 40)

    print(f"\nTotal papers processed: {stats['total']:,}")
    print(f"Papers with abstract: {stats['with_abstract']:,} ({100*stats['with_abstract']/stats['total']:.1f}%)")
    print(f"Papers with methods: {stats['with_methods']:,} ({100*stats['with_methods']/stats['total']:.1f}%)")
    print(f"Papers with results: {stats['with_results']:,} ({100*stats['with_results']/stats['total']:.1f}%)")

    print(f"\nAverage text length: {stats['avg_text_length']:,.0f} chars")
    print(f"Average embedding text length: {stats['avg_embedding_length']:,.0f} chars")
    print(f"Compression ratio: {100*stats['avg_embedding_length']/max(stats['avg_text_length'],1):.1f}%")

    print("\nSection counts:")
    for section, count in sorted(stats['section_counts'].items(), key=lambda x: -x[1]):
        print(f"  {section}: {count:,} ({100*count/stats['total']:.1f}%)")

    print("\nYears distribution (top 10):")
    sorted_years = sorted(stats['years'].items(), key=lambda x: -x[1])[:10]
    for year, count in sorted_years:
        print(f"  {year}: {count:,} ({100*count/stats['total']:.1f}%)")

    print("\nFields distribution:")
    for field, count in sorted(stats['fields'].items(), key=lambda x: -x[1]):
        print(f"  {field}: {count:,} ({100*count/stats['total']:.1f}%)")

    return stats


def validate_file_integrity():
    """Check all data files for basic integrity."""
    print("=" * 60)
    print("VALIDATING FILE INTEGRITY")
    print("=" * 60)

    data_dir = Path("/home/abheekp/ai_research/s2orc_data")
    issues = []
    file_count = 0
    total_size = 0

    for item in data_dir.iterdir():
        if not item.is_dir():
            continue

        for split_dir in item.iterdir():
            if not split_dir.is_dir():
                continue

            for data_file in split_dir.glob('*'):
                if data_file.suffix in ['.json', '.jsonl', '.gz']:
                    file_count += 1
                    size = data_file.stat().st_size
                    total_size += size

                    # Check for empty files
                    if size == 0:
                        issues.append(f"Empty file: {data_file}")

                    # Check for very small files
                    elif size < 100:
                        issues.append(f"Suspiciously small: {data_file} ({size} bytes)")

    print(f"\nTotal data files: {file_count:,}")
    print(f"Total size: {total_size / (1024**3):.2f} GB")

    if issues:
        print(f"\nIssues found: {len(issues)}")
        for issue in issues[:10]:
            print(f"  - {issue}")
    else:
        print("\nNo integrity issues found!")

    return file_count, total_size, issues


def show_sample_paper():
    """Show a sample preprocessed paper for manual inspection."""
    print("=" * 60)
    print("SAMPLE PREPROCESSED PAPER")
    print("=" * 60)

    iterator = PaperIterator(
        data_dir="/home/abheekp/ai_research/s2orc_data",
        batch_size=5,
        preprocess=True
    )

    for batch in iterator.iterate(max_papers=5, resume=False, show_progress=False):
        for paper in batch:
            print(f"\nPaper ID: {paper['paper_id']}")
            print(f"Title: {paper['title'][:100]}...")
            print(f"Year: {paper['year']}")
            print(f"Field: {paper['source_field']}")
            print(f"Original length: {paper['text_length']:,} chars")
            print(f"Embedding length: {len(paper['embedding_text']):,} chars")
            print(f"Sections: {paper['sections_found']}")
            print(f"\nEmbedding text preview (first 500 chars):")
            print("-" * 40)
            print(paper['embedding_text'][:500])
            print("-" * 40)
            return  # Just show one


def main():
    """Run all validations."""
    print("\n" + "=" * 60)
    print("FULL HYBRID DATA VALIDATION")
    print("=" * 60 + "\n")

    # Check file integrity
    file_count, total_size, issues = validate_file_integrity()

    # Show sample paper
    show_sample_paper()

    # Validate preprocessing
    stats = validate_preprocessing(sample_size=500)

    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Files: {file_count:,}")
    print(f"Total size: {total_size / (1024**3):.2f} GB")
    print(f"Sample papers validated: {stats['total']:,}")
    print(f"Integrity issues: {len(issues)}")
    print(f"Ready for embedding generation: {'YES' if len(issues) == 0 else 'NO - fix issues first'}")


if __name__ == "__main__":
    main()
