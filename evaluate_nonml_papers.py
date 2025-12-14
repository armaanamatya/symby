"""
Evaluate all non-ML papers and save each as individual JSON file
"""

import json
import sys
from pathlib import Path
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src' / 'data'))

from paper_evaluator import PaperEvaluator
from jsonl_processor import JSONLProcessor


def evaluate_all_nonml_papers(
    input_file: str = "Symby_AI/modded_s2orc_nonml.jsonl",
    output_dir: str = "nonml_evaluations",
    batch_size: int = 1000
):
    """
    Evaluate all non-ML papers and save each as individual JSON file.

    Args:
        input_file: Path to non-ML JSONL file
        output_dir: Directory to save individual JSON files
        batch_size: Process and save in batches
    """
    # Setup
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    processor = JSONLProcessor()
    evaluator = PaperEvaluator()

    print("="*60)
    print("EVALUATING ALL NON-ML PAPERS")
    print("="*60)
    print(f"Input: {input_file}")
    print(f"Output directory: {output_path.absolute()}")

    # Read all papers
    print("\nReading papers...")
    papers = list(processor.read_jsonl(Path(input_file)))
    total_papers = len(papers)
    print(f"Total papers to evaluate: {total_papers:,}")

    # Statistics tracking
    stats = {
        'total_processed': 0,
        'total_errors': 0,
        'ml_adoption_levels': {},
        'papers_with_code': 0,
        'papers_with_data': 0,
        'papers_with_novelty': 0,
    }

    # Process papers
    print("\nEvaluating papers...")
    errors = []

    for i, paper in enumerate(tqdm(papers, desc="Processing papers")):
        try:
            # Evaluate paper
            evaluation = evaluator.evaluate_paper(paper)

            # Generate filename from paper_id (sanitize for filesystem)
            paper_id = evaluation['paper_id']
            if not paper_id:
                paper_id = f"unknown_{i}"

            # Sanitize filename (remove invalid characters)
            safe_id = str(paper_id).replace('/', '_').replace('\\', '_')
            output_file = output_path / f"{safe_id}.json"

            # Save individual JSON file
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(evaluation, f, indent=2, ensure_ascii=False)

            # Update statistics
            stats['total_processed'] += 1

            ml_level = evaluation['ml_adoption']['ml_adoption_level']
            stats['ml_adoption_levels'][ml_level] = stats['ml_adoption_levels'].get(ml_level, 0) + 1

            if evaluation['reproducibility']['code_availability_mentioned']:
                stats['papers_with_code'] += 1

            if evaluation['reproducibility']['data_availability_mentioned']:
                stats['papers_with_data'] += 1

            if evaluation['impact_indicators']['claims_novelty']:
                stats['papers_with_novelty'] += 1

        except Exception as e:
            stats['total_errors'] += 1
            errors.append({
                'paper_id': paper.get('id', f'unknown_{i}'),
                'error': str(e)
            })

            # Print error for first few
            if stats['total_errors'] <= 5:
                print(f"\nError processing paper {paper.get('id', i)}: {e}")

    # Save summary statistics
    summary_file = output_path / "_summary.json"
    summary = {
        'total_papers': total_papers,
        'successfully_processed': stats['total_processed'],
        'errors': stats['total_errors'],
        'error_rate': stats['total_errors'] / total_papers if total_papers > 0 else 0,
        'ml_adoption_levels': stats['ml_adoption_levels'],
        'code_availability_rate': stats['papers_with_code'] / stats['total_processed'] if stats['total_processed'] > 0 else 0,
        'data_availability_rate': stats['papers_with_data'] / stats['total_processed'] if stats['total_processed'] > 0 else 0,
        'novelty_claims_rate': stats['papers_with_novelty'] / stats['total_processed'] if stats['total_processed'] > 0 else 0,
    }

    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved summary to: {summary_file}")

    # Save error log if any errors
    if errors:
        error_file = output_path / "_errors.json"
        with open(error_file, 'w', encoding='utf-8') as f:
            json.dump(errors, f, indent=2)
        print(f"Saved error log to: {error_file}")

    # Print final report
    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)
    print(f"Total papers: {total_papers:,}")
    print(f"Successfully processed: {stats['total_processed']:,}")
    print(f"Errors: {stats['total_errors']}")
    print(f"\nML Adoption Levels:")
    for level, count in sorted(stats['ml_adoption_levels'].items()):
        pct = count / stats['total_processed'] * 100 if stats['total_processed'] > 0 else 0
        print(f"  {level}: {count:,} ({pct:.1f}%)")
    print(f"\nReproducibility:")
    print(f"  Code available: {stats['papers_with_code']:,} ({summary['code_availability_rate']:.1%})")
    print(f"  Data available: {stats['papers_with_data']:,} ({summary['data_availability_rate']:.1%})")
    print(f"\nImpact:")
    print(f"  Novelty claims: {stats['papers_with_novelty']:,} ({summary['novelty_claims_rate']:.1%})")
    print(f"\nOutput directory: {output_path.absolute()}")
    print(f"Individual JSON files: {stats['total_processed']:,}")

    return summary


if __name__ == "__main__":
    # Run evaluation
    summary = evaluate_all_nonml_papers(
        input_file="Symby_AI/modded_s2orc_nonml.jsonl",
        output_dir="nonml_evaluations"
    )

    print("\n✅ All non-ML papers evaluated and saved!")
