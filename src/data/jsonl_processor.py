"""
JSONL Processor for Symby AI Hackathon Data
Reads simple JSONL files and produces enriched evaluations
"""

import json
import gzip
from pathlib import Path
from typing import Generator, Dict, List, Optional
from tqdm import tqdm
from paper_evaluator import PaperEvaluator


class JSONLProcessor:
    """
    Process JSONL files from Symby_AI directory and generate evaluations.

    Input format: {"id": "", "text": "", "field": "", "publication_date": 2020, "matched_term": ""}
    Output format: Full evaluation structure from evaluation_structure.pdf
    """

    def __init__(self, data_dir: str = "/home/asus/symby/Symby_AI"):
        self.data_dir = Path(data_dir)
        self.evaluator = PaperEvaluator()

    def read_jsonl(self, filepath: Path, max_papers: Optional[int] = None) -> Generator[Dict, None, None]:
        """
        Read a JSONL file (supports both .jsonl and .jsonl.gz).

        Args:
            filepath: Path to JSONL file
            max_papers: Maximum number of papers to read (None = all)

        Yields:
            Paper dicts
        """
        is_gzipped = str(filepath).endswith('.gz')
        count = 0

        try:
            if is_gzipped:
                with gzip.open(filepath, 'rt', encoding='utf-8') as f:
                    for line in f:
                        if max_papers and count >= max_papers:
                            break
                        try:
                            paper = json.loads(line.strip())
                            yield paper
                            count += 1
                        except json.JSONDecodeError as e:
                            print(f"Error decoding JSON: {e}")
                            continue
            else:
                with open(filepath, 'r', encoding='utf-8') as f:
                    for line in f:
                        if max_papers and count >= max_papers:
                            break
                        try:
                            paper = json.loads(line.strip())
                            yield paper
                            count += 1
                        except json.JSONDecodeError as e:
                            print(f"Error decoding JSON: {e}")
                            continue
        except Exception as e:
            print(f"Error reading {filepath}: {e}")

    def process_file(self,
                     input_file: str,
                     output_file: Optional[str] = None,
                     max_papers: Optional[int] = None,
                     show_progress: bool = True) -> List[Dict]:
        """
        Process a JSONL file and generate evaluations.

        Args:
            input_file: Path to input JSONL file (relative to data_dir)
            output_file: Path to output JSONL file (optional)
            max_papers: Maximum number of papers to process
            show_progress: Show progress bar

        Returns:
            List of evaluated papers
        """
        input_path = self.data_dir / input_file
        if not input_path.exists():
            # Try without data_dir prefix
            input_path = Path(input_file)

        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        print(f"Processing: {input_path}")

        # Read papers
        papers = list(self.read_jsonl(input_path, max_papers))
        print(f"Read {len(papers)} papers")

        # Evaluate papers
        evaluations = []
        iterator = tqdm(papers, desc="Evaluating papers") if show_progress else papers

        for paper in iterator:
            try:
                evaluation = self.evaluator.evaluate_paper(paper)
                evaluations.append(evaluation)
            except Exception as e:
                print(f"Error evaluating paper {paper.get('id', 'unknown')}: {e}")
                continue

        # Save to output file if specified
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w', encoding='utf-8') as f:
                for evaluation in evaluations:
                    f.write(json.dumps(evaluation) + '\n')

            print(f"Saved {len(evaluations)} evaluations to: {output_path}")

        return evaluations

    def process_all_datasets(self,
                            max_papers_per_file: Optional[int] = None,
                            output_dir: Optional[str] = None) -> Dict[str, List[Dict]]:
        """
        Process all datasets (ml, nonml, validation).

        Args:
            max_papers_per_file: Maximum papers to process per file
            output_dir: Directory to save outputs

        Returns:
            Dict mapping dataset name to evaluations
        """
        datasets = {
            'ml': 'modded_s2orc_ml.jsonl',
            'nonml': 'modded_s2orc_nonml.jsonl',
            'validation': 'modded_s2orc_validation.jsonl',
        }

        results = {}

        for name, filename in datasets.items():
            print(f"\n{'='*60}")
            print(f"Processing {name} dataset")
            print(f"{'='*60}")

            try:
                output_file = None
                if output_dir:
                    output_file = Path(output_dir) / f"{name}_evaluated.jsonl"

                evaluations = self.process_file(
                    filename,
                    output_file=output_file,
                    max_papers=max_papers_per_file
                )

                results[name] = evaluations

                # Print summary statistics
                self.print_summary(name, evaluations)

            except FileNotFoundError as e:
                print(f"Skipping {name}: {e}")
                continue

        return results

    def print_summary(self, dataset_name: str, evaluations: List[Dict]):
        """Print summary statistics for a dataset"""
        if not evaluations:
            print(f"No evaluations for {dataset_name}")
            return

        total = len(evaluations)

        # ML adoption levels
        ml_levels = {}
        for e in evaluations:
            level = e['ml_adoption']['ml_adoption_level']
            ml_levels[level] = ml_levels.get(level, 0) + 1

        # Reproducibility
        code_available = sum(1 for e in evaluations
                            if e['reproducibility']['code_availability_mentioned'])
        data_available = sum(1 for e in evaluations
                            if e['reproducibility']['data_availability_mentioned'])

        # Impact
        novel_claims = sum(1 for e in evaluations
                          if e['impact_indicators']['claims_novelty'])

        print(f"\n{dataset_name} Summary ({total} papers):")
        print(f"  ML Adoption Levels:")
        for level, count in sorted(ml_levels.items()):
            print(f"    {level}: {count} ({count/total*100:.1f}%)")
        print(f"  Code Available: {code_available} ({code_available/total*100:.1f}%)")
        print(f"  Data Available: {data_available} ({data_available/total*100:.1f}%)")
        print(f"  Novelty Claims: {novel_claims} ({novel_claims/total*100:.1f}%)")

    def get_statistics(self, evaluations: List[Dict]) -> Dict:
        """
        Get detailed statistics from evaluations.

        Args:
            evaluations: List of evaluated papers

        Returns:
            Dict with statistics
        """
        if not evaluations:
            return {}

        total = len(evaluations)

        stats = {
            'total_papers': total,
            'ml_adoption': {},
            'reproducibility': {},
            'impact': {},
            'outcomes': {},
        }

        # ML adoption distribution
        ml_levels = {}
        for e in evaluations:
            level = e['ml_adoption']['ml_adoption_level']
            ml_levels[level] = ml_levels.get(level, 0) + 1

        stats['ml_adoption']['levels'] = ml_levels
        stats['ml_adoption']['ml_papers'] = total - ml_levels.get('none', 0)
        stats['ml_adoption']['ml_rate'] = stats['ml_adoption']['ml_papers'] / total

        # Most common frameworks
        frameworks = {}
        for e in evaluations:
            for fw in e['ml_adoption']['ml_frameworks_mentioned']:
                frameworks[fw] = frameworks.get(fw, 0) + 1
        stats['ml_adoption']['top_frameworks'] = sorted(frameworks.items(),
                                                        key=lambda x: x[1],
                                                        reverse=True)[:5]

        # Reproducibility stats
        stats['reproducibility']['code_available'] = sum(
            1 for e in evaluations if e['reproducibility']['code_availability_mentioned']
        )
        stats['reproducibility']['data_available'] = sum(
            1 for e in evaluations if e['reproducibility']['data_availability_mentioned']
        )
        stats['reproducibility']['code_rate'] = stats['reproducibility']['code_available'] / total
        stats['reproducibility']['data_rate'] = stats['reproducibility']['data_available'] / total

        # Impact stats
        stats['impact']['novelty_claims'] = sum(
            1 for e in evaluations if e['impact_indicators']['claims_novelty']
        )
        stats['impact']['improvement_claims'] = sum(
            1 for e in evaluations if e['impact_indicators']['claims_improvement_over_existing']
        )

        impact_scopes = {}
        for e in evaluations:
            scope = e['impact_indicators']['potential_impact_scope']
            impact_scopes[scope] = impact_scopes.get(scope, 0) + 1
        stats['impact']['scopes'] = impact_scopes

        # Research outcomes
        stats['outcomes']['clinical_trials'] = sum(
            1 for e in evaluations if e['research_outcomes']['mentions_clinical_trials']
        )
        stats['outcomes']['real_world_apps'] = sum(
            1 for e in evaluations if e['research_outcomes']['real_world_application_mentioned']
        )

        return stats


if __name__ == "__main__":
    # Example usage
    processor = JSONLProcessor()

    # Process a sample from validation set
    print("Processing validation dataset (first 100 papers)...")
    evaluations = processor.process_file(
        'modded_s2orc_validation.jsonl',
        output_file='outputs/validation_sample_evaluated.jsonl',
        max_papers=100
    )

    # Get statistics
    stats = processor.get_statistics(evaluations)

    print("\n" + "="*60)
    print("STATISTICS")
    print("="*60)
    print(f"Total Papers: {stats['total_papers']}")
    print(f"ML Papers: {stats['ml_adoption']['ml_papers']} ({stats['ml_adoption']['ml_rate']:.1%})")
    print(f"Code Availability: {stats['reproducibility']['code_rate']:.1%}")
    print(f"Data Availability: {stats['reproducibility']['data_rate']:.1%}")
    print(f"Novelty Claims: {stats['impact']['novelty_claims']}")
    print(f"\nTop Frameworks: {stats['ml_adoption']['top_frameworks']}")
