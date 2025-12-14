"""
Validation Script for AI Research Impact Observatory
Tests evaluation metrics against validation dataset
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src' / 'data'))

from jsonl_processor import JSONLProcessor
from paper_evaluator import PaperEvaluator


class EvaluationValidator:
    """
    Validate paper evaluations against ground truth.

    Compares generated evaluations with validation dataset to measure accuracy.
    """

    def __init__(self):
        self.evaluator = PaperEvaluator()
        self.processor = JSONLProcessor()

    def load_validation_data(self, filepath: str) -> List[Dict]:
        """
        Load validation data (can be pre-evaluated or raw).

        Args:
            filepath: Path to validation JSONL file

        Returns:
            List of validation papers
        """
        papers = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    paper = json.loads(line.strip())
                    papers.append(paper)
                except json.JSONDecodeError:
                    continue
        return papers

    def is_evaluated(self, paper: Dict) -> bool:
        """Check if paper has evaluation structure"""
        return 'ml_adoption' in paper and isinstance(paper['ml_adoption'], dict)

    def validate_ml_adoption(self, predicted: Dict, ground_truth: Dict) -> Dict:
        """
        Validate ML adoption predictions.

        Args:
            predicted: Predicted evaluation
            ground_truth: Ground truth evaluation

        Returns:
            Dict with accuracy metrics
        """
        metrics = {}

        pred_ml = predicted['ml_adoption']
        true_ml = ground_truth['ml_adoption']

        # Adoption level accuracy
        metrics['adoption_level_match'] = (
            pred_ml['ml_adoption_level'] == true_ml['ml_adoption_level']
        )

        # Framework detection (Jaccard similarity)
        pred_fw = set(pred_ml['ml_frameworks_mentioned'])
        true_fw = set(true_ml['ml_frameworks_mentioned'])

        if pred_fw or true_fw:
            metrics['framework_jaccard'] = (
                len(pred_fw & true_fw) / len(pred_fw | true_fw) if (pred_fw | true_fw) else 0.0
            )
        else:
            metrics['framework_jaccard'] = 1.0  # Both empty = perfect

        # Model detection
        pred_models = set(pred_ml['specific_models_architectures'])
        true_models = set(true_ml['specific_models_architectures'])

        if pred_models or true_models:
            metrics['model_jaccard'] = (
                len(pred_models & true_models) / len(pred_models | true_models)
                if (pred_models | true_models) else 0.0
            )
        else:
            metrics['model_jaccard'] = 1.0

        return metrics

    def validate_reproducibility(self, predicted: Dict, ground_truth: Dict) -> Dict:
        """Validate reproducibility predictions"""
        metrics = {}

        pred_repro = predicted['reproducibility']
        true_repro = ground_truth['reproducibility']

        # Binary indicators
        metrics['code_availability_match'] = (
            pred_repro['code_availability_mentioned'] == true_repro['code_availability_mentioned']
        )
        metrics['data_availability_match'] = (
            pred_repro['data_availability_mentioned'] == true_repro['data_availability_mentioned']
        )

        # Methodology level
        metrics['methodology_level_match'] = (
            pred_repro['methodology_detail_level'] == true_repro['methodology_detail_level']
        )

        # Feasibility
        metrics['feasibility_match'] = (
            pred_repro['replication_feasibility'] == true_repro['replication_feasibility']
        )

        return metrics

    def validate_impact(self, predicted: Dict, ground_truth: Dict) -> Dict:
        """Validate impact indicator predictions"""
        metrics = {}

        pred_impact = predicted['impact_indicators']
        true_impact = ground_truth['impact_indicators']

        # Binary indicators
        metrics['novelty_match'] = (
            pred_impact['claims_novelty'] == true_impact['claims_novelty']
        )
        metrics['improvement_match'] = (
            pred_impact['claims_improvement_over_existing'] == true_impact['claims_improvement_over_existing']
        )

        # Impact scope
        metrics['impact_scope_match'] = (
            pred_impact['potential_impact_scope'] == true_impact['potential_impact_scope']
        )

        return metrics

    def validate_single_paper(self, predicted: Dict, ground_truth: Dict) -> Dict:
        """
        Validate a single paper evaluation.

        Args:
            predicted: Predicted evaluation
            ground_truth: Ground truth evaluation

        Returns:
            Dict with all validation metrics
        """
        metrics = {
            'paper_id': predicted.get('paper_id', ''),
            'ml_adoption': self.validate_ml_adoption(predicted, ground_truth),
            'reproducibility': self.validate_reproducibility(predicted, ground_truth),
            'impact': self.validate_impact(predicted, ground_truth),
        }

        # Overall accuracy (average of key metrics)
        key_metrics = [
            metrics['ml_adoption']['adoption_level_match'],
            metrics['ml_adoption']['framework_jaccard'],
            metrics['reproducibility']['code_availability_match'],
            metrics['reproducibility']['data_availability_match'],
            metrics['impact']['novelty_match'],
        ]

        metrics['overall_accuracy'] = sum(key_metrics) / len(key_metrics)

        return metrics

    def validate_dataset(self,
                        predictions: List[Dict],
                        ground_truth: List[Dict]) -> Dict:
        """
        Validate an entire dataset.

        Args:
            predictions: List of predicted evaluations
            ground_truth: List of ground truth evaluations

        Returns:
            Aggregated metrics
        """
        # Create lookup by paper_id
        gt_lookup = {p['paper_id']: p for p in ground_truth}

        results = []
        for pred in predictions:
            paper_id = pred['paper_id']
            if paper_id in gt_lookup:
                result = self.validate_single_paper(pred, gt_lookup[paper_id])
                results.append(result)

        if not results:
            print("Warning: No matching papers found between predictions and ground truth")
            return {}

        # Aggregate metrics
        aggregated = {
            'total_papers': len(results),
            'ml_adoption': {},
            'reproducibility': {},
            'impact': {},
            'overall': {}
        }

        # ML Adoption metrics
        aggregated['ml_adoption']['adoption_level_accuracy'] = (
            sum(r['ml_adoption']['adoption_level_match'] for r in results) / len(results)
        )
        aggregated['ml_adoption']['avg_framework_jaccard'] = (
            sum(r['ml_adoption']['framework_jaccard'] for r in results) / len(results)
        )
        aggregated['ml_adoption']['avg_model_jaccard'] = (
            sum(r['ml_adoption']['model_jaccard'] for r in results) / len(results)
        )

        # Reproducibility metrics
        aggregated['reproducibility']['code_accuracy'] = (
            sum(r['reproducibility']['code_availability_match'] for r in results) / len(results)
        )
        aggregated['reproducibility']['data_accuracy'] = (
            sum(r['reproducibility']['data_availability_match'] for r in results) / len(results)
        )
        aggregated['reproducibility']['methodology_accuracy'] = (
            sum(r['reproducibility']['methodology_level_match'] for r in results) / len(results)
        )

        # Impact metrics
        aggregated['impact']['novelty_accuracy'] = (
            sum(r['impact']['novelty_match'] for r in results) / len(results)
        )
        aggregated['impact']['improvement_accuracy'] = (
            sum(r['impact']['improvement_match'] for r in results) / len(results)
        )
        aggregated['impact']['scope_accuracy'] = (
            sum(r['impact']['impact_scope_match'] for r in results) / len(results)
        )

        # Overall accuracy
        aggregated['overall']['accuracy'] = (
            sum(r['overall_accuracy'] for r in results) / len(results)
        )

        return aggregated

    def run_validation(self,
                      validation_file: str = "Symby_AI/modded_s2orc_validation.jsonl",
                      max_papers: int = None) -> Dict:
        """
        Run full validation pipeline.

        Args:
            validation_file: Path to validation file
            max_papers: Maximum papers to validate (None = all)

        Returns:
            Validation metrics
        """
        print("="*60)
        print("VALIDATION PIPELINE")
        print("="*60)

        # Load validation data
        print(f"\nLoading validation data from: {validation_file}")
        validation_papers = list(self.processor.read_jsonl(Path(validation_file), max_papers))
        print(f"Loaded {len(validation_papers)} papers")

        # Check if already evaluated
        if validation_papers and self.is_evaluated(validation_papers[0]):
            print("Validation data is already evaluated (has ground truth)")
            ground_truth = validation_papers

            # Generate predictions
            print("\nGenerating predictions...")
            predictions = []
            for i, paper in enumerate(validation_papers):
                if i % 100 == 0:
                    print(f"  Processing {i}/{len(validation_papers)}...")

                # Create raw paper dict
                raw_paper = {
                    'id': paper['paper_id'],
                    'text': paper.get('text', ''),  # May not have text in evaluated data
                    'field': paper.get('field', ''),
                    'publication_date': paper.get('publication_date')
                }

                pred = self.evaluator.evaluate_paper(raw_paper)
                predictions.append(pred)

        else:
            print("Validation data is raw (no ground truth)")
            print("Cannot validate without ground truth - generating evaluations only")

            predictions = []
            for i, paper in enumerate(validation_papers):
                if i % 100 == 0:
                    print(f"  Processing {i}/{len(validation_papers)}...")

                pred = self.evaluator.evaluate_paper(paper)
                predictions.append(pred)

            # Save predictions
            output_file = "validation_predictions.jsonl"
            with open(output_file, 'w') as f:
                for pred in predictions:
                    f.write(json.dumps(pred) + '\n')

            print(f"\nSaved predictions to: {output_file}")
            print("To validate, provide a ground truth file with evaluation structure")

            return {'predictions': predictions, 'metrics': None}

        # Validate
        print("\nValidating predictions against ground truth...")
        metrics = self.validate_dataset(predictions, ground_truth)

        return {'predictions': predictions, 'ground_truth': ground_truth, 'metrics': metrics}

    def print_validation_report(self, results: Dict):
        """Print a formatted validation report"""
        metrics = results.get('metrics')

        if not metrics:
            print("\nNo metrics available (no ground truth)")
            return

        print("\n" + "="*60)
        print("VALIDATION RESULTS")
        print("="*60)

        print(f"\nTotal Papers Validated: {metrics['total_papers']}")

        print(f"\nOverall Accuracy: {metrics['overall']['accuracy']:.2%}")

        print("\nML Adoption Metrics:")
        print(f"  Adoption Level Accuracy: {metrics['ml_adoption']['adoption_level_accuracy']:.2%}")
        print(f"  Framework Detection (Jaccard): {metrics['ml_adoption']['avg_framework_jaccard']:.2%}")
        print(f"  Model Detection (Jaccard): {metrics['ml_adoption']['avg_model_jaccard']:.2%}")

        print("\nReproducibility Metrics:")
        print(f"  Code Availability Accuracy: {metrics['reproducibility']['code_accuracy']:.2%}")
        print(f"  Data Availability Accuracy: {metrics['reproducibility']['data_accuracy']:.2%}")
        print(f"  Methodology Level Accuracy: {metrics['reproducibility']['methodology_accuracy']:.2%}")

        print("\nImpact Metrics:")
        print(f"  Novelty Detection Accuracy: {metrics['impact']['novelty_accuracy']:.2%}")
        print(f"  Improvement Detection Accuracy: {metrics['impact']['improvement_accuracy']:.2%}")
        print(f"  Impact Scope Accuracy: {metrics['impact']['scope_accuracy']:.2%}")


if __name__ == "__main__":
    validator = EvaluationValidator()

    # Run validation on sample
    print("Running validation on first 200 papers...")
    results = validator.run_validation(
        validation_file="/home/asus/symby/Symby_AI/modded_s2orc_validation.jsonl",
        max_papers=200
    )

    # Print report
    validator.print_validation_report(results)

    print("\n" + "="*60)
    print("Validation complete!")
    print("="*60)
