"""
Test a single paper/document with the paper evaluator
"""

import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src' / 'data'))

from paper_evaluator import PaperEvaluator


def test_paper_from_text(text: str, paper_id: str = "test", field: str = "Unknown"):
    """
    Test evaluation on a single paper from text.

    Args:
        text: Full text of the paper
        paper_id: ID for the paper (optional)
        field: Field/domain of the paper (optional)
    """
    evaluator = PaperEvaluator()

    # Create paper object in S2ORC format
    paper = {
        'id': paper_id,
        'text': text,
        'field': field,
        'publication_date': 2024,
        'matched_term': None
    }

    print("="*70)
    print("EVALUATING PAPER")
    print("="*70)
    print(f"ID: {paper_id}")
    print(f"Field: {field}")
    print(f"Text length: {len(text)} characters")
    print(f"Text preview: {text[:200]}...")
    print()

    # Evaluate
    print("Running evaluation...")
    evaluation = evaluator.evaluate_paper(paper)

    # Display results
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    print(json.dumps(evaluation, indent=2))

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"ML Adoption Level: {evaluation['ml_adoption']['ml_adoption_level']}")
    print(f"Code Available: {evaluation['reproducibility']['code_availability_mentioned']}")
    print(f"Data Available: {evaluation['reproducibility']['data_availability_mentioned']}")
    print(f"Claims Novelty: {evaluation['impact_indicators']['claims_novelty']}")
    print(f"Overall Assessment: {evaluation['overall_assessment']}")
    print(f"Confidence: {evaluation['confidence_score']}")

    return evaluation


def test_paper_from_file(file_path: str):
    """
    Test evaluation on a paper from a text file.

    Args:
        file_path: Path to text file containing paper content
    """
    path = Path(file_path)

    if not path.exists():
        print(f"❌ Error: File not found: {file_path}")
        return None

    print(f"Reading from: {path}")
    text = path.read_text(encoding='utf-8')

    # Use filename as ID
    paper_id = path.stem

    return test_paper_from_text(text, paper_id=paper_id)


def interactive_test():
    """
    Interactive mode - prompts user for input
    """
    print("="*70)
    print("PAPER EVALUATION - INTERACTIVE MODE")
    print("="*70)
    print()
    print("Choose input method:")
    print("  1. Paste text directly")
    print("  2. Load from file")
    print()

    choice = input("Enter choice (1 or 2): ").strip()

    if choice == "1":
        print("\nPaste your paper text below (Ctrl+D or Ctrl+Z when done):")
        lines = []
        try:
            while True:
                line = input()
                lines.append(line)
        except EOFError:
            pass

        text = '\n'.join(lines)

        if not text.strip():
            print("❌ No text provided")
            return

        paper_id = input("\nEnter paper ID (or press Enter for 'test'): ").strip() or "test"
        field = input("Enter field/domain (or press Enter for 'Unknown'): ").strip() or "Unknown"

        test_paper_from_text(text, paper_id=paper_id, field=field)

    elif choice == "2":
        file_path = input("\nEnter file path: ").strip()
        test_paper_from_file(file_path)

    else:
        print("❌ Invalid choice")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # File path provided as argument
        file_path = sys.argv[1]
        test_paper_from_file(file_path)
    else:
        # Interactive mode
        interactive_test()
