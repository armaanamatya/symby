#!/usr/bin/env python3
"""
Test script for the paper analysis module.

Tests:
1. Schema creation and JSON serialization
2. Heuristic analysis (no LLM required)
3. Code extraction
4. JSON output format matching target schema
"""

import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.paper_analysis import (
    PaperAnalyzer,
    PaperAnalysisResult,
    MLAdoption,
    Reproducibility,
    ResearchOutcomes,
    ImpactIndicators,
    ExtractedCode
)


# Sample paper text (from CLAUDE.md example)
SAMPLE_PAPER = """
Analysis of Sediment Distribution at the Intake Structure

Abstract

As a Maritime country, most of Indonesia's population lives near the coast or estuary.
The main problem felt by the community is sanitation, clean water. The utilization of
groundwater is very limited in number because of the influence of seawater intrusion.

Introduction

We present a novel deep learning approach using a transformer architecture to analyze
sediment distribution patterns. Our model, based on BERT, achieves state-of-the-art
performance on the benchmark dataset.

Methods

The model was implemented using PyTorch and trained on NVIDIA A100 GPUs. We used the
following hyperparameters:
- Learning rate: 0.001
- Batch size: 32
- Epochs: 100

Code is available at: https://github.com/research/sediment-analysis

```python
import torch
import numpy as np

def analyze_sediment(data):
    model = TransformerModel()
    predictions = model(data)
    return predictions
```

Results

Our approach achieved 95% accuracy, outperforming the baseline by 15%. The model
showed significant improvements in precision (0.92) and recall (0.89).

Statistical validation was performed using cross-validation with p-value < 0.05.

Conclusion

This study demonstrates the effectiveness of deep learning for sediment analysis.
The code and data are available for replication.

Acknowledgments

This work was funded by NSF and NIH.
"""


def test_schema_creation():
    """Test schema dataclass creation."""
    print("\n=== Test 1: Schema Creation ===\n")

    # Create ML adoption
    ml = MLAdoption(
        ml_frameworks_mentioned=["pytorch", "tensorflow"],
        specific_models_architectures=["BERT", "transformer"],
        compute_resources_mentioned=["GPU", "A100"],
        ml_libraries_tools=["wandb"],
        ml_adoption_level="extensive",
        ml_method_primary="BERT",
        integration_with_traditional_methods=True
    )

    # Create reproducibility
    repro = Reproducibility(
        code_availability_mentioned=True,
        code_repository_type="github",
        data_availability_mentioned=True,
        methodology_detail_level="detailed",
        hyperparameters_specified=True,
        replication_feasibility="feasible"
    )

    # Create full result
    result = PaperAnalysisResult(
        paper_id="test_paper_001",
        field="Environmental Science",
        publication_date="2020",
        title="Analysis of Sediment Distribution",
        ml_adoption=ml,
        reproducibility=repro,
        overall_assessment="ML adoption: extensive; real-world application",
        confidence_score="high"
    )

    # Convert to dict
    result_dict = result.to_dict()

    print(f"Paper ID: {result_dict['paper_id']}")
    print(f"Field: {result_dict['field']}")
    print(f"ML Adoption Level: {result_dict['ml_adoption']['ml_adoption_level']}")
    print(f"Replication Feasibility: {result_dict['reproducibility']['replication_feasibility']}")

    # Test JSON serialization
    json_str = result.to_json()
    print(f"\nJSON output length: {len(json_str)} chars")

    # Verify round-trip
    parsed = json.loads(json_str)
    assert parsed['paper_id'] == 'test_paper_001'
    assert parsed['ml_adoption']['ml_adoption_level'] == 'extensive'

    print("\n✅ Schema creation test passed!")
    return True


def test_heuristic_analysis():
    """Test heuristic-based analysis (no LLM)."""
    print("\n=== Test 2: Heuristic Analysis ===\n")

    analyzer = PaperAnalyzer(use_llm=False)

    result = analyzer.analyze(
        text=SAMPLE_PAPER,
        paper_id="sample_paper_001",
        title="Analysis of Sediment Distribution"
    )

    print(f"Detected Field: {result.field}")
    print(f"ML Adoption Level: {result.ml_adoption.ml_adoption_level}")
    print(f"ML Frameworks Found: {result.ml_adoption.ml_frameworks_mentioned}")
    print(f"ML Architectures Found: {result.ml_adoption.specific_models_architectures}")
    print(f"Compute Resources: {result.ml_adoption.compute_resources_mentioned}")
    print(f"\nCode Available: {result.reproducibility.code_availability_mentioned}")
    print(f"Repo Type: {result.reproducibility.code_repository_type}")
    print(f"Hyperparams Specified: {result.reproducibility.hyperparameters_specified}")
    print(f"Replication Feasibility: {result.reproducibility.replication_feasibility}")
    print(f"\nClaims Novelty: {result.impact_indicators.claims_novelty}")
    print(f"Claims Improvement: {result.impact_indicators.claims_improvement_over_existing}")
    print(f"Impact Scope: {result.impact_indicators.potential_impact_scope}")
    print(f"\nGitHub Repos: {result.github_repos}")
    print(f"Code Blocks Extracted: {len(result.code_blocks)}")
    print(f"\nOverall Assessment: {result.overall_assessment}")
    print(f"Confidence: {result.confidence_score}")

    # Verify key findings
    assert result.ml_adoption.ml_adoption_level in ['moderate', 'extensive']
    assert 'pytorch' in [f.lower() for f in result.ml_adoption.ml_frameworks_mentioned]
    assert result.reproducibility.code_availability_mentioned
    assert len(result.github_repos) > 0

    print("\n✅ Heuristic analysis test passed!")
    return True


def test_json_format():
    """Test that output matches target JSON format."""
    print("\n=== Test 3: JSON Format Verification ===\n")

    analyzer = PaperAnalyzer(use_llm=False)
    result = analyzer.analyze(text=SAMPLE_PAPER, paper_id="format_test")

    json_output = json.loads(result.to_json())

    # Check all required top-level keys
    required_keys = [
        'paper_id', 'field', 'ml_adoption', 'reproducibility',
        'research_outcomes', 'impact_indicators', 'overall_assessment',
        'confidence_score', 'notes'
    ]

    for key in required_keys:
        assert key in json_output, f"Missing key: {key}"
        print(f"✓ {key}")

    # Check ml_adoption structure
    ml_keys = [
        'ml_frameworks_mentioned', 'specific_models_architectures',
        'compute_resources_mentioned', 'ml_libraries_tools',
        'ml_adoption_level', 'ml_method_primary', 'integration_with_traditional_methods'
    ]

    print("\nml_adoption subkeys:")
    for key in ml_keys:
        assert key in json_output['ml_adoption'], f"Missing ml_adoption key: {key}"
        print(f"  ✓ {key}")

    # Check reproducibility structure
    repro_keys = [
        'code_availability_mentioned', 'code_repository_type',
        'data_availability_mentioned', 'methodology_detail_level',
        'hyperparameters_specified', 'replication_feasibility'
    ]

    print("\nreproducibility subkeys:")
    for key in repro_keys:
        assert key in json_output['reproducibility'], f"Missing reproducibility key: {key}"
        print(f"  ✓ {key}")

    print("\n✅ JSON format test passed!")
    return True


def test_code_extraction():
    """Test code block extraction."""
    print("\n=== Test 4: Code Extraction ===\n")

    analyzer = PaperAnalyzer(use_llm=False)
    result = analyzer.analyze(text=SAMPLE_PAPER, paper_id="code_test")

    print(f"Extracted {len(result.code_blocks)} code block(s)")

    for i, block in enumerate(result.code_blocks):
        print(f"\nBlock {i+1}:")
        print(f"  Language: {block.language}")
        print(f"  Executable: {block.executable}")
        print(f"  Code preview: {block.code[:50]}...")

    if result.code_blocks:
        # Verify code was extracted
        assert result.code_blocks[0].language == 'python'
        assert 'import' in result.code_blocks[0].code.lower()

    print("\n✅ Code extraction test passed!")
    return True


def test_full_json_output():
    """Generate and print full JSON output."""
    print("\n=== Test 5: Full JSON Output ===\n")

    analyzer = PaperAnalyzer(use_llm=False)
    result = analyzer.analyze(
        text=SAMPLE_PAPER,
        paper_id="254371605",
        title="Analysis of Sediment Distribution",
        publication_date="2020"
    )

    json_output = result.to_json(indent=2)
    print(json_output)

    print("\n✅ Full JSON output test passed!")
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("Paper Analysis Module Test Suite")
    print("=" * 60)

    tests = [
        test_schema_creation,
        test_heuristic_analysis,
        test_json_format,
        test_code_extraction,
        test_full_json_output
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"\n❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
