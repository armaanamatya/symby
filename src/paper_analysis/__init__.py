"""
Paper Analysis Module

Provides comprehensive analysis of research papers with structured JSON output.

Features:
- PDF/text paper analysis with structured JSON output
- ML adoption detection and classification
- Reproducibility assessment
- Code extraction and sandbox execution
- Paper Q&A/inference using LLM

Usage:
    from src.paper_analysis import PaperAnalyzer, PaperAnalysisResult

    # Analyze a paper
    analyzer = PaperAnalyzer(use_llm=True)
    result = analyzer.analyze(text=paper_text, paper_id="my_paper")

    # Get JSON output
    json_output = result.to_json()

    # Query the paper
    answer = analyzer.query_paper(paper_text, "What ML methods are used?")
"""

from .schemas import (
    PaperAnalysisResult,
    MLAdoption,
    Reproducibility,
    ResearchOutcomes,
    ImpactIndicators,
    ExtractedCode,
    MLAdoptionLevel,
    ReplicationFeasibility,
    MethodologyDetailLevel,
    PotentialImpactScope,
    ConfidenceLevel
)

from .analyzer import PaperAnalyzer, analyze_paper_to_json

# Import executor components
try:
    from .executor import (
        PaperCodeExecutor,
        PaperQueryEngine,
        run_paper_analysis_pipeline
    )
    EXECUTOR_AVAILABLE = True
except ImportError:
    EXECUTOR_AVAILABLE = False
    PaperCodeExecutor = None
    PaperQueryEngine = None
    run_paper_analysis_pipeline = None

__all__ = [
    # Schemas
    'PaperAnalysisResult',
    'MLAdoption',
    'Reproducibility',
    'ResearchOutcomes',
    'ImpactIndicators',
    'ExtractedCode',
    'MLAdoptionLevel',
    'ReplicationFeasibility',
    'MethodologyDetailLevel',
    'PotentialImpactScope',
    'ConfidenceLevel',
    # Analyzer
    'PaperAnalyzer',
    'analyze_paper_to_json',
    # Executor
    'PaperCodeExecutor',
    'PaperQueryEngine',
    'run_paper_analysis_pipeline',
    'EXECUTOR_AVAILABLE'
]
