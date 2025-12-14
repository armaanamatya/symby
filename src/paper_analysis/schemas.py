"""
Paper Analysis Schemas

Defines the JSON output structure for comprehensive paper analysis.
Matches the target format from s2orc_data analysis.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any
import json


class MLAdoptionLevel(str, Enum):
    """Level of ML adoption in a paper."""
    NONE = "none"
    MINIMAL = "minimal"
    MODERATE = "moderate"
    EXTENSIVE = "extensive"


class ReplicationFeasibility(str, Enum):
    """How feasible it is to replicate the paper's work."""
    NOT_FEASIBLE = "not_feasible"
    DIFFICULT = "difficult"
    MODERATE = "moderate"
    FEASIBLE = "feasible"
    HIGHLY_FEASIBLE = "highly_feasible"


class MethodologyDetailLevel(str, Enum):
    """Level of detail in methodology description."""
    SPARSE = "sparse"
    BASIC = "basic"
    MODERATE = "moderate"
    DETAILED = "detailed"
    COMPREHENSIVE = "comprehensive"


class PotentialImpactScope(str, Enum):
    """Scope of potential impact."""
    NARROW = "narrow"
    MODERATE = "moderate"
    BROAD = "broad"
    TRANSFORMATIONAL = "transformational"


class ConfidenceLevel(str, Enum):
    """Confidence in the analysis."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class MLAdoption:
    """ML adoption analysis for a paper."""
    ml_frameworks_mentioned: List[str] = field(default_factory=list)
    specific_models_architectures: List[str] = field(default_factory=list)
    compute_resources_mentioned: List[str] = field(default_factory=list)
    datasets_referenced: List[str] = field(default_factory=list)
    ml_libraries_tools: List[str] = field(default_factory=list)
    ml_adoption_level: str = "none"
    ml_application_domain: Optional[str] = None
    ml_method_primary: Optional[str] = None
    integration_with_traditional_methods: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ml_frameworks_mentioned": self.ml_frameworks_mentioned,
            "specific_models_architectures": self.specific_models_architectures,
            "compute_resources_mentioned": self.compute_resources_mentioned,
            "datasets_referenced": self.datasets_referenced,
            "ml_libraries_tools": self.ml_libraries_tools,
            "ml_adoption_level": self.ml_adoption_level,
            "ml_application_domain": self.ml_application_domain,
            "ml_method_primary": self.ml_method_primary,
            "integration_with_traditional_methods": self.integration_with_traditional_methods
        }


@dataclass
class Reproducibility:
    """Reproducibility analysis for a paper."""
    code_availability_mentioned: bool = False
    code_repository_type: Optional[str] = None  # github, gitlab, zenodo, etc.
    data_availability_mentioned: bool = False
    data_sharing_statement: Optional[str] = None
    methodology_detail_level: str = "basic"
    hyperparameters_specified: bool = False
    computational_environment_described: bool = False
    preprocessing_steps_detailed: bool = False
    statistical_methods_described: bool = False
    replication_feasibility: str = "not_feasible"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code_availability_mentioned": self.code_availability_mentioned,
            "code_repository_type": self.code_repository_type,
            "data_availability_mentioned": self.data_availability_mentioned,
            "data_sharing_statement": self.data_sharing_statement,
            "methodology_detail_level": self.methodology_detail_level,
            "hyperparameters_specified": self.hyperparameters_specified,
            "computational_environment_described": self.computational_environment_described,
            "preprocessing_steps_detailed": self.preprocessing_steps_detailed,
            "statistical_methods_described": self.statistical_methods_described,
            "replication_feasibility": self.replication_feasibility
        }


@dataclass
class ResearchOutcomes:
    """Research outcomes analysis."""
    mentions_clinical_trials: bool = False
    clinical_trial_identifiers: List[str] = field(default_factory=list)
    mentions_patents: bool = False
    patent_numbers: List[str] = field(default_factory=list)
    mentions_corrections: bool = False
    mentions_retractions: bool = False
    validation_type: List[str] = field(default_factory=list)
    real_world_application_mentioned: bool = False
    commercialization_mentioned: bool = False
    regulatory_approval_mentioned: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mentions_clinical_trials": self.mentions_clinical_trials,
            "clinical_trial_identifiers": self.clinical_trial_identifiers,
            "mentions_patents": self.mentions_patents,
            "patent_numbers": self.patent_numbers,
            "mentions_corrections": self.mentions_corrections,
            "mentions_retractions": self.mentions_retractions,
            "validation_type": self.validation_type,
            "real_world_application_mentioned": self.real_world_application_mentioned,
            "commercialization_mentioned": self.commercialization_mentioned,
            "regulatory_approval_mentioned": self.regulatory_approval_mentioned
        }


@dataclass
class ImpactIndicators:
    """Impact indicators for the paper."""
    claims_novelty: bool = False
    claims_improvement_over_existing: bool = False
    quantitative_improvements_mentioned: bool = False
    comparison_to_baseline: bool = False
    mentions_policy_implications: bool = False
    mentions_clinical_guidelines: bool = False
    mentions_media_coverage: bool = False
    collaboration_indicators: List[str] = field(default_factory=list)
    funding_sources_mentioned: List[str] = field(default_factory=list)
    potential_impact_scope: str = "narrow"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claims_novelty": self.claims_novelty,
            "claims_improvement_over_existing": self.claims_improvement_over_existing,
            "quantitative_improvements_mentioned": self.quantitative_improvements_mentioned,
            "comparison_to_baseline": self.comparison_to_baseline,
            "mentions_policy_implications": self.mentions_policy_implications,
            "mentions_clinical_guidelines": self.mentions_clinical_guidelines,
            "mentions_media_coverage": self.mentions_media_coverage,
            "collaboration_indicators": self.collaboration_indicators,
            "funding_sources_mentioned": self.funding_sources_mentioned,
            "potential_impact_scope": self.potential_impact_scope
        }


@dataclass
class ExtractedCode:
    """Code extracted from the paper."""
    language: str = "python"
    code: str = ""
    description: Optional[str] = None
    line_start: Optional[int] = None
    dependencies: List[str] = field(default_factory=list)
    executable: bool = False
    execution_result: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "language": self.language,
            "code": self.code,
            "description": self.description,
            "line_start": self.line_start,
            "dependencies": self.dependencies,
            "executable": self.executable,
            "execution_result": self.execution_result
        }


@dataclass
class PaperAnalysisResult:
    """
    Complete analysis result for a paper.

    Matches the target JSON schema from s2orc_data format.
    """
    paper_id: str
    field: str
    publication_date: Optional[str] = None
    title: Optional[str] = None
    abstract: Optional[str] = None

    # Core analysis sections
    ml_adoption: MLAdoption = field(default_factory=MLAdoption)
    reproducibility: Reproducibility = field(default_factory=Reproducibility)
    research_outcomes: ResearchOutcomes = field(default_factory=ResearchOutcomes)
    impact_indicators: ImpactIndicators = field(default_factory=ImpactIndicators)

    # Extracted code
    code_blocks: List[ExtractedCode] = field(default_factory=list)
    github_repos: List[str] = field(default_factory=list)

    # Summary
    overall_assessment: str = ""
    confidence_score: str = "low"
    notes: str = ""

    # Analysis metadata
    analyzed_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    text_length: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary matching the target JSON format."""
        return {
            "paper_id": self.paper_id,
            "field": self.field,
            "publication_date": self.publication_date,
            "title": self.title,
            "abstract": self.abstract,
            "ml_adoption": self.ml_adoption.to_dict(),
            "reproducibility": self.reproducibility.to_dict(),
            "research_outcomes": self.research_outcomes.to_dict(),
            "impact_indicators": self.impact_indicators.to_dict(),
            "code_blocks": [cb.to_dict() for cb in self.code_blocks],
            "github_repos": self.github_repos,
            "overall_assessment": self.overall_assessment,
            "confidence_score": self.confidence_score,
            "notes": self.notes,
            "analyzed_at": self.analyzed_at,
            "text_length": self.text_length
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PaperAnalysisResult':
        """Create from dictionary."""
        result = cls(
            paper_id=data.get("paper_id", ""),
            field=data.get("field", ""),
            publication_date=data.get("publication_date"),
            title=data.get("title"),
            abstract=data.get("abstract"),
            overall_assessment=data.get("overall_assessment", ""),
            confidence_score=data.get("confidence_score", "low"),
            notes=data.get("notes", ""),
            analyzed_at=data.get("analyzed_at", datetime.utcnow().isoformat()),
            text_length=data.get("text_length", 0),
            github_repos=data.get("github_repos", [])
        )

        # Parse ML adoption
        if "ml_adoption" in data:
            ma = data["ml_adoption"]
            result.ml_adoption = MLAdoption(
                ml_frameworks_mentioned=ma.get("ml_frameworks_mentioned", []),
                specific_models_architectures=ma.get("specific_models_architectures", []),
                compute_resources_mentioned=ma.get("compute_resources_mentioned", []),
                datasets_referenced=ma.get("datasets_referenced", []),
                ml_libraries_tools=ma.get("ml_libraries_tools", []),
                ml_adoption_level=ma.get("ml_adoption_level", "none"),
                ml_application_domain=ma.get("ml_application_domain"),
                ml_method_primary=ma.get("ml_method_primary"),
                integration_with_traditional_methods=ma.get("integration_with_traditional_methods", False)
            )

        # Parse reproducibility
        if "reproducibility" in data:
            r = data["reproducibility"]
            result.reproducibility = Reproducibility(
                code_availability_mentioned=r.get("code_availability_mentioned", False),
                code_repository_type=r.get("code_repository_type"),
                data_availability_mentioned=r.get("data_availability_mentioned", False),
                data_sharing_statement=r.get("data_sharing_statement"),
                methodology_detail_level=r.get("methodology_detail_level", "basic"),
                hyperparameters_specified=r.get("hyperparameters_specified", False),
                computational_environment_described=r.get("computational_environment_described", False),
                preprocessing_steps_detailed=r.get("preprocessing_steps_detailed", False),
                statistical_methods_described=r.get("statistical_methods_described", False),
                replication_feasibility=r.get("replication_feasibility", "not_feasible")
            )

        # Parse research outcomes
        if "research_outcomes" in data:
            ro = data["research_outcomes"]
            result.research_outcomes = ResearchOutcomes(
                mentions_clinical_trials=ro.get("mentions_clinical_trials", False),
                clinical_trial_identifiers=ro.get("clinical_trial_identifiers", []),
                mentions_patents=ro.get("mentions_patents", False),
                patent_numbers=ro.get("patent_numbers", []),
                mentions_corrections=ro.get("mentions_corrections", False),
                mentions_retractions=ro.get("mentions_retractions", False),
                validation_type=ro.get("validation_type", []),
                real_world_application_mentioned=ro.get("real_world_application_mentioned", False),
                commercialization_mentioned=ro.get("commercialization_mentioned", False),
                regulatory_approval_mentioned=ro.get("regulatory_approval_mentioned", False)
            )

        # Parse impact indicators
        if "impact_indicators" in data:
            ii = data["impact_indicators"]
            result.impact_indicators = ImpactIndicators(
                claims_novelty=ii.get("claims_novelty", False),
                claims_improvement_over_existing=ii.get("claims_improvement_over_existing", False),
                quantitative_improvements_mentioned=ii.get("quantitative_improvements_mentioned", False),
                comparison_to_baseline=ii.get("comparison_to_baseline", False),
                mentions_policy_implications=ii.get("mentions_policy_implications", False),
                mentions_clinical_guidelines=ii.get("mentions_clinical_guidelines", False),
                mentions_media_coverage=ii.get("mentions_media_coverage", False),
                collaboration_indicators=ii.get("collaboration_indicators", []),
                funding_sources_mentioned=ii.get("funding_sources_mentioned", []),
                potential_impact_scope=ii.get("potential_impact_scope", "narrow")
            )

        # Parse code blocks
        if "code_blocks" in data:
            result.code_blocks = [
                ExtractedCode(
                    language=cb.get("language", "python"),
                    code=cb.get("code", ""),
                    description=cb.get("description"),
                    line_start=cb.get("line_start"),
                    dependencies=cb.get("dependencies", []),
                    executable=cb.get("executable", False),
                    execution_result=cb.get("execution_result")
                )
                for cb in data["code_blocks"]
            ]

        return result
