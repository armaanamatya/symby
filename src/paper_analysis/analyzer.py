"""
Paper Analyzer Module

Analyzes research papers (PDF or text) and produces structured JSON output.
Integrates with LLM for intelligent analysis, code extraction, and Q&A.
"""

import os
import re
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = None

from .schemas import (
    PaperAnalysisResult,
    MLAdoption,
    Reproducibility,
    ResearchOutcomes,
    ImpactIndicators,
    ExtractedCode
)

logger = logging.getLogger(__name__)


# Pattern definitions for text analysis
ML_FRAMEWORKS = [
    "pytorch", "tensorflow", "keras", "jax", "mxnet", "caffe", "theano",
    "scikit-learn", "sklearn", "xgboost", "lightgbm", "catboost"
]

ML_ARCHITECTURES = [
    "transformer", "bert", "gpt", "llama", "t5", "vit", "resnet", "vgg",
    "inception", "efficientnet", "unet", "gan", "vae", "diffusion",
    "lstm", "gru", "rnn", "cnn", "mlp", "attention", "autoencoder",
    "encoder-decoder", "seq2seq", "yolo", "faster-rcnn", "mask-rcnn",
    "alphafold", "esm", "clip", "dall-e", "stable diffusion", "whisper"
]

COMPUTE_RESOURCES = [
    "gpu", "tpu", "cuda", "a100", "v100", "h100", "dgx", "hpc",
    "cloud", "aws", "gcp", "azure", "nvidia", "amd"
]

ML_TOOLS = [
    "wandb", "mlflow", "tensorboard", "ray", "dask", "spark", "horovod",
    "deepspeed", "huggingface", "transformers", "datasets", "accelerate",
    "pytorch-lightning", "fastai", "optuna", "hydra"
]

VALIDATION_TYPES = [
    "cross-validation", "k-fold", "holdout", "bootstrap", "ablation",
    "statistical validation", "experimental validation", "clinical validation",
    "benchmark", "baseline comparison"
]

GITHUB_PATTERN = re.compile(
    r'github\.com/([a-zA-Z0-9_-]+)/([a-zA-Z0-9_.-]+)',
    re.IGNORECASE
)

CODE_BLOCK_PATTERN = re.compile(
    r'```(?:python|py)?\s*(.*?)```',
    re.DOTALL | re.IGNORECASE
)

CLINICAL_TRIAL_PATTERN = re.compile(
    r'(NCT\d{8}|ISRCTN\d{8}|ACTRN\d{14}|ChiCTR\d+)',
    re.IGNORECASE
)

PATENT_PATTERN = re.compile(
    r'(US\d{7,10}|EP\d{7}|WO\d{4}/\d{6})',
    re.IGNORECASE
)


class PaperAnalyzer:
    """
    Analyzes research papers and produces structured JSON output.

    Features:
    - PDF parsing for text extraction
    - LLM-powered intelligent analysis
    - ML adoption detection
    - Reproducibility assessment
    - Code extraction
    - Q&A interface for paper queries
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "meta/llama-3.3-70b-instruct",
        use_llm: bool = True
    ):
        """
        Initialize the paper analyzer.

        Args:
            api_key: NVIDIA API key (or set NVIDIA_API_KEY env var)
            model: LLM model to use for analysis
            use_llm: Whether to use LLM for enhanced analysis
        """
        self.use_llm = use_llm
        self.model = model
        self.client = None

        if use_llm:
            if not OPENAI_AVAILABLE:
                logger.warning("OpenAI module not available. LLM features disabled.")
                self.use_llm = False
            else:
                api_key = api_key or os.environ.get('NVIDIA_API_KEY')
                if api_key and api_key.startswith("nvapi-"):
                    self.client = OpenAI(
                        base_url="https://integrate.api.nvidia.com/v1",
                        api_key=api_key
                    )
                else:
                    logger.warning("NVIDIA API key not available. LLM features disabled.")
                    self.use_llm = False

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text content from a PDF file."""
        if not PYMUPDF_AVAILABLE:
            raise RuntimeError("PyMuPDF not installed. Run: pip install pymupdf")

        text_parts = []
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text_parts.append(page.get_text())

        return "\n".join(text_parts)

    def extract_text_from_pdf_bytes(self, pdf_content: bytes) -> str:
        """Extract text content from PDF bytes."""
        if not PYMUPDF_AVAILABLE:
            raise RuntimeError("PyMuPDF not installed. Run: pip install pymupdf")

        text_parts = []
        with fitz.open(stream=pdf_content, filetype="pdf") as doc:
            for page in doc:
                text_parts.append(page.get_text())

        return "\n".join(text_parts)

    def _find_patterns(self, text: str, patterns: List[str]) -> List[str]:
        """Find all matching patterns in text."""
        found = []
        text_lower = text.lower()
        for pattern in patterns:
            if pattern.lower() in text_lower:
                found.append(pattern)
        return found

    def _analyze_ml_adoption_heuristic(self, text: str) -> MLAdoption:
        """Analyze ML adoption using heuristic pattern matching."""
        text_lower = text.lower()

        frameworks = self._find_patterns(text, ML_FRAMEWORKS)
        architectures = self._find_patterns(text, ML_ARCHITECTURES)
        compute = self._find_patterns(text, COMPUTE_RESOURCES)
        tools = self._find_patterns(text, ML_TOOLS)

        # Determine adoption level
        ml_indicators = len(frameworks) + len(architectures) + len(tools)
        if ml_indicators == 0:
            level = "none"
        elif ml_indicators <= 2:
            level = "minimal"
        elif ml_indicators <= 5:
            level = "moderate"
        else:
            level = "extensive"

        # Find primary method
        primary_method = architectures[0] if architectures else None

        # Check for integration with traditional methods
        traditional_keywords = ["regression", "statistical", "hypothesis", "p-value", "anova"]
        has_traditional = any(kw in text_lower for kw in traditional_keywords)
        integration = has_traditional and ml_indicators > 0

        return MLAdoption(
            ml_frameworks_mentioned=frameworks,
            specific_models_architectures=architectures,
            compute_resources_mentioned=compute,
            ml_libraries_tools=tools,
            ml_adoption_level=level,
            ml_method_primary=primary_method,
            integration_with_traditional_methods=integration
        )

    def _analyze_reproducibility_heuristic(self, text: str) -> Reproducibility:
        """Analyze reproducibility using heuristic pattern matching."""
        text_lower = text.lower()

        # Code availability
        code_mentioned = any(kw in text_lower for kw in [
            "code available", "code is available", "github", "gitlab",
            "source code", "implementation available", "repository"
        ])

        # Detect repository type
        repo_type = None
        if "github" in text_lower:
            repo_type = "github"
        elif "gitlab" in text_lower:
            repo_type = "gitlab"
        elif "zenodo" in text_lower:
            repo_type = "zenodo"
        elif "figshare" in text_lower:
            repo_type = "figshare"

        # Data availability
        data_mentioned = any(kw in text_lower for kw in [
            "data available", "dataset available", "data is available",
            "publicly available", "data can be found", "data shared"
        ])

        # Hyperparameters
        hyperparams = any(kw in text_lower for kw in [
            "hyperparameter", "learning rate", "batch size", "epochs",
            "optimizer", "dropout", "weight decay"
        ])

        # Computational environment
        env_described = any(kw in text_lower for kw in [
            "nvidia", "cuda version", "python version", "torch version",
            "tensorflow version", "training time", "compute"
        ])

        # Preprocessing
        preprocessing = any(kw in text_lower for kw in [
            "preprocessing", "data augmentation", "normalization",
            "tokenization", "feature extraction"
        ])

        # Statistical methods
        stats = any(kw in text_lower for kw in [
            "p-value", "confidence interval", "statistical significance",
            "t-test", "chi-square", "correlation", "regression analysis"
        ])

        # Determine methodology detail level
        detail_score = sum([
            hyperparams, env_described, preprocessing, stats
        ])
        if detail_score >= 3:
            detail_level = "detailed"
        elif detail_score >= 2:
            detail_level = "moderate"
        elif detail_score >= 1:
            detail_level = "basic"
        else:
            detail_level = "sparse"

        # Determine replication feasibility
        feasibility_score = sum([
            code_mentioned * 2, data_mentioned * 2,
            hyperparams, env_described, preprocessing
        ])
        if feasibility_score >= 6:
            feasibility = "highly_feasible"
        elif feasibility_score >= 4:
            feasibility = "feasible"
        elif feasibility_score >= 2:
            feasibility = "moderate"
        elif feasibility_score >= 1:
            feasibility = "difficult"
        else:
            feasibility = "not_feasible"

        return Reproducibility(
            code_availability_mentioned=code_mentioned,
            code_repository_type=repo_type,
            data_availability_mentioned=data_mentioned,
            methodology_detail_level=detail_level,
            hyperparameters_specified=hyperparams,
            computational_environment_described=env_described,
            preprocessing_steps_detailed=preprocessing,
            statistical_methods_described=stats,
            replication_feasibility=feasibility
        )

    def _analyze_research_outcomes_heuristic(self, text: str) -> ResearchOutcomes:
        """Analyze research outcomes using heuristic pattern matching."""
        text_lower = text.lower()

        # Clinical trials
        clinical_trials = "clinical trial" in text_lower or "clinical study" in text_lower
        trial_ids = CLINICAL_TRIAL_PATTERN.findall(text)

        # Patents
        patents = "patent" in text_lower
        patent_nums = PATENT_PATTERN.findall(text)

        # Corrections/retractions
        corrections = "correction" in text_lower or "erratum" in text_lower
        retractions = "retraction" in text_lower or "retracted" in text_lower

        # Validation types
        validation = []
        for vtype in VALIDATION_TYPES:
            if vtype.lower() in text_lower:
                validation.append(vtype)

        # Real-world applications
        real_world = any(kw in text_lower for kw in [
            "real-world", "deployed", "production", "clinical practice",
            "industry", "application", "practical"
        ])

        # Commercialization
        commercial = any(kw in text_lower for kw in [
            "commercial", "product", "startup", "company", "market"
        ])

        # Regulatory
        regulatory = any(kw in text_lower for kw in [
            "fda", "ce mark", "regulatory", "approved", "clearance"
        ])

        return ResearchOutcomes(
            mentions_clinical_trials=clinical_trials,
            clinical_trial_identifiers=trial_ids,
            mentions_patents=patents,
            patent_numbers=patent_nums,
            mentions_corrections=corrections,
            mentions_retractions=retractions,
            validation_type=validation,
            real_world_application_mentioned=real_world,
            commercialization_mentioned=commercial,
            regulatory_approval_mentioned=regulatory
        )

    def _analyze_impact_indicators_heuristic(self, text: str) -> ImpactIndicators:
        """Analyze impact indicators using heuristic pattern matching."""
        text_lower = text.lower()

        # Novelty claims
        novelty = any(kw in text_lower for kw in [
            "novel", "first", "new approach", "unprecedented", "innovative"
        ])

        # Improvement claims
        improvement = any(kw in text_lower for kw in [
            "improve", "outperform", "better than", "state-of-the-art",
            "sota", "benchmark", "surpass"
        ])

        # Quantitative improvements
        quantitative = any(kw in text_lower for kw in [
            "% improvement", "percentage", "accuracy", "f1 score",
            "precision", "recall", "auc", "rmse", "mae"
        ])

        # Baseline comparison
        baseline = any(kw in text_lower for kw in [
            "baseline", "compared to", "comparison", "ablation"
        ])

        # Policy implications
        policy = any(kw in text_lower for kw in [
            "policy", "regulation", "guideline", "recommendation"
        ])

        # Clinical guidelines
        clinical = "clinical guideline" in text_lower

        # Media coverage
        media = any(kw in text_lower for kw in [
            "media", "news", "press", "coverage"
        ])

        # Funding sources
        funding = []
        funding_keywords = ["nih", "nsf", "darpa", "european", "grant", "funded by"]
        for kw in funding_keywords:
            if kw in text_lower:
                funding.append(kw.upper())

        # Determine impact scope
        impact_score = sum([
            novelty, improvement * 2, quantitative, baseline,
            policy, clinical
        ])
        if impact_score >= 5:
            scope = "transformational"
        elif impact_score >= 3:
            scope = "broad"
        elif impact_score >= 1:
            scope = "moderate"
        else:
            scope = "narrow"

        return ImpactIndicators(
            claims_novelty=novelty,
            claims_improvement_over_existing=improvement,
            quantitative_improvements_mentioned=quantitative,
            comparison_to_baseline=baseline,
            mentions_policy_implications=policy,
            mentions_clinical_guidelines=clinical,
            mentions_media_coverage=media,
            funding_sources_mentioned=funding,
            potential_impact_scope=scope
        )

    def _extract_code_blocks(self, text: str) -> List[ExtractedCode]:
        """Extract code blocks from text."""
        code_blocks = []

        for match in CODE_BLOCK_PATTERN.finditer(text):
            code = match.group(1).strip()
            if len(code) > 20:  # Minimum length
                code_blocks.append(ExtractedCode(
                    language="python",
                    code=code,
                    description="Extracted from paper",
                    executable=self._is_executable_code(code)
                ))

        return code_blocks

    def _extract_github_repos(self, text: str) -> List[str]:
        """Extract GitHub repository URLs from text."""
        repos = set()
        for match in GITHUB_PATTERN.finditer(text):
            username, repo = match.groups()
            repos.add(f"https://github.com/{username}/{repo}")
        return list(repos)

    def _is_executable_code(self, code: str) -> bool:
        """Check if code looks executable."""
        indicators = ["import ", "def ", "class ", "if ", "for ", "while "]
        return any(ind in code for ind in indicators)

    def _detect_field(self, text: str) -> str:
        """Detect the scientific field from text."""
        text_lower = text.lower()

        field_keywords = {
            "Biology": ["protein", "gene", "cell", "dna", "rna", "genome", "biological"],
            "Medicine": ["patient", "clinical", "disease", "treatment", "diagnosis", "medical"],
            "Computer Science": ["algorithm", "software", "programming", "neural network", "machine learning"],
            "Physics": ["quantum", "particle", "energy", "physics", "atom", "electron"],
            "Chemistry": ["molecule", "chemical", "reaction", "compound", "synthesis"],
            "Environmental Science": ["climate", "environment", "ecosystem", "pollution"],
            "Economics": ["market", "economic", "finance", "investment"],
            "Psychology": ["cognitive", "behavioral", "psychology", "mental"],
            "Mathematics": ["theorem", "proof", "mathematical", "equation"]
        }

        max_score = 0
        detected_field = "General Science"

        for field, keywords in field_keywords.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > max_score:
                max_score = score
                detected_field = field

        return detected_field

    def _generate_overall_assessment(self, result: PaperAnalysisResult) -> str:
        """Generate overall assessment string."""
        parts = []

        # ML adoption
        parts.append(f"ML adoption: {result.ml_adoption.ml_adoption_level}")

        # Key findings
        if result.research_outcomes.real_world_application_mentioned:
            parts.append("real-world application")
        if result.reproducibility.code_availability_mentioned:
            parts.append("code available")
        if result.impact_indicators.claims_novelty:
            parts.append("claims novelty")

        return "; ".join(parts)

    def _determine_confidence(self, text_length: int, use_llm: bool) -> str:
        """Determine analysis confidence level."""
        if use_llm and text_length > 5000:
            return "high"
        elif text_length > 2000:
            return "medium"
        else:
            return "low"

    def analyze_with_llm(self, text: str, paper_id: str) -> Dict[str, Any]:
        """
        Use LLM for enhanced analysis.

        Returns structured analysis from the LLM.
        """
        if not self.client:
            return {}

        prompt = f"""Analyze this research paper and extract structured information.

PAPER TEXT (first 4000 chars):
{text[:4000]}

Provide a JSON response with the following structure:
{{
    "field": "detected scientific field",
    "ml_adoption": {{
        "ml_frameworks_mentioned": ["list of ML frameworks"],
        "specific_models_architectures": ["list of models/architectures"],
        "compute_resources_mentioned": ["GPU, TPU, etc."],
        "ml_libraries_tools": ["list of tools"],
        "ml_adoption_level": "none|minimal|moderate|extensive",
        "ml_method_primary": "primary ML method or null",
        "integration_with_traditional_methods": true/false
    }},
    "reproducibility": {{
        "code_availability_mentioned": true/false,
        "code_repository_type": "github|gitlab|zenodo|null",
        "data_availability_mentioned": true/false,
        "methodology_detail_level": "sparse|basic|moderate|detailed",
        "hyperparameters_specified": true/false,
        "replication_feasibility": "not_feasible|difficult|moderate|feasible|highly_feasible"
    }},
    "research_outcomes": {{
        "validation_type": ["list of validation types"],
        "real_world_application_mentioned": true/false,
        "commercialization_mentioned": true/false
    }},
    "impact_indicators": {{
        "claims_novelty": true/false,
        "claims_improvement_over_existing": true/false,
        "quantitative_improvements_mentioned": true/false,
        "potential_impact_scope": "narrow|moderate|broad|transformational"
    }},
    "summary": "brief summary of the paper"
}}"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a research paper analysis expert. Provide accurate, structured analysis in JSON format."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=2048,
                response_format={"type": "json_object"}
            )

            return json.loads(response.choices[0].message.content)

        except Exception as e:
            logger.error(f"LLM analysis failed: {e}")
            return {}

    def analyze(
        self,
        text: Optional[str] = None,
        pdf_path: Optional[str] = None,
        pdf_bytes: Optional[bytes] = None,
        paper_id: Optional[str] = None,
        title: Optional[str] = None,
        abstract: Optional[str] = None,
        publication_date: Optional[str] = None
    ) -> PaperAnalysisResult:
        """
        Analyze a paper and return structured JSON result.

        Args:
            text: Paper text content
            pdf_path: Path to PDF file
            pdf_bytes: PDF file bytes
            paper_id: Paper identifier
            title: Paper title (optional)
            abstract: Paper abstract (optional)
            publication_date: Publication date (optional)

        Returns:
            PaperAnalysisResult with comprehensive analysis
        """
        # Extract text if needed
        if pdf_path:
            text = self.extract_text_from_pdf(pdf_path)
        elif pdf_bytes:
            text = self.extract_text_from_pdf_bytes(pdf_bytes)
        elif not text:
            raise ValueError("Must provide text, pdf_path, or pdf_bytes")

        # Generate paper ID if not provided
        if not paper_id:
            paper_id = f"paper_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

        # Detect field
        field = self._detect_field(text)

        # Run heuristic analysis
        ml_adoption = self._analyze_ml_adoption_heuristic(text)
        reproducibility = self._analyze_reproducibility_heuristic(text)
        research_outcomes = self._analyze_research_outcomes_heuristic(text)
        impact_indicators = self._analyze_impact_indicators_heuristic(text)

        # Extract code and repos
        code_blocks = self._extract_code_blocks(text)
        github_repos = self._extract_github_repos(text)

        # Enhance with LLM if available
        llm_analysis = {}
        if self.use_llm and self.client:
            llm_analysis = self.analyze_with_llm(text, paper_id)

            # Merge LLM results with heuristic results
            if llm_analysis:
                if "field" in llm_analysis:
                    field = llm_analysis["field"]

                if "ml_adoption" in llm_analysis:
                    llm_ml = llm_analysis["ml_adoption"]
                    ml_adoption.ml_adoption_level = llm_ml.get("ml_adoption_level", ml_adoption.ml_adoption_level)
                    ml_adoption.ml_method_primary = llm_ml.get("ml_method_primary", ml_adoption.ml_method_primary)
                    # Merge lists
                    ml_adoption.ml_frameworks_mentioned = list(set(
                        ml_adoption.ml_frameworks_mentioned + llm_ml.get("ml_frameworks_mentioned", [])
                    ))
                    ml_adoption.specific_models_architectures = list(set(
                        ml_adoption.specific_models_architectures + llm_ml.get("specific_models_architectures", [])
                    ))

        # Build result
        result = PaperAnalysisResult(
            paper_id=paper_id,
            field=field,
            publication_date=publication_date,
            title=title,
            abstract=abstract,
            ml_adoption=ml_adoption,
            reproducibility=reproducibility,
            research_outcomes=research_outcomes,
            impact_indicators=impact_indicators,
            code_blocks=code_blocks,
            github_repos=github_repos,
            text_length=len(text)
        )

        # Generate assessment
        result.overall_assessment = self._generate_overall_assessment(result)
        result.confidence_score = self._determine_confidence(len(text), bool(llm_analysis))
        result.notes = f"Evaluated from {len(text)} chars of text"

        return result

    def query_paper(
        self,
        text: str,
        question: str,
        paper_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Answer questions about a paper using RAG.

        Args:
            text: Paper text content
            question: Question to answer
            paper_id: Paper identifier (optional)

        Returns:
            Dict with answer, confidence, and sources
        """
        if not self.client:
            return {
                "answer": "LLM not available for Q&A",
                "confidence": 0.0,
                "error": "NVIDIA API key not configured"
            }

        prompt = f"""Based on the following paper content, answer the question.

PAPER CONTENT:
{text[:6000]}

QUESTION: {question}

Provide a JSON response with:
{{
    "answer": "your detailed answer",
    "confidence": 0.0-1.0,
    "relevant_quotes": ["quotes from paper supporting answer"],
    "reasoning": "brief explanation of your reasoning"
}}"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a research paper expert. Answer questions accurately based only on the provided paper content. If the answer is not in the paper, say so."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1024,
                response_format={"type": "json_object"}
            )

            result = json.loads(response.choices[0].message.content)
            result["paper_id"] = paper_id
            return result

        except Exception as e:
            logger.error(f"Q&A failed: {e}")
            return {
                "answer": f"Error processing question: {str(e)}",
                "confidence": 0.0,
                "error": str(e)
            }


def analyze_paper_to_json(
    text: Optional[str] = None,
    pdf_path: Optional[str] = None,
    paper_id: Optional[str] = None,
    use_llm: bool = True
) -> str:
    """
    Convenience function to analyze a paper and return JSON string.

    Args:
        text: Paper text content
        pdf_path: Path to PDF file
        paper_id: Paper identifier
        use_llm: Whether to use LLM for enhanced analysis

    Returns:
        JSON string with analysis results
    """
    analyzer = PaperAnalyzer(use_llm=use_llm)
    result = analyzer.analyze(text=text, pdf_path=pdf_path, paper_id=paper_id)
    return result.to_json()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python analyzer.py <pdf_path_or_text_file>")
        sys.exit(1)

    input_path = sys.argv[1]

    analyzer = PaperAnalyzer(use_llm=True)

    if input_path.endswith('.pdf'):
        result = analyzer.analyze(pdf_path=input_path)
    else:
        with open(input_path, 'r') as f:
            text = f.read()
        result = analyzer.analyze(text=text)

    print(result.to_json())
