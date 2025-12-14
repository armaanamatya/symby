"""
Paper Evaluator for AI Research Impact Observatory
Extracts structured metadata from papers according to evaluation_structure.pdf

Transforms simple JSONL (id, text, field, publication_date) into rich evaluation
with ml_adoption, reproducibility, research_outcomes, and impact_indicators.
"""

import re
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, asdict
from collections import defaultdict


@dataclass
class MLAdoption:
    """ML Adoption markers for a paper"""
    ml_frameworks_mentioned: List[str]
    specific_models_architectures: List[str]
    compute_resources_mentioned: List[str]
    datasets_referenced: List[str]
    ml_libraries_tools: List[str]
    ml_adoption_level: str  # "none" | "minimal" | "moderate" | "substantial" | "core"
    ml_application_domain: str
    ml_method_primary: Optional[str]
    integration_with_traditional_methods: bool


@dataclass
class Reproducibility:
    """Reproducibility indicators for a paper"""
    code_availability_mentioned: bool
    code_repository_type: Optional[str]
    data_availability_mentioned: bool
    data_sharing_statement: Optional[str]
    methodology_detail_level: str  # "minimal" | "moderate" | "detailed" | "comprehensive"
    hyperparameters_specified: bool
    computational_environment_described: bool
    preprocessing_steps_detailed: bool
    statistical_methods_described: bool
    replication_feasibility: str  # "not_feasible" | "difficult" | "moderate" | "straightforward"


@dataclass
class ResearchOutcomes:
    """Research outcomes and impact signals"""
    mentions_clinical_trials: bool
    clinical_trial_identifiers: List[str]
    mentions_patents: bool
    patent_numbers: List[str]
    mentions_corrections: bool
    mentions_retractions: bool
    validation_type: List[str]
    real_world_application_mentioned: bool
    commercialization_mentioned: bool
    regulatory_approval_mentioned: bool


@dataclass
class ImpactIndicators:
    """Impact and significance indicators"""
    claims_novelty: bool
    claims_improvement_over_existing: bool
    quantitative_improvements_mentioned: bool
    comparison_to_baseline: bool
    mentions_policy_implications: bool
    mentions_clinical_guidelines: bool
    mentions_media_coverage: bool
    collaboration_indicators: List[str]
    funding_sources_mentioned: List[str]
    potential_impact_scope: str  # "narrow" | "moderate" | "broad" | "transformative"


class PaperEvaluator:
    """
    Comprehensive paper evaluator that extracts structured metadata.

    Uses keyword matching, regex patterns, and heuristics to extract
    all fields from the evaluation structure.
    """

    # ML Frameworks
    ML_FRAMEWORKS = {
        'PyTorch': ['pytorch', 'torch.nn', 'torch.cuda'],
        'TensorFlow': ['tensorflow', 'keras', 'tf.keras'],
        'JAX': ['jax', 'flax', 'haiku'],
        'scikit-learn': ['scikit-learn', 'sklearn', 'scikitlearn'],
        'Hugging Face': ['huggingface', 'transformers', 'diffusers'],
        'MXNet': ['mxnet', 'gluon'],
        'Caffe': ['caffe'],
        'Theano': ['theano'],
    }

    # ML Models/Architectures
    ML_MODELS = {
        'Transformer': ['transformer', 'attention mechanism', 'self-attention'],
        'BERT': ['bert', 'roberta', 'distilbert', 'albert'],
        'GPT': ['gpt', 'gpt-2', 'gpt-3', 'gpt-4', 'generative pre-trained'],
        'ResNet': ['resnet', 'residual network'],
        'CNN': ['convolutional neural', 'cnn', 'convnet'],
        'RNN': ['recurrent neural', 'rnn', 'lstm', 'gru'],
        'GAN': ['generative adversarial', 'gan'],
        'VAE': ['variational autoencoder', 'vae'],
        'Diffusion': ['diffusion model', 'stable diffusion', 'ddpm'],
        'U-Net': ['u-net', 'unet'],
        'YOLO': ['yolo', 'you only look once'],
        'AlphaFold': ['alphafold', 'alpha fold'],
        'Graph Neural Network': ['graph neural', 'gnn', 'gcn'],
    }

    # Compute Resources
    COMPUTE_RESOURCES = {
        'GPU': ['gpu', 'nvidia', 'cuda', 'cudnn'],
        'TPU': ['tpu', 'tensor processing unit'],
        'HPC': ['hpc', 'high-performance computing', 'supercomputer'],
        'Cloud': ['aws', 'azure', 'gcp', 'google cloud', 'cloud computing'],
        'DGX': ['dgx', 'a100', 'v100', 'h100'],
        'Distributed Training': ['distributed training', 'multi-gpu', 'data parallel'],
    }

    # ML Libraries/Tools
    ML_LIBRARIES = {
        'NumPy': ['numpy', 'np.array'],
        'Pandas': ['pandas', 'dataframe'],
        'Matplotlib': ['matplotlib', 'pyplot'],
        'OpenCV': ['opencv', 'cv2'],
        'SpaCy': ['spacy'],
        'NLTK': ['nltk', 'natural language toolkit'],
        'XGBoost': ['xgboost', 'xgb'],
        'LightGBM': ['lightgbm', 'lgbm'],
        'CatBoost': ['catboost'],
        'Ray': ['ray', 'ray tune'],
        'Weights & Biases': ['wandb', 'weights and biases'],
        'MLflow': ['mlflow'],
        'TensorBoard': ['tensorboard'],
        'Optuna': ['optuna'],
    }

    # Datasets (common ones)
    DATASETS = {
        'ImageNet': ['imagenet'],
        'COCO': ['coco dataset', 'ms-coco'],
        'CIFAR': ['cifar-10', 'cifar-100', 'cifar'],
        'MNIST': ['mnist'],
        'WikiText': ['wikitext'],
        'SQuAD': ['squad dataset', 'stanford question'],
        'GLUE': ['glue benchmark', 'general language understanding'],
        'CommonCrawl': ['common crawl', 'commoncrawl'],
        'BookCorpus': ['bookcorpus', 'book corpus'],
        'OpenWebText': ['openwebtext'],
    }

    # Code availability patterns
    CODE_PATTERNS = [
        r'github\.com/[\w-]+/[\w-]+',
        r'gitlab\.com/[\w-]+/[\w-]+',
        r'code\s+(?:is\s+)?available',
        r'source\s+code',
        r'implementation\s+available',
        r'publicly\s+available\s+code',
        r'our\s+code',
        r'released\s+code',
    ]

    # Repository types
    REPO_TYPES = {
        'GitHub': ['github.com'],
        'GitLab': ['gitlab.com'],
        'Bitbucket': ['bitbucket.org'],
        'Hugging Face': ['huggingface.co'],
    }

    # Data sharing patterns
    DATA_SHARING_PATTERNS = [
        r'data\s+(?:is\s+)?available',
        r'dataset\s+(?:is\s+)?available',
        r'publicly\s+available\s+data',
        r'open\s+data',
        r'data\s+sharing',
        r'supplementary\s+(?:data|material)',
    ]

    # Clinical trial patterns
    CLINICAL_TRIAL_PATTERNS = [
        r'clinical\s+trial',
        r'NCT\d{8}',  # ClinicalTrials.gov identifier
        r'randomized\s+controlled\s+trial',
        r'phase\s+[I-IV]+\s+trial',
    ]

    # Patent patterns
    PATENT_PATTERNS = [
        r'patent',
        r'US\d{7,10}',  # US patent number
        r'EP\d{7}',  # European patent
        r'WO\d{4}/\d{6}',  # WIPO patent
    ]

    # Retraction/correction patterns
    RETRACTION_PATTERNS = [
        r'retract(?:ed|ion)',
        r'correction',
        r'erratum',
        r'corrigendum',
        r'expression\s+of\s+concern',
        r'withdrawn',
    ]

    # Novelty claims
    NOVELTY_PATTERNS = [
        r'novel',
        r'first\s+(?:to|time)',
        r'new\s+(?:method|approach|technique)',
        r'breakthrough',
        r'unprecedented',
        r'state-of-the-art',
        r'sota',
    ]

    # Improvement claims
    IMPROVEMENT_PATTERNS = [
        r'outperform',
        r'better\s+than',
        r'improve(?:s|d|ment)',
        r'superior\s+to',
        r'exceed',
        r'surpass',
    ]

    # Quantitative improvements
    QUANTITATIVE_PATTERNS = [
        r'\d+(?:\.\d+)?%\s+(?:improvement|increase|reduction)',
        r'(?:improved|increased|reduced)\s+by\s+\d+',
        r'\d+(?:\.\d+)?x\s+(?:faster|speedup|acceleration)',
    ]

    # Baseline comparison
    BASELINE_PATTERNS = [
        r'baseline',
        r'compared\s+to',
        r'versus',
        r'vs\.?',
        r'benchmark',
    ]

    # Funding sources
    FUNDING_PATTERNS = {
        'NSF': ['national science foundation', 'nsf grant'],
        'NIH': ['national institutes of health', 'nih grant'],
        'DARPA': ['darpa'],
        'DOE': ['department of energy', 'doe grant'],
        'ERC': ['european research council', 'erc grant'],
        'Industry': ['sponsored by', 'funding from'],
    }

    def __init__(self):
        """Initialize patterns and compile regex"""
        self._compile_patterns()

    def _compile_patterns(self):
        """Compile regex patterns for efficiency"""
        self.code_regex = [re.compile(p, re.IGNORECASE) for p in self.CODE_PATTERNS]
        self.data_regex = [re.compile(p, re.IGNORECASE) for p in self.DATA_SHARING_PATTERNS]
        self.clinical_regex = [re.compile(p, re.IGNORECASE) for p in self.CLINICAL_TRIAL_PATTERNS]
        self.patent_regex = [re.compile(p, re.IGNORECASE) for p in self.PATENT_PATTERNS]
        self.retraction_regex = [re.compile(p, re.IGNORECASE) for p in self.RETRACTION_PATTERNS]
        self.novelty_regex = [re.compile(p, re.IGNORECASE) for p in self.NOVELTY_PATTERNS]
        self.improvement_regex = [re.compile(p, re.IGNORECASE) for p in self.IMPROVEMENT_PATTERNS]
        self.quantitative_regex = [re.compile(p, re.IGNORECASE) for p in self.QUANTITATIVE_PATTERNS]
        self.baseline_regex = [re.compile(p, re.IGNORECASE) for p in self.BASELINE_PATTERNS]

    def _match_keywords(self, text: str, keyword_dict: Dict[str, List[str]]) -> List[str]:
        """Match keywords from a dictionary against text"""
        text_lower = text.lower()
        matches = []
        for category, keywords in keyword_dict.items():
            if any(kw in text_lower for kw in keywords):
                matches.append(category)
        return matches

    def _match_patterns(self, text: str, regex_list: List[re.Pattern]) -> bool:
        """Check if any pattern matches"""
        return any(pattern.search(text) for pattern in regex_list)

    def _extract_matches(self, text: str, regex_list: List[re.Pattern]) -> List[str]:
        """Extract all matches from patterns"""
        matches = []
        for pattern in regex_list:
            found = pattern.findall(text)
            matches.extend(found)
        return matches[:10]  # Limit to 10

    def evaluate_ml_adoption(self, text: str, field: str) -> MLAdoption:
        """Evaluate ML adoption level and extract ML-related features"""
        frameworks = self._match_keywords(text, self.ML_FRAMEWORKS)
        models = self._match_keywords(text, self.ML_MODELS)
        compute = self._match_keywords(text, self.COMPUTE_RESOURCES)
        datasets = self._match_keywords(text, self.DATASETS)
        libraries = self._match_keywords(text, self.ML_LIBRARIES)

        # Determine adoption level
        ml_score = len(frameworks) * 2 + len(models) * 3 + len(libraries)

        if ml_score == 0:
            adoption_level = "none"
        elif ml_score <= 3:
            adoption_level = "minimal"
        elif ml_score <= 8:
            adoption_level = "moderate"
        elif ml_score <= 15:
            adoption_level = "substantial"
        else:
            adoption_level = "core"

        # Primary method
        primary_method = models[0] if models else None

        # Integration with traditional methods
        text_lower = text.lower()
        integration_keywords = ['traditional', 'classical', 'conventional', 'hybrid approach', 'combined with']
        integration = any(kw in text_lower for kw in integration_keywords) and adoption_level != "none"

        # Application domain
        domain = field if field else "unknown"

        return MLAdoption(
            ml_frameworks_mentioned=frameworks,
            specific_models_architectures=models,
            compute_resources_mentioned=compute,
            datasets_referenced=datasets,
            ml_libraries_tools=libraries,
            ml_adoption_level=adoption_level,
            ml_application_domain=domain,
            ml_method_primary=primary_method,
            integration_with_traditional_methods=integration
        )

    def evaluate_reproducibility(self, text: str) -> Reproducibility:
        """Evaluate reproducibility and code/data availability"""
        text_lower = text.lower()

        # Code availability
        code_available = self._match_patterns(text, self.code_regex)

        # Repository type
        repo_type = None
        for repo, patterns in self.REPO_TYPES.items():
            if any(p in text_lower for p in patterns):
                repo_type = repo
                break

        # Data availability
        data_available = self._match_patterns(text, self.data_regex)

        # Data sharing statement
        data_statement = None
        if data_available:
            data_statement = "available"

        # Methodology detail level
        method_indicators = {
            'comprehensive': ['detailed', 'comprehensive', 'step-by-step', 'thoroughly'],
            'detailed': ['method', 'procedure', 'algorithm', 'implementation'],
            'moderate': ['approach', 'technique'],
            'minimal': []
        }

        detail_scores = {level: sum(1 for kw in keywords if kw in text_lower)
                        for level, keywords in method_indicators.items()}

        if detail_scores['comprehensive'] >= 3:
            methodology_level = "comprehensive"
        elif detail_scores['detailed'] >= 2:
            methodology_level = "detailed"
        elif detail_scores['moderate'] >= 1:
            methodology_level = "moderate"
        else:
            methodology_level = "minimal"

        # Hyperparameters
        hyperparams = any(kw in text_lower for kw in [
            'hyperparameter', 'learning rate', 'batch size', 'epoch', 'dropout rate'
        ])

        # Computational environment
        comp_env = any(kw in text_lower for kw in [
            'gpu', 'cpu', 'memory', 'computation time', 'hardware', 'nvidia', 'cuda'
        ])

        # Preprocessing steps
        preprocessing = any(kw in text_lower for kw in [
            'preprocessing', 'normalization', 'data cleaning', 'feature engineering', 'augmentation'
        ])

        # Statistical methods
        stats = any(kw in text_lower for kw in [
            'statistical', 'significance', 'p-value', 'confidence interval', 't-test', 'anova'
        ])

        # Replication feasibility
        repro_score = sum([code_available, data_available, hyperparams, comp_env])
        if repro_score >= 3:
            feasibility = "straightforward"
        elif repro_score == 2:
            feasibility = "moderate"
        elif repro_score == 1:
            feasibility = "difficult"
        else:
            feasibility = "not_feasible"

        return Reproducibility(
            code_availability_mentioned=code_available,
            code_repository_type=repo_type,
            data_availability_mentioned=data_available,
            data_sharing_statement=data_statement,
            methodology_detail_level=methodology_level,
            hyperparameters_specified=hyperparams,
            computational_environment_described=comp_env,
            preprocessing_steps_detailed=preprocessing,
            statistical_methods_described=stats,
            replication_feasibility=feasibility
        )

    def evaluate_research_outcomes(self, text: str) -> ResearchOutcomes:
        """Evaluate research outcomes and real-world impact"""
        text_lower = text.lower()

        # Clinical trials
        mentions_trials = self._match_patterns(text, self.clinical_regex)
        trial_ids = self._extract_matches(text, [re.compile(r'NCT\d{8}', re.IGNORECASE)])

        # Patents
        mentions_patents = self._match_patterns(text, self.patent_regex)
        patent_nums = self._extract_matches(text, [
            re.compile(r'US\d{7,10}|EP\d{7}|WO\d{4}/\d{6}', re.IGNORECASE)
        ])

        # Corrections/retractions
        mentions_corrections = 'correction' in text_lower or 'erratum' in text_lower
        mentions_retractions = self._match_patterns(text, self.retraction_regex)

        # Validation types
        validation_types = []
        validation_keywords = {
            'cross-validation': ['cross-validation', 'k-fold', 'cv'],
            'statistical validation': ['statistical', 'significance test'],
            'experimental validation': ['experiment', 'validated experimentally'],
            'clinical validation': ['clinical trial', 'patient cohort'],
        }

        for val_type, keywords in validation_keywords.items():
            if any(kw in text_lower for kw in keywords):
                validation_types.append(val_type)

        # Real-world application
        real_world = any(kw in text_lower for kw in [
            'real-world', 'practical application', 'deployment', 'production', 'clinical use'
        ])

        # Commercialization
        commercialization = any(kw in text_lower for kw in [
            'commercial', 'product', 'startup', 'company', 'market'
        ])

        # Regulatory approval
        regulatory = any(kw in text_lower for kw in [
            'fda approval', 'fda cleared', 'ema approval', 'regulatory', 'ce mark'
        ])

        return ResearchOutcomes(
            mentions_clinical_trials=mentions_trials,
            clinical_trial_identifiers=trial_ids,
            mentions_patents=mentions_patents,
            patent_numbers=patent_nums,
            mentions_corrections=mentions_corrections,
            mentions_retractions=mentions_retractions,
            validation_type=validation_types,
            real_world_application_mentioned=real_world,
            commercialization_mentioned=commercialization,
            regulatory_approval_mentioned=regulatory
        )

    def evaluate_impact_indicators(self, text: str) -> ImpactIndicators:
        """Evaluate impact and significance indicators"""
        text_lower = text.lower()

        # Novelty claims
        claims_novelty = self._match_patterns(text, self.novelty_regex)

        # Improvement claims
        claims_improvement = self._match_patterns(text, self.improvement_regex)

        # Quantitative improvements
        quant_improvements = self._match_patterns(text, self.quantitative_regex)

        # Baseline comparison
        baseline_comparison = self._match_patterns(text, self.baseline_regex)

        # Policy implications
        policy = any(kw in text_lower for kw in [
            'policy', 'regulation', 'guideline', 'recommendation'
        ])

        # Clinical guidelines
        clinical_guidelines = any(kw in text_lower for kw in [
            'clinical guideline', 'treatment guideline', 'diagnostic criteria'
        ])

        # Media coverage
        media = any(kw in text_lower for kw in [
            'press release', 'media', 'news', 'public attention'
        ])

        # Collaboration indicators
        collaboration = []
        if 'collaboration' in text_lower or 'collaborative' in text_lower:
            collaboration.append('multi-institutional')
        if 'international' in text_lower:
            collaboration.append('international')
        if 'consortium' in text_lower:
            collaboration.append('consortium')

        # Funding sources
        funding = self._match_keywords(text, self.FUNDING_PATTERNS)

        # Impact scope
        impact_score = sum([
            claims_novelty * 2,
            claims_improvement,
            quant_improvements,
            len(collaboration),
            len(funding),
        ])

        if impact_score >= 8:
            impact_scope = "transformative"
        elif impact_score >= 5:
            impact_scope = "broad"
        elif impact_score >= 2:
            impact_scope = "moderate"
        else:
            impact_scope = "narrow"

        return ImpactIndicators(
            claims_novelty=claims_novelty,
            claims_improvement_over_existing=claims_improvement,
            quantitative_improvements_mentioned=quant_improvements,
            comparison_to_baseline=baseline_comparison,
            mentions_policy_implications=policy,
            mentions_clinical_guidelines=clinical_guidelines,
            mentions_media_coverage=media,
            collaboration_indicators=collaboration,
            funding_sources_mentioned=funding,
            potential_impact_scope=impact_scope
        )

    def evaluate_paper(self, paper_dict: Dict) -> Dict:
        """
        Evaluate a paper and return full evaluation structure.

        Args:
            paper_dict: Dict with 'id', 'text', 'field', 'publication_date'

        Returns:
            Dict with full evaluation structure matching evaluation_structure.pdf
        """
        paper_id = paper_dict.get('id', '')
        text = paper_dict.get('text', '')
        field = paper_dict.get('field', '')
        pub_date = paper_dict.get('publication_date')

        # Evaluate all aspects
        ml_adoption = self.evaluate_ml_adoption(text, field)
        reproducibility = self.evaluate_reproducibility(text)
        research_outcomes = self.evaluate_research_outcomes(text)
        impact_indicators = self.evaluate_impact_indicators(text)

        # Generate overall assessment
        assessment_parts = []
        if ml_adoption.ml_adoption_level != "none":
            assessment_parts.append(f"ML adoption: {ml_adoption.ml_adoption_level}")
        if reproducibility.code_availability_mentioned:
            assessment_parts.append("code available")
        if research_outcomes.real_world_application_mentioned:
            assessment_parts.append("real-world application")

        overall_assessment = "; ".join(assessment_parts) if assessment_parts else "Traditional research"

        # Confidence score
        text_length = len(text)
        if text_length > 5000:
            confidence = "high"
        elif text_length > 1000:
            confidence = "medium"
        else:
            confidence = "low"

        return {
            "paper_id": paper_id,
            "field": field,
            "publication_date": str(pub_date) if pub_date else None,
            "ml_adoption": asdict(ml_adoption),
            "reproducibility": asdict(reproducibility),
            "research_outcomes": asdict(research_outcomes),
            "impact_indicators": asdict(impact_indicators),
            "overall_assessment": overall_assessment,
            "confidence_score": confidence,
            "notes": f"Evaluated from {text_length} chars of text"
        }


if __name__ == "__main__":
    # Test the evaluator
    evaluator = PaperEvaluator()

    # Test paper
    test_paper = {
        'id': 'test123',
        'text': '''Deep Learning for Drug Discovery

We present a novel deep learning approach for molecular property prediction
using graph neural networks. Our method, implemented in PyTorch, achieves
state-of-the-art results on the MoleculeNet benchmark, outperforming previous
methods by 15%. We trained on NVIDIA A100 GPUs and make our code available
on GitHub. The model uses a transformer-based architecture with attention
mechanisms. We validated our approach on FDA-approved drugs and show
significant improvement over traditional methods.''',
        'field': 'Chemistry',
        'publication_date': '2023'
    }

    evaluation = evaluator.evaluate_paper(test_paper)

    print("Paper Evaluation:")
    print(f"Field: {evaluation['field']}")
    print(f"\nML Adoption Level: {evaluation['ml_adoption']['ml_adoption_level']}")
    print(f"Frameworks: {evaluation['ml_adoption']['ml_frameworks_mentioned']}")
    print(f"Models: {evaluation['ml_adoption']['specific_models_architectures']}")
    print(f"\nCode Available: {evaluation['reproducibility']['code_availability_mentioned']}")
    print(f"Replication: {evaluation['reproducibility']['replication_feasibility']}")
    print(f"\nNovelty Claims: {evaluation['impact_indicators']['claims_novelty']}")
    print(f"Impact Scope: {evaluation['impact_indicators']['potential_impact_scope']}")
    print(f"\nOverall: {evaluation['overall_assessment']}")
    print(f"Confidence: {evaluation['confidence_score']}")
