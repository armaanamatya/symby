"""
Configuration and Constants for AI Research Impact Observatory
DGX Spark Frontier Hackathon - Symby AI Track
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import os

# Base paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "s2orc_data"
CACHE_DIR = PROJECT_ROOT / "cache"
GRAPH_CACHE_DIR = PROJECT_ROOT / "graph_cache"

# Ensure directories exist
CACHE_DIR.mkdir(exist_ok=True)
GRAPH_CACHE_DIR.mkdir(exist_ok=True)


@dataclass
class MLKeywords:
    """ML/AI related keywords for classification"""

    # Core ML techniques
    techniques: List[str] = field(default_factory=lambda: [
        'transformer', 'bert', 'gpt', 'llm', 'attention',
        'cnn', 'convolutional', 'resnet', 'vgg', 'inception',
        'rnn', 'lstm', 'gru', 'recurrent',
        'gan', 'generative adversarial', 'vae', 'autoencoder',
        'diffusion', 'stable diffusion', 'ddpm',
        'reinforcement learning', 'q-learning', 'policy gradient',
        'graph neural', 'gnn', 'gcn', 'message passing',
    ])

    # General ML terms
    general: List[str] = field(default_factory=lambda: [
        'deep learning', 'neural network', 'machine learning',
        'supervised learning', 'unsupervised learning', 'self-supervised',
        'transfer learning', 'fine-tuning', 'pre-training',
        'few-shot', 'zero-shot', 'in-context learning',
        'backpropagation', 'gradient descent', 'optimization',
        'classification', 'regression', 'clustering', 'embedding',
    ])

    # Frameworks and tools
    frameworks: List[str] = field(default_factory=lambda: [
        'pytorch', 'tensorflow', 'keras', 'jax', 'flax',
        'huggingface', 'transformers', 'scikit-learn',
        'cuda', 'gpu', 'nvidia', 'dgx', 'rapids', 'cugraph',
    ])

    # Landmark models/methods
    landmarks: List[str] = field(default_factory=lambda: [
        'alphafold', 'alexnet', 'imagenet', 'word2vec', 'glove',
        'elmo', 'roberta', 'xlnet', 't5', 'clip', 'dall-e',
        'chatgpt', 'gpt-4', 'gemini', 'claude', 'llama',
        'segment anything', 'sam', 'dino', 'moco', 'simclr',
    ])

    @property
    def all_keywords(self) -> List[str]:
        return self.techniques + self.general + self.frameworks + self.landmarks


@dataclass
class ReproducibilityKeywords:
    """Keywords indicating reproducibility/quality"""

    code: List[str] = field(default_factory=lambda: [
        'github', 'gitlab', 'bitbucket', 'code available',
        'source code', 'repository', 'implementation available',
        'colab', 'jupyter', 'notebook', 'docker', 'container',
    ])

    data: List[str] = field(default_factory=lambda: [
        'dataset available', 'data available', 'benchmark',
        'open data', 'public dataset', 'released dataset',
    ])

    reproducibility: List[str] = field(default_factory=lambda: [
        'reproducible', 'replication', 'replicated',
        'supplementary material', 'appendix', 'ablation',
        'error bar', 'confidence interval', 'statistical significance',
    ])

    @property
    def all_keywords(self) -> List[str]:
        return self.code + self.data + self.reproducibility


@dataclass
class LandmarkMethods:
    """Historical ML breakthrough methods with years"""

    methods: Dict[str, Dict] = field(default_factory=lambda: {
        'alexnet': {'year': 2012, 'field': 'Computer Vision', 'impact': 'high'},
        'word2vec': {'year': 2013, 'field': 'NLP', 'impact': 'high'},
        'gan': {'year': 2014, 'field': 'Generative', 'impact': 'high'},
        'resnet': {'year': 2015, 'field': 'Computer Vision', 'impact': 'high'},
        'transformer': {'year': 2017, 'field': 'NLP', 'impact': 'transformational'},
        'bert': {'year': 2018, 'field': 'NLP', 'impact': 'transformational'},
        'gpt': {'year': 2018, 'field': 'NLP', 'impact': 'transformational'},
        'alphafold': {'year': 2020, 'field': 'Biology', 'impact': 'transformational'},
        'clip': {'year': 2021, 'field': 'Multimodal', 'impact': 'high'},
        'diffusion': {'year': 2020, 'field': 'Generative', 'impact': 'high'},
        'chatgpt': {'year': 2022, 'field': 'NLP', 'impact': 'transformational'},
    })


@dataclass
class ScientificFields:
    """Scientific field mappings"""

    # Mapping from S2 fields to normalized names
    field_mapping: Dict[str, str] = field(default_factory=lambda: {
        'Computer Science': 'Computer Science',
        'Biology': 'Biology',
        'Medicine': 'Medicine',
        'Physics': 'Physics',
        'Chemistry': 'Chemistry',
        'Materials Science': 'Materials Science',
        'Environmental Science': 'Environmental Science',
        'Psychology': 'Psychology',
        'Economics': 'Economics',
        'Sociology': 'Sociology',
        'Engineering': 'Engineering',
        'Mathematics': 'Mathematics',
        'Geology': 'Geology',
        'Geography': 'Geography',
        'Political Science': 'Political Science',
        'NA': 'Multidisciplinary',
    })

    # Domain science categories (non-CS)
    domain_sciences: List[str] = field(default_factory=lambda: [
        'Biology', 'Medicine', 'Physics', 'Chemistry',
        'Materials Science', 'Environmental Science',
        'Psychology', 'Economics', 'Engineering',
    ])


@dataclass
class APIConfig:
    """Semantic Scholar API configuration"""

    base_url: str = "https://api.semanticscholar.org/graph/v1"
    api_key: Optional[str] = field(default_factory=lambda: os.environ.get('S2_API_KEY'))

    # Rate limiting
    requests_per_second_with_key: float = 10.0
    requests_per_second_without_key: float = 0.33  # 1 per 3 seconds

    # Batch settings
    batch_size: int = 100
    max_papers_per_request: int = 500

    # Timeouts
    request_timeout: int = 30
    batch_timeout: int = 60

    # Fields to fetch
    paper_fields: List[str] = field(default_factory=lambda: [
        'title', 'year', 'citationCount', 'referenceCount',
        'fieldsOfStudy', 'authors', 'abstract',
        'citations.paperId', 'citations.title', 'citations.year',
        'references.paperId', 'references.title', 'references.year',
    ])


@dataclass
class DashboardConfig:
    """Dashboard configuration"""

    # Colors
    primary_color: str = "#76b900"  # NVIDIA green
    secondary_color: str = "#667eea"
    accent_color: str = "#764ba2"
    warning_color: str = "#ffa500"
    error_color: str = "#dc3545"
    success_color: str = "#28a745"

    # Layout
    default_papers_to_load: int = 5000
    max_papers_to_load: int = 100000
    cache_ttl_hours: int = 1

    # Visualizations
    chart_height: int = 400
    network_height: int = 500


# Global instances
ML_KEYWORDS = MLKeywords()
REPRO_KEYWORDS = ReproducibilityKeywords()
LANDMARK_METHODS = LandmarkMethods()
SCIENTIFIC_FIELDS = ScientificFields()
API_CONFIG = APIConfig()
DASHBOARD_CONFIG = DashboardConfig()
