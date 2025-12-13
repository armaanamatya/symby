"""
Data Loader for S2ORC Scientific Papers Dataset
Optimized for DGX Spark with RAPIDS/cuDF support

Handles both .json (uncompressed JSONL) and .jsonl files
"""

import os
import gzip
import json
from pathlib import Path
from typing import Generator, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import hashlib
import pickle

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    def tqdm(x, **kwargs):
        return x

# Try GPU-accelerated loading first (for DGX Spark)
try:
    import cudf as pd
    GPU_AVAILABLE = True
    print("[RAPIDS] Using cuDF for GPU-accelerated data loading")
except ImportError:
    import pandas as pd
    GPU_AVAILABLE = False

try:
    import orjson
    USE_ORJSON = True
except ImportError:
    USE_ORJSON = False


class MLClassifier:
    """ML/AI paper classification with comprehensive keyword matching"""

    # Core ML techniques (high signal)
    TECHNIQUES = {
        'transformer', 'attention mechanism', 'self-attention',
        'bert', 'gpt', 'llm', 'large language model',
        'cnn', 'convolutional neural', 'resnet', 'vgg', 'unet',
        'rnn', 'lstm', 'gru', 'recurrent neural',
        'gan', 'generative adversarial', 'vae', 'variational autoencoder',
        'diffusion model', 'stable diffusion', 'ddpm', 'score matching',
        'reinforcement learning', 'q-learning', 'policy gradient', 'ppo', 'dqn',
        'graph neural', 'gnn', 'gcn', 'message passing', 'graph attention',
        'autoencoder', 'encoder-decoder', 'seq2seq',
    }

    # General ML terms (medium signal)
    GENERAL = {
        'deep learning', 'neural network', 'machine learning',
        'supervised learning', 'unsupervised learning', 'self-supervised',
        'transfer learning', 'fine-tuning', 'pre-training', 'pretrained',
        'few-shot', 'zero-shot', 'in-context learning', 'prompt engineering',
        'backpropagation', 'gradient descent', 'stochastic gradient',
        'batch normalization', 'dropout', 'regularization',
        'classification', 'object detection', 'semantic segmentation',
        'natural language processing', 'computer vision', 'speech recognition',
        'embedding', 'word2vec', 'glove', 'fasttext',
    }

    # Frameworks (medium signal)
    FRAMEWORKS = {
        'pytorch', 'tensorflow', 'keras', 'jax', 'flax',
        'huggingface', 'transformers library', 'scikit-learn', 'sklearn',
        'cuda', 'cudnn', 'tensorrt', 'onnx',
    }

    # Landmark models (high signal)
    LANDMARKS = {
        'alphafold', 'alexnet', 'imagenet', 'clip', 'dall-e', 'dalle',
        'chatgpt', 'gpt-4', 'gpt-3', 'gemini', 'llama', 'mistral',
        'stable diffusion', 'midjourney', 'sam', 'segment anything',
        'yolo', 'faster rcnn', 'mask rcnn',
    }

    # Reproducibility indicators
    REPRO_CODE = {
        'github.com', 'gitlab.com', 'bitbucket', 'code available',
        'source code', 'implementation available', 'our code',
        'colab', 'jupyter notebook', 'docker', 'reproducible',
    }

    REPRO_DATA = {
        'dataset available', 'data available', 'benchmark dataset',
        'open data', 'public dataset', 'released dataset',
        'supplementary material', 'appendix',
    }

    @classmethod
    def classify(cls, text: str) -> Dict:
        """Classify a paper and extract features"""
        text_lower = text.lower()

        # Count keyword matches
        technique_score = sum(2 for kw in cls.TECHNIQUES if kw in text_lower)
        general_score = sum(1 for kw in cls.GENERAL if kw in text_lower)
        framework_score = sum(1 for kw in cls.FRAMEWORKS if kw in text_lower)
        landmark_score = sum(3 for kw in cls.LANDMARKS if kw in text_lower)

        ml_score = technique_score + general_score + framework_score + landmark_score
        is_ml_paper = ml_score >= 3

        # Reproducibility
        code_score = sum(1 for kw in cls.REPRO_CODE if kw in text_lower)
        data_score = sum(1 for kw in cls.REPRO_DATA if kw in text_lower)
        repro_score = code_score + data_score
        has_code = code_score >= 1
        has_data = data_score >= 1

        # Extract mentioned techniques
        techniques_found = [kw for kw in cls.TECHNIQUES if kw in text_lower]
        landmarks_found = [kw for kw in cls.LANDMARKS if kw in text_lower]

        return {
            'ml_score': ml_score,
            'is_ml_paper': is_ml_paper,
            'repro_score': repro_score,
            'has_code': has_code,
            'has_data': has_data,
            'techniques': techniques_found[:5],  # Limit for storage
            'landmarks': landmarks_found[:3],
        }


class DataLoader:
    """
    High-performance data loader for S2ORC dataset

    Supports:
    - Both .json and .jsonl files (uncompressed JSONL format)
    - Optional .gz compressed files
    - GPU acceleration via cuDF when available
    - Caching for fast repeated loads
    """

    def __init__(self, data_dir: str = "s2orc_data", cache_dir: str = "cache"):
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

    def discover_datasets(self) -> List[Dict]:
        """Discover all available dataset directories"""
        datasets = []

        if not self.data_dir.exists():
            print(f"Warning: Data directory {self.data_dir} does not exist")
            return datasets

        for item in self.data_dir.iterdir():
            if item.is_dir() and ',' in item.name:
                parts = item.name.split(',')
                field = parts[0]
                year_range = parts[1] if len(parts) > 1 else "unknown"

                # Count files
                file_count = sum(1 for _ in item.rglob('*.json')) + sum(1 for _ in item.rglob('*.jsonl'))

                datasets.append({
                    'path': item,
                    'field': field,
                    'year_range': year_range,
                    'name': item.name,
                    'file_count': file_count
                })

        return sorted(datasets, key=lambda x: x['name'])

    def _load_jsonl_file(self, filepath: Path) -> Generator[Dict, None, None]:
        """Load a JSONL file (supports both compressed and uncompressed)"""
        is_gzipped = str(filepath).endswith('.gz')

        try:
            if is_gzipped:
                with gzip.open(filepath, 'rt', encoding='utf-8') as f:
                    for line in f:
                        try:
                            if USE_ORJSON:
                                yield orjson.loads(line)
                            else:
                                yield json.loads(line)
                        except (json.JSONDecodeError, Exception):
                            continue
            else:
                with open(filepath, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            if USE_ORJSON:
                                yield orjson.loads(line)
                            else:
                                yield json.loads(line)
                        except (json.JSONDecodeError, Exception):
                            continue
        except Exception as e:
            print(f"Error loading {filepath}: {e}")

    def _extract_paper_features(self, paper: Dict, source_field: str, split: str) -> Dict:
        """Extract all relevant features from a paper record"""
        text = paper.get('text', '')
        metadata = paper.get('metadata', {})

        # Get title (first line of text)
        title = text.split('\n')[0][:300] if text else ""

        # ML classification
        ml_features = MLClassifier.classify(text)

        # Extract year
        year = metadata.get('year', 0)
        if not year:
            # Try to extract from created field
            created = paper.get('created', '')
            if created and len(created) >= 4:
                try:
                    year = int(created[:4])
                except:
                    year = 0

        # Get fields of study
        s2_fields = metadata.get('s2fieldsofstudy', []) or []
        ext_fields = metadata.get('extfieldsofstudy', []) or []
        all_fields = list(set(s2_fields + ext_fields))

        return {
            'id': str(paper.get('id', '')),
            'title': title,
            'year': year,
            'source_field': source_field,
            'split': split,
            's2_fields': s2_fields,
            'ext_fields': ext_fields,
            'primary_field': all_fields[0] if all_fields else source_field,
            'text_length': len(text),
            'created': paper.get('created', ''),
            **ml_features
        }

    def _get_cache_key(self, max_papers: Optional[int], fields_filter: Optional[List[str]]) -> str:
        """Generate cache key based on parameters"""
        key_str = f"{max_papers}_{fields_filter}"
        return hashlib.md5(key_str.encode()).hexdigest()[:12]

    def load_all_papers(self,
                        max_papers: Optional[int] = None,
                        fields_filter: Optional[List[str]] = None,
                        splits: List[str] = None,
                        use_cache: bool = True,
                        show_progress: bool = True) -> pd.DataFrame:
        """
        Load all papers from available datasets

        Args:
            max_papers: Maximum number of papers to load (None = all)
            fields_filter: Only load specific fields
            splits: Which splits to load ('train', 'val', 'test')
            use_cache: Whether to use/save cache
            show_progress: Show progress bar

        Returns:
            DataFrame with paper features
        """
        if splits is None:
            splits = ['train', 'val', 'test']

        # Check cache
        cache_key = self._get_cache_key(max_papers, fields_filter)
        cache_file = self.cache_dir / f"papers_{cache_key}.pkl"

        if use_cache and cache_file.exists():
            print(f"Loading from cache: {cache_file}")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)

        datasets = self.discover_datasets()
        all_papers = []
        paper_count = 0

        print(f"Found {len(datasets)} dataset directories")

        iterator = tqdm(datasets, desc="Loading datasets") if show_progress else datasets

        for dataset in iterator:
            # Apply field filter
            if fields_filter and dataset['field'] not in fields_filter:
                continue

            source_field = self._normalize_field(dataset['field'])

            for split in splits:
                split_dir = dataset['path'] / split
                if not split_dir.exists():
                    continue

                # Find all data files
                data_files = (
                    list(split_dir.glob('*.json')) +
                    list(split_dir.glob('*.jsonl')) +
                    list(split_dir.glob('*.json.gz')) +
                    list(split_dir.glob('*.jsonl.gz'))
                )

                for data_file in data_files:
                    for paper in self._load_jsonl_file(data_file):
                        features = self._extract_paper_features(paper, source_field, split)
                        all_papers.append(features)
                        paper_count += 1

                        if max_papers and paper_count >= max_papers:
                            break

                    if max_papers and paper_count >= max_papers:
                        break

                if max_papers and paper_count >= max_papers:
                    break

            if max_papers and paper_count >= max_papers:
                break

        print(f"Loaded {len(all_papers)} papers")

        # Convert to DataFrame
        if GPU_AVAILABLE:
            # For cuDF, first create pandas then convert
            pdf = pd.DataFrame(all_papers)
            df = pdf  # cudf.DataFrame(pdf) if needed
        else:
            df = pd.DataFrame(all_papers)

        # Save cache
        if use_cache and len(all_papers) > 0:
            with open(cache_file, 'wb') as f:
                pickle.dump(df, f)
            print(f"Cached to: {cache_file}")

        return df

    def _normalize_field(self, field: str) -> str:
        """Normalize field names"""
        mapping = {
            'ComputerScience': 'Computer Science',
            'MaterialsScience': 'Materials Science',
            'EnvironmentalScience': 'Environmental Science',
            'PoliticalScience': 'Political Science',
            'AgriculturalAndFoodSciences': 'Agricultural Sciences',
        }
        return mapping.get(field, field)

    def load_sample(self, n: int = 1000) -> pd.DataFrame:
        """Load a sample of papers for quick testing"""
        return self.load_all_papers(max_papers=n, use_cache=False)

    def get_field_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Get statistics by field of study"""
        stats = df.groupby('source_field').agg({
            'id': 'count',
            'is_ml_paper': 'sum',
            'has_code': 'sum',
            'ml_score': 'mean',
            'repro_score': 'mean',
        }).reset_index()

        stats.columns = ['field', 'total_papers', 'ml_papers',
                        'papers_with_code', 'avg_ml_score', 'avg_repro_score']

        stats['ml_adoption_rate'] = stats['ml_papers'] / stats['total_papers']
        stats['code_availability_rate'] = stats['papers_with_code'] / stats['total_papers']

        return stats.sort_values('total_papers', ascending=False)

    def get_temporal_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Get statistics by year"""
        stats = df.groupby('year').agg({
            'id': 'count',
            'is_ml_paper': 'sum',
            'has_code': 'sum',
            'ml_score': 'mean',
            'repro_score': 'mean',
        }).reset_index()

        stats.columns = ['year', 'total_papers', 'ml_papers',
                        'papers_with_code', 'avg_ml_score', 'avg_repro_score']

        stats['ml_adoption_rate'] = stats['ml_papers'] / stats['total_papers']
        stats['code_availability_rate'] = stats['papers_with_code'] / stats['total_papers']

        return stats[stats['year'] > 2000].sort_values('year')

    def get_dataset_summary(self) -> Dict:
        """Get summary of available datasets"""
        datasets = self.discover_datasets()

        fields = set()
        years = set()
        total_files = 0

        for ds in datasets:
            fields.add(ds['field'])
            total_files += ds['file_count']

            # Parse year range
            yr = ds['year_range']
            if '-' in yr:
                start, end = yr.split('-')
                try:
                    years.update(range(int(start), int(end) + 1))
                except:
                    pass

        return {
            'num_datasets': len(datasets),
            'fields': sorted(fields),
            'num_fields': len(fields),
            'year_range': (min(years), max(years)) if years else (None, None),
            'total_files': total_files,
        }


if __name__ == "__main__":
    # Test the loader
    loader = DataLoader("/home/abheekp/ai_research/s2orc_data")

    print("\n=== Dataset Summary ===")
    summary = loader.get_dataset_summary()
    print(f"Datasets: {summary['num_datasets']}")
    print(f"Fields: {summary['num_fields']}")
    print(f"Year range: {summary['year_range']}")
    print(f"Total files: {summary['total_files']}")
    print(f"Fields: {summary['fields']}")

    print("\n=== Loading Sample ===")
    df = loader.load_sample(500)
    print(f"Sample shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nML papers: {df['is_ml_paper'].sum()} ({df['is_ml_paper'].mean():.1%})")
    print(f"Papers with code: {df['has_code'].sum()} ({df['has_code'].mean():.1%})")

    print("\n=== Field Stats ===")
    field_stats = loader.get_field_stats(df)
    print(field_stats.to_string())
