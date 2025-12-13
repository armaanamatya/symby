"""
Data Preprocessor for ML Impact Analysis
Extracts ML techniques, builds citation proxies, computes metrics
"""

import re
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

try:
    import cudf as pd
    GPU_AVAILABLE = True
except ImportError:
    import pandas as pd
    GPU_AVAILABLE = False


class Preprocessor:
    """Preprocess papers for ML impact analysis"""

    # ML Technique Categories with keywords
    ML_TECHNIQUES = {
        'deep_learning': ['deep learning', 'neural network', 'deep neural'],
        'transformer': ['transformer', 'attention mechanism', 'self-attention', 'bert', 'gpt'],
        'cnn': ['convolutional', 'cnn', 'convnet', 'resnet', 'vgg', 'inception'],
        'rnn': ['recurrent', 'rnn', 'lstm', 'gru', 'sequence model'],
        'gan': ['generative adversarial', 'gan', 'discriminator', 'generator network'],
        'reinforcement_learning': ['reinforcement learning', 'q-learning', 'policy gradient', 'actor-critic'],
        'graph_neural': ['graph neural', 'gnn', 'graph convolutional', 'message passing'],
        'diffusion': ['diffusion model', 'denoising', 'score matching', 'stable diffusion'],
        'llm': ['large language model', 'llm', 'gpt-3', 'gpt-4', 'chatgpt', 'llama', 'claude'],
        'autoencoder': ['autoencoder', 'vae', 'variational autoencoder'],
        'transfer_learning': ['transfer learning', 'fine-tuning', 'pre-training', 'pretrained'],
        'few_shot': ['few-shot', 'zero-shot', 'meta-learning', 'prompt learning'],
    }

    # Domain categories
    DOMAINS = {
        'drug_discovery': ['drug discovery', 'pharmaceutical', 'molecule', 'compound screening', 'drug design'],
        'materials_science': ['materials science', 'crystal structure', 'alloy', 'polymer', 'semiconductor'],
        'climate_science': ['climate', 'weather prediction', 'atmospheric', 'carbon', 'emissions'],
        'physics': ['quantum', 'particle physics', 'cosmology', 'astrophysics', 'condensed matter'],
        'neuroscience': ['neuroscience', 'brain', 'neural activity', 'cognitive', 'fmri'],
        'biology': ['genomics', 'proteomics', 'cell biology', 'molecular biology', 'genetics'],
        'medicine': ['clinical', 'diagnosis', 'treatment', 'patient', 'medical imaging'],
    }

    # Breakthrough indicators
    BREAKTHROUGH_KEYWORDS = [
        'state-of-the-art', 'sota', 'breakthrough', 'novel', 'first',
        'outperforms', 'significant improvement', 'remarkable', 'unprecedented'
    ]

    def __init__(self):
        self.technique_patterns = self._compile_patterns(self.ML_TECHNIQUES)
        self.domain_patterns = self._compile_patterns(self.DOMAINS)

    def _compile_patterns(self, keyword_dict: Dict[str, List[str]]) -> Dict[str, re.Pattern]:
        """Compile regex patterns for efficient matching"""
        patterns = {}
        for category, keywords in keyword_dict.items():
            pattern = '|'.join(re.escape(kw) for kw in keywords)
            patterns[category] = re.compile(pattern, re.IGNORECASE)
        return patterns

    def extract_ml_techniques(self, text: str) -> List[str]:
        """Extract ML techniques mentioned in paper"""
        techniques = []
        text_lower = text.lower()
        for technique, pattern in self.technique_patterns.items():
            if pattern.search(text_lower):
                techniques.append(technique)
        return techniques

    def extract_domains(self, text: str) -> List[str]:
        """Extract scientific domains from paper"""
        domains = []
        text_lower = text.lower()
        for domain, pattern in self.domain_patterns.items():
            if pattern.search(text_lower):
                domains.append(domain)
        return domains

    def compute_breakthrough_score(self, text: str) -> float:
        """Compute a breakthrough/novelty score based on language"""
        text_lower = text.lower()
        score = sum(1 for kw in self.BREAKTHROUGH_KEYWORDS if kw in text_lower)
        return min(score / len(self.BREAKTHROUGH_KEYWORDS), 1.0)

    def extract_references_count(self, text: str) -> int:
        """Estimate number of references from text patterns"""
        # Look for citation patterns like [1], [2], etc.
        bracket_refs = len(re.findall(r'\[\d+\]', text))
        # Look for author-year patterns
        author_year = len(re.findall(r'\([A-Z][a-z]+(?:\s+et\s+al\.?)?,?\s*\d{4}\)', text))
        return bracket_refs + author_year

    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add extracted features to dataframe"""
        print("Extracting ML techniques...")

        # Convert to pandas for text processing if using cuDF
        if GPU_AVAILABLE:
            df_cpu = df.to_pandas()
        else:
            df_cpu = df

        # Extract techniques
        df_cpu['ml_techniques'] = df_cpu['title'].apply(
            lambda x: self.extract_ml_techniques(str(x))
        )

        # Extract domains
        df_cpu['domains'] = df_cpu['title'].apply(
            lambda x: self.extract_domains(str(x))
        )

        # Breakthrough score
        df_cpu['breakthrough_score'] = df_cpu['title'].apply(
            lambda x: self.compute_breakthrough_score(str(x))
        )

        # Flatten techniques for analysis
        df_cpu['technique_count'] = df_cpu['ml_techniques'].apply(len)
        df_cpu['domain_count'] = df_cpu['domains'].apply(len)

        # Primary technique (first one found)
        df_cpu['primary_technique'] = df_cpu['ml_techniques'].apply(
            lambda x: x[0] if x else 'none'
        )

        # Primary domain
        df_cpu['primary_domain'] = df_cpu['domains'].apply(
            lambda x: x[0] if x else 'other'
        )

        # Convert back to cuDF if needed
        if GPU_AVAILABLE:
            # Note: cuDF doesn't support list columns well, keep as pandas for now
            return df_cpu
        return df_cpu

    def compute_adoption_metrics(self, df: pd.DataFrame) -> Dict:
        """Compute ML adoption metrics by field and year"""
        metrics = {
            'by_year': {},
            'by_field': {},
            'by_technique': {},
            'technique_by_year': {},
        }

        # Adoption by year
        year_groups = df.groupby('year')
        for year, group in year_groups:
            total = len(group)
            ml_papers = group['is_ml_paper'].sum()
            metrics['by_year'][year] = {
                'total': total,
                'ml_papers': ml_papers,
                'adoption_rate': ml_papers / total if total > 0 else 0,
                'avg_ml_score': group['ml_score'].mean(),
            }

        # Adoption by field
        field_groups = df.groupby('source_field')
        for field, group in field_groups:
            total = len(group)
            ml_papers = group['is_ml_paper'].sum()
            metrics['by_field'][field] = {
                'total': total,
                'ml_papers': ml_papers,
                'adoption_rate': ml_papers / total if total > 0 else 0,
                'avg_repro_score': group['repro_score'].mean(),
            }

        # Technique popularity
        technique_counts = defaultdict(int)
        for techniques in df['ml_techniques']:
            if isinstance(techniques, list):
                for tech in techniques:
                    technique_counts[tech] += 1

        metrics['by_technique'] = dict(technique_counts)

        return metrics


if __name__ == "__main__":
    from loader import DataLoader

    # Test preprocessing
    loader = DataLoader("/home/abheekp/ai_research/s2orc_data")
    df = loader.load_sample(200)

    preprocessor = Preprocessor()
    df_processed = preprocessor.process_dataframe(df)

    print(f"\nProcessed columns: {df_processed.columns.tolist()}")
    print(f"\nML techniques found: {df_processed['primary_technique'].value_counts()}")
    print(f"\nDomains found: {df_processed['primary_domain'].value_counts()}")
