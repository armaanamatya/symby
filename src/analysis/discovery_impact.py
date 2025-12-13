"""
Discovery Impact Tracing
Track ML method → domain application → real-world impact journeys
"""

import re
from typing import Dict, List, Optional
from collections import defaultdict

try:
    import cudf as pd
    GPU_AVAILABLE = True
except ImportError:
    import pandas as pd
    GPU_AVAILABLE = False


class DiscoveryTracer:
    """Trace discovery impact from ML methods to domain applications"""

    # Landmark ML papers/methods to track
    LANDMARK_METHODS = {
        'alphafold': {
            'keywords': ['alphafold', 'protein structure prediction', 'deepmind protein'],
            'year': 2020,
            'domain': 'biology'
        },
        'transformer': {
            'keywords': ['transformer', 'attention is all you need', 'self-attention'],
            'year': 2017,
            'domain': 'nlp'
        },
        'bert': {
            'keywords': ['bert', 'bidirectional encoder', 'masked language model'],
            'year': 2018,
            'domain': 'nlp'
        },
        'gpt': {
            'keywords': ['gpt', 'generative pre-trained', 'gpt-2', 'gpt-3', 'gpt-4'],
            'year': 2018,
            'domain': 'nlp'
        },
        'resnet': {
            'keywords': ['resnet', 'residual network', 'skip connection', 'he et al'],
            'year': 2015,
            'domain': 'computer_vision'
        },
        'gan': {
            'keywords': ['generative adversarial', 'gan', 'goodfellow'],
            'year': 2014,
            'domain': 'generative'
        },
        'diffusion': {
            'keywords': ['diffusion model', 'stable diffusion', 'ddpm', 'denoising diffusion'],
            'year': 2020,
            'domain': 'generative'
        },
        'graph_neural': {
            'keywords': ['graph neural network', 'gnn', 'message passing neural'],
            'year': 2017,
            'domain': 'graph_ml'
        },
    }

    # Impact indicators
    IMPACT_INDICATORS = {
        'clinical': ['clinical trial', 'fda', 'patient outcome', 'therapeutic'],
        'patent': ['patent', 'intellectual property', 'commercialized'],
        'policy': ['policy', 'regulation', 'government', 'legislation'],
        'industry': ['deployed', 'production', 'commercial', 'startup', 'company'],
    }

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self._compile_patterns()

    def _compile_patterns(self):
        """Compile regex patterns for landmark detection"""
        self.landmark_patterns = {}
        for name, info in self.LANDMARK_METHODS.items():
            pattern = '|'.join(re.escape(kw) for kw in info['keywords'])
            self.landmark_patterns[name] = re.compile(pattern, re.IGNORECASE)

        self.impact_patterns = {}
        for impact_type, keywords in self.IMPACT_INDICATORS.items():
            pattern = '|'.join(re.escape(kw) for kw in keywords)
            self.impact_patterns[impact_type] = re.compile(pattern, re.IGNORECASE)

    def detect_landmark_references(self, text: str) -> List[str]:
        """Detect references to landmark ML methods"""
        text_lower = text.lower()
        found = []
        for name, pattern in self.landmark_patterns.items():
            if pattern.search(text_lower):
                found.append(name)
        return found

    def detect_impact_indicators(self, text: str) -> List[str]:
        """Detect real-world impact indicators"""
        text_lower = text.lower()
        found = []
        for impact_type, pattern in self.impact_patterns.items():
            if pattern.search(text_lower):
                found.append(impact_type)
        return found

    def trace_method_adoption(self, method_name: str) -> pd.DataFrame:
        """Trace adoption of a specific ML method over time and fields"""
        if method_name not in self.LANDMARK_METHODS:
            return pd.DataFrame()

        pattern = self.landmark_patterns[method_name]
        method_info = self.LANDMARK_METHODS[method_name]

        # Find papers mentioning this method
        mentions = []
        for _, row in self.df.iterrows():
            text = str(row.get('title', ''))
            if pattern.search(text.lower()):
                mentions.append({
                    'id': row['id'],
                    'year': row['year'],
                    'field': row['source_field'],
                    'is_ml_paper': row['is_ml_paper'],
                    'title': row['title'][:100] if row.get('title') else '',
                })

        df_mentions = pd.DataFrame(mentions)

        if len(df_mentions) > 0:
            # Filter to after method introduction
            df_mentions = df_mentions[df_mentions['year'] >= method_info['year']]

        return df_mentions

    def compute_diffusion_timeline(self, method_name: str) -> Dict:
        """Compute how a method diffused across fields over time"""
        mentions = self.trace_method_adoption(method_name)

        if len(mentions) == 0:
            return {'method': method_name, 'total_mentions': 0, 'timeline': {}}

        timeline = mentions.groupby(['year', 'field']).size().reset_index(name='count')

        # Pivot to year x field
        pivot = timeline.pivot_table(
            values='count',
            index='year',
            columns='field',
            fill_value=0
        )

        return {
            'method': method_name,
            'total_mentions': len(mentions),
            'first_adoption_by_field': mentions.groupby('field')['year'].min().to_dict(),
            'timeline': pivot.to_dict(),
        }

    def find_high_impact_papers(self) -> pd.DataFrame:
        """Find papers with real-world impact indicators"""
        high_impact = []

        for _, row in self.df.iterrows():
            text = str(row.get('title', ''))
            impacts = self.detect_impact_indicators(text)
            landmarks = self.detect_landmark_references(text)

            if impacts or (landmarks and row.get('is_ml_paper')):
                high_impact.append({
                    'id': row['id'],
                    'year': row['year'],
                    'field': row['source_field'],
                    'title': row.get('title', '')[:150],
                    'impact_types': impacts,
                    'landmark_refs': landmarks,
                    'is_ml_paper': row['is_ml_paper'],
                    'impact_score': len(impacts) + len(landmarks) * 0.5,
                })

        df_impact = pd.DataFrame(high_impact)
        if len(df_impact) > 0:
            df_impact = df_impact.sort_values('impact_score', ascending=False)
        return df_impact

    def compute_method_impact_summary(self) -> pd.DataFrame:
        """Summarize impact of each landmark method"""
        summaries = []

        for method_name, method_info in self.LANDMARK_METHODS.items():
            mentions = self.trace_method_adoption(method_name)

            if len(mentions) == 0:
                continue

            # Count by field
            field_counts = mentions['field'].value_counts().to_dict()

            # Years since introduction
            years_since = mentions['year'].max() - method_info['year']

            summaries.append({
                'method': method_name,
                'introduced_year': method_info['year'],
                'origin_domain': method_info['domain'],
                'total_mentions': len(mentions),
                'fields_adopted': len(field_counts),
                'top_field': max(field_counts, key=field_counts.get) if field_counts else None,
                'years_tracked': years_since,
                'mentions_per_year': len(mentions) / max(years_since, 1),
            })

        return pd.DataFrame(summaries).sort_values('total_mentions', ascending=False)

    def generate_case_study(self, method_name: str) -> Dict:
        """Generate a case study for a specific method"""
        if method_name not in self.LANDMARK_METHODS:
            return {'error': f'Unknown method: {method_name}'}

        method_info = self.LANDMARK_METHODS[method_name]
        mentions = self.trace_method_adoption(method_name)
        diffusion = self.compute_diffusion_timeline(method_name)

        # Find example papers in different fields
        examples = {}
        if len(mentions) > 0:
            for field in mentions['field'].unique():
                field_papers = mentions[mentions['field'] == field].head(2)
                examples[field] = field_papers[['year', 'title']].to_dict('records')

        return {
            'method': method_name,
            'info': method_info,
            'total_papers': len(mentions),
            'diffusion': diffusion,
            'examples_by_field': examples,
        }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '/home/abheekp/ai_research/src')
    from data.loader import DataLoader
    from data.preprocessor import Preprocessor

    loader = DataLoader("/home/abheekp/ai_research/s2orc_data")
    df = loader.load_sample(1000)

    preprocessor = Preprocessor()
    df = preprocessor.process_dataframe(df)

    tracer = DiscoveryTracer(df)

    print("\nMethod Impact Summary:")
    print(tracer.compute_method_impact_summary())

    print("\nHigh Impact Papers:")
    print(tracer.find_high_impact_papers().head(10))
