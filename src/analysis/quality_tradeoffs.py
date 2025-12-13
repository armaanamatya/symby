"""
Quality Trade-offs Analysis
Investigate whether ML adoption correlates with research reproducibility
"""

from typing import Dict, List, Optional
from collections import defaultdict
import numpy as np

try:
    import cudf as pd
    GPU_AVAILABLE = True
except ImportError:
    import pandas as pd
    GPU_AVAILABLE = False


class QualityAnalyzer:
    """Analyze quality and reproducibility trade-offs with ML adoption"""

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def compute_reproducibility_by_ml_adoption(self) -> pd.DataFrame:
        """Compare reproducibility metrics between ML and non-ML papers"""
        ml_papers = self.df[self.df['is_ml_paper'] == True]
        non_ml_papers = self.df[self.df['is_ml_paper'] == False]

        comparison = pd.DataFrame({
            'metric': ['Code Availability', 'Avg Repro Score', 'Paper Count'],
            'ml_papers': [
                ml_papers['has_code'].mean() if len(ml_papers) > 0 else 0,
                ml_papers['repro_score'].mean() if len(ml_papers) > 0 else 0,
                len(ml_papers),
            ],
            'non_ml_papers': [
                non_ml_papers['has_code'].mean() if len(non_ml_papers) > 0 else 0,
                non_ml_papers['repro_score'].mean() if len(non_ml_papers) > 0 else 0,
                len(non_ml_papers),
            ],
        })

        comparison['difference'] = comparison['ml_papers'] - comparison['non_ml_papers']
        return comparison

    def compute_reproducibility_by_field(self) -> pd.DataFrame:
        """Compute reproducibility metrics by field"""
        stats = self.df.groupby('source_field').agg({
            'has_code': 'mean',
            'repro_score': 'mean',
            'is_ml_paper': 'mean',
            'id': 'count',
        }).reset_index()

        stats.columns = ['field', 'code_availability', 'avg_repro_score',
                        'ml_adoption_rate', 'paper_count']

        # Compute correlation between ML adoption and reproducibility
        if len(stats) > 2:
            corr = stats['ml_adoption_rate'].corr(stats['code_availability'])
            stats['ml_repro_correlation'] = corr

        return stats.sort_values('code_availability', ascending=False)

    def compute_reproducibility_trend(self) -> pd.DataFrame:
        """Track reproducibility over time"""
        stats = self.df.groupby('year').agg({
            'has_code': 'mean',
            'repro_score': 'mean',
            'is_ml_paper': 'mean',
            'id': 'count',
        }).reset_index()

        stats.columns = ['year', 'code_availability', 'avg_repro_score',
                        'ml_adoption_rate', 'paper_count']

        return stats.sort_values('year')

    def compute_ml_score_vs_reproducibility(self) -> pd.DataFrame:
        """Analyze relationship between ML intensity and reproducibility"""
        # Bin papers by ML score
        bins = [0, 1, 3, 5, 10, 100]
        labels = ['0', '1-2', '3-4', '5-9', '10+']

        df_copy = self.df.copy()
        df_copy['ml_score_bin'] = pd.cut(
            df_copy['ml_score'],
            bins=bins,
            labels=labels,
            include_lowest=True
        )

        stats = df_copy.groupby('ml_score_bin').agg({
            'has_code': 'mean',
            'repro_score': 'mean',
            'id': 'count',
        }).reset_index()

        stats.columns = ['ml_score_range', 'code_availability',
                        'avg_repro_score', 'paper_count']

        return stats

    def identify_reproducibility_leaders(self) -> pd.DataFrame:
        """Identify fields that are leaders in reproducibility"""
        field_stats = self.compute_reproducibility_by_field()

        # Rank fields
        field_stats['code_rank'] = field_stats['code_availability'].rank(ascending=False)
        field_stats['repro_rank'] = field_stats['avg_repro_score'].rank(ascending=False)
        field_stats['combined_rank'] = (field_stats['code_rank'] + field_stats['repro_rank']) / 2

        return field_stats.sort_values('combined_rank')

    def compute_technique_reproducibility(self) -> pd.DataFrame:
        """Analyze reproducibility by ML technique used"""
        technique_stats = defaultdict(lambda: {'code_count': 0, 'total': 0, 'repro_sum': 0})

        for _, row in self.df.iterrows():
            techniques = row.get('ml_techniques', [])
            if isinstance(techniques, list):
                for tech in techniques:
                    technique_stats[tech]['total'] += 1
                    technique_stats[tech]['repro_sum'] += row.get('repro_score', 0)
                    if row.get('has_code', False):
                        technique_stats[tech]['code_count'] += 1

        records = []
        for tech, stats in technique_stats.items():
            if stats['total'] > 0:
                records.append({
                    'technique': tech,
                    'paper_count': stats['total'],
                    'code_availability': stats['code_count'] / stats['total'],
                    'avg_repro_score': stats['repro_sum'] / stats['total'],
                })

        df = pd.DataFrame(records)
        if len(df) > 0:
            df = df.sort_values('code_availability', ascending=False)
        return df

    def generate_quality_report(self) -> Dict:
        """Generate comprehensive quality analysis report"""
        ml_comparison = self.compute_reproducibility_by_ml_adoption()
        field_stats = self.compute_reproducibility_by_field()
        trends = self.compute_reproducibility_trend()

        # Calculate key insights
        overall_code_rate = self.df['has_code'].mean()
        ml_papers_code_rate = self.df[self.df['is_ml_paper']]['has_code'].mean() if self.df['is_ml_paper'].sum() > 0 else 0

        return {
            'overall_code_availability': float(overall_code_rate),
            'ml_papers_code_availability': float(ml_papers_code_rate),
            'code_availability_gap': float(ml_papers_code_rate - overall_code_rate),
            'top_reproducible_fields': field_stats.head(5)['field'].tolist(),
            'lowest_reproducible_fields': field_stats.tail(5)['field'].tolist(),
            'trend_improving': bool(trends['code_availability'].iloc[-1] > trends['code_availability'].iloc[0]) if len(trends) > 1 else None,
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

    analyzer = QualityAnalyzer(df)

    print("\nML vs Non-ML Reproducibility:")
    print(analyzer.compute_reproducibility_by_ml_adoption())

    print("\nReproducibility by Field:")
    print(analyzer.compute_reproducibility_by_field())

    print("\nQuality Report:")
    print(analyzer.generate_quality_report())
