"""
ML Impact Quantification
Measure how much ML actually contributes to scientific breakthroughs
"""

from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import numpy as np

try:
    import cudf as pd
    GPU_AVAILABLE = True
except ImportError:
    import pandas as pd
    GPU_AVAILABLE = False


class MLImpactQuantifier:
    """Quantify ML's real impact on scientific progress"""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self._compute_baselines()

    def _compute_baselines(self):
        """Compute baseline metrics for comparison"""
        self.total_papers = len(self.df)
        self.ml_paper_count = int(self.df['is_ml_paper'].sum())
        self.ml_ratio = self.ml_paper_count / self.total_papers if self.total_papers > 0 else 0

    def compute_attribution_scores(self) -> pd.DataFrame:
        """
        Compute ML attribution scores for each field.
        Attribution = how much ML contributes to that field's research
        """
        field_stats = self.df.groupby('source_field').agg({
            'is_ml_paper': ['sum', 'mean'],
            'ml_score': ['mean', 'sum'],
            'breakthrough_score': 'mean',
            'id': 'count',
        }).reset_index()

        # Flatten column names
        field_stats.columns = ['field', 'ml_paper_count', 'ml_adoption_rate',
                              'avg_ml_intensity', 'total_ml_score',
                              'avg_breakthrough', 'total_papers']

        # Attribution score: combination of adoption rate and intensity
        field_stats['ml_attribution'] = (
            field_stats['ml_adoption_rate'] * 0.4 +
            (field_stats['avg_ml_intensity'] / field_stats['avg_ml_intensity'].max()) * 0.3 +
            (field_stats['avg_breakthrough'] / field_stats['avg_breakthrough'].max()) * 0.3
        ).fillna(0)

        # Normalize to 0-100
        field_stats['ml_attribution_pct'] = (
            field_stats['ml_attribution'] / field_stats['ml_attribution'].max() * 100
        ).fillna(0)

        return field_stats.sort_values('ml_attribution_pct', ascending=False)

    def compute_acceleration_metrics(self) -> pd.DataFrame:
        """
        Estimate how ML might be accelerating research in each field.
        Uses proxy: rate of increase in papers and ML adoption velocity.
        """
        yearly = self.df.groupby(['source_field', 'year']).agg({
            'id': 'count',
            'is_ml_paper': 'sum',
            'ml_score': 'mean',
        }).reset_index()

        yearly.columns = ['field', 'year', 'paper_count', 'ml_papers', 'avg_ml_score']

        acceleration_metrics = []

        for field in self.df['source_field'].unique():
            field_data = yearly[yearly['field'] == field].sort_values('year')

            if len(field_data) < 2:
                continue

            years = field_data['year'].values
            paper_counts = field_data['paper_count'].values
            ml_counts = field_data['ml_papers'].values

            # Calculate growth rates
            if len(years) >= 2 and years[-1] != years[0]:
                paper_growth = (paper_counts[-1] - paper_counts[0]) / (years[-1] - years[0])
                ml_growth = (ml_counts[-1] - ml_counts[0]) / (years[-1] - years[0])

                # Acceleration proxy: is ML adoption accelerating paper production?
                # Compare ML adoption rate change with paper count change
                early_ml_rate = ml_counts[0] / paper_counts[0] if paper_counts[0] > 0 else 0
                late_ml_rate = ml_counts[-1] / paper_counts[-1] if paper_counts[-1] > 0 else 0
                ml_adoption_delta = late_ml_rate - early_ml_rate

                acceleration_metrics.append({
                    'field': field,
                    'paper_growth_rate': paper_growth,
                    'ml_paper_growth_rate': ml_growth,
                    'ml_adoption_change': ml_adoption_delta,
                    'years_observed': int(years[-1] - years[0]),
                    'start_year': int(years[0]),
                    'end_year': int(years[-1]),
                })

        return pd.DataFrame(acceleration_metrics).sort_values(
            'ml_adoption_change', ascending=False
        )

    def compute_efficiency_metrics(self) -> pd.DataFrame:
        """
        Compute efficiency metrics: how efficiently is ML being used?
        """
        field_stats = self.df.groupby('source_field').agg({
            'ml_score': ['mean', 'std'],
            'repro_score': 'mean',
            'has_code': 'mean',
            'technique_count': 'mean',
            'is_ml_paper': 'mean',
            'id': 'count',
        }).reset_index()

        field_stats.columns = ['field', 'avg_ml_score', 'ml_score_std',
                              'avg_repro', 'code_rate', 'avg_techniques',
                              'ml_adoption', 'paper_count']

        # Efficiency score: high ML impact with good reproducibility
        field_stats['efficiency_score'] = (
            field_stats['avg_ml_score'] * field_stats['code_rate'] *
            (1 + field_stats['avg_repro'])
        ).fillna(0)

        # Normalize
        max_eff = field_stats['efficiency_score'].max()
        if max_eff > 0:
            field_stats['efficiency_pct'] = field_stats['efficiency_score'] / max_eff * 100
        else:
            field_stats['efficiency_pct'] = 0

        return field_stats.sort_values('efficiency_pct', ascending=False)

    def identify_transformational_vs_hype(self) -> Dict[str, List[str]]:
        """
        Classify fields/techniques as transformational vs hype.
        Transformational: high ML adoption + high reproducibility + sustained growth
        Hype: high ML adoption but low reproducibility or declining
        """
        attribution = self.compute_attribution_scores()
        efficiency = self.compute_efficiency_metrics()

        # Merge metrics
        combined = attribution.merge(
            efficiency[['field', 'efficiency_pct', 'code_rate']],
            on='field'
        )

        # Classify
        transformational = []
        hype_risk = []
        emerging = []

        for _, row in combined.iterrows():
            if row['ml_adoption_rate'] > 0.3:
                if row['code_rate'] > 0.1 and row['efficiency_pct'] > 50:
                    transformational.append(row['field'])
                elif row['code_rate'] < 0.05:
                    hype_risk.append(row['field'])
                else:
                    emerging.append(row['field'])
            elif row['ml_adoption_rate'] > 0.1:
                emerging.append(row['field'])

        return {
            'transformational': transformational,
            'hype_risk': hype_risk,
            'emerging': emerging,
        }

    def compute_technique_staying_power(self) -> pd.DataFrame:
        """
        Identify which ML techniques have staying power vs fads.
        Based on: sustained usage over years, spread across fields.
        """
        technique_stats = defaultdict(lambda: {
            'years': set(),
            'fields': set(),
            'total_papers': 0,
            'repro_sum': 0,
        })

        for _, row in self.df.iterrows():
            techniques = row.get('ml_techniques', [])
            if isinstance(techniques, list):
                for tech in techniques:
                    technique_stats[tech]['years'].add(row['year'])
                    technique_stats[tech]['fields'].add(row['source_field'])
                    technique_stats[tech]['total_papers'] += 1
                    technique_stats[tech]['repro_sum'] += row.get('repro_score', 0)

        records = []
        for tech, stats in technique_stats.items():
            years_active = len(stats['years'])
            fields_spread = len(stats['fields'])
            total = stats['total_papers']

            # Staying power score
            staying_power = (
                years_active * 0.4 +
                fields_spread * 0.3 +
                min(total / 100, 1) * 0.3
            )

            records.append({
                'technique': tech,
                'years_active': years_active,
                'fields_spread': fields_spread,
                'total_papers': total,
                'avg_repro_score': stats['repro_sum'] / total if total > 0 else 0,
                'staying_power_score': staying_power,
            })

        df = pd.DataFrame(records)
        if len(df) > 0:
            # Normalize staying power
            max_sp = df['staying_power_score'].max()
            if max_sp > 0:
                df['staying_power_pct'] = df['staying_power_score'] / max_sp * 100
            df = df.sort_values('staying_power_pct', ascending=False)

        return df

    def generate_impact_summary(self) -> Dict:
        """Generate comprehensive ML impact summary"""
        attribution = self.compute_attribution_scores()
        classification = self.identify_transformational_vs_hype()
        technique_power = self.compute_technique_staying_power()

        return {
            'total_papers_analyzed': self.total_papers,
            'ml_paper_count': self.ml_paper_count,
            'overall_ml_ratio': round(self.ml_ratio, 4),
            'top_ml_attributed_fields': attribution.head(5)['field'].tolist(),
            'top_attribution_scores': attribution.head(5)['ml_attribution_pct'].tolist(),
            'transformational_fields': classification['transformational'],
            'hype_risk_fields': classification['hype_risk'],
            'top_staying_power_techniques': technique_power.head(5)['technique'].tolist() if len(technique_power) > 0 else [],
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

    quantifier = MLImpactQuantifier(df)

    print("\nAttribution Scores:")
    print(quantifier.compute_attribution_scores())

    print("\nAcceleration Metrics:")
    print(quantifier.compute_acceleration_metrics())

    print("\nTechnique Staying Power:")
    print(quantifier.compute_technique_staying_power())

    print("\nImpact Summary:")
    print(quantifier.generate_impact_summary())
