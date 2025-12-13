"""
Adoption Dynamics Analysis
Track how ML techniques spread across scientific disciplines over time
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from scipy.optimize import curve_fit

try:
    import cudf as pd
    GPU_AVAILABLE = True
except ImportError:
    import pandas as pd
    GPU_AVAILABLE = False


class AdoptionAnalyzer:
    """Analyze ML adoption patterns across fields and time"""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self._precompute_stats()

    def _precompute_stats(self):
        """Precompute common statistics"""
        self.years = sorted(self.df['year'].unique())
        self.fields = self.df['source_field'].unique().tolist()

    @staticmethod
    def s_curve(x: np.ndarray, L: float, k: float, x0: float) -> np.ndarray:
        """Logistic S-curve function for adoption modeling"""
        return L / (1 + np.exp(-k * (x - x0)))

    def fit_s_curve(self, years: List[int], adoption_rates: List[float]) -> Dict:
        """Fit an S-curve to adoption data"""
        try:
            years_normalized = np.array(years) - min(years)
            popt, _ = curve_fit(
                self.s_curve,
                years_normalized,
                adoption_rates,
                p0=[max(adoption_rates), 0.5, len(years) / 2],
                bounds=([0, 0, 0], [1, 10, len(years)]),
                maxfev=5000
            )
            return {
                'L': popt[0],  # Maximum adoption
                'k': popt[1],  # Steepness
                'x0': popt[2] + min(years),  # Inflection point (year)
                'fitted': True
            }
        except:
            return {'fitted': False}

    def compute_adoption_by_year(self) -> pd.DataFrame:
        """Compute ML adoption rates by year"""
        stats = self.df.groupby('year').agg({
            'id': 'count',
            'is_ml_paper': 'sum',
            'ml_score': 'mean',
            'has_code': 'sum',
        }).reset_index()

        stats.columns = ['year', 'total_papers', 'ml_papers', 'avg_ml_score', 'papers_with_code']
        stats['adoption_rate'] = stats['ml_papers'] / stats['total_papers']
        stats['code_rate'] = stats['papers_with_code'] / stats['total_papers']

        return stats.sort_values('year')

    def compute_adoption_by_field(self) -> pd.DataFrame:
        """Compute ML adoption rates by field"""
        stats = self.df.groupby('source_field').agg({
            'id': 'count',
            'is_ml_paper': 'sum',
            'ml_score': 'mean',
            'repro_score': 'mean',
            'has_code': 'sum',
        }).reset_index()

        stats.columns = ['field', 'total_papers', 'ml_papers', 'avg_ml_score',
                        'avg_repro_score', 'papers_with_code']
        stats['adoption_rate'] = stats['ml_papers'] / stats['total_papers']
        stats['code_rate'] = stats['papers_with_code'] / stats['total_papers']

        return stats.sort_values('adoption_rate', ascending=False)

    def compute_adoption_heatmap(self) -> pd.DataFrame:
        """Create field x year heatmap of adoption rates"""
        pivot = self.df.pivot_table(
            values='is_ml_paper',
            index='source_field',
            columns='year',
            aggfunc='mean',
            fill_value=0
        )
        return pivot

    def compute_technique_trends(self) -> pd.DataFrame:
        """Track technique popularity over time"""
        technique_year_counts = defaultdict(lambda: defaultdict(int))
        total_by_year = defaultdict(int)

        for _, row in self.df.iterrows():
            year = row['year']
            total_by_year[year] += 1
            techniques = row.get('ml_techniques', [])
            if isinstance(techniques, list):
                for tech in techniques:
                    technique_year_counts[tech][year] += 1

        # Convert to DataFrame
        records = []
        for tech, year_counts in technique_year_counts.items():
            for year, count in year_counts.items():
                records.append({
                    'technique': tech,
                    'year': year,
                    'count': count,
                    'rate': count / total_by_year[year] if total_by_year[year] > 0 else 0
                })

        return pd.DataFrame(records)

    def compute_field_technique_matrix(self) -> pd.DataFrame:
        """Create field x technique adoption matrix"""
        field_tech_counts = defaultdict(lambda: defaultdict(int))
        field_totals = defaultdict(int)

        for _, row in self.df.iterrows():
            field = row['source_field']
            field_totals[field] += 1
            techniques = row.get('ml_techniques', [])
            if isinstance(techniques, list):
                for tech in techniques:
                    field_tech_counts[field][tech] += 1

        # Convert to DataFrame
        records = []
        for field, tech_counts in field_tech_counts.items():
            for tech, count in tech_counts.items():
                records.append({
                    'field': field,
                    'technique': tech,
                    'count': count,
                    'rate': count / field_totals[field] if field_totals[field] > 0 else 0
                })

        df = pd.DataFrame(records)
        if len(df) > 0:
            pivot = df.pivot_table(
                values='rate',
                index='field',
                columns='technique',
                fill_value=0
            )
            return pivot
        return pd.DataFrame()

    def identify_adoption_leaders(self) -> Dict[str, List[str]]:
        """Identify early vs late adopter fields for each technique"""
        technique_adoption = defaultdict(list)

        field_technique_df = self.compute_field_technique_matrix()
        if len(field_technique_df) == 0:
            return {}

        for tech in field_technique_df.columns:
            sorted_fields = field_technique_df[tech].sort_values(ascending=False)
            top_fields = sorted_fields.head(3).index.tolist()
            technique_adoption[tech] = top_fields

        return dict(technique_adoption)

    def compute_adoption_velocity(self) -> pd.DataFrame:
        """Compute how fast each field is adopting ML"""
        yearly = self.df.groupby(['source_field', 'year']).agg({
            'is_ml_paper': 'mean'
        }).reset_index()

        velocities = []
        for field in self.fields:
            field_data = yearly[yearly['source_field'] == field].sort_values('year')
            if len(field_data) >= 2:
                # Compute year-over-year change
                rates = field_data['is_ml_paper'].values
                years = field_data['year'].values
                if len(rates) > 1:
                    velocity = (rates[-1] - rates[0]) / (years[-1] - years[0])
                    velocities.append({
                        'field': field,
                        'start_rate': rates[0],
                        'end_rate': rates[-1],
                        'velocity': velocity,
                        'years_span': years[-1] - years[0]
                    })

        return pd.DataFrame(velocities).sort_values('velocity', ascending=False)

    def get_summary_stats(self) -> Dict:
        """Get summary statistics for the dashboard"""
        return {
            'total_papers': len(self.df),
            'ml_papers': int(self.df['is_ml_paper'].sum()),
            'overall_adoption_rate': float(self.df['is_ml_paper'].mean()),
            'papers_with_code': int(self.df['has_code'].sum()),
            'code_availability_rate': float(self.df['has_code'].mean()),
            'year_range': (int(self.df['year'].min()), int(self.df['year'].max())),
            'num_fields': len(self.fields),
            'fields': self.fields,
        }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '/home/abheekp/ai_research/src')
    from data.loader import DataLoader
    from data.preprocessor import Preprocessor

    # Test
    loader = DataLoader("/home/abheekp/ai_research/s2orc_data")
    df = loader.load_sample(500)

    preprocessor = Preprocessor()
    df = preprocessor.process_dataframe(df)

    analyzer = AdoptionAnalyzer(df)

    print("\nSummary Stats:")
    print(analyzer.get_summary_stats())

    print("\nAdoption by Year:")
    print(analyzer.compute_adoption_by_year())

    print("\nAdoption by Field:")
    print(analyzer.compute_adoption_by_field())
