"""
Unified Impact Metrics Engine for AI Research Impact Observatory
DGX Spark Frontier Hackathon - Symby AI Track

Implements all 4 Build Objectives:
1. Quantify ML Impact - Attribution scoring, acceleration metrics
2. Visualize Adoption Dynamics - S-curves, temporal evolution
3. Analyze Quality Trade-offs - Reproducibility vs ML adoption
4. Trace Discovery Impact - ML method → domain → real-world paths

Optimized for DGX Spark with 128GB unified memory
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
from dataclasses import dataclass, asdict, field
import warnings

try:
    from scipy import stats
    from scipy.optimize import curve_fit
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("scipy not available, some metrics will be limited")

try:
    import pandas as pd
except ImportError:
    import pandas as pd


# =============================================================================
# DATA CLASSES FOR STRUCTURED OUTPUT
# =============================================================================

@dataclass
class AdoptionMetrics:
    """Objective 2: Adoption Dynamics metrics for a field"""
    field: str
    total_papers: int
    ml_papers: int
    adoption_rate: float
    adoption_velocity: float  # Year-over-year change rate
    first_adoption_year: Optional[int]
    peak_adoption_year: Optional[int]
    s_curve_phase: str  # 'early', 'growth', 'mature', 'declining'
    growth_rate: float
    years_to_50pct: Optional[int]  # Years to reach 50% adoption
    current_trend: str  # 'accelerating', 'growing', 'stable', 'declining'


@dataclass
class QualityMetrics:
    """Objective 3: Quality Trade-offs metrics for a field"""
    field: str
    total_papers: int
    ml_papers: int
    ml_adoption_rate: float
    code_availability_rate: float
    data_availability_rate: float
    avg_reproducibility_score: float
    ml_code_rate: float  # Code availability for ML papers
    nonml_code_rate: float  # Code availability for non-ML papers
    quality_delta: float  # Difference in reproducibility ML vs non-ML
    ml_quality_correlation: float  # Correlation between ML adoption and quality
    quality_trend: str  # 'improving', 'stable', 'declining'
    hype_risk_score: float  # High adoption + low reproducibility = risk
    retraction_signal_rate: float  # Rate of papers with retraction/concern signals
    ml_retraction_rate: float  # Retraction signals in ML papers
    nonml_retraction_rate: float  # Retraction signals in non-ML papers
    reproducibility_concern_score: float  # Composite quality concern indicator


@dataclass
class ImpactMetrics:
    """Objective 1: ML Impact Quantification for a field"""
    field: str
    total_papers: int
    ml_papers: int
    ml_attribution_score: float  # 0-1: How much ML contributes
    adoption_intensity: float  # Average ML score for ML papers
    acceleration_factor: float  # Research output growth rate
    efficiency_score: float  # Impact per ML adoption unit
    breakthrough_potential: float  # Based on landmark methods
    technique_diversity: float  # Number of different ML techniques used
    cross_pollination_score: float  # Methods from other fields
    compute_cost_proxy: float  # 0-1: Estimated compute intensity (GPU/HPC mentions)
    cost_efficiency_ratio: float  # Impact per compute cost unit


@dataclass
class DiscoveryPath:
    """Objective 4: Discovery Impact path from ML → Domain → Impact"""
    ml_technique: str
    origin_year: int
    target_field: str
    first_adoption_year: int
    adoption_lag_years: int
    papers_count: int
    adoption_rate_in_field: float
    impact_trajectory: str  # 'growing', 'peaked', 'declining'


@dataclass
class LandmarkImpact:
    """Impact analysis for landmark ML methods"""
    method: str
    origin_year: int
    total_mentions: int
    fields_reached: List[str]
    adoption_by_field: Dict[str, int]
    peak_year: Optional[int]
    adoption_velocity: float
    transformational_score: float


# =============================================================================
# LANDMARK ML METHODS DATABASE
# =============================================================================

LANDMARK_METHODS = {
    # Computer Vision
    'alexnet': {'year': 2012, 'domain': 'Computer Vision', 'keywords': ['alexnet', 'alex net']},
    'resnet': {'year': 2015, 'domain': 'Computer Vision', 'keywords': ['resnet', 'residual network', 'res-net']},
    'yolo': {'year': 2016, 'domain': 'Computer Vision', 'keywords': ['yolo', 'you only look once']},
    'unet': {'year': 2015, 'domain': 'Computer Vision', 'keywords': ['u-net', 'unet']},

    # NLP/Language
    'word2vec': {'year': 2013, 'domain': 'NLP', 'keywords': ['word2vec', 'word embedding']},
    'transformer': {'year': 2017, 'domain': 'NLP', 'keywords': ['transformer', 'attention is all']},
    'bert': {'year': 2018, 'domain': 'NLP', 'keywords': ['bert', 'bidirectional encoder']},
    'gpt': {'year': 2018, 'domain': 'NLP', 'keywords': ['gpt', 'gpt-2', 'gpt-3', 'gpt-4', 'generative pre-trained']},

    # Generative
    'gan': {'year': 2014, 'domain': 'Generative', 'keywords': ['gan', 'generative adversarial']},
    'vae': {'year': 2013, 'domain': 'Generative', 'keywords': ['vae', 'variational autoencoder']},
    'diffusion': {'year': 2020, 'domain': 'Generative', 'keywords': ['diffusion model', 'stable diffusion', 'ddpm']},

    # Scientific Breakthroughs
    'alphafold': {'year': 2020, 'domain': 'Biology', 'keywords': ['alphafold', 'alpha fold', 'protein structure prediction']},

    # COVID-19 Drug Discovery (Objective 4 case study)
    'covid_drug': {'year': 2020, 'domain': 'Medicine', 'keywords': [
        'covid-19 drug', 'sars-cov-2 drug', 'coronavirus drug discovery',
        'covid treatment', 'antiviral screening', 'drug repurposing covid'
    ]},

    # Reinforcement Learning
    'dqn': {'year': 2015, 'domain': 'RL', 'keywords': ['dqn', 'deep q-network', 'deep q network']},
    'ppo': {'year': 2017, 'domain': 'RL', 'keywords': ['ppo', 'proximal policy']},

    # Graph
    'gcn': {'year': 2017, 'domain': 'Graph', 'keywords': ['gcn', 'graph convolutional', 'graph neural']},
}

# =============================================================================
# COMPUTE COST KEYWORDS (for efficiency/cost per discovery metric)
# =============================================================================

COMPUTE_COST_KEYWORDS = {
    # High cost indicators (weight: 3)
    'a100': 3, 'v100': 3, 'tpu': 3, 'dgx': 3, 'supercomputer': 3,
    'petaflop': 3, 'exascale': 3, 'hpc cluster': 3,

    # Medium cost indicators (weight: 2)
    'gpu cluster': 2, 'multi-gpu': 2, 'distributed training': 2,
    'teraflop': 2, 'cloud computing': 2, 'aws': 2, 'gcp': 2, 'azure': 2,

    # Low cost indicators (weight: 1)
    'gpu': 1, 'cuda': 1, 'nvidia': 1, 'single gpu': 1,
    'colab': 1, 'kaggle': 1,
}

# =============================================================================
# RETRACTION/QUALITY SIGNAL KEYWORDS (for quality trade-offs)
# =============================================================================

RETRACTION_KEYWORDS = {
    # Direct retraction signals
    'retracted', 'retraction notice', 'withdrawn', 'correction published',
    'erratum', 'corrigendum', 'expression of concern', 'editorial concern',

    # Quality concern signals
    'failed to replicate', 'could not reproduce', 'irreproducible',
    'data manipulation', 'fabrication', 'falsification', 'misconduct',
    'p-hacking', 'selective reporting',
}

# =============================================================================
# CITATION FLOW PROXY KEYWORDS (for adoption dynamics)
# =============================================================================

CITATION_SIGNAL_KEYWORDS = {
    # Indicates paper cites/builds on ML methods
    'based on', 'inspired by', 'extending', 'building upon',
    'adapted from', 'modified from', 'following', 'using the approach',
    'applying', 'employing', 'leveraging', 'utilizing',
}


# =============================================================================
# HELPER FUNCTIONS FOR NEW METRICS
# =============================================================================

def compute_cost_score(text: str) -> float:
    """
    Compute compute-cost proxy score from paper text.

    Higher score = mentions of expensive compute resources (A100, HPC, etc.)
    Returns score 0-1.
    """
    if not text:
        return 0.0

    text_lower = text.lower()
    score = 0
    max_score = 0

    for keyword, weight in COMPUTE_COST_KEYWORDS.items():
        max_score += weight
        if keyword in text_lower:
            score += weight

    # Normalize to 0-1
    return min(score / 15, 1.0)  # 15 is a reasonable max for well-resourced papers


def detect_retraction_signals(text: str) -> bool:
    """
    Detect if paper text contains retraction or quality concern signals.

    Returns True if any retraction keywords found.
    """
    if not text:
        return False

    text_lower = text.lower()

    for keyword in RETRACTION_KEYWORDS:
        if keyword in text_lower:
            return True

    return False


def compute_reproducibility_concern_score(code_rate: float, retraction_rate: float,
                                          data_rate: float, adoption_rate: float) -> float:
    """
    Compute composite reproducibility concern score.

    High score = high concern (low quality indicators + high adoption).
    Formula: (1 - code_rate) * 0.3 + (1 - data_rate) * 0.2 + retraction_rate * 0.3 + adoption_rate * 0.2
    """
    concern = (
        (1 - code_rate) * 0.3 +    # Low code availability
        (1 - data_rate) * 0.2 +    # Low data availability
        retraction_rate * 0.3 +     # Retraction signals present
        adoption_rate * 0.2         # High adoption without quality = concern
    )
    return min(concern, 1.0)


# =============================================================================
# UNIFIED METRICS ENGINE
# =============================================================================

class UnifiedMetricsEngine:
    """
    Comprehensive metrics engine implementing all 4 hackathon objectives.

    Designed for DGX Spark with 128GB unified memory:
    - Efficient DataFrame operations
    - Vectorized computations
    - Memory-conscious aggregations
    """

    def __init__(self, papers_df: pd.DataFrame):
        """
        Initialize with paper data.

        Args:
            papers_df: DataFrame with columns:
                - id, title, year, source_field, is_ml_paper, has_code,
                - has_data, ml_score, repro_score, techniques, landmarks
        """
        self.df = papers_df
        self._precompute()

    def _precompute(self):
        """Precompute common aggregations for efficiency"""
        # Basic stats
        self.years = sorted(self.df['year'].unique())
        self.years = [y for y in self.years if y > 2000]  # Filter valid years
        self.fields = sorted(self.df['source_field'].unique())

        # Precompute field aggregations
        self._field_stats = self._compute_field_aggregations()
        self._year_stats = self._compute_year_aggregations()

    def _compute_field_aggregations(self) -> Dict[str, Dict]:
        """Precompute aggregations by field"""
        stats = {}
        for field in self.fields:
            field_df = self.df[self.df['source_field'] == field]
            ml_df = field_df[field_df['is_ml_paper'] == True]
            nonml_df = field_df[field_df['is_ml_paper'] == False]

            stats[field] = {
                'total': len(field_df),
                'ml_count': len(ml_df),
                'nonml_count': len(nonml_df),
                'ml_rate': len(ml_df) / len(field_df) if len(field_df) > 0 else 0,
                'code_rate': field_df['has_code'].mean() if len(field_df) > 0 else 0,
                'ml_code_rate': ml_df['has_code'].mean() if len(ml_df) > 0 else 0,
                'nonml_code_rate': nonml_df['has_code'].mean() if len(nonml_df) > 0 else 0,
                'avg_ml_score': field_df['ml_score'].mean() if len(field_df) > 0 else 0,
                'avg_repro': field_df['repro_score'].mean() if len(field_df) > 0 else 0,
            }
        return stats

    def _compute_year_aggregations(self) -> Dict[int, Dict]:
        """Precompute aggregations by year"""
        stats = {}
        for year in self.years:
            year_df = self.df[self.df['year'] == year]
            stats[year] = {
                'total': len(year_df),
                'ml_count': year_df['is_ml_paper'].sum(),
                'ml_rate': year_df['is_ml_paper'].mean() if len(year_df) > 0 else 0,
                'code_rate': year_df['has_code'].mean() if len(year_df) > 0 else 0,
            }
        return stats

    # =========================================================================
    # OBJECTIVE 1: QUANTIFY ML IMPACT
    # =========================================================================

    def compute_impact_metrics(self) -> List[ImpactMetrics]:
        """
        Compute ML impact quantification metrics for each field.

        Measures:
        - Attribution scoring: What % of breakthrough comes from ML
        - Acceleration metrics: Research output growth
        - Efficiency measures: Impact per ML unit
        """
        results = []

        for field in self.fields:
            field_df = self.df[self.df['source_field'] == field]
            stats = self._field_stats[field]

            if stats['total'] < 10:
                continue

            # ML Attribution Score (0-1)
            # Based on: adoption rate, intensity, technique diversity
            adoption_component = min(stats['ml_rate'] * 2, 0.4)  # Cap at 0.4
            intensity_component = min(stats['avg_ml_score'] / 20, 0.3)  # Cap at 0.3

            # Technique diversity
            all_techniques = []
            if 'techniques' in field_df.columns:
                for tech_list in field_df['techniques'].dropna():
                    if isinstance(tech_list, list):
                        all_techniques.extend(tech_list)
            unique_techniques = len(set(all_techniques))
            diversity_component = min(unique_techniques / 10, 0.3)  # Cap at 0.3

            attribution_score = adoption_component + intensity_component + diversity_component

            # Acceleration Factor
            # Compare paper output growth before/after ML adoption
            yearly = field_df.groupby('year').size()
            if len(yearly) >= 4:
                mid_idx = len(yearly) // 2
                early_avg = yearly.iloc[:mid_idx].mean()
                late_avg = yearly.iloc[mid_idx:].mean()
                acceleration = late_avg / early_avg if early_avg > 0 else 1.0
            else:
                acceleration = 1.0

            # Efficiency Score
            # High impact (citations proxy via text_length) per ML adoption
            ml_papers = field_df[field_df['is_ml_paper']]
            if len(ml_papers) > 0 and stats['ml_rate'] > 0:
                avg_length = ml_papers['text_length'].mean()
                efficiency = (avg_length / 10000) / stats['ml_rate']
            else:
                efficiency = 0

            # Breakthrough Potential
            landmark_count = 0
            if 'landmarks' in field_df.columns:
                for lm_list in field_df['landmarks'].dropna():
                    if isinstance(lm_list, list):
                        landmark_count += len(lm_list)
            breakthrough = min(landmark_count / stats['total'] * 100, 1.0)

            # Cross-pollination (methods from Computer Science in non-CS fields)
            if field != 'Computer Science':
                cross_poll = stats['ml_rate'] * 0.5 + (unique_techniques / 20)
            else:
                cross_poll = 0

            # Compute cost proxy (Gap 1: cost/efficiency metrics)
            if 'text' in field_df.columns:
                ml_papers_df = field_df[field_df['is_ml_paper'] == True]
                if len(ml_papers_df) > 0:
                    cost_scores = ml_papers_df['text'].apply(
                        lambda x: compute_cost_score(str(x) if x else '')
                    )
                    cost_proxy = cost_scores.mean()
                else:
                    cost_proxy = 0.0
            else:
                # Fallback: estimate from text_length as proxy
                avg_length = field_df['text_length'].mean() if len(field_df) > 0 else 0
                cost_proxy = min(avg_length / 50000, 1.0)  # Longer papers often = more compute

            # Cost efficiency ratio: impact / cost
            if cost_proxy > 0:
                cost_efficiency = attribution_score / (cost_proxy + 0.1)
            else:
                cost_efficiency = attribution_score * 2  # Low cost = high efficiency

            results.append(ImpactMetrics(
                field=field,
                total_papers=stats['total'],
                ml_papers=stats['ml_count'],
                ml_attribution_score=round(attribution_score, 3),
                adoption_intensity=round(stats['avg_ml_score'], 2),
                acceleration_factor=round(acceleration, 2),
                efficiency_score=round(efficiency, 2),
                breakthrough_potential=round(breakthrough, 3),
                technique_diversity=unique_techniques,
                cross_pollination_score=round(cross_poll, 3),
                compute_cost_proxy=round(cost_proxy, 3),
                cost_efficiency_ratio=round(cost_efficiency, 3)
            ))

        return sorted(results, key=lambda x: x.ml_attribution_score, reverse=True)

    # =========================================================================
    # OBJECTIVE 2: VISUALIZE ADOPTION DYNAMICS
    # =========================================================================

    def compute_adoption_dynamics(self) -> List[AdoptionMetrics]:
        """
        Compute adoption dynamics metrics for S-curve analysis.

        Tracks:
        - Adoption rates over time
        - S-curve phase detection
        - Growth velocity
        - Time to 50% adoption
        """
        results = []

        for field in self.fields:
            field_df = self.df[self.df['source_field'] == field]
            stats = self._field_stats[field]

            if stats['total'] < 10:
                continue

            # Yearly adoption rates
            yearly = field_df.groupby('year').agg({
                'is_ml_paper': ['sum', 'count']
            }).reset_index()
            yearly.columns = ['year', 'ml_count', 'total']
            yearly['rate'] = yearly['ml_count'] / yearly['total']
            yearly = yearly[yearly['year'] > 2000].sort_values('year')

            if len(yearly) < 2:
                continue

            # Current adoption rate
            current_rate = yearly.iloc[-1]['rate']

            # Adoption velocity (year-over-year change)
            if len(yearly) >= 2:
                rates = yearly['rate'].values
                velocity = (rates[-1] - rates[0]) / len(rates)
            else:
                velocity = 0

            # First adoption year
            ml_years = yearly[yearly['ml_count'] > 0]
            first_adoption = int(ml_years['year'].min()) if len(ml_years) > 0 else None

            # Peak adoption year
            peak_idx = yearly['rate'].idxmax()
            peak_year = int(yearly.loc[peak_idx, 'year']) if peak_idx is not None else None

            # S-curve phase detection
            if current_rate < 0.1:
                phase = 'early'
            elif current_rate < 0.3 and velocity > 0.02:
                phase = 'growth'
            elif current_rate >= 0.3 and velocity < 0.01:
                phase = 'mature'
            elif velocity < -0.01:
                phase = 'declining'
            else:
                phase = 'growth'

            # Growth rate (compound annual)
            if len(yearly) >= 2 and yearly.iloc[0]['rate'] > 0:
                n_years = len(yearly) - 1
                growth_rate = (yearly.iloc[-1]['rate'] / yearly.iloc[0]['rate']) ** (1/n_years) - 1
            else:
                growth_rate = 0

            # Years to 50% adoption (extrapolated)
            years_to_50 = None
            if current_rate < 0.5 and velocity > 0:
                years_to_50 = int((0.5 - current_rate) / velocity) if velocity > 0.001 else None

            # Current trend
            if len(yearly) >= 3:
                recent = yearly.tail(3)['rate'].values
                recent_velocity = (recent[-1] - recent[0]) / 2
                if recent_velocity > 0.03:
                    trend = 'accelerating'
                elif recent_velocity > 0.01:
                    trend = 'growing'
                elif recent_velocity > -0.01:
                    trend = 'stable'
                else:
                    trend = 'declining'
            else:
                trend = 'unknown'

            results.append(AdoptionMetrics(
                field=field,
                total_papers=stats['total'],
                ml_papers=stats['ml_count'],
                adoption_rate=round(current_rate, 4),
                adoption_velocity=round(velocity, 4),
                first_adoption_year=first_adoption,
                peak_adoption_year=peak_year,
                s_curve_phase=phase,
                growth_rate=round(growth_rate, 4),
                years_to_50pct=years_to_50,
                current_trend=trend
            ))

        return sorted(results, key=lambda x: x.adoption_rate, reverse=True)

    def compute_temporal_evolution(self) -> pd.DataFrame:
        """
        Compute temporal evolution data for animated visualization.

        Returns DataFrame with year x field adoption rates.
        Enhanced for Gap 4: animated temporal visualization support.
        """
        records = []

        # Track cumulative papers for animation
        cumulative_by_field = defaultdict(lambda: {'total': 0, 'ml': 0})

        for year in sorted(self.years):
            year_df = self.df[self.df['year'] == year]

            year_records = []
            for field in self.fields:
                field_year_df = year_df[year_df['source_field'] == field]
                if len(field_year_df) == 0:
                    continue

                # Update cumulative counts
                cumulative_by_field[field]['total'] += len(field_year_df)
                cumulative_by_field[field]['ml'] += int(field_year_df['is_ml_paper'].sum())

                # Calculate cumulative rate
                cum_total = cumulative_by_field[field]['total']
                cum_ml = cumulative_by_field[field]['ml']
                cumulative_rate = cum_ml / cum_total if cum_total > 0 else 0

                year_records.append({
                    'year': year,
                    'field': field,
                    'total_papers': len(field_year_df),
                    'ml_papers': int(field_year_df['is_ml_paper'].sum()),
                    'ml_rate': field_year_df['is_ml_paper'].mean(),
                    'code_rate': field_year_df['has_code'].mean(),
                    'avg_ml_score': field_year_df['ml_score'].mean(),
                    # Animation-friendly additions
                    'cumulative_papers': cum_total,
                    'cumulative_ml_papers': cum_ml,
                    'cumulative_ml_rate': cumulative_rate,
                })

            # Add ranking within each year for animated bar chart races
            year_records_sorted = sorted(year_records, key=lambda x: x['ml_rate'], reverse=True)
            for rank, rec in enumerate(year_records_sorted, 1):
                rec['rank'] = rank
                records.append(rec)

        return pd.DataFrame(records)

    def compute_s_curve_data(self) -> pd.DataFrame:
        """
        Compute S-curve fitting data for adoption dynamics visualization.

        Returns data suitable for plotting S-curves by field.
        """
        temporal = self.compute_temporal_evolution()

        if len(temporal) == 0:
            return pd.DataFrame()

        # Pivot to get adoption rates by year and field
        pivot = temporal.pivot_table(
            index='year',
            columns='field',
            values='cumulative_ml_rate',
            aggfunc='first'
        ).fillna(0)

        # Create records with S-curve phase annotations
        records = []
        for field in pivot.columns:
            field_data = pivot[field].dropna()
            if len(field_data) < 2:
                continue

            for year, rate in field_data.items():
                # Determine S-curve phase based on rate and velocity
                if len(field_data) >= 2:
                    idx = list(field_data.index).index(year)
                    if idx > 0:
                        velocity = rate - field_data.iloc[idx - 1]
                    else:
                        velocity = 0
                else:
                    velocity = 0

                if rate < 0.1:
                    phase = 'early'
                elif rate < 0.3 and velocity > 0:
                    phase = 'growth'
                elif rate >= 0.3:
                    phase = 'mature'
                else:
                    phase = 'emerging'

                records.append({
                    'year': year,
                    'field': field,
                    'cumulative_ml_rate': rate,
                    's_curve_phase': phase,
                    'velocity': velocity,
                })

        return pd.DataFrame(records)

    # =========================================================================
    # OBJECTIVE 3: ANALYZE QUALITY TRADE-OFFS
    # =========================================================================

    def compute_quality_metrics(self) -> List[QualityMetrics]:
        """
        Analyze quality trade-offs between ML adoption and reproducibility.

        Investigates:
        - Code/data availability rates
        - ML vs non-ML reproducibility
        - Correlation between adoption and quality
        - Hype risk detection
        """
        results = []

        for field in self.fields:
            field_df = self.df[self.df['source_field'] == field]
            stats = self._field_stats[field]

            if stats['total'] < 10:
                continue

            # Split ML vs non-ML
            ml_df = field_df[field_df['is_ml_paper'] == True]
            nonml_df = field_df[field_df['is_ml_paper'] == False]

            # Data availability
            data_rate = field_df['has_data'].mean() if 'has_data' in field_df.columns else 0

            # Quality delta (ML vs non-ML)
            ml_repro = ml_df['repro_score'].mean() if len(ml_df) > 0 else 0
            nonml_repro = nonml_df['repro_score'].mean() if len(nonml_df) > 0 else 0
            quality_delta = ml_repro - nonml_repro

            # Correlation between ML adoption and quality
            if SCIPY_AVAILABLE and len(field_df) > 10:
                ml_binary = field_df['is_ml_paper'].astype(float)
                repro = field_df['repro_score']
                if ml_binary.std() > 0 and repro.std() > 0:
                    correlation, _ = stats_module_correlation(ml_binary, repro)
                else:
                    correlation = 0
            else:
                correlation = 0

            # Quality trend
            yearly_quality = field_df.groupby('year')['has_code'].mean()
            if len(yearly_quality) >= 3:
                recent = yearly_quality.tail(3).values
                if recent[-1] > recent[0] * 1.1:
                    trend = 'improving'
                elif recent[-1] < recent[0] * 0.9:
                    trend = 'declining'
                else:
                    trend = 'stable'
            else:
                trend = 'unknown'

            # Hype Risk Score
            # High ML adoption + low reproducibility = high risk
            if stats['code_rate'] > 0:
                hype_risk = (stats['ml_rate'] / (stats['code_rate'] + 0.1)) * 0.5
            else:
                hype_risk = stats['ml_rate'] * 2  # High risk if no code at all

            hype_risk = min(hype_risk, 1.0)  # Cap at 1.0

            # Gap 2: Retraction signal detection
            if 'text' in field_df.columns:
                retraction_signals = field_df['text'].apply(
                    lambda x: detect_retraction_signals(str(x) if x else '')
                )
                retraction_rate = retraction_signals.mean()

                # ML vs non-ML retraction rates
                if len(ml_df) > 0 and 'text' in ml_df.columns:
                    ml_retraction = ml_df['text'].apply(
                        lambda x: detect_retraction_signals(str(x) if x else '')
                    ).mean()
                else:
                    ml_retraction = 0.0

                if len(nonml_df) > 0 and 'text' in nonml_df.columns:
                    nonml_retraction = nonml_df['text'].apply(
                        lambda x: detect_retraction_signals(str(x) if x else '')
                    ).mean()
                else:
                    nonml_retraction = 0.0
            else:
                # Fallback: estimate from title text
                if 'title' in field_df.columns:
                    retraction_signals = field_df['title'].apply(
                        lambda x: detect_retraction_signals(str(x) if x else '')
                    )
                    retraction_rate = retraction_signals.mean()
                else:
                    retraction_rate = 0.0
                ml_retraction = 0.0
                nonml_retraction = 0.0

            # Compute reproducibility concern score
            concern_score = compute_reproducibility_concern_score(
                stats['code_rate'],
                retraction_rate,
                data_rate,
                stats['ml_rate']
            )

            results.append(QualityMetrics(
                field=field,
                total_papers=stats['total'],
                ml_papers=stats['ml_count'],
                ml_adoption_rate=round(stats['ml_rate'], 4),
                code_availability_rate=round(stats['code_rate'], 4),
                data_availability_rate=round(data_rate, 4),
                avg_reproducibility_score=round(stats['avg_repro'], 2),
                ml_code_rate=round(stats['ml_code_rate'], 4),
                nonml_code_rate=round(stats['nonml_code_rate'], 4),
                quality_delta=round(quality_delta, 3),
                ml_quality_correlation=round(correlation, 3),
                quality_trend=trend,
                hype_risk_score=round(hype_risk, 3),
                retraction_signal_rate=round(retraction_rate, 4),
                ml_retraction_rate=round(ml_retraction, 4),
                nonml_retraction_rate=round(nonml_retraction, 4),
                reproducibility_concern_score=round(concern_score, 3)
            ))

        return sorted(results, key=lambda x: x.hype_risk_score, reverse=True)

    def compute_ml_vs_nonml_comparison(self) -> Dict:
        """
        Direct comparison of ML vs non-ML papers across all fields.
        """
        ml_papers = self.df[self.df['is_ml_paper'] == True]
        nonml_papers = self.df[self.df['is_ml_paper'] == False]

        return {
            'ml_papers': {
                'count': len(ml_papers),
                'code_rate': round(ml_papers['has_code'].mean(), 4),
                'data_rate': round(ml_papers['has_data'].mean(), 4) if 'has_data' in ml_papers.columns else 0,
                'avg_repro_score': round(ml_papers['repro_score'].mean(), 2),
                'avg_text_length': int(ml_papers['text_length'].mean()),
            },
            'nonml_papers': {
                'count': len(nonml_papers),
                'code_rate': round(nonml_papers['has_code'].mean(), 4),
                'data_rate': round(nonml_papers['has_data'].mean(), 4) if 'has_data' in nonml_papers.columns else 0,
                'avg_repro_score': round(nonml_papers['repro_score'].mean(), 2),
                'avg_text_length': int(nonml_papers['text_length'].mean()),
            },
            'delta': {
                'code_rate': round(ml_papers['has_code'].mean() - nonml_papers['has_code'].mean(), 4),
                'repro_score': round(ml_papers['repro_score'].mean() - nonml_papers['repro_score'].mean(), 2),
            }
        }

    # =========================================================================
    # OBJECTIVE 4: TRACE DISCOVERY IMPACT
    # =========================================================================

    def compute_discovery_paths(self) -> List[DiscoveryPath]:
        """
        Trace discovery impact from ML methods to domain applications.

        Tracks:
        - ML technique → domain adoption paths
        - Time lag from origin to domain adoption
        - Impact trajectory in target field
        """
        results = []

        for method, info in LANDMARK_METHODS.items():
            keywords = info['keywords']
            origin_year = info['year']

            # Find mentions across all fields
            for field in self.fields:
                if field == 'Computer Science' and info['domain'] != 'Biology':
                    continue  # Skip CS-to-CS unless it's AlphaFold

                field_df = self.df[self.df['source_field'] == field]

                # Count mentions by checking title
                mentions = field_df[
                    field_df['title'].str.lower().str.contains('|'.join(keywords), na=False)
                ]

                if len(mentions) == 0:
                    continue

                # First adoption year in this field
                first_year = mentions['year'].min()
                adoption_lag = first_year - origin_year if first_year > origin_year else 0

                # Adoption rate in field
                adoption_rate = len(mentions) / len(field_df) if len(field_df) > 0 else 0

                # Impact trajectory
                yearly = mentions.groupby('year').size()
                if len(yearly) >= 2:
                    recent = yearly.tail(2).values
                    if recent[-1] > recent[0]:
                        trajectory = 'growing'
                    elif recent[-1] < recent[0]:
                        trajectory = 'declining'
                    else:
                        trajectory = 'stable'
                else:
                    trajectory = 'emerging'

                results.append(DiscoveryPath(
                    ml_technique=method,
                    origin_year=origin_year,
                    target_field=field,
                    first_adoption_year=int(first_year),
                    adoption_lag_years=adoption_lag,
                    papers_count=len(mentions),
                    adoption_rate_in_field=round(adoption_rate, 5),
                    impact_trajectory=trajectory
                ))

        return sorted(results, key=lambda x: x.papers_count, reverse=True)

    def compute_landmark_impact(self) -> List[LandmarkImpact]:
        """
        Analyze impact of landmark ML methods across all fields.
        """
        results = []

        for method, info in LANDMARK_METHODS.items():
            keywords = info['keywords']
            origin_year = info['year']

            # Find all mentions
            mask = self.df['title'].str.lower().str.contains('|'.join(keywords), na=False)
            mentions = self.df[mask]

            if len(mentions) == 0:
                continue

            # Fields reached
            fields_reached = mentions['source_field'].unique().tolist()

            # Adoption by field
            adoption_by_field = mentions.groupby('source_field').size().to_dict()

            # Peak year
            yearly = mentions.groupby('year').size()
            peak_year = int(yearly.idxmax()) if len(yearly) > 0 else None

            # Adoption velocity
            if len(yearly) >= 2:
                velocity = (yearly.iloc[-1] - yearly.iloc[0]) / len(yearly)
            else:
                velocity = 0

            # Transformational score
            # Based on: spread across fields, total mentions, adoption speed
            field_spread = len(fields_reached) / len(self.fields)
            mention_score = min(len(mentions) / 100, 1.0)
            transformational = (field_spread * 0.4 + mention_score * 0.4 + min(velocity / 10, 0.2))

            results.append(LandmarkImpact(
                method=method,
                origin_year=origin_year,
                total_mentions=len(mentions),
                fields_reached=fields_reached,
                adoption_by_field=adoption_by_field,
                peak_year=peak_year,
                adoption_velocity=round(velocity, 2),
                transformational_score=round(transformational, 3)
            ))

        return sorted(results, key=lambda x: x.transformational_score, reverse=True)

    # =========================================================================
    # CLASSIFICATION & INSIGHTS
    # =========================================================================

    def classify_fields(self) -> Dict[str, List[str]]:
        """
        Classify fields into categories based on ML impact.

        Categories:
        - transformational: High ML impact + high quality
        - emerging: Growing ML adoption with good quality
        - hype_risk: High adoption + low reproducibility
        - traditional: Low ML adoption
        """
        impact = {m.field: m for m in self.compute_impact_metrics()}
        quality = {m.field: m for m in self.compute_quality_metrics()}
        adoption = {m.field: m for m in self.compute_adoption_dynamics()}

        classification = {
            'transformational': [],
            'emerging': [],
            'hype_risk': [],
            'traditional': [],
        }

        for field in self.fields:
            if field not in impact or field not in quality:
                continue

            imp = impact[field]
            qual = quality[field]
            adop = adoption.get(field)

            # Transformational: High impact + good quality
            if imp.ml_attribution_score > 0.3 and qual.code_availability_rate > 0.1:
                classification['transformational'].append(field)
            # Hype risk: High adoption + low quality
            elif qual.hype_risk_score > 0.5:
                classification['hype_risk'].append(field)
            # Emerging: Growing adoption
            elif adop and adop.current_trend in ['accelerating', 'growing']:
                classification['emerging'].append(field)
            # Traditional: Low ML presence
            elif imp.ml_attribution_score < 0.1:
                classification['traditional'].append(field)
            else:
                classification['emerging'].append(field)

        return classification

    def generate_key_insights(self) -> List[Dict]:
        """
        Generate key actionable insights from the data.
        """
        insights = []

        # Overall stats
        total = len(self.df)
        ml_count = self.df['is_ml_paper'].sum()
        ml_rate = ml_count / total if total > 0 else 0

        insights.append({
            'type': 'overview',
            'title': 'ML Penetration Across Science',
            'value': f'{ml_rate:.1%}',
            'description': f'{ml_count:,} of {total:,} papers use ML methods'
        })

        # Fastest growing field
        adoption = self.compute_adoption_dynamics()
        if adoption:
            fastest = max(adoption, key=lambda x: x.adoption_velocity)
            insights.append({
                'type': 'adoption',
                'title': 'Fastest ML Adoption',
                'value': fastest.field,
                'description': f'+{fastest.adoption_velocity:.1%}/year growth rate'
            })

        # Quality comparison
        comparison = self.compute_ml_vs_nonml_comparison()
        code_delta = comparison['delta']['code_rate']
        insights.append({
            'type': 'quality',
            'title': 'ML vs Traditional: Code Availability',
            'value': f'{code_delta:+.1%}',
            'description': 'ML papers are more likely to share code' if code_delta > 0 else 'Traditional papers share more code'
        })

        # Most impactful landmark
        landmarks = self.compute_landmark_impact()
        if landmarks:
            top_landmark = landmarks[0]
            insights.append({
                'type': 'landmark',
                'title': 'Most Transformational Method',
                'value': top_landmark.method.upper(),
                'description': f'Reached {len(top_landmark.fields_reached)} fields with {top_landmark.total_mentions:,} mentions'
            })

        return insights

    # =========================================================================
    # DASHBOARD DATA EXPORT
    # =========================================================================

    def get_dashboard_data(self) -> Dict:
        """
        Get all metrics for dashboard display.

        Returns comprehensive dict with all 4 objectives.
        Enhanced with animation-friendly data structures.
        """
        impact_metrics = self.compute_impact_metrics()
        adoption_metrics = self.compute_adoption_dynamics()
        quality_metrics = self.compute_quality_metrics()
        discovery_paths = self.compute_discovery_paths()
        landmarks = self.compute_landmark_impact()
        classification = self.classify_fields()
        insights = self.generate_key_insights()
        temporal_df = self.compute_temporal_evolution()
        s_curve_df = self.compute_s_curve_data()

        return {
            'summary': {
                'total_papers': len(self.df),
                'ml_papers': int(self.df['is_ml_paper'].sum()),
                'overall_ml_rate': float(self.df['is_ml_paper'].mean()),
                'overall_code_rate': float(self.df['has_code'].mean()),
                'fields_analyzed': len(self.fields),
                'year_range': (min(self.years), max(self.years)) if self.years else (None, None),
            },
            'ml_impact': [asdict(m) for m in impact_metrics],
            'adoption_dynamics': [asdict(m) for m in adoption_metrics],
            'quality_tradeoffs': [asdict(m) for m in quality_metrics],
            'discovery_paths': [asdict(p) for p in discovery_paths[:50]],  # Top 50
            'landmark_impact': [asdict(lm) for lm in landmarks],
            'classification': classification,
            'ml_vs_nonml': self.compute_ml_vs_nonml_comparison(),
            'temporal_evolution': temporal_df.to_dict('records') if len(temporal_df) > 0 else [],
            's_curve_data': s_curve_df.to_dict('records') if len(s_curve_df) > 0 else [],
            'insights': insights,
        }


def stats_module_correlation(x, y):
    """Helper for scipy correlation with fallback"""
    if SCIPY_AVAILABLE:
        return stats.pearsonr(x, y)
    return (0, 1)


# =============================================================================
# CLI TESTING
# =============================================================================

if __name__ == "__main__":
    import sys
    sys.path.insert(0, '/home/abheekp/ai_research/src')
    from data.loader import DataLoader

    print("Loading data...")
    loader = DataLoader("/home/abheekp/ai_research/s2orc_data")
    df = loader.load_sample(2000)

    print(f"\nInitializing metrics engine with {len(df)} papers...")
    engine = UnifiedMetricsEngine(df)

    print("\n" + "="*60)
    print("OBJECTIVE 1: ML IMPACT QUANTIFICATION")
    print("="*60)
    for m in engine.compute_impact_metrics()[:5]:
        print(f"  {m.field}: attribution={m.ml_attribution_score:.2f}, accel={m.acceleration_factor:.2f}x")

    print("\n" + "="*60)
    print("OBJECTIVE 2: ADOPTION DYNAMICS")
    print("="*60)
    for m in engine.compute_adoption_dynamics()[:5]:
        print(f"  {m.field}: rate={m.adoption_rate:.1%}, phase={m.s_curve_phase}, trend={m.current_trend}")

    print("\n" + "="*60)
    print("OBJECTIVE 3: QUALITY TRADE-OFFS")
    print("="*60)
    for m in engine.compute_quality_metrics()[:5]:
        print(f"  {m.field}: code={m.code_availability_rate:.1%}, hype_risk={m.hype_risk_score:.2f}")

    print("\n" + "="*60)
    print("OBJECTIVE 4: DISCOVERY IMPACT")
    print("="*60)
    for p in engine.compute_discovery_paths()[:5]:
        print(f"  {p.ml_technique} → {p.target_field}: lag={p.adoption_lag_years}yr, papers={p.papers_count}")

    print("\n" + "="*60)
    print("FIELD CLASSIFICATION")
    print("="*60)
    classification = engine.classify_fields()
    for cat, fields in classification.items():
        if fields:
            print(f"  {cat}: {', '.join(fields[:3])}")

    print("\n" + "="*60)
    print("KEY INSIGHTS")
    print("="*60)
    for insight in engine.generate_key_insights():
        print(f"  {insight['title']}: {insight['value']}")
