"""
AI Research Impact Observatory
DGX Spark Frontier Hackathon - Symby AI Track

Interactive dashboard quantifying ML's real impact on scientific progress.

All 4 Build Objectives:
1. Quantify ML Impact - Attribution scoring, acceleration metrics
2. Visualize Adoption Dynamics - S-curves, temporal evolution
3. Analyze Quality Trade-offs - Reproducibility vs ML adoption
4. Trace Discovery Impact - ML method → domain → real-world paths

Optimized for DGX Spark with 128GB unified memory
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import sys
import os
import requests
import time
import re

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data.loader import DataLoader
from metrics.impact_metrics import UnifiedMetricsEngine

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="AI Research Impact Observatory",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# NVIDIA THEME COLORS
# =============================================================================

NVIDIA_GREEN = "#76b900"
NVIDIA_DARK = "#1a1a2e"
COLORS = {
    'primary': NVIDIA_GREEN,
    'secondary': '#667eea',
    'accent': '#764ba2',
    'success': '#28a745',
    'warning': '#ffa500',
    'danger': '#dc3545',
    'info': '#17a2b8',
}

# =============================================================================
# CUSTOM CSS
# =============================================================================

st.markdown("""
<style>
    /* Main theme */
    .main-header {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #76b900 0%, #1a1a2e 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        padding: 1rem 0;
    }

    .sub-header {
        font-size: 1.3rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        padding: 1.5rem;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }

    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
    }

    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }

    /* Insight boxes */
    .insight-box {
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
        border-left: 5px solid #76b900;
        padding: 1.2rem;
        margin: 1rem 0;
        border-radius: 0 12px 12px 0;
    }

    .insight-title {
        font-weight: bold;
        color: #1a1a2e;
        margin-bottom: 0.5rem;
    }

    /* NVIDIA badge */
    .nvidia-badge {
        background: linear-gradient(135deg, #76b900 0%, #5a9100 100%);
        color: white;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
    }

    /* Objective tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6;
        border-radius: 8px;
        padding: 10px 20px;
    }

    .stTabs [aria-selected="true"] {
        background-color: #76b900 !important;
        color: white !important;
    }

    /* Section headers */
    .section-header {
        font-size: 1.5rem;
        font-weight: 700;
        color: #1a1a2e;
        border-bottom: 3px solid #76b900;
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
    }

    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        background: linear-gradient(135deg, #1a1a2e 0%, #2d2d44 100%);
        color: white;
        border-radius: 12px;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# DATA LOADING
# =============================================================================

@st.cache_data(ttl=3600)
def load_data(data_path: str, max_papers: int = None):
    """Load and preprocess data with caching"""
    loader = DataLoader(data_path)
    df = loader.load_all_papers(max_papers=max_papers)
    return df


@st.cache_data(ttl=3600)
def compute_metrics(_df):
    """Compute all metrics with caching"""
    engine = UnifiedMetricsEngine(_df)
    return engine.get_dashboard_data()


# =============================================================================
# HEADER
# =============================================================================

def render_header():
    """Render the main header with NVIDIA branding"""
    col1, col2, col3 = st.columns([1, 4, 1])

    with col1:
        st.image("https://upload.wikimedia.org/wikipedia/sco/thumb/2/21/Nvidia_logo.svg/200px-Nvidia_logo.svg.png", width=100)

    with col2:
        st.markdown('<p class="main-header">AI Research Impact Observatory</p>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">Quantifying Machine Learning\'s Real Impact on Scientific Progress</p>', unsafe_allow_html=True)

    with col3:
        st.markdown('<span class="nvidia-badge">DGX Spark</span>', unsafe_allow_html=True)
        st.markdown('<span class="nvidia-badge">RAPIDS</span>', unsafe_allow_html=True)

    st.markdown("---")


# =============================================================================
# SIDEBAR
# =============================================================================

def render_sidebar():
    """Render sidebar with data loading and filters"""
    st.sidebar.markdown("### 🔧 Configuration")

    data_path = st.sidebar.text_input(
        "📂 Data Path",
        value="s2orc_data"
    )

    max_papers = st.sidebar.slider(
        "📊 Papers to Load",
        min_value=1000,
        max_value=100000,
        value=10000,
        step=1000,
        help="More papers = better insights but slower loading"
    )

    load_button = st.sidebar.button("🚀 Load Data", type="primary")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📋 Build Objectives")
    st.sidebar.markdown("""
    1. **Quantify ML Impact** - Attribution & acceleration
    2. **Adoption Dynamics** - S-curves & temporal evolution
    3. **Quality Trade-offs** - Reproducibility analysis
    4. **Discovery Impact** - ML → Domain → Impact paths
    """)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🖥️ Spark Story")
    st.sidebar.info("""
    **Why DGX Spark?**
    - 128GB unified memory holds entire graph + LLM
    - RAPIDS cuDF for GPU-accelerated data loading
    - cuGraph for real-time citation analysis
    - Local inference for privacy & low latency
    """)

    return data_path, max_papers, load_button


# =============================================================================
# KEY METRICS OVERVIEW
# =============================================================================

def render_key_metrics(dashboard_data):
    """Render key metrics overview"""
    summary = dashboard_data['summary']

    cols = st.columns(5)

    metrics = [
        ("📄 Total Papers", f"{summary['total_papers']:,}", ""),
        ("🤖 ML Papers", f"{summary['ml_papers']:,}", f"{summary['overall_ml_rate']:.1%}"),
        ("📈 ML Adoption", f"{summary['overall_ml_rate']:.1%}", "overall"),
        ("💻 Code Available", f"{summary['overall_code_rate']:.1%}", ""),
        ("🔬 Fields", f"{summary['fields_analyzed']}", "analyzed"),
    ]

    for col, (label, value, delta) in zip(cols, metrics):
        with col:
            st.metric(label, value, delta if delta else None)


def render_insights(dashboard_data):
    """Render key insights"""
    insights = dashboard_data.get('insights', [])

    if not insights:
        return

    st.markdown("### 💡 Key Insights")

    cols = st.columns(len(insights))

    icons = {'overview': '📊', 'adoption': '📈', 'quality': '✅', 'landmark': '🏆'}

    for col, insight in zip(cols, insights):
        with col:
            icon = icons.get(insight['type'], '💡')
            st.markdown(f"""
            <div class="insight-box">
                <div class="insight-title">{icon} {insight['title']}</div>
                <div style="font-size: 1.8rem; font-weight: bold; color: #76b900;">{insight['value']}</div>
                <div style="font-size: 0.85rem; color: #666;">{insight['description']}</div>
            </div>
            """, unsafe_allow_html=True)


# =============================================================================
# OBJECTIVE 1: ML IMPACT QUANTIFICATION
# =============================================================================

def render_ml_impact(dashboard_data):
    """Render Objective 1: ML Impact Quantification"""
    st.markdown('<p class="section-header">📊 Objective 1: Quantify ML Impact</p>', unsafe_allow_html=True)
    st.markdown("*Measuring how much ML actually contributes to scientific breakthroughs*")

    impact_data = dashboard_data.get('ml_impact', [])

    if not impact_data:
        st.info("Insufficient data for impact analysis")
        return

    col1, col2 = st.columns(2)

    with col1:
        # Attribution Scores
        st.subheader("🎯 ML Attribution Score by Field")

        fields = [d['field'] for d in impact_data]
        scores = [d['ml_attribution_score'] * 100 for d in impact_data]

        fig = go.Figure(go.Bar(
            x=scores,
            y=fields,
            orientation='h',
            marker=dict(
                color=scores,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Score %")
            ),
            text=[f"{s:.1f}%" for s in scores],
            textposition='outside'
        ))

        fig.update_layout(
            xaxis_title="Attribution Score (%)",
            yaxis_title="",
            height=400,
            margin=dict(l=150)
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Acceleration Factors
        st.subheader("🚀 Research Acceleration Factor")

        accel = [d['acceleration_factor'] for d in impact_data]

        colors = [COLORS['success'] if a > 1 else COLORS['danger'] for a in accel]

        fig = go.Figure(go.Bar(
            x=accel,
            y=fields,
            orientation='h',
            marker_color=colors,
            text=[f"{a:.2f}x" for a in accel],
            textposition='outside'
        ))

        fig.add_vline(x=1, line_dash="dash", line_color="gray",
                     annotation_text="Baseline")

        fig.update_layout(
            xaxis_title="Acceleration Factor",
            yaxis_title="",
            height=400,
            margin=dict(l=150)
        )
        st.plotly_chart(fig, use_container_width=True)

    # Technique diversity & cross-pollination
    st.subheader("🔄 Technique Diversity & Cross-Pollination")

    col1, col2 = st.columns(2)

    with col1:
        diversity = [d['technique_diversity'] for d in impact_data]

        fig = go.Figure(go.Bar(
            x=diversity,
            y=fields,
            orientation='h',
            marker_color=COLORS['secondary'],
            text=diversity,
            textposition='outside'
        ))

        fig.update_layout(
            title="Unique ML Techniques per Field",
            xaxis_title="# Techniques",
            height=400,
            margin=dict(l=150)
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        cross_poll = [d['cross_pollination_score'] for d in impact_data]

        fig = go.Figure(go.Bar(
            x=cross_poll,
            y=fields,
            orientation='h',
            marker_color=COLORS['accent'],
            text=[f"{c:.2f}" for c in cross_poll],
            textposition='outside'
        ))

        fig.update_layout(
            title="Cross-Pollination Score",
            xaxis_title="Score",
            height=400,
            margin=dict(l=150)
        )
        st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# OBJECTIVE 2: ADOPTION DYNAMICS
# =============================================================================

def render_adoption_dynamics(dashboard_data, df):
    """Render Objective 2: Adoption Dynamics Visualization"""
    st.markdown('<p class="section-header">📈 Objective 2: Visualize Adoption Dynamics</p>', unsafe_allow_html=True)
    st.markdown("*How ML techniques spread across scientific disciplines over time*")

    adoption_data = dashboard_data.get('adoption_dynamics', [])
    temporal_data = dashboard_data.get('temporal_evolution', [])

    if not adoption_data:
        st.info("Insufficient data for adoption analysis")
        return

    col1, col2 = st.columns(2)

    with col1:
        # Current Adoption Rates
        st.subheader("📊 Current ML Adoption Rates")

        fields = [d['field'] for d in adoption_data]
        rates = [d['adoption_rate'] * 100 for d in adoption_data]
        phases = [d['s_curve_phase'] for d in adoption_data]

        phase_colors = {
            'early': COLORS['info'],
            'growth': COLORS['success'],
            'mature': COLORS['warning'],
            'declining': COLORS['danger']
        }
        colors = [phase_colors.get(p, COLORS['secondary']) for p in phases]

        fig = go.Figure(go.Bar(
            x=rates,
            y=fields,
            orientation='h',
            marker_color=colors,
            text=[f"{r:.1f}%" for r in rates],
            textposition='outside'
        ))

        fig.update_layout(
            xaxis_title="Adoption Rate (%)",
            height=400,
            margin=dict(l=150)
        )
        st.plotly_chart(fig, use_container_width=True)

        # Legend for phases
        st.markdown("**S-Curve Phases:** 🔵 Early | 🟢 Growth | 🟡 Mature | 🔴 Declining")

    with col2:
        # Adoption Velocity
        st.subheader("⚡ Adoption Velocity")

        velocities = [d['adoption_velocity'] * 100 for d in adoption_data]

        fig = go.Figure(go.Bar(
            x=velocities,
            y=fields,
            orientation='h',
            marker_color=[COLORS['success'] if v > 0 else COLORS['danger'] for v in velocities],
            text=[f"{v:+.2f}%/yr" for v in velocities],
            textposition='outside'
        ))

        fig.add_vline(x=0, line_dash="dash", line_color="gray")

        fig.update_layout(
            xaxis_title="Velocity (%/year)",
            height=400,
            margin=dict(l=150)
        )
        st.plotly_chart(fig, use_container_width=True)

    # Temporal Evolution Chart
    st.subheader("📅 Temporal Evolution (2007-2022)")

    if temporal_data:
        temp_df = pd.DataFrame(temporal_data)

        fig = px.line(
            temp_df,
            x='year',
            y='ml_rate',
            color='field',
            title='ML Adoption Rate Over Time by Field',
            markers=True
        )

        fig.update_layout(
            xaxis_title="Year",
            yaxis_title="ML Adoption Rate",
            yaxis_tickformat='.0%',
            height=500,
            legend=dict(orientation="h", yanchor="bottom", y=-0.3)
        )
        st.plotly_chart(fig, use_container_width=True)

    # S-Curve Analysis
    st.subheader("📉 S-Curve Phase Distribution")

    phase_counts = {}
    for d in adoption_data:
        phase = d['s_curve_phase']
        phase_counts[phase] = phase_counts.get(phase, 0) + 1

    fig = go.Figure(go.Pie(
        labels=list(phase_counts.keys()),
        values=list(phase_counts.values()),
        hole=0.4,
        marker_colors=[phase_colors.get(p, COLORS['secondary']) for p in phase_counts.keys()]
    ))

    fig.update_layout(
        title="Fields by S-Curve Phase",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# OBJECTIVE 3: QUALITY TRADE-OFFS
# =============================================================================

def render_quality_tradeoffs(dashboard_data):
    """Render Objective 3: Quality Trade-offs Analysis"""
    st.markdown('<p class="section-header">⚖️ Objective 3: Analyze Quality Trade-offs</p>', unsafe_allow_html=True)
    st.markdown("*Investigating whether ML adoption correlates with better or worse research reproducibility*")

    quality_data = dashboard_data.get('quality_tradeoffs', [])
    ml_vs_nonml = dashboard_data.get('ml_vs_nonml', {})

    if not quality_data:
        st.info("Insufficient data for quality analysis")
        return

    # ML vs Non-ML Comparison
    st.subheader("🔬 ML vs Non-ML Papers: Quality Comparison")

    col1, col2, col3 = st.columns(3)

    with col1:
        ml_count = ml_vs_nonml.get('ml_papers', {}).get('count', 0)
        nonml_count = ml_vs_nonml.get('nonml_papers', {}).get('count', 0)

        fig = go.Figure(go.Pie(
            labels=['ML Papers', 'Non-ML Papers'],
            values=[ml_count, nonml_count],
            hole=0.4,
            marker_colors=[COLORS['primary'], COLORS['secondary']]
        ))
        fig.update_layout(title="Paper Distribution", height=300)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        ml_code = ml_vs_nonml.get('ml_papers', {}).get('code_rate', 0) * 100
        nonml_code = ml_vs_nonml.get('nonml_papers', {}).get('code_rate', 0) * 100

        fig = go.Figure(go.Bar(
            x=['ML Papers', 'Non-ML Papers'],
            y=[ml_code, nonml_code],
            marker_color=[COLORS['primary'], COLORS['secondary']],
            text=[f"{ml_code:.1f}%", f"{nonml_code:.1f}%"],
            textposition='outside'
        ))
        fig.update_layout(title="Code Availability", yaxis_title="%", height=300)
        st.plotly_chart(fig, use_container_width=True)

    with col3:
        delta = ml_vs_nonml.get('delta', {}).get('code_rate', 0) * 100
        color = COLORS['success'] if delta > 0 else COLORS['danger']

        st.markdown(f"""
        <div style="text-align: center; padding: 2rem;">
            <div style="font-size: 3rem; font-weight: bold; color: {color};">{delta:+.1f}%</div>
            <div style="font-size: 1rem; color: #666;">Code Availability Delta</div>
            <div style="font-size: 0.9rem; margin-top: 1rem;">
                {"✅ ML papers share more code!" if delta > 0 else "⚠️ Traditional papers share more code"}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Quality by Field
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("💻 Code Availability by Field")

        fields = [d['field'] for d in quality_data]
        code_rates = [d['code_availability_rate'] * 100 for d in quality_data]

        fig = go.Figure(go.Bar(
            x=code_rates,
            y=fields,
            orientation='h',
            marker_color=COLORS['primary'],
            text=[f"{r:.1f}%" for r in code_rates],
            textposition='outside'
        ))

        fig.update_layout(
            xaxis_title="Code Availability (%)",
            height=400,
            margin=dict(l=150)
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("⚠️ Hype Risk Score")

        hype_scores = [d['hype_risk_score'] for d in quality_data]

        colors = [COLORS['danger'] if h > 0.5 else COLORS['warning'] if h > 0.3 else COLORS['success']
                 for h in hype_scores]

        fig = go.Figure(go.Bar(
            x=hype_scores,
            y=fields,
            orientation='h',
            marker_color=colors,
            text=[f"{h:.2f}" for h in hype_scores],
            textposition='outside'
        ))

        fig.add_vline(x=0.5, line_dash="dash", line_color="red",
                     annotation_text="High Risk Threshold")

        fig.update_layout(
            xaxis_title="Hype Risk Score",
            height=400,
            margin=dict(l=150)
        )
        st.plotly_chart(fig, use_container_width=True)

    # ML vs Non-ML by Field
    st.subheader("📊 ML vs Non-ML Code Availability by Field")

    ml_rates = [d['ml_code_rate'] * 100 for d in quality_data]
    nonml_rates = [d['nonml_code_rate'] * 100 for d in quality_data]

    fig = go.Figure()
    fig.add_trace(go.Bar(name='ML Papers', x=fields, y=ml_rates, marker_color=COLORS['primary']))
    fig.add_trace(go.Bar(name='Non-ML Papers', x=fields, y=nonml_rates, marker_color=COLORS['secondary']))

    fig.update_layout(
        barmode='group',
        xaxis_title="Field",
        yaxis_title="Code Availability (%)",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# OBJECTIVE 4: DISCOVERY IMPACT TRACING
# =============================================================================

def render_discovery_impact(dashboard_data):
    """Render Objective 4: Discovery Impact Tracing"""
    st.markdown('<p class="section-header">🔍 Objective 4: Trace Discovery Impact</p>', unsafe_allow_html=True)
    st.markdown("*Following the path from ML method development to real-world scientific impact*")

    discovery_paths = dashboard_data.get('discovery_paths', [])
    landmark_impact = dashboard_data.get('landmark_impact', [])

    if not discovery_paths and not landmark_impact:
        st.info("Insufficient data for discovery impact analysis")
        return

    # Landmark Methods Impact
    st.subheader("🏆 Landmark ML Methods Impact")

    if landmark_impact:
        col1, col2 = st.columns(2)

        with col1:
            methods = [d['method'].upper() for d in landmark_impact]
            mentions = [d['total_mentions'] for d in landmark_impact]

            fig = go.Figure(go.Bar(
                x=mentions,
                y=methods,
                orientation='h',
                marker=dict(
                    color=mentions,
                    colorscale='Plasma'
                ),
                text=mentions,
                textposition='outside'
            ))

            fig.update_layout(
                title="Total Mentions by Method",
                xaxis_title="Paper Count",
                height=400,
                margin=dict(l=100)
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            trans_scores = [d['transformational_score'] for d in landmark_impact]

            fig = go.Figure(go.Bar(
                x=trans_scores,
                y=methods,
                orientation='h',
                marker_color=COLORS['primary'],
                text=[f"{s:.2f}" for s in trans_scores],
                textposition='outside'
            ))

            fig.update_layout(
                title="Transformational Score",
                xaxis_title="Score",
                height=400,
                margin=dict(l=100)
            )
            st.plotly_chart(fig, use_container_width=True)

    # Discovery Paths Sankey
    st.subheader("🌊 ML Method → Domain Flow")

    if discovery_paths:
        # Create Sankey data
        methods = list(set(d['ml_technique'] for d in discovery_paths[:30]))
        fields = list(set(d['target_field'] for d in discovery_paths[:30]))

        all_nodes = methods + fields
        node_indices = {n: i for i, n in enumerate(all_nodes)}

        sources = []
        targets = []
        values = []

        for path in discovery_paths[:30]:
            if path['ml_technique'] in node_indices and path['target_field'] in node_indices:
                sources.append(node_indices[path['ml_technique']])
                targets.append(node_indices[path['target_field']])
                values.append(path['papers_count'])

        if sources:
            fig = go.Figure(go.Sankey(
                node=dict(
                    pad=15,
                    thickness=20,
                    label=all_nodes,
                    color=[COLORS['primary'] if n in methods else COLORS['secondary'] for n in all_nodes]
                ),
                link=dict(
                    source=sources,
                    target=targets,
                    value=values,
                    color='rgba(118, 185, 0, 0.3)'
                )
            ))

            fig.update_layout(
                title="ML Methods Flow to Domain Sciences",
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)

    # Adoption Lag Analysis
    st.subheader("⏱️ Adoption Lag: Years from ML Origin to Domain Application")

    if discovery_paths:
        path_df = pd.DataFrame(discovery_paths)

        fig = px.scatter(
            path_df,
            x='origin_year',
            y='adoption_lag_years',
            size='papers_count',
            color='target_field',
            hover_data=['ml_technique'],
            title='Adoption Lag vs Origin Year'
        )

        fig.update_layout(
            xaxis_title="ML Method Origin Year",
            yaxis_title="Years to Domain Adoption",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# FIELD CLASSIFICATION
# =============================================================================

def render_classification(dashboard_data):
    """Render field classification results"""
    st.markdown("### 🏷️ Field Classification")

    classification = dashboard_data.get('classification', {})

    cols = st.columns(4)

    categories = [
        ('transformational', '✅ Transformational', 'High ML impact + High quality', COLORS['success']),
        ('emerging', '🌱 Emerging', 'Growing ML adoption', COLORS['warning']),
        ('hype_risk', '⚠️ Hype Risk', 'High adoption, low reproducibility', COLORS['danger']),
        ('traditional', '📚 Traditional', 'Low ML presence', COLORS['info']),
    ]

    for col, (key, title, desc, color) in zip(cols, categories):
        with col:
            fields = classification.get(key, [])
            st.markdown(f"**{title}**")
            st.markdown(f"*{desc}*")

            if fields:
                for field in fields[:5]:
                    st.markdown(f"• {field}")
            else:
                st.markdown("*None identified*")


# =============================================================================
# PAPER CODE EXECUTOR TAB
# =============================================================================

def extract_metrics_from_output(stdout: str) -> dict:
    """
    Extract numeric metrics from execution output.
    Looks for common patterns like 'accuracy: 0.95', 'loss=2.3', etc.
    """
    metrics = {}

    # Common metric patterns
    patterns = [
        r'(?:accuracy|acc)[:\s=]+([0-9.]+)%?',
        r'(?:loss)[:\s=]+([0-9.]+)',
        r'(?:precision)[:\s=]+([0-9.]+)%?',
        r'(?:recall)[:\s=]+([0-9.]+)%?',
        r'(?:f1[-_]?score|f1)[:\s=]+([0-9.]+)%?',
        r'(?:auc|auroc)[:\s=]+([0-9.]+)',
        r'(?:top-?1)[:\s=]+([0-9.]+)%?',
        r'(?:top-?5)[:\s=]+([0-9.]+)%?',
        r'(?:mse|mae|rmse)[:\s=]+([0-9.]+)',
        r'(?:bleu)[:\s=]+([0-9.]+)',
        r'(?:perplexity|ppl)[:\s=]+([0-9.]+)',
    ]

    for pattern in patterns:
        matches = re.findall(pattern, stdout, re.IGNORECASE)
        if matches:
            # Extract metric name from pattern
            metric_name = pattern.split('(?:')[1].split(')')[0].replace('|', '_').replace('[-_]?', '_')
            # Take the last value found (usually the final result)
            try:
                metrics[metric_name] = float(matches[-1])
            except ValueError:
                pass

    return metrics


def render_pce_tab():
    """Render the Paper Code Executor tab"""
    st.markdown('<p class="section-header">📄 Paper Code Executor</p>', unsafe_allow_html=True)
    st.markdown("*Extract, execute, and validate code from research papers in isolated sandboxes*")

    # Initialize session state for PCE
    if "pce_code_manifest" not in st.session_state:
        st.session_state.pce_code_manifest = None
    if "pce_execution_result" not in st.session_state:
        st.session_state.pce_execution_result = None
    if "pce_validation_result" not in st.session_state:
        st.session_state.pce_validation_result = None
    if "pce_error" not in st.session_state:
        st.session_state.pce_error = None

    # API base URL - configurable
    api_base_url = st.sidebar.text_input(
        "🔗 PCE API URL",
        value="http://localhost:8000",
        help="Base URL for the Paper Code Executor API"
    )

    # ==========================================================================
    # INPUT SECTION
    # ==========================================================================
    st.subheader("🔍 Paper Input")

    col1, col2 = st.columns([4, 1])
    with col1:
        paper_id = st.text_input(
            "Paper ID",
            placeholder="arxiv:2010.11929 or DOI",
            help="Enter arXiv ID (e.g., arxiv:2010.11929) or DOI"
        )
    with col2:
        st.write("")  # Spacing
        st.write("")
        extract_btn = st.button("🔍 Extract Code", type="primary")

    # ==========================================================================
    # EXTRACTION SECTION
    # ==========================================================================
    if extract_btn and paper_id:
        st.session_state.pce_code_manifest = None
        st.session_state.pce_execution_result = None
        st.session_state.pce_validation_result = None
        st.session_state.pce_error = None

        with st.spinner("Extracting code from paper..."):
            try:
                # Start extraction job
                response = requests.post(
                    f"{api_base_url}/api/pce/extract",
                    json={"paper_id": paper_id},
                    timeout=30
                )
                response.raise_for_status()
                job_data = response.json()
                job_id = job_data.get("job_id")

                if not job_id:
                    st.error("Failed to start extraction job: No job ID returned")
                    st.session_state.pce_error = "No job ID returned"
                else:
                    # Poll for completion
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    max_polls = 60  # Max 60 seconds

                    for i in range(max_polls):
                        status_response = requests.get(
                            f"{api_base_url}/api/pce/extract/{job_id}",
                            timeout=10
                        )
                        status_response.raise_for_status()
                        status_data = status_response.json()

                        status = status_data.get("status", "unknown")
                        status_text.text(f"Status: {status}")
                        progress_bar.progress((i + 1) / max_polls)

                        if status == "complete":
                            st.session_state.pce_code_manifest = status_data.get("code_manifest")
                            progress_bar.progress(1.0)
                            st.success("✅ Code extraction complete!")
                            break
                        elif status == "error":
                            st.session_state.pce_error = status_data.get("error", "Unknown error")
                            st.error(f"❌ Extraction failed: {st.session_state.pce_error}")
                            break

                        time.sleep(1)
                    else:
                        st.warning("⏱️ Extraction timed out. Please try again.")

            except requests.exceptions.ConnectionError:
                st.error(f"❌ Cannot connect to API at {api_base_url}. Please check if the server is running.")
                st.session_state.pce_error = "Connection error"
            except requests.exceptions.Timeout:
                st.error("❌ Request timed out. Please try again.")
                st.session_state.pce_error = "Timeout"
            except requests.exceptions.HTTPError as e:
                st.error(f"❌ HTTP Error: {e}")
                st.session_state.pce_error = str(e)
            except Exception as e:
                st.error(f"❌ Unexpected error: {e}")
                st.session_state.pce_error = str(e)

    # ==========================================================================
    # EXTRACTED CODE DISPLAY
    # ==========================================================================
    if st.session_state.pce_code_manifest:
        st.markdown("---")
        st.subheader("📦 Extracted Code")

        manifest = st.session_state.pce_code_manifest
        code_blocks = manifest.get("code_blocks", [])

        with st.expander(f"📄 {len(code_blocks)} code blocks found", expanded=True):
            for i, block in enumerate(code_blocks):
                st.markdown(f"**Block {i + 1}** ({block.get('language', 'unknown')})")
                st.code(block.get("code", ""), language=block.get("language", "python"))

        # Metadata display
        col1, col2, col3 = st.columns(3)
        with col1:
            framework = manifest.get("framework", "Unknown")
            st.info(f"🔧 **Framework:** {framework}")
        with col2:
            deps = manifest.get("dependencies", [])
            st.info(f"📦 **Dependencies:** {len(deps)} packages")
        with col3:
            repos = manifest.get("github_repos", [])
            st.info(f"🔗 **GitHub Repos:** {len(repos)} found")

        # Show dependencies in expander
        if deps:
            with st.expander("📋 Dependencies"):
                for dep in deps:
                    st.text(f"• {dep}")

        # Show GitHub repos if found
        if repos:
            with st.expander("🔗 GitHub Repositories"):
                for repo in repos:
                    st.markdown(f"• [{repo}]({repo})")

        # ==========================================================================
        # EXECUTION SECTION
        # ==========================================================================
        st.markdown("---")
        st.subheader("▶️ Execution")

        col1, col2 = st.columns(2)
        with col1:
            env = st.selectbox(
                "Environment",
                ["pytorch", "tensorflow", "jax"],
                help="Select the ML framework environment for execution"
            )
        with col2:
            timeout = st.number_input(
                "Timeout (seconds)",
                min_value=60,
                max_value=600,
                value=300,
                help="Maximum execution time in seconds"
            )

        execute_btn = st.button("▶️ Execute in Sandbox", type="primary")

        if execute_btn:
            st.session_state.pce_execution_result = None
            st.session_state.pce_validation_result = None

            with st.spinner("Running code in isolated container..."):
                try:
                    # Start execution job
                    response = requests.post(
                        f"{api_base_url}/api/pce/execute",
                        json={
                            "code_manifest_id": manifest.get("id"),
                            "environment": env,
                            "timeout_seconds": timeout
                        },
                        timeout=30
                    )
                    response.raise_for_status()
                    job_data = response.json()
                    job_id = job_data.get("job_id")

                    if not job_id:
                        st.error("Failed to start execution job: No job ID returned")
                    else:
                        # Poll for completion with live output
                        output_placeholder = st.empty()
                        progress_bar = st.progress(0)
                        max_polls = timeout + 30  # Extra buffer for setup

                        for i in range(max_polls):
                            status_response = requests.get(
                                f"{api_base_url}/api/pce/execute/{job_id}",
                                timeout=10
                            )
                            status_response.raise_for_status()
                            result_data = status_response.json()

                            status = result_data.get("status", "unknown")
                            progress_bar.progress(min((i + 1) / max_polls, 0.95))

                            # Show live output if available
                            if result_data.get("result"):
                                stdout = result_data["result"].get("stdout", "")
                                if stdout:
                                    output_placeholder.code(stdout[-2000:], language="bash")

                            if status in ["complete", "error", "timeout"]:
                                st.session_state.pce_execution_result = result_data.get("result")
                                progress_bar.progress(1.0)
                                if status == "complete":
                                    st.success("✅ Execution complete!")
                                elif status == "timeout":
                                    st.warning("⏱️ Execution timed out")
                                else:
                                    st.error("❌ Execution failed")
                                break

                            time.sleep(1)
                        else:
                            st.warning("⏱️ Polling timed out")

                except requests.exceptions.ConnectionError:
                    st.error(f"❌ Cannot connect to API at {api_base_url}")
                except requests.exceptions.Timeout:
                    st.error("❌ Request timed out")
                except requests.exceptions.HTTPError as e:
                    st.error(f"❌ HTTP Error: {e}")
                except Exception as e:
                    st.error(f"❌ Unexpected error: {e}")

    # ==========================================================================
    # EXECUTION RESULTS DISPLAY
    # ==========================================================================
    if st.session_state.pce_execution_result:
        result = st.session_state.pce_execution_result

        st.markdown("---")
        st.subheader("📊 Execution Output")

        # Main output
        stdout = result.get("stdout", "")
        if stdout:
            st.code(stdout, language="bash")
        else:
            st.info("No stdout output")

        # Stderr in expander
        stderr = result.get("stderr", "")
        if stderr:
            with st.expander("⚠️ Stderr", expanded=False):
                st.code(stderr, language="bash")

        # Metrics display
        cols = st.columns(3)

        status = result.get("status", "unknown")
        status_icon = "✅" if status == "success" else "❌"
        cols[0].metric("Status", f"{status_icon} {status.capitalize()}")

        exec_time = result.get("execution_time_seconds", 0)
        cols[1].metric("Time", f"{exec_time:.1f}s")

        memory = result.get("memory_peak_mb", 0)
        cols[2].metric("Memory", f"{memory:.0f} MB")

        # ==========================================================================
        # VALIDATION SECTION
        # ==========================================================================
        st.markdown("---")
        st.subheader("📊 Validation")

        validate_btn = st.button("📊 Validate Against Paper Claims", type="primary")

        if validate_btn:
            with st.spinner("Comparing results to paper claims..."):
                try:
                    extracted_metrics = extract_metrics_from_output(stdout)

                    response = requests.post(
                        f"{api_base_url}/api/pce/validate",
                        json={
                            "paper_id": st.session_state.pce_code_manifest.get("paper_id"),
                            "execution_results": {
                                "stdout": stdout,
                                "metrics": extracted_metrics
                            }
                        },
                        timeout=60
                    )
                    response.raise_for_status()
                    validation_data = response.json()
                    st.session_state.pce_validation_result = validation_data.get("validation_result")
                    st.success("✅ Validation complete!")

                except requests.exceptions.ConnectionError:
                    st.error(f"❌ Cannot connect to API at {api_base_url}")
                except requests.exceptions.Timeout:
                    st.error("❌ Validation request timed out")
                except requests.exceptions.HTTPError as e:
                    st.error(f"❌ HTTP Error: {e}")
                except Exception as e:
                    st.error(f"❌ Unexpected error: {e}")

    # ==========================================================================
    # VALIDATION RESULTS DISPLAY
    # ==========================================================================
    if st.session_state.pce_validation_result:
        validation = st.session_state.pce_validation_result

        st.markdown("---")
        st.subheader("✅ Validation Results")

        claims_tested = validation.get("claims_tested", [])

        if claims_tested:
            for claim in claims_tested:
                with st.container():
                    cols = st.columns([3, 1, 1])

                    metric_name = claim.get("metric_name", "Unknown")
                    claimed_value = claim.get("claimed_value", "N/A")
                    actual_value = claim.get("actual_value", "N/A")
                    passed = claim.get("passed", False)
                    difference_pct = claim.get("difference_pct", 0)

                    cols[0].markdown(f"**{metric_name}**: Claimed {claimed_value}")
                    cols[1].markdown(f"Actual: **{actual_value}**")

                    if passed:
                        cols[2].success("✅ Pass")
                    else:
                        cols[2].warning(f"⚠️ {difference_pct:.1f}% off")

            # Overall summary
            claims_passed = validation.get("claims_passed", 0)
            total_claims = validation.get("total_claims", len(claims_tested))

            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    "Overall Verification",
                    f"{claims_passed}/{total_claims} claims verified"
                )
            with col2:
                pass_rate = (claims_passed / total_claims * 100) if total_claims > 0 else 0
                if pass_rate >= 80:
                    st.success(f"🎉 High reproducibility ({pass_rate:.0f}%)")
                elif pass_rate >= 50:
                    st.warning(f"⚠️ Partial reproducibility ({pass_rate:.0f}%)")
                else:
                    st.error(f"❌ Low reproducibility ({pass_rate:.0f}%)")
        else:
            st.info("No claims were tested. The validation system may not have found comparable metrics.")

    # ==========================================================================
    # EMPTY STATE
    # ==========================================================================
    if not st.session_state.pce_code_manifest and not st.session_state.pce_error:
        st.markdown("---")
        st.info("""
        👆 **Getting Started:**
        1. Enter a paper ID (arXiv or DOI format)
        2. Click "Extract Code" to analyze the paper
        3. Review extracted code blocks and dependencies
        4. Execute in an isolated sandbox environment
        5. Validate results against paper claims
        """)


# =============================================================================
# MAIN APP
# =============================================================================

def main():
    render_header()

    # Initialize session state
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'dashboard_data' not in st.session_state:
        st.session_state.dashboard_data = None
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False

    # Sidebar
    data_path, max_papers, load_button = render_sidebar()

    # Load data if not loaded or button pressed
    if not st.session_state.data_loaded or load_button:
        with st.spinner("🔄 Loading and analyzing data..."):
            try:
                st.session_state.df = load_data(data_path, max_papers)
                st.session_state.dashboard_data = compute_metrics(st.session_state.df)
                st.session_state.data_loaded = True
                st.sidebar.success(f"✅ Loaded {len(st.session_state.df):,} papers")
            except Exception as e:
                st.error(f"Error loading data: {str(e)}")
                st.info("Please check that the data path is correct and data files exist.")
                st.stop()

    # Check if data is available
    if st.session_state.df is None or st.session_state.dashboard_data is None:
        st.warning("⏳ Please wait for data to load...")
        st.stop()

    df = st.session_state.df
    dashboard_data = st.session_state.dashboard_data

    # Key Metrics
    render_key_metrics(dashboard_data)
    render_insights(dashboard_data)

    st.markdown("---")

    # Tabs for objectives
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 ML Impact",
        "📈 Adoption Dynamics",
        "⚖️ Quality Trade-offs",
        "🔍 Discovery Impact",
        "🏷️ Classification",
        "📄 Paper Code Executor"
    ])

    with tab1:
        render_ml_impact(dashboard_data)

    with tab2:
        render_adoption_dynamics(dashboard_data, df)

    with tab3:
        render_quality_tradeoffs(dashboard_data)

    with tab4:
        render_discovery_impact(dashboard_data)

    with tab5:
        render_classification(dashboard_data)

    with tab6:
        render_pce_tab()

    # Footer
    st.markdown("---")
    st.markdown("""
    <div class="footer">
        <h3>AI Research Impact Observatory</h3>
        <p>DGX Spark Frontier Hackathon | Symby AI Track</p>
        <p>Built with <span style="color: #76b900;">NVIDIA RAPIDS</span> • cuGraph • Streamlit • Plotly</p>
        <p style="font-size: 0.85rem; margin-top: 1rem;">
            🖥️ <strong>Spark Story:</strong> 128GB Unified Memory enables holding entire citation graph + LLM simultaneously
        </p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
