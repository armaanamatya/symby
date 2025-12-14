"""
SymbyAI - AI-Powered Research Paper Analysis Dashboard
Streamlines the journey from hypothesis to published paper
"""

import streamlit as st
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from collections import defaultdict
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src' / 'data'))
from paper_evaluator import PaperEvaluator


# Page config
st.set_page_config(
    page_title="SymbyAI - Research Paper Analysis",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    .success-box {
        background-color: #d4edda;
        border-color: #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border-color: #bee5eb;
        color: #0c5460;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_evaluations(dataset_type='ml'):
    """Load all evaluations from directory"""
    eval_dir = Path(f'{dataset_type}_evaluations')

    if not eval_dir.exists():
        return []

    evaluations = []
    for file in eval_dir.glob('*.json'):
        if file.name.startswith('_'):
            continue
        with open(file, 'r') as f:
            evaluations.append(json.load(f))

    return evaluations


@st.cache_data
def load_summary(dataset_type='ml'):
    """Load summary statistics"""
    summary_file = Path(f'{dataset_type}_evaluations/_summary.json')

    if not summary_file.exists():
        return None

    with open(summary_file, 'r') as f:
        return json.load(f)


def render_hero():
    """Render hero section"""
    st.markdown('<div class="main-header">🔬 SymbyAI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-Powered Research Paper Analysis Platform</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
    <h3>Streamline Your Research Journey</h3>
    <p><strong>SymbyAI</strong> streamlines the journey from hypothesis to published paper:</p>
    <ul>
        <li>🚀 <strong>Rapid Submission</strong>: Submit your paper and associated data</li>
        <li>🤖 <strong>AI Analysis</strong>: Comprehensive review of all materials</li>
        <li>🔄 <strong>Method Replication</strong>: Automated reproducibility checks</li>
        <li>📊 <strong>Comprehensive Insights</strong>: Detailed analysis in minutes</li>
    </ul>
    <p>Join the scientific AI revolution and improve the quality of your research!</p>
    </div>
    """, unsafe_allow_html=True)


def render_overview_stats():
    """Render overview statistics"""
    st.header("📊 Dataset Overview")

    # Load summaries
    ml_summary = load_summary('ml')
    nonml_summary = load_summary('nonml')

    if not ml_summary or not nonml_summary:
        st.error("❌ Evaluation data not found. Please run evaluations first.")
        return

    # Top metrics
    col1, col2, col3, col4 = st.columns(4)

    total_papers = ml_summary['total_papers'] + nonml_summary['total_papers']

    with col1:
        st.metric("Total Papers Analyzed", f"{total_papers:,}")

    with col2:
        st.metric("ML Papers", f"{ml_summary['total_papers']:,}")

    with col3:
        st.metric("Non-ML Papers", f"{nonml_summary['total_papers']:,}")

    with col4:
        success_rate = ((ml_summary['successfully_processed'] + nonml_summary['successfully_processed']) / total_papers) * 100
        st.metric("Success Rate", f"{success_rate:.1f}%")

    st.markdown("---")

    # Comparison metrics
    st.subheader("🔬 Key Comparison Metrics")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### ML Papers")
        ml_metrics = {
            "Code Availability": f"{ml_summary['code_availability_rate']:.1%}",
            "Data Availability": f"{ml_summary['data_availability_rate']:.1%}",
            "Novelty Claims": f"{ml_summary['novelty_claims_rate']:.1%}",
        }
        for metric, value in ml_metrics.items():
            st.metric(metric, value)

    with col2:
        st.markdown("### Non-ML Papers")
        nonml_metrics = {
            "Code Availability": f"{nonml_summary['code_availability_rate']:.1%}",
            "Data Availability": f"{nonml_summary['data_availability_rate']:.1%}",
            "Novelty Claims": f"{nonml_summary['novelty_claims_rate']:.1%}",
        }
        for metric, value in nonml_metrics.items():
            st.metric(metric, value)


def render_ml_adoption_analysis():
    """Render ML adoption analysis"""
    st.header("🤖 ML Adoption Analysis")

    ml_summary = load_summary('ml')
    nonml_summary = load_summary('nonml')

    if not ml_summary or not nonml_summary:
        return

    # ML Adoption levels comparison
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("ML Papers - Adoption Levels")
        ml_adoption = ml_summary['ml_adoption_levels']
        fig = px.pie(
            names=list(ml_adoption.keys()),
            values=list(ml_adoption.values()),
            title="Distribution of ML Adoption Levels",
            color_discrete_sequence=px.colors.sequential.Blues_r
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Non-ML Papers - Adoption Levels")
        nonml_adoption = nonml_summary['ml_adoption_levels']
        fig = px.pie(
            names=list(nonml_adoption.keys()),
            values=list(nonml_adoption.values()),
            title="Distribution of ML Adoption Levels",
            color_discrete_sequence=px.colors.sequential.Greens_r
        )
        st.plotly_chart(fig, use_container_width=True)

    # Bar chart comparison
    st.subheader("Side-by-Side Comparison")

    # Prepare data
    levels = ['core', 'substantial', 'moderate', 'minimal', 'none']
    ml_values = [ml_summary['ml_adoption_levels'].get(level, 0) for level in levels]
    nonml_values = [nonml_summary['ml_adoption_levels'].get(level, 0) for level in levels]

    fig = go.Figure(data=[
        go.Bar(name='ML Papers', x=levels, y=ml_values, marker_color='#1f77b4'),
        go.Bar(name='Non-ML Papers', x=levels, y=nonml_values, marker_color='#2ca02c')
    ])

    fig.update_layout(
        title="ML Adoption Levels Comparison",
        xaxis_title="Adoption Level",
        yaxis_title="Number of Papers",
        barmode='group',
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)


def render_reproducibility_analysis():
    """Render reproducibility analysis"""
    st.header("🔄 Reproducibility Analysis")

    ml_evals = load_evaluations('ml')
    nonml_evals = load_evaluations('nonml')

    if not ml_evals or not nonml_evals:
        st.warning("Loading evaluations...")
        return

    # Code and data availability
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Code Availability")

        ml_code = sum(1 for e in ml_evals if e['reproducibility']['code_availability_mentioned'])
        nonml_code = sum(1 for e in nonml_evals if e['reproducibility']['code_availability_mentioned'])

        data = {
            'Dataset': ['ML Papers', 'Non-ML Papers'],
            'With Code': [ml_code, nonml_code],
            'Without Code': [len(ml_evals) - ml_code, len(nonml_evals) - nonml_code]
        }

        df = pd.DataFrame(data)

        fig = go.Figure(data=[
            go.Bar(name='With Code', x=df['Dataset'], y=df['With Code'], marker_color='#2ca02c'),
            go.Bar(name='Without Code', x=df['Dataset'], y=df['Without Code'], marker_color='#d62728')
        ])

        fig.update_layout(
            barmode='stack',
            height=400,
            yaxis_title="Number of Papers"
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Data Availability")

        ml_data = sum(1 for e in ml_evals if e['reproducibility']['data_availability_mentioned'])
        nonml_data = sum(1 for e in nonml_evals if e['reproducibility']['data_availability_mentioned'])

        data = {
            'Dataset': ['ML Papers', 'Non-ML Papers'],
            'With Data': [ml_data, nonml_data],
            'Without Data': [len(ml_evals) - ml_data, len(nonml_evals) - nonml_data]
        }

        df = pd.DataFrame(data)

        fig = go.Figure(data=[
            go.Bar(name='With Data', x=df['Dataset'], y=df['With Data'], marker_color='#ff7f0e'),
            go.Bar(name='Without Data', x=df['Dataset'], y=df['Without Data'], marker_color='#bcbd22')
        ])

        fig.update_layout(
            barmode='stack',
            height=400,
            yaxis_title="Number of Papers"
        )

        st.plotly_chart(fig, use_container_width=True)

    # Methodology detail levels
    st.subheader("Methodology Detail Levels")

    ml_methodology = defaultdict(int)
    nonml_methodology = defaultdict(int)

    for e in ml_evals:
        level = e['reproducibility']['methodology_detail_level']
        ml_methodology[level] += 1

    for e in nonml_evals:
        level = e['reproducibility']['methodology_detail_level']
        nonml_methodology[level] += 1

    levels = ['high', 'moderate', 'low']
    ml_values = [ml_methodology.get(level, 0) for level in levels]
    nonml_values = [nonml_methodology.get(level, 0) for level in levels]

    fig = go.Figure(data=[
        go.Bar(name='ML Papers', x=levels, y=ml_values, marker_color='#1f77b4'),
        go.Bar(name='Non-ML Papers', x=levels, y=nonml_values, marker_color='#2ca02c')
    ])

    fig.update_layout(
        title="Methodology Detail Level Distribution",
        xaxis_title="Detail Level",
        yaxis_title="Number of Papers",
        barmode='group',
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)


def render_paper_upload():
    """Render paper upload and analysis section"""
    st.header("📄 Analyze Your Paper")

    st.markdown("""
    <div class="success-box">
    <h3>🚀 Get Instant AI Analysis</h3>
    <p>Upload your research paper and get comprehensive insights in minutes!</p>
    </div>
    """, unsafe_allow_html=True)

    # Input method selection
    input_method = st.radio(
        "Choose input method:",
        ["Paste Text", "Upload File"],
        horizontal=True
    )

    paper_text = None
    paper_id = None
    field = None

    if input_method == "Paste Text":
        paper_text = st.text_area(
            "Paste your paper text here:",
            height=300,
            placeholder="Paste the full text of your research paper..."
        )

        col1, col2 = st.columns(2)
        with col1:
            paper_id = st.text_input("Paper ID (optional)", value="uploaded_paper")
        with col2:
            field = st.text_input("Field/Domain (optional)", value="Unknown")

    else:  # Upload File
        uploaded_file = st.file_uploader(
            "Upload your paper (TXT file)",
            type=['txt'],
            help="Upload a plain text file containing your research paper"
        )

        if uploaded_file:
            paper_text = uploaded_file.read().decode('utf-8')
            paper_id = uploaded_file.name.replace('.txt', '')
            field = st.text_input("Field/Domain (optional)", value="Unknown")

    # Analyze button
    if st.button("🔍 Analyze Paper", type="primary", use_container_width=True):
        if not paper_text or not paper_text.strip():
            st.error("❌ Please provide paper text")
            return

        with st.spinner("🤖 AI Analysis in progress..."):
            # Create evaluator
            evaluator = PaperEvaluator()

            # Create paper object
            paper = {
                'id': paper_id,
                'text': paper_text,
                'field': field,
                'publication_date': 2024,
                'matched_term': None
            }

            # Evaluate
            evaluation = evaluator.evaluate_paper(paper)

            # Display results
            st.success("✅ Analysis Complete!")

            # Summary metrics
            st.subheader("📊 Quick Summary")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("ML Adoption", evaluation['ml_adoption']['ml_adoption_level'].upper())

            with col2:
                code_status = "✅ Yes" if evaluation['reproducibility']['code_availability_mentioned'] else "❌ No"
                st.metric("Code Available", code_status)

            with col3:
                data_status = "✅ Yes" if evaluation['reproducibility']['data_availability_mentioned'] else "❌ No"
                st.metric("Data Available", data_status)

            with col4:
                st.metric("Confidence", evaluation['confidence_score'].upper())

            # Detailed results
            st.subheader("🔬 Detailed Analysis")

            tab1, tab2, tab3, tab4 = st.tabs([
                "ML Adoption",
                "Reproducibility",
                "Research Outcomes",
                "Impact Indicators"
            ])

            with tab1:
                st.json(evaluation['ml_adoption'])

            with tab2:
                st.json(evaluation['reproducibility'])

            with tab3:
                st.json(evaluation['research_outcomes'])

            with tab4:
                st.json(evaluation['impact_indicators'])

            # Full evaluation
            with st.expander("📋 View Full Evaluation JSON"):
                st.json(evaluation)


def render_field_analysis():
    """Render field-based analysis"""
    st.header("🏆 Analysis by Research Field")

    ml_evals = load_evaluations('ml')
    nonml_evals = load_evaluations('nonml')

    if not ml_evals or not nonml_evals:
        st.warning("Loading evaluations...")
        return

    # Group by field
    field_stats = defaultdict(lambda: {
        'total': 0,
        'ml': 0,
        'code_available': 0,
        'data_available': 0,
        'claims_novelty': 0
    })

    for e in ml_evals + nonml_evals:
        field = e.get('field', 'Unknown')
        field_stats[field]['total'] += 1

        if e['ml_adoption']['ml_adoption_level'] in ['substantial', 'core']:
            field_stats[field]['ml'] += 1

        if e['reproducibility']['code_availability_mentioned']:
            field_stats[field]['code_available'] += 1

        if e['reproducibility']['data_availability_mentioned']:
            field_stats[field]['data_available'] += 1

        if e['impact_indicators']['claims_novelty']:
            field_stats[field]['claims_novelty'] += 1

    # Convert to dataframe
    df_data = []
    for field, stats in field_stats.items():
        df_data.append({
            'Field': field,
            'Total Papers': stats['total'],
            'ML Adoption %': (stats['ml'] / stats['total'] * 100) if stats['total'] > 0 else 0,
            'Code Availability %': (stats['code_available'] / stats['total'] * 100) if stats['total'] > 0 else 0,
            'Data Availability %': (stats['data_available'] / stats['total'] * 100) if stats['total'] > 0 else 0,
            'Novelty Claims %': (stats['claims_novelty'] / stats['total'] * 100) if stats['total'] > 0 else 0,
        })

    df = pd.DataFrame(df_data).sort_values('Total Papers', ascending=False)

    # Display table
    st.dataframe(
        df.style.format({
            'Total Papers': '{:,.0f}',
            'ML Adoption %': '{:.1f}%',
            'Code Availability %': '{:.1f}%',
            'Data Availability %': '{:.1f}%',
            'Novelty Claims %': '{:.1f}%',
        }),
        use_container_width=True,
        height=400
    )

    # Top fields chart
    st.subheader("Top 10 Research Fields by Paper Count")

    top_fields = df.head(10)

    fig = px.bar(
        top_fields,
        x='Field',
        y='Total Papers',
        color='ML Adoption %',
        color_continuous_scale='Blues',
        title="Research Fields Distribution"
    )

    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)


def main():
    """Main app"""

    # Sidebar navigation
    st.sidebar.title("🧭 Navigation")
    page = st.sidebar.radio(
        "Go to:",
        [
            "🏠 Home",
            "📊 Overview Statistics",
            "🤖 ML Adoption Analysis",
            "🔄 Reproducibility Analysis",
            "🏆 Field Analysis",
            "📄 Analyze Your Paper"
        ]
    )

    # Add info in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### About SymbyAI

    SymbyAI is an AI-powered platform for analyzing research papers and improving scientific quality.

    **Features:**
    - 🚀 Rapid paper analysis
    - 🤖 AI-powered insights
    - 🔄 Reproducibility checks
    - 📊 Comprehensive statistics

    **Dataset:**
    - ML Papers: 3,739
    - Non-ML Papers: 25,882
    - Total: 29,621 papers
    """)

    # Render hero on all pages
    render_hero()
    st.markdown("---")

    # Route to appropriate page
    if page == "🏠 Home":
        st.info("👈 Use the sidebar to navigate to different sections")
        render_overview_stats()

    elif page == "📊 Overview Statistics":
        render_overview_stats()

    elif page == "🤖 ML Adoption Analysis":
        render_ml_adoption_analysis()

    elif page == "🔄 Reproducibility Analysis":
        render_reproducibility_analysis()

    elif page == "🏆 Field Analysis":
        render_field_analysis()

    elif page == "📄 Analyze Your Paper":
        render_paper_upload()


if __name__ == "__main__":
    main()
