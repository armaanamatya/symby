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
import io

# PDF extraction
try:
    import fitz  # PyMuPDF
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src' / 'data'))
sys.path.insert(0, str(Path(__file__).parent / 'src'))
from paper_evaluator import PaperEvaluator

# Standard imports (always available)
import re as regex_module
import asyncio

# PCE imports for sandbox execution
try:
    from pce.sandbox_runner import SandboxRunner, ExecutionResult
    from pce.code_extractor import CodeExtractor, CodeManifest, CodeBlock
    SANDBOX_SUPPORT = True
except ImportError as e:
    SANDBOX_SUPPORT = False
    print(f"Sandbox support not available: {e}")


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


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract text from PDF bytes using PyMuPDF"""
    if not PDF_SUPPORT:
        raise ImportError("PyMuPDF (fitz) is required for PDF extraction. Install with: pip install PyMuPDF")

    text_parts = []
    try:
        # Open PDF from bytes
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")

        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            text = page.get_text()
            if text.strip():
                text_parts.append(text)

        pdf_document.close()

        return "\n\n".join(text_parts)
    except Exception as e:
        raise ValueError(f"Failed to extract text from PDF: {str(e)}")


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
        # Determine supported file types
        supported_types = ['txt']
        type_help = "Upload a plain text file"

        if PDF_SUPPORT:
            supported_types.append('pdf')
            type_help = "Upload a plain text (.txt) or PDF (.pdf) file"
        else:
            st.warning("PDF support not available. Install PyMuPDF for PDF uploads: `pip install PyMuPDF`")

        uploaded_file = st.file_uploader(
            "Upload your paper",
            type=supported_types,
            help=type_help + " containing your research paper"
        )

        if uploaded_file:
            file_extension = uploaded_file.name.lower().split('.')[-1]

            if file_extension == 'pdf':
                # Extract text from PDF
                try:
                    with st.spinner("Extracting text from PDF..."):
                        pdf_bytes = uploaded_file.read()
                        paper_text = extract_text_from_pdf(pdf_bytes)

                    if not paper_text.strip():
                        st.error("Could not extract text from PDF. The PDF may be image-based or corrupted.")
                        paper_text = None
                    else:
                        st.success(f"Successfully extracted {len(paper_text):,} characters from PDF")
                except Exception as e:
                    st.error(f"Error reading PDF: {str(e)}")
                    paper_text = None
            else:
                # Handle text file
                paper_text = uploaded_file.read().decode('utf-8')

            paper_id = uploaded_file.name.rsplit('.', 1)[0]  # Remove any extension
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

            # Store evaluation and paper text in session state for sandbox
            st.session_state['paper_evaluation'] = evaluation
            st.session_state['paper_text'] = paper_text
            st.session_state['paper_id'] = paper_id

    # ==========================================================================
    # SANDBOX EXECUTION SECTION (persisted via session state)
    # ==========================================================================
    # Render sandbox section if we have analyzed a paper
    if 'paper_evaluation' in st.session_state and 'paper_text' in st.session_state:
        st.markdown("---")
        render_sandbox_section(
            st.session_state['paper_text'],
            st.session_state['paper_evaluation'],
            st.session_state['paper_id']
        )


def extract_code_blocks_from_text(text: str) -> list:
    """Extract code blocks from paper text using regex patterns."""
    code_blocks = []

    # Pattern 1: Markdown code blocks with language
    pattern1 = regex_module.compile(r'```(\w+)?\s*(.*?)```', regex_module.DOTALL)
    for match in pattern1.finditer(text):
        lang = match.group(1) or 'python'
        code = match.group(2).strip()
        if code and len(code) > 20:  # Filter out very short snippets
            code_blocks.append({
                'language': lang.lower(),
                'code': code,
                'source': 'paper'
            })

    # Pattern 2: Indented code blocks (4+ spaces)
    lines = text.split('\n')
    current_block = []
    in_code_block = False

    for line in lines:
        if line.startswith('    ') or line.startswith('\t'):
            current_block.append(line.strip())
            in_code_block = True
        else:
            if in_code_block and current_block:
                code = '\n'.join(current_block)
                # Check if it looks like code (has common programming constructs)
                if any(kw in code for kw in ['import ', 'def ', 'class ', 'for ', 'if ', '= ', '(', ')']):
                    if len(code) > 50:
                        code_blocks.append({
                            'language': 'python',
                            'code': code,
                            'source': 'paper'
                        })
                current_block = []
                in_code_block = False

    # Pattern 3: Algorithm/pseudocode blocks
    algo_pattern = regex_module.compile(
        r'(?:Algorithm|Procedure|Function|Code)[:\s]*\d*\s*[\n\r]+(.*?)(?=\n\n|\Z)',
        regex_module.DOTALL | regex_module.IGNORECASE
    )
    for match in algo_pattern.finditer(text):
        code = match.group(1).strip()
        if code and len(code) > 30:
            code_blocks.append({
                'language': 'python',
                'code': code,
                'source': 'paper'
            })

    return code_blocks[:10]  # Limit to 10 blocks


def run_code_in_sandbox(code: str, timeout: int = 60) -> dict:
    """Execute code in a sandbox container."""
    if not SANDBOX_SUPPORT:
        return {
            'status': 'error',
            'stdout': '',
            'stderr': 'Sandbox support not available. Please install docker and pce dependencies.',
            'execution_time_seconds': 0,
            'memory_peak_mb': 0
        }

    try:
        runner = SandboxRunner()

        # Run the code synchronously
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                runner.run(code, timeout_seconds=timeout)
            )
            return {
                'status': result.status,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'exit_code': result.exit_code,
                'execution_time_seconds': result.execution_time_seconds,
                'memory_peak_mb': result.memory_peak_mb,
                'gpu_utilization_pct': result.gpu_utilization_pct,
                'gpu_memory_used_mb': result.gpu_memory_used_mb
            }
        finally:
            loop.close()
    except Exception as e:
        return {
            'status': 'error',
            'stdout': '',
            'stderr': str(e),
            'execution_time_seconds': 0,
            'memory_peak_mb': 0
        }


def extract_code_from_arxiv(arxiv_id: str) -> dict:
    """Extract code from an arXiv paper using PCE module."""
    if not SANDBOX_SUPPORT:
        return {
            'success': False,
            'error': 'PCE module not available',
            'code_blocks': [],
            'github_repos': [],
            'framework': 'unknown',
            'dependencies': []
        }

    try:
        extractor = CodeExtractor()
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            manifest = loop.run_until_complete(extractor.extract_from_arxiv(arxiv_id))
            return {
                'success': True,
                'code_blocks': [
                    {'language': b.language, 'code': b.code, 'source': b.source}
                    for b in manifest.code_blocks
                ],
                'github_repos': manifest.github_repos,
                'framework': manifest.framework,
                'dependencies': list(manifest.dependencies)
            }
        finally:
            loop.run_until_complete(extractor.close())
            loop.close()
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'code_blocks': [],
            'github_repos': [],
            'framework': 'unknown',
            'dependencies': []
        }


def extract_code_from_github(repo_url: str) -> dict:
    """Extract code from a GitHub repository using PCE module."""
    if not SANDBOX_SUPPORT:
        return {
            'success': False,
            'error': 'PCE module not available',
            'code_blocks': []
        }

    try:
        extractor = CodeExtractor()
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            code_blocks = loop.run_until_complete(extractor.extract_from_github(repo_url))
            return {
                'success': True,
                'code_blocks': [
                    {'language': b.language, 'code': b.code, 'source': b.source}
                    for b in code_blocks
                ]
            }
        finally:
            loop.run_until_complete(extractor.close())
            loop.close()
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'code_blocks': []
        }


def render_sandbox_section(paper_text: str, evaluation: dict, paper_id: str):
    """Render the sandbox execution section after paper analysis."""
    st.subheader("🚀 Code Sandbox")

    if not SANDBOX_SUPPORT:
        st.warning("⚠️ Docker sandbox not available. Code extraction is still available, but execution requires Docker.")
        sandbox_disabled = True
    else:
        sandbox_disabled = False

    st.markdown("""
    <div class="info-box">
    <strong>Execute extracted code in an isolated Docker container</strong><br>
    Extract and run code from the paper to verify results and test reproducibility.
    </div>
    """, unsafe_allow_html=True)

    # Initialize session state for sandbox
    if 'sandbox_code_blocks' not in st.session_state:
        st.session_state.sandbox_code_blocks = []
    if 'sandbox_execution_result' not in st.session_state:
        st.session_state.sandbox_execution_result = None
    if 'sandbox_selected_code' not in st.session_state:
        st.session_state.sandbox_selected_code = ""
    if 'sandbox_github_repos' not in st.session_state:
        st.session_state.sandbox_github_repos = []
    if 'sandbox_framework' not in st.session_state:
        st.session_state.sandbox_framework = 'unknown'
    if 'sandbox_dependencies' not in st.session_state:
        st.session_state.sandbox_dependencies = []

    # Code Extraction Options
    st.markdown("### 📥 Code Extraction Source")
    extraction_source = st.radio(
        "Choose extraction source:",
        ["From Paper Text", "From arXiv ID", "From GitHub URL"],
        horizontal=True,
        key="extraction_source"
    )

    if extraction_source == "From arXiv ID":
        # arXiv ID input
        arxiv_col1, arxiv_col2 = st.columns([3, 1])
        with arxiv_col1:
            arxiv_id = st.text_input(
                "Enter arXiv ID:",
                placeholder="e.g., 2010.11929 (Vision Transformer paper)",
                help="Enter the arXiv paper ID to extract code and find GitHub repos"
            )
        with arxiv_col2:
            st.write("")
            st.write("")
            extract_arxiv_btn = st.button("🔍 Extract from arXiv", type="primary", use_container_width=True)

        if extract_arxiv_btn and arxiv_id.strip():
            with st.spinner(f"Extracting code from arXiv:{arxiv_id}..."):
                result = extract_code_from_arxiv(arxiv_id.strip())

                if result['success']:
                    st.session_state.sandbox_code_blocks = result['code_blocks']
                    st.session_state.sandbox_github_repos = result['github_repos']
                    st.session_state.sandbox_framework = result['framework']
                    st.session_state.sandbox_dependencies = result['dependencies']

                    st.success(f"✅ Extracted {len(result['code_blocks'])} code block(s)")

                    if result['github_repos']:
                        st.info(f"🔗 Found {len(result['github_repos'])} GitHub repo(s): {', '.join(result['github_repos'])}")

                    if result['framework'] != 'other':
                        st.info(f"🔧 Detected framework: {result['framework']}")

                    if result['dependencies']:
                        st.info(f"📦 Dependencies: {', '.join(result['dependencies'][:10])}")
                else:
                    st.error(f"❌ Failed to extract: {result.get('error', 'Unknown error')}")

        # Show GitHub repos and allow extraction
        if st.session_state.sandbox_github_repos:
            st.markdown("#### 🔗 GitHub Repositories Found in Paper")
            for i, repo_url in enumerate(st.session_state.sandbox_github_repos):
                repo_col1, repo_col2 = st.columns([3, 1])
                with repo_col1:
                    st.markdown(f"**{i+1}.** [{repo_url}]({repo_url})")
                with repo_col2:
                    if st.button(f"Extract Code", key=f"extract_github_{i}"):
                        with st.spinner(f"Cloning and extracting from {repo_url}..."):
                            github_result = extract_code_from_github(repo_url)
                            if github_result['success']:
                                # Append to existing code blocks
                                st.session_state.sandbox_code_blocks.extend(github_result['code_blocks'])
                                st.success(f"✅ Added {len(github_result['code_blocks'])} code block(s) from GitHub")
                            else:
                                st.error(f"❌ Failed: {github_result.get('error', 'Unknown error')}")

    elif extraction_source == "From GitHub URL":
        # GitHub URL input
        github_col1, github_col2 = st.columns([3, 1])
        with github_col1:
            github_url = st.text_input(
                "Enter GitHub URL:",
                placeholder="e.g., https://github.com/google-research/vision_transformer",
                help="Enter a GitHub repository URL to extract code"
            )
        with github_col2:
            st.write("")
            st.write("")
            extract_github_btn = st.button("🔍 Extract from GitHub", type="primary", use_container_width=True)

        if extract_github_btn and github_url.strip():
            with st.spinner(f"Cloning and extracting from {github_url}..."):
                result = extract_code_from_github(github_url.strip())

                if result['success']:
                    st.session_state.sandbox_code_blocks = result['code_blocks']
                    st.success(f"✅ Extracted {len(result['code_blocks'])} code block(s) from GitHub")
                else:
                    st.error(f"❌ Failed to extract: {result.get('error', 'Unknown error')}")

    else:  # From Paper Text
        # Extract Code Button
        col1, col2 = st.columns([1, 1])

        with col1:
            if st.button("🔍 Extract Code from Paper Text", type="secondary", use_container_width=True):
                with st.spinner("Extracting code blocks from paper..."):
                    code_blocks = extract_code_blocks_from_text(paper_text)
                    st.session_state.sandbox_code_blocks = code_blocks

                    if code_blocks:
                        st.success(f"✅ Found {len(code_blocks)} code block(s)")
                    else:
                        st.info("No executable code blocks found in the paper text.")

        with col2:
            # Framework detection from ML adoption analysis
            ml_adoption = evaluation.get('ml_adoption', {})
            frameworks = ml_adoption.get('ml_frameworks_mentioned', [])

            if frameworks:
                st.info(f"🔧 Detected frameworks: {', '.join(frameworks)}")
            else:
                st.info("🔧 No specific ML frameworks detected")

    # Debug info
    with st.expander("🐛 Debug Info", expanded=False):
        st.write(f"Code blocks in session state: {len(st.session_state.sandbox_code_blocks)}")
        st.write(f"Paper text length: {len(paper_text)} chars")
        if st.session_state.sandbox_code_blocks:
            st.write("First block preview:")
            st.json(st.session_state.sandbox_code_blocks[0])

    # Display extracted code blocks
    if st.session_state.sandbox_code_blocks:
        st.markdown("### 📦 Extracted Code Blocks")

        for i, block in enumerate(st.session_state.sandbox_code_blocks):
            source = block.get('source', 'unknown')
            with st.expander(f"Code Block {i + 1} - {source} ({block.get('language', 'unknown')})", expanded=(i == 0)):
                st.code(block.get('code', ''), language=block.get('language', 'python'))

                if st.button(f"▶️ Run Block {i + 1}", key=f"run_block_{i}"):
                    st.session_state.sandbox_selected_code = block.get('code', '')
                    st.rerun()

    # Custom code editor
    st.markdown("### ✏️ Code Editor")
    st.markdown("*Paste or edit code to run in the sandbox:*")

    # Pre-fill with selected code or first extracted block
    default_code = st.session_state.sandbox_selected_code
    if not default_code and st.session_state.sandbox_code_blocks:
        default_code = st.session_state.sandbox_code_blocks[0].get('code', '')

    custom_code = st.text_area(
        "Code to execute:",
        value=default_code,
        height=300,
        placeholder="# Paste Python code here to run in sandbox\nimport numpy as np\nprint('Hello from sandbox!')"
    )

    # Execution options
    col1, col2, col3 = st.columns(3)

    with col1:
        timeout = st.number_input(
            "Timeout (seconds)",
            min_value=10,
            max_value=600,
            value=60,
            help="Maximum execution time"
        )

    with col2:
        env_option = st.selectbox(
            "Environment",
            ["PyTorch", "TensorFlow", "JAX", "CPU-only"],
            help="Select ML framework environment"
        )

    with col3:
        st.write("")  # Spacing
        st.write("")
        run_btn = st.button(
            "▶️ Run in Sandbox",
            type="primary",
            use_container_width=True,
            disabled=sandbox_disabled
        )

    # Execute code
    if run_btn and custom_code.strip() and not sandbox_disabled:
        st.session_state.sandbox_execution_result = None

        with st.spinner("🔄 Running code in isolated container..."):
            result = run_code_in_sandbox(custom_code, timeout=timeout)
            st.session_state.sandbox_execution_result = result

    # Display execution results
    if st.session_state.sandbox_execution_result:
        result = st.session_state.sandbox_execution_result

        st.markdown("### 📊 Execution Results")

        # Status indicator
        status = result.get('status', 'unknown')
        if status == 'success':
            st.success(f"✅ Execution completed successfully!")
        elif status == 'timeout':
            st.warning(f"⏱️ Execution timed out after {timeout} seconds")
        elif status == 'oom':
            st.error("💾 Out of memory error")
        else:
            st.error(f"❌ Execution failed: {status}")

        # Metrics
        cols = st.columns(4)
        cols[0].metric("Status", status.capitalize())
        cols[1].metric("Time", f"{result.get('execution_time_seconds', 0):.2f}s")
        cols[2].metric("Memory", f"{result.get('memory_peak_mb', 0):.1f} MB")

        gpu_util = result.get('gpu_utilization_pct')
        if gpu_util is not None:
            cols[3].metric("GPU", f"{gpu_util:.1f}%")
        else:
            cols[3].metric("GPU", "N/A")

        # Output
        stdout = result.get('stdout', '')
        stderr = result.get('stderr', '')

        if stdout:
            st.markdown("**Output (stdout):**")
            st.code(stdout, language="bash")

        if stderr:
            with st.expander("⚠️ Errors/Warnings (stderr)", expanded=bool(status != 'success')):
                st.code(stderr, language="bash")

        # Save results option
        with st.expander("💾 Export Results"):
            export_data = {
                'paper_id': paper_id,
                'code': custom_code,
                'execution_result': result,
                'environment': env_option
            }
            st.download_button(
                "📥 Download Execution Report (JSON)",
                data=json.dumps(export_data, indent=2),
                file_name=f"sandbox_result_{paper_id}.json",
                mime="application/json"
            )


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
