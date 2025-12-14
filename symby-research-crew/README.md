# 🔬 Symby Research Crew

**Autonomous multi-agent system for research paper analysis, code discovery, and implementation verification.**

Built with [CrewAI](https://crewai.com), this system deploys a team of specialized AI agents that work together to:

1. 📄 **Analyze** uploaded research papers
2. 📚 **Research** related literature on arXiv and Wikipedia  
3. 💻 **Find** code implementations on GitHub
4. ✅ **Verify** that code actually implements what papers claim

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Paper Research Crew                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐   │
│  │    Paper     │    │  Literature  │    │     Code     │   │
│  │   Analyst    │───▶│  Researcher  │───▶│    Hunter    │   │
│  └──────────────┘    └──────────────┘    └──────────────┘   │
│         │                   │                    │           │
│         │                   │                    ▼           │
│         │                   │            ┌──────────────┐   │
│         │                   │            │     Code     │   │
│         │                   │            │   Verifier   │   │
│         │                   │            └──────────────┘   │
│         │                   │                    │           │
│         ▼                   ▼                    ▼           │
│  ┌──────────────────────────────────────────────────────┐   │
│  │                    Synthesizer                        │   │
│  │            (Creates final research report)            │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/symby-research-crew
cd symby-research-crew

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys
```

### Basic Usage

```bash
# Full research on a paper
python main.py research paper.pdf --output report.md

# Quick topic research
python main.py quick "transformer architectures attention mechanism"

# Find code for a paper
python main.py code "Attention Is All You Need" --keywords pytorch attention

# Verify a repository
python main.py verify https://github.com/user/repo "implements multi-head attention"
```

### Python API

```python
from crew import create_research_crew

# Initialize crew
crew = create_research_crew(
    model="gpt-4o",
    provider="openai",
    verbose=True,
)

# Full research workflow
results = crew.research_paper(
    paper_path="paper.pdf",
    output_path="report.md",
)

# Quick research
summary = crew.quick_research("self-attention mechanisms")

# Find code
code_results = crew.find_code(
    paper_title="Attention Is All You Need",
    keywords=["pytorch", "transformer"],
)

# Verify implementation
verification = crew.verify_repository(
    repo_url="https://github.com/user/repo",
    paper_claims="Implements multi-head self-attention with positional encoding",
)
```

## 🤖 Agents

### Paper Analyst
- Parses PDF research papers
- Extracts title, abstract, methods, results
- Identifies key contributions and claims
- **Tools**: PDF Parser, Section Extractor, Metadata Extractor

### Literature Researcher  
- Searches arXiv for related papers
- Finds prior work and citations
- Explains background concepts via Wikipedia
- **Tools**: ArXiv Search, Wikipedia Search, Concept Definer

### Code Hunter
- Searches GitHub and Papers With Code
- Identifies official vs community implementations
- Evaluates repository quality and activity
- **Tools**: GitHub Search, Repo Analyzer, Papers With Code

### Code Verifier
- Analyzes code structure and implementation
- Verifies alignment with paper claims
- Identifies gaps and missing features
- **Tools**: Static Analyzer, Code Verification, Repo Structure Analyzer

### Synthesizer
- Coordinates findings from all agents
- Produces comprehensive research reports
- Provides actionable recommendations

## 🛠️ Tools

### ArXiv Tools
| Tool | Description |
|------|-------------|
| `arxiv_search` | Search papers by topic |
| `arxiv_get_paper` | Get paper details by ID |
| `arxiv_find_related` | Find related papers |

### GitHub Tools
| Tool | Description |
|------|-------------|
| `github_search_repos` | Search repositories |
| `github_analyze_repo` | Analyze repo structure |
| `github_fetch_code` | Fetch specific files |
| `papers_with_code_search` | Search Papers With Code |

### Wikipedia Tools
| Tool | Description |
|------|-------------|
| `wikipedia_search` | Search articles |
| `wikipedia_get_page` | Get full page content |
| `define_concept` | Quick concept definitions |

### PDF Tools
| Tool | Description |
|------|-------------|
| `parse_paper_pdf` | Full paper analysis |
| `extract_paper_section` | Get specific sections |
| `get_paper_metadata` | Extract metadata |

### Code Tools
| Tool | Description |
|------|-------------|
| `analyze_code_static` | Static code analysis |
| `verify_code_claims` | Match code to paper |
| `analyze_repo_structure` | Repo structure analysis |

## ⚙️ Configuration

### Environment Variables

```bash
# LLM Provider
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# Tool APIs (optional)
GITHUB_TOKEN=ghp_...

# Crew Settings
CREW_MODEL=gpt-4o
CREW_PROVIDER=openai
CREW_VERBOSE=true
```

### Model Options

```bash
# OpenAI (default)
python main.py research paper.pdf --model gpt-4o --provider openai

# Anthropic Claude
python main.py research paper.pdf --model claude-sonnet-4-20250514 --provider anthropic

# Local Ollama
python main.py research paper.pdf --model llama3.1:8b --provider ollama
```

## 📁 Project Structure

```
symby-research-crew/
├── agents/
│   ├── __init__.py
│   └── research_agents.py    # Agent definitions
├── tasks/
│   ├── __init__.py
│   └── research_tasks.py     # Task definitions
├── tools/
│   ├── __init__.py
│   ├── arxiv_tools.py        # ArXiv research tools
│   ├── github_tools.py       # GitHub code search tools
│   ├── wikipedia_tools.py    # Wikipedia research tools
│   ├── pdf_tools.py          # PDF parsing tools
│   └── code_tools.py         # Code analysis tools
├── config/
│   └── crew_config.yaml      # Crew configuration
├── utils/
│   └── ...
├── crew.py                   # Main crew orchestration
├── main.py                   # CLI entry point
├── requirements.txt
├── .env.example
└── README.md
```

## 🔄 Workflow

```
1. Upload Paper
      │
      ▼
2. Paper Analysis
   - Extract metadata
   - Parse sections
   - Identify claims
      │
      ▼
3. Literature Research
   - Search arXiv
   - Find related work
   - Background context
      │
      ▼
4. Code Discovery
   - Search GitHub
   - Papers With Code
   - Rank implementations
      │
      ▼
5. Code Verification
   - Analyze structure
   - Compare to paper
   - Identify gaps
      │
      ▼
6. Synthesis
   - Compile findings
   - Generate report
   - Recommendations
```

## 📊 Example Output

```markdown
# Research Report: "Attention Is All You Need"

## Executive Summary
This paper introduces the Transformer architecture, which relies 
entirely on attention mechanisms without recurrence or convolution...

## Paper Analysis
- **Authors**: Vaswani et al. (Google)
- **Key Contribution**: Self-attention mechanism for sequence modeling
- **Methods**: Multi-head attention, positional encoding, layer normalization

## Related Literature
1. Bahdanau Attention (2014) - Foundation for attention mechanisms
2. Neural Machine Translation - Application domain
...

## Code Implementations
| Repo | Stars | Official | Framework |
|------|-------|----------|-----------|
| tensorflow/tensor2tensor | 12.5k | ✅ | TensorFlow |
| huggingface/transformers | 85k | ❌ | PyTorch |

## Verification Results
✅ **HIGH ALIGNMENT** (87%)
- Multi-head attention: Implemented correctly
- Positional encoding: Present and matches paper
- Gap: Label smoothing not implemented in some repos

## Recommendations
1. Use `huggingface/transformers` for PyTorch
2. Review positional encoding implementation
3. Check hyperparameters match paper Table 3
```

## 🤝 Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

MIT License - see [LICENSE](LICENSE)

---

Built with ❤️ using [CrewAI](https://crewai.com)
