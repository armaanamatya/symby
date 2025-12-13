# AI Research Impact Observatory

## DGX Spark Frontier Hackathon - Symby AI Track

**Comprehensive Technical Documentation**

---

## Executive Summary

This system **quantifies machine learning's real impact on scientific progress** across 14 scientific disciplines using 3.2+ million papers from S2ORC. It implements all 4 hackathon build objectives:

| Objective | Description | Key Metrics |
|-----------|-------------|-------------|
| **1. Quantify ML Impact** | Attribution scoring, acceleration | `ml_attribution_score`, `acceleration_factor` |
| **2. Visualize Adoption Dynamics** | S-curves, temporal evolution | `adoption_rate`, `s_curve_phase`, `velocity` |
| **3. Analyze Quality Trade-offs** | Reproducibility vs ML adoption | `hype_risk_score`, `code_availability_rate` |
| **4. Trace Discovery Impact** | ML → Domain paths | `adoption_lag_years`, `transformational_score` |

### The Spark Story (Why DGX Spark?)

- **128GB Unified Memory**: Holds entire paper corpus + metrics in memory simultaneously
- **RAPIDS cuDF**: GPU-accelerated DataFrame operations (10x faster than pandas)
- **cuGraph Support**: GPU-accelerated graph analysis when citation networks are loaded
- **Local Inference**: Privacy-preserving, low-latency analysis

---

## Quick Start

### Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt

# On DGX Spark (optional - for GPU acceleration)
conda install -c rapidsai -c nvidia -c conda-forge rapids=24.10
```

### Run Dashboard

```bash
# Streamlit Dashboard (all 4 objectives)
streamlit run app.py

# FastAPI Backend (for Next.js/custom frontends)
uvicorn api:app --host 0.0.0.0 --port 8000
```

### Test Pipeline

```bash
# Test data loader
python src/data/loader.py

# Test metrics engine
python src/metrics/impact_metrics.py
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     USER INTERFACES                                  │
│  ┌─────────────────────┐    ┌─────────────────────────────────┐    │
│  │  Streamlit Dashboard│    │  FastAPI REST API               │    │
│  │     (app.py)        │    │     (api.py)                    │    │
│  └─────────────────────┘    └─────────────────────────────────┘    │
├─────────────────────────────────────────────────────────────────────┤
│                     METRICS ENGINE                                   │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │            UnifiedMetricsEngine                              │   │
│  │         (src/metrics/impact_metrics.py)                     │   │
│  │                                                              │   │
│  │  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌──────────┐ │   │
│  │  │ Objective 1│ │ Objective 2│ │ Objective 3│ │Objective4│ │   │
│  │  │ ML Impact  │ │ Adoption   │ │ Quality    │ │ Discovery│ │   │
│  │  └────────────┘ └────────────┘ └────────────┘ └──────────┘ │   │
│  └─────────────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────────────┤
│                     DATA LAYER                                       │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │               DataLoader + MLClassifier                      │   │
│  │              (src/data/loader.py)                           │   │
│  │                                                              │   │
│  │  - Supports .json, .jsonl, .gz formats                      │   │
│  │  - cuDF/pandas backend auto-selection                       │   │
│  │  - Caching for fast repeated loads                          │   │
│  │  - ML paper detection (50+ keywords)                        │   │
│  │  - Reproducibility scoring                                  │   │
│  └─────────────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────────────┤
│                     S2ORC DATASET (3.2M+ papers)                    │
│  14 Fields | 2007-2022 | train/val/test splits                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Dataset Structure

### 3.2M+ Papers Across 14 Fields

| Field | Papers | % of Total |
|-------|--------|------------|
| Medicine | 890,702 | 27.4% |
| Biology | 477,169 | 14.7% |
| Computer Science | 312,225 | 9.6% |
| Physics | 309,073 | 9.5% |
| Engineering | 205,695 | 6.3% |
| Materials Science | 188,405 | 5.8% |
| Mathematics | 177,845 | 5.5% |
| Environmental Science | 177,513 | 5.5% |
| Agricultural Sciences | 141,743 | 4.4% |
| Psychology | 138,833 | 4.3% |
| Chemistry | 83,241 | 2.6% |
| Economics | 71,615 | 2.2% |
| NA (Multidisciplinary) | 38,511 | 1.2% |
| Political Science | 36,320 | 1.1% |

### Directory Structure

```
s2orc_data/
├── Biology,2017-2017/
│   ├── train/
│   │   └── Biology-2017.gz-0000.json   # JSONL format
│   ├── val/
│   │   └── Biology-2017.jsonl
│   └── test/
│       └── Biology-2017.jsonl
├── ComputerScience,2017-2017/
│   └── ...
└── [48 total directories]
```

### Paper Schema

```json
{
  "id": "12345678",
  "text": "Paper title and full text...",
  "created": "2020-01-15T00:00:00.000Z",
  "metadata": {
    "year": 2020,
    "s2fieldsofstudy": ["Computer Science", "Biology"],
    "extfieldsofstudy": ["Artificial Intelligence"]
  }
}
```

---

## Build Objectives Implementation

### Objective 1: Quantify ML Impact

**Goal**: Measure how much ML actually contributes to scientific breakthroughs.

**Implementation** (`compute_impact_metrics()`):

```python
@dataclass
class ImpactMetrics:
    field: str
    total_papers: int
    ml_papers: int
    ml_attribution_score: float    # 0-1: ML contribution %
    adoption_intensity: float      # Avg ML score for ML papers
    acceleration_factor: float     # Research output growth
    efficiency_score: float        # Impact per ML unit
    breakthrough_potential: float  # Landmark method presence
    technique_diversity: float     # Unique ML techniques
    cross_pollination_score: float # Methods from other fields
```

**Key Formulas**:

```
ml_attribution_score = min(adoption_rate * 2, 0.4)
                     + min(avg_ml_score / 20, 0.3)
                     + min(unique_techniques / 10, 0.3)

acceleration_factor = late_period_output / early_period_output
```

### Objective 2: Visualize Adoption Dynamics

**Goal**: Show how ML techniques spread across disciplines over time.

**Implementation** (`compute_adoption_dynamics()`):

```python
@dataclass
class AdoptionMetrics:
    field: str
    total_papers: int
    ml_papers: int
    adoption_rate: float           # Current ML adoption %
    adoption_velocity: float       # YoY change rate
    first_adoption_year: int
    peak_adoption_year: int
    s_curve_phase: str            # 'early'|'growth'|'mature'|'declining'
    growth_rate: float            # Compound annual growth
    years_to_50pct: int           # Extrapolated time to 50%
    current_trend: str            # 'accelerating'|'growing'|'stable'|'declining'
```

**S-Curve Phase Detection**:

| Rate | Velocity | Phase |
|------|----------|-------|
| < 10% | any | early |
| 10-30% | > 2%/yr | growth |
| > 30% | < 1%/yr | mature |
| any | < -1%/yr | declining |

### Objective 3: Analyze Quality Trade-offs

**Goal**: Investigate if ML adoption correlates with better/worse reproducibility.

**Implementation** (`compute_quality_metrics()`):

```python
@dataclass
class QualityMetrics:
    field: str
    ml_adoption_rate: float
    code_availability_rate: float   # Papers with code
    data_availability_rate: float   # Papers with data
    ml_code_rate: float            # Code rate for ML papers
    nonml_code_rate: float         # Code rate for non-ML papers
    quality_delta: float           # ML - NonML reproducibility
    hype_risk_score: float         # High adoption + low quality = risk
    quality_trend: str             # 'improving'|'stable'|'declining'
```

**Hype Risk Formula**:

```
hype_risk_score = (ml_adoption_rate / (code_availability_rate + 0.1)) * 0.5

# High risk if: high ML adoption + low code availability
# Score capped at 1.0
```

### Objective 4: Trace Discovery Impact

**Goal**: Follow paths from ML method → domain application → impact.

**Implementation** (`compute_discovery_paths()`, `compute_landmark_impact()`):

```python
@dataclass
class DiscoveryPath:
    ml_technique: str              # e.g., 'transformer', 'gan'
    origin_year: int               # When method was invented
    target_field: str              # Where it was adopted
    first_adoption_year: int       # When first seen in field
    adoption_lag_years: int        # Time from origin to adoption
    papers_count: int              # Papers using method in field
    adoption_rate_in_field: float
    impact_trajectory: str         # 'growing'|'stable'|'declining'

@dataclass
class LandmarkImpact:
    method: str
    origin_year: int
    total_mentions: int
    fields_reached: List[str]
    adoption_by_field: Dict[str, int]
    peak_year: int
    adoption_velocity: float
    transformational_score: float  # Cross-field spread + mentions
```

**Tracked Landmark Methods**:

| Method | Year | Domain |
|--------|------|--------|
| AlexNet | 2012 | Computer Vision |
| Word2Vec | 2013 | NLP |
| GAN | 2014 | Generative |
| ResNet | 2015 | Computer Vision |
| Transformer | 2017 | NLP |
| BERT | 2018 | NLP |
| GPT | 2018 | NLP |
| AlphaFold | 2020 | Biology |
| Diffusion | 2020 | Generative |

---

## ML Paper Classification

### Keyword Categories (50+ keywords)

**Techniques** (2 points each):
```
transformer, bert, gpt, llm, cnn, convolutional neural,
rnn, lstm, gan, generative adversarial, diffusion model,
reinforcement learning, graph neural, autoencoder...
```

**General ML** (1 point each):
```
deep learning, neural network, machine learning,
transfer learning, classification, embedding...
```

**Frameworks** (1 point each):
```
pytorch, tensorflow, keras, huggingface, cuda...
```

**Landmarks** (3 points each):
```
alphafold, alexnet, chatgpt, stable diffusion...
```

**Classification Rule**:
```python
is_ml_paper = (ml_score >= 3)
```

### Reproducibility Detection

**Code Indicators**:
```
github.com, gitlab.com, code available, source code,
implementation available, colab, jupyter notebook, docker
```

**Data Indicators**:
```
dataset available, data available, benchmark dataset,
open data, public dataset, supplementary material
```

---

## API Reference

### REST Endpoints (api.py)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API info and health |
| `/api/load` | POST | Load data (required first) |
| `/api/summary` | GET | Dataset summary |
| `/api/fields` | GET | All fields with stats |
| `/api/objectives/ml-impact` | GET | Objective 1 metrics |
| `/api/objectives/adoption` | GET | Objective 2 metrics |
| `/api/objectives/quality` | GET | Objective 3 metrics |
| `/api/objectives/discovery` | GET | Objective 4 metrics |
| `/api/field/{name}` | GET | All metrics for one field |
| `/api/dashboard` | GET | Complete dashboard data |
| `/api/insights` | GET | Key actionable insights |
| `/api/classification` | GET | Field classification |

### Example API Usage

```bash
# Load data (required first)
curl -X POST "http://localhost:8000/api/load?max_papers=50000"

# Get ML impact metrics
curl "http://localhost:8000/api/objectives/ml-impact"

# Get specific field
curl "http://localhost:8000/api/field/Biology"

# Get complete dashboard
curl "http://localhost:8000/api/dashboard"
```

---

## File Structure

```
ai_research/
├── app.py                      # Streamlit dashboard
├── api.py                      # FastAPI backend
├── requirements.txt            # Dependencies
├── DOCUMENTATION.md            # This file
├── CLAUDE.md                   # Context for AI assistants
│
├── src/
│   ├── __init__.py
│   ├── config.py               # Global configuration
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   └── loader.py           # DataLoader + MLClassifier
│   │
│   ├── metrics/
│   │   ├── __init__.py
│   │   └── impact_metrics.py   # UnifiedMetricsEngine
│   │
│   └── graph/
│       ├── __init__.py
│       ├── citation_graph.py   # Semantic Scholar API client
│       ├── graph_analysis.py   # cuGraph/NetworkX algorithms
│       └── graph_rag.py        # Natural language queries
│
├── cache/                      # Cached loaded data
└── s2orc_data/                 # 3.2M+ papers
    ├── Biology,2017-2017/
    ├── ComputerScience,2017-2017/
    └── [48 directories]
```

---

## Performance Optimizations

### DGX Spark Optimizations

1. **cuDF Backend**: Auto-detects and uses GPU DataFrames when available
   ```python
   try:
       import cudf as pd
       GPU_AVAILABLE = True
   except ImportError:
       import pandas as pd
   ```

2. **Caching**: Loaded data is cached to disk for fast restarts
   ```python
   cache_file = cache_dir / f"papers_{hash}.pkl"
   ```

3. **Precomputation**: Field and year aggregations computed once
   ```python
   self._field_stats = self._compute_field_aggregations()
   self._year_stats = self._compute_year_aggregations()
   ```

### Memory Efficiency

- Streaming JSONL loading (generator-based)
- Only extract needed features from papers
- Limit stored techniques/landmarks to top 5

---

## Evaluation Criteria Mapping

| Criteria (100 pts) | Our Implementation |
|--------------------|-------------------|
| **Technical Execution (30)** | Complete pipeline: load → analyze → visualize |
| **NVIDIA Stack (30)** | RAPIDS cuDF, cuGraph-ready, DGX Spark optimized |
| **Value & Impact (20)** | Non-obvious insights: hype risk, cross-pollination |
| **Frontier Factor (20)** | 3.2M paper scale, 14-field coverage, real-time |

### The Spark Story

> "We use 128GB unified memory to hold the entire 3.2M paper corpus in-memory,
> enabling real-time field-by-field analysis without disk I/O. RAPIDS cuDF
> provides 10x faster DataFrame operations for ML classification across
> millions of papers. cuGraph integration is ready for citation network
> analysis when Semantic Scholar API data is fetched."

---

## Future Enhancements

1. **Citation Graph Integration**: Fetch citations via Semantic Scholar API
2. **GraphRAG Queries**: Natural language questions over the data
3. **Real-time Updates**: Streaming paper ingestion
4. **Embedding Search**: Semantic similarity for related papers
5. **Prediction Models**: Forecast field adoption curves

---

## Credits

Built for **DGX Spark Frontier Hackathon** - Symby AI Track

**Technologies**: NVIDIA RAPIDS, cuGraph, Streamlit, FastAPI, Plotly
