"""
FastAPI Backend for AI Research Impact Observatory
DGX Spark Frontier Hackathon - Symby AI Track

REST API for Next.js integration - serves all 4 objectives:
1. ML Impact Quantification
2. Adoption Dynamics
3. Quality Trade-offs
4. Discovery Impact Tracing

Run: uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Optional, List
from pydantic import BaseModel
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data.loader import DataLoader
from metrics.impact_metrics import UnifiedMetricsEngine

# =============================================================================
# APP CONFIGURATION
# =============================================================================

app = FastAPI(
    title="AI Research Impact Observatory API",
    description="""
    REST API for quantifying ML's real impact on scientific progress.

    **Build Objectives:**
    1. **ML Impact** - Attribution scoring, acceleration metrics
    2. **Adoption Dynamics** - S-curves, temporal evolution
    3. **Quality Trade-offs** - Reproducibility analysis
    4. **Discovery Impact** - ML → Domain paths

    **Optimized for DGX Spark** with RAPIDS/cuGraph support.
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS for Next.js
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# DATA MANAGEMENT
# =============================================================================

# Global state for data (loaded once)
_data_state = {
    "df": None,
    "engine": None,
    "dashboard_data": None,
    "loaded": False,
    "papers_count": 0
}

DATA_PATH = os.environ.get("DATA_PATH", "s2orc_data")
DEFAULT_PAPERS = int(os.environ.get("DEFAULT_PAPERS", "50000"))


def get_engine():
    """Get or initialize the metrics engine"""
    if not _data_state["loaded"]:
        raise HTTPException(
            status_code=503,
            detail="Data not loaded. Call POST /api/load first."
        )
    return _data_state["engine"]


def get_dashboard_data():
    """Get cached dashboard data"""
    if not _data_state["loaded"]:
        raise HTTPException(
            status_code=503,
            detail="Data not loaded. Call POST /api/load first."
        )
    return _data_state["dashboard_data"]


# =============================================================================
# RESPONSE MODELS
# =============================================================================

class LoadResponse(BaseModel):
    status: str
    papers_loaded: int
    fields: List[str]
    year_range: List[int]


class FieldInfo(BaseModel):
    field: str
    total_papers: int
    ml_papers: int
    ml_rate: float
    code_rate: float


class SummaryResponse(BaseModel):
    total_papers: int
    ml_papers: int
    overall_ml_rate: float
    overall_code_rate: float
    fields_analyzed: int
    year_range: List[int]


# =============================================================================
# API ROUTES
# =============================================================================

@app.get("/")
async def root():
    """API health check and info"""
    return {
        "name": "AI Research Impact Observatory API",
        "version": "1.0.0",
        "status": "running",
        "data_loaded": _data_state["loaded"],
        "papers_loaded": _data_state["papers_count"],
        "endpoints": {
            "docs": "/docs",
            "load_data": "POST /api/load",
            "summary": "GET /api/summary",
            "fields": "GET /api/fields",
            "ml_impact": "GET /api/objectives/ml-impact",
            "adoption": "GET /api/objectives/adoption",
            "quality": "GET /api/objectives/quality",
            "discovery": "GET /api/objectives/discovery",
            "field_detail": "GET /api/field/{field_name}",
            "dashboard": "GET /api/dashboard"
        }
    }


@app.post("/api/load", response_model=LoadResponse)
async def load_data(max_papers: int = Query(default=DEFAULT_PAPERS, ge=1000, le=1000000)):
    """
    Load paper data and initialize metrics engine.

    Call this first before using other endpoints.
    """
    try:
        loader = DataLoader(DATA_PATH)
        df = loader.load_all_papers(max_papers=max_papers)

        engine = UnifiedMetricsEngine(df)
        dashboard_data = engine.get_dashboard_data()

        _data_state["df"] = df
        _data_state["engine"] = engine
        _data_state["dashboard_data"] = dashboard_data
        _data_state["loaded"] = True
        _data_state["papers_count"] = len(df)

        year_range = dashboard_data["summary"].get("year_range", (None, None))

        return LoadResponse(
            status="success",
            papers_loaded=len(df),
            fields=engine.fields,
            year_range=list(year_range) if year_range[0] else []
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/summary", response_model=SummaryResponse)
async def get_summary():
    """Get overall dataset summary"""
    data = get_dashboard_data()
    summary = data["summary"]

    return SummaryResponse(
        total_papers=summary["total_papers"],
        ml_papers=summary["ml_papers"],
        overall_ml_rate=summary["overall_ml_rate"],
        overall_code_rate=summary["overall_code_rate"],
        fields_analyzed=summary["fields_analyzed"],
        year_range=list(summary.get("year_range", (0, 0)))
    )


@app.get("/api/fields")
async def get_fields():
    """Get list of all available scientific fields with basic stats"""
    data = get_dashboard_data()
    impact = {m["field"]: m for m in data["ml_impact"]}
    quality = {m["field"]: m for m in data["quality_tradeoffs"]}

    fields = []
    for field in get_engine().fields:
        imp = impact.get(field, {})
        qual = quality.get(field, {})

        fields.append({
            "field": field,
            "total_papers": imp.get("total_papers", 0),
            "ml_papers": imp.get("ml_papers", 0),
            "ml_rate": imp.get("ml_attribution_score", 0),
            "code_rate": qual.get("code_availability_rate", 0),
            "hype_risk": qual.get("hype_risk_score", 0)
        })

    return {"fields": fields, "count": len(fields)}


# =============================================================================
# OBJECTIVE 1: ML IMPACT QUANTIFICATION
# =============================================================================

@app.get("/api/objectives/ml-impact")
async def get_ml_impact(field: Optional[str] = None):
    """
    Objective 1: ML Impact Quantification

    Metrics:
    - ml_attribution_score: How much ML contributes (0-1)
    - acceleration_factor: Research output growth rate
    - efficiency_score: Impact per ML unit
    - technique_diversity: Number of ML techniques used
    - cross_pollination_score: Methods from other fields
    """
    data = get_dashboard_data()
    impact = data["ml_impact"]

    if field:
        impact = [m for m in impact if m["field"].lower() == field.lower()]
        if not impact:
            raise HTTPException(status_code=404, detail=f"Field '{field}' not found")

    return {
        "objective": "ML Impact Quantification",
        "description": "Measuring how much ML contributes to scientific breakthroughs",
        "metrics": impact
    }


# =============================================================================
# OBJECTIVE 2: ADOPTION DYNAMICS
# =============================================================================

@app.get("/api/objectives/adoption")
async def get_adoption_dynamics(field: Optional[str] = None):
    """
    Objective 2: Adoption Dynamics Visualization

    Metrics:
    - adoption_rate: Current ML adoption rate
    - adoption_velocity: Year-over-year change
    - s_curve_phase: early/growth/mature/declining
    - first_adoption_year: When ML first appeared
    - current_trend: accelerating/growing/stable/declining
    """
    data = get_dashboard_data()
    adoption = data["adoption_dynamics"]

    if field:
        adoption = [m for m in adoption if m["field"].lower() == field.lower()]
        if not adoption:
            raise HTTPException(status_code=404, detail=f"Field '{field}' not found")

    return {
        "objective": "Adoption Dynamics",
        "description": "How ML techniques spread across disciplines over time",
        "metrics": adoption
    }


@app.get("/api/objectives/adoption/temporal")
async def get_temporal_evolution():
    """
    Get temporal evolution data for animated visualization.

    Returns year-by-year ML adoption rates for each field.
    """
    data = get_dashboard_data()
    return {
        "objective": "Temporal Evolution",
        "description": "Year-by-year adoption rates for animation",
        "data": data["temporal_evolution"]
    }


# =============================================================================
# OBJECTIVE 3: QUALITY TRADE-OFFS
# =============================================================================

@app.get("/api/objectives/quality")
async def get_quality_tradeoffs(field: Optional[str] = None):
    """
    Objective 3: Quality Trade-offs Analysis

    Metrics:
    - code_availability_rate: % of papers with code
    - ml_code_rate vs nonml_code_rate: Comparison
    - hype_risk_score: High adoption + low reproducibility
    - quality_delta: ML vs non-ML reproducibility difference
    - quality_trend: improving/stable/declining
    """
    data = get_dashboard_data()
    quality = data["quality_tradeoffs"]

    if field:
        quality = [m for m in quality if m["field"].lower() == field.lower()]
        if not quality:
            raise HTTPException(status_code=404, detail=f"Field '{field}' not found")

    return {
        "objective": "Quality Trade-offs",
        "description": "ML adoption vs reproducibility correlation",
        "metrics": quality
    }


@app.get("/api/objectives/quality/comparison")
async def get_ml_vs_nonml():
    """
    Direct comparison of ML vs non-ML papers.
    """
    data = get_dashboard_data()
    return {
        "objective": "ML vs Non-ML Comparison",
        "data": data["ml_vs_nonml"]
    }


# =============================================================================
# OBJECTIVE 4: DISCOVERY IMPACT
# =============================================================================

@app.get("/api/objectives/discovery")
async def get_discovery_impact(method: Optional[str] = None, field: Optional[str] = None):
    """
    Objective 4: Discovery Impact Tracing

    Tracks ML method → domain application paths:
    - ml_technique: The landmark method
    - target_field: Where it was adopted
    - adoption_lag_years: Time from origin to adoption
    - impact_trajectory: growing/stable/declining
    """
    data = get_dashboard_data()
    paths = data["discovery_paths"]

    if method:
        paths = [p for p in paths if p["ml_technique"].lower() == method.lower()]
    if field:
        paths = [p for p in paths if p["target_field"].lower() == field.lower()]

    return {
        "objective": "Discovery Impact",
        "description": "ML method → domain application → real-world impact paths",
        "paths": paths
    }


@app.get("/api/objectives/discovery/landmarks")
async def get_landmark_methods():
    """
    Impact analysis of landmark ML methods.
    """
    data = get_dashboard_data()
    return {
        "objective": "Landmark ML Methods",
        "description": "Transformational ML methods and their cross-field impact",
        "landmarks": data["landmark_impact"]
    }


# =============================================================================
# FIELD-SPECIFIC ENDPOINT
# =============================================================================

@app.get("/api/field/{field_name}")
async def get_field_detail(field_name: str):
    """
    Get comprehensive metrics for a specific field.

    Returns all 4 objectives for the given field.
    """
    data = get_dashboard_data()

    # Find field (case-insensitive)
    field_match = None
    for f in get_engine().fields:
        if f.lower() == field_name.lower() or f.replace(" ", "").lower() == field_name.lower():
            field_match = f
            break

    if not field_match:
        raise HTTPException(status_code=404, detail=f"Field '{field_name}' not found")

    # Gather all metrics for this field
    impact = next((m for m in data["ml_impact"] if m["field"] == field_match), None)
    adoption = next((m for m in data["adoption_dynamics"] if m["field"] == field_match), None)
    quality = next((m for m in data["quality_tradeoffs"] if m["field"] == field_match), None)
    paths = [p for p in data["discovery_paths"] if p["target_field"] == field_match]

    # Temporal data for this field
    temporal = [t for t in data["temporal_evolution"] if t["field"] == field_match]

    return {
        "field": field_match,
        "ml_impact": impact,
        "adoption_dynamics": adoption,
        "quality_tradeoffs": quality,
        "discovery_paths": paths[:10],  # Top 10
        "temporal_evolution": temporal
    }


# =============================================================================
# DASHBOARD DATA (ALL AT ONCE)
# =============================================================================

@app.get("/api/dashboard")
async def get_full_dashboard():
    """
    Get complete dashboard data for all 4 objectives.

    This returns everything needed to render the full dashboard.
    Use for initial page load, then use specific endpoints for updates.
    """
    return get_dashboard_data()


@app.get("/api/insights")
async def get_insights():
    """Get key actionable insights"""
    data = get_dashboard_data()
    return {
        "insights": data["insights"]
    }


@app.get("/api/classification")
async def get_field_classification():
    """
    Get field classification by ML impact category.

    Categories:
    - transformational: High ML impact + high quality
    - emerging: Growing ML adoption
    - hype_risk: High adoption + low reproducibility
    - traditional: Low ML presence
    """
    data = get_dashboard_data()
    return {
        "classification": data["classification"]
    }


# =============================================================================
# HEALTH & METRICS
# =============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "data_loaded": _data_state["loaded"],
        "papers_count": _data_state["papers_count"]
    }


# =============================================================================
# RUN SERVER
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
