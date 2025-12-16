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

from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime
import sys
import os
import uuid
import asyncio
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
# PAPER CODE EXECUTOR (PCE) MODELS
# =============================================================================

class JobStatus(str, Enum):
    """Status enum for PCE jobs"""
    PROCESSING = "processing"
    COMPLETE = "complete"
    ERROR = "error"
    QUEUED = "queued"
    RUNNING = "running"
    TIMEOUT = "timeout"


class Environment(str, Enum):
    """Supported execution environments"""
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    JAX = "jax"


class CodeBlock(BaseModel):
    """Individual code block extracted from paper"""
    language: str = "python"
    code: str
    description: Optional[str] = None
    line_start: Optional[int] = None
    dependencies: List[str] = Field(default_factory=list)


class CodeManifest(BaseModel):
    """Complete code manifest extracted from a paper"""
    id: str
    paper_id: str
    title: Optional[str] = None
    extracted_at: datetime
    code_blocks: List[CodeBlock] = Field(default_factory=list)
    main_algorithm: Optional[str] = None
    dependencies: List[str] = Field(default_factory=list)
    estimated_runtime_seconds: Optional[int] = None
    hardware_requirements: Optional[Dict[str, Any]] = None


class ExecutionResult(BaseModel):
    """Result of code execution"""
    success: bool
    stdout: Optional[str] = None
    stderr: Optional[str] = None
    execution_time_seconds: float
    memory_usage_mb: Optional[float] = None
    gpu_usage_percent: Optional[float] = None
    output_artifacts: List[str] = Field(default_factory=list)


class ValidationResult(BaseModel):
    """Result of validating execution against paper claims"""
    paper_id: str
    execution_job_id: str
    validated_at: datetime
    claims_verified: int
    claims_failed: int
    claims_total: int
    verification_rate: float
    details: List[Dict[str, Any]] = Field(default_factory=list)
    overall_status: str  # "verified", "partially_verified", "failed"


# PCE Request/Response Models
class ExtractRequest(BaseModel):
    """Request to extract code from a paper"""
    paper_id: str = Field(..., description="Paper ID (e.g., 'arxiv:2010.11929')")


class ExtractResponse(BaseModel):
    """Response for code extraction request"""
    job_id: str
    status: str
    code_manifest: Optional[CodeManifest] = None


class ExtractJobResponse(BaseModel):
    """Response for extraction job status"""
    status: str
    code_manifest: Optional[CodeManifest] = None
    error: Optional[str] = None


class ExecuteRequest(BaseModel):
    """Request to execute code from a manifest"""
    code_manifest_id: str
    environment: Environment = Environment.PYTORCH
    timeout_seconds: int = Field(default=300, ge=10, le=3600)


class ExecuteResponse(BaseModel):
    """Response for code execution request"""
    job_id: str
    status: str


class ExecuteJobResponse(BaseModel):
    """Response for execution job status"""
    status: str
    result: Optional[ExecutionResult] = None
    error: Optional[str] = None


class ValidateRequest(BaseModel):
    """Request to validate execution against paper claims"""
    paper_id: str
    execution_job_id: str


class ValidateResponse(BaseModel):
    """Response for validation request"""
    validation_result: ValidationResult


class PCEStatusResponse(BaseModel):
    """Response for PCE system status"""
    active_jobs: int
    completed_today: int
    gpu_utilization: float


# =============================================================================
# PCE JOB STORE (In-memory, use Redis in production)
# =============================================================================

_pce_jobs: Dict[str, Dict[str, Any]] = {}
_pce_manifests: Dict[str, CodeManifest] = {}
_pce_execution_results: Dict[str, ExecutionResult] = {}
_pce_stats = {
    "completed_today": 0,
    "last_reset": datetime.utcnow().date()
}


def _reset_daily_stats():
    """Reset daily stats if it's a new day"""
    today = datetime.utcnow().date()
    if _pce_stats["last_reset"] != today:
        _pce_stats["completed_today"] = 0
        _pce_stats["last_reset"] = today


# =============================================================================
# PCE BACKGROUND TASKS
# =============================================================================

async def run_extraction(job_id: str, paper_id: str):
    """Background task to extract code from a paper"""
    try:
        # Simulate extraction process (replace with actual implementation)
        await asyncio.sleep(2)  # Simulated processing time

        # Create a sample manifest (in production, use actual extraction logic)
        manifest = CodeManifest(
            id=str(uuid.uuid4()),
            paper_id=paper_id,
            title=f"Extracted code from {paper_id}",
            extracted_at=datetime.utcnow(),
            code_blocks=[
                CodeBlock(
                    language="python",
                    code="# Placeholder: Actual extraction would parse the paper",
                    description="Main algorithm implementation",
                    dependencies=["torch", "numpy"]
                )
            ],
            dependencies=["torch>=1.9.0", "numpy>=1.20.0"],
            estimated_runtime_seconds=60
        )

        _pce_manifests[manifest.id] = manifest
        _pce_jobs[job_id] = {
            "status": JobStatus.COMPLETE,
            "result": manifest,
            "error": None
        }

    except Exception as e:
        _pce_jobs[job_id] = {
            "status": JobStatus.ERROR,
            "result": None,
            "error": str(e)
        }


async def run_execution(job_id: str, manifest_id: str, environment: str, timeout: int):
    """Background task to execute code from a manifest"""
    try:
        _pce_jobs[job_id]["status"] = JobStatus.RUNNING

        # Simulate execution (replace with actual Docker/sandbox execution)
        await asyncio.sleep(3)  # Simulated execution time

        result = ExecutionResult(
            success=True,
            stdout="Execution completed successfully.\nModel accuracy: 0.95",
            stderr=None,
            execution_time_seconds=2.5,
            memory_usage_mb=512.0,
            gpu_usage_percent=75.0,
            output_artifacts=["model_weights.pt", "results.json"]
        )

        _pce_execution_results[job_id] = result
        _pce_jobs[job_id] = {
            "status": JobStatus.COMPLETE,
            "result": result,
            "error": None
        }

        _reset_daily_stats()
        _pce_stats["completed_today"] += 1

    except asyncio.TimeoutError:
        _pce_jobs[job_id] = {
            "status": JobStatus.TIMEOUT,
            "result": None,
            "error": f"Execution timed out after {timeout} seconds"
        }
    except Exception as e:
        _pce_jobs[job_id] = {
            "status": JobStatus.ERROR,
            "result": None,
            "error": str(e)
        }


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
            "dashboard": "GET /api/dashboard",
            "pce_extract": "POST /api/pce/extract",
            "pce_extract_status": "GET /api/pce/extract/{job_id}",
            "pce_execute": "POST /api/pce/execute",
            "pce_execute_status": "GET /api/pce/execute/{job_id}",
            "pce_validate": "POST /api/pce/validate",
            "pce_status": "GET /api/pce/status"
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
# PAPER CODE EXECUTOR (PCE) ENDPOINTS
# =============================================================================

@app.post("/api/pce/extract", response_model=ExtractResponse, tags=["Paper Code Executor"])
async def extract_code(
    request: ExtractRequest,
    background_tasks: BackgroundTasks
) -> ExtractResponse:
    """
    Extract code from a paper.

    Initiates an async extraction job that parses the paper and extracts
    code blocks, algorithms, and dependencies.

    **Supported paper_id formats:**
    - arxiv:2010.11929
    - doi:10.1234/example
    - s2:123456789

    Returns a job_id to poll for results.
    """
    job_id = str(uuid.uuid4())
    _pce_jobs[job_id] = {
        "status": JobStatus.PROCESSING,
        "result": None,
        "error": None,
        "type": "extraction",
        "paper_id": request.paper_id,
        "created_at": datetime.utcnow()
    }

    background_tasks.add_task(run_extraction, job_id, request.paper_id)

    return ExtractResponse(
        job_id=job_id,
        status=JobStatus.PROCESSING.value,
        code_manifest=None
    )


@app.get("/api/pce/extract/{job_id}", response_model=ExtractJobResponse, tags=["Paper Code Executor"])
async def get_extraction_status(job_id: str) -> ExtractJobResponse:
    """
    Get the status of a code extraction job.

    Poll this endpoint to check if extraction is complete.
    Once status is 'complete', the code_manifest will contain the extracted code.
    """
    if job_id not in _pce_jobs:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    job = _pce_jobs[job_id]

    if job.get("type") != "extraction":
        raise HTTPException(status_code=400, detail=f"Job '{job_id}' is not an extraction job")

    code_manifest = None
    if job["status"] == JobStatus.COMPLETE and job.get("result"):
        code_manifest = job["result"]

    return ExtractJobResponse(
        status=job["status"].value if isinstance(job["status"], JobStatus) else job["status"],
        code_manifest=code_manifest,
        error=job.get("error")
    )


@app.post("/api/pce/execute", response_model=ExecuteResponse, tags=["Paper Code Executor"])
async def execute_code(
    request: ExecuteRequest,
    background_tasks: BackgroundTasks
) -> ExecuteResponse:
    """
    Execute code from a code manifest.

    Runs the extracted code in a sandboxed environment with the specified
    ML framework (PyTorch, TensorFlow, or JAX).

    **Parameters:**
    - code_manifest_id: ID from a completed extraction job
    - environment: ML framework to use (pytorch, tensorflow, jax)
    - timeout_seconds: Max execution time (10-3600 seconds)

    Returns a job_id to poll for execution results.
    """
    if request.code_manifest_id not in _pce_manifests:
        raise HTTPException(
            status_code=404,
            detail=f"Code manifest '{request.code_manifest_id}' not found. "
                   "Ensure extraction is complete first."
        )

    job_id = str(uuid.uuid4())
    _pce_jobs[job_id] = {
        "status": JobStatus.QUEUED,
        "result": None,
        "error": None,
        "type": "execution",
        "manifest_id": request.code_manifest_id,
        "environment": request.environment.value,
        "timeout": request.timeout_seconds,
        "created_at": datetime.utcnow()
    }

    background_tasks.add_task(
        run_execution,
        job_id,
        request.code_manifest_id,
        request.environment.value,
        request.timeout_seconds
    )

    return ExecuteResponse(
        job_id=job_id,
        status=JobStatus.QUEUED.value
    )


@app.get("/api/pce/execute/{job_id}", response_model=ExecuteJobResponse, tags=["Paper Code Executor"])
async def get_execution_status(job_id: str) -> ExecuteJobResponse:
    """
    Get the status of a code execution job.

    Poll this endpoint to check if execution is complete.

    **Possible statuses:**
    - queued: Waiting to start
    - running: Currently executing
    - complete: Finished successfully
    - error: Failed with an error
    - timeout: Exceeded time limit
    """
    if job_id not in _pce_jobs:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    job = _pce_jobs[job_id]

    if job.get("type") != "execution":
        raise HTTPException(status_code=400, detail=f"Job '{job_id}' is not an execution job")

    result = None
    if job["status"] == JobStatus.COMPLETE and job.get("result"):
        result = job["result"]

    return ExecuteJobResponse(
        status=job["status"].value if isinstance(job["status"], JobStatus) else job["status"],
        result=result,
        error=job.get("error")
    )


@app.post("/api/pce/validate", response_model=ValidateResponse, tags=["Paper Code Executor"])
async def validate_execution(request: ValidateRequest) -> ValidateResponse:
    """
    Validate execution results against paper claims.

    Compares the execution output with claims made in the paper
    (e.g., accuracy metrics, performance benchmarks).

    **Parameters:**
    - paper_id: The paper ID used for extraction
    - execution_job_id: Job ID from a completed execution

    Returns a validation result with verification statistics.
    """
    if request.execution_job_id not in _pce_jobs:
        raise HTTPException(
            status_code=404,
            detail=f"Execution job '{request.execution_job_id}' not found"
        )

    job = _pce_jobs[request.execution_job_id]

    if job.get("type") != "execution":
        raise HTTPException(
            status_code=400,
            detail=f"Job '{request.execution_job_id}' is not an execution job"
        )

    if job["status"] != JobStatus.COMPLETE:
        raise HTTPException(
            status_code=400,
            detail=f"Execution job is not complete (status: {job['status']})"
        )

    # Perform validation (simulated - replace with actual validation logic)
    # In production, this would compare execution outputs with paper claims
    claims_total = 5
    claims_verified = 4
    claims_failed = 1

    verification_rate = claims_verified / claims_total if claims_total > 0 else 0.0

    if verification_rate >= 0.9:
        overall_status = "verified"
    elif verification_rate >= 0.5:
        overall_status = "partially_verified"
    else:
        overall_status = "failed"

    validation_result = ValidationResult(
        paper_id=request.paper_id,
        execution_job_id=request.execution_job_id,
        validated_at=datetime.utcnow(),
        claims_verified=claims_verified,
        claims_failed=claims_failed,
        claims_total=claims_total,
        verification_rate=verification_rate,
        details=[
            {"claim": "Model accuracy >= 0.95", "verified": True, "actual": 0.95},
            {"claim": "Training time < 1 hour", "verified": True, "actual": "45 minutes"},
            {"claim": "GPU memory < 8GB", "verified": True, "actual": "6.5GB"},
            {"claim": "Convergence in 100 epochs", "verified": True, "actual": 98},
            {"claim": "FLOPs < 10B", "verified": False, "actual": "12B"}
        ],
        overall_status=overall_status
    )

    return ValidateResponse(validation_result=validation_result)


@app.get("/api/pce/status", response_model=PCEStatusResponse, tags=["Paper Code Executor"])
async def get_pce_status() -> PCEStatusResponse:
    """
    Get Paper Code Executor system status.

    Returns metrics about the PCE system including:
    - active_jobs: Number of currently running jobs
    - completed_today: Jobs completed in the last 24 hours
    - gpu_utilization: Current GPU usage percentage (0.0-1.0)
    """
    _reset_daily_stats()

    # Count active jobs (processing, queued, or running)
    active_statuses = {JobStatus.PROCESSING, JobStatus.QUEUED, JobStatus.RUNNING}
    active_jobs = sum(
        1 for job in _pce_jobs.values()
        if job.get("status") in active_statuses
    )

    # Simulated GPU utilization (replace with actual monitoring)
    # In production, query nvidia-smi or use pynvml
    gpu_utilization = 0.0
    if active_jobs > 0:
        gpu_utilization = min(0.25 * active_jobs, 1.0)

    return PCEStatusResponse(
        active_jobs=active_jobs,
        completed_today=_pce_stats["completed_today"],
        gpu_utilization=gpu_utilization
    )


# =============================================================================
# PAPER ANALYSIS ENDPOINTS
# =============================================================================

# Import paper analysis module
try:
    from src.paper_analysis import PaperAnalyzer, PaperAnalysisResult
    from src.paper_analysis.executor import PaperCodeExecutor, PaperQueryEngine
    PAPER_ANALYSIS_AVAILABLE = True
except ImportError:
    PAPER_ANALYSIS_AVAILABLE = False

# Paper analysis state
_paper_state = {
    "analyzer": None,
    "executor": None,
    "query_engine": None,
    "papers": {}  # paper_id -> analysis cache
}


def get_paper_analyzer():
    """Get or initialize paper analyzer."""
    if not PAPER_ANALYSIS_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Paper analysis module not available"
        )
    if _paper_state["analyzer"] is None:
        _paper_state["analyzer"] = PaperAnalyzer(use_llm=True)
    return _paper_state["analyzer"]


def get_paper_executor():
    """Get or initialize paper executor."""
    if not PAPER_ANALYSIS_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Paper analysis module not available"
        )
    if _paper_state["executor"] is None:
        _paper_state["executor"] = PaperCodeExecutor()
    return _paper_state["executor"]


def get_query_engine():
    """Get or initialize query engine."""
    if not PAPER_ANALYSIS_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Paper analysis module not available"
        )
    if _paper_state["query_engine"] is None:
        _paper_state["query_engine"] = PaperQueryEngine()
    return _paper_state["query_engine"]


class PaperAnalyzeRequest(BaseModel):
    """Request to analyze a paper."""
    text: Optional[str] = Field(None, description="Paper text content")
    paper_id: Optional[str] = Field(None, description="Paper identifier")
    title: Optional[str] = Field(None, description="Paper title")
    abstract: Optional[str] = Field(None, description="Paper abstract")
    publication_date: Optional[str] = Field(None, description="Publication date")


class PaperQueryRequest(BaseModel):
    """Request to query a paper."""
    paper_id: str = Field(..., description="Paper identifier")
    question: str = Field(..., description="Question to answer")


class CodeExecuteRequest(BaseModel):
    """Request to execute code."""
    code: str = Field(..., description="Code to execute")
    language: str = Field(default="python", description="Programming language")
    paper_id: Optional[str] = Field(None, description="Associated paper ID")


class PaperCodeExecuteRequest(BaseModel):
    """Request to execute code from a paper."""
    paper_id: str = Field(..., description="Paper identifier")
    code_block_index: int = Field(default=0, description="Index of code block to execute")


@app.post("/api/paper/analyze", tags=["Paper Analysis"])
async def analyze_paper(request: PaperAnalyzeRequest):
    """
    Analyze a paper and return structured JSON output.

    Returns comprehensive analysis including:
    - ML adoption metrics
    - Reproducibility assessment
    - Research outcomes
    - Impact indicators
    - Extracted code blocks
    - GitHub repositories

    The output format matches the s2orc_data JSON schema.
    """
    analyzer = get_paper_analyzer()

    if not request.text:
        raise HTTPException(
            status_code=400,
            detail="Paper text is required"
        )

    try:
        result = analyzer.analyze(
            text=request.text,
            paper_id=request.paper_id,
            title=request.title,
            abstract=request.abstract,
            publication_date=request.publication_date
        )

        # Cache for future queries
        if result.paper_id:
            _paper_state["papers"][result.paper_id] = {
                "analysis": result,
                "text": request.text
            }

        return result.to_dict()

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/paper/analyze/upload", tags=["Paper Analysis"])
async def analyze_paper_upload(
    file: bytes = None,
    paper_id: Optional[str] = Query(None),
    title: Optional[str] = Query(None)
):
    """
    Analyze an uploaded PDF file.

    Upload a PDF file to get structured JSON analysis.
    """
    analyzer = get_paper_analyzer()

    if not file:
        raise HTTPException(
            status_code=400,
            detail="PDF file is required"
        )

    try:
        result = analyzer.analyze(
            pdf_bytes=file,
            paper_id=paper_id,
            title=title
        )

        # Cache for future queries
        if result.paper_id:
            text = analyzer.extract_text_from_pdf_bytes(file)
            _paper_state["papers"][result.paper_id] = {
                "analysis": result,
                "text": text
            }

        return result.to_dict()

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/paper/query", tags=["Paper Analysis"])
async def query_paper(request: PaperQueryRequest):
    """
    Answer questions about an analyzed paper.

    The paper must be analyzed first using /api/paper/analyze.
    Uses RAG-style approach to answer questions based on paper content.
    """
    if request.paper_id not in _paper_state["papers"]:
        raise HTTPException(
            status_code=404,
            detail=f"Paper '{request.paper_id}' not found. Analyze it first."
        )

    analyzer = get_paper_analyzer()
    cache = _paper_state["papers"][request.paper_id]
    text = cache.get("text", "")

    if not text:
        raise HTTPException(
            status_code=400,
            detail="Paper text not available for querying"
        )

    try:
        result = analyzer.query_paper(
            text=text,
            question=request.question,
            paper_id=request.paper_id
        )
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/paper/{paper_id}", tags=["Paper Analysis"])
async def get_paper_analysis(paper_id: str):
    """
    Get cached analysis for a paper.

    Returns the stored analysis result for a previously analyzed paper.
    """
    if paper_id not in _paper_state["papers"]:
        raise HTTPException(
            status_code=404,
            detail=f"Paper '{paper_id}' not found"
        )

    cache = _paper_state["papers"][paper_id]
    analysis = cache.get("analysis")

    if analysis:
        return analysis.to_dict()
    else:
        raise HTTPException(
            status_code=404,
            detail=f"Analysis not found for paper '{paper_id}'"
        )


@app.get("/api/paper/{paper_id}/code", tags=["Paper Analysis"])
async def get_paper_code(paper_id: str):
    """
    Get extracted code blocks from a paper.

    Returns all code blocks extracted during analysis.
    """
    if paper_id not in _paper_state["papers"]:
        raise HTTPException(
            status_code=404,
            detail=f"Paper '{paper_id}' not found"
        )

    cache = _paper_state["papers"][paper_id]
    analysis = cache.get("analysis")

    if analysis:
        return {
            "paper_id": paper_id,
            "code_blocks": [cb.to_dict() for cb in analysis.code_blocks],
            "github_repos": analysis.github_repos
        }
    else:
        raise HTTPException(
            status_code=404,
            detail=f"Analysis not found for paper '{paper_id}'"
        )


@app.post("/api/paper/execute", tags=["Paper Analysis"])
async def execute_code(request: CodeExecuteRequest):
    """
    Execute code in a sandbox environment.

    Runs the provided code in an isolated Docker container.
    Returns execution results including stdout, stderr, and metrics.
    """
    executor = get_paper_executor()

    try:
        result = await executor.execute_code(
            code=request.code,
            language=request.language
        )

        if request.paper_id:
            result["paper_id"] = request.paper_id

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/paper/{paper_id}/execute", tags=["Paper Analysis"])
async def execute_paper_code(paper_id: str, request: PaperCodeExecuteRequest):
    """
    Execute a code block from an analyzed paper.

    Runs the specified code block in an isolated sandbox.
    The paper must be analyzed first.
    """
    if paper_id not in _paper_state["papers"]:
        raise HTTPException(
            status_code=404,
            detail=f"Paper '{paper_id}' not found. Analyze it first."
        )

    cache = _paper_state["papers"][paper_id]
    analysis = cache.get("analysis")

    if not analysis or not analysis.code_blocks:
        raise HTTPException(
            status_code=400,
            detail="No code blocks found in paper"
        )

    executor = get_paper_executor()

    try:
        result = await executor.execute_paper_code(
            analysis_result=analysis,
            code_block_index=request.code_block_index
        )
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/papers", tags=["Paper Analysis"])
async def list_papers():
    """
    List all analyzed papers.

    Returns a list of paper IDs that have been analyzed.
    """
    papers = []
    for paper_id, cache in _paper_state["papers"].items():
        analysis = cache.get("analysis")
        if analysis:
            papers.append({
                "paper_id": paper_id,
                "field": analysis.field,
                "ml_adoption_level": analysis.ml_adoption.ml_adoption_level,
                "code_blocks_count": len(analysis.code_blocks),
                "analyzed_at": analysis.analyzed_at
            })

    return {
        "papers": papers,
        "count": len(papers)
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
        "papers_count": _data_state["papers_count"],
        "paper_analysis_available": PAPER_ANALYSIS_AVAILABLE,
        "analyzed_papers_count": len(_paper_state["papers"])
    }


# =============================================================================
# RUN SERVER
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
