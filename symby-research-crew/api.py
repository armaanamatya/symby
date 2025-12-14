"""
Symby Research Crew - REST API Server
Provides HTTP API endpoints for paper research operations.

Run with: uvicorn api:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List
from pathlib import Path
import tempfile
import uuid
import os
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor

from crew import create_research_crew

app = FastAPI(
    title="Symby Research Crew API",
    description="Autonomous paper research, code discovery, and verification",
    version="1.0.0",
)

# Thread pool for running synchronous crew operations
executor = ThreadPoolExecutor(max_workers=4)

# In-memory job storage (use Redis/DB in production)
jobs = {}


# =============================================================================
# Request/Response Models
# =============================================================================

class QuickResearchRequest(BaseModel):
    topic: str = Field(..., description="Research topic")
    model: str = Field(default="gpt-4o", description="LLM model to use")
    provider: str = Field(default="openai", description="LLM provider")


class CodeSearchRequest(BaseModel):
    paper_title: str = Field(..., description="Title of the paper")
    keywords: List[str] = Field(default=[], description="Additional search keywords")
    model: str = Field(default="gpt-4o")
    provider: str = Field(default="openai")


class VerifyRequest(BaseModel):
    repo_url: str = Field(..., description="GitHub repository URL")
    claims: str = Field(..., description="Paper claims to verify against")
    model: str = Field(default="gpt-4o")
    provider: str = Field(default="openai")


class ResearchJob(BaseModel):
    job_id: str
    status: str  # "pending", "running", "completed", "failed"
    created_at: str
    completed_at: Optional[str] = None
    result: Optional[str] = None
    error: Optional[str] = None


class ResearchResponse(BaseModel):
    success: bool
    result: Optional[str] = None
    error: Optional[str] = None


# =============================================================================
# Helper Functions
# =============================================================================

def run_research_sync(paper_path: str, job_id: str, model: str, provider: str):
    """Run research in background thread."""
    try:
        jobs[job_id]["status"] = "running"
        
        crew = create_research_crew(model=model, provider=provider, verbose=False)
        results = crew.research_paper(paper_path=paper_path)
        
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["result"] = results["report"]
        jobs[job_id]["completed_at"] = datetime.now().isoformat()
        
    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
        jobs[job_id]["completed_at"] = datetime.now().isoformat()


# =============================================================================
# API Endpoints
# =============================================================================

@app.get("/")
async def root():
    """API root - health check."""
    return {
        "service": "Symby Research Crew",
        "status": "healthy",
        "version": "1.0.0",
    }


@app.post("/research/paper", response_model=ResearchJob)
async def research_paper(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    model: str = "gpt-4o",
    provider: str = "openai",
):
    """
    Upload a paper PDF and start async research.
    Returns a job ID to track progress.
    """
    # Validate file
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")
    
    # Save uploaded file
    job_id = str(uuid.uuid4())
    temp_dir = Path(tempfile.gettempdir()) / "symby_research"
    temp_dir.mkdir(exist_ok=True)
    
    paper_path = temp_dir / f"{job_id}.pdf"
    
    with open(paper_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    # Create job record
    jobs[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "created_at": datetime.now().isoformat(),
        "paper_path": str(paper_path),
    }
    
    # Start background research
    loop = asyncio.get_event_loop()
    loop.run_in_executor(
        executor,
        run_research_sync,
        str(paper_path),
        job_id,
        model,
        provider,
    )
    
    return ResearchJob(
        job_id=job_id,
        status="pending",
        created_at=jobs[job_id]["created_at"],
    )


@app.get("/research/status/{job_id}", response_model=ResearchJob)
async def get_job_status(job_id: str):
    """Get the status of a research job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    return ResearchJob(
        job_id=job["job_id"],
        status=job["status"],
        created_at=job["created_at"],
        completed_at=job.get("completed_at"),
        result=job.get("result"),
        error=job.get("error"),
    )


@app.post("/research/quick", response_model=ResearchResponse)
async def quick_research(request: QuickResearchRequest):
    """
    Run a quick research sweep on a topic.
    Synchronous - returns result directly.
    """
    try:
        crew = create_research_crew(
            model=request.model,
            provider=request.provider,
            verbose=False,
        )
        
        # Run in thread pool
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            executor,
            crew.quick_research,
            request.topic,
        )
        
        return ResearchResponse(success=True, result=result)
        
    except Exception as e:
        return ResearchResponse(success=False, error=str(e))


@app.post("/code/search", response_model=ResearchResponse)
async def search_code(request: CodeSearchRequest):
    """Search for code implementations of a paper."""
    try:
        crew = create_research_crew(
            model=request.model,
            provider=request.provider,
            verbose=False,
        )
        
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            executor,
            lambda: crew.find_code(request.paper_title, request.keywords or None),
        )
        
        return ResearchResponse(success=True, result=result)
        
    except Exception as e:
        return ResearchResponse(success=False, error=str(e))


@app.post("/code/verify", response_model=ResearchResponse)
async def verify_code(request: VerifyRequest):
    """Verify a code repository against paper claims."""
    try:
        crew = create_research_crew(
            model=request.model,
            provider=request.provider,
            verbose=False,
        )
        
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            executor,
            crew.verify_repository,
            request.repo_url,
            request.claims,
        )
        
        return ResearchResponse(success=True, result=result)
        
    except Exception as e:
        return ResearchResponse(success=False, error=str(e))


@app.get("/jobs")
async def list_jobs():
    """List all research jobs."""
    return {
        "jobs": [
            {
                "job_id": j["job_id"],
                "status": j["status"],
                "created_at": j["created_at"],
            }
            for j in jobs.values()
        ]
    }


@app.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    """Delete a job and its associated files."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    # Delete paper file if exists
    paper_path = job.get("paper_path")
    if paper_path and os.path.exists(paper_path):
        os.remove(paper_path)
    
    del jobs[job_id]
    
    return {"deleted": job_id}


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
