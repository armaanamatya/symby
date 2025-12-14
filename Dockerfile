# SymbyAI - AI-Powered Research Paper Analysis
# DGX Spark Frontier Hackathon - Symby AI Track
#
# Multi-stage build for efficient Docker image

# ============================================================
# STAGE 1: Base Python environment
# ============================================================
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# ============================================================
# STAGE 2: Dependencies
# ============================================================
FROM base as dependencies

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# ============================================================
# STAGE 3: Production image
# ============================================================
FROM dependencies as production

# Create non-root user for security
RUN groupadd --gid 1000 symby && \
    useradd --uid 1000 --gid symby --shell /bin/bash --create-home symby

# Copy application code
COPY --chown=symby:symby . .

# Create necessary directories
RUN mkdir -p /app/cache /app/graph_cache /app/checkpoints && \
    chown -R symby:symby /app/cache /app/graph_cache /app/checkpoints

# Switch to non-root user
USER symby

# Expose ports for API and Streamlit
EXPOSE 8000 8501

# Default environment variables
ENV DATA_PATH=/app/s2orc_data \
    DEFAULT_PAPERS=50000 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Default command runs Streamlit
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
