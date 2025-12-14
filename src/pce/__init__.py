"""
Paper Code Execution (PCE) module for running code in isolated containers.

Optimized for NVIDIA DGX Spark with GB10 GPU (CUDA 13.0, Python 3.10).
"""

from .sandbox_runner import (
    DEFAULT_CUDA_IMAGE,
    DGX_SPARK_CUDA_IMAGE,
    ExecutionResult,
    GPUInfo,
    SandboxRunner,
    create_dgx_spark_runner,
)

__all__ = [
    "ExecutionResult",
    "GPUInfo",
    "SandboxRunner",
    "create_dgx_spark_runner",
    "DEFAULT_CUDA_IMAGE",
    "DGX_SPARK_CUDA_IMAGE",
]
