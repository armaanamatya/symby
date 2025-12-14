"""
Paper Code Execution (PCE) Module

Provides code extraction, sandbox execution, Dockerfile generation, and validation
of execution results against paper claims.
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

from .dockerfile_generator import (
    CodeManifest as DockerCodeManifest,
    DockerConfig,
    DockerfileGenerator,
    KNOWN_PACKAGE_VERSIONS,
    BASE_IMAGES,
)

from .code_extractor import (
    CodeBlock,
    CodeManifest,
    CodeExtractor,
)

from .validation_agent import (
    # Enums
    ClaimType,
    ComparisonType,
    ValidationStatus,
    DiscrepancySeverity,

    # Dataclasses
    PaperClaim,
    Discrepancy,
    ValidationResult,

    # Classes
    LLMClient,
    ValidationAgent,

    # Constants
    METRIC_ALIASES,

    # Convenience functions
    validate_paper_execution,
)

__all__ = [
    # Sandbox runner
    "ExecutionResult",
    "GPUInfo",
    "SandboxRunner",
    "create_dgx_spark_runner",
    "DEFAULT_CUDA_IMAGE",
    "DGX_SPARK_CUDA_IMAGE",

    # Dockerfile generator
    "DockerCodeManifest",
    "DockerConfig",
    "DockerfileGenerator",
    "KNOWN_PACKAGE_VERSIONS",
    "BASE_IMAGES",

    # Code extractor
    "CodeBlock",
    "CodeManifest",
    "CodeExtractor",

    # Validation enums
    "ClaimType",
    "ComparisonType",
    "ValidationStatus",
    "DiscrepancySeverity",

    # Validation dataclasses
    "PaperClaim",
    "Discrepancy",
    "ValidationResult",

    # Validation classes
    "LLMClient",
    "ValidationAgent",

    # Constants
    "METRIC_ALIASES",

    # Convenience functions
    "validate_paper_execution",
]
