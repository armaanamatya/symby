"""
Paper Code Execution (PCE) Module

Provides validation of execution results against paper claims.
"""

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
    # Enums
    "ClaimType",
    "ComparisonType",
    "ValidationStatus",
    "DiscrepancySeverity",

    # Dataclasses
    "PaperClaim",
    "Discrepancy",
    "ValidationResult",

    # Classes
    "LLMClient",
    "ValidationAgent",

    # Constants
    "METRIC_ALIASES",

    # Convenience functions
    "validate_paper_execution",
]
