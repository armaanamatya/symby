"""
Tasks Package
"""

from .research_tasks import (
    create_paper_analysis_task,
    create_literature_search_task,
    create_code_search_task,
    create_code_verification_task,
    create_synthesis_task,
    create_quick_research_task,
)

from .code_explanation_tasks import (
    create_code_explanation_task,
    create_code_execution_task,
    create_step_by_step_execution_task,
    create_code_tutoring_task,
    create_code_debugging_task,
    create_code_comparison_task,
    create_paper_code_explanation_task,
)

__all__ = [
    # Research tasks
    "create_paper_analysis_task",
    "create_literature_search_task",
    "create_code_search_task",
    "create_code_verification_task",
    "create_synthesis_task",
    "create_quick_research_task",
    # Code explanation tasks
    "create_code_explanation_task",
    "create_code_execution_task",
    "create_step_by_step_execution_task",
    "create_code_tutoring_task",
    "create_code_debugging_task",
    "create_code_comparison_task",
    "create_paper_code_explanation_task",
]
