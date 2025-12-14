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

__all__ = [
    "create_paper_analysis_task",
    "create_literature_search_task",
    "create_code_search_task",
    "create_code_verification_task",
    "create_synthesis_task",
    "create_quick_research_task",
]
