"""
Agents Package
"""

from .research_agents import (
    create_paper_analyst_agent,
    create_literature_researcher_agent,
    create_code_hunter_agent,
    create_code_verifier_agent,
    create_synthesis_agent,
    get_all_agents,
)

from .code_explanation_agent import (
    create_code_explanation_agent,
    create_code_executor_agent,
    create_code_tutor_agent,
    get_code_agents,
)

__all__ = [
    # Research agents
    "create_paper_analyst_agent",
    "create_literature_researcher_agent",
    "create_code_hunter_agent",
    "create_code_verifier_agent",
    "create_synthesis_agent",
    "get_all_agents",
    # Code explanation agents
    "create_code_explanation_agent",
    "create_code_executor_agent",
    "create_code_tutor_agent",
    "get_code_agents",
]
