"""
Code Explanation Crew - Main Orchestration
Autonomous agent system for running code in containers and explaining each step.
"""

from crewai import Crew, Process
from typing import Optional, Dict, Any, List
from pathlib import Path
import logging
import json
from datetime import datetime

from agents import (
    create_code_explanation_agent,
    create_code_executor_agent,
    create_code_tutor_agent,
)

from tasks import (
    create_code_explanation_task,
    create_code_execution_task,
    create_step_by_step_execution_task,
    create_code_tutoring_task,
    create_code_debugging_task,
    create_code_comparison_task,
    create_paper_code_explanation_task,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CodeExplanationCrew:
    """
    Autonomous crew for executing and explaining code.

    Features:
    1. Execute code in isolated Docker containers
    2. Break down code into understandable steps
    3. Explain what each part does
    4. Show execution results with explanations
    5. Debug code and identify issues
    6. Compare different implementations
    """

    def __init__(
        self,
        llm=None,
        verbose: bool = True,
        memory: bool = True,
        max_rpm: int = 10,
    ):
        """
        Initialize the code explanation crew.

        Args:
            llm: LLM instance to use (defaults to configured LLM)
            verbose: Enable verbose logging
            memory: Enable crew memory
            max_rpm: Maximum requests per minute
        """
        self.llm = llm
        self.verbose = verbose
        self.memory = memory
        self.max_rpm = max_rpm

        # Initialize agents
        self.code_explainer = create_code_explanation_agent(llm)
        self.code_executor = create_code_executor_agent(llm)
        self.code_tutor = create_code_tutor_agent(llm)

        logger.info("Code Explanation Crew initialized with 3 agents")

    def explain_code(
        self,
        code: str,
        detail_level: str = "detailed",
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Explain code step by step with execution.

        Args:
            code: Python code to explain
            detail_level: Level of detail ('brief', 'detailed', 'comprehensive')
            output_path: Optional path to save the explanation

        Returns:
            Dictionary containing explanation results
        """
        logger.info(f"Explaining code ({len(code)} chars, detail={detail_level})")

        # Create the explanation task
        task = create_code_explanation_task(
            agent=self.code_explainer,
            code=code,
            detail_level=detail_level,
        )

        # Create and run the crew
        crew = Crew(
            agents=[self.code_explainer],
            tasks=[task],
            process=Process.sequential,
            verbose=self.verbose,
            memory=self.memory,
            max_rpm=self.max_rpm,
        )

        # Execute
        logger.info("Executing code explanation crew...")
        result = crew.kickoff()

        # Package results
        explanation_result = {
            "code": code,
            "detail_level": detail_level,
            "timestamp": datetime.now().isoformat(),
            "explanation": str(result),
        }

        # Save if output path provided
        if output_path:
            self._save_result(output_path, explanation_result, str(result))

        return explanation_result

    def execute_code(
        self,
        code: str,
        timeout_seconds: int = 60,
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute code in a sandbox container.

        Args:
            code: Python code to execute
            timeout_seconds: Maximum execution time
            output_path: Optional path to save results

        Returns:
            Dictionary containing execution results
        """
        logger.info(f"Executing code in sandbox ({len(code)} chars)")

        # Create the execution task
        task = create_code_execution_task(
            agent=self.code_executor,
            code=code,
            timeout_seconds=timeout_seconds,
        )

        # Create and run the crew
        crew = Crew(
            agents=[self.code_executor],
            tasks=[task],
            process=Process.sequential,
            verbose=self.verbose,
            memory=self.memory,
            max_rpm=self.max_rpm,
        )

        # Execute
        logger.info("Executing code...")
        result = crew.kickoff()

        # Package results
        execution_result = {
            "code": code,
            "timeout_seconds": timeout_seconds,
            "timestamp": datetime.now().isoformat(),
            "result": str(result),
        }

        # Save if output path provided
        if output_path:
            self._save_result(output_path, execution_result, str(result))

        return execution_result

    def step_by_step(
        self,
        code: str,
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute code step by step with explanations.

        Args:
            code: Python code to execute step by step
            output_path: Optional path to save results

        Returns:
            Dictionary containing step-by-step results
        """
        logger.info(f"Step-by-step execution ({len(code)} chars)")

        # Create the step-by-step task
        task = create_step_by_step_execution_task(
            agent=self.code_explainer,
            code=code,
        )

        # Create and run the crew
        crew = Crew(
            agents=[self.code_explainer],
            tasks=[task],
            process=Process.sequential,
            verbose=self.verbose,
            memory=self.memory,
            max_rpm=self.max_rpm,
        )

        # Execute
        logger.info("Running step-by-step execution...")
        result = crew.kickoff()

        # Package results
        step_result = {
            "code": code,
            "timestamp": datetime.now().isoformat(),
            "steps": str(result),
        }

        # Save if output path provided
        if output_path:
            self._save_result(output_path, step_result, str(result))

        return step_result

    def tutor_code(
        self,
        code: str,
        learner_level: str = "intermediate",
        focus_topics: Optional[List[str]] = None,
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Teach code concepts with execution examples.

        Args:
            code: Code to teach
            learner_level: Skill level ('beginner', 'intermediate', 'advanced')
            focus_topics: Specific topics to emphasize
            output_path: Optional path to save results

        Returns:
            Dictionary containing tutorial content
        """
        logger.info(f"Tutoring code for {learner_level} level")

        # Create the tutoring task
        task = create_code_tutoring_task(
            agent=self.code_tutor,
            code=code,
            learner_level=learner_level,
            focus_topics=focus_topics,
        )

        # Create and run the crew
        crew = Crew(
            agents=[self.code_tutor],
            tasks=[task],
            process=Process.sequential,
            verbose=self.verbose,
            memory=self.memory,
            max_rpm=self.max_rpm,
        )

        # Execute
        logger.info("Running tutoring session...")
        result = crew.kickoff()

        # Package results
        tutorial_result = {
            "code": code,
            "learner_level": learner_level,
            "focus_topics": focus_topics or [],
            "timestamp": datetime.now().isoformat(),
            "tutorial": str(result),
        }

        # Save if output path provided
        if output_path:
            self._save_result(output_path, tutorial_result, str(result))

        return tutorial_result

    def debug_code(
        self,
        code: str,
        error_description: str = "",
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Debug code and explain issues.

        Args:
            code: Code to debug
            error_description: Description of the error or unexpected behavior
            output_path: Optional path to save results

        Returns:
            Dictionary containing debugging results
        """
        logger.info(f"Debugging code ({len(code)} chars)")

        # Create the debugging task
        task = create_code_debugging_task(
            agent=self.code_explainer,
            code=code,
            error_description=error_description,
        )

        # Create and run the crew
        crew = Crew(
            agents=[self.code_explainer],
            tasks=[task],
            process=Process.sequential,
            verbose=self.verbose,
            memory=self.memory,
            max_rpm=self.max_rpm,
        )

        # Execute
        logger.info("Running debugging session...")
        result = crew.kickoff()

        # Package results
        debug_result = {
            "code": code,
            "error_description": error_description,
            "timestamp": datetime.now().isoformat(),
            "analysis": str(result),
        }

        # Save if output path provided
        if output_path:
            self._save_result(output_path, debug_result, str(result))

        return debug_result

    def compare_code(
        self,
        code_v1: str,
        code_v2: str,
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Compare two code implementations.

        Args:
            code_v1: First code version
            code_v2: Second code version
            output_path: Optional path to save results

        Returns:
            Dictionary containing comparison results
        """
        logger.info("Comparing two code implementations")

        # Create the comparison task
        task = create_code_comparison_task(
            agent=self.code_explainer,
            code_v1=code_v1,
            code_v2=code_v2,
        )

        # Create and run the crew
        crew = Crew(
            agents=[self.code_explainer],
            tasks=[task],
            process=Process.sequential,
            verbose=self.verbose,
            memory=self.memory,
            max_rpm=self.max_rpm,
        )

        # Execute
        logger.info("Running code comparison...")
        result = crew.kickoff()

        # Package results
        comparison_result = {
            "code_v1": code_v1,
            "code_v2": code_v2,
            "timestamp": datetime.now().isoformat(),
            "comparison": str(result),
        }

        # Save if output path provided
        if output_path:
            self._save_result(output_path, comparison_result, str(result))

        return comparison_result

    def explain_paper_code(
        self,
        code: str,
        paper_context: str,
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Explain code from a research paper.

        Args:
            code: Code extracted from a paper
            paper_context: Description of the paper and its methods
            output_path: Optional path to save results

        Returns:
            Dictionary containing paper code explanation
        """
        logger.info("Explaining paper code with context")

        # Create the paper code explanation task
        task = create_paper_code_explanation_task(
            agent=self.code_explainer,
            code=code,
            paper_context=paper_context,
        )

        # Create and run the crew
        crew = Crew(
            agents=[self.code_explainer],
            tasks=[task],
            process=Process.sequential,
            verbose=self.verbose,
            memory=self.memory,
            max_rpm=self.max_rpm,
        )

        # Execute
        logger.info("Running paper code explanation...")
        result = crew.kickoff()

        # Package results
        paper_result = {
            "code": code,
            "paper_context": paper_context,
            "timestamp": datetime.now().isoformat(),
            "explanation": str(result),
        }

        # Save if output path provided
        if output_path:
            self._save_result(output_path, paper_result, str(result))

        return paper_result

    def _save_result(
        self,
        output_path: str,
        metadata: Dict[str, Any],
        report: str,
    ) -> None:
        """Save results to files."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Save markdown report
        with open(output_file, "w") as f:
            f.write(report)

        # Save JSON metadata
        json_path = output_file.with_suffix(".json")
        with open(json_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Results saved to: {output_path}")


def create_code_explanation_crew(
    model: str = "gpt-4o",
    provider: str = "openai",
    api_key: Optional[str] = None,
    **kwargs,
) -> CodeExplanationCrew:
    """
    Factory function to create a code explanation crew with specific LLM configuration.

    Args:
        model: Model name (e.g., "gpt-4o", "claude-3-sonnet", "llama3.1")
        provider: LLM provider ("openai", "anthropic", "ollama")
        api_key: API key (or set via environment variable)
        **kwargs: Additional arguments passed to CodeExplanationCrew

    Returns:
        Configured CodeExplanationCrew instance
    """
    import os

    llm = None

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            model=model,
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            temperature=0.1,
        )

    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        llm = ChatAnthropic(
            model=model,
            api_key=api_key or os.getenv("ANTHROPIC_API_KEY"),
            temperature=0.1,
        )

    elif provider == "ollama":
        from langchain_ollama import ChatOllama
        llm = ChatOllama(
            model=model,
            temperature=0.1,
        )

    elif provider == "nvidia":
        from langchain_nvidia_ai_endpoints import ChatNVIDIA
        llm = ChatNVIDIA(
            model=model,
            api_key=api_key or os.getenv("NVIDIA_API_KEY"),
            temperature=0.1,
        )

    else:
        raise ValueError(f"Unknown provider: {provider}")

    return CodeExplanationCrew(llm=llm, **kwargs)


# Convenience aliases
ExplanationCrew = CodeExplanationCrew
