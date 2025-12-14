"""
Paper Research Crew - Main Orchestration
Autonomous multi-agent system for researching papers, finding code, and verification.
"""

from crewai import Crew, Process
from typing import Optional, Dict, Any, List
from pathlib import Path
import logging
import json
from datetime import datetime

from agents import (
    create_paper_analyst_agent,
    create_literature_researcher_agent,
    create_code_hunter_agent,
    create_code_verifier_agent,
    create_synthesis_agent,
)

from tasks import (
    create_paper_analysis_task,
    create_literature_search_task,
    create_code_search_task,
    create_code_verification_task,
    create_synthesis_task,
    create_quick_research_task,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PaperResearchCrew:
    """
    Autonomous research crew that:
    1. Analyzes uploaded research papers
    2. Searches arXiv and Wikipedia for context
    3. Finds code implementations on GitHub
    4. Verifies code matches paper claims
    """
    
    def __init__(
        self,
        llm=None,
        verbose: bool = True,
        memory: bool = True,
        max_rpm: int = 10,
    ):
        """
        Initialize the research crew.
        
        Args:
            llm: LLM instance to use (defaults to OpenAI GPT-4)
            verbose: Enable verbose logging
            memory: Enable crew memory
            max_rpm: Maximum requests per minute
        """
        self.llm = llm
        self.verbose = verbose
        self.memory = memory
        self.max_rpm = max_rpm
        
        # Initialize agents
        self.paper_analyst = create_paper_analyst_agent(llm)
        self.literature_researcher = create_literature_researcher_agent(llm)
        self.code_hunter = create_code_hunter_agent(llm)
        self.code_verifier = create_code_verifier_agent(llm)
        self.synthesizer = create_synthesis_agent(llm)
        
        logger.info("Paper Research Crew initialized with 5 agents")
    
    def research_paper(
        self,
        paper_path: str,
        verify_code: bool = True,
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run full research workflow on an uploaded paper.
        
        Args:
            paper_path: Path to the PDF paper file
            verify_code: Whether to verify found code implementations
            output_path: Optional path to save the report
            
        Returns:
            Dictionary containing research results
        """
        logger.info(f"Starting research on paper: {paper_path}")
        
        # Validate paper exists
        if not Path(paper_path).exists():
            raise FileNotFoundError(f"Paper not found: {paper_path}")
        
        # Create tasks
        tasks = []
        
        # Task 1: Analyze the paper
        analysis_task = create_paper_analysis_task(
            agent=self.paper_analyst,
            paper_path=paper_path,
        )
        tasks.append(analysis_task)
        
        # Task 2: Research related literature
        literature_task = create_literature_search_task(
            agent=self.literature_researcher,
            paper_topic="{{analysis_task.output}}",  # Will be filled from context
            keywords=["machine learning", "deep learning"],  # Default, will be refined
            context=[analysis_task],
        )
        tasks.append(literature_task)
        
        # Task 3: Find code implementations
        code_search_task = create_code_search_task(
            agent=self.code_hunter,
            paper_title="{{analysis_task.output.title}}",
            authors=[],
            keywords=["implementation", "code"],
            context=[analysis_task, literature_task],
        )
        tasks.append(code_search_task)
        
        # Task 4: Verify code (if requested)
        if verify_code:
            verification_task = create_code_verification_task(
                agent=self.code_verifier,
                repo_url="{{code_search_task.output.best_repo}}",
                paper_claims="{{analysis_task.output.claims}}",
                context=[analysis_task, code_search_task],
            )
            tasks.append(verification_task)
        
        # Task 5: Synthesize findings
        synthesis_task = create_synthesis_task(
            agent=self.synthesizer,
            context=tasks,
        )
        tasks.append(synthesis_task)
        
        # Create and run the crew
        crew = Crew(
            agents=[
                self.paper_analyst,
                self.literature_researcher,
                self.code_hunter,
                self.code_verifier if verify_code else None,
                self.synthesizer,
            ],
            tasks=tasks,
            process=Process.sequential,  # Sequential for dependency handling
            verbose=self.verbose,
            memory=self.memory,
            max_rpm=self.max_rpm,
        )
        
        # Remove None agents
        crew.agents = [a for a in crew.agents if a is not None]
        
        # Execute
        logger.info("Executing research crew...")
        result = crew.kickoff()
        
        # Package results
        research_results = {
            "paper_path": paper_path,
            "timestamp": datetime.now().isoformat(),
            "report": str(result),
            "tasks_completed": len(tasks),
        }
        
        # Save if output path provided
        if output_path:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Save markdown report
            with open(output_file, "w") as f:
                f.write(str(result))
            
            # Save JSON metadata
            json_path = output_file.with_suffix(".json")
            with open(json_path, "w") as f:
                json.dump(research_results, f, indent=2)
            
            logger.info(f"Results saved to: {output_path}")
        
        return research_results
    
    def quick_research(self, topic: str) -> str:
        """
        Run a quick research sweep on a topic.
        
        Args:
            topic: Research topic or query
            
        Returns:
            Quick research summary
        """
        logger.info(f"Quick research on: {topic}")
        
        task = create_quick_research_task(
            agent=self.literature_researcher,
            topic=topic,
        )
        
        crew = Crew(
            agents=[self.literature_researcher],
            tasks=[task],
            process=Process.sequential,
            verbose=self.verbose,
        )
        
        result = crew.kickoff()
        return str(result)
    
    def find_code(self, paper_title: str, keywords: List[str] = None) -> str:
        """
        Search for code implementations of a paper.
        
        Args:
            paper_title: Title of the paper
            keywords: Additional search keywords
            
        Returns:
            Code search results
        """
        logger.info(f"Searching code for: {paper_title}")
        
        task = create_code_search_task(
            agent=self.code_hunter,
            paper_title=paper_title,
            authors=[],
            keywords=keywords or ["implementation", "pytorch", "tensorflow"],
        )
        
        crew = Crew(
            agents=[self.code_hunter],
            tasks=[task],
            process=Process.sequential,
            verbose=self.verbose,
        )
        
        result = crew.kickoff()
        return str(result)
    
    def verify_repository(self, repo_url: str, paper_claims: str) -> str:
        """
        Verify a code repository against paper claims.
        
        Args:
            repo_url: GitHub repository URL
            paper_claims: Description of what the paper claims
            
        Returns:
            Verification report
        """
        logger.info(f"Verifying repository: {repo_url}")
        
        task = create_code_verification_task(
            agent=self.code_verifier,
            repo_url=repo_url,
            paper_claims=paper_claims,
        )
        
        crew = Crew(
            agents=[self.code_verifier],
            tasks=[task],
            process=Process.sequential,
            verbose=self.verbose,
        )
        
        result = crew.kickoff()
        return str(result)


def create_research_crew(
    model: str = "gpt-4o",
    provider: str = "openai",
    api_key: Optional[str] = None,
    **kwargs,
) -> PaperResearchCrew:
    """
    Factory function to create a research crew with specific LLM configuration.
    
    Args:
        model: Model name (e.g., "gpt-4o", "claude-3-sonnet", "llama3.1")
        provider: LLM provider ("openai", "anthropic", "ollama")
        api_key: API key (or set via environment variable)
        **kwargs: Additional arguments passed to PaperResearchCrew
        
    Returns:
        Configured PaperResearchCrew instance
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
    
    else:
        raise ValueError(f"Unknown provider: {provider}")
    
    return PaperResearchCrew(llm=llm, **kwargs)


# Convenience aliases
ResearchCrew = PaperResearchCrew
