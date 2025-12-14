"""
Research Agents for CrewAI
Specialized agents for autonomous paper research and code verification.
"""

from crewai import Agent
from typing import List, Optional
from tools import (
    ArxivSearchTool,
    ArxivPaperTool,
    ArxivCitationSearchTool,
    GitHubSearchTool,
    GitHubRepoAnalyzerTool,
    GitHubCodeFetchTool,
    PapersWithCodeTool,
    WikipediaSearchTool,
    WikipediaPageTool,
    ConceptDefinitionTool,
    PDFParserTool,
    PDFExtractSectionTool,
    PDFMetadataTool,
    CodeStaticAnalysisTool,
    CodeVerificationTool,
    RepoStructureAnalyzer,
)


def create_paper_analyst_agent(llm=None) -> Agent:
    """
    Creates the Paper Analyst agent.
    
    Responsibilities:
    - Parse and understand uploaded research papers
    - Extract key concepts, methods, and claims
    - Identify the paper's main contributions
    """
    tools = [
        PDFParserTool(),
        PDFExtractSectionTool(),
        PDFMetadataTool(),
    ]
    
    return Agent(
        role="Paper Analyst",
        goal="Thoroughly analyze uploaded research papers to extract key information, methods, claims, and identify the paper's main contributions",
        backstory="""You are an expert research analyst with a PhD-level understanding of 
        scientific literature across multiple domains. You excel at quickly parsing dense 
        academic papers and extracting the essential information: the problem being solved, 
        the novel methods proposed, the key experimental results, and the paper's claimed 
        contributions to the field. You have a keen eye for identifying the core innovations 
        that distinguish a paper from prior work.""",
        tools=tools,
        llm=llm,
        verbose=True,
        allow_delegation=True,
        memory=True,
    )


def create_literature_researcher_agent(llm=None) -> Agent:
    """
    Creates the Literature Researcher agent.
    
    Responsibilities:
    - Search arXiv for related papers
    - Find prior work and citations
    - Build context around the research topic
    """
    tools = [
        ArxivSearchTool(),
        ArxivPaperTool(),
        ArxivCitationSearchTool(),
        WikipediaSearchTool(),
        WikipediaPageTool(),
        ConceptDefinitionTool(),
    ]
    
    return Agent(
        role="Literature Researcher",
        goal="Research the academic landscape around a paper's topic, finding related work on arXiv, understanding background concepts, and placing the paper in its proper research context",
        backstory="""You are a seasoned research librarian and literature review specialist 
        with deep expertise in navigating academic databases and scientific literature. You 
        understand how research builds upon prior work and can quickly identify the key 
        papers that influenced a given work, as well as subsequent papers that cite it.
        You're skilled at using Wikipedia and other resources to explain complex concepts 
        in accessible terms. Your literature reviews are comprehensive yet focused.""",
        tools=tools,
        llm=llm,
        verbose=True,
        allow_delegation=True,
        memory=True,
    )


def create_code_hunter_agent(llm=None) -> Agent:
    """
    Creates the Code Hunter agent.
    
    Responsibilities:
    - Find code implementations of papers
    - Search GitHub and Papers With Code
    - Identify official vs. unofficial implementations
    """
    tools = [
        GitHubSearchTool(),
        GitHubRepoAnalyzerTool(),
        GitHubCodeFetchTool(),
        PapersWithCodeTool(),
    ]
    
    return Agent(
        role="Code Hunter",
        goal="Find and catalog all available code implementations of a research paper, identifying official repositories, high-quality community implementations, and related projects",
        backstory="""You are a skilled software archaeologist who specializes in tracking 
        down code implementations of research papers. You know all the best places to look: 
        official project pages, GitHub, Papers With Code, and various research lab 
        repositories. You can distinguish between official implementations by paper authors 
        and community reproductions, and you know how to evaluate code quality and 
        completeness. You're familiar with the conventions of ML/AI research code.""",
        tools=tools,
        llm=llm,
        verbose=True,
        allow_delegation=True,
        memory=True,
    )


def create_code_verifier_agent(llm=None) -> Agent:
    """
    Creates the Code Verifier agent.
    
    Responsibilities:
    - Analyze code structure and implementation
    - Verify code matches paper descriptions
    - Assess code quality and completeness
    """
    tools = [
        GitHubRepoAnalyzerTool(),
        GitHubCodeFetchTool(),
        CodeStaticAnalysisTool(),
        CodeVerificationTool(),
        RepoStructureAnalyzer(),
    ]
    
    return Agent(
        role="Code Verifier",
        goal="Thoroughly analyze and verify that code implementations actually match what the paper describes, checking for completeness, correctness, and potential gaps",
        backstory="""You are a meticulous code reviewer and research engineer who has 
        reviewed hundreds of paper implementations. You know exactly what to look for: 
        does the code implement the algorithm as described? Are the hyperparameters 
        correct? Is the model architecture faithful to the paper? Are there hidden 
        assumptions or shortcuts? You can read code across multiple languages and 
        frameworks, and you're not afraid to dig deep into implementation details. 
        Your verification reports are thorough and actionable.""",
        tools=tools,
        llm=llm,
        verbose=True,
        allow_delegation=True,
        memory=True,
    )


def create_synthesis_agent(llm=None) -> Agent:
    """
    Creates the Synthesis Agent (Coordinator).
    
    Responsibilities:
    - Coordinate findings from all agents
    - Produce final comprehensive research report
    - Identify gaps and recommend next steps
    """
    return Agent(
        role="Research Synthesizer",
        goal="Synthesize findings from all research agents into a comprehensive, well-structured report that clearly communicates the paper's content, available implementations, and verification results",
        backstory="""You are a senior research manager and science communicator who excels 
        at bringing together disparate pieces of information into coherent narratives. 
        You've written hundreds of research summaries and technical reports. You know how 
        to highlight the most important findings while providing enough context for 
        readers at various expertise levels. Your reports are clear, well-organized, 
        and actionable, with specific recommendations for next steps.""",
        tools=[],  # Synthesis agent primarily coordinates, doesn't need tools
        llm=llm,
        verbose=True,
        allow_delegation=True,
        memory=True,
    )


def get_all_agents(llm=None) -> dict:
    """
    Create and return all research agents.
    
    Args:
        llm: Optional LLM instance to use for all agents
        
    Returns:
        Dictionary of agent name -> Agent instance
    """
    return {
        "paper_analyst": create_paper_analyst_agent(llm),
        "literature_researcher": create_literature_researcher_agent(llm),
        "code_hunter": create_code_hunter_agent(llm),
        "code_verifier": create_code_verifier_agent(llm),
        "synthesizer": create_synthesis_agent(llm),
    }
