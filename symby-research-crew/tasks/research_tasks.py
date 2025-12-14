"""
Research Tasks for CrewAI
Defines the workflow tasks for autonomous paper research.
"""

from crewai import Task
from typing import Optional


def create_paper_analysis_task(agent, paper_path: str, context: Optional[list] = None) -> Task:
    """
    Create task for analyzing an uploaded paper.
    """
    return Task(
        description=f"""Analyze the research paper located at: {paper_path}

        Your analysis should include:
        1. **Paper Metadata**: Title, authors, publication date, venue
        2. **Problem Statement**: What problem does this paper address?
        3. **Key Methods**: What novel methods or approaches does the paper propose?
        4. **Main Contributions**: List the paper's claimed contributions
        5. **Datasets Used**: What datasets were used for evaluation?
        6. **Key Results**: Summarize the main experimental results
        7. **Technical Details**: Extract any important hyperparameters, architectures, or algorithms
        8. **Keywords**: Identify key terms for searching related work
        
        Be thorough but concise. Focus on information that will help find and verify implementations.""",
        expected_output="""A structured analysis report containing:
        - Paper metadata (title, authors, date)
        - Problem statement summary
        - List of key methods/approaches
        - Claimed contributions
        - Datasets and benchmarks mentioned
        - Summary of results
        - Technical specifications
        - Keywords for further research""",
        agent=agent,
        context=context,
    )


def create_literature_search_task(agent, paper_topic: str, keywords: list, context: Optional[list] = None) -> Task:
    """
    Create task for researching related literature.
    """
    keywords_str = ", ".join(keywords)
    
    return Task(
        description=f"""Research the academic landscape around this topic: {paper_topic}

        Use these keywords to guide your search: {keywords_str}
        
        Your research should:
        1. **Search arXiv** for related papers in the field
        2. **Find Prior Work**: Identify foundational papers that this work builds upon
        3. **Find Related Methods**: Search for alternative approaches to the same problem
        4. **Background Concepts**: Use Wikipedia to explain key technical concepts
        5. **Research Context**: Understand where this paper fits in the research timeline
        
        Focus on finding papers that:
        - Directly influenced this work (cited by the paper)
        - Propose competing or complementary methods
        - Apply similar techniques to different domains
        - Have available code implementations""",
        expected_output="""A literature review containing:
        - List of 5-10 most relevant related papers with arXiv IDs
        - Brief explanation of how each relates to the main paper
        - Background explanations of key concepts
        - Research timeline/context
        - Recommendations for papers likely to have code""",
        agent=agent,
        context=context,
    )


def create_code_search_task(agent, paper_title: str, authors: list, keywords: list, context: Optional[list] = None) -> Task:
    """
    Create task for finding code implementations.
    """
    authors_str = ", ".join(authors[:3]) if authors else "Unknown"
    keywords_str = ", ".join(keywords)
    
    return Task(
        description=f"""Find all available code implementations for the paper: "{paper_title}"

        Authors: {authors_str}
        Search keywords: {keywords_str}
        
        Your search should:
        1. **Search Papers With Code** for official implementations
        2. **Search GitHub** using:
           - Paper title
           - Author names + topic keywords
           - Method names mentioned in the paper
        3. **Identify Official vs Community**: Note which repos are by paper authors
        4. **Evaluate Quality**: Check stars, forks, recency, and documentation
        5. **Find Multiple Implementations**: Look for both original and reimplementations
        
        For each repository found, provide:
        - Repository URL
        - Stars/forks count
        - Whether it's official or community-made
        - Programming language and framework
        - Last update date
        - Quality assessment""",
        expected_output="""A catalog of code implementations including:
        - List of repositories with URLs
        - Classification (official/community)
        - Quality metrics (stars, activity)
        - Framework information
        - Ranked recommendations of which to use""",
        agent=agent,
        context=context,
    )


def create_code_verification_task(agent, repo_url: str, paper_claims: str, context: Optional[list] = None) -> Task:
    """
    Create task for verifying code implementation.
    """
    return Task(
        description=f"""Verify that the code implementation matches the paper's claims.

        Repository: {repo_url}
        
        Paper's Key Claims:
        {paper_claims}
        
        Your verification should:
        1. **Analyze Repository Structure**: Examine file organization and key components
        2. **Read Documentation**: Check README for setup instructions and claimed features
        3. **Examine Core Code**: Look at main implementation files
        4. **Verify Architecture**: Check if model/algorithm matches paper description
        5. **Check for Completeness**: 
           - Is the full pipeline implemented?
           - Are training scripts included?
           - Are evaluation scripts present?
           - Are pretrained models available?
        6. **Identify Gaps**: Note anything missing from the paper
        7. **Assess Reproducibility**: Can results be reproduced with this code?
        
        Be specific about what is and isn't implemented.""",
        expected_output="""A verification report containing:
        - Repository structure overview
        - Alignment score (how well code matches paper)
        - List of implemented features
        - List of missing/incomplete features
        - Code quality assessment
        - Reproducibility assessment
        - Specific recommendations or warnings""",
        agent=agent,
        context=context,
    )


def create_synthesis_task(agent, context: list) -> Task:
    """
    Create task for synthesizing all findings into a final report.
    """
    return Task(
        description="""Synthesize all research findings into a comprehensive final report.

        Review all the previous analysis and create a unified report that:
        
        1. **Executive Summary**: Brief overview of the paper and findings
        2. **Paper Analysis**: 
           - What the paper does
           - Key contributions and methods
           - Main results
        3. **Research Context**:
           - Related work
           - Where this fits in the field
        4. **Code Availability**:
           - Available implementations
           - Recommended code to use
        5. **Verification Results**:
           - Does the code match the paper?
           - What's implemented vs missing?
        6. **Recommendations**:
           - Best code to start with
           - Gaps to be aware of
           - Suggested next steps
        
        Make the report accessible to both researchers and practitioners.""",
        expected_output="""A comprehensive research report in markdown format with:
        - Executive summary (2-3 sentences)
        - Detailed paper analysis
        - Literature context
        - Code implementation status
        - Verification findings
        - Clear recommendations and next steps""",
        agent=agent,
        context=context,
    )


def create_quick_research_task(agent, topic: str) -> Task:
    """
    Create a quick research task for a single topic.
    """
    return Task(
        description=f"""Quickly research the topic: {topic}

        Perform a focused research sweep:
        1. Search arXiv for the most relevant papers
        2. Look up background information on Wikipedia
        3. Search GitHub for related implementations
        
        Provide a quick summary of what you find.""",
        expected_output="""A brief research summary containing:
        - Top 3-5 relevant papers
        - Key concepts explained
        - Available code repositories
        - Quick recommendation for next steps""",
        agent=agent,
    )
