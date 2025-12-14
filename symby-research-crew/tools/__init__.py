"""
Research Tools Package for CrewAI
Exports all available tools for the research crew.
"""

from .arxiv_tools import (
    ArxivSearchTool,
    ArxivPaperTool,
    ArxivCitationSearchTool,
)

from .github_tools import (
    GitHubSearchTool,
    GitHubRepoAnalyzerTool,
    GitHubCodeFetchTool,
    PapersWithCodeTool,
)

from .wikipedia_tools import (
    WikipediaSearchTool,
    WikipediaPageTool,
    WikipediaSectionTool,
    ConceptDefinitionTool,
)

from .pdf_tools import (
    PDFParserTool,
    PDFExtractSectionTool,
    PDFMetadataTool,
)

from .code_tools import (
    CodeStaticAnalysisTool,
    CodeVerificationTool,
    RepoStructureAnalyzer,
)

from .code_execution_tools import (
    SandboxCodeExecutionTool,
    CodeBreakdownTool,
    StepByStepExecutionTool,
)

__all__ = [
    # ArXiv Tools
    "ArxivSearchTool",
    "ArxivPaperTool",
    "ArxivCitationSearchTool",

    # GitHub Tools
    "GitHubSearchTool",
    "GitHubRepoAnalyzerTool",
    "GitHubCodeFetchTool",
    "PapersWithCodeTool",

    # Wikipedia Tools
    "WikipediaSearchTool",
    "WikipediaPageTool",
    "WikipediaSectionTool",
    "ConceptDefinitionTool",

    # PDF Tools
    "PDFParserTool",
    "PDFExtractSectionTool",
    "PDFMetadataTool",

    # Code Analysis Tools
    "CodeStaticAnalysisTool",
    "CodeVerificationTool",
    "RepoStructureAnalyzer",

    # Code Execution Tools
    "SandboxCodeExecutionTool",
    "CodeBreakdownTool",
    "StepByStepExecutionTool",
]


def get_all_tools():
    """Get instances of all available tools."""
    return [
        # Research tools
        ArxivSearchTool(),
        ArxivPaperTool(),
        ArxivCitationSearchTool(),

        # Code search tools
        GitHubSearchTool(),
        GitHubRepoAnalyzerTool(),
        GitHubCodeFetchTool(),
        PapersWithCodeTool(),

        # Background research
        WikipediaSearchTool(),
        WikipediaPageTool(),
        WikipediaSectionTool(),
        ConceptDefinitionTool(),

        # Paper analysis
        PDFParserTool(),
        PDFExtractSectionTool(),
        PDFMetadataTool(),

        # Code verification
        CodeStaticAnalysisTool(),
        CodeVerificationTool(),
        RepoStructureAnalyzer(),

        # Code execution
        SandboxCodeExecutionTool(),
        CodeBreakdownTool(),
        StepByStepExecutionTool(),
    ]
