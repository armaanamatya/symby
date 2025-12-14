"""
Wikipedia Research Tool for CrewAI
Searches and retrieves information from Wikipedia for background context.
"""

from typing import Optional, List
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
import wikipedia
import httpx
import logging

logger = logging.getLogger(__name__)


class WikipediaSearchInput(BaseModel):
    """Input schema for Wikipedia search."""
    query: str = Field(..., description="Search query for Wikipedia")
    max_results: int = Field(default=5, description="Maximum search results to return")


class WikipediaPageInput(BaseModel):
    """Input schema for fetching a Wikipedia page."""
    title: str = Field(..., description="Title of the Wikipedia page to fetch")
    sentences: int = Field(default=10, description="Number of sentences to return from summary")


class WikipediaSearchTool(BaseTool):
    """Tool to search Wikipedia for relevant articles."""
    
    name: str = "wikipedia_search"
    description: str = """
    Search Wikipedia for articles related to a topic.
    Use this to find background information, definitions, and context for research topics.
    Returns: List of relevant Wikipedia article titles.
    """
    args_schema: type[BaseModel] = WikipediaSearchInput
    
    def _run(self, query: str, max_results: int = 5) -> str:
        """Search Wikipedia for relevant articles."""
        try:
            wikipedia.set_lang("en")
            results = wikipedia.search(query, results=max_results)
            
            if not results:
                return f"No Wikipedia articles found for: '{query}'"
            
            output = f"Found {len(results)} Wikipedia articles for: '{query}'\n\n"
            
            for i, title in enumerate(results, 1):
                output += f"{i}. {title}\n"
                
                # Try to get a brief summary
                try:
                    summary = wikipedia.summary(title, sentences=2, auto_suggest=False)
                    output += f"   {summary[:200]}...\n\n"
                except wikipedia.exceptions.DisambiguationError as e:
                    output += f"   (Multiple meanings - see options: {', '.join(e.options[:3])})\n\n"
                except Exception:
                    output += f"   (Summary unavailable)\n\n"
            
            return output
            
        except Exception as e:
            logger.error(f"Wikipedia search error: {e}")
            return f"Error searching Wikipedia: {str(e)}"


class WikipediaPageTool(BaseTool):
    """Tool to fetch detailed Wikipedia page content."""
    
    name: str = "wikipedia_get_page"
    description: str = """
    Fetch detailed content from a specific Wikipedia page.
    Use this to get comprehensive background information about a concept or topic.
    Returns: Page summary, sections, and key information.
    """
    args_schema: type[BaseModel] = WikipediaPageInput
    
    def _run(self, title: str, sentences: int = 10) -> str:
        """Fetch Wikipedia page content."""
        try:
            wikipedia.set_lang("en")
            
            try:
                page = wikipedia.page(title, auto_suggest=True)
            except wikipedia.exceptions.DisambiguationError as e:
                # Return disambiguation options
                return f"'{title}' refers to multiple topics:\n" + "\n".join(f"- {opt}" for opt in e.options[:10])
            except wikipedia.exceptions.PageError:
                # Try searching instead
                results = wikipedia.search(title, results=3)
                if results:
                    return f"Page '{title}' not found. Did you mean:\n" + "\n".join(f"- {r}" for r in results)
                return f"No Wikipedia page found for: '{title}'"
            
            output = f"# {page.title}\n\n"
            output += f"**URL**: {page.url}\n\n"
            output += f"## Summary\n{wikipedia.summary(title, sentences=sentences)}\n\n"
            
            # Get section titles
            if page.sections:
                output += "## Sections\n"
                for section in page.sections[:15]:  # First 15 sections
                    output += f"- {section}\n"
                output += "\n"
            
            # Get categories
            if page.categories:
                output += "## Categories\n"
                output += ", ".join(page.categories[:10]) + "\n\n"
            
            # Get links to related topics
            if page.links:
                output += "## Related Topics (first 10)\n"
                for link in page.links[:10]:
                    output += f"- {link}\n"
            
            return output
            
        except Exception as e:
            logger.error(f"Wikipedia page fetch error: {e}")
            return f"Error fetching Wikipedia page: {str(e)}"


class WikipediaSectionTool(BaseTool):
    """Tool to fetch a specific section from a Wikipedia page."""
    
    name: str = "wikipedia_get_section"
    description: str = """
    Fetch a specific section from a Wikipedia page.
    Use this when you need detailed information about a particular aspect of a topic.
    """
    
    def _run(self, title: str, section: str) -> str:
        """Fetch a specific section from a Wikipedia page."""
        try:
            wikipedia.set_lang("en")
            page = wikipedia.page(title, auto_suggest=True)
            
            # Get full content and try to find section
            content = page.content
            
            # Simple section extraction (Wikipedia sections are marked with ==)
            section_pattern = f"== {section} =="
            alt_pattern = f"=={section}=="
            
            start_idx = content.find(section_pattern)
            if start_idx == -1:
                start_idx = content.find(alt_pattern)
            
            if start_idx == -1:
                # Try case-insensitive
                content_lower = content.lower()
                section_lower = section.lower()
                for pattern in [f"== {section_lower} ==", f"=={section_lower}=="]:
                    start_idx = content_lower.find(pattern)
                    if start_idx != -1:
                        break
            
            if start_idx == -1:
                available_sections = page.sections[:10]
                return f"Section '{section}' not found. Available sections:\n" + "\n".join(f"- {s}" for s in available_sections)
            
            # Find end of section (next == or end of content)
            end_idx = content.find("\n==", start_idx + 1)
            if end_idx == -1:
                end_idx = len(content)
            
            section_content = content[start_idx:end_idx].strip()
            
            # Truncate if too long
            if len(section_content) > 3000:
                section_content = section_content[:3000] + "\n\n... [Section truncated]"
            
            return f"# {title} - {section}\n\n{section_content}"
            
        except wikipedia.exceptions.DisambiguationError as e:
            return f"'{title}' is ambiguous. Options:\n" + "\n".join(f"- {opt}" for opt in e.options[:5])
        except Exception as e:
            logger.error(f"Wikipedia section fetch error: {e}")
            return f"Error fetching section: {str(e)}"


class ConceptDefinitionTool(BaseTool):
    """Tool to quickly get definitions of technical concepts."""
    
    name: str = "define_concept"
    description: str = """
    Get a quick definition and explanation of a technical or scientific concept.
    Use this to understand terminology and jargon in research papers.
    Returns: Brief definition and context from Wikipedia.
    """
    args_schema: type[BaseModel] = WikipediaSearchInput
    
    def _run(self, query: str, max_results: int = 1) -> str:
        """Get definition of a concept."""
        try:
            wikipedia.set_lang("en")
            
            # Try direct lookup first
            try:
                summary = wikipedia.summary(query, sentences=3, auto_suggest=True)
                return f"**{query}**: {summary}"
            except wikipedia.exceptions.DisambiguationError as e:
                # Pick the first option that seems most relevant
                for option in e.options[:5]:
                    try:
                        summary = wikipedia.summary(option, sentences=3, auto_suggest=False)
                        return f"**{option}**: {summary}"
                    except Exception:
                        continue
                return f"Multiple meanings for '{query}':\n" + "\n".join(f"- {opt}" for opt in e.options[:5])
            except wikipedia.exceptions.PageError:
                # Try searching
                results = wikipedia.search(query, results=3)
                if results:
                    try:
                        summary = wikipedia.summary(results[0], sentences=3, auto_suggest=False)
                        return f"**{results[0]}**: {summary}"
                    except Exception:
                        return f"Could not define '{query}'. Related topics: {', '.join(results)}"
                return f"No definition found for: '{query}'"
                
        except Exception as e:
            logger.error(f"Concept definition error: {e}")
            return f"Error defining concept: {str(e)}"
