"""
ArXiv Research Tool for CrewAI
Searches and retrieves papers from arXiv.org
"""

from typing import Optional, List, Dict, Any
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
import arxiv
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential
import logging

logger = logging.getLogger(__name__)


class ArxivSearchInput(BaseModel):
    """Input schema for ArXiv search."""
    query: str = Field(..., description="Search query for arXiv papers")
    max_results: int = Field(default=10, description="Maximum number of results to return")
    sort_by: str = Field(
        default="relevance",
        description="Sort by: 'relevance', 'lastUpdatedDate', 'submittedDate'"
    )


class ArxivPaperInput(BaseModel):
    """Input schema for fetching a specific paper."""
    arxiv_id: str = Field(..., description="ArXiv paper ID (e.g., '2301.07041' or 'arxiv:2301.07041')")


class ArxivSearchTool(BaseTool):
    """Tool to search arXiv for research papers."""
    
    name: str = "arxiv_search"
    description: str = """
    Search arXiv for research papers. Use this to find papers related to a topic,
    discover related work, or find papers by specific authors.
    Returns: List of papers with titles, abstracts, authors, and arXiv IDs.
    """
    args_schema: type[BaseModel] = ArxivSearchInput
    
    def _run(self, query: str, max_results: int = 10, sort_by: str = "relevance") -> str:
        """Execute arXiv search."""
        try:
            # Map sort options
            sort_map = {
                "relevance": arxiv.SortCriterion.Relevance,
                "lastUpdatedDate": arxiv.SortCriterion.LastUpdatedDate,
                "submittedDate": arxiv.SortCriterion.SubmittedDate,
            }
            sort_criterion = sort_map.get(sort_by, arxiv.SortCriterion.Relevance)
            
            # Create search client
            client = arxiv.Client()
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=sort_criterion,
            )
            
            results = []
            for paper in client.results(search):
                results.append({
                    "arxiv_id": paper.entry_id.split("/")[-1],
                    "title": paper.title,
                    "authors": [a.name for a in paper.authors[:5]],  # First 5 authors
                    "abstract": paper.summary[:500] + "..." if len(paper.summary) > 500 else paper.summary,
                    "published": paper.published.strftime("%Y-%m-%d"),
                    "updated": paper.updated.strftime("%Y-%m-%d"),
                    "categories": paper.categories,
                    "pdf_url": paper.pdf_url,
                    "primary_category": paper.primary_category,
                })
            
            if not results:
                return f"No papers found for query: '{query}'"
            
            # Format output
            output = f"Found {len(results)} papers for query: '{query}'\n\n"
            for i, paper in enumerate(results, 1):
                output += f"## {i}. {paper['title']}\n"
                output += f"**ArXiv ID**: {paper['arxiv_id']}\n"
                output += f"**Authors**: {', '.join(paper['authors'])}\n"
                output += f"**Published**: {paper['published']} | **Updated**: {paper['updated']}\n"
                output += f"**Categories**: {', '.join(paper['categories'])}\n"
                output += f"**PDF**: {paper['pdf_url']}\n"
                output += f"**Abstract**: {paper['abstract']}\n\n"
            
            return output
            
        except Exception as e:
            logger.error(f"ArXiv search error: {e}")
            return f"Error searching arXiv: {str(e)}"


class ArxivPaperTool(BaseTool):
    """Tool to fetch details of a specific arXiv paper."""
    
    name: str = "arxiv_get_paper"
    description: str = """
    Fetch detailed information about a specific arXiv paper by its ID.
    Use this when you have an arXiv ID and need the full paper details.
    Returns: Paper title, authors, full abstract, categories, and download links.
    """
    args_schema: type[BaseModel] = ArxivPaperInput
    
    def _run(self, arxiv_id: str) -> str:
        """Fetch paper details by arXiv ID."""
        try:
            # Clean up arxiv_id
            arxiv_id = arxiv_id.replace("arxiv:", "").replace("arXiv:", "")
            arxiv_id = arxiv_id.split("v")[0]  # Remove version if present
            
            client = arxiv.Client()
            search = arxiv.Search(id_list=[arxiv_id])
            
            paper = next(client.results(search), None)
            if not paper:
                return f"Paper not found with ID: {arxiv_id}"
            
            output = f"# {paper.title}\n\n"
            output += f"**ArXiv ID**: {arxiv_id}\n"
            output += f"**Authors**: {', '.join([a.name for a in paper.authors])}\n"
            output += f"**Published**: {paper.published.strftime('%Y-%m-%d')}\n"
            output += f"**Updated**: {paper.updated.strftime('%Y-%m-%d')}\n"
            output += f"**Primary Category**: {paper.primary_category}\n"
            output += f"**All Categories**: {', '.join(paper.categories)}\n"
            output += f"**PDF URL**: {paper.pdf_url}\n"
            output += f"**Abstract Page**: {paper.entry_id}\n\n"
            output += f"## Abstract\n{paper.summary}\n\n"
            
            # Try to find code links in the paper
            if paper.links:
                output += "## Links\n"
                for link in paper.links:
                    output += f"- {link.href}\n"
            
            return output
            
        except Exception as e:
            logger.error(f"ArXiv paper fetch error: {e}")
            return f"Error fetching paper {arxiv_id}: {str(e)}"


class ArxivCitationSearchTool(BaseTool):
    """Tool to find papers that cite or are cited by a given paper."""
    
    name: str = "arxiv_find_related"
    description: str = """
    Find related papers by searching for papers that reference similar concepts.
    Use when you need to understand the context around a paper or find related work.
    """
    args_schema: type[BaseModel] = ArxivPaperInput
    
    def _run(self, arxiv_id: str) -> str:
        """Find related papers."""
        try:
            # First get the original paper
            arxiv_id = arxiv_id.replace("arxiv:", "").replace("arXiv:", "")
            client = arxiv.Client()
            search = arxiv.Search(id_list=[arxiv_id])
            paper = next(client.results(search), None)
            
            if not paper:
                return f"Paper not found: {arxiv_id}"
            
            # Extract key terms from title
            title_words = paper.title.lower().split()
            stopwords = {'a', 'an', 'the', 'in', 'on', 'at', 'for', 'to', 'of', 'and', 'or', 'with', 'by'}
            keywords = [w for w in title_words if w not in stopwords and len(w) > 3][:5]
            
            # Search for related papers using keywords
            query = " AND ".join(f'ti:"{k}"' if " " not in k else k for k in keywords[:3])
            
            related_search = arxiv.Search(
                query=query,
                max_results=10,
                sort_by=arxiv.SortCriterion.Relevance,
            )
            
            results = []
            for related in client.results(related_search):
                if related.entry_id != paper.entry_id:  # Exclude original paper
                    results.append({
                        "arxiv_id": related.entry_id.split("/")[-1],
                        "title": related.title,
                        "authors": [a.name for a in related.authors[:3]],
                        "published": related.published.strftime("%Y-%m-%d"),
                    })
            
            output = f"# Related Papers for: {paper.title}\n\n"
            output += f"Search terms: {query}\n\n"
            
            if not results:
                output += "No closely related papers found.\n"
            else:
                for i, r in enumerate(results[:10], 1):
                    output += f"{i}. **{r['title']}**\n"
                    output += f"   - ID: {r['arxiv_id']} | Authors: {', '.join(r['authors'])} | {r['published']}\n\n"
            
            return output
            
        except Exception as e:
            logger.error(f"Related papers search error: {e}")
            return f"Error finding related papers: {str(e)}"
