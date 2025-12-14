"""
GitHub Code Research Tool for CrewAI
Searches for code repositories related to papers and analyzes their contents.
"""

from typing import Optional, List, Dict, Any
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
import httpx
import os
import tempfile
import subprocess
import re
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class GitHubSearchInput(BaseModel):
    """Input schema for GitHub repository search."""
    query: str = Field(..., description="Search query for GitHub repositories")
    max_results: int = Field(default=10, description="Maximum number of results")
    language: Optional[str] = Field(default=None, description="Filter by programming language (e.g., 'python')")


class GitHubRepoInput(BaseModel):
    """Input schema for analyzing a specific repository."""
    repo_url: str = Field(..., description="GitHub repository URL (e.g., 'https://github.com/owner/repo')")


class GitHubCodeSearchInput(BaseModel):
    """Input schema for searching code within a repo."""
    repo_url: str = Field(..., description="GitHub repository URL")
    search_pattern: str = Field(..., description="Code pattern or function name to search for")


class GitHubSearchTool(BaseTool):
    """Tool to search GitHub for repositories related to a research topic."""
    
    name: str = "github_search_repos"
    description: str = """
    Search GitHub for repositories related to a research paper or topic.
    Use this to find implementations of papers, related code, or relevant projects.
    Returns: List of repositories with descriptions, stars, and languages.
    """
    args_schema: type[BaseModel] = GitHubSearchInput
    
    def _run(self, query: str, max_results: int = 10, language: Optional[str] = None) -> str:
        """Search GitHub repositories."""
        try:
            # Build search query
            search_query = query
            if language:
                search_query += f" language:{language}"
            
            # GitHub Search API
            headers = {
                "Accept": "application/vnd.github.v3+json",
            }
            
            # Add token if available
            github_token = os.getenv("GITHUB_TOKEN")
            if github_token:
                headers["Authorization"] = f"token {github_token}"
            
            params = {
                "q": search_query,
                "sort": "stars",
                "order": "desc",
                "per_page": max_results,
            }
            
            with httpx.Client(timeout=30) as client:
                response = client.get(
                    "https://api.github.com/search/repositories",
                    headers=headers,
                    params=params,
                )
                response.raise_for_status()
                data = response.json()
            
            repos = data.get("items", [])
            
            if not repos:
                return f"No repositories found for query: '{query}'"
            
            output = f"Found {len(repos)} repositories for: '{query}'\n\n"
            
            for i, repo in enumerate(repos, 1):
                output += f"## {i}. {repo['full_name']}\n"
                output += f"**URL**: {repo['html_url']}\n"
                output += f"**Stars**: {repo['stargazers_count']} | **Forks**: {repo['forks_count']}\n"
                output += f"**Language**: {repo.get('language', 'N/A')}\n"
                
                if repo.get('description'):
                    output += f"**Description**: {repo['description'][:200]}\n"
                
                if repo.get('topics'):
                    output += f"**Topics**: {', '.join(repo['topics'][:5])}\n"
                
                output += f"**Last Updated**: {repo['updated_at'][:10]}\n\n"
            
            return output
            
        except Exception as e:
            logger.error(f"GitHub search error: {e}")
            return f"Error searching GitHub: {str(e)}"


class GitHubRepoAnalyzerTool(BaseTool):
    """Tool to analyze a GitHub repository structure and README."""
    
    name: str = "github_analyze_repo"
    description: str = """
    Analyze a GitHub repository to understand its structure, purpose, and usage.
    Use this to verify what code does and understand implementation details.
    Returns: README content, file structure, and key code files.
    """
    args_schema: type[BaseModel] = GitHubRepoInput
    
    def _run(self, repo_url: str) -> str:
        """Analyze a GitHub repository."""
        try:
            # Extract owner and repo from URL
            match = re.match(r'https?://github\.com/([^/]+)/([^/]+)/?', repo_url)
            if not match:
                return f"Invalid GitHub URL format: {repo_url}"
            
            owner, repo = match.groups()
            repo = repo.rstrip('.git')
            
            headers = {"Accept": "application/vnd.github.v3+json"}
            github_token = os.getenv("GITHUB_TOKEN")
            if github_token:
                headers["Authorization"] = f"token {github_token}"
            
            with httpx.Client(timeout=30) as client:
                # Get repo info
                repo_response = client.get(
                    f"https://api.github.com/repos/{owner}/{repo}",
                    headers=headers,
                )
                repo_response.raise_for_status()
                repo_data = repo_response.json()
                
                # Get README
                readme_content = "No README found."
                try:
                    readme_response = client.get(
                        f"https://api.github.com/repos/{owner}/{repo}/readme",
                        headers=headers,
                    )
                    if readme_response.status_code == 200:
                        readme_data = readme_response.json()
                        import base64
                        readme_content = base64.b64decode(readme_data['content']).decode('utf-8')
                        # Truncate if too long
                        if len(readme_content) > 3000:
                            readme_content = readme_content[:3000] + "\n\n... [README truncated]"
                except Exception:
                    pass
                
                # Get file tree (root level)
                tree_response = client.get(
                    f"https://api.github.com/repos/{owner}/{repo}/contents/",
                    headers=headers,
                )
                file_tree = []
                if tree_response.status_code == 200:
                    contents = tree_response.json()
                    for item in contents:
                        icon = "📁" if item['type'] == 'dir' else "📄"
                        file_tree.append(f"{icon} {item['name']}")
            
            output = f"# Repository Analysis: {owner}/{repo}\n\n"
            output += f"**URL**: {repo_url}\n"
            output += f"**Description**: {repo_data.get('description', 'N/A')}\n"
            output += f"**Stars**: {repo_data['stargazers_count']} | **Forks**: {repo_data['forks_count']}\n"
            output += f"**Language**: {repo_data.get('language', 'N/A')}\n"
            output += f"**Created**: {repo_data['created_at'][:10]} | **Updated**: {repo_data['updated_at'][:10]}\n"
            
            if repo_data.get('topics'):
                output += f"**Topics**: {', '.join(repo_data['topics'])}\n"
            
            output += f"\n## File Structure\n"
            output += "\n".join(file_tree) + "\n"
            
            output += f"\n## README\n{readme_content}\n"
            
            return output
            
        except Exception as e:
            logger.error(f"GitHub repo analysis error: {e}")
            return f"Error analyzing repository: {str(e)}"


class GitHubCodeFetchTool(BaseTool):
    """Tool to fetch and analyze specific code files from a repository."""
    
    name: str = "github_fetch_code"
    description: str = """
    Fetch specific code files from a GitHub repository.
    Use this to read implementation details, understand algorithms, or verify functionality.
    Returns: Content of the requested file.
    """
    
    def _run(self, repo_url: str, file_path: str) -> str:
        """Fetch a specific file from a repository."""
        try:
            # Extract owner and repo
            match = re.match(r'https?://github\.com/([^/]+)/([^/]+)/?', repo_url)
            if not match:
                return f"Invalid GitHub URL: {repo_url}"
            
            owner, repo = match.groups()
            repo = repo.rstrip('.git')
            
            headers = {"Accept": "application/vnd.github.v3+json"}
            github_token = os.getenv("GITHUB_TOKEN")
            if github_token:
                headers["Authorization"] = f"token {github_token}"
            
            with httpx.Client(timeout=30) as client:
                response = client.get(
                    f"https://api.github.com/repos/{owner}/{repo}/contents/{file_path}",
                    headers=headers,
                )
                
                if response.status_code == 404:
                    return f"File not found: {file_path}"
                
                response.raise_for_status()
                data = response.json()
                
                if data.get('type') == 'dir':
                    # List directory contents
                    files = [f"{'📁' if f['type'] == 'dir' else '📄'} {f['name']}" for f in data]
                    return f"Directory: {file_path}\n\n" + "\n".join(files)
                
                import base64
                content = base64.b64decode(data['content']).decode('utf-8')
                
                # Truncate if too large
                if len(content) > 5000:
                    content = content[:5000] + "\n\n... [File truncated, showing first 5000 chars]"
                
                return f"# File: {file_path}\n\n```\n{content}\n```"
                
        except Exception as e:
            logger.error(f"GitHub file fetch error: {e}")
            return f"Error fetching file: {str(e)}"


class PapersWithCodeTool(BaseTool):
    """Tool to search Papers With Code for implementations."""
    
    name: str = "papers_with_code_search"
    description: str = """
    Search Papers With Code to find official implementations and benchmarks for research papers.
    Use this to find verified code implementations of papers.
    Returns: Papers with their associated code repositories.
    """
    args_schema: type[BaseModel] = GitHubSearchInput
    
    def _run(self, query: str, max_results: int = 10, language: Optional[str] = None) -> str:
        """Search Papers With Code."""
        try:
            with httpx.Client(timeout=30) as client:
                response = client.get(
                    "https://paperswithcode.com/api/v1/search/",
                    params={"q": query, "page": 1},
                )
                
                if response.status_code != 200:
                    # Fallback to GitHub search
                    return f"Papers With Code API unavailable. Use github_search_repos for code search."
                
                data = response.json()
                results = data.get('results', [])[:max_results]
                
                if not results:
                    return f"No papers found on Papers With Code for: '{query}'"
                
                output = f"Found {len(results)} results on Papers With Code for: '{query}'\n\n"
                
                for i, paper in enumerate(results, 1):
                    output += f"## {i}. {paper.get('paper', {}).get('title', 'Unknown')}\n"
                    
                    if paper.get('repository'):
                        repo = paper['repository']
                        output += f"**Code**: {repo.get('url', 'N/A')}\n"
                        output += f"**Stars**: {repo.get('stars', 'N/A')}\n"
                        output += f"**Framework**: {repo.get('framework', 'N/A')}\n"
                    
                    if paper.get('paper', {}).get('arxiv_id'):
                        output += f"**ArXiv**: {paper['paper']['arxiv_id']}\n"
                    
                    output += "\n"
                
                return output
                
        except Exception as e:
            logger.error(f"Papers With Code search error: {e}")
            return f"Error searching Papers With Code: {str(e)}. Try github_search_repos instead."
