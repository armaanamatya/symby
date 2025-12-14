"""
Utility functions for the Research Crew
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import yaml

logger = logging.getLogger(__name__)


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load crew configuration from YAML file.
    
    Args:
        config_path: Path to config file (defaults to config/crew_config.yaml)
        
    Returns:
        Configuration dictionary
    """
    if config_path is None:
        config_path = Path(__file__).parent.parent / "config" / "crew_config.yaml"
    else:
        config_path = Path(config_path)
    
    if not config_path.exists():
        logger.warning(f"Config file not found: {config_path}")
        return {}
    
    with open(config_path) as f:
        return yaml.safe_load(f)


def setup_logging(
    level: str = "INFO",
    format_string: Optional[str] = None,
) -> None:
    """
    Configure logging for the research crew.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        format_string: Custom log format
    """
    if format_string is None:
        format_string = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_string,
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def validate_environment() -> Dict[str, bool]:
    """
    Validate required environment variables are set.
    
    Returns:
        Dictionary of variable names and their availability
    """
    required_vars = {
        "OPENAI_API_KEY": "OpenAI API (required for default provider)",
        "GITHUB_TOKEN": "GitHub API (optional, improves rate limits)",
    }
    
    optional_vars = {
        "ANTHROPIC_API_KEY": "Anthropic API (for Claude models)",
    }
    
    results = {}
    
    for var, description in required_vars.items():
        value = os.getenv(var)
        results[var] = bool(value)
        if not value:
            logger.warning(f"Missing {var}: {description}")
    
    for var, description in optional_vars.items():
        results[var] = bool(os.getenv(var))
    
    return results


def get_paper_path(path: str) -> Path:
    """
    Resolve and validate paper path.
    
    Args:
        path: Path to paper (relative or absolute)
        
    Returns:
        Resolved Path object
        
    Raises:
        FileNotFoundError: If paper doesn't exist
    """
    paper_path = Path(path).resolve()
    
    if not paper_path.exists():
        raise FileNotFoundError(f"Paper not found: {paper_path}")
    
    if not paper_path.suffix.lower() == ".pdf":
        logger.warning(f"Expected PDF file, got: {paper_path.suffix}")
    
    return paper_path


def format_report(
    report: str,
    format_type: str = "markdown",
    include_toc: bool = True,
) -> str:
    """
    Format a research report for output.
    
    Args:
        report: Raw report text
        format_type: Output format (markdown, html, plain)
        include_toc: Whether to include table of contents
        
    Returns:
        Formatted report
    """
    if format_type == "markdown":
        if include_toc:
            # Generate simple TOC from headers
            lines = report.split("\n")
            toc_lines = ["## Table of Contents\n"]
            for line in lines:
                if line.startswith("## ") and not line.startswith("## Table"):
                    title = line[3:].strip()
                    anchor = title.lower().replace(" ", "-").replace(":", "")
                    toc_lines.append(f"- [{title}](#{anchor})")
            
            # Insert TOC after first header
            first_header_end = report.find("\n", report.find("# "))
            if first_header_end > 0:
                report = (
                    report[:first_header_end + 1] + 
                    "\n".join(toc_lines) + "\n\n" +
                    report[first_header_end + 1:]
                )
        
        return report
    
    elif format_type == "html":
        try:
            import markdown
            return markdown.markdown(report, extensions=['tables', 'fenced_code'])
        except ImportError:
            logger.warning("markdown library not installed, returning raw text")
            return report
    
    else:
        # Plain text - strip markdown formatting
        import re
        plain = report
        plain = re.sub(r'\*\*([^*]+)\*\*', r'\1', plain)  # Bold
        plain = re.sub(r'\*([^*]+)\*', r'\1', plain)       # Italic
        plain = re.sub(r'`([^`]+)`', r'\1', plain)         # Code
        plain = re.sub(r'#{1,6}\s*', '', plain)            # Headers
        return plain
