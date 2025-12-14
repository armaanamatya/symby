#!/usr/bin/env python3
"""
Symby Research Crew - CLI Entry Point
Autonomous paper research, code finding, and verification.

Usage:
    # Full paper research
    python main.py research paper.pdf --output report.md
    
    # Quick topic research  
    python main.py quick "transformer architectures"
    
    # Find code for a paper
    python main.py code "Attention Is All You Need"
    
    # Verify a repository
    python main.py verify https://github.com/user/repo "implements multi-head attention"
"""

import argparse
import sys
import os
from pathlib import Path
from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

# Load environment variables
load_dotenv()

console = Console()


def main():
    parser = argparse.ArgumentParser(
        description="Autonomous paper research and code verification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s research paper.pdf --output report.md
  %(prog)s quick "transformer attention mechanisms"
  %(prog)s code "Attention Is All You Need" --keywords pytorch attention
  %(prog)s verify https://github.com/user/repo "multi-head attention mechanism"
        """
    )
    
    # Global options
    parser.add_argument(
        "--model", "-m",
        default=os.getenv("CREW_MODEL", "gpt-4o"),
        help="LLM model to use (default: gpt-4o)"
    )
    parser.add_argument(
        "--provider", "-p",
        default=os.getenv("CREW_PROVIDER", "openai"),
        choices=["openai", "anthropic", "ollama"],
        help="LLM provider (default: openai)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Minimal output"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Research command
    research_parser = subparsers.add_parser(
        "research",
        help="Run full research on a paper PDF"
    )
    research_parser.add_argument(
        "paper",
        type=str,
        help="Path to the PDF paper file"
    )
    research_parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output file path for the report"
    )
    research_parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip code verification step"
    )
    
    # Quick research command
    quick_parser = subparsers.add_parser(
        "quick",
        help="Quick research on a topic"
    )
    quick_parser.add_argument(
        "topic",
        type=str,
        help="Topic to research"
    )
    
    # Code search command
    code_parser = subparsers.add_parser(
        "code",
        help="Search for code implementations"
    )
    code_parser.add_argument(
        "title",
        type=str,
        help="Paper title to search for"
    )
    code_parser.add_argument(
        "--keywords", "-k",
        nargs="+",
        default=[],
        help="Additional search keywords"
    )
    
    # Verify command
    verify_parser = subparsers.add_parser(
        "verify",
        help="Verify a code repository"
    )
    verify_parser.add_argument(
        "repo_url",
        type=str,
        help="GitHub repository URL"
    )
    verify_parser.add_argument(
        "claims",
        type=str,
        help="Paper claims to verify against"
    )
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Import crew (delayed to speed up --help)
    from crew import create_research_crew
    
    # Create crew
    if not args.quiet:
        console.print(Panel(
            f"[bold blue]Symby Research Crew[/bold blue]\n"
            f"Model: {args.model} | Provider: {args.provider}",
            title="🔬 Initializing",
        ))
    
    try:
        crew = create_research_crew(
            model=args.model,
            provider=args.provider,
            verbose=args.verbose,
        )
    except Exception as e:
        console.print(f"[red]Error initializing crew: {e}[/red]")
        sys.exit(1)
    
    # Execute command
    try:
        if args.command == "research":
            result = run_research(crew, args)
        elif args.command == "quick":
            result = run_quick(crew, args)
        elif args.command == "code":
            result = run_code_search(crew, args)
        elif args.command == "verify":
            result = run_verify(crew, args)
        else:
            parser.print_help()
            sys.exit(1)
        
        # Display result
        if not args.quiet:
            console.print("\n")
            console.print(Panel(
                Markdown(result),
                title="📋 Results",
                border_style="green",
            ))
        else:
            print(result)
            
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        sys.exit(130)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def run_research(crew, args):
    """Run full paper research."""
    paper_path = Path(args.paper)
    
    if not paper_path.exists():
        raise FileNotFoundError(f"Paper not found: {paper_path}")
    
    console.print(f"[cyan]Researching paper: {paper_path.name}[/cyan]")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Running research crew...", total=None)
        
        results = crew.research_paper(
            paper_path=str(paper_path),
            verify_code=not args.no_verify,
            output_path=args.output,
        )
    
    return results["report"]


def run_quick(crew, args):
    """Run quick topic research."""
    console.print(f"[cyan]Quick research: {args.topic}[/cyan]")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Researching...", total=None)
        result = crew.quick_research(args.topic)
    
    return result


def run_code_search(crew, args):
    """Run code search."""
    console.print(f"[cyan]Searching code for: {args.title}[/cyan]")
    
    keywords = args.keywords if args.keywords else None
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Searching GitHub...", total=None)
        result = crew.find_code(args.title, keywords=keywords)
    
    return result


def run_verify(crew, args):
    """Run repository verification."""
    console.print(f"[cyan]Verifying: {args.repo_url}[/cyan]")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Analyzing repository...", total=None)
        result = crew.verify_repository(args.repo_url, args.claims)
    
    return result


if __name__ == "__main__":
    main()
