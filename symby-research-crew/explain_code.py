#!/usr/bin/env python3
"""
Code Explanation CLI - Run and Explain Code Step by Step

This CLI provides an easy way to:
- Execute code in isolated Docker containers
- Get step-by-step breakdowns of what code does
- Debug code with detailed explanations
- Compare different code implementations

Usage:
    # Explain code from a file
    python explain_code.py explain code.py --detail detailed

    # Execute code in sandbox
    python explain_code.py execute code.py --timeout 60

    # Step-by-step execution
    python explain_code.py steps code.py

    # Debug code
    python explain_code.py debug code.py --error "IndexError on line 5"

    # Compare two implementations
    python explain_code.py compare v1.py v2.py

    # Interactive mode with inline code
    python explain_code.py explain --code "print('Hello, World!')"
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
from rich.syntax import Syntax

# Load environment variables
load_dotenv()

console = Console()


def read_code_file(file_path: str) -> str:
    """Read code from a file."""
    path = Path(file_path)
    if not path.exists():
        console.print(f"[red]Error: File not found: {file_path}[/red]")
        sys.exit(1)
    return path.read_text()


def main():
    parser = argparse.ArgumentParser(
        description="Execute and explain code step by step using AI agents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s explain mycode.py --detail comprehensive
  %(prog)s execute script.py --timeout 120
  %(prog)s steps algorithm.py --output explanation.md
  %(prog)s debug buggy.py --error "TypeError: unsupported operand"
  %(prog)s compare old.py new.py
  %(prog)s tutor sorting.py --level beginner
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
        choices=["openai", "anthropic", "ollama", "nvidia"],
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

    # Explain command
    explain_parser = subparsers.add_parser(
        "explain",
        help="Explain code step by step"
    )
    explain_parser.add_argument(
        "file",
        type=str,
        nargs="?",
        help="Path to the code file"
    )
    explain_parser.add_argument(
        "--code", "-c",
        type=str,
        help="Inline code to explain (alternative to file)"
    )
    explain_parser.add_argument(
        "--detail", "-d",
        choices=["brief", "detailed", "comprehensive"],
        default="detailed",
        help="Level of detail (default: detailed)"
    )
    explain_parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output file path for the explanation"
    )

    # Execute command
    execute_parser = subparsers.add_parser(
        "execute",
        help="Execute code in a sandbox container"
    )
    execute_parser.add_argument(
        "file",
        type=str,
        nargs="?",
        help="Path to the code file"
    )
    execute_parser.add_argument(
        "--code", "-c",
        type=str,
        help="Inline code to execute"
    )
    execute_parser.add_argument(
        "--timeout", "-t",
        type=int,
        default=60,
        help="Execution timeout in seconds (default: 60)"
    )
    execute_parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output file path"
    )

    # Steps command
    steps_parser = subparsers.add_parser(
        "steps",
        help="Execute code step by step with explanations"
    )
    steps_parser.add_argument(
        "file",
        type=str,
        nargs="?",
        help="Path to the code file"
    )
    steps_parser.add_argument(
        "--code", "-c",
        type=str,
        help="Inline code"
    )
    steps_parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output file path"
    )

    # Debug command
    debug_parser = subparsers.add_parser(
        "debug",
        help="Debug code and explain issues"
    )
    debug_parser.add_argument(
        "file",
        type=str,
        nargs="?",
        help="Path to the code file"
    )
    debug_parser.add_argument(
        "--code", "-c",
        type=str,
        help="Inline code"
    )
    debug_parser.add_argument(
        "--error", "-e",
        type=str,
        default="",
        help="Description of the error or unexpected behavior"
    )
    debug_parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output file path"
    )

    # Compare command
    compare_parser = subparsers.add_parser(
        "compare",
        help="Compare two code implementations"
    )
    compare_parser.add_argument(
        "file1",
        type=str,
        help="Path to the first code file"
    )
    compare_parser.add_argument(
        "file2",
        type=str,
        help="Path to the second code file"
    )
    compare_parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output file path"
    )

    # Tutor command
    tutor_parser = subparsers.add_parser(
        "tutor",
        help="Get a tutorial-style explanation of code"
    )
    tutor_parser.add_argument(
        "file",
        type=str,
        nargs="?",
        help="Path to the code file"
    )
    tutor_parser.add_argument(
        "--code", "-c",
        type=str,
        help="Inline code"
    )
    tutor_parser.add_argument(
        "--level", "-l",
        choices=["beginner", "intermediate", "advanced"],
        default="intermediate",
        help="Learner level (default: intermediate)"
    )
    tutor_parser.add_argument(
        "--topics",
        nargs="+",
        help="Specific topics to focus on"
    )
    tutor_parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output file path"
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Import crew (delayed to speed up --help)
    from code_explanation_crew import create_code_explanation_crew

    # Create crew
    if not args.quiet:
        console.print(Panel(
            f"[bold blue]Code Explanation Crew[/bold blue]\n"
            f"Model: {args.model} | Provider: {args.provider}",
            title="🔧 Initializing",
        ))

    try:
        crew = create_code_explanation_crew(
            model=args.model,
            provider=args.provider,
            verbose=args.verbose,
        )
    except Exception as e:
        console.print(f"[red]Error initializing crew: {e}[/red]")
        sys.exit(1)

    # Execute command
    try:
        if args.command == "explain":
            result = run_explain(crew, args)
        elif args.command == "execute":
            result = run_execute(crew, args)
        elif args.command == "steps":
            result = run_steps(crew, args)
        elif args.command == "debug":
            result = run_debug(crew, args)
        elif args.command == "compare":
            result = run_compare(crew, args)
        elif args.command == "tutor":
            result = run_tutor(crew, args)
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


def get_code(args) -> str:
    """Get code from file or inline argument."""
    if hasattr(args, 'code') and args.code:
        return args.code
    if hasattr(args, 'file') and args.file:
        return read_code_file(args.file)
    console.print("[red]Error: Please provide either a file or --code argument[/red]")
    sys.exit(1)


def run_explain(crew, args):
    """Run code explanation."""
    code = get_code(args)

    # Show the code being analyzed
    console.print(Panel(
        Syntax(code, "python", theme="monokai", line_numbers=True),
        title="📄 Code to Explain",
    ))

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task("Analyzing and explaining code...", total=None)
        result = crew.explain_code(
            code=code,
            detail_level=args.detail,
            output_path=args.output,
        )

    return result["explanation"]


def run_execute(crew, args):
    """Run code execution."""
    code = get_code(args)

    console.print(Panel(
        Syntax(code, "python", theme="monokai", line_numbers=True),
        title="📄 Code to Execute",
    ))

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task(f"Executing in sandbox (timeout: {args.timeout}s)...", total=None)
        result = crew.execute_code(
            code=code,
            timeout_seconds=args.timeout,
            output_path=args.output,
        )

    return result["result"]


def run_steps(crew, args):
    """Run step-by-step execution."""
    code = get_code(args)

    console.print(Panel(
        Syntax(code, "python", theme="monokai", line_numbers=True),
        title="📄 Code for Step-by-Step Execution",
    ))

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task("Executing step by step...", total=None)
        result = crew.step_by_step(
            code=code,
            output_path=args.output,
        )

    return result["steps"]


def run_debug(crew, args):
    """Run code debugging."""
    code = get_code(args)

    console.print(Panel(
        Syntax(code, "python", theme="monokai", line_numbers=True),
        title="📄 Code to Debug",
    ))

    if args.error:
        console.print(f"[yellow]Reported error: {args.error}[/yellow]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task("Debugging and analyzing...", total=None)
        result = crew.debug_code(
            code=code,
            error_description=args.error,
            output_path=args.output,
        )

    return result["analysis"]


def run_compare(crew, args):
    """Run code comparison."""
    code_v1 = read_code_file(args.file1)
    code_v2 = read_code_file(args.file2)

    console.print(Panel(
        Syntax(code_v1, "python", theme="monokai", line_numbers=True),
        title=f"📄 Version 1: {args.file1}",
    ))
    console.print(Panel(
        Syntax(code_v2, "python", theme="monokai", line_numbers=True),
        title=f"📄 Version 2: {args.file2}",
    ))

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task("Comparing implementations...", total=None)
        result = crew.compare_code(
            code_v1=code_v1,
            code_v2=code_v2,
            output_path=args.output,
        )

    return result["comparison"]


def run_tutor(crew, args):
    """Run code tutoring."""
    code = get_code(args)

    console.print(Panel(
        Syntax(code, "python", theme="monokai", line_numbers=True),
        title="📄 Code to Learn",
    ))

    console.print(f"[cyan]Learner level: {args.level}[/cyan]")
    if args.topics:
        console.print(f"[cyan]Focus topics: {', '.join(args.topics)}[/cyan]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task("Creating tutorial...", total=None)
        result = crew.tutor_code(
            code=code,
            learner_level=args.level,
            focus_topics=args.topics,
            output_path=args.output,
        )

    return result["tutorial"]


if __name__ == "__main__":
    main()
