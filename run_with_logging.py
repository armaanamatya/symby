#!/usr/bin/env python3
"""
Run any demo with automatic logging

Usage:
    python run_with_logging.py demo_nvidia_rag.py
    python run_with_logging.py demo_fixed_with_vectorstore.py
    python run_with_logging.py test_nvidia_integration.py
"""

import sys
import os
import subprocess
from datetime import datetime
from pathlib import Path


def run_with_logging(script_path: str, args: list = None):
    """
    Run a script and log all output.

    Args:
        script_path: Path to script to run
        args: Additional arguments
    """
    script_path = Path(script_path)
    if not script_path.exists():
        print(f"❌ Script not found: {script_path}")
        return 1

    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # Generate log filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    script_name = script_path.stem
    log_file = log_dir / f"{script_name}_{timestamp}.log"

    print(f"\n📝 Running: {script_path}")
    print(f"📝 Logging to: {log_file}\n")

    # Run script and capture output
    command = [sys.executable, str(script_path)]
    if args:
        command.extend(args)

    try:
        # Run with both stdout and stderr captured
        with open(log_file, 'w') as f:
            # Write header
            f.write("="*70 + "\n")
            f.write(f"Script: {script_path}\n")
            f.write(f"Started: {datetime.now().isoformat()}\n")
            f.write(f"Command: {' '.join(command)}\n")
            f.write("="*70 + "\n\n")
            f.flush()

            # Run with live output
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )

            # Stream output to both console and file
            for line in process.stdout:
                print(line, end='')
                f.write(line)
                f.flush()

            # Wait for completion
            returncode = process.wait()

            # Write footer
            f.write("\n" + "="*70 + "\n")
            f.write(f"Completed: {datetime.now().isoformat()}\n")
            f.write(f"Exit code: {returncode}\n")
            f.write("="*70 + "\n")

        if returncode == 0:
            print(f"\n✅ Script completed successfully")
        else:
            print(f"\n❌ Script exited with code {returncode}")

        print(f"📝 Full log saved to: {log_file}\n")
        return returncode

    except KeyboardInterrupt:
        print(f"\n\n⏸️  Interrupted by user")
        print(f"📝 Partial log saved to: {log_file}\n")
        return 130

    except Exception as e:
        print(f"\n❌ Error running script: {e}")
        return 1


def list_demo_scripts():
    """List available demo scripts."""
    demos = [
        "demo_nvidia_rag.py",
        "demo_fixed_with_vectorstore.py",
        "test_nvidia_integration.py",
        "check_api_key.py"
    ]

    print("\nAvailable demos:")
    for demo in demos:
        exists = "✅" if Path(demo).exists() else "❌"
        print(f"  {exists} {demo}")


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("""
Usage: python run_with_logging.py <script> [args...]

Examples:
    python run_with_logging.py demo_nvidia_rag.py
    python run_with_logging.py demo_fixed_with_vectorstore.py
    python run_with_logging.py test_nvidia_integration.py

Features:
    ✅ Captures all output (stdout + stderr)
    ✅ Shows output live in console
    ✅ Saves to timestamped log file in logs/
    ✅ Handles interrupts gracefully
""")
        list_demo_scripts()
        return 1

    script = sys.argv[1]
    args = sys.argv[2:] if len(sys.argv) > 2 else []

    return run_with_logging(script, args)


if __name__ == "__main__":
    sys.exit(main())
