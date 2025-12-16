"""
Logging utility to capture demo outputs

Logs both to console and file with timestamps
"""

import sys
import os
from datetime import datetime
from pathlib import Path


class DualLogger:
    """
    Logs output to both console and file.

    Usage:
        logger = DualLogger("demo_run")
        logger.log("This goes to both console and file")
    """

    def __init__(self, log_name: str = "demo", log_dir: str = "logs"):
        """
        Initialize dual logger.

        Args:
            log_name: Base name for log file
            log_dir: Directory to store logs
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        # Create timestamped log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"{log_name}_{timestamp}.log"

        # Keep original stdout
        self.terminal = sys.stdout

        # Open log file
        self.log_handle = open(self.log_file, 'w', encoding='utf-8')

        # Write header
        self.log_handle.write("="*70 + "\n")
        self.log_handle.write(f"Log started: {datetime.now().isoformat()}\n")
        self.log_handle.write(f"Log file: {self.log_file}\n")
        self.log_handle.write("="*70 + "\n\n")
        self.log_handle.flush()

        print(f"\n📝 Logging to: {self.log_file}\n")

    def write(self, message):
        """Write to both console and log file."""
        self.terminal.write(message)
        self.log_handle.write(message)
        self.log_handle.flush()  # Ensure it's written immediately

    def flush(self):
        """Flush both outputs."""
        self.terminal.flush()
        self.log_handle.flush()

    def close(self):
        """Close log file and restore stdout."""
        self.log_handle.write("\n" + "="*70 + "\n")
        self.log_handle.write(f"Log ended: {datetime.now().isoformat()}\n")
        self.log_handle.write("="*70 + "\n")
        self.log_handle.close()
        print(f"\n📝 Log saved to: {self.log_file}\n")

    def __enter__(self):
        """Context manager entry."""
        # Redirect stdout to logger
        sys.stdout = self
        return self

    def __exit__(self, *args):
        """Context manager exit."""
        # Restore stdout
        sys.stdout = self.terminal
        self.close()


class TimestampedLogger(DualLogger):
    """Logger that adds timestamps to each line."""

    def write(self, message):
        """Write with timestamp prefix."""
        if message.strip():  # Only timestamp non-empty lines
            timestamp = datetime.now().strftime("%H:%M:%S")
            message = f"[{timestamp}] {message}"
        super().write(message)


def setup_logging(log_name: str = "demo", timestamped: bool = False):
    """
    Setup logging for a demo/script.

    Args:
        log_name: Base name for log file
        timestamped: Whether to add timestamps to each line

    Returns:
        Context manager logger

    Usage:
        with setup_logging("my_demo"):
            print("This is logged!")
            # Your code here
    """
    if timestamped:
        return TimestampedLogger(log_name)
    else:
        return DualLogger(log_name)


# Example usage
if __name__ == "__main__":
    print("\n=== Testing Logger ===\n")

    # Test 1: Basic logging
    print("Test 1: Basic logging")
    with setup_logging("test_basic"):
        print("This goes to both console and file!")
        print("Multiple lines work too.")
        print("Numbers: 1, 2, 3")

    # Test 2: Timestamped logging
    print("\nTest 2: Timestamped logging")
    with setup_logging("test_timestamped", timestamped=True):
        print("This line has a timestamp")
        print("So does this one")

    print("\n✅ Logger test complete")
    print("Check the logs/ directory for output files")
