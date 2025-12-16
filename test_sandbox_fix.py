#!/usr/bin/env python3
"""
Test script for the fixed SandboxRunner with subprocess fallback.

Tests:
1. Simple code execution
2. Code with auto-detected requirements
3. Fallback when Docker fails
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from pce.sandbox_runner import SandboxRunner


async def test_simple_execution():
    """Test 1: Simple code execution."""
    print("\n" + "="*60)
    print("TEST 1: Simple Code Execution")
    print("="*60)

    code = """
import sys
print(f"Hello from sandbox!")
print(f"Python version: {sys.version}")
print("Success!")
"""

    runner = SandboxRunner(
        max_memory_gb=2.0,
        max_timeout_seconds=60,
        enable_gpu=False
    )

    try:
        result = await runner.run(image="python:3.10-slim", code=code)

        print(f"\n✓ Status: {result.status}")
        print(f"✓ Exit Code: {result.exit_code}")
        print(f"✓ Execution Time: {result.execution_time_seconds:.2f}s")
        print(f"\n--- STDOUT ---")
        print(result.stdout)
        if result.stderr:
            print(f"\n--- STDERR ---")
            print(result.stderr)

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()


async def test_with_requirements():
    """Test 2: Code with auto-detected requirements."""
    print("\n" + "="*60)
    print("TEST 2: Code with Auto-Detected Requirements")
    print("="*60)

    code = """
import json
import sys

# Test that we can use json (stdlib, should work)
data = {"status": "success", "python": sys.version}
print(json.dumps(data, indent=2))
"""

    runner = SandboxRunner(
        max_memory_gb=2.0,
        max_timeout_seconds=60,
        enable_gpu=False
    )

    try:
        result = await runner.run(image="python:3.10-slim", code=code)

        print(f"\n✓ Status: {result.status}")
        print(f"✓ Exit Code: {result.exit_code}")
        print(f"\n--- STDOUT ---")
        print(result.stdout)

    except Exception as e:
        print(f"\n✗ Test failed: {e}")


async def test_fallback():
    """Test 3: Fallback when Docker image doesn't exist."""
    print("\n" + "="*60)
    print("TEST 3: Subprocess Fallback (when Docker fails)")
    print("="*60)

    code = """
import sys
print("Running via subprocess fallback!")
print(f"Python: {sys.version}")
"""

    runner = SandboxRunner(
        max_memory_gb=2.0,
        max_timeout_seconds=60,
        enable_gpu=False
    )

    try:
        # Use a non-existent image to force fallback
        result = await runner.run(image="nonexistent:image", code=code)

        print(f"\n✓ Status: {result.status}")
        print(f"✓ Exit Code: {result.exit_code}")
        print(f"\n--- STDOUT ---")
        print(result.stdout)
        if result.stderr:
            print(f"\n--- STDERR ---")
            print(result.stderr)

    except Exception as e:
        print(f"\n✗ Test failed: {e}")


async def test_large_code():
    """Test 4: Large code that would trigger 'argument list too long'."""
    print("\n" + "="*60)
    print("TEST 4: Large Code (would fail with -c flag)")
    print("="*60)

    # Generate a large code block
    code = "# " + ("Large code block " * 100) + "\n"
    code += """
import sys

# This is a large code file that would exceed argument limits
# if passed via python -c "..."

def large_function():
    '''A large function with lots of code.'''
    data = []
"""

    # Add lots of lines to make it large
    for i in range(1000):
        code += f"    data.append({i})  # Line {i}\n"

    code += """
    return sum(data)

result = large_function()
print(f"Result: {result}")
print("Large code execution successful!")
"""

    print(f"Code size: {len(code)} bytes")

    runner = SandboxRunner(
        max_memory_gb=2.0,
        max_timeout_seconds=60,
        enable_gpu=False
    )

    try:
        result = await runner.run(image="python:3.10-slim", code=code)

        print(f"\n✓ Status: {result.status}")
        print(f"✓ Exit Code: {result.exit_code}")
        print(f"\n--- STDOUT ---")
        print(result.stdout)

    except Exception as e:
        print(f"\n✗ Test failed: {e}")


async def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("SANDBOX RUNNER TESTS - With Subprocess Fallback & Auto-Requirements")
    print("="*60)

    await test_simple_execution()
    await test_with_requirements()
    await test_fallback()
    await test_large_code()

    print("\n" + "="*60)
    print("ALL TESTS COMPLETED")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())
