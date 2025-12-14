"""
Code Execution and Explanation Tools for CrewAI
Runs code in isolated Docker containers and provides step-by-step explanations.
"""

import asyncio
import re
import sys
import os
from typing import Optional, Dict, Any, List
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
import logging

# Add parent directories to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

logger = logging.getLogger(__name__)


class CodeExecutionInput(BaseModel):
    """Input for code execution."""
    code: str = Field(..., description="Python code to execute in a sandbox container")
    language: str = Field(default="python", description="Programming language (default: python)")
    timeout: int = Field(default=60, description="Maximum execution time in seconds")


class CodeBreakdownInput(BaseModel):
    """Input for code breakdown/explanation."""
    code: str = Field(..., description="Code to analyze and explain step by step")
    detail_level: str = Field(
        default="detailed",
        description="Level of detail: 'brief', 'detailed', or 'comprehensive'"
    )


class CodeStepExecutionInput(BaseModel):
    """Input for step-by-step code execution."""
    code: str = Field(..., description="Code to execute step by step")
    explain_each_step: bool = Field(default=True, description="Whether to explain each step")


class SandboxCodeExecutionTool(BaseTool):
    """
    Tool for executing code in isolated Docker containers.

    Uses the existing SandboxRunner from the pce module for secure execution.
    """

    name: str = "execute_code_sandbox"
    description: str = """
    Execute code in an isolated Docker sandbox container.
    This provides a secure environment for running untrusted code.
    Returns: stdout, stderr, exit code, execution time, and resource usage.
    """
    args_schema: type[BaseModel] = CodeExecutionInput

    def __init__(self):
        super().__init__()
        self._sandbox = None

    def _get_sandbox(self):
        """Lazy-load the sandbox runner."""
        if self._sandbox is None:
            try:
                from src.pce.sandbox_runner import SandboxRunner
                self._sandbox = SandboxRunner(
                    max_memory_gb=4.0,
                    max_timeout_seconds=120,
                    max_cpus=2,
                    enable_gpu=False,
                    require_gpu=False
                )
            except Exception as e:
                logger.warning(f"Failed to initialize SandboxRunner: {e}")
                self._sandbox = "unavailable"
        return self._sandbox if self._sandbox != "unavailable" else None

    def _run(self, code: str, language: str = "python", timeout: int = 60) -> str:
        """Execute code in sandbox container."""
        sandbox = self._get_sandbox()

        if sandbox is None:
            # Fallback: Execute in subprocess if sandbox unavailable
            return self._fallback_execute(code, language, timeout)

        try:
            # Run the async execution
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        try:
            result = loop.run_until_complete(
                sandbox.run(
                    image="python:3.10-slim",
                    code=code,
                    gpu_monitor=False
                )
            )

            output = f"""# Code Execution Results

## Status: {result.status.upper()}

## Output (stdout):
```
{result.stdout or '(no output)'}
```

## Errors (stderr):
```
{result.stderr or '(no errors)'}
```

## Execution Details:
- Exit Code: {result.exit_code}
- Execution Time: {result.execution_time_seconds:.2f} seconds
- Memory Peak: {result.memory_peak_mb:.1f} MB
"""

            if result.gpu_utilization_pct is not None:
                output += f"- GPU Utilization: {result.gpu_utilization_pct:.1f}%\n"
            if result.gpu_memory_used_mb is not None:
                output += f"- GPU Memory Used: {result.gpu_memory_used_mb:.1f} MB\n"

            return output

        except Exception as e:
            logger.error(f"Sandbox execution error: {e}")
            return f"# Execution Error\n\nFailed to execute code in sandbox: {str(e)}"

    def _fallback_execute(self, code: str, language: str, timeout: int) -> str:
        """Fallback execution using subprocess."""
        import subprocess
        import tempfile
        import time

        start_time = time.time()

        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_file = f.name

            result = subprocess.run(
                [sys.executable, temp_file],
                capture_output=True,
                timeout=timeout,
                text=True
            )

            execution_time = time.time() - start_time

            status = "success" if result.returncode == 0 else "error"

            output = f"""# Code Execution Results (Subprocess Fallback)

## Status: {status.upper()}

## Output (stdout):
```
{result.stdout or '(no output)'}
```

## Errors (stderr):
```
{result.stderr or '(no errors)'}
```

## Execution Details:
- Exit Code: {result.returncode}
- Execution Time: {execution_time:.2f} seconds
"""
            return output

        except subprocess.TimeoutExpired:
            return f"# Execution Timeout\n\nCode execution exceeded the {timeout} second limit."
        except Exception as e:
            return f"# Execution Error\n\nFailed to execute code: {str(e)}"
        finally:
            try:
                os.unlink(temp_file)
            except:
                pass


class CodeBreakdownTool(BaseTool):
    """
    Tool for breaking down code into explained steps.

    Analyzes code structure and provides detailed explanations of each component.
    """

    name: str = "breakdown_code"
    description: str = """
    Break down code into individual steps and explain what each part does.
    This helps understand complex code by analyzing it piece by piece.
    Returns: Structured breakdown with explanations for each code section.
    """
    args_schema: type[BaseModel] = CodeBreakdownInput

    def _run(self, code: str, detail_level: str = "detailed") -> str:
        """Break down and explain code."""
        import ast

        output = "# Code Breakdown and Explanation\n\n"

        # Add the original code
        output += "## Original Code\n```python\n" + code + "\n```\n\n"

        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return output + f"## Syntax Error\nCould not parse code: {e}\n"

        output += "## Step-by-Step Breakdown\n\n"

        # Track step number
        step_num = 1

        # Extract and explain imports
        imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
        if imports:
            output += f"### Step {step_num}: Import Statements\n"
            step_num += 1
            for imp in imports:
                if isinstance(imp, ast.Import):
                    for alias in imp.names:
                        module = alias.name
                        as_name = f" as {alias.asname}" if alias.asname else ""
                        output += f"- `import {module}{as_name}`: "
                        output += self._explain_import(module, detail_level) + "\n"
                elif isinstance(imp, ast.ImportFrom):
                    module = imp.module or ""
                    names = ", ".join(alias.name for alias in imp.names)
                    output += f"- `from {module} import {names}`: "
                    output += self._explain_import(module, detail_level) + "\n"
            output += "\n"

        # Extract and explain global variables/constants
        assignments = [node for node in ast.iter_child_nodes(tree)
                      if isinstance(node, ast.Assign)]
        if assignments:
            output += f"### Step {step_num}: Variable Definitions\n"
            step_num += 1
            for assign in assignments:
                targets = [self._get_name(t) for t in assign.targets]
                value_repr = self._get_value_repr(assign.value)
                output += f"- `{', '.join(targets)} = {value_repr}`: "
                output += self._explain_assignment(targets[0], assign.value, detail_level) + "\n"
            output += "\n"

        # Extract and explain functions
        functions = [node for node in ast.iter_child_nodes(tree)
                    if isinstance(node, ast.FunctionDef)]
        for func in functions:
            output += f"### Step {step_num}: Function `{func.name}`\n"
            step_num += 1

            # Function signature
            args = [arg.arg for arg in func.args.args]
            output += f"**Signature**: `def {func.name}({', '.join(args)})`\n\n"

            # Docstring
            docstring = ast.get_docstring(func)
            if docstring:
                output += f"**Purpose**: {docstring[:200]}...\n\n" if len(docstring) > 200 else f"**Purpose**: {docstring}\n\n"

            # Function body breakdown
            if detail_level in ["detailed", "comprehensive"]:
                output += "**Body Breakdown**:\n"
                for i, stmt in enumerate(func.body, 1):
                    if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
                        continue  # Skip docstring
                    stmt_desc = self._explain_statement(stmt, detail_level)
                    output += f"  {i}. {stmt_desc}\n"
            output += "\n"

        # Extract and explain classes
        classes = [node for node in ast.iter_child_nodes(tree)
                  if isinstance(node, ast.ClassDef)]
        for cls in classes:
            output += f"### Step {step_num}: Class `{cls.name}`\n"
            step_num += 1

            bases = [self._get_name(b) for b in cls.bases]
            if bases:
                output += f"**Inherits from**: {', '.join(bases)}\n\n"

            docstring = ast.get_docstring(cls)
            if docstring:
                output += f"**Purpose**: {docstring[:200]}\n\n"

            # Methods
            methods = [n for n in cls.body if isinstance(n, ast.FunctionDef)]
            if methods:
                output += "**Methods**:\n"
                for method in methods:
                    args = [arg.arg for arg in method.args.args]
                    method_doc = ast.get_docstring(method) or "No description"
                    output += f"  - `{method.name}({', '.join(args)})`: {method_doc[:100]}\n"
            output += "\n"

        # Main execution block
        main_block = None
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.If):
                test = ast.unparse(node.test) if hasattr(ast, 'unparse') else str(node.test)
                if "__name__" in test and "__main__" in test:
                    main_block = node
                    break

        if main_block:
            output += f"### Step {step_num}: Main Execution Block\n"
            output += "This code runs when the script is executed directly (not imported).\n\n"
            if detail_level in ["detailed", "comprehensive"]:
                output += "**Actions**:\n"
                for i, stmt in enumerate(main_block.body, 1):
                    stmt_desc = self._explain_statement(stmt, detail_level)
                    output += f"  {i}. {stmt_desc}\n"

        # Summary
        output += "\n## Summary\n"
        output += f"- Total imports: {len(imports)}\n"
        output += f"- Total functions: {len(functions)}\n"
        output += f"- Total classes: {len(classes)}\n"
        output += f"- Has main block: {'Yes' if main_block else 'No'}\n"

        return output

    def _get_name(self, node) -> str:
        """Get name from AST node."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_name(node.value)}.{node.attr}"
        elif isinstance(node, ast.Subscript):
            return f"{self._get_name(node.value)}[...]"
        return str(type(node).__name__)

    def _get_value_repr(self, node) -> str:
        """Get string representation of a value node."""
        if hasattr(ast, 'unparse'):
            try:
                return ast.unparse(node)[:50]
            except:
                pass

        if isinstance(node, ast.Constant):
            return repr(node.value)[:50]
        elif isinstance(node, ast.List):
            return f"[... {len(node.elts)} items ...]"
        elif isinstance(node, ast.Dict):
            return f"{{... {len(node.keys)} items ...}}"
        elif isinstance(node, ast.Call):
            return f"{self._get_name(node.func)}(...)"
        return "..."

    def _explain_import(self, module: str, detail_level: str) -> str:
        """Explain what a module provides."""
        explanations = {
            "numpy": "Numerical computing library for arrays and mathematical operations",
            "pandas": "Data manipulation and analysis library with DataFrames",
            "torch": "PyTorch deep learning framework for neural networks",
            "tensorflow": "TensorFlow machine learning and deep learning library",
            "sklearn": "Scikit-learn machine learning algorithms and utilities",
            "matplotlib": "Plotting and visualization library",
            "os": "Operating system interface for file/process operations",
            "sys": "System-specific parameters and functions",
            "json": "JSON encoding and decoding",
            "re": "Regular expression pattern matching",
            "asyncio": "Asynchronous I/O framework",
            "typing": "Type hints support",
            "datetime": "Date and time handling",
            "pathlib": "Object-oriented filesystem paths",
            "logging": "Logging facility for Python",
            "requests": "HTTP library for making web requests",
            "transformers": "Hugging Face library for NLP models",
        }

        base_module = module.split('.')[0]
        return explanations.get(base_module, f"External module providing {base_module} functionality")

    def _explain_assignment(self, name: str, value, detail_level: str) -> str:
        """Explain a variable assignment."""
        if isinstance(value, ast.Constant):
            return f"Sets `{name}` to the constant value `{repr(value.value)[:30]}`"
        elif isinstance(value, ast.List):
            return f"Creates a list with {len(value.elts)} elements"
        elif isinstance(value, ast.Dict):
            return f"Creates a dictionary with {len(value.keys)} key-value pairs"
        elif isinstance(value, ast.Call):
            func_name = self._get_name(value.func)
            return f"Calls `{func_name}()` and stores the result"
        return f"Assigns a computed value to `{name}`"

    def _explain_statement(self, stmt, detail_level: str) -> str:
        """Explain a statement."""
        if isinstance(stmt, ast.Return):
            return "Returns a value from the function"
        elif isinstance(stmt, ast.Assign):
            targets = [self._get_name(t) for t in stmt.targets]
            return f"Assigns value to `{', '.join(targets)}`"
        elif isinstance(stmt, ast.If):
            return "Conditional check (if/else block)"
        elif isinstance(stmt, ast.For):
            target = self._get_name(stmt.target)
            return f"Loop iterating with variable `{target}`"
        elif isinstance(stmt, ast.While):
            return "While loop (repeats until condition is false)"
        elif isinstance(stmt, ast.With):
            return "Context manager (with statement) for resource management"
        elif isinstance(stmt, ast.Try):
            return "Try/except block for error handling"
        elif isinstance(stmt, ast.Expr):
            if isinstance(stmt.value, ast.Call):
                func_name = self._get_name(stmt.value.func)
                return f"Calls `{func_name}()`"
            return "Expression statement"
        elif isinstance(stmt, ast.AugAssign):
            target = self._get_name(stmt.target)
            return f"Updates `{target}` in place"
        elif isinstance(stmt, ast.Raise):
            return "Raises an exception"
        elif isinstance(stmt, ast.Assert):
            return "Assertion check (fails if condition is false)"

        return f"Statement: {type(stmt).__name__}"


class StepByStepExecutionTool(BaseTool):
    """
    Tool for executing code step by step with explanations.

    Breaks code into logical steps, executes each step, and explains the results.
    """

    name: str = "execute_step_by_step"
    description: str = """
    Execute code step by step, showing the output and explanation for each step.
    This is ideal for understanding how code flows and what each part produces.
    Returns: Step-by-step execution log with outputs and explanations.
    """
    args_schema: type[BaseModel] = CodeStepExecutionInput

    def _run(self, code: str, explain_each_step: bool = True) -> str:
        """Execute code step by step."""
        output = "# Step-by-Step Code Execution\n\n"
        output += "## Original Code\n```python\n" + code + "\n```\n\n"

        # Parse code into executable steps
        try:
            steps = self._parse_into_steps(code)
        except SyntaxError as e:
            return output + f"## Syntax Error\nCould not parse code: {e}\n"

        output += f"## Execution Log ({len(steps)} steps)\n\n"

        # Create a shared namespace for execution
        namespace = {}

        for i, (step_code, step_desc) in enumerate(steps, 1):
            output += f"### Step {i}: {step_desc}\n"
            output += f"```python\n{step_code}\n```\n"

            # Execute this step
            try:
                # Capture stdout
                import io
                from contextlib import redirect_stdout, redirect_stderr

                stdout_capture = io.StringIO()
                stderr_capture = io.StringIO()

                with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                    exec(step_code, namespace)

                stdout_output = stdout_capture.getvalue()
                stderr_output = stderr_capture.getvalue()

                if stdout_output:
                    output += f"\n**Output:**\n```\n{stdout_output}\n```\n"

                if stderr_output:
                    output += f"\n**Warnings/Info:**\n```\n{stderr_output}\n```\n"

                # Show any new variables created
                new_vars = self._detect_new_variables(step_code, namespace)
                if new_vars:
                    output += "\n**Variables updated:**\n"
                    for var, val in new_vars.items():
                        val_repr = repr(val)[:100]
                        output += f"  - `{var}` = `{val_repr}`\n"

                output += "\n**Status:** ✅ Success\n"

            except Exception as e:
                output += f"\n**Status:** ❌ Error\n"
                output += f"**Error:** `{type(e).__name__}: {str(e)}`\n"

            if explain_each_step:
                explanation = self._explain_step(step_code, step_desc)
                output += f"\n**Explanation:** {explanation}\n"

            output += "\n---\n\n"

        # Final state summary
        output += "## Final State\n\n"
        output += "**Variables in scope:**\n"
        for name, value in namespace.items():
            if not name.startswith('_'):
                val_repr = repr(value)[:80]
                val_type = type(value).__name__
                output += f"  - `{name}` ({val_type}): `{val_repr}`\n"

        return output

    def _parse_into_steps(self, code: str) -> List[tuple]:
        """Parse code into executable steps."""
        import ast

        tree = ast.parse(code)
        steps = []

        for node in ast.iter_child_nodes(tree):
            # Get the source code for this node
            if hasattr(ast, 'unparse'):
                step_code = ast.unparse(node)
            else:
                # Fallback: extract lines based on line numbers
                lines = code.splitlines()
                start = node.lineno - 1
                end = getattr(node, 'end_lineno', start + 1)
                step_code = '\n'.join(lines[start:end])

            # Generate description
            step_desc = self._describe_node(node)

            steps.append((step_code, step_desc))

        return steps

    def _describe_node(self, node) -> str:
        """Generate a brief description of an AST node."""
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
            return f"Import {', '.join(names)}"
        elif isinstance(node, ast.ImportFrom):
            return f"Import from {node.module}"
        elif isinstance(node, ast.Assign):
            targets = [self._get_target_name(t) for t in node.targets]
            return f"Assign to {', '.join(targets)}"
        elif isinstance(node, ast.FunctionDef):
            return f"Define function {node.name}"
        elif isinstance(node, ast.ClassDef):
            return f"Define class {node.name}"
        elif isinstance(node, ast.If):
            return "Conditional block"
        elif isinstance(node, ast.For):
            return "For loop"
        elif isinstance(node, ast.While):
            return "While loop"
        elif isinstance(node, ast.Expr):
            if isinstance(node.value, ast.Call):
                func = self._get_target_name(node.value.func)
                return f"Call {func}"
            return "Expression"
        return type(node).__name__

    def _get_target_name(self, node) -> str:
        """Get the name from a target node."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_target_name(node.value)}.{node.attr}"
        return "?"

    def _detect_new_variables(self, step_code: str, namespace: dict) -> dict:
        """Detect variables that were created/modified in this step."""
        # Simple heuristic: check for assignments in the step code
        import ast
        new_vars = {}

        try:
            tree = ast.parse(step_code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            name = target.id
                            if name in namespace:
                                new_vars[name] = namespace[name]
        except:
            pass

        return new_vars

    def _explain_step(self, step_code: str, step_desc: str) -> str:
        """Generate an explanation for the step."""
        explanations = {
            "Import": "Loads external libraries/modules to use their functionality",
            "Assign": "Stores a value in a variable for later use",
            "Define function": "Creates a reusable block of code",
            "Define class": "Creates a template for objects with attributes and methods",
            "Conditional": "Checks a condition and executes code based on the result",
            "For loop": "Iterates over a sequence, executing code for each item",
            "While loop": "Repeats code until a condition becomes false",
            "Call": "Executes a function and returns its result",
        }

        for key, explanation in explanations.items():
            if key.lower() in step_desc.lower():
                return explanation

        return "Executes the given code statement"
