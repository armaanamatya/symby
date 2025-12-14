"""
Code Verification Tools for CrewAI
Analyzes and verifies code repositories against paper claims.
"""

from typing import Optional, List, Dict, Any
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
import subprocess
import tempfile
import os
import re
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class CodeAnalysisInput(BaseModel):
    """Input for code analysis."""
    code: str = Field(..., description="Code to analyze")
    language: str = Field(default="python", description="Programming language")


class RepoCloneInput(BaseModel):
    """Input for cloning and analyzing a repository."""
    repo_url: str = Field(..., description="Git repository URL to clone and analyze")
    branch: str = Field(default="main", description="Branch to clone")


class CodeStaticAnalysisTool(BaseTool):
    """Tool for static code analysis."""
    
    name: str = "analyze_code_static"
    description: str = """
    Perform static analysis on code to understand its structure and functionality.
    Use this to verify what code does without running it.
    Returns: Functions, classes, imports, and code structure analysis.
    """
    args_schema: type[BaseModel] = CodeAnalysisInput
    
    def _run(self, code: str, language: str = "python") -> str:
        """Analyze code statically."""
        try:
            if language.lower() == "python":
                return self._analyze_python(code)
            else:
                return self._generic_analysis(code, language)
        except Exception as e:
            logger.error(f"Code analysis error: {e}")
            return f"Error analyzing code: {str(e)}"
    
    def _analyze_python(self, code: str) -> str:
        """Analyze Python code using AST."""
        import ast
        
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return f"Syntax error in code: {e}"
        
        output = "# Python Code Analysis\n\n"
        
        # Extract imports
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    imports.append(f"{module}.{alias.name}")
        
        if imports:
            output += "## Imports\n"
            for imp in sorted(set(imports)):
                output += f"- {imp}\n"
            output += "\n"
        
        # Extract classes
        classes = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
                bases = [self._get_name(b) for b in node.bases]
                classes.append({
                    'name': node.name,
                    'bases': bases,
                    'methods': methods,
                    'docstring': ast.get_docstring(node),
                })
        
        if classes:
            output += "## Classes\n"
            for cls in classes:
                output += f"### class {cls['name']}"
                if cls['bases']:
                    output += f"({', '.join(cls['bases'])})"
                output += "\n"
                if cls['docstring']:
                    output += f"*{cls['docstring'][:200]}*\n"
                if cls['methods']:
                    output += f"Methods: {', '.join(cls['methods'])}\n"
                output += "\n"
        
        # Extract top-level functions
        functions = []
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.FunctionDef):
                args = [a.arg for a in node.args.args]
                functions.append({
                    'name': node.name,
                    'args': args,
                    'docstring': ast.get_docstring(node),
                    'decorators': [self._get_name(d) for d in node.decorator_list],
                })
        
        if functions:
            output += "## Functions\n"
            for func in functions:
                output += f"### def {func['name']}({', '.join(func['args'])})\n"
                if func['decorators']:
                    output += f"Decorators: {', '.join(func['decorators'])}\n"
                if func['docstring']:
                    output += f"*{func['docstring'][:200]}*\n"
                output += "\n"
        
        # Basic metrics
        output += "## Metrics\n"
        output += f"- Lines of code: {len(code.splitlines())}\n"
        output += f"- Classes: {len(classes)}\n"
        output += f"- Functions: {len(functions)}\n"
        output += f"- Imports: {len(imports)}\n"
        
        return output
    
    def _get_name(self, node) -> str:
        """Get name from AST node."""
        import ast
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_name(node.value)}.{node.attr}"
        elif isinstance(node, ast.Call):
            return self._get_name(node.func)
        return str(type(node).__name__)
    
    def _generic_analysis(self, code: str, language: str) -> str:
        """Generic code analysis for non-Python languages."""
        output = f"# {language.title()} Code Analysis\n\n"
        
        lines = code.splitlines()
        output += f"## Metrics\n"
        output += f"- Lines of code: {len(lines)}\n"
        output += f"- Non-empty lines: {len([l for l in lines if l.strip()])}\n"
        output += f"- Comment lines: {len([l for l in lines if l.strip().startswith(('#', '//', '/*', '*'))])}\n"
        
        # Extract function patterns
        func_patterns = {
            'javascript': r'(?:function\s+(\w+)|(\w+)\s*=\s*(?:async\s+)?function|(\w+)\s*=\s*(?:async\s+)?\([^)]*\)\s*=>)',
            'java': r'(?:public|private|protected)?\s*(?:static\s+)?[\w<>\[\]]+\s+(\w+)\s*\([^)]*\)',
            'cpp': r'[\w:]+\s+(\w+)\s*\([^)]*\)\s*(?:const)?\s*(?:override)?\s*[{;]',
            'rust': r'fn\s+(\w+)',
        }
        
        if language.lower() in func_patterns:
            pattern = func_patterns[language.lower()]
            matches = re.findall(pattern, code)
            if matches:
                output += f"\n## Functions Found\n"
                for match in matches[:20]:
                    name = match if isinstance(match, str) else next((m for m in match if m), None)
                    if name:
                        output += f"- {name}\n"
        
        return output


class CodeVerificationTool(BaseTool):
    """Tool to verify code claims against paper descriptions."""
    
    name: str = "verify_code_claims"
    description: str = """
    Verify that code implementation matches paper descriptions.
    Use this to check if a repository actually implements what a paper describes.
    Returns: Analysis of alignment between code and claimed functionality.
    """
    
    def _run(self, paper_claims: str, code_analysis: str) -> str:
        """Verify code against paper claims."""
        output = "# Code Verification Report\n\n"
        
        # Extract key terms from paper claims
        claim_keywords = self._extract_keywords(paper_claims)
        code_keywords = self._extract_keywords(code_analysis)
        
        # Find overlaps
        common = claim_keywords & code_keywords
        missing_in_code = claim_keywords - code_keywords
        extra_in_code = code_keywords - claim_keywords
        
        output += "## Keyword Analysis\n\n"
        output += f"**Matching concepts**: {', '.join(sorted(common)[:20]) if common else 'None found'}\n\n"
        
        if missing_in_code:
            output += f"**Potentially missing from code**: {', '.join(sorted(missing_in_code)[:10])}\n\n"
        
        if extra_in_code:
            output += f"**Additional in code**: {', '.join(sorted(extra_in_code)[:10])}\n\n"
        
        # Calculate alignment score
        if claim_keywords:
            alignment = len(common) / len(claim_keywords) * 100
            output += f"## Alignment Score: {alignment:.1f}%\n\n"
            
            if alignment > 70:
                output += "✅ **HIGH ALIGNMENT**: Code appears to implement most paper concepts.\n"
            elif alignment > 40:
                output += "⚠️ **MODERATE ALIGNMENT**: Some concepts present, verification recommended.\n"
            else:
                output += "❌ **LOW ALIGNMENT**: Significant gaps between paper and code.\n"
        
        output += "\n## Recommendations\n"
        output += "1. Review the code's README and documentation for implementation details\n"
        output += "2. Check if missing concepts are implemented under different names\n"
        output += "3. Look for test files that demonstrate the claimed functionality\n"
        output += "4. Compare numerical results if benchmarks are available\n"
        
        return output
    
    def _extract_keywords(self, text: str) -> set:
        """Extract relevant keywords from text."""
        # Common technical terms to look for
        text_lower = text.lower()
        
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                     'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                     'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare',
                     'ought', 'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by',
                     'from', 'as', 'into', 'through', 'during', 'before', 'after', 'above',
                     'below', 'between', 'under', 'again', 'further', 'then', 'once', 'and',
                     'but', 'or', 'nor', 'so', 'yet', 'both', 'either', 'neither', 'not',
                     'only', 'same', 'than', 'too', 'very', 'just', 'also', 'now', 'this',
                     'that', 'these', 'those', 'such', 'which', 'who', 'whom', 'whose',
                     'what', 'when', 'where', 'why', 'how', 'all', 'each', 'every', 'any',
                     'some', 'most', 'other', 'none', 'def', 'class', 'import', 'return'}
        
        # Extract words
        words = re.findall(r'\b[a-z][a-z_]+\b', text_lower)
        
        # Filter and return
        keywords = {w for w in words if w not in stop_words and len(w) > 3}
        
        return keywords


class RepoStructureAnalyzer(BaseTool):
    """Tool to analyze repository structure for implementation verification."""
    
    name: str = "analyze_repo_structure"
    description: str = """
    Analyze a cloned repository's structure to verify implementation completeness.
    Use this to understand how code is organized and what components are implemented.
    """
    args_schema: type[BaseModel] = RepoCloneInput
    
    def _run(self, repo_url: str, branch: str = "main") -> str:
        """Analyze repository structure."""
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                # Clone repo
                clone_cmd = ["git", "clone", "--depth", "1", "--branch", branch, repo_url, tmpdir]
                result = subprocess.run(clone_cmd, capture_output=True, text=True, timeout=60)
                
                if result.returncode != 0:
                    # Try without branch specification
                    clone_cmd = ["git", "clone", "--depth", "1", repo_url, tmpdir]
                    result = subprocess.run(clone_cmd, capture_output=True, text=True, timeout=60)
                    if result.returncode != 0:
                        return f"Failed to clone repository: {result.stderr}"
                
                # Analyze structure
                output = f"# Repository Structure Analysis\n\n"
                output += f"**Repository**: {repo_url}\n\n"
                
                # Get file tree
                files = []
                dirs = []
                code_files = []
                
                for root, dirnames, filenames in os.walk(tmpdir):
                    # Skip hidden directories
                    dirnames[:] = [d for d in dirnames if not d.startswith('.')]
                    
                    rel_root = os.path.relpath(root, tmpdir)
                    if rel_root != '.':
                        dirs.append(rel_root)
                    
                    for filename in filenames:
                        if not filename.startswith('.'):
                            rel_path = os.path.join(rel_root, filename) if rel_root != '.' else filename
                            files.append(rel_path)
                            
                            # Track code files
                            ext = os.path.splitext(filename)[1].lower()
                            if ext in ['.py', '.js', '.ts', '.java', '.cpp', '.c', '.rs', '.go']:
                                code_files.append(rel_path)
                
                output += f"## Statistics\n"
                output += f"- Total files: {len(files)}\n"
                output += f"- Directories: {len(dirs)}\n"
                output += f"- Code files: {len(code_files)}\n\n"
                
                # Key files
                output += "## Key Files\n"
                key_files = ['README.md', 'readme.md', 'README', 'setup.py', 'pyproject.toml',
                            'requirements.txt', 'package.json', 'Cargo.toml', 'Makefile',
                            'config.yaml', 'config.yml', 'main.py', 'app.py', 'train.py']
                
                for kf in key_files:
                    if kf in files or kf.lower() in [f.lower() for f in files]:
                        output += f"- ✅ {kf}\n"
                
                output += "\n## Directory Structure (top level)\n"
                for d in sorted(dirs)[:20]:
                    if '/' not in d:  # Only top-level dirs
                        output += f"📁 {d}/\n"
                
                output += "\n## Code Files\n"
                for cf in sorted(code_files)[:30]:
                    output += f"📄 {cf}\n"
                
                return output
                
        except subprocess.TimeoutExpired:
            return "Error: Repository clone timed out"
        except Exception as e:
            logger.error(f"Repo analysis error: {e}")
            return f"Error analyzing repository: {str(e)}"
