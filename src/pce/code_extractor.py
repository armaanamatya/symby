"""
Code Extractor Module

Extracts code blocks from research papers (arXiv) and GitHub repositories.
Supports detection of ML frameworks and dependency extraction.
"""

from __future__ import annotations

import asyncio
import os
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Set, Tuple

import aiohttp
import fitz  # PyMuPDF
from git import Repo
from git.exc import GitCommandError


# Constants for safety limits
MAX_CODE_BLOCKS_PER_PAPER = 10
MAX_GITHUB_REPOS_PER_PAPER = 5
MAX_TOTAL_CODE_SIZE_BYTES = 1 * 1024 * 1024  # 1MB
GITHUB_CLONE_TIMEOUT_SECONDS = 60
MAX_PYTHON_FILES_FROM_REPO = 10


@dataclass
class CodeBlock:
    """Represents a single block of code extracted from a paper or repository.

    Attributes:
        language: Programming language of the code (e.g., 'python', 'r').
        code: The actual code content.
        source: Origin of the code - either 'paper' or 'github'.
        line_numbers: Optional tuple of (start_line, end_line) if available.
    """
    language: str
    code: str
    source: str  # 'paper' or 'github'
    line_numbers: Optional[Tuple[int, int]] = None

    def __post_init__(self) -> None:
        """Validate source field."""
        if self.source not in ('paper', 'github'):
            raise ValueError(f"source must be 'paper' or 'github', got '{self.source}'")


@dataclass
class CodeManifest:
    """Manifest containing all code artifacts extracted from a research paper.

    Attributes:
        paper_id: Unique identifier for the paper (e.g., arXiv ID).
        code_blocks: List of extracted code blocks.
        github_repos: List of GitHub repository URLs found in the paper.
        dependencies: Set of Python package dependencies inferred from imports.
        framework: Primary ML framework detected (pytorch|tensorflow|jax|other).
    """
    paper_id: str
    code_blocks: List[CodeBlock] = field(default_factory=list)
    github_repos: List[str] = field(default_factory=list)
    dependencies: Set[str] = field(default_factory=set)
    framework: str = "other"


class CodeExtractor:
    """Extracts code from research papers and GitHub repositories.

    This class provides async methods to:
    - Extract code blocks from arXiv papers (PDF parsing)
    - Clone and extract code from GitHub repositories
    - Infer ML framework usage from imports
    - Extract Python package dependencies

    Example:
        >>> extractor = CodeExtractor()
        >>> manifest = await extractor.extract_from_arxiv("2010.11929")
        >>> print(manifest.framework)
        'pytorch'
    """

    # Regex patterns for code extraction
    PYTHON_CODE_BLOCK_PATTERN = re.compile(
        r'```python\s*(.*?)```',
        re.DOTALL | re.IGNORECASE
    )

    GENERIC_CODE_BLOCK_PATTERN = re.compile(
        r'```\s*(.*?)```',
        re.DOTALL
    )

    # Pattern for code after "Code:" or "Algorithm:" headers
    ALGORITHM_CODE_PATTERN = re.compile(
        r'(?:Code|Algorithm|Pseudocode|Implementation)[\s:]*\n((?:[ \t]+.+\n)+)',
        re.IGNORECASE
    )

    # Pattern for indented code blocks (4+ spaces or tab)
    INDENTED_CODE_PATTERN = re.compile(
        r'(?:^|\n)((?:(?:[ ]{4,}|\t).+\n)+)',
        re.MULTILINE
    )

    # GitHub URL pattern
    GITHUB_URL_PATTERN = re.compile(
        r'(?:https?://)?(?:www\.)?github\.com/([a-zA-Z0-9_-]+)/([a-zA-Z0-9_.-]+)(?:/[^\s\)]*)?',
        re.IGNORECASE
    )

    # R code pattern
    R_CODE_PATTERN = re.compile(
        r'```r\s*(.*?)```',
        re.DOTALL | re.IGNORECASE
    )

    # Framework import patterns with priority
    FRAMEWORK_PATTERNS = [
        ('pytorch', re.compile(r'(?:^|\s)import\s+torch|from\s+torch(?:\.|\s)', re.MULTILINE)),
        ('tensorflow', re.compile(r'(?:^|\s)import\s+tensorflow|from\s+tensorflow(?:\.|\s)', re.MULTILINE)),
        ('jax', re.compile(r'(?:^|\s)import\s+jax|from\s+jax(?:\.|\s)', re.MULTILINE)),
        ('sklearn', re.compile(r'(?:^|\s)import\s+sklearn|from\s+sklearn(?:\.|\s)', re.MULTILINE)),
    ]

    # Import extraction pattern
    IMPORT_PATTERN = re.compile(
        r'(?:^|\n)\s*(?:import\s+([\w.]+)|from\s+([\w.]+)\s+import)',
        re.MULTILINE
    )

    # Common standard library modules to filter out
    STDLIB_MODULES = {
        'os', 'sys', 're', 'math', 'json', 'typing', 'collections', 'itertools',
        'functools', 'operator', 'pathlib', 'glob', 'shutil', 'tempfile', 'io',
        'abc', 'copy', 'pickle', 'datetime', 'time', 'random', 'string', 'struct',
        'logging', 'warnings', 'argparse', 'configparser', 'hashlib', 'hmac',
        'secrets', 'threading', 'multiprocessing', 'subprocess', 'asyncio',
        'concurrent', 'queue', 'socket', 'http', 'urllib', 'email', 'html', 'xml',
        'csv', 'sqlite3', 'zipfile', 'tarfile', 'gzip', 'bz2', 'lzma', 'zlib',
        'unittest', 'doctest', 'pdb', 'profile', 'cProfile', 'timeit', 'trace',
        'gc', 'inspect', 'dis', 'code', 'codeop', 'types', 'weakref', 'enum',
        'dataclasses', 'contextlib', 'textwrap', 'difflib', 'pprint', 'reprlib',
        'locale', 'gettext', 'codecs', 'unicodedata', 'decimal', 'fractions',
        'statistics', 'cmath', 'numbers', 'array', 'bisect', 'heapq', 'calendar',
        '__future__', 'builtins', 'importlib', 'pkgutil', 'modulefinder',
    }

    # Package name mappings (import name -> pip package name)
    PACKAGE_MAPPINGS = {
        'cv2': 'opencv-python',
        'PIL': 'Pillow',
        'sklearn': 'scikit-learn',
        'skimage': 'scikit-image',
        'yaml': 'pyyaml',
        'bs4': 'beautifulsoup4',
        'tf': 'tensorflow',
    }

    def __init__(self, session: Optional[aiohttp.ClientSession] = None) -> None:
        """Initialize the CodeExtractor.

        Args:
            session: Optional aiohttp ClientSession for HTTP requests.
                     If not provided, a new session will be created when needed.
        """
        self._session = session
        self._owns_session = session is None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create an aiohttp session."""
        if self._session is None:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self) -> None:
        """Close the HTTP session if we own it."""
        if self._owns_session and self._session is not None:
            await self._session.close()
            self._session = None

    async def __aenter__(self) -> 'CodeExtractor':
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()

    async def extract_from_arxiv(self, arxiv_id: str) -> CodeManifest:
        """Fetch paper from arXiv and extract code blocks.

        Downloads the PDF from arXiv, extracts text, and uses regex patterns
        to identify Python/R code blocks. Also identifies GitHub URLs mentioned
        in the paper.

        Args:
            arxiv_id: The arXiv paper ID (e.g., "2010.11929").

        Returns:
            CodeManifest containing extracted code blocks and metadata.

        Raises:
            aiohttp.ClientError: If the paper cannot be downloaded.
            ValueError: If the PDF cannot be parsed.

        Example:
            >>> extractor = CodeExtractor()
            >>> manifest = await extractor.extract_from_arxiv("2010.11929")
            >>> print(len(manifest.code_blocks))
            5
        """
        # Clean up arxiv_id (remove URL prefix if present)
        arxiv_id = arxiv_id.strip()
        if 'arxiv.org' in arxiv_id:
            arxiv_id = arxiv_id.split('/')[-1]

        manifest = CodeManifest(paper_id=arxiv_id)

        # Download PDF from arXiv
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        session = await self._get_session()

        async with session.get(pdf_url) as response:
            if response.status != 200:
                raise ValueError(f"Failed to download paper {arxiv_id}: HTTP {response.status}")
            pdf_content = await response.read()

        # Parse PDF and extract text
        text = self._extract_text_from_pdf(pdf_content)

        # Extract code blocks from text
        code_blocks = self._extract_code_blocks_from_text(text)
        manifest.code_blocks = code_blocks[:MAX_CODE_BLOCKS_PER_PAPER]

        # Extract GitHub URLs
        github_repos = self._extract_github_urls(text)
        manifest.github_repos = github_repos[:MAX_GITHUB_REPOS_PER_PAPER]

        # Infer framework and dependencies
        manifest.framework = self.infer_framework(manifest.code_blocks)
        manifest.dependencies = self.extract_dependencies(manifest.code_blocks)

        return manifest

    def _extract_text_from_pdf(self, pdf_content: bytes) -> str:
        """Extract text content from PDF bytes.

        Args:
            pdf_content: Raw PDF file content.

        Returns:
            Extracted text from all pages.
        """
        text_parts = []

        with fitz.open(stream=pdf_content, filetype="pdf") as doc:
            for page in doc:
                text_parts.append(page.get_text())

        return "\n".join(text_parts)

    def _extract_code_blocks_from_text(self, text: str) -> List[CodeBlock]:
        """Extract code blocks from paper text using regex patterns.

        Args:
            text: Full text content of the paper.

        Returns:
            List of CodeBlock objects extracted from the text.
        """
        code_blocks = []
        total_size = 0

        # Extract Python code blocks (```python ... ```)
        for match in self.PYTHON_CODE_BLOCK_PATTERN.finditer(text):
            code = match.group(1).strip()
            if code and self._is_valid_code(code, 'python'):
                if total_size + len(code) > MAX_TOTAL_CODE_SIZE_BYTES:
                    break
                code_blocks.append(CodeBlock(
                    language='python',
                    code=code,
                    source='paper'
                ))
                total_size += len(code)

        # Extract R code blocks (```r ... ```)
        for match in self.R_CODE_PATTERN.finditer(text):
            code = match.group(1).strip()
            if code and self._is_valid_code(code, 'r'):
                if total_size + len(code) > MAX_TOTAL_CODE_SIZE_BYTES:
                    break
                code_blocks.append(CodeBlock(
                    language='r',
                    code=code,
                    source='paper'
                ))
                total_size += len(code)

        # Extract code after "Algorithm:" or "Code:" headers
        for match in self.ALGORITHM_CODE_PATTERN.finditer(text):
            code = match.group(1).strip()
            if code and len(code) > 20:  # Minimum length for algorithm blocks
                if total_size + len(code) > MAX_TOTAL_CODE_SIZE_BYTES:
                    break
                # Try to detect language
                lang = self._detect_language(code)
                code_blocks.append(CodeBlock(
                    language=lang,
                    code=code,
                    source='paper'
                ))
                total_size += len(code)

        return code_blocks

    def _is_valid_code(self, code: str, language: str) -> bool:
        """Check if extracted text looks like valid code.

        Args:
            code: Code string to validate.
            language: Expected programming language.

        Returns:
            True if the code appears to be valid.
        """
        if len(code) < 10:
            return False

        if language == 'python':
            # Check for Python-like syntax
            python_indicators = ['def ', 'class ', 'import ', 'from ', 'if ', 'for ',
                                 'while ', 'return ', 'print(', '= ', ':', '#']
            return any(indicator in code for indicator in python_indicators)

        if language == 'r':
            # Check for R-like syntax
            r_indicators = ['<-', 'function(', 'library(', 'require(', 'c(',
                           'data.frame', 'matrix(', 'plot(', 'ggplot(']
            return any(indicator in code for indicator in r_indicators)

        return True

    def _detect_language(self, code: str) -> str:
        """Attempt to detect the programming language of code.

        Args:
            code: Code string to analyze.

        Returns:
            Detected language name (default: 'unknown').
        """
        # Python indicators
        python_score = sum([
            'def ' in code,
            'import ' in code,
            'class ' in code,
            ':' in code and '    ' in code,
            'print(' in code,
            'self.' in code,
        ])

        # R indicators
        r_score = sum([
            '<-' in code,
            'function(' in code,
            'library(' in code,
            'c(' in code,
            '%>%' in code,
        ])

        if python_score > r_score and python_score >= 2:
            return 'python'
        elif r_score > python_score and r_score >= 2:
            return 'r'

        return 'unknown'

    def _extract_github_urls(self, text: str) -> List[str]:
        """Extract unique GitHub repository URLs from text.

        Args:
            text: Text content to search.

        Returns:
            List of unique GitHub repository URLs.
        """
        repos = set()

        for match in self.GITHUB_URL_PATTERN.finditer(text):
            username = match.group(1)
            repo_name = match.group(2)
            # Clean up repo name (remove .git suffix if present)
            repo_name = repo_name.rstrip('.git')
            repo_url = f"https://github.com/{username}/{repo_name}"
            repos.add(repo_url)

        return list(repos)

    async def extract_from_github(self, repo_url: str) -> List[CodeBlock]:
        """Clone a GitHub repository and extract key Python code.

        Performs a shallow clone of the repository and extracts code from
        the main Python files (limited to top 10 by size). Focuses on
        models, training loops, and core logic.

        Args:
            repo_url: URL of the GitHub repository.

        Returns:
            List of CodeBlock objects extracted from the repository.

        Raises:
            GitCommandError: If the repository cannot be cloned.
            asyncio.TimeoutError: If cloning takes longer than 60 seconds.

        Example:
            >>> extractor = CodeExtractor()
            >>> blocks = await extractor.extract_from_github(
            ...     "https://github.com/google-research/vision_transformer"
            ... )
        """
        code_blocks = []
        temp_dir = None

        try:
            # Create temporary directory for clone
            temp_dir = tempfile.mkdtemp(prefix="code_extractor_")

            # Clone repository with timeout (shallow clone)
            try:
                await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: Repo.clone_from(
                            repo_url,
                            temp_dir,
                            depth=1,
                            single_branch=True
                        )
                    ),
                    timeout=GITHUB_CLONE_TIMEOUT_SECONDS
                )
            except asyncio.TimeoutError:
                raise asyncio.TimeoutError(
                    f"GitHub clone timed out after {GITHUB_CLONE_TIMEOUT_SECONDS} seconds"
                )
            except GitCommandError as e:
                raise GitCommandError(f"Failed to clone repository: {e}", 128)

            # Find Python files
            python_files = self._find_python_files(Path(temp_dir))

            # Sort by file size (descending) and take top N
            python_files.sort(key=lambda p: p.stat().st_size, reverse=True)
            python_files = python_files[:MAX_PYTHON_FILES_FROM_REPO]

            # Extract code from each file
            total_size = 0
            for py_file in python_files:
                if total_size >= MAX_TOTAL_CODE_SIZE_BYTES:
                    break

                try:
                    code = py_file.read_text(encoding='utf-8', errors='ignore')

                    # Check if this is a key file (models, training, etc.)
                    if self._is_key_file(py_file, code):
                        remaining_size = MAX_TOTAL_CODE_SIZE_BYTES - total_size
                        if len(code) > remaining_size:
                            code = code[:remaining_size]

                        # Calculate relative path from repo root
                        rel_path = py_file.relative_to(temp_dir)

                        code_blocks.append(CodeBlock(
                            language='python',
                            code=code,
                            source='github',
                            line_numbers=(1, code.count('\n') + 1)
                        ))
                        total_size += len(code)
                except Exception:
                    # Skip files that can't be read
                    continue

        finally:
            # Clean up temporary directory
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)

        return code_blocks

    def _find_python_files(self, root: Path) -> List[Path]:
        """Find all Python files in a directory tree.

        Args:
            root: Root directory to search.

        Returns:
            List of paths to Python files.
        """
        python_files = []

        for py_file in root.rglob("*.py"):
            # Skip test files, setup files, and hidden directories
            path_str = str(py_file)
            if any(skip in path_str for skip in [
                '/test', '/tests', '/.', '/venv/', '/env/',
                '__pycache__', '/site-packages/', '/dist/',
                'setup.py', 'conftest.py'
            ]):
                continue

            # Skip very small files (likely __init__.py with just imports)
            if py_file.stat().st_size < 100:
                continue

            python_files.append(py_file)

        return python_files

    def _is_key_file(self, path: Path, code: str) -> bool:
        """Check if a Python file contains key ML code.

        Args:
            path: Path to the Python file.
            code: Content of the file.

        Returns:
            True if the file appears to contain important ML code.
        """
        filename = path.name.lower()

        # Priority filenames
        key_names = ['model', 'train', 'network', 'layers', 'attention',
                     'transformer', 'encoder', 'decoder', 'loss', 'optim',
                     'dataset', 'dataloader', 'main', 'run', 'eval']

        if any(key in filename for key in key_names):
            return True

        # Check code content for key patterns
        key_patterns = [
            'class.*Model', 'class.*Network', 'class.*Layer',
            'def forward', 'def train', 'def fit',
            'nn.Module', 'tf.keras', 'nn.Linear', 'nn.Conv',
            'optim.Adam', 'optim.SGD', 'optimizer.minimize',
            'loss.backward', 'gradients', 'backprop',
        ]

        for pattern in key_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                return True

        return False

    def infer_framework(self, code_blocks: List[CodeBlock]) -> str:
        """Analyze imports to detect the primary ML framework.

        Checks code blocks for framework imports with priority:
        torch > tensorflow > jax > sklearn > other

        Args:
            code_blocks: List of code blocks to analyze.

        Returns:
            Framework name: 'pytorch', 'tensorflow', 'jax', 'sklearn', or 'other'.

        Example:
            >>> extractor = CodeExtractor()
            >>> blocks = [CodeBlock(language='python',
            ...                      code='import torch\\nmodel = torch.nn.Linear(10, 5)',
            ...                      source='paper')]
            >>> extractor.infer_framework(blocks)
            'pytorch'
        """
        # Combine all code for analysis
        all_code = "\n".join(block.code for block in code_blocks if block.language == 'python')

        if not all_code:
            return "other"

        # Check each framework in priority order
        for framework_name, pattern in self.FRAMEWORK_PATTERNS:
            if pattern.search(all_code):
                return framework_name

        return "other"

    def extract_dependencies(self, code_blocks: List[CodeBlock]) -> Set[str]:
        """Parse imports to list required pip packages.

        Extracts import statements from code blocks and converts them
        to pip-installable package names, filtering out standard library
        modules.

        Args:
            code_blocks: List of code blocks to analyze.

        Returns:
            Set of pip package names.

        Example:
            >>> extractor = CodeExtractor()
            >>> blocks = [CodeBlock(language='python',
            ...                      code='import numpy as np\\nfrom PIL import Image',
            ...                      source='paper')]
            >>> deps = extractor.extract_dependencies(blocks)
            >>> 'numpy' in deps and 'Pillow' in deps
            True
        """
        dependencies = set()

        for block in code_blocks:
            if block.language != 'python':
                continue

            for match in self.IMPORT_PATTERN.finditer(block.code):
                # Get the module name (either from 'import X' or 'from X import')
                module = match.group(1) or match.group(2)
                if module:
                    # Get the top-level package name
                    top_level = module.split('.')[0]

                    # Skip standard library modules
                    if top_level in self.STDLIB_MODULES:
                        continue

                    # Map to pip package name if needed
                    package_name = self.PACKAGE_MAPPINGS.get(top_level, top_level)
                    dependencies.add(package_name)

        return dependencies
