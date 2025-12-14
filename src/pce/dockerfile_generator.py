"""
Dockerfile Generator for Research Code

This module provides tools for generating Dockerfiles tailored to research code,
with automatic framework detection and environment configuration.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set
import re


# Known package versions for reproducibility
KNOWN_PACKAGE_VERSIONS: Dict[str, str] = {
    "torch": "2.1.0",
    "tensorflow": "2.15.0",
    "transformers": "4.36.0",
    "numpy": "1.26.0",
    "pandas": "2.1.0",
}

# Base image mappings
BASE_IMAGES: Dict[str, str] = {
    "pytorch": "nvcr.io/nvidia/pytorch:24.01-py3",
    "tensorflow": "nvcr.io/nvidia/tensorflow:24.01-tf2-py3",
    "jax": "ghcr.io/google/jax:jax-cuda12.3",
    "huggingface": "huggingface/transformers-pytorch-gpu:latest",
    "cpu-only": "python:3.11-slim",
}

# Framework detection patterns
FRAMEWORK_PATTERNS: Dict[str, List[str]] = {
    "pytorch": ["torch", "torchvision", "torchaudio", "pytorch"],
    "tensorflow": ["tensorflow", "keras", "tf"],
    "jax": ["jax", "flax", "optax"],
    "huggingface": ["transformers", "datasets", "tokenizers", "huggingface_hub"],
    "cpu-only": ["sklearn", "scikit-learn", "pandas", "numpy", "scipy"],
}

# Dangerous patterns to check in Dockerfile validation
DANGEROUS_PATTERNS: List[Tuple[str, str]] = [
    (r"curl\s+.*\|\s*bash", "Piping curl to bash is insecure"),
    (r"wget\s+.*\|\s*bash", "Piping wget to bash is insecure"),
    (r"curl\s+.*\|\s*sh", "Piping curl to sh is insecure"),
    (r"wget\s+.*\|\s*sh", "Piping wget to sh is insecure"),
    (r"chmod\s+777", "chmod 777 grants excessive permissions"),
    (r"--trusted-host", "Using --trusted-host bypasses SSL verification"),
]


@dataclass
class CodeManifest:
    """
    Represents metadata about research code for Docker environment generation.

    Attributes:
        imports: List of Python import statements or package names found in code.
        python_files: List of Python file paths in the project.
        requirements_file: Optional path to an existing requirements.txt file.
        entry_point: The main script to run (defaults to 'main.py').
        extra_apt_packages: Additional system packages required.
        extra_env_vars: Additional environment variables needed.
    """
    imports: List[str] = field(default_factory=list)
    python_files: List[str] = field(default_factory=list)
    requirements_file: Optional[str] = None
    entry_point: str = "main.py"
    extra_apt_packages: List[str] = field(default_factory=list)
    extra_env_vars: Dict[str, str] = field(default_factory=dict)


@dataclass
class DockerConfig:
    """
    Configuration for Docker environment generation.

    Attributes:
        base_image: The base Docker image to use.
        python_version: The Python version for the environment.
        gpu_required: Whether GPU support is needed.
        pip_packages: List of Python packages to install via pip.
        apt_packages: List of system packages to install via apt-get.
        env_vars: Environment variables to set in the container.
    """
    base_image: str
    python_version: str
    gpu_required: bool
    pip_packages: List[str] = field(default_factory=list)
    apt_packages: List[str] = field(default_factory=list)
    env_vars: Dict[str, str] = field(default_factory=dict)


class DockerfileGenerator:
    """
    Generates Dockerfiles for research code with automatic framework detection.

    This class analyzes code manifests to determine the appropriate Docker
    environment, generates Dockerfiles, and validates them for common issues.

    Example:
        >>> generator = DockerfileGenerator()
        >>> manifest = CodeManifest(imports=["torch", "transformers"])
        >>> config = generator.analyze_dependencies(manifest)
        >>> dockerfile = generator.generate_dockerfile(config)
    """

    def __init__(
        self,
        known_versions: Optional[Dict[str, str]] = None,
        base_images: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Initialize the DockerfileGenerator.

        Args:
            known_versions: Optional custom mapping of package names to versions.
            base_images: Optional custom mapping of framework names to base images.
        """
        self.known_versions = known_versions or KNOWN_PACKAGE_VERSIONS.copy()
        self.base_images = base_images or BASE_IMAGES.copy()

    def analyze_dependencies(self, code_manifest: CodeManifest) -> DockerConfig:
        """
        Analyze code to determine environment needs.

        Framework detection:
        - torch/torchvision → nvcr.io/nvidia/pytorch:24.01-py3
        - tensorflow/keras → nvcr.io/nvidia/tensorflow:24.01-tf2-py3
        - jax/flax → custom JAX CUDA image
        - transformers → HuggingFace base
        - sklearn/pandas → CPU-only base

        Args:
            code_manifest: A CodeManifest containing information about the code.

        Returns:
            DockerConfig with appropriate settings for the detected framework.
        """
        imports_lower = {imp.lower() for imp in code_manifest.imports}

        # Detect framework and determine base image
        detected_framework = self._detect_framework(imports_lower)
        base_image = self.base_images.get(detected_framework, self.base_images["cpu-only"])

        # Determine if GPU is required
        gpu_required = detected_framework in ["pytorch", "tensorflow", "jax", "huggingface"]

        # Determine Python version
        python_version = self._determine_python_version(detected_framework)

        # Collect pip packages
        pip_packages = self._collect_pip_packages(code_manifest.imports)

        # Collect apt packages
        apt_packages = self._collect_apt_packages(imports_lower, code_manifest.extra_apt_packages)

        # Collect environment variables
        env_vars = self._collect_env_vars(detected_framework, code_manifest.extra_env_vars)

        return DockerConfig(
            base_image=base_image,
            python_version=python_version,
            gpu_required=gpu_required,
            pip_packages=pip_packages,
            apt_packages=apt_packages,
            env_vars=env_vars,
        )

    def _detect_framework(self, imports_lower: Set[str]) -> str:
        """
        Detect the primary ML framework from imports.

        Priority order: pytorch > tensorflow > jax > huggingface > cpu-only

        Args:
            imports_lower: Set of lowercased import names.

        Returns:
            The detected framework name.
        """
        # Check in priority order
        for framework in ["pytorch", "tensorflow", "jax", "huggingface", "cpu-only"]:
            patterns = FRAMEWORK_PATTERNS.get(framework, [])
            for pattern in patterns:
                if pattern in imports_lower:
                    return framework

        return "cpu-only"

    def _determine_python_version(self, framework: str) -> str:
        """
        Determine the appropriate Python version for the framework.

        Args:
            framework: The detected framework name.

        Returns:
            Python version string.
        """
        # Most modern ML frameworks work well with Python 3.11
        # NVIDIA containers typically come with a specific Python version
        if framework in ["pytorch", "tensorflow"]:
            return "3.10"  # NVIDIA containers often use 3.10
        elif framework == "jax":
            return "3.11"
        else:
            return "3.11"

    def _collect_pip_packages(self, imports: List[str]) -> List[str]:
        """
        Collect pip packages from imports, adding known versions where available.

        Args:
            imports: List of import names.

        Returns:
            List of pip package specifications.
        """
        packages = []
        seen = set()

        for imp in imports:
            # Normalize package name
            package_name = self._normalize_package_name(imp)

            if package_name in seen:
                continue
            seen.add(package_name)

            # Add version if known
            if package_name in self.known_versions:
                packages.append(f"{package_name}=={self.known_versions[package_name]}")
            else:
                packages.append(package_name)

        return sorted(packages)

    def _normalize_package_name(self, import_name: str) -> str:
        """
        Normalize an import name to a pip package name.

        Args:
            import_name: The import name from code.

        Returns:
            Normalized pip package name.
        """
        # Common mappings from import names to pip package names
        import_to_pip = {
            "sklearn": "scikit-learn",
            "cv2": "opencv-python",
            "PIL": "Pillow",
            "yaml": "PyYAML",
            "bs4": "beautifulsoup4",
        }

        return import_to_pip.get(import_name, import_name.lower())

    def _collect_apt_packages(
        self,
        imports_lower: Set[str],
        extra_packages: List[str],
    ) -> List[str]:
        """
        Collect required apt packages based on detected imports.

        Args:
            imports_lower: Set of lowercased import names.
            extra_packages: Additional apt packages specified by user.

        Returns:
            List of apt package names.
        """
        packages = set(extra_packages)

        # Common apt dependencies for Python packages
        apt_dependencies = {
            "cv2": ["libgl1-mesa-glx", "libglib2.0-0"],
            "opencv": ["libgl1-mesa-glx", "libglib2.0-0"],
            "pillow": ["libjpeg-dev", "zlib1g-dev"],
            "pil": ["libjpeg-dev", "zlib1g-dev"],
            "lxml": ["libxml2-dev", "libxslt1-dev"],
            "psycopg2": ["libpq-dev"],
            "mysql": ["libmysqlclient-dev"],
        }

        for imp in imports_lower:
            if imp in apt_dependencies:
                packages.update(apt_dependencies[imp])

        return sorted(packages)

    def _collect_env_vars(
        self,
        framework: str,
        extra_env_vars: Dict[str, str],
    ) -> Dict[str, str]:
        """
        Collect environment variables for the container.

        Args:
            framework: The detected framework name.
            extra_env_vars: Additional environment variables specified by user.

        Returns:
            Dictionary of environment variables.
        """
        env_vars = {
            "PYTHONUNBUFFERED": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        }

        # Framework-specific environment variables
        if framework == "pytorch":
            env_vars["TORCH_HOME"] = "/app/.cache/torch"
        elif framework == "tensorflow":
            env_vars["TF_CPP_MIN_LOG_LEVEL"] = "2"
        elif framework == "huggingface":
            env_vars["HF_HOME"] = "/app/.cache/huggingface"
            env_vars["TRANSFORMERS_CACHE"] = "/app/.cache/huggingface"

        # Add extra environment variables
        env_vars.update(extra_env_vars)

        return env_vars

    def generate_dockerfile(self, config: DockerConfig) -> str:
        """
        Generate Dockerfile content from configuration.

        Template:
        FROM {base_image}
        WORKDIR /app

        # System packages
        RUN apt-get update && apt-get install -y {apt_packages}

        # Python packages
        COPY requirements.txt .
        RUN pip install --no-cache-dir -r requirements.txt

        # Copy code
        COPY . .

        # Default command
        CMD ["python", "main.py"]

        Args:
            config: DockerConfig with environment settings.

        Returns:
            Complete Dockerfile content as a string.
        """
        lines = []

        # Base image
        lines.append(f"FROM {config.base_image}")
        lines.append("")

        # Working directory
        lines.append("WORKDIR /app")
        lines.append("")

        # Environment variables
        if config.env_vars:
            lines.append("# Environment variables")
            for key, value in sorted(config.env_vars.items()):
                lines.append(f"ENV {key}={value}")
            lines.append("")

        # System packages
        if config.apt_packages:
            lines.append("# System packages")
            packages_str = " \\\n    ".join(sorted(config.apt_packages))
            lines.append(
                f"RUN apt-get update && apt-get install -y --no-install-recommends \\\n"
                f"    {packages_str} \\\n"
                f"    && rm -rf /var/lib/apt/lists/*"
            )
            lines.append("")

        # Python packages
        lines.append("# Python packages")
        lines.append("COPY requirements.txt .")
        lines.append("RUN pip install --no-cache-dir -r requirements.txt")
        lines.append("")

        # Copy code
        lines.append("# Copy code")
        lines.append("COPY . .")
        lines.append("")

        # Default command
        lines.append("# Default command")
        lines.append('CMD ["python", "main.py"]')
        lines.append("")

        return "\n".join(lines)

    def generate_requirements_txt(self, packages: List[str]) -> str:
        """
        Generate requirements.txt with pinned versions where known.

        Args:
            packages: List of package names or specifications.

        Returns:
            requirements.txt content as a string.
        """
        lines = ["# Auto-generated requirements.txt", "# Pinned versions for reproducibility", ""]

        processed_packages = []
        for package in packages:
            # If package already has a version, use as-is
            if "==" in package or ">=" in package or "<=" in package:
                processed_packages.append(package)
            else:
                # Try to add known version
                package_lower = package.lower()
                if package_lower in self.known_versions:
                    processed_packages.append(f"{package_lower}=={self.known_versions[package_lower]}")
                else:
                    processed_packages.append(package)

        # Sort alphabetically for consistency
        processed_packages = sorted(set(processed_packages), key=lambda x: x.lower())
        lines.extend(processed_packages)
        lines.append("")

        return "\n".join(lines)

    def validate_dockerfile(self, dockerfile: str) -> Tuple[bool, List[str]]:
        """
        Validate Dockerfile syntax and check for common issues.

        Checks for common issues:
        - Missing FROM
        - Invalid commands
        - Dangerous patterns (curl | bash)

        Args:
            dockerfile: The Dockerfile content to validate.

        Returns:
            Tuple of (is_valid, list_of_issues) where is_valid is True if
            no critical issues were found.
        """
        issues: List[str] = []
        lines = dockerfile.strip().split("\n")

        # Check for empty Dockerfile
        if not lines or not dockerfile.strip():
            issues.append("Dockerfile is empty")
            return False, issues

        # Check for FROM instruction
        has_from = False
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("FROM "):
                has_from = True
                break
            # Skip comments and empty lines
            if stripped and not stripped.startswith("#"):
                break

        if not has_from:
            issues.append("Dockerfile must start with a FROM instruction")

        # Valid Dockerfile instructions
        valid_instructions = {
            "FROM", "RUN", "CMD", "LABEL", "MAINTAINER", "EXPOSE", "ENV",
            "ADD", "COPY", "ENTRYPOINT", "VOLUME", "USER", "WORKDIR",
            "ARG", "ONBUILD", "STOPSIGNAL", "HEALTHCHECK", "SHELL",
        }

        # Check each line for valid instructions
        continuation = False
        for line_num, line in enumerate(lines, 1):
            stripped = line.strip()

            # Skip empty lines and comments
            if not stripped or stripped.startswith("#"):
                continuation = False
                continue

            # Skip continuation lines
            if continuation:
                continuation = stripped.endswith("\\")
                continue

            # Check if line starts with valid instruction
            parts = stripped.split(None, 1)
            if parts:
                instruction = parts[0].upper()
                if instruction not in valid_instructions:
                    issues.append(
                        f"Line {line_num}: Unknown instruction '{parts[0]}'"
                    )

            continuation = stripped.endswith("\\")

        # Check for dangerous patterns
        dockerfile_lower = dockerfile.lower()
        for pattern, message in DANGEROUS_PATTERNS:
            if re.search(pattern, dockerfile_lower):
                issues.append(f"Security warning: {message}")

        # Determine if valid (only FROM missing is critical)
        is_valid = has_from and not any(
            "must start with" in issue or "is empty" in issue
            for issue in issues
        )

        return is_valid, issues
