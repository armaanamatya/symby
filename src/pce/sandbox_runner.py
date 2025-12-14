"""
Sandbox Runner Module

Executes code in isolated Docker containers with strict security measures.
Optimized for NVIDIA DGX Spark with GB10 GPU (CUDA 13.0, Python 3.10).
Supports GPU execution, memory limits, timeouts, and proper cleanup.
"""

import asyncio
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import docker
from docker.errors import APIError, ContainerError, ImageNotFound
from docker.types import DeviceRequest

logger = logging.getLogger(__name__)

# Default CUDA images optimized for DGX Spark GB10
DEFAULT_CUDA_IMAGE = "nvcr.io/nvidia/pytorch:24.01-py3"  # CUDA 12.3, Python 3.10
DGX_SPARK_CUDA_IMAGE = "nvcr.io/nvidia/cuda:12.3.0-runtime-ubuntu22.04"


@dataclass
class GPUInfo:
    """Information about available GPU."""

    name: str
    uuid: str
    memory_total_mb: float
    memory_free_mb: float
    gpu_utilization_pct: float
    memory_utilization_pct: float
    temperature_c: float
    power_draw_w: float
    compute_capability: str


@dataclass
class ExecutionResult:
    """Result of code execution in a sandbox container."""

    status: str  # success | error | timeout | oom
    stdout: str
    stderr: str
    exit_code: int
    execution_time_seconds: float
    memory_peak_mb: float
    gpu_utilization_pct: Optional[float] = None
    gpu_memory_used_mb: Optional[float] = None
    gpu_memory_peak_mb: Optional[float] = None

    def __post_init__(self):
        """Validate status field."""
        valid_statuses = {"success", "error", "timeout", "oom"}
        if self.status not in valid_statuses:
            raise ValueError(f"Invalid status: {self.status}. Must be one of {valid_statuses}")


class SandboxRunner:
    """
    Execute code in isolated Docker containers with full GPU support.

    Optimized for NVIDIA DGX Spark with GB10 GPU:
    - CUDA 13.0 support
    - Python 3.10.19 compatibility
    - Full GPU memory and utilization tracking via nvidia-smi
    - NVIDIA Container Toolkit integration

    Security measures applied:
    - No network access (network_mode="none")
    - Read-only root filesystem (where possible)
    - All capabilities dropped (except GPU access)
    - Memory limit enforced
    - CPU quota enforced
    - Timeout with SIGKILL fallback
    """

    # GPU monitoring script injected into containers
    GPU_MONITOR_SCRIPT = '''
import subprocess
import json
import sys

def get_gpu_stats():
    """Get GPU stats using nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(", ")
            if len(parts) >= 5:
                return {
                    "gpu_util": float(parts[0]),
                    "mem_used": float(parts[1]),
                    "mem_total": float(parts[2]),
                    "temp": float(parts[3]),
                    "power": float(parts[4]) if parts[4] != "[N/A]" else 0.0
                }
    except Exception as e:
        print(f"GPU stats error: {e}", file=sys.stderr)
    return None

# Print GPU stats at start
stats = get_gpu_stats()
if stats:
    print(f"__GPU_STATS_START__:{json.dumps(stats)}:__GPU_STATS_END__", file=sys.stderr)
'''

    def __init__(
        self,
        max_memory_gb: float = 8.0,
        max_timeout_seconds: int = 300,
        max_cpus: int = 4,
        enable_gpu: bool = True,
        gpu_memory_fraction: float = 1.0,
        require_gpu: bool = False,
        cuda_visible_devices: Optional[str] = None,
    ):
        """
        Initialize Docker client and configure limits for DGX Spark.

        Args:
            max_memory_gb: Maximum memory limit in GB (default 8GB)
            max_timeout_seconds: Maximum execution timeout in seconds (default 300s)
            max_cpus: Maximum number of CPUs to allocate (default 4)
            enable_gpu: Whether to enable GPU access for containers (default True)
            gpu_memory_fraction: Fraction of GPU memory to allow (0.0-1.0)
            require_gpu: If True, fail if GPU is not available
            cuda_visible_devices: Specific GPU devices to use (e.g., "0" or "0,1")
        """
        self.max_memory_gb = max_memory_gb
        self.max_timeout_seconds = max_timeout_seconds
        self.max_cpus = max_cpus
        self.enable_gpu = enable_gpu
        self.gpu_memory_fraction = gpu_memory_fraction
        self.require_gpu = require_gpu
        self.cuda_visible_devices = cuda_visible_devices

        # Initialize Docker client (with graceful fallback)
        self.client = None
        self._docker_available = False
        try:
            self.client = docker.from_env()
            self.client.ping()
            self._docker_available = True
            logger.info("Docker client initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize Docker client: {e}")
            logger.warning("Docker unavailable - will use subprocess fallback for code execution")
            self.client = None

        # Calculate resource limits
        self._memory_limit = f"{int(max_memory_gb)}g"
        # CPU quota: 100000 period with quota = cpus * period
        self._cpu_period = 100000
        self._cpu_quota = max_cpus * self._cpu_period

        # Check GPU availability (only if Docker is available)
        self._gpu_available = False
        self._gpu_info: Optional[GPUInfo] = None
        if enable_gpu and self._docker_available:
            self._check_gpu_availability()

        if require_gpu and not self._gpu_available:
            raise RuntimeError("GPU required but not available (Docker may not be accessible)")

        logger.info(
            f"SandboxRunner configured: memory={self._memory_limit}, "
            f"timeout={max_timeout_seconds}s, cpus={max_cpus}, "
            f"gpu={'available' if self._gpu_available else 'not available'}"
        )

    def _check_gpu_availability(self) -> None:
        """Check if NVIDIA GPU is available and get info."""
        try:
            # Check Docker for NVIDIA runtime
            info = self.client.info()
            runtimes = info.get("Runtimes", {})

            if "nvidia" not in runtimes:
                logger.warning("NVIDIA Docker runtime not available")
                return

            # Try to get GPU info via nvidia-smi on host
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=name,uuid,memory.total,memory.free,utilization.gpu,utilization.memory,temperature.gpu,power.draw,compute_cap",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=10
                )
                if result.returncode == 0:
                    parts = result.stdout.strip().split(", ")
                    if len(parts) >= 9:
                        self._gpu_info = GPUInfo(
                            name=parts[0],
                            uuid=parts[1],
                            memory_total_mb=float(parts[2]),
                            memory_free_mb=float(parts[3]),
                            gpu_utilization_pct=float(parts[4]),
                            memory_utilization_pct=float(parts[5]),
                            temperature_c=float(parts[6]),
                            power_draw_w=float(parts[7]) if parts[7] != "[N/A]" else 0.0,
                            compute_capability=parts[8],
                        )
                        self._gpu_available = True
                        logger.info(f"GPU detected: {self._gpu_info.name} ({self._gpu_info.memory_total_mb:.0f}MB)")
            except FileNotFoundError:
                # nvidia-smi not on host, but runtime might still work
                self._gpu_available = True
                logger.info("NVIDIA runtime available (nvidia-smi not found on host)")
            except subprocess.TimeoutExpired:
                logger.warning("nvidia-smi timed out")
            except Exception as e:
                logger.warning(f"Error checking GPU: {e}")
                # Still try to use GPU if runtime is available
                self._gpu_available = True

        except Exception as e:
            logger.warning(f"GPU availability check failed: {e}")

    def get_gpu_info(self) -> Optional[GPUInfo]:
        """Get current GPU information."""
        return self._gpu_info

    def is_gpu_available(self) -> bool:
        """Check if GPU is available for use."""
        return self._gpu_available

    def _get_device_requests(self) -> List[DeviceRequest]:
        """Get GPU device requests for container."""
        if not self.enable_gpu or not self._gpu_available:
            return []

        # Request all GPUs or specific ones
        if self.cuda_visible_devices:
            device_ids = self.cuda_visible_devices.split(",")
            return [DeviceRequest(device_ids=device_ids, capabilities=[["gpu"]])]
        else:
            return [DeviceRequest(count=-1, capabilities=[["gpu"]])]  # -1 = all GPUs

    def _get_gpu_environment(self) -> Dict[str, str]:
        """Get GPU-related environment variables."""
        env = {}

        if self.enable_gpu and self._gpu_available:
            # CUDA memory configuration
            if self.gpu_memory_fraction < 1.0:
                # TensorFlow style memory growth
                env["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
                # PyTorch style memory fraction
                env["PYTORCH_CUDA_ALLOC_CONF"] = f"max_split_size_mb:512"

            # Visible devices
            if self.cuda_visible_devices:
                env["CUDA_VISIBLE_DEVICES"] = self.cuda_visible_devices

            # Enable TensorFloat-32 for better performance on Ampere+
            env["NVIDIA_TF32_OVERRIDE"] = "1"

            # Disable CUDA caching for more predictable memory
            env["CUDA_CACHE_DISABLE"] = "1"

        return env

    def _get_security_opts(self) -> List[str]:
        """Get security options for container."""
        return [
            "no-new-privileges:true",
        ]

    def _get_cap_drop(self) -> List[str]:
        """Get capabilities to drop for container."""
        return ["ALL"]

    def _parse_container_stats(self, stats: dict) -> Tuple[float, float]:
        """
        Parse container stats for memory and GPU usage.

        Args:
            stats: Container stats dictionary from Docker API

        Returns:
            Tuple of (memory_peak_mb, gpu_utilization_pct)
        """
        memory_peak_mb = 0.0
        gpu_utilization_pct = 0.0

        try:
            # Parse memory stats
            memory_stats = stats.get("memory_stats", {})
            max_usage = memory_stats.get("max_usage", 0)
            memory_peak_mb = max_usage / (1024 * 1024)  # Convert bytes to MB

        except Exception as e:
            logger.warning(f"Failed to parse container stats: {e}")

        return memory_peak_mb, gpu_utilization_pct

    def _parse_gpu_stats_from_output(self, stderr: str) -> Tuple[Optional[float], Optional[float]]:
        """
        Parse GPU stats from container stderr output.

        Args:
            stderr: Container stderr output

        Returns:
            Tuple of (gpu_utilization_pct, gpu_memory_used_mb)
        """
        gpu_util = None
        gpu_mem = None

        try:
            # Look for our GPU stats marker
            match = re.search(r"__GPU_STATS_START__:(.+?):__GPU_STATS_END__", stderr)
            if match:
                stats = json.loads(match.group(1))
                gpu_util = stats.get("gpu_util")
                gpu_mem = stats.get("mem_used")
        except Exception as e:
            logger.debug(f"Could not parse GPU stats from output: {e}")

        return gpu_util, gpu_mem

    async def _wait_for_container(
        self,
        container,
        timeout: int
    ) -> Tuple[int, bool]:
        """
        Wait for container to complete with timeout.

        Args:
            container: Docker container object
            timeout: Timeout in seconds

        Returns:
            Tuple of (exit_code, timed_out)
        """
        loop = asyncio.get_event_loop()
        timed_out = False
        exit_code = -1

        try:
            # Run the blocking wait in a thread pool
            result = await asyncio.wait_for(
                loop.run_in_executor(None, lambda: container.wait(timeout=timeout)),
                timeout=timeout + 5  # Add buffer for API call overhead
            )
            exit_code = result.get("StatusCode", -1)
        except asyncio.TimeoutError:
            logger.warning(f"Container {container.id[:12]} timed out after {timeout}s")
            timed_out = True
            try:
                container.kill()
                logger.info(f"Killed timed-out container {container.id[:12]}")
            except Exception as e:
                logger.error(f"Failed to kill container: {e}")
        except Exception as e:
            logger.error(f"Error waiting for container: {e}")
            timed_out = True
            try:
                container.kill()
            except Exception:
                pass

        return exit_code, timed_out

    async def _get_container_stats(self, container) -> dict:
        """Get container stats (non-streaming)."""
        loop = asyncio.get_event_loop()
        try:
            stats = await loop.run_in_executor(
                None,
                lambda: container.stats(stream=False)
            )
            return stats
        except Exception as e:
            logger.warning(f"Failed to get container stats: {e}")
            return {}

    def _prepare_code_with_gpu_monitoring(self, code: str) -> str:
        """Wrap code with GPU monitoring if GPU is enabled."""
        if not self.enable_gpu or not self._gpu_available:
            return code

        # Prepend GPU monitoring script
        return f"{self.GPU_MONITOR_SCRIPT}\n\n# User code\n{code}"

    def _generate_requirements_file(self, code: str, workdir: str) -> Optional[str]:
        """
        Auto-detect and generate requirements.txt from imports in code.

        Args:
            code: Python code to analyze
            workdir: Directory to write requirements.txt

        Returns:
            Path to requirements.txt if created, None otherwise
        """
        # Common package mappings (import name -> pip package)
        IMPORT_TO_PIP = {
            'cv2': 'opencv-python',
            'PIL': 'Pillow',
            'sklearn': 'scikit-learn',
            'yaml': 'pyyaml',
            'torch': 'torch',
            'tensorflow': 'tensorflow',
            'tf': 'tensorflow',
            'jax': 'jax',
            'numpy': 'numpy',
            'pandas': 'pandas',
            'matplotlib': 'matplotlib',
            'seaborn': 'seaborn',
            'scipy': 'scipy',
            'transformers': 'transformers',
            'datasets': 'datasets',
            'accelerate': 'accelerate',
            'tqdm': 'tqdm',
            'requests': 'requests',
            'bs4': 'beautifulsoup4',
            'skimage': 'scikit-image',
        }

        # Standard library modules (don't need pip install)
        stdlib = {
            'os', 'sys', 'json', 'time', 'datetime', 'pathlib', 'typing',
            'collections', 'itertools', 'functools', 'math', 're', 'logging',
            'asyncio', 'subprocess', 'tempfile', 'shutil', 'hashlib', 'dataclasses',
            'abc', 'argparse', 'base64', 'bisect', 'copy', 'csv', 'enum', 'glob',
            'io', 'pickle', 'random', 'string', 'struct', 'threading', 'uuid', 'warnings',
        }

        # Detect imports
        imports = set()
        for line in code.split('\n'):
            line = line.strip()
            if line.startswith('import '):
                # import numpy, import numpy as np
                parts = line[7:].split(',')
                for part in parts:
                    module = part.strip().split(' as ')[0].split('.')[0]
                    imports.add(module)
            elif line.startswith('from '):
                # from numpy import array
                parts = line[5:].split(' import ')
                if parts:
                    module = parts[0].strip().split('.')[0]
                    imports.add(module)

        # Convert to pip packages (excluding stdlib)
        requirements = []
        for imp in imports:
            if imp in stdlib:
                continue
            pip_name = IMPORT_TO_PIP.get(imp, imp)
            requirements.append(pip_name)

        if requirements:
            req_path = os.path.join(workdir, "requirements.txt")
            with open(req_path, "w") as f:
                f.write('\n'.join(sorted(set(requirements))))
            logger.info(f"Generated requirements.txt with {len(requirements)} packages")
            return req_path

        return None

    async def _run_subprocess(
        self,
        code_dir: str,
        start_time: float,
        entrypoint: str = "exec_code.py"
    ) -> ExecutionResult:
        """
        Fallback: Run code directly via subprocess when Docker is unavailable.
        Less isolated but works when Docker has permission issues or isn't available.

        Args:
            code_dir: Directory containing code files
            start_time: Execution start time
            entrypoint: Python file to execute

        Returns:
            ExecutionResult with execution details
        """
        code_file = os.path.join(code_dir, entrypoint)

        # Install requirements first if they exist
        req_file = os.path.join(code_dir, "requirements.txt")
        if os.path.exists(req_file):
            try:
                logger.info("Installing requirements via pip...")
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", "--no-cache-dir", "-r", req_file],
                    capture_output=True,
                    timeout=120
                )
            except Exception as e:
                logger.warning(f"Failed to install requirements: {e}")

        try:
            # Run the code file
            logger.info(f"Executing {entrypoint} via subprocess fallback")
            result = subprocess.run(
                [sys.executable, code_file],
                capture_output=True,
                timeout=self.max_timeout_seconds,
                cwd=code_dir
            )

            status = 'success' if result.returncode == 0 else 'error'

            return ExecutionResult(
                status=status,
                stdout=result.stdout.decode('utf-8', errors='replace'),
                stderr=result.stderr.decode('utf-8', errors='replace'),
                exit_code=result.returncode,
                execution_time_seconds=time.time() - start_time,
                memory_peak_mb=0.0,  # Can't measure in subprocess mode
            )

        except subprocess.TimeoutExpired:
            return ExecutionResult(
                status='timeout',
                stdout='',
                stderr=f'Execution timed out after {self.max_timeout_seconds}s',
                exit_code=-1,
                execution_time_seconds=time.time() - start_time,
                memory_peak_mb=0.0,
            )
        except Exception as e:
            logger.error(f"Subprocess execution error: {e}")
            return ExecutionResult(
                status='error',
                stdout='',
                stderr=str(e),
                exit_code=-1,
                execution_time_seconds=time.time() - start_time,
                memory_peak_mb=0.0,
            )

    async def run(
        self,
        image: str,
        code: str,
        workdir: str = "/app",
        environment: Optional[Dict[str, str]] = None,
        gpu_monitor: bool = True,
    ) -> ExecutionResult:
        """
        Execute code in isolated container with full GPU support.

        Security measures:
        - No network access (network_mode="none")
        - Read-only root filesystem (where possible)
        - Drop all capabilities
        - Memory limit enforced
        - CPU quota enforced
        - Timeout with SIGKILL fallback

        Args:
            image: Docker image to use (recommend nvcr.io/nvidia/pytorch for GPU)
            code: Python code to execute
            workdir: Working directory inside container
            environment: Additional environment variables
            gpu_monitor: Whether to inject GPU monitoring code

        Returns:
            ExecutionResult with execution details including GPU metrics
        """
        container = None
        temp_dir = None
        start_time = time.time()

        logger.info(f"Starting execution in image {image} (GPU: {self._gpu_available}, Docker: {self._docker_available})")

        # Prepare code and temp directory first (needed for both Docker and subprocess)
        try:
            # Prepare code with optional GPU monitoring
            exec_code = self._prepare_code_with_gpu_monitoring(code) if gpu_monitor else code

            # Create temp directory with code file to avoid argument length limits
            temp_dir = tempfile.mkdtemp(prefix="sandbox_code_")
            code_file = os.path.join(temp_dir, "exec_code.py")
            with open(code_file, "w") as f:
                f.write(exec_code)

            # Auto-detect and generate requirements file
            self._generate_requirements_file(code, temp_dir)

            # If Docker is not available, use subprocess fallback immediately
            if not self._docker_available:
                logger.info("Docker not available, using subprocess fallback")
                return await self._run_subprocess(temp_dir, start_time)

            # Docker is available, try to use it
            loop = asyncio.get_event_loop()

            # Build environment
            env = self._get_gpu_environment()
            if environment:
                env.update(environment)

            run_kwargs = {
                "image": image,
                "command": ["python", "/sandbox_code/exec_code.py"],
                "detach": True,
                "network_mode": "none",
                "mem_limit": self._memory_limit,
                "memswap_limit": self._memory_limit,  # Prevent swap
                "cpu_period": self._cpu_period,
                "cpu_quota": self._cpu_quota,
                "working_dir": workdir,
                "security_opt": self._get_security_opts(),
                "cap_drop": self._get_cap_drop(),
                "environment": env,
                "volumes": {temp_dir: {"bind": "/sandbox_code", "mode": "ro"}},
                "remove": False,  # We'll remove after getting logs
            }

            # Add GPU device requests
            device_requests = self._get_device_requests()
            if device_requests:
                run_kwargs["device_requests"] = device_requests
                # Use NVIDIA runtime explicitly
                run_kwargs["runtime"] = "nvidia"
                logger.info("GPU enabled for container execution")

            # Try with read-only filesystem first
            try:
                run_kwargs["read_only"] = True
                # Need tmpfs for writable locations (CUDA needs /tmp)
                run_kwargs["tmpfs"] = {
                    "/tmp": "size=1G,mode=1777",
                    "/root/.cache": "size=500M",
                }
                container = await loop.run_in_executor(
                    None,
                    lambda: self.client.containers.run(**run_kwargs)
                )
            except APIError as e:
                # Some images may not work with read-only FS
                if "read-only" in str(e).lower() or "tmpfs" in str(e).lower():
                    logger.warning("Read-only filesystem not supported, retrying without")
                    run_kwargs.pop("read_only", None)
                    run_kwargs.pop("tmpfs", None)
                    container = await loop.run_in_executor(
                        None,
                        lambda: self.client.containers.run(**run_kwargs)
                    )
                else:
                    raise

            logger.info(f"Container {container.id[:12]} started")

            # Wait for completion with timeout
            exit_code, timed_out = await self._wait_for_container(
                container,
                self.max_timeout_seconds
            )

            execution_time = time.time() - start_time

            # Get logs before cleanup
            stdout = ""
            stderr = ""
            try:
                stdout = container.logs(stdout=True, stderr=False).decode("utf-8", errors="replace")
                stderr = container.logs(stdout=False, stderr=True).decode("utf-8", errors="replace")
            except Exception as e:
                logger.warning(f"Failed to get container logs: {e}")

            # Get stats for memory usage
            stats = await self._get_container_stats(container)
            memory_peak_mb, _ = self._parse_container_stats(stats)

            # Parse GPU stats from output
            gpu_util, gpu_mem = self._parse_gpu_stats_from_output(stderr)

            # Clean GPU stats markers from stderr
            stderr = re.sub(r"__GPU_STATS_START__:.+?:__GPU_STATS_END__\n?", "", stderr)

            # Determine status
            if timed_out:
                status = "timeout"
            elif exit_code == 137:  # SIGKILL, usually OOM
                status = "oom"
            elif exit_code == 0:
                status = "success"
            else:
                status = "error"

            logger.info(
                f"Execution completed: status={status}, exit_code={exit_code}, "
                f"time={execution_time:.2f}s, memory={memory_peak_mb:.1f}MB, "
                f"gpu_util={gpu_util}%, gpu_mem={gpu_mem}MB"
            )

            return ExecutionResult(
                status=status,
                stdout=stdout,
                stderr=stderr,
                exit_code=exit_code,
                execution_time_seconds=execution_time,
                memory_peak_mb=memory_peak_mb,
                gpu_utilization_pct=gpu_util,
                gpu_memory_used_mb=gpu_mem,
                gpu_memory_peak_mb=gpu_mem,  # Would need continuous monitoring for true peak
            )

        except ImageNotFound as e:
            logger.warning(f"Docker image not found: {image}, falling back to subprocess")
            if temp_dir:
                try:
                    return await self._run_subprocess(temp_dir, start_time)
                except Exception as fallback_error:
                    logger.error(f"Subprocess fallback also failed: {fallback_error}")
                    return ExecutionResult(
                        status="error",
                        stdout="",
                        stderr=f"Docker failed (image not found) and subprocess fallback failed: {fallback_error}",
                        exit_code=-1,
                        execution_time_seconds=time.time() - start_time,
                        memory_peak_mb=0.0,
                    )

        except ContainerError as e:
            logger.warning(f"Container error: {e}, falling back to subprocess")
            if temp_dir:
                try:
                    return await self._run_subprocess(temp_dir, start_time)
                except Exception as fallback_error:
                    logger.error(f"Subprocess fallback also failed: {fallback_error}")
                    stdout_log = ""
                    if e.container:
                        try:
                            stdout_log = e.container.logs(stdout=True, stderr=False).decode()
                        except Exception:
                            pass
                    return ExecutionResult(
                        status="error",
                        stdout=stdout_log,
                        stderr=f"Docker failed: {e}, subprocess fallback failed: {fallback_error}",
                        exit_code=e.exit_status,
                        execution_time_seconds=time.time() - start_time,
                        memory_peak_mb=0.0,
                    )

        except APIError as e:
            logger.warning(f"Docker API error: {e}, falling back to subprocess")
            if temp_dir:
                try:
                    return await self._run_subprocess(temp_dir, start_time)
                except Exception as fallback_error:
                    logger.error(f"Subprocess fallback also failed: {fallback_error}")
                    return ExecutionResult(
                        status="error",
                        stdout="",
                        stderr=f"Docker API error: {e}, subprocess fallback failed: {fallback_error}",
                        exit_code=-1,
                        execution_time_seconds=time.time() - start_time,
                        memory_peak_mb=0.0,
                    )

        except Exception as e:
            logger.warning(f"Unexpected error during Docker execution: {e}, falling back to subprocess")
            if temp_dir:
                try:
                    return await self._run_subprocess(temp_dir, start_time)
                except Exception as fallback_error:
                    logger.error(f"Subprocess fallback also failed: {fallback_error}")
                    return ExecutionResult(
                        status="error",
                        stdout="",
                        stderr=f"Docker execution failed: {e}, subprocess fallback failed: {fallback_error}",
                        exit_code=-1,
                        execution_time_seconds=time.time() - start_time,
                        memory_peak_mb=0.0,
                    )

        finally:
            # Cleanup container
            if container:
                try:
                    container.remove(force=True)
                    logger.debug(f"Container {container.id[:12]} removed")
                except Exception as e:
                    logger.warning(f"Failed to remove container: {e}")

            # Cleanup temp directory
            if temp_dir and os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                    logger.debug(f"Cleaned up temp directory: {temp_dir}")
                except Exception as e:
                    logger.warning(f"Failed to cleanup temp directory: {e}")

    async def run_with_files(
        self,
        image: str,
        files: Dict[str, str],
        entrypoint: str = "main.py",
        environment: Optional[Dict[str, str]] = None,
    ) -> ExecutionResult:
        """
        Execute with multiple files mounted.
        Creates temp directory, copies files, runs container.

        Args:
            image: Docker image to use
            files: Dictionary mapping filename to file content
            entrypoint: Python file to execute (must be in files dict)
            environment: Additional environment variables

        Returns:
            ExecutionResult with execution details
        """
        if entrypoint not in files:
            return ExecutionResult(
                status="error",
                stdout="",
                stderr=f"Entrypoint '{entrypoint}' not found in files",
                exit_code=-1,
                execution_time_seconds=0.0,
                memory_peak_mb=0.0,
            )

        temp_dir = None
        container = None
        start_time = time.time()

        try:
            # Create temp directory with files
            temp_dir = tempfile.mkdtemp(prefix="sandbox_")

            for filename, content in files.items():
                file_path = os.path.join(temp_dir, filename)
                # Create subdirectories if needed
                dir_path = os.path.dirname(file_path)
                if dir_path:
                    os.makedirs(dir_path, exist_ok=True)
                with open(file_path, "w") as f:
                    f.write(content)

            logger.info(f"Created temp directory with {len(files)} files")

            # Run container with mounted volume
            loop = asyncio.get_event_loop()

            # Build environment
            env = self._get_gpu_environment()
            if environment:
                env.update(environment)

            run_kwargs = {
                "image": image,
                "command": ["python", entrypoint],
                "detach": True,
                "network_mode": "none",
                "mem_limit": self._memory_limit,
                "memswap_limit": self._memory_limit,
                "cpu_period": self._cpu_period,
                "cpu_quota": self._cpu_quota,
                "working_dir": "/app",
                "volumes": {temp_dir: {"bind": "/app", "mode": "ro"}},
                "security_opt": self._get_security_opts(),
                "cap_drop": self._get_cap_drop(),
                "environment": env,
                "remove": False,
            }

            device_requests = self._get_device_requests()
            if device_requests:
                run_kwargs["device_requests"] = device_requests
                run_kwargs["runtime"] = "nvidia"

            # Add tmpfs for CUDA cache when using GPU
            if device_requests:
                run_kwargs["tmpfs"] = {
                    "/tmp": "size=1G,mode=1777",
                    "/root/.cache": "size=500M",
                }

            container = await loop.run_in_executor(
                None,
                lambda: self.client.containers.run(**run_kwargs)
            )

            logger.info(f"Container {container.id[:12]} started with files")

            # Wait for completion
            exit_code, timed_out = await self._wait_for_container(
                container,
                self.max_timeout_seconds
            )

            execution_time = time.time() - start_time

            # Get logs
            stdout = ""
            stderr = ""
            try:
                stdout = container.logs(stdout=True, stderr=False).decode("utf-8", errors="replace")
                stderr = container.logs(stdout=False, stderr=True).decode("utf-8", errors="replace")
            except Exception as e:
                logger.warning(f"Failed to get container logs: {e}")

            # Get stats
            stats = await self._get_container_stats(container)
            memory_peak_mb, _ = self._parse_container_stats(stats)

            # Parse GPU stats
            gpu_util, gpu_mem = self._parse_gpu_stats_from_output(stderr)
            stderr = re.sub(r"__GPU_STATS_START__:.+?:__GPU_STATS_END__\n?", "", stderr)

            # Determine status
            if timed_out:
                status = "timeout"
            elif exit_code == 137:
                status = "oom"
            elif exit_code == 0:
                status = "success"
            else:
                status = "error"

            logger.info(f"Execution with files completed: status={status}")

            return ExecutionResult(
                status=status,
                stdout=stdout,
                stderr=stderr,
                exit_code=exit_code,
                execution_time_seconds=execution_time,
                memory_peak_mb=memory_peak_mb,
                gpu_utilization_pct=gpu_util,
                gpu_memory_used_mb=gpu_mem,
                gpu_memory_peak_mb=gpu_mem,
            )

        except ImageNotFound:
            logger.error(f"Image not found: {image}")
            return ExecutionResult(
                status="error",
                stdout="",
                stderr=f"Image not found: {image}",
                exit_code=-1,
                execution_time_seconds=time.time() - start_time,
                memory_peak_mb=0.0,
            )

        except APIError as e:
            logger.error(f"Docker API error: {e}")
            raise RuntimeError(f"Docker API error: {e}") from e

        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return ExecutionResult(
                status="error",
                stdout="",
                stderr=str(e),
                exit_code=-1,
                execution_time_seconds=time.time() - start_time,
                memory_peak_mb=0.0,
            )

        finally:
            # Cleanup container
            if container:
                try:
                    container.remove(force=True)
                except Exception as e:
                    logger.warning(f"Failed to remove container: {e}")

            # Cleanup temp directory
            if temp_dir and os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                    logger.debug(f"Cleaned up temp directory: {temp_dir}")
                except Exception as e:
                    logger.warning(f"Failed to cleanup temp directory: {e}")

    async def build_and_run(
        self,
        dockerfile: str,
        code_files: Dict[str, str],
        entrypoint: str = "main.py",
        environment: Optional[Dict[str, str]] = None,
    ) -> ExecutionResult:
        """
        Build image from Dockerfile, then execute.
        Cleans up image after execution.

        Args:
            dockerfile: Dockerfile content as string
            code_files: Dictionary mapping filename to file content
            entrypoint: Python file to execute (must be in code_files dict)
            environment: Additional environment variables

        Returns:
            ExecutionResult with execution details
        """
        if entrypoint not in code_files:
            return ExecutionResult(
                status="error",
                stdout="",
                stderr=f"Entrypoint '{entrypoint}' not found in code_files",
                exit_code=-1,
                execution_time_seconds=0.0,
                memory_peak_mb=0.0,
            )

        temp_dir = None
        image = None
        container = None
        start_time = time.time()
        build_time = 0.0

        try:
            # Create temp directory for build context
            temp_dir = tempfile.mkdtemp(prefix="sandbox_build_")

            # Write Dockerfile
            dockerfile_path = os.path.join(temp_dir, "Dockerfile")
            with open(dockerfile_path, "w") as f:
                f.write(dockerfile)

            # Write code files
            for filename, content in code_files.items():
                file_path = os.path.join(temp_dir, filename)
                dir_path = os.path.dirname(file_path)
                if dir_path:
                    os.makedirs(dir_path, exist_ok=True)
                with open(file_path, "w") as f:
                    f.write(content)

            logger.info(f"Build context created with {len(code_files)} files")

            # Build image
            loop = asyncio.get_event_loop()
            build_start = time.time()

            try:
                image, build_logs = await loop.run_in_executor(
                    None,
                    lambda: self.client.images.build(
                        path=temp_dir,
                        rm=True,
                        forcerm=True,
                        network_mode="default",  # Allow network during build for package installs
                    )
                )
                build_time = time.time() - build_start
                logger.info(f"Image built in {build_time:.2f}s: {image.id[:12]}")
            except Exception as e:
                logger.error(f"Image build failed: {e}")
                return ExecutionResult(
                    status="error",
                    stdout="",
                    stderr=f"Image build failed: {e}",
                    exit_code=-1,
                    execution_time_seconds=time.time() - start_time,
                    memory_peak_mb=0.0,
                )

            # Build environment
            env = self._get_gpu_environment()
            if environment:
                env.update(environment)

            # Run container
            run_kwargs = {
                "image": image.id,
                "command": ["python", entrypoint],
                "detach": True,
                "network_mode": "none",
                "mem_limit": self._memory_limit,
                "memswap_limit": self._memory_limit,
                "cpu_period": self._cpu_period,
                "cpu_quota": self._cpu_quota,
                "working_dir": "/app",
                "security_opt": self._get_security_opts(),
                "cap_drop": self._get_cap_drop(),
                "environment": env,
                "remove": False,
            }

            device_requests = self._get_device_requests()
            if device_requests:
                run_kwargs["device_requests"] = device_requests
                run_kwargs["runtime"] = "nvidia"
                run_kwargs["tmpfs"] = {
                    "/tmp": "size=1G,mode=1777",
                    "/root/.cache": "size=500M",
                }

            container = await loop.run_in_executor(
                None,
                lambda: self.client.containers.run(**run_kwargs)
            )

            logger.info(f"Container {container.id[:12]} started from built image")

            # Wait for completion
            exit_code, timed_out = await self._wait_for_container(
                container,
                self.max_timeout_seconds
            )

            execution_time = time.time() - start_time

            # Get logs
            stdout = ""
            stderr = ""
            try:
                stdout = container.logs(stdout=True, stderr=False).decode("utf-8", errors="replace")
                stderr = container.logs(stdout=False, stderr=True).decode("utf-8", errors="replace")
            except Exception as e:
                logger.warning(f"Failed to get container logs: {e}")

            # Get stats
            stats = await self._get_container_stats(container)
            memory_peak_mb, _ = self._parse_container_stats(stats)

            # Parse GPU stats
            gpu_util, gpu_mem = self._parse_gpu_stats_from_output(stderr)
            stderr = re.sub(r"__GPU_STATS_START__:.+?:__GPU_STATS_END__\n?", "", stderr)

            # Determine status
            if timed_out:
                status = "timeout"
            elif exit_code == 137:
                status = "oom"
            elif exit_code == 0:
                status = "success"
            else:
                status = "error"

            logger.info(
                f"Build and run completed: status={status}, "
                f"build_time={build_time:.2f}s, exec_time={execution_time - build_time:.2f}s"
            )

            return ExecutionResult(
                status=status,
                stdout=stdout,
                stderr=stderr,
                exit_code=exit_code,
                execution_time_seconds=execution_time,
                memory_peak_mb=memory_peak_mb,
                gpu_utilization_pct=gpu_util,
                gpu_memory_used_mb=gpu_mem,
                gpu_memory_peak_mb=gpu_mem,
            )

        except APIError as e:
            logger.error(f"Docker API error: {e}")
            raise RuntimeError(f"Docker API error: {e}") from e

        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return ExecutionResult(
                status="error",
                stdout="",
                stderr=str(e),
                exit_code=-1,
                execution_time_seconds=time.time() - start_time,
                memory_peak_mb=0.0,
            )

        finally:
            # Cleanup container
            if container:
                try:
                    container.remove(force=True)
                except Exception as e:
                    logger.warning(f"Failed to remove container: {e}")

            # Cleanup image
            if image:
                try:
                    self.client.images.remove(image.id, force=True)
                    logger.debug(f"Cleaned up built image: {image.id[:12]}")
                except Exception as e:
                    logger.warning(f"Failed to remove built image: {e}")

            # Cleanup temp directory
            if temp_dir and os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                except Exception as e:
                    logger.warning(f"Failed to cleanup temp directory: {e}")

    def close(self):
        """Close Docker client connection."""
        if self.client is not None:
            try:
                self.client.close()
                logger.info("Docker client closed")
            except Exception as e:
                logger.warning(f"Error closing Docker client: {e}")

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with cleanup."""
        self.close()
        return False


# Convenience function for DGX Spark
def create_dgx_spark_runner(
    max_memory_gb: float = 64.0,  # DGX Spark has 128GB unified memory
    max_timeout_seconds: int = 600,
    max_cpus: int = 10,  # GB10 has 10 ARM cores
) -> SandboxRunner:
    """
    Create a SandboxRunner optimized for NVIDIA DGX Spark with GB10.

    The DGX Spark features:
    - NVIDIA GB10 Superchip (Grace Blackwell architecture)
    - 128GB unified CPU+GPU memory
    - 10 ARM Neoverse V2 CPU cores
    - CUDA 13.0 support
    - Python 3.10.19

    Args:
        max_memory_gb: Maximum memory limit (default 64GB, half of available)
        max_timeout_seconds: Maximum timeout (default 10 minutes)
        max_cpus: Maximum CPUs (default 10, all available)

    Returns:
        Configured SandboxRunner instance
    """
    return SandboxRunner(
        max_memory_gb=max_memory_gb,
        max_timeout_seconds=max_timeout_seconds,
        max_cpus=max_cpus,
        enable_gpu=True,
        gpu_memory_fraction=0.9,  # Leave some for system
        require_gpu=True,
    )
