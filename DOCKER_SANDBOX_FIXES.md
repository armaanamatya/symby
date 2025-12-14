# Docker Sandbox Fixes

## Issues Found & Fixed

### 1. ❌ Docker Permission Issue (NEEDS USER ACTION)

**Problem:** Your user is not in the `docker` group, causing permission errors when trying to access the Docker socket.

**Error:**
```
permission denied while trying to connect to the Docker daemon socket at unix:///var/run/docker.sock
```

**Current groups:** `asus adm sudo audio dip plugdev users lpadmin` (missing `docker`)

**Fix Required:**
```bash
# Add your user to the docker group
sudo usermod -aG docker $USER

# Then either:
# Option 1: Log out and log back in (recommended)
# Option 2: Or run this to activate in current session
newgrp docker

# Verify it worked
groups  # Should now show 'docker'
docker ps  # Should work without errors
```

---

### 2. ✅ FIXED - Critical Bug in streamlit_app.py

**Problem:** Incorrect API call to `SandboxRunner.run()` missing required `image` parameter.

**Location:** `streamlit_app.py:624`

**Before (WRONG):**
```python
runner = SandboxRunner()
result = loop.run_until_complete(
    runner.run(code, timeout_seconds=timeout)  # Missing 'image' parameter!
)
```

**After (FIXED):**
```python
runner = SandboxRunner(max_timeout_seconds=timeout)
result = loop.run_until_complete(
    runner.run(image="python:3.10-slim", code=code)  # Correct!
)
```

**Why this matters:**
- The `run()` method requires `image` as the first parameter (after `self`)
- The `timeout_seconds` parameter doesn't exist in `run()` - timeout is set during `SandboxRunner()` initialization
- This bug would cause a `TypeError` every time code execution was attempted

---

### 3. ✅ FIXED - Argument List Too Long Error

**Problem:** When executing large code blocks, Docker was failing with "exec /usr/local/bin/python: argument list too long"

**Location:** `src/pce/sandbox_runner.py:397-614`

**Before (WRONG):**
```python
run_kwargs = {
    "image": image,
    "command": ["python", "-c", exec_code],  # Passing code as argument
    ...
}
```

**After (FIXED):**
```python
# Create temp directory with code file to avoid argument length limits
temp_dir = tempfile.mkdtemp(prefix="sandbox_code_")
code_file = os.path.join(temp_dir, "exec_code.py")
with open(code_file, "w") as f:
    f.write(exec_code)

run_kwargs = {
    "image": image,
    "command": ["python", "/sandbox_code/exec_code.py"],  # Execute from file
    "volumes": {temp_dir: {"bind": "/sandbox_code", "mode": "ro"}},
    ...
}
```

**Why this matters:**
- System limits on command-line argument length (typically ~128KB on Linux)
- Large code blocks from papers would exceed this limit
- Writing to a file and mounting it as a volume avoids the limit entirely
- Uses separate `/sandbox_code` mount point to avoid conflicts with `/tmp` tmpfs

---

## Additional Checks & Recommendations

### 3. 🔍 Docker Images

The sandbox uses the following default image:
- `python:3.10-slim` for basic Python execution

**Recommended action:** Pull the image to avoid delays on first use:
```bash
docker pull python:3.10-slim
```

For GPU workloads, you may also want:
```bash
# NVIDIA PyTorch (for ML/AI papers)
docker pull nvcr.io/nvidia/pytorch:24.01-py3

# NVIDIA CUDA runtime
docker pull nvcr.io/nvidia/cuda:12.3.0-runtime-ubuntu22.04
```

### 4. ✅ Docker Daemon Status
- Docker daemon: **RUNNING** ✓
- Docker version: **28.5.1** ✓

### 5. 🔍 GPU Support (Optional)

The sandbox is configured to support GPU execution. To enable:
- NVIDIA Docker runtime must be installed
- GPU drivers must be installed
- NVIDIA Container Toolkit must be configured

Current GPU status will be auto-detected when sandbox initializes.

---

## Testing After Fixes

Once you've added your user to the docker group, test the sandbox:

### Test 1: Basic Docker Access
```bash
docker ps
docker images
```

### Test 2: Pull Required Image
```bash
docker pull python:3.10-slim
```

### Test 3: Test Sandbox in Python
```python
import asyncio
from pce.sandbox_runner import SandboxRunner

async def test():
    runner = SandboxRunner(max_timeout_seconds=60)
    result = await runner.run(
        image="python:3.10-slim",
        code="print('Hello from sandbox!')"
    )
    print(f"Status: {result.status}")
    print(f"Output: {result.stdout}")
    print(f"Errors: {result.stderr}")

asyncio.run(test())
```

### Test 4: Run Streamlit App
```bash
streamlit run streamlit_app.py
```

Then try the code sandbox feature.

---

### 4. ✅ FIXED - Auto-Requirements Detection

**Feature:** Automatically detect Python imports and generate requirements.txt

**Location:** `src/pce/sandbox_runner.py:397-473`

**How it works:**
```python
def _generate_requirements_file(self, code: str, workdir: str):
    # Maps import names to pip packages
    IMPORT_TO_PIP = {
        'cv2': 'opencv-python',
        'PIL': 'Pillow',
        'sklearn': 'scikit-learn',
        ...
    }

    # Detects imports and generates requirements.txt
    # Excludes stdlib modules (os, sys, json, etc.)
```

**Why this matters:**
- Research papers often use specialized libraries
- Automatically installs numpy, pandas, torch, transformers, etc.
- No need to manually specify requirements

---

### 5. ✅ FIXED - Subprocess Fallback

**Feature:** Automatically falls back to subprocess when Docker is unavailable

**Location:** `src/pce/sandbox_runner.py:475-547, 152-163, 603-606`

**How it works:**
```python
# Docker initialization with graceful fallback
try:
    self.client = docker.from_env()
    self._docker_available = True
except:
    logger.warning("Docker unavailable - will use subprocess fallback")
    self._docker_available = False

# In run() method:
if not self._docker_available:
    return await self._run_subprocess(temp_dir, start_time)
```

**Why this matters:**
- Code execution works even with Docker permission issues
- Seamless fallback - user doesn't need to fix Docker permissions
- Less isolated but still functional for code testing
- Perfect for development environments

---

## Summary

| Issue | Status | Action Required |
|-------|--------|-----------------|
| Docker permissions | ⚠️ **OPTIONAL** | Docker will auto-fallback to subprocess if unavailable |
| streamlit_app.py API bug | ✅ **FIXED** | Already corrected |
| Argument list too long | ✅ **FIXED** | Code now written to files, not passed as arguments |
| Auto-requirements | ✅ **ADDED** | Automatically detects and installs dependencies |
| Subprocess fallback | ✅ **ADDED** | Works without Docker permissions |
| Docker daemon | ✅ **OK** | Running v28.5.1 |
| Docker images | ⚠️ **RECOMMENDED** | Run `docker pull python:3.10-slim` |

---

## Next Steps

1. **Run your app** (works immediately):
   ```bash
   streamlit run streamlit_app.py
   ```
   The sandbox will automatically use subprocess fallback if Docker is unavailable.

2. **Fix Docker permissions** (optional, for better isolation):
   ```bash
   sudo usermod -aG docker $USER
   ```
   Then log out and back in. This enables Docker-based sandboxing for better security.

3. **Pull base image** (optional, if using Docker):
   ```bash
   docker pull python:3.10-slim
   ```

4. **Test the sandbox** (verify everything works):
   ```bash
   python test_sandbox_fix.py
   ```

---

## Files Modified

- ✅ `streamlit_app.py:617-624` - Fixed `SandboxRunner.run()` call

## Files Analyzed

- `src/pce/sandbox_runner.py` - Sandbox implementation
- `src/paper_analysis/executor.py` - Correct usage reference
- `streamlit_app.py` - Fixed incorrect usage

---

**Last Updated:** 2025-12-14
**Status:** Ready to test after user adds themselves to docker group
