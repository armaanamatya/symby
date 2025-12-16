"""
Load environment variables from .env file
"""

from pathlib import Path
import os


def load_env():
    """Load .env file into environment variables."""
    env_file = Path(__file__).parent / ".env"

    if not env_file.exists():
        print("Warning: .env file not found")
        return

    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                key, value = line.split('=', 1)
                os.environ[key.strip()] = value.strip()

    print(f"✅ Loaded {env_file}")


# Auto-load when imported
load_env()
