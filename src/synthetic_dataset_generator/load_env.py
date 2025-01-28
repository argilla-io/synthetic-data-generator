import os
from pathlib import Path
from typing import Dict
from dotenv import load_dotenv

def load_env_file(env_path: Path | str) -> Dict[str, str]:
    """
    Load environment variables from a .env file.
    
    Args:
        env_path: Path to the .env file
        
    Returns:
        Dict of environment variables loaded
    
    Raises:
        FileNotFoundError: If the .env file doesn't exist
    """
    env_vars = {}
    
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                os.environ[key] = value
                env_vars[key] = value
                
    return env_vars


def init_environment():
    """Initialize environment variables."""
    root_dir = Path(__file__).parent.parent.parent
    env_file = root_dir / ".env"

    # Load .env if it exists, but do not fail if it does not exist
    if env_file.exists():
        load_dotenv(env_file)
    
    # Verify required variables are set
    if not os.getenv("HF_TOKEN"):
        raise ValueError(
            "HF_TOKEN environment variable is required. "
            "Please set it in your .env file or environment."
        ) 