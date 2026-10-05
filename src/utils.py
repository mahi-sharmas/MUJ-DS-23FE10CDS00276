"""Small shared helpers."""
from pathlib import Path

import yaml


def load_config(path="config.yaml"):
    """Read the YAML config file into a dictionary."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path):
    """Create a folder (and parents) if it doesn't exist; return it as a Path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
