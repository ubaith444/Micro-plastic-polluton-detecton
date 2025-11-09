"""Helper Functions"""

from pathlib import Path
import yaml
import json


def create_dirs(*dirs):
    """Create directories"""
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)


def save_config(config: dict, path: str):
    """Save configuration"""
    with open(path, 'w') as f:
        yaml.dump(config, f)


def load_config(path: str) -> dict:
    """Load configuration"""
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def save_results(results: dict, path: str):
    """Save results"""
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)
