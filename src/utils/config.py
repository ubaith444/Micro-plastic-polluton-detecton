"""Configuration Management"""

import yaml
from pathlib import Path


class Config:
    """Configuration handler"""
    
    def __init__(self, config_path: str = None):
        self.config = {}
        if config_path:
            self.load(config_path)
    
    def load(self, config_path: str):
        """Load configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def save(self, config_path: str):
        """Save configuration"""
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f)
    
    def get(self, key: str, default=None):
        """Get value"""
        return self.config.get(key, default)
    
    def __getitem__(self, key):
        return self.config[key]
