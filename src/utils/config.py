import yaml
from pathlib import Path
from typing import Dict, Any

def load_config(config_path: str | Path) -> Dict[str, Any]:
    """Loads a YAML configuration file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config or {}

def save_config(config: Dict[str, Any], save_path: str | Path) -> None:
    """Saves a dictionary to a YAML configuration file."""
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
