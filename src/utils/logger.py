import logging
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, Any

def setup_logger(name: str, log_dir: str | Path = "results") -> logging.Logger:
    """Sets up a logger that outputs to console and file."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = Path(log_dir) / f"{name}_{timestamp}.log"
        
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        # File handler
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
    return logger

def log_metrics(metrics: Dict[str, Any], save_path: str | Path) -> None:
    """Saves a metrics dictionary to a JSON file.

    Handles numpy types (np.float64, np.int64, np.ndarray, etc.)
    by converting them to native Python types before serialization.
    """
    import numpy as np

    def _convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return obj

    converted = {k: _convert(v) for k, v in metrics.items()}
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(converted, f, indent=4)
