"""
SFR Framework - Configuration Utilities
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import yaml


ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT_DIR / "data"
SPLITS_DIR = ROOT_DIR / "splits"
RESULTS_DIR = ROOT_DIR / "results"
CONFIGS_DIR = ROOT_DIR / "configs"


@dataclass
class PathConfig:
    root_dir: Path = ROOT_DIR
    data_dir: Path = DATA_DIR
    splits_dir: Path = SPLITS_DIR
    results_dir: Path = RESULTS_DIR
    configs_dir: Path = CONFIGS_DIR
    
    # Dataset paths
    ninfea_dir: Path = DATA_DIR / "ninfea"
    nifeadb_dir: Path = DATA_DIR / "nifeadb"
    cinc2013_dir: Path = DATA_DIR / "cinc2013"
    heartbeat_dir: Path = DATA_DIR / "heartbeat"
    cardium_dir: Path = DATA_DIR / "cardium_clinical"

    def ensure_dirs(self):
        for path in [
            self.data_dir, self.splits_dir, self.results_dir, self.configs_dir,
            self.ninfea_dir, self.nifeadb_dir, self.cinc2013_dir,
            self.heartbeat_dir, self.cardium_dir
        ]:
            path.mkdir(parents=True, exist_ok=True)


def load_yaml_config(config_path: Path | str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
