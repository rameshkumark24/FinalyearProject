"""Locations of data, runs and reports. Data folders are never committed."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA = ROOT / "data"
FETALDATA = ROOT / "fetaldata"
RUNS = ROOT / "runs"
SPLITS = ROOT / "splits"
REPORTS = ROOT / "reports"
MODELS = ROOT / "models_cache"

CINC_DIR = DATA / "cinc2013" / "set-a"
NIFEADB_DIR = DATA / "nifeadb"
NINFEA_DIR = DATA / "ninfea"
NINFEA_ECG_DIR = NINFEA_DIR / "ecg"
NINFEA_PWD_DIR = NINFEA_DIR / "pwd"
CARDIUM_DIR = DATA / "cardium_clinical"
HEARTBEAT_DIR = DATA / "heartbeat"

MENDELEY_PWD_DIR = (
    FETALDATA / "Dataset_Impact_of_pulsed-wave-Doppler_velocity-envelope_extraction_technique"
)
FCU_DIR = FETALDATA / "fcu(echocardiogram)-data" / "fcu-data"
