"""
SFR Framework — Download PhysioNet Open Datasets
================================================
Downloads:
1. NInFEA (Non-Invasive Multimodal Fetal ECG-Doppler Dataset)
   https://physionet.org/content/ninfea/1.0.0/
2. NIFEADB (Non-Invasive Fetal ECG Database)
   https://physionet.org/content/nifeadb/1.0.0/
3. CinC Challenge 2013 (Set A - Noninvasive Fetal ECG with annotated peaks)
   https://physionet.org/content/challenge-2013/1.0.0/
"""

import sys
from pathlib import Path
import argparse

# Add repo root to path
repo_root = Path(__file__).resolve().parent.parent
sys.path.append(str(repo_root))

from src.utils.config import PathConfig


def download_with_wfdb(db_name: str, target_dir: Path):
    """Attempt download using wfdb package if available."""
    try:
        import wfdb
        print(f"[*] Downloading {db_name} via wfdb into {target_dir}...")
        target_dir.mkdir(parents=True, exist_ok=True)
        wfdb.dl_database(db_name, dl_dir=str(target_dir))
        print(f"[+] Successfully downloaded {db_name} to {target_dir}")
        return True
    except Exception as e:
        print(f"[-] wfdb download for {db_name} encountered: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download open fetal cardiovascular datasets.")
    parser.add_argument("--dataset", choices=["all", "ninfea", "nifeadb", "cinc2013"], default="all")
    args = parser.parse_args()

    paths = PathConfig()
    paths.ensure_dirs()

    datasets = {
        "ninfea": ("ninfea/1.0.0", paths.ninfea_dir),
        "nifeadb": ("nifeadb/1.0.0", paths.nifeadb_dir),
        "cinc2013": ("challenge-2013/1.0.0", paths.cinc2013_dir),
    }

    targets = datasets.keys() if args.dataset == "all" else [args.dataset]

    for ds in targets:
        db_name, target_dir = datasets[ds]
        success = download_with_wfdb(db_name, target_dir)
        if not success:
            print(f"[i] To download {ds} manually or via wget/curl:")
            print(f"    wget -r -N -c -np https://physionet.org/files/{db_name}/ -P {target_dir}")


if __name__ == "__main__":
    main()
