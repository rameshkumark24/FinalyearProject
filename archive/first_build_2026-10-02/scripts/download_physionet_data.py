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


def download_with_wfdb(db_name: str, target_dir: Path, records: list = None):
    """Attempt download using wfdb package if available."""
    try:
        import wfdb
        print(f"[*] Downloading {db_name} via wfdb into {target_dir}...")
        target_dir.mkdir(parents=True, exist_ok=True)
        wfdb.dl_database(db_name, dl_dir=str(target_dir), records=records)
        print(f"[+] Successfully downloaded {db_name} to {target_dir}")
        return True
    except Exception as e:
        print(f"[-] wfdb download for {db_name} encountered: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download open fetal cardiovascular datasets.")
    parser.add_argument("--dataset", choices=["all", "ninfea", "nifeadb", "cinc2013"], default="cinc2013")
    parser.add_argument("--limit", type=int, default=5, help="Number of records to download (default 5 for quick setup)")
    args = parser.parse_args()

    paths = PathConfig()
    paths.ensure_dirs()

    # Pre-configure record subsets for targeted downloads
    cinc_records = [f"set-a/a{i:02d}" for i in range(1, args.limit + 1)] if args.limit else None

    datasets = {
        "cinc2013": ("challenge-2013", paths.cinc2013_dir, cinc_records),
        "nifeadb": ("nifeadb", paths.nifeadb_dir, None),
        "ninfea": ("ninfea/1.0.0", paths.ninfea_dir, None),
    }

    targets = datasets.keys() if args.dataset == "all" else [args.dataset]

    for ds in targets:
        db_name, target_dir, rec_list = datasets[ds]
        success = download_with_wfdb(db_name, target_dir, records=rec_list)
        if not success:
            print(f"[i] Direct download note: {ds} files can also be retrieved via:")
            print(f"    https://physionet.org/content/{db_name}/")


if __name__ == "__main__":
    main()
