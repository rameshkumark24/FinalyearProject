"""
Test downloading a single record from PhysioNet Challenge 2013 (Set A)
"""

import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
sys.path.append(str(repo_root))

from src.utils.config import PathConfig

def test_single_record_download():
    paths = PathConfig()
    paths.ensure_dirs()
    target_dir = paths.cinc2013_dir
    print(f"[*] Testing download of record a01 into {target_dir}...")

    import wfdb
    try:
        # Challenge 2013 Set A records are located in set-a/
        wfdb.dl_database("challenge-2013", dl_dir=str(target_dir), records=["set-a/a01"])
        print("[+] Record set-a/a01 downloaded successfully!")
        
        # Test loading it
        record = wfdb.rdrecord(str(target_dir / "set-a" / "a01"))
        print(f"[+] Loaded record a01 successfully! Shape: {record.p_signal.shape}, fs: {record.fs}")
    except Exception as e:
        print(f"[-] Download failed: {e}")

if __name__ == "__main__":
    test_single_record_download()
