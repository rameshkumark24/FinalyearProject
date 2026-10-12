"""
SFR Framework — Batch Downloader for NIFEADB (Non-Invasive Fetal ECG Database)
==============================================================================
Downloads the complete clinical cohort of multi-lead abdominal fetal ECG records
from PhysioNet (ARR_01 to ARR_12 abnormal cases, plus normal controls).
Total size: ~50-80 MB.
"""

import sys
import re
from pathlib import Path
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

repo_root = Path(__file__).resolve().parent.parent
sys.path.append(str(repo_root))

from src.utils.config import PathConfig

BASE_URL = "https://physionet.org/files/nifeadb/1.0.0"

def get_remote_file_list() -> list:
    """Fetch all .dat and .hea files listed in NIFEADB repository."""
    req = urllib.request.Request(
        f"{BASE_URL}/",
        headers={"User-Agent": "Mozilla/5.0"}
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        html = resp.read().decode("utf-8")
    
    files = re.findall(r'href="([^"]+\.(?:dat|hea))"', html)
    return sorted(list(set(files)))

def download_file(file_name: str, target_dir: Path, max_retries: int = 3) -> bool:
    dest = target_dir / file_name
    if dest.exists() and dest.stat().st_size > 0:
        return True

    url = f"{BASE_URL}/{file_name}"
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=20) as resp, open(dest, "wb") as f:
                f.write(resp.read())
            return True
        except Exception:
            time.sleep(1)
    return False

def main():
    paths = PathConfig()
    target_dir = paths.nifeadb_dir
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"[*] Discovering NIFEADB records on PhysioNet...")
    files = get_remote_file_list()
    print(f"[+] Discovered {len(files)} files to download.")

    completed = 0
    total = len(files)
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {executor.submit(download_file, fname, target_dir): fname for fname in files}
        for future in as_completed(futures):
            if future.result():
                completed += 1
            if completed % 10 == 0 or completed == total:
                print(f"[+] Downloaded {completed}/{total} files ({(completed/total)*100:.1f}%)")

    elapsed = time.time() - start_time
    print(f"\n[+] Successfully finished NIFEADB download in {elapsed:.1f}s!")

if __name__ == "__main__":
    main()
