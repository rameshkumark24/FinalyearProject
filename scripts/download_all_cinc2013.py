"""
SFR Framework — Batch Downloader for CinC 2013 Challenge Set A
==============================================================
Downloads all 75 clinical recordings (a01 to a75) with:
- Signals (.dat)
- Header descriptions (.hea)
- Ground-truth fetal QRS annotations (.fqrs)
Directly from PhysioNet via multi-threaded streaming.
Total size: ~36 MB.
"""

import sys
import os
from pathlib import Path
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

repo_root = Path(__file__).resolve().parent.parent
sys.path.append(str(repo_root))

from src.utils.config import PathConfig

BASE_URL = "https://physionet.org/files/challenge-2013/1.0.0/set-a"

def download_file(url: str, dest_path: Path, max_retries: int = 3) -> bool:
    if dest_path.exists() and dest_path.stat().st_size > 0:
        return True  # Already cached

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
            )
            with urllib.request.urlopen(req, timeout=15) as response, open(dest_path, "wb") as out_file:
                out_file.write(response.read())
            return True
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"[-] Failed {url}: {e}")
                return False
            time.sleep(1)
    return False

def download_single_record(record_name: str, target_dir: Path) -> tuple:
    results = []
    extensions = ["dat", "hea", "fqrs"]
    for ext in extensions:
        url = f"{BASE_URL}/{record_name}.{ext}"
        dest = target_dir / f"{record_name}.{ext}"
        success = download_file(url, dest)
        results.append(success)
    return record_name, all(results)

def main():
    paths = PathConfig()
    target_dir = paths.cinc2013_dir / "set-a"
    target_dir.mkdir(parents=True, exist_ok=True)

    record_names = [f"a{i:02d}" for i in range(1, 76)]
    print(f"[*] Starting parallel download of {len(record_names)} CinC 2013 Set A records into {target_dir}...")

    total_records = len(record_names)
    completed = 0
    failed = []

    start_time = time.time()
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(download_single_record, rec, target_dir): rec
            for rec in record_names
        }
        for future in as_completed(futures):
            rec_name, success = future.result()
            if success:
                completed += 1
            else:
                failed.append(rec_name)
            if completed % 10 == 0 or completed == total_records:
                print(f"[+] Progress: {completed}/{total_records} records downloaded ({(completed/total_records)*100:.1f}%)")

    elapsed = time.time() - start_time
    print(f"\n[+] Finished download in {elapsed:.1f}s!")
    print(f"    Total successful: {completed}/{total_records}")
    if failed:
        print(f"    Failed records: {failed}")

if __name__ == "__main__":
    main()
