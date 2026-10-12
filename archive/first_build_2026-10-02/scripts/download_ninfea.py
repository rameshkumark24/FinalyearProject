"""
SFR Framework — NInFEA Multimodal Dataset Downloader
=====================================================
Downloads the NInFEA dataset from PhysioNet:
- Multi-lead abdominal & thoracic ECG waveforms
- Pulse Wave Doppler ultrasound sonograms (pwd_images/)
- Subject metadata and gestational age annotations
Supports targeted subject downloads (e.g. --subjects 5 or all 60).
"""

import sys
import re
import argparse
from pathlib import Path
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

repo_root = Path(__file__).resolve().parent.parent
sys.path.append(str(repo_root))

from src.utils.config import PathConfig

BASE_URL = "https://physionet.org/files/ninfea/1.0.0"

def get_ninfea_records() -> list:
    """Fetch RECORDS file listing all 60 NInFEA subjects."""
    url = f"{BASE_URL}/RECORDS"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            lines = resp.read().decode("utf-8").splitlines()
        return [l.strip() for l in lines if l.strip()]
    except Exception as e:
        print(f"[-] Could not fetch RECORDS: {e}")
        return [f"{i:02d}" for i in range(1, 61)]

def download_file(url: str, dest: Path) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp, open(dest, "wb") as f:
            f.write(resp.read())
        return True
    except Exception:
        return False

def download_subject(sub_id: str, target_dir: Path) -> tuple:
    # 1. WFDB ECG files: .hea and .dat
    sub_clean = sub_id.replace("wfdb_format_ecg_and_respiration/", "")
    ecg_base = f"{BASE_URL}/wfdb_format_ecg_and_respiration/{sub_clean}"
    
    success_files = 0
    total_files = 0
    for ext in ["hea", "dat"]:
        url = f"{ecg_base}.{ext}"
        dest = target_dir / "ecg" / f"{sub_clean}.{ext}"
        total_files += 1
        if download_file(url, dest):
            success_files += 1

    # 2. PWD Doppler image (if present for subject)
    pwd_url = f"{BASE_URL}/pwd_images/{sub_clean}.png"
    dest_pwd = target_dir / "pwd" / f"{sub_clean}.png"
    download_file(pwd_url, dest_pwd)

    return sub_clean, (success_files > 0)

def main():
    parser = argparse.ArgumentParser(description="Download NInFEA multimodal dataset.")
    parser.add_argument("--subjects", type=int, default=10, help="Number of subjects to download (default 10; use 60 for all)")
    args = parser.parse_args()

    paths = PathConfig()
    target_dir = paths.ninfea_dir
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"[*] Discovering NInFEA subjects...")
    all_records = get_ninfea_records()
    selected = all_records[:args.subjects]
    print(f"[+] Selected {len(selected)} subjects to download into {target_dir}...")

    completed = 0
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(download_subject, s, target_dir): s for s in selected}
        for future in as_completed(futures):
            sub_name, ok = future.result()
            if ok:
                completed += 1
            print(f"[+] Downloaded subject {sub_name} ({completed}/{len(selected)})")

    elapsed = time.time() - start_time
    print(f"\n[+] Successfully completed NInFEA download for {completed} subjects in {elapsed:.1f}s!")

if __name__ == "__main__":
    main()
