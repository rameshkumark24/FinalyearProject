"""Download NInFEA (PhysioNet, v1.0.0): 60 ECG records and their 60 pulsed-wave Doppler images.

Every file is checked against PhysioNet's own SHA256SUMS.txt. A file that fails the check
is deleted and reported; the script exits with an error if anything is missing.

    python scripts/download_ninfea.py
"""

from __future__ import annotations

import hashlib
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sfr.paths import NINFEA_DIR, NINFEA_ECG_DIR, NINFEA_PWD_DIR  # noqa: E402

# The checksum list comes from physionet.org. The files themselves are fetched from PhysioNet's
# open-data mirror on Amazon S3, which is far faster; every file is verified against the list.
BASE = "https://physionet.org/files/ninfea/1.0.0"
MIRROR = "https://physionet-open.s3.amazonaws.com/ninfea/1.0.0"
HEADERS = {"User-Agent": "Mozilla/5.0 (research download; final-year project)"}


def fetch(url: str, timeout: int = 60) -> bytes:
    with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=timeout) as resp:
        return resp.read()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_one(remote: str, dest: Path, expected: str, retries: int = 5) -> tuple[str, bool, str]:
    if dest.exists() and sha256(dest) == expected:
        return remote, True, "already present"
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    last_error = ""
    for attempt in range(retries):
        try:
            request = urllib.request.Request(f"{MIRROR}/{remote}", headers=HEADERS)
            with urllib.request.urlopen(request, timeout=120) as resp, open(part, "wb") as out:
                while True:
                    chunk = resp.read(1 << 18)
                    if not chunk:
                        break
                    out.write(chunk)
            if sha256(part) == expected:
                part.replace(dest)
                return remote, True, "downloaded"
            last_error = "checksum mismatch"
            part.unlink(missing_ok=True)
        except Exception as error:  # network errors are retried
            last_error = repr(error)
            time.sleep(2 + 2 * attempt)
    return remote, False, last_error


def main() -> int:
    NINFEA_DIR.mkdir(parents=True, exist_ok=True)
    sums_file = NINFEA_DIR / "SHA256SUMS.txt"
    if not sums_file.exists():
        sums_file.write_bytes(fetch(f"{BASE}/SHA256SUMS.txt"))
    sums_text = sums_file.read_text(encoding="utf-8")
    expected = {}
    for line in sums_text.splitlines():
        parts = line.split()
        if len(parts) >= 2:
            expected[parts[1]] = parts[0].lower()

    jobs = []
    for number in range(1, 61):
        for ext in ("hea", "dat"):
            remote = f"wfdb_format_ecg_and_respiration/{number}.{ext}"
            jobs.append((remote, NINFEA_ECG_DIR / f"{number}.{ext}"))
        remote = f"pwd_images/{number}.bmp"
        jobs.append((remote, NINFEA_PWD_DIR / f"{number}.bmp"))
    for name in ("code/envelope_extraction.m", "code/ReadBinaryFile.m", "code/Interfaccia_dataset.m", "LICENSE.txt"):
        jobs.append((name, NINFEA_DIR / name))

    missing_in_manifest = [remote for remote, _ in jobs if remote not in expected]
    if missing_in_manifest:
        print("Files not listed in SHA256SUMS.txt:", missing_in_manifest)
        return 2

    failures = []
    done = 0
    start = time.time()
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(download_one, remote, dest, expected[remote]) for remote, dest in jobs]
        for future in as_completed(futures):
            remote, ok, note = future.result()
            done += 1
            if not ok:
                failures.append((remote, note))
            if done % 10 == 0 or not ok or done == len(jobs):
                print(f"[{done}/{len(jobs)}] {remote}: {'ok' if ok else 'FAILED'} ({note})", flush=True)

    print(f"Finished in {time.time() - start:.0f} s. Files verified: {len(jobs) - len(failures)} of {len(jobs)}.")
    if failures:
        for remote, note in failures:
            print("FAILED:", remote, note)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
