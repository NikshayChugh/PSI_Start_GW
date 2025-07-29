#!/usr/bin/env python3
"""
TNG300-1 DM Coordinates Downloader (Per-snapshot, chunked, resumable)
Downloads only coords_XXX.npy files per snapshot without combining or postprocessing.
"""

import os
import sys
import time
import requests
import numpy as np
import h5py
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import argparse

TNG_API_KEY = os.environ.get("TNG_API_KEY", "")
BASE_URL = 'http://www.tng-project.org/api/'
HEADERS = {"api-key": TNG_API_KEY}
SNAPSHOT_IDS = [55, 65, 75, 85, 95, 99]

def get_snapshot_file_count(snapshot_id):
    # Assumes fixed 600 files per snapshot
    return 600

def download_file_chunked(url, local_filename, chunk_size=1024*1024*50):
    """Download file with resume support."""
    resume_byte_pos = 0
    if os.path.exists(local_filename):
        resume_byte_pos = os.path.getsize(local_filename)
    
    headers = HEADERS.copy()
    if resume_byte_pos > 0:
        headers['Range'] = f'bytes={resume_byte_pos}-'
    
    for attempt in range(5):
        try:
            r = requests.get(url, headers=headers, stream=True, timeout=60)
            r.raise_for_status()
            mode = 'ab' if resume_byte_pos > 0 else 'wb'
            with open(local_filename, mode) as f:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
            return True
        except Exception as e:
            print(f"Retry {attempt+1}/5 failed: {str(e)}")
            time.sleep(2**attempt)
    return False

def download_single_file(snapshot_id, file_num, total_files, outdir):
    file_url = f"{BASE_URL}TNG300-1/files/snapshot-{snapshot_id}.{file_num}.hdf5"
    hdf5_file = os.path.join(outdir, f"snapshot-{snapshot_id}.{file_num}.hdf5")
    coords_file = os.path.join(outdir, f"coords_{file_num:03d}.npy")

    if os.path.exists(coords_file):
        print(f"[{snapshot_id}][{file_num:03d}] Skipping (already exists).")
        return file_num, True

    print(f"[{snapshot_id}][{file_num:03d}] Downloading...")
    success = download_file_chunked(file_url, hdf5_file)
    if not success:
        return file_num, False

    try:
        with h5py.File(hdf5_file, 'r') as f:
            coords = f['PartType1']['Coordinates'][:] / 1000.0
            np.save(coords_file, coords)
            print(f"[{snapshot_id}][{file_num:03d}] Saved {len(coords):,} coords.")
    except Exception as e:
        print(f"[{snapshot_id}][{file_num:03d}] HDF5 read error: {e}")
        os.remove(hdf5_file)
        return file_num, False

    os.remove(hdf5_file)
    return file_num, True

def process_snapshot(snapshot_id, workers, base_dir):
    print(f"\n--- Processing snapshot {snapshot_id} ---")
    outdir = os.path.join(base_dir, f"snap-{snapshot_id}")
    os.makedirs(outdir, exist_ok=True)

    total_files = get_snapshot_file_count(snapshot_id)
    success, fail = [], []

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(download_single_file, snapshot_id, i, total_files, outdir): i
            for i in range(total_files)
        }
        for f in as_completed(futures):
            file_num, ok = f.result()
            if ok:
                success.append(file_num)
            else:
                fail.append(file_num)

    print(f"\nSnapshot {snapshot_id} summary:")
    print(f"  Success: {len(success)} / {total_files}")
    if fail:
        print(f"  Failed: {fail}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--base-dir', type=str, default='/gpfs/nchugh/dmcoordssnaps')
    parser.add_argument('--snapshots', type=int, nargs='*', default=SNAPSHOT_IDS)
    args = parser.parse_args()

    for snap in args.snapshots:
        process_snapshot(snap, args.workers, args.base_dir)

if __name__ == "__main__":
    main()
