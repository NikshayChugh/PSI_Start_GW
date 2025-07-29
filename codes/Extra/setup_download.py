#!/usr/bin/env python3
"""
Setup script to test TNG API connection and get basic info about snapshots.
Run this first to make sure everything is working.
"""

import requests
import sys

def test_api_connection(api_key=None):
    """Test connection to TNG API."""
    base_url = "https://www.tng-project.org/api/TNG300-1"
    headers = {'api-key': api_key} if api_key else {}
    
    print("Testing TNG API connection...")
    print(f"Base URL: {base_url}")
    
    try:
        response = requests.get(base_url, headers=headers)
        response.raise_for_status()
        
        data = response.json()
        print(f"✓ Connection successful!")
        print(f"Simulation: {data.get('name', 'Unknown')}")
        print(f"Available snapshots: {data.get('num_snapshots', 'Unknown')}")
        
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"✗ Connection failed: {e}")
        return False

def get_snapshot_info(snapshot_num, api_key=None):
    """Get basic info about a specific snapshot."""
    base_url = f"https://www.tng-project.org/api/TNG300-1/snapshots/{snapshot_num}/"
    headers = {'api-key': api_key} if api_key else {}
    
    try:
        response = requests.get(base_url, headers=headers)
        response.raise_for_status()
        
        data = response.json()
        print(f"\nSnapshot {snapshot_num}:")
        print(f"  Redshift: {data.get('redshift', 'Unknown')}")
        print(f"  Scale factor: {data.get('scale_factor', 'Unknown')}")
        print(f"  Time: {data.get('time', 'Unknown')} Gyr")
        
        return data
        
    except requests.exceptions.RequestException as e:
        print(f"Error getting info for snapshot {snapshot_num}: {e}")
        return None

def check_file_structure(snapshot_num, api_key=None):
    """Check file structure for a snapshot."""
    files_url = f"https://www.tng-project.org/api/TNG300-1/files/snapshot-{snapshot_num}/"
    headers = {'api-key': api_key} if api_key else {}
    
    try:
        response = requests.get(files_url, headers=headers)
        response.raise_for_status()
        
        data = response.json()
        print(f"  File chunks: {data.get('count', 'Unknown')}")
        
        if data.get('results'):
            first_file = data['results'][0]
            print(f"  First file: {first_file.get('name', 'Unknown')}")
        
        return data.get('count', 0)
        
    except requests.exceptions.RequestException as e:
        print(f"Error checking files for snapshot {snapshot_num}: {e}")
        return 0

def main():
    """Main setup function."""
    print("TNG300-1 Dark Matter Coordinates Download Setup")
    print("=" * 50)
    
    api_key = None
    if len(sys.argv) > 1:
        api_key = sys.argv[1]
        print(f"Using provided API key: {api_key[:10]}...")
    else:
        print("No API key provided. Some features may be limited.")
        print("Get your API key from: https://www.tng-project.org/users/profile/")
    
    # Test basic connection
    if not test_api_connection(api_key):
        print("\nSetup failed. Please check your internet connection and API key.")
        return
    
    # Check target snapshots
    snapshots = [55, 65, 75, 85, 95]
    print(f"\nChecking target snapshots: {snapshots}")
    
    total_files = 0
    for snap in snapshots:
        snap_info = get_snapshot_info(snap, api_key)
        if snap_info:
            file_count = check_file_structure(snap, api_key)
            total_files += file_count
    
    print(f"\nTotal file chunks to download: {total_files}")
    
    # Estimate download size (very rough)
    if total_files > 0:
        estimated_size_gb = total_files * 0.5  # Rough estimate: 0.5 GB per chunk
        print(f"Estimated total download size: ~{estimated_size_gb:.1f} GB")
        print(f"Estimated download time: ~{estimated_size_gb * 2:.0f} minutes (assuming 4 MB/s)")
    
    print("\nSetup complete! You can now run the download script.")
    print("Commands:")
    print("  1. Submit SLURM job: sbatch download_dm_coords.slurm")
    print("  2. Or run directly: python3 download_dm_coords.py [API_KEY]")

if __name__ == "__main__":
    main()