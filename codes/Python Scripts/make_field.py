import os
import requests
import h5py
import numpy as np
import MAS_library as MASL
import sys

# Read snapshot number from command line argument if given
if len(sys.argv) > 1:
    snapshot_number = int(sys.argv[1])
else:
    snapshot_number = 55  # default

# === CONFIGURATION ===
api_key = "a6e92d93311aa4bd2349be55ec31c930"
num_chunks = 600
output_dir = "/gpfs/nchugh/DMCOORDS"
base_url = "https://www.tng-project.org/api/TNG300-1/files"
cutout_query = "dm=Coordinates"

# Density field params (3D)
grid = 256
BoxSize = 205.0  # Mpc/h
MAS = 'CIC'
verbose = True

# Headers for API authentication
headers = {"api-key": api_key}

# Make sure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Initialize 3D density field grid
delta = np.zeros((grid, grid, grid), dtype=np.float32)

# Function to load DM particle coordinates from chunk file
def load_particle_positions(filename):
    with h5py.File(filename, 'r') as f:
        pos = f['PartType1']['Coordinates'][:]/1000  # shape (N_particles, 3)
    return pos.astype(np.float32)

# Loop over chunks
for chunk in range(num_chunks):
    file_url = f"{base_url}/snapshot-{snapshot_number}.{chunk}.hdf5?{cutout_query}"
    output_path = os.path.join(output_dir, f"snap{snapshot_number}_{chunk}.hdf5")

    print(f"⬇️  Downloading chunk {chunk}/{num_chunks - 1}...")

    try:
        # Download chunk with API key header
        with requests.get(file_url, headers=headers, stream=True) as r:
            r.raise_for_status()
            with open(output_path, "wb") as f_out:
                for chunk_data in r.iter_content(chunk_size=8192):
                    f_out.write(chunk_data)

        # Load 3D particle positions
        pos = load_particle_positions(output_path)

        # Add particles to density grid with CIC assignment
        MASL.MA(pos, delta, BoxSize, MAS, verbose=verbose)

        # Remove chunk file to save space
        os.remove(output_path)

        print(f"✅ Processed chunk {chunk}")

    except Exception as e:
        print(f"❌ Failed on chunk {chunk}: {e}")

# Normalize density to overdensity
mean_delta = np.mean(delta, dtype=np.float64)
delta /= mean_delta
delta -= 1.0

# Save overdensity field
output_field_path = os.path.join(output_dir, f"field_256_{snapshot_number}.npy")
np.save(output_field_path, delta)

print("✅ Finished all chunks and computed 3D overdensity field.")
print(f"✅ Saved overdensity field to {output_field_path}")