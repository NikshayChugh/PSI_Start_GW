#!/usr/bin/env python3
"""
TNG300-1 Snapshot 99 Density Field Calculator
Computes density field from coords_*.npy files using Cloud-in-Cell method
Incrementally accumulates results for batches of files specified by start and end indices.
"""

import numpy as np
import os
import glob
import argparse

# Configuration
BASE_DIR = "/gpfs/nchugh/dmcoordssnaps/snap-99"

def standardize_coords(coords, ngrid, xmax=None, xmin=None):
    """
    Standardize 3D coordinates to [0, ngrid) for each axis.

    Parameters:
        coords: ndarray of shape (N, 3)
        ngrid: int or tuple of ints (nx, ny, nz)
        xmax, xmin: Optional max/min for each axis (each should be shape (3,))

    Returns:
        coords_std: ndarray of shape (N, 3)
    """
    coords = np.asarray(coords)
    if isinstance(ngrid, int):
        ngrid = (ngrid,) * 3

    coords_std = np.empty_like(coords)

    for i in range(3):
        if xmax is None: 
            x_max = coords[:, i].max()
        else: 
            x_max = xmax[i]
        if xmin is None: 
            x_min = coords[:, i].min()
        else: 
            x_min = xmin[i]

        if x_max != x_min:
            coords_std[:, i] = (coords[:, i] - x_min) / (x_max - x_min) * ngrid[i]
        else:
            # Place particle(s) at center of grid
            coords_std[:, i] = ngrid[i] / 2.0

    return coords_std

def cic_density_field(coords, nx, ny=None, nz=None, mass=None, wraparound=False):
    """
    Compute density field using Cloud-in-Cell method.
    
    Parameters:
        coords: ndarray of shape (N, 3) - particle coordinates (standardized to grid)
        nx, ny, nz: grid dimensions
        mass: particle masses (if None, assumes equal mass particles)
        wraparound: periodic boundary conditions
    
    Returns:
        field: ndarray of shape (nx, ny, nz) - density field
    """
    if ny is None:
        ny = nx
    if nz is None:
        nz = nx
        
    coords = np.asarray(coords)
    nparticles = coords.shape[0]
    
    if mass is None:
        mass = np.ones(nparticles, dtype=np.float32)
    else:
        mass = np.asarray(mass, dtype=np.float32)
    
    print(f'Computing CIC density field for {nparticles:,} particles on {nx}x{ny}x{nz} grid')
    
    def findweights(pos, ngrid):
        """Calculate CIC weights."""
        if wraparound:
            ngp = np.fix(pos + 0.5)
        else:
            ngp = np.fix(pos) + 0.5

        distngp = ngp - pos
        weight2 = np.abs(distngp)
        weight1 = 1.0 - weight2

        if wraparound:
            ind1 = ngp
        else:
            ind1 = ngp - 0.5
        ind1 = ind1.astype(int)

        ind2 = ind1 - 1
        ind2[distngp < 0] += 2

        bad = (ind2 == -1)
        ind2[bad] = ngrid - 1
        if not wraparound:
            weight2[bad] = 0.0
            
        bad = (ind2 == ngrid)
        ind2[bad] = 0
        if not wraparound:
            weight2[bad] = 0.0

        if wraparound:
            ind1[ind1 == ngrid] = 0

        return dict(weight=weight1, ind=ind1), dict(weight=weight2, ind=ind2)

    def update_field(field, a, b, c, mass_values):
        """Update field with CIC contributions."""
        nxny = nx * ny
        indices = (a['ind'] + b['ind'] * nx + c['ind'] * nxny) % (nx * ny * nz)
        # Ensure indices are within bounds
        weights = a['weight'] * b['weight'] * c['weight']
        contribution = weights * mass_values
        
        # Use numpy's bincount for efficient accumulation
        field += np.bincount(indices, weights=contribution, minlength=nx*ny*nz)

    # Get CIC weights for each dimension
    x1, x2 = findweights(coords[:, 0], nx)
    y1, y2 = findweights(coords[:, 1], ny)
    z1, z2 = findweights(coords[:, 2], nz)
    # Initialize field
    field = np.zeros(nx * ny * nz, dtype=np.float32)
    # Add all 8 contributions (2^3 for 3D CIC)
    print("Adding CIC contributions...")
    update_field(field, x1, y1, z1, mass)
    update_field(field, x2, y1, z1, mass)
    update_field(field, x1, y2, z1, mass)
    update_field(field, x2, y2, z1, mass)
    update_field(field, x1, y1, z2, mass)
    update_field(field, x2, y1, z2, mass)
    update_field(field, x1, y2, z2, mass)
    update_field(field, x2, y2, z2, mass)
    return field.reshape((nx, ny, nz))

def compute_density_field(grid_size=128, start=0, end=None):
    """
    Compute and accumulate the density field from coords_*.npy files
    in the range [start, end].
    """
    coord_pattern = os.path.join(BASE_DIR, "coords_*.npy")
    coord_files = sorted(glob.glob(coord_pattern))
    
    if end is None or end >= len(coord_files):
        end = len(coord_files) - 1
    
    if start < 0 or start > end:
        raise ValueError(f"Invalid start ({start}) and end ({end}) indices.")
    
    files_to_process = coord_files[start:end+1]
    if not files_to_process:
        print(f"No files found in range {start} to {end}. Exiting.")
        return
    
    print(f"Processing files from index {start} to {end} ({len(files_to_process)} files)")
    
    # Load coordinates for all files in this batch
    coords_list = []
    total_particles = 0
    for filepath in files_to_process:
        coords = np.load(filepath)
        coords_list.append(coords)
        total_particles += coords.shape[0]
        print(f"Loaded {os.path.basename(filepath)} with {coords.shape[0]:,} particles")
    coords_all = np.vstack(coords_list)
    print(f"Total particles in batch: {total_particles:,}")
    
    # Calculate coordinate bounds for this batch
    coord_min = coords_all.min(axis=0)
    coord_max = coords_all.max(axis=0)
    box_size = coord_max - coord_min
    
    print(f"Coordinate bounds for batch:")
    print(f"  X: [{coord_min[0]:.3f}, {coord_max[0]:.3f}] (size: {box_size[0]:.3f})")
    print(f"  Y: [{coord_min[1]:.3f}, {coord_max[1]:.3f}] (size: {box_size[1]:.3f})")
    print(f"  Z: [{coord_min[2]:.3f}, {coord_max[2]:.3f}] (size: {box_size[2]:.3f})")
    
    # Standardize coordinates to grid
    coords_std = standardize_coords(coords_all, grid_size, xmax=coord_max, xmin=coord_min)
    coords_std = np.clip(coords_std, 0, grid_size - 1e-6)
    
    # Compute density for this batch
    density_batch = cic_density_field(coords_std, grid_size, grid_size, grid_size)
    
    # Load existing density field if exists, else initialize
    density_file = os.path.join(BASE_DIR, f"tng300_snap99_density_field_{grid_size}^3.npy")
    overdensity_file = os.path.join(BASE_DIR, f"tng300_snap99_density_contrast_{grid_size}^3.npy")
    metadata_file = os.path.join(BASE_DIR, f"tng300_snap99_metadata_{grid_size}^3.npy")
    
    if os.path.exists(density_file):
        print("Loading existing density field to accumulate...")
        density_total = np.load(density_file)
    else:
        print("No existing density field found, starting fresh.")
        density_total = np.zeros_like(density_batch)
    
    # Accumulate density field
    density_total += density_batch
    
    # Calculate overdensity (density contrast)
    mean_density = density_total.mean()
    density_over = density_total / mean_density - 1.0
    
    # Save updated fields
    print("Saving updated density and overdensity fields...")
    np.save(density_file, density_total)
    np.save(overdensity_file, density_over)
    
    # Save metadata
    metadata = {
        'grid_size': grid_size,
        'batch_start_index': start,
        'batch_end_index': end,
        'total_particles_batch': total_particles,
        'coord_min': coord_min,
        'coord_max': coord_max,
        'box_size': box_size,
        'mean_density': mean_density,
    }
    np.save(metadata_file, metadata)
    
    print(f"Batch {start}-{end} processed successfully.")
    print(f"Saved files:\n  {density_file}\n  {overdensity_file}\n  {metadata_file}")
    
    return density_total, density_over, metadata

def main():
    parser = argparse.ArgumentParser(description="Incremental CIC Density Field Calculator for TNG300")
    parser.add_argument("--grid-size", type=int, default=128, help="Grid size for density field (default: 128)")
    parser.add_argument("--start", type=int, default=0, help="Start file index (default: 0)")
    parser.add_argument("--end", type=int, default=None, help="End file index (default: last)")
    
    args = parser.parse_args()
    
    compute_density_field(grid_size=args.grid_size, start=args.start, end=args.end)

if __name__ == "__main__":
    main()
