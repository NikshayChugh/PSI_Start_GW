#!/usr/bin/env python3
"""
CIC Method Comparison Tool (Fixed Version)
Compares the original and new CIC implementations with test cases
"""

import numpy as np
import matplotlib.pyplot as plt

def standardize_coords(coords, ngrid, xmax=None, xmin=None):
    """
    Standardize 3D coordinates to [0, ngrid) for each axis.
    Fixed to handle single particles and edge cases.
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

        # Handle case where x_min == x_max (single particle or all particles at same position)
        if x_max == x_min:
            # Place particle(s) at center of grid
            coords_std[:, i] = ngrid[i] / 2.0
        else:
            coords_std[:, i] = (coords[:, i] - x_min) / (x_max - x_min) * ngrid[i]

    return coords_std

def cic_density_field_original(coords, nx, ny=None, nz=None, mass=None, wraparound=False):
    """
    Original CIC method using bincount for accumulation.
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
    
    print(f'Original CIC: {nparticles:,} particles on {nx}x{ny}x{nz} grid')
    
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
        indices = a['ind'] + b['ind'] * nx + c['ind'] * nxny
        weights = a['weight'] * b['weight'] * c['weight']
        contribution = weights * mass_values
        
        # Ensure indices are valid before bincount
        valid_mask = (indices >= 0) & (indices < nx*ny*nz)
        if not np.all(valid_mask):
            print(f"Warning: Invalid indices detected. Valid: {np.sum(valid_mask)}/{len(indices)}")
            indices = indices[valid_mask]
            contribution = contribution[valid_mask]
        
        # Use numpy's bincount for efficient accumulation
        field += np.bincount(indices, weights=contribution, minlength=nx*ny*nz)

    # Get CIC weights for each dimension
    x1, x2 = findweights(coords[:, 0], nx)
    y1, y2 = findweights(coords[:, 1], ny)
    z1, z2 = findweights(coords[:, 2], nz)

    # Initialize field
    field = np.zeros(nx * ny * nz, dtype=np.float32)

    # Add all 8 contributions (2^3 for 3D CIC)
    update_field(field, x1, y1, z1, mass)
    update_field(field, x2, y1, z1, mass)
    update_field(field, x1, y2, z1, mass)
    update_field(field, x2, y2, z1, mass)
    update_field(field, x1, y1, z2, mass)
    update_field(field, x2, y1, z2, mass)
    update_field(field, x1, y2, z2, mass)
    update_field(field, x2, y2, z2, mass)

    return field.reshape((nx, ny, nz))

def cic_new(value, x, nx, y=None, ny=1, z=None, nz=1, wraparound=False, average=False):
    """
    New CIC method using explicit loops for accumulation.
    """
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
            weight2[bad] = 0.
        bad = (ind2 == ngrid)
        ind2[bad] = 0
        if not wraparound:
            weight2[bad] = 0.

        if wraparound:
            ind1[ind1 == ngrid] = 0

        return dict(weight=weight1, ind=ind1), dict(weight=weight2, ind=ind2)

    def update_field_vals(field, totalweight, a, b, c, value):
        """Update field using explicit loops."""
        indices = a['ind'] + b['ind'] * nx + c['ind'] * nxny
        weights = a['weight'] * b['weight'] * c['weight']
        value_weighted = weights * value 
        
        # Ensure indices are valid
        valid_mask = (indices >= 0) & (indices < nx*ny*nz)
        if not np.all(valid_mask):
            print(f"Warning: Invalid indices in new method. Valid: {np.sum(valid_mask)}/{len(indices)}")
            indices = indices[valid_mask]
            value_weighted = value_weighted[valid_mask]
            weights = weights[valid_mask]
        
        if average:
            for i, ind in enumerate(indices):
                field[ind] += value_weighted[i]
                totalweight[ind] += weights[i]
        else:
            for i, ind in enumerate(indices):
                field[ind] += value_weighted[i]

    nx, ny, nz = (int(i) for i in (nx, ny, nz))
    nxny = nx * ny
    value = np.asarray(value)

    print(f'New CIC: Resampling {len(value)} values to {nx}x{ny}x{nz} grid')

    x1, x2 = findweights(np.asarray(x), nx)
    y1 = z1 = dict(weight=np.ones_like(x), ind=np.zeros_like(x, dtype=int))
    if y is not None:
        y1, y2 = findweights(np.asarray(y), ny)
        if z is not None:
            z1, z2 = findweights(np.asarray(z), nz)

    field = np.zeros(nx * ny * nz, np.float32)

    if average:
        totalweight = np.zeros(nx * ny * nz, np.float32)
    else:
        totalweight = None

    update_field_vals(field, totalweight, x1, y1, z1, value)
    update_field_vals(field, totalweight, x2, y1, z1, value)
    if y is not None:
        update_field_vals(field, totalweight, x1, y2, z1, value)
        update_field_vals(field, totalweight, x2, y2, z1, value)
        if z is not None:
            update_field_vals(field, totalweight, x1, y1, z2, value)
            update_field_vals(field, totalweight, x2, y1, z2, value)
            update_field_vals(field, totalweight, x1, y2, z2, value)
            update_field_vals(field, totalweight, x2, y2, z2, value)

    if average:
        good = totalweight > 0
        field[good] /= totalweight[good]

    return field.reshape((nx, ny, nz)).squeeze()

def run_comparison_tests():
    """Run comprehensive comparison tests."""
    print("=" * 60)
    print("CIC METHOD COMPARISON TESTS")
    print("=" * 60)
    
    # Test parameters
    grid_size = 4
    
    # Test cases
    test_cases = {
        "Corner particles": np.array([
            [0.0, 0.0, 0.0],
            [3.999, 3.999, 3.999],
        ]),
        "Near corner particles": np.array([
            [0.001, 0.001, 0.001],
            [3.998, 3.998, 3.998],
        ]),
        "Center particle": np.array([
            [2.0, 2.0, 2.0],
        ]),
        "Edge particles": np.array([
            [0.0, 2.0, 2.0],
            [3.999, 2.0, 2.0],
            [2.0, 0.0, 2.0],
            [2.0, 3.999, 2.0],
            [2.0, 2.0, 0.0],
            [2.0, 2.0, 3.999],
        ]),
        "Mixed particles": np.array([
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
            [3.0, 3.0, 3.0],
            [3.5, 3.5, 3.5],
            [3.999, 3.999, 3.999],
        ]),
        "Identical particles": np.array([
            [1.5, 1.5, 1.5],
            [1.5, 1.5, 1.5],
            [1.5, 1.5, 1.5],
        ])
    }
    
    for test_name, test_coords in test_cases.items():
        print(f"\n{'-' * 40}")
        print(f"TEST: {test_name}")
        print(f"Coordinates:\n{test_coords}")
        print(f"{'-' * 40}")
        
        # For test cases, use coordinates directly in grid units
        coords_std = test_coords.copy()
        coords_std = np.clip(coords_std, 0, grid_size - 1e-6)
        
        print(f"Grid coordinates:\n{coords_std}")
        
        # Create uniform mass
        masses = np.ones(len(test_coords), dtype=np.float32)
        
        # Test original method
        try:
            field_orig = cic_density_field_original(coords_std, grid_size, 
                                                   wraparound=False, mass=masses)
        except Exception as e:
            print(f"Original method failed: {e}")
            continue
        
        # Test new method
        try:
            field_new = cic_new(masses, coords_std[:, 0], grid_size,
                               coords_std[:, 1], grid_size, coords_std[:, 2], grid_size,
                               wraparound=False, average=False)
        except Exception as e:
            print(f"New method failed: {e}")
            continue
        
        # Compare results
        diff = field_orig - field_new
        max_diff = np.max(np.abs(diff))
        total_orig = np.sum(field_orig)
        total_new = np.sum(field_new)
        
        print(f"\nRESULTS:")
        print(f"Original method - Total mass: {total_orig:.6f}")
        print(f"New method - Total mass: {total_new:.6f}")
        print(f"Max absolute difference: {max_diff:.10f}")
        if max(total_orig, total_new) > 0:
            print(f"Relative difference: {max_diff/max(total_orig, total_new):.10f}")
        
        if max_diff > 1e-6:
            print(f"⚠️  SIGNIFICANT DIFFERENCE DETECTED!")
            print(f"Original field shape: {field_orig.shape}")
            print(f"New field shape: {field_new.shape}")
            
            # Find where differences occur
            diff_indices = np.where(np.abs(diff) > 1e-10)
            if len(diff_indices[0]) > 0:
                print(f"Differences at indices: {list(zip(*diff_indices))[:10]}")  # Show first 10
                for i, (x, y, z) in enumerate(zip(*diff_indices)):
                    if i >= 5:  # Limit output
                        break
                    print(f"  [{x},{y},{z}]: orig={field_orig[x,y,z]:.6f}, "
                          f"new={field_new[x,y,z]:.6f}, diff={diff[x,y,z]:.6f}")
        else:
            print(f"✅ Fields match within tolerance")
        
        # Test with wraparound
        print(f"\nTesting with wraparound=True:")
        try:
            field_orig_wrap = cic_density_field_original(coords_std, grid_size, 
                                                        wraparound=True, mass=masses)
            field_new_wrap = cic_new(masses, coords_std[:, 0], grid_size,
                                    coords_std[:, 1], grid_size, coords_std[:, 2], grid_size,
                                    wraparound=True, average=False)
            
            diff_wrap = field_orig_wrap - field_new_wrap
            max_diff_wrap = np.max(np.abs(diff_wrap))
            
            print(f"Wraparound - Max difference: {max_diff_wrap:.10f}")
            if max_diff_wrap > 1e-6:
                print(f"⚠️  WRAPAROUND DIFFERENCE DETECTED!")
            else:
                print(f"✅ Wraparound fields match")
        except Exception as e:
            print(f"Wraparound test failed: {e}")

def detailed_particle_analysis():
    """Analyze how individual particles are handled."""
    print(f"\n{'=' * 60}")
    print("DETAILED PARTICLE ANALYSIS")
    print(f"{'=' * 60}")
    
    # Single particle tests
    grid_size = 4
    single_particles = [
        [0.0, 0.0, 0.0],
        [0.1, 0.1, 0.1],
        [0.5, 0.5, 0.5],
        [1.0, 1.0, 1.0],
        [1.5, 1.5, 1.5],
        [2.0, 2.0, 2.0],
        [3.9, 3.9, 3.9],
        [3.99, 3.99, 3.99],
        [3.999, 3.999, 3.999],
    ]
    
    for particle_pos in single_particles:
        print(f"\nAnalyzing particle at {particle_pos}")
        
        coords = np.array([particle_pos])
        coords_std = np.clip(coords, 0, grid_size - 1e-6)
        
        print(f"Grid coordinates: {coords_std[0]}")
        
        # Test both methods
        try:
            field_orig = cic_density_field_original(coords_std, grid_size)
            field_new = cic_new(np.ones(1), coords_std[:, 0], grid_size,
                               coords_std[:, 1], grid_size, coords_std[:, 2], grid_size,
                               wraparound=False, average=False)
            
            # Find non-zero elements
            orig_nonzero = np.where(field_orig > 1e-10)
            new_nonzero = np.where(field_new > 1e-10)
            
            print(f"Original method - Non-zero at: {list(zip(*orig_nonzero))}")
            print(f"New method - Non-zero at: {list(zip(*new_nonzero))}")
            
            # Compare values
            diff = field_orig - field_new
            max_diff = np.max(np.abs(diff))
            if max_diff > 1e-10:
                print(f"⚠️  Difference: {max_diff:.10f}")
            else:
                print(f"✅ Match")
                
        except Exception as e:
            print(f"Error: {e}")

def debug_weight_calculation():
    """Debug the weight calculation for specific cases."""
    print(f"\n{'=' * 60}")
    print("DEBUG WEIGHT CALCULATION")
    print(f"{'=' * 60}")
    
    # Test specific problematic cases
    test_positions = [2.0, 1.5, 0.0, 3.999]
    grid_size = 4
    
    for pos in test_positions:
        print(f"\nDebugging position: {pos}")
        
        # Original method calculation
        if False:  # wraparound
            ngp = np.fix(pos + 0.5)
        else:
            ngp = np.fix(pos) + 0.5
        
        distngp = ngp - pos
        weight2 = np.abs(distngp)
        weight1 = 1.0 - weight2
        
        if False:  # wraparound
            ind1 = ngp
        else:
            ind1 = ngp - 0.5
        ind1 = int(ind1)
        
        ind2 = ind1 - 1
        if distngp < 0:
            ind2 += 2
        
        print(f"  ngp: {ngp}, distngp: {distngp}")
        print(f"  weight1: {weight1}, weight2: {weight2}")
        print(f"  ind1: {ind1}, ind2: {ind2}")
        print(f"  Total weight: {weight1 + weight2}")

if __name__ == "__main__":
    run_comparison_tests()
    detailed_particle_analysis()
    debug_weight_calculation()