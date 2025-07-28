#!/usr/bin/env python3
"""
Simplified Siren Bias Analysis
Uses improved linear bias fitting method for robustness
Modified to handle multiple snapshots via command line arguments
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import MAS_library as MASL
import Pk_library as PKL
import glob
import os
import argparse
from scipy.interpolate import griddata
from scipy.optimize import curve_fit
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings('ignore')

# Configuration
GRID = 256
BOXSIZE = 205.001  # Mpc/h
MAS = 'CIC'
SIREN_PATH = "/gpfs/nchugh/gw/results/"
DM_PATH = "/gpfs/nchugh/DMCOORDS/"
KMIN_FIT = 0.04
KMAX_FIT = 1.0

def parse_filename(filename):
    """Parse siren filename to extract parameters"""
    basename = os.path.basename(filename)
    parts = basename.replace('.pkl', '').split('_')
    
    params = {}
    for i, part in enumerate(parts):
        if part == 'snap' and i+1 < len(parts):
            params['snap'] = int(parts[i+1])
        elif part == 'dl' and i+1 < len(parts):
            params['dl'] = float(parts[i+1])
        elif part == 'dh' and i+1 < len(parts):
            params['dh'] = float(parts[i+1])
        elif part == 'mc' and i+1 < len(parts):
            params['mc'] = float(parts[i+1])
    
    return params

def compute_power_spectrum(delta, boxsize, MAS):
    """Compute power spectrum using Pylians"""
    Pk = PKL.Pk(delta, boxsize, axis=0, MAS=MAS, threads=1)
    return Pk.k3D, Pk.Pk[:,0], Pk.Nmodes3D

def load_and_process_siren_catalog(filepath):
    """Load siren catalog and convert to density field"""
    try:
        df = pd.read_pickle(filepath)
        
        # Extract positions (convert from kpc/h to Mpc/h)
        x = df['x'].values / 1000.0
        y = df['y'].values / 1000.0
        z = df['z_pos'].values / 1000.0
        positions = np.vstack([x, y, z]).T.astype(np.float32)
        
        # Create density field
        delta = np.zeros((GRID, GRID, GRID), dtype=np.float32)
        MASL.MA(positions, delta, BOXSIZE, MAS)
        
        # Normalize to get overdensity field
        mean_delta = np.mean(delta)
        if mean_delta > 0:
            delta /= mean_delta
            delta -= 1.0
        else:
            print(f"Warning: Zero mean density in {filepath}")
            return None, 0
        
        return delta, len(df)
    
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None, 0

def load_dm_field(filepath):
    """Load DM overdensity field"""
    try:
        delta_dm = np.load(filepath).astype(np.float32)
        return delta_dm
    except Exception as e:
        print(f"Error loading DM field {filepath}: {e}")
        return None

def compute_bias_linear_fit(k_siren, P_siren, k_dm, P_dm, nmodes_siren, nmodes_dm):
    """
    Compute bias using improved linear fitting method
    Fits: P_siren = (b0 + b1*k)^2 * P_dm
    Returns bias at k=0.1 h/Mpc reference scale
    """
    
    # Find overlapping k range
    k_min = max(k_siren.min(), k_dm.min(), KMIN_FIT)
    k_max = min(k_siren.max(), k_dm.max(), KMAX_FIT)
    
    if k_max <= k_min:
        return np.nan, np.nan, np.nan, 0
    
    # Create common k grid
    k_common = np.logspace(np.log10(k_min), np.log10(k_max), 15)
    
    # Interpolate to common grid
    P_siren_interp = np.interp(k_common, k_siren, P_siren)
    P_dm_interp = np.interp(k_common, k_dm, P_dm)
    nmodes_siren_interp = np.interp(k_common, k_siren, nmodes_siren)
    nmodes_dm_interp = np.interp(k_common, k_dm, nmodes_dm)
    
    # Compute errors on power spectra
    err_siren = P_siren_interp / np.sqrt(2 * nmodes_siren_interp + 1e-10)
    err_dm = P_dm_interp / np.sqrt(2 * nmodes_dm_interp + 1e-10)
    
    # Remove invalid points
    valid = (P_siren_interp > 0) & (P_dm_interp > 0) & np.isfinite(err_siren) & np.isfinite(err_dm) & (nmodes_siren_interp > 0)
    
    if np.sum(valid) < 3:
        return np.nan, np.nan, np.nan, np.sum(valid)
    
    k_fit = k_common[valid]
    P_siren_fit = P_siren_interp[valid]
    P_dm_fit = P_dm_interp[valid]
    err_siren_fit = err_siren[valid]
    
    # Define linear bias model
    def model(k, b0, b1):
        bias_k = b0 + b1 * k
        return bias_k**2 * np.interp(k, k_fit, P_dm_fit)
    
    try:
        # Initial guess: constant bias around 1.5, small slope
        p0 = [1.5, 0.0]
        popt, pcov = curve_fit(model, k_fit, P_siren_fit, p0=p0, 
                              sigma=err_siren_fit, absolute_sigma=True)
        
        b0, b1 = popt
        b0_err, b1_err = np.sqrt(np.diag(pcov))
        
        # Compute chi-squared
        model_pred = model(k_fit, b0, b1)
        chi2 = np.sum(((P_siren_fit - model_pred) / err_siren_fit)**2)
        dof = len(k_fit) - 2  # 2 parameters
        chi2_reduced = chi2 / dof if dof > 0 else np.inf
        
        # Return effective bias at k=0.1 h/Mpc (typical scale)
        k_ref = 0.1
        bias_ref = b0 + b1 * k_ref
        # Error propagation for bias at k_ref
        bias_ref_err = np.sqrt(b0_err**2 + (k_ref * b1_err)**2)
        
        return bias_ref, bias_ref_err, chi2_reduced, len(k_fit)
        
    except Exception as e:
        print(f"    Warning: Linear fit failed, falling back to simple average: {e}")
        # Fallback to simple bias calculation if fitting fails
        return compute_bias_simple_average(k_fit, P_siren_fit, P_dm_fit, err_siren_fit, nmodes_siren_interp[valid])

def compute_bias_simple_average(k, P_siren, P_dm, err_siren, nmodes_siren):
    """Fallback method: simple weighted average of b(k) = sqrt(P_siren/P_dm)"""
    
    # Compute bias as sqrt(P_siren / P_dm)
    bias = np.sqrt(P_siren / P_dm)
    
    # Error propagation
    err_dm = P_dm / np.sqrt(2 * nmodes_siren + 1e-10)  # Approximate
    err_bias = 0.5 * bias * np.sqrt((err_siren / P_siren)**2 + (err_dm / P_dm)**2)
    
    # Remove invalid points
    valid = np.isfinite(bias) & np.isfinite(err_bias) & (err_bias > 0)
    
    if np.sum(valid) < 1:
        return np.nan, np.nan, np.nan, 0
    
    bias_fit = bias[valid]
    err_fit = err_bias[valid]
    
    # Weighted average
    weights = 1.0 / err_fit**2
    bias_mean = np.sum(weights * bias_fit) / np.sum(weights)
    bias_error = 1.0 / np.sqrt(np.sum(weights))
    
    # Chi-squared
    chi2 = np.sum(weights * (bias_fit - bias_mean)**2)
    dof = len(bias_fit) - 1
    chi2_reduced = chi2 / dof if dof > 0 else np.inf
    
    return bias_mean, bias_error, chi2_reduced, len(bias_fit)

def analyze_snapshot(snapshot):
    """Analyze a single snapshot and return results"""
    print(f"\n{'='*60}")
    print(f"ANALYZING SNAPSHOT {snapshot}")
    print(f"{'='*60}")
    
    # Find all siren files for this snapshot
    pattern = os.path.join(SIREN_PATH, f"siren_cat_snap_{snapshot}_*.pkl")
    siren_files = glob.glob(pattern)
    
    if len(siren_files) == 0:
        print(f"No siren files found for snapshot {snapshot} matching: {pattern}")
        return []
    
    print(f"Found {len(siren_files)} siren catalog files for snapshot {snapshot}")
    
    # Parse and organize files
    catalog_info = []
    for filepath in siren_files:
        params = parse_filename(filepath)
        if all(key in params for key in ['snap', 'mc', 'dl', 'dh']):
            params['filepath'] = filepath
            catalog_info.append(params)
        else:
            print(f"Could not parse filename: {os.path.basename(filepath)}")
    
    if len(catalog_info) == 0:
        print(f"No valid catalog files found for snapshot {snapshot}")
        return []
    
    df_catalogs = pd.DataFrame(catalog_info)
    df_catalogs = df_catalogs[df_catalogs['snap'] == snapshot]
    print(f"Valid catalogs for snapshot {snapshot}: {len(df_catalogs)}")
    print(f"Parameter ranges:")
    print(f"  mc: {df_catalogs['mc'].min():.3f} - {df_catalogs['mc'].max():.3f}")
    print(f"  dl: {df_catalogs['dl'].min():.2f} - {df_catalogs['dl'].max():.2f}")
    print(f"  dh: {df_catalogs['dh'].min():.2f} - {df_catalogs['dh'].max():.2f}")
    
    # Process each catalog
    results = []
    
    for idx, row in df_catalogs.iterrows():
        snap = row['snap']
        mc = row['mc']
        dl = row['dl']
        dh = row['dh']
        filepath = row['filepath']
        
        print(f"\n[{idx+1}/{len(df_catalogs)}] Processing snap={snap}, mc={mc:.3f}, dl={dl:.2f}, dh={dh:.2f}")
        
        # Check DM field exists
        dm_filepath = os.path.join(DM_PATH, f"field_256_{snap}.npy")
        if not os.path.exists(dm_filepath):
            print(f"  ❌ DM field not found: {dm_filepath}")
            continue
        
        # Load data
        delta_siren, n_sirens = load_and_process_siren_catalog(filepath)
        if delta_siren is None:
            print(f"  ❌ Failed to load siren catalog")
            continue
        
        delta_dm = load_dm_field(dm_filepath)
        if delta_dm is None:
            print(f"  ❌ Failed to load DM field")
            continue
        
        print(f"  ✓ Loaded {n_sirens} sirens and DM field")
        
        # Compute power spectra
        try:
            k_siren, P_siren, nmodes_siren = compute_power_spectrum(delta_siren, BOXSIZE, MAS)
            k_dm, P_dm, nmodes_dm = compute_power_spectrum(delta_dm, BOXSIZE, MAS)
            
            # Compute bias using improved linear fitting
            bias_fit, bias_err, chi2_red, n_points = compute_bias_linear_fit(
                k_siren, P_siren, k_dm, P_dm, nmodes_siren, nmodes_dm)
            
            result = {
                'snap': snap,
                'mc': mc,
                'dl': dl,
                'dh': dh,
                'n_sirens': n_sirens,
                'bias': bias_fit,
                'bias_error': bias_err,
                'chi2_reduced': chi2_red,
                'n_fit_points': n_points
            }
            results.append(result)
            
            if np.isfinite(bias_fit):
                print(f"  ✓ Bias: {bias_fit:.3f} ± {bias_err:.3f}, χ²ᵣ = {chi2_red:.2f} ({n_points} points)")
            else:
                print(f"  ❌ Bias fit failed")
                
        except Exception as e:
            print(f"  ❌ Error computing power spectrum: {e}")
            continue
    
    print(f"\n✓ Snapshot {snapshot} analysis complete! {len(results)} successful results.")
    return results

def create_plots(df_results, snapshot):
    """Create visualization plots for a specific snapshot"""
    
    print(f"\nCreating plots for snapshot {snapshot}...")
    
    # Filter valid results for this snapshot
    df_snap = df_results[df_results['snap'] == snapshot].copy()
    valid_mask = np.isfinite(df_snap['bias']) & (df_snap['chi2_reduced'] < 10)  # Stricter chi2 cut for linear fits
    df_valid = df_snap[valid_mask].copy()
    
    if len(df_valid) == 0:
        print(f"No valid results for plotting snapshot {snapshot}")
        return
    
    print(f"Using {len(df_valid)} valid results for snapshot {snapshot} (χ²ᵣ < 10)")
    
    # Create output directory for plots
    plot_dir = f'plots_snap_{snapshot}'
    os.makedirs(plot_dir, exist_ok=True)
    
    # Plot 1: 3D scatter
    if len(df_valid) >= 4:
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        scatter = ax.scatter(df_valid['mc'], df_valid['dl'], df_valid['dh'], 
                           c=df_valid['bias'], cmap='plasma', s=80, alpha=0.8,
                           edgecolors='black', linewidth=0.5)
        
        ax.set_xlabel('mc', fontsize=12)
        ax.set_ylabel('dl', fontsize=12)
        ax.set_zlabel('dh', fontsize=12)
        ax.set_title(f'Bias Parameter Distribution (Linear Fit, Snapshot {snapshot})', fontsize=14)
        
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, pad=0.15)
        cbar.set_label('Bias @ k=0.1', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f'bias_3d_linear_largek_snap_{snapshot}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    # Plot 2: 2D projections
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # mc vs dl
    scatter1 = axes[0,0].scatter(df_valid['mc'], df_valid['dl'], c=df_valid['bias'], 
                                cmap='plasma', s=60, alpha=0.8, edgecolors='black', linewidth=0.5)
    axes[0,0].set_xlabel('mc')
    axes[0,0].set_ylabel('dl')
    axes[0,0].set_title('mc vs dl')
    plt.colorbar(scatter1, ax=axes[0,0])
    
    # mc vs dh
    scatter2 = axes[0,1].scatter(df_valid['mc'], df_valid['dh'], c=df_valid['bias'], 
                                cmap='plasma', s=60, alpha=0.8, edgecolors='black', linewidth=0.5)
    axes[0,1].set_xlabel('mc')
    axes[0,1].set_ylabel('dh')
    axes[0,1].set_title('mc vs dh')
    plt.colorbar(scatter2, ax=axes[0,1])
    
    # dl vs dh
    scatter3 = axes[0,2].scatter(df_valid['dl'], df_valid['dh'], c=df_valid['bias'], 
                                cmap='plasma', s=60, alpha=0.8, edgecolors='black', linewidth=0.5)
    axes[0,2].set_xlabel('dl')
    axes[0,2].set_ylabel('dh')
    axes[0,2].set_title('dl vs dh')
    plt.colorbar(scatter3, ax=axes[0,2])
    
    # Individual parameter trends
    axes[1,0].scatter(df_valid['mc'], df_valid['bias'], alpha=0.6, s=40)
    axes[1,0].set_xlabel('mc')
    axes[1,0].set_ylabel('Bias @ k=0.1')
    axes[1,0].grid(True, alpha=0.3)
    
    axes[1,1].scatter(df_valid['dl'], df_valid['bias'], alpha=0.6, s=40)
    axes[1,1].set_xlabel('dl')
    axes[1,1].set_ylabel('Bias @ k=0.1')
    axes[1,1].grid(True, alpha=0.3)
    
    axes[1,2].scatter(df_valid['dh'], df_valid['bias'], alpha=0.6, s=40)
    axes[1,2].set_xlabel('dh')
    axes[1,2].set_ylabel('Bias @ k=0.1')
    axes[1,2].grid(True, alpha=0.3)
    
    plt.suptitle(f'Bias Parameter Analysis (Linear Fit) - Snapshot {snapshot}', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'bias_2d_linear_analysis_largek_snap_{snapshot}.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Quality metrics
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].hist(df_valid['bias'], bins=20, alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('Bias @ k=0.1')
    axes[0].set_ylabel('Count')
    axes[0].set_title(f'Bias Distribution (Linear Fit, Snap {snapshot})\nMean: {df_valid["bias"].mean():.3f} ± {df_valid["bias"].std():.3f}')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].hist(df_valid['chi2_reduced'], bins=20, alpha=0.7, edgecolor='black')
    axes[1].set_xlabel('χ² reduced')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Linear Fit Quality')
    axes[1].grid(True, alpha=0.3)
    
    axes[2].scatter(df_valid['n_sirens'], df_valid['bias'], alpha=0.6)
    axes[2].set_xlabel('N sirens')
    axes[2].set_ylabel('Bias @ k=0.1')
    axes[2].set_xscale('log')
    axes[2].set_title('Bias vs Sample Size')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'bias_quality_metrics_linear_largek_snap_{snapshot}.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()

def print_summary_statistics(df_results, snapshot):
    """Print summary statistics for a snapshot"""
    df_snap = df_results[df_results['snap'] == snapshot].copy()
    valid_mask = np.isfinite(df_snap['bias']) & (df_snap['chi2_reduced'] < 10)  # Stricter cut
    df_valid = df_snap[valid_mask].copy()
    
    if len(df_valid) == 0:
        print(f"No valid results for snapshot {snapshot}")
        return
    
    print(f"\n{'='*60}")
    print(f"SUMMARY STATISTICS (LINEAR FIT) - SNAPSHOT {snapshot}")
    print(f"{'='*60}")
    print(f"Total valid results: {len(df_valid)}")
    print(f"Bias @ k=0.1 h/Mpc statistics:")
    print(f"  Mean: {df_valid['bias'].mean():.4f}")
    print(f"  Std:  {df_valid['bias'].std():.4f}")
    print(f"  Min:  {df_valid['bias'].min():.4f}")
    print(f"  Max:  {df_valid['bias'].max():.4f}")
    print(f"  Median: {df_valid['bias'].median():.4f}")
    
    print(f"\nLinear fit quality:")
    print(f"  Mean χ²ᵣ: {df_valid['chi2_reduced'].mean():.3f}")
    print(f"  Median χ²ᵣ: {df_valid['chi2_reduced'].median():.3f}")
    
    # Find best-fit parameters (lowest chi-squared)
    if len(df_valid) > 0:
        best_idx = df_valid['chi2_reduced'].idxmin()
        best_result = df_valid.loc[best_idx]
        
        print(f"\nBest fit (lowest χ²ᵣ = {best_result['chi2_reduced']:.3f}):")
        print(f"  snap = {best_result['snap']}")
        print(f"  mc = {best_result['mc']:.3f}")
        print(f"  dl = {best_result['dl']:.2f}")
        print(f"  dh = {best_result['dh']:.2f}")
        print(f"  bias = {best_result['bias']:.4f} ± {best_result['bias_error']:.4f}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Siren Bias Analysis with Linear Fitting')
    parser.add_argument('--snapshot', type=int, required=True,
                       help='Snapshot number to analyze')
    args = parser.parse_args()
    
    snapshot = args.snapshot
    
    # Analyze the specified snapshot
    results = analyze_snapshot(snapshot)
    
    if len(results) == 0:
        print(f"\n❌ No results for snapshot {snapshot}!")
        return None
    
    # Convert to DataFrame
    df_results = pd.DataFrame(results)
    
    # Save results for this snapshot
    output_dir = 'bias_results_linear_largek'
    os.makedirs(output_dir, exist_ok=True)
    
    df_results.to_pickle(os.path.join(output_dir, f'siren_bias_linear_results_snap_{snapshot}.pkl'))
    df_results.to_csv(os.path.join(output_dir, f'siren_bias_linear_results_snap_{snapshot}.csv'), index=False)
    
    print(f"\n✓ Results saved for snapshot {snapshot} (linear fitting method)!")
    
    # Create visualizations
    create_plots(df_results, snapshot)
    
    # Print summary statistics
    print_summary_statistics(df_results, snapshot)
    
    return df_results

if __name__ == "__main__":
    import sys
    
    # Set up better error handling
    try:
        results = main()
        if results is not None:
            print("\n✅ Analysis completed successfully!")
            sys.exit(0)
        else:
            print("\n❌ Analysis failed!")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n⚠️  Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)