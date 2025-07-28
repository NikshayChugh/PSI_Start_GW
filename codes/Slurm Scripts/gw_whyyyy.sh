#!/bin/bash
#SBATCH --job-name=siren_bias_linear_analysis
#SBATCH --output=logs_gw/bias_linear_analysis_%j.out
#SBATCH --error=logs_gw/bias_linear_analysis_%j.err
#SBATCH --partition=preq
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --mem=32G

# Load Conda environment
source /gpfs/nchugh/anaconda3/etc/profile.d/conda.sh
conda activate gw_env

# Navigate to working directory
cd /gpfs/nchugh/gw

# Create necessary directories if they don't exist
mkdir -p bias_results_linear_largek

# List of snapshots to analyze
SNAPSHOTS=(50 55 60 65 70 75 80 85 90 99)

echo "Starting linear bias analysis for snapshots: ${SNAPSHOTS[@]}"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"

# Run the analysis script for each desired snapshot
for SNAP in "${SNAPSHOTS[@]}"; do
    echo ""
    echo "========================================="
    echo "Running linear bias analysis for snapshot ${SNAP}"
    echo "========================================="
    echo "Time: $(date)"
    
    # Run the analysis
    python gw_analysis_maybe_final.py --snapshot ${SNAP}
    
    # Check if the analysis was successful
    if [ $? -eq 0 ]; then
        echo "✅ Snapshot ${SNAP} completed successfully"
    else
        echo "❌ Snapshot ${SNAP} failed"
    fi
    
    echo "Snapshot ${SNAP} finished at: $(date)"
done

echo ""
echo "========================================="
echo "All analyses completed!"
echo "End time: $(date)"
echo "========================================="

# Create a combined summary of all results
echo ""
echo "Creating combined results summary..."

# Run Python script to combine all results
cat << 'EOF' > combine_linear_results.py
#!/usr/bin/env python3
"""
Combine all snapshot linear bias results into a single file
"""
import pandas as pd
import glob
import os
import numpy as np
import matplotlib.pyplot as plt

def combine_all_results():
    # Find all result files (updated for linear results)
    result_files = glob.glob('bias_results_linear_largek/siren_bias_linear_results_snap_*.pkl')
    
    if len(result_files) == 0:
        print("No linear bias result files found!")
        return
    
    print(f"Found {len(result_files)} linear bias result files")
    
    # Load and combine all results
    all_results = []
    for file in sorted(result_files):
        try:
            df = pd.read_pickle(file)
            all_results.append(df)
            snapshot = df['snap'].iloc[0]
            print(f"Loaded {len(df)} results from snapshot {snapshot}")
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    if len(all_results) == 0:
        print("No valid results to combine!")
        return
    
    # Combine all results
    df_combined = pd.concat(all_results, ignore_index=True)
    
    # Save combined results
    df_combined.to_pickle('bias_results_linear_largek/siren_bias_linear_results_all_snapshots.pkl')
    df_combined.to_csv('bias_results_linear_largek/siren_bias_linear_results_all_snapshots.csv', index=False)
    
    print(f"\n✅ Combined linear bias results saved! Total entries: {len(df_combined)}")
    
    # Create evolution plot
    create_evolution_plot(df_combined)
    
    # Print overall summary
    print_combined_summary(df_combined)

def create_evolution_plot(df_combined):
    """Create plots showing linear bias evolution across snapshots"""
    
    # Filter valid results (stricter chi2 cut for linear fits)
    valid_mask = np.isfinite(df_combined['bias']) & (df_combined['chi2_reduced'] < 10)
    df_valid = df_combined[valid_mask].copy()
    
    if len(df_valid) == 0:
        print("No valid results for evolution plot")
        return
    
    # Group by snapshot and compute statistics
    snap_stats = df_valid.groupby('snap')['bias'].agg(['mean', 'std', 'count']).reset_index()
    
    # Create evolution plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Bias vs snapshot
    axes[0,0].errorbar(snap_stats['snap'], snap_stats['mean'], 
                      yerr=snap_stats['std'], fmt='o-', capsize=5, capthick=2)
    axes[0,0].set_xlabel('Snapshot')
    axes[0,0].set_ylabel('Mean Bias @ k=0.1')
    axes[0,0].set_title('Linear Bias Evolution Across Snapshots')
    axes[0,0].grid(True, alpha=0.3)
    
    # Number of valid results per snapshot
    axes[0,1].bar(snap_stats['snap'], snap_stats['count'], alpha=0.7)
    axes[0,1].set_xlabel('Snapshot')
    axes[0,1].set_ylabel('Number of Valid Results')
    axes[0,1].set_title('Valid Results Count (χ²ᵣ < 10)')
    axes[0,1].grid(True, alpha=0.3)
    
    # Bias distribution across all snapshots
    for snap in sorted(df_valid['snap'].unique()):
        df_snap = df_valid[df_valid['snap'] == snap]
        axes[1,0].hist(df_snap['bias'], alpha=0.5, bins=15, label=f'Snap {snap}')
    axes[1,0].set_xlabel('Bias @ k=0.1')
    axes[1,0].set_ylabel('Count')
    axes[1,0].set_title('Linear Bias Distribution by Snapshot')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # Overall bias vs parameters (all snapshots combined)
    scatter = axes[1,1].scatter(df_valid['mc'], df_valid['bias'], 
                               c=df_valid['snap'], cmap='viridis', alpha=0.6)
    axes[1,1].set_xlabel('mc')
    axes[1,1].set_ylabel('Bias @ k=0.1')
    axes[1,1].set_title('Linear Bias vs mc (colored by snapshot)')
    plt.colorbar(scatter, ax=axes[1,1], label='Snapshot')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('bias_results_linear_largek/bias_linear_evolution_all_snapshots.png', dpi=600, bbox_inches='tight')
    plt.close()
    
    print("✅ Linear bias evolution plots saved!")

def print_combined_summary(df_combined):
    """Print summary statistics for all snapshots"""
    
    # Use stricter chi2 cut for linear fits
    valid_mask = np.isfinite(df_combined['bias']) & (df_combined['chi2_reduced'] < 10)
    df_valid = df_combined[valid_mask].copy()
    
    print(f"\n{'='*60}")
    print(f"COMBINED SUMMARY - LINEAR BIAS ANALYSIS - ALL SNAPSHOTS")
    print(f"{'='*60}")
    print(f"Total entries: {len(df_combined)}")
    print(f"Valid entries (χ²ᵣ < 10): {len(df_valid)}")
    print(f"Snapshots analyzed: {sorted(df_combined['snap'].unique())}")
    
    if len(df_valid) > 0:
        print(f"\nOverall linear bias @ k=0.1 h/Mpc statistics:")
        print(f"  Mean: {df_valid['bias'].mean():.4f}")
        print(f"  Std:  {df_valid['bias'].std():.4f}")
        print(f"  Min:  {df_valid['bias'].min():.4f}")
        print(f"  Max:  {df_valid['bias'].max():.4f}")
        print(f"  Median: {df_valid['bias'].median():.4f}")
        
        print(f"\nLinear fit quality statistics:")
        print(f"  Mean χ²ᵣ: {df_valid['chi2_reduced'].mean():.3f}")
        print(f"  Median χ²ᵣ: {df_valid['chi2_reduced'].median():.3f}")
        print(f"  Max χ²ᵣ: {df_valid['chi2_reduced'].max():.3f}")
        
        # Per-snapshot summary
        print(f"\nPer-snapshot summary:")
        snap_stats = df_valid.groupby('snap')['bias'].agg(['count', 'mean', 'std']).round(4)
        for snap, stats in snap_stats.iterrows():
            print(f"  Snapshot {snap}: {stats['count']} results, "
                  f"bias = {stats['mean']:.4f} ± {stats['std']:.4f}")
        
        # Find best overall fit
        best_idx = df_valid['chi2_reduced'].idxmin()
        best_result = df_valid.loc[best_idx]
        
        print(f"\nBest overall linear fit (χ²ᵣ = {best_result['chi2_reduced']:.3f}):")
        print(f"  snap = {best_result['snap']}")
        print(f"  mc = {best_result['mc']:.3f}")
        print(f"  dl = {best_result['dl']:.2f}")
        print(f"  dh = {best_result['dh']:.2f}")
        print(f"  bias @ k=0.1 = {best_result['bias']:.4f} ± {best_result['bias_error']:.4f}")
        print(f"  n_sirens = {best_result['n_sirens']}")

if __name__ == "__main__":
    combine_all_results()
EOF

python combine_linear_results.py

# Clean up temporary script
rm combine_linear_results.py

echo ""
echo "========================================="
echo "FINAL SUMMARY - LINEAR BIAS ANALYSIS"
echo "========================================="
echo "All analyses completed at: $(date)"
echo "Results saved in: bias_results_linear_largek/"
echo "Plots saved in: plots_snap_*/"
echo "Combined results: bias_results_linear_largek/siren_bias_linear_results_all_snapshots.*"
echo "Evolution plot: bias_results_linear_largek/bias_linear_evolution_all_snapshots.png"
echo "Note: Linear bias method fits b(k) = b0 + b1*k and reports bias @ k=0.1 h/Mpc"
echo "========================================="