#!/bin/bash
#SBATCH --job-name=dmfield
#SBATCH --output=/gpfs/nchugh/gw/dmfieldslurmouts/dmfield_%j.out
#SBATCH --error=/gpfs/nchugh/gw/dmfieldslurmouts/dmfield_%j.err
#SBATCH --time=24:00:00       # adjust time as needed
#SBATCH --partition=defq     # choose partition as appropriate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8   # adjust if your code uses threads or multiprocessing
#SBATCH --mem=32G           # adjust memory as needed

# Initialize conda for batch job
source /gpfs/nchugh/anaconda3/etc/profile.d/conda.sh
conda activate gw_env

# Change to working directory
cd /gpfs/nchugh/gw

# Set OpenMP threads to match SLURM allocation
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Snapshot number passed as first argument to this script
snapshot=$1
echo "Running snapshot $snapshot"
echo "Working directory: $(pwd)"
echo "Using $OMP_NUM_THREADS threads"

# Run the main script
python make_field.py $snapshot

echo "dmfield calculation completed!"
echo "Check output files in /gpfs/nchugh/gw/"