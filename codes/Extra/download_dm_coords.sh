#!/bin/bash
#SBATCH --job-name=tng_dm_download
#SBATCH --output=/gpfs/nchugh/dmcoordssnaps/download_%j.out
#SBATCH --error=/gpfs/nchugh/dmcoordssnaps/download_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=amdq
#SBATCH --nodelist=cn094

source /gpfs/nchugh/anaconda3/etc/profile.d/conda.sh
conda activate gw_env

export TNG_API_KEY="a6e92d93311aa4bd2349be55ec31c930"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

mkdir -p /gpfs/nchugh/dmcoordssnaps
cd /gpfs/nchugh/dmcoordssnaps

echo "=== JOB START ==="
echo "Start time: $(date)"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "=================="

python /gpfs/nchugh/TNG300-1/download_dm_coords.py \
  --workers $SLURM_CPUS_PER_TASK \
  --base-dir /gpfs/nchugh/dmcoordssnaps \
  --snapshots 65 75

echo "=== JOB END ==="
echo "End time: $(date)"