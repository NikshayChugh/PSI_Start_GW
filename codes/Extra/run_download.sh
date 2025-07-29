#!/bin/bash
#SBATCH --job-name=tng_download
#SBATCH --output=/gpfs/nchugh/dmcoords/download_%j.out
#SBATCH --error=/gpfs/nchugh/dmcoords/download_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=defq

# Initialize conda for batch job
source /gpfs/nchugh/anaconda3/etc/profile.d/conda.sh
conda activate pyenv

cd /gpfs/nchugh/TNG300-1

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Download all 600 files using 6 workers
python tng_downloader.py --workers 8 --max-files 600