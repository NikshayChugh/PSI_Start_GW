#!/bin/bash
#SBATCH --job-name=tng_density
#SBATCH --output=/gpfs/nchugh/dmcoordssnaps/snap-55/density_%j.out
#SBATCH --error=/gpfs/nchugh/dmcoordssnaps/snap-55/density_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --partition=amdpreq

# Initialize conda for batch job
source /gpfs/nchugh/anaconda3/etc/profile.d/conda.sh
conda activate gw_env

cd /gpfs/nchugh/TNG300-1

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Run density field calculation
# For 256^3 grid using available coord files
python cic_density.py --grid-size 256 --start $START --end $END

echo "Density field calculation completed!"
echo "Check output files in /gpfs/nchugh/dmcoordssnaps/snap-55/"