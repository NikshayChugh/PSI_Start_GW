#!/bin/bash
#SBATCH --job-name=submit_driver
#SBATCH --output=submit_driver_%j.out
#SBATCH --error=submit_driver_%j.err
#SBATCH --time=00:10:00
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G

# Run your submit_jobs.sh script
bash submit_jobs.sh