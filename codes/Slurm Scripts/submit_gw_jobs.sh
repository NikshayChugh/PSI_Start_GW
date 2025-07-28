#!/bin/bash
# submit_gw_jobs.sh - Submit and monitor the GW analysis jobs
#SBATCH --partition=amdpreq  # Adjust to your cluster's partition name

# Configuration
OUTPUT_DIR="/gpfs/nchugh/gw/results"
DATA_DIR="/gpfs/nchugh/groupcat"
LOG_DIR="/gpfs/nchugh/gw/logs"

# Create necessary directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

# Display job information
echo "=== GW Siren Analysis Job Submission ==="
echo "Total jobs: 2400"
echo "Snapshots: 10 (50, 55, 60, 65, 70, 75, 80, 85, 90, 99)"
echo "Delta_l values: 4 (0.9 to 1.5)"
echo "Delta_h values: 4 (4.0 to 2.0)" 
echo "M_c values: 15 (0.1 to 100, logspaced)"
echo "Output directory: $OUTPUT_DIR"
echo "Log directory: $LOG_DIR"
echo ""

# Submit job arrays in chunks of 800
ARRAY_SIZE=800
TOTAL_JOBS=2400

for start in 1 801 1601
do
  end=$((start + ARRAY_SIZE - 1))
  if [ $end -gt $TOTAL_JOBS ]; then
    end=$TOTAL_JOBS
  fi

  echo "Submitting job array $start-$end..."
  JOB_ID=$(sbatch --parsable --array=${start}-${end} gw_analysis.sh)

  if [ $? -eq 0 ]; then
    echo "Jobs $start-$end submitted successfully!"
    echo "Job ID: $JOB_ID"
    echo ""
  else
    echo "Failed to submit jobs $start-$end"
    exit 1
  fi
done

echo "All job arrays submitted."

if [ $? -eq 0 ]; then
    echo "Jobs submitted successfully!"
    echo "Job ID: $JOB_ID"
    echo ""
    
    # Display monitoring commands
    echo "=== Monitoring Commands ==="
    echo "Check job status:     squeue -j $JOB_ID"
    echo "Check running jobs:   squeue -j $JOB_ID -t RUNNING"
    echo "Check pending jobs:   squeue -j $JOB_ID -t PENDING"
    echo "Check failed jobs:    sacct -j $JOB_ID --state=FAILED"
    echo "Cancel all jobs:      scancel $JOB_ID"
    echo ""
    
    # Create monitoring script
    cat > monitor_jobs.sh << EOF
#!/bin/bash
# Auto-generated monitoring script for job $JOB_ID

echo "=== Job Status Summary ==="
echo "Job ID: $JOB_ID"
echo "Total jobs: 2400"
echo ""

echo "Current status:"
squeue -j $JOB_ID --format="%.10i %.8T %.10M %.10l %.6D %.20S" | head -20

echo ""
echo "Job state counts:"
squeue -j $JOB_ID --format="%.8T" --noheader | sort | uniq -c

echo ""
echo "Completed jobs:"
COMPLETED=\$(find $OUTPUT_DIR -name "siren_cat_*.pkl" | wc -l)
echo "\$COMPLETED / 2400 completed (\$(echo "scale=1; \$COMPLETED * 100 / 2400" | bc)%)"

echo ""
echo "Failed jobs (if any):"
sacct -j $JOB_ID --state=FAILED --format="JobID,State,ExitCode" --noheader | head -10
EOF
    
    chmod +x monitor_jobs.sh
    echo "Created monitoring script: monitor_jobs.sh"
    echo ""
    
    # Create resubmission script for failed jobs
    cat > resubmit_failed.sh << EOF
#!/bin/bash
# Auto-generated resubmission script for failed jobs

echo "Checking for failed jobs..."
FAILED_JOBS=\$(sacct -j $JOB_ID --state=FAILED --format="JobID" --noheader | grep -oE "[0-9]+_[0-9]+" | cut -d'_' -f2)

if [ -z "\$FAILED_JOBS" ]; then
    echo "No failed jobs found!"
else
    echo "Found failed jobs. Creating resubmission array..."
    FAILED_ARRAY=\$(echo "\$FAILED_JOBS" | tr '\n' ',' | sed 's/,\$//')
    echo "Failed job indices: \$FAILED_ARRAY"
    
    # Resubmit only failed jobs
    sbatch --array=\$FAILED_ARRAY gw_analysis.sbatch
    echo "Resubmitted failed jobs"
fi
EOF
    
    chmod +x resubmit_failed.sh
    echo "Created resubmission script: resubmit_failed.sh"
    echo ""
    
    # Display efficiency tips
    echo "=== Efficiency Tips ==="
    echo "1. Monitor progress: ./monitor_jobs.sh"
    echo "2. Check logs: tail -f $LOG_DIR/job_${JOB_ID}_*.err"
    echo "3. Resubmit failed jobs: ./resubmit_failed.sh"
    echo "4. Expected runtime: ~1-2 hours per job"
    echo "5. Total expected data: ~24GB (10MB per catalog)"
    
else
    echo "Job submission failed!"
    exit 1
fi