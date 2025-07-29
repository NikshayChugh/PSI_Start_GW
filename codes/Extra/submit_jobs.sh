#!/bin/bash

#SBATCH --partition=amdpreq


# Loop over file indices in steps of 10
for (( i=0; i<=599; i+=10 ))
do
    START=$i
    END=$(( i+9 ))
    if (( END > 599 )); then
        END=599
    fi

    # Submit job with START and END as environment variables on amdq partition
    sbatch --partition=amdpreq --export=START=$START,END=$END run_cic.sh

    echo "Submitted job for files $START to $END on amdq partition"
done