#!/bin/bash
# fmriprep job.
#
# Replace <ACCOUNT> with your account name before submitting.
#
#SBATCH -A nklab          # Set Account name
#SBATCH --job-name=batch  # The job name
#SBATCH -c 1                   # Number of cores
#SBATCH -t 00:05              # Runtime in D-HH:MM

for (( i=16; i>=1; i-- )); do
    sbatch --export=ALL,vda=$i run_task.sh
done