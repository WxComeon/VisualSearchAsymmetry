#!/bin/bash
# fmriprep job.
#
# Replace <ACCOUNT> with your account name before submitting.
#
#SBATCH -A nklab          # Set Account name
#SBATCH --job-name=batch  # The job name
#SBATCH -c 1                   # Number of cores
#SBATCH -t 00:05              # Runtime in D-HH:MM

batch_size=5
layers=('block1_conv2' 'block2_conv2') # 'block1_conv2' 'block2_conv2' 'block3_conv3'

for (( j=0; j<batch_size; j++ )); do
    for (( i=0; i<${#layers[@]}; i++ )); do
        sbatch --export=ALL,layer=${layers[$i]},i_batch=$j run_metamers.sh
    done
done