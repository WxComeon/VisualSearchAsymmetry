#!/bin/bash
# fmriprep job.
#
# Replace <ACCOUNT> with your account name before submitting.
#
#SBATCH -A nklab          # Set Account name
#SBATCH --job-name=batch  # The job name
#SBATCH --gres=gpu:1
#SBATCH -c 6                   # Number of cores
#SBATCH --time=03-00:00
#SBATCH --exclude=ax[03-19]
#SBATCH --mail-user=wg2361@columbia.edu
#SBATCH --mail-type=ALL

cd /home/wg2361/VisualSearchAsymmetry

source activate vsa_nips_klab
python -u metamers/generate_metamers.py --i_batch $i_batch --batch_size 20 --layer_num $layer