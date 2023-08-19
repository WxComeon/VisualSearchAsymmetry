#!/bin/sh
#
#
#SBATCH --account=nklab
#SBATCH --job-name=ecc_task  # The job name.
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=01-00:00
#SBATCH --exclude=ax[03-17]

cd /home/wg2361/VisualSearchAsymmetry

source activate vsa_nips_klab

echo "Running with VDA $vda"
python -u change_detection/covert_attention_task.py --vda $vda