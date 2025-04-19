#!/bin/bash

#SBATCH --job-name=RunReAlign
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:rtx:2
#SBATCH --mem=8G
#SBATCH --output=logs/%j.log

ml purge
ml GCCcore/13.3.0
ml Miniconda3/23.10.0-1
source /sw/eb/sw/Miniconda3/23.10.0-1/etc/profile.d/conda.sh
conda activate re-align

cd $SCRATCH/Re-Align
chmod +x train_lora_rdpo.sh

echo "Starting Main!!!"
ls -lh



echo "Done!!!"
