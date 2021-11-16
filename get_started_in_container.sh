#!/bin/bash -l
#SBATCH --job-name=fastMRI
#SBATCH --export=NONE
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
unset SLURM_EXPORT_ENV 

cd /home/woody/iwi5/iwi5044h/Code/fastMRI
source /home/hpc/iwi5/iwi5044h/.bashrc
conda activate fast
export PYTHONPATH="/home/woody/iwi5/iwi5044h/Code/fastMRI:$PYTHONPATH"
python fastmri_examples/varnet/train_varnet_4d.py