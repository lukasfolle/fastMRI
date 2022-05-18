#!/bin/bash -l
#SBATCH --job-name=fastMRI
#SBATCH --export=NONE
#SBATCH --time=24:00:00
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:8
#SBATCH --dependency=afterany:43794
#SBATCH --begin=now+0hour

unset SLURM_EXPORT_ENV 

module load gcc
module load git
module load cuda
module load cudnn
module load mkl

mkdir -p /scratch/cache
mkdir -p /scratch/cache/cache_train
mkdir -p /scratch/cache/cache_val

#echo "Clearing cache at /scratch/cache/"
#rm -rf /scratch/cache/cache_train/*
#rm -rf /scratch/cache/cache_val/*

cd /home/woody/iwi5/iwi5044h/Code/fastMRI
source /home/hpc/iwi5/iwi5044h/.bashrc
conda activate fast
export PYTHONPATH="/home/woody/iwi5/iwi5044h/Code/fastMRI:$PYTHONPATH"
python fastmri_examples/varnet/train_varnet_4d.py --accelerations=6 --loss=combined
