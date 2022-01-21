#!/bin/bash -l
#SBATCH --job-name=fastMRI
#SBATCH --export=NONE
#SBATCH --time=24:00:00
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:8
#SBATCH --dependency=afterany:20
#SBATCH --begin=now+0hour

unset SLURM_EXPORT_ENV 

mkdir -p /scratch/cache
mkdir -p /scratch/cache/cache_train
mkdir -p /scratch/cache/cache_val

# module load python/3.9-anaconda

cd /home/woody/iwi5/iwi5044h/Code/fastMRI
source /home/hpc/iwi5/iwi5044h/.bashrc
conda activate fast
export PYTHONPATH="/home/woody/iwi5/iwi5044h/Code/fastMRI:$PYTHONPATH"
python fastmri_examples/varnet/train_varnet_4d.py --accelerations=6 --loss=combined
