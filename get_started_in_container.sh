#!/bin/bash -l
#SBATCH --job-name=fastMRI
#SBATCH --export=NONE
#SBATCH --time=24:00:00
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --signal=SIGUSR1@90
unset SLURM_EXPORT_ENV 

cd /home/woody/iwi5/iwi5044h/Code/fastMRI
source /home/hpc/iwi5/iwi5044h/.bashrc
conda activate fast
export PYTHONPATH="/home/woody/iwi5/iwi5044h/Code/fastMRI:$PYTHONPATH"
nvidia-smi --query-gpu=timestamp,name,pstate,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv -l 1 > /home/woody/iwi5/iwi5044h/Code/results-file.csv &
python fastmri_examples/varnet/train_varnet_4d.py