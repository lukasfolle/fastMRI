#!/bin/bash -l
#SBATCH --job-name=fastMRI
#SBATCH --export=NONE
#SBATCH --time=24:00:00
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --dependency=afterany:100
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

mkdir -p /scratch/fastMRI
mkdir -p /scratch/fastMRI/multicoil_train
mkdir -p /scratch/fastMRI/multicoil_val

tar --skip-old-files -xf /lustre/iwi5/iwi5044h/multicoil_train.tar -C /scratch/fastMRI/multicoil_train --strip-components 6 &
tar --skip-old-files -xf /lustre/iwi5/iwi5044h/multicoil_val.tar -C /scratch/fastMRI/multicoil_val --strip-components 6 &

wait
echo "Done unpacking archives to lustre"

cd /home/woody/iwi5/iwi5044h/Code/fastMRI
source /home/hpc/iwi5/iwi5044h/.bashrc
conda activate fast
export PYTHONPATH="/home/woody/iwi5/iwi5044h/Code/fastMRI:$PYTHONPATH"
python fastmri_examples/varnet/train_varnet_4d_fastmri.py

