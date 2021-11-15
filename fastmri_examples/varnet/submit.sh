#!/bin/bash
#SBATCH --job-name=fastMRI
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=2
#SBATCH --mem=24000
#SBATCH --gres=gpu:4
#SBATCH -o /home/%u/%j-%x-on-%N.out
#SBATCH -e /home/%u/%j-%x-on-%N.err
#SBATCH --mail-type=ALL
#Timelimit format: "hours:minutes:seconds" -- max is 24h
#SBATCH --time=24:00:00

# Tell's pipenv to install the virtualenvs in the cluster folder
export WORKON_HOME=/cluster/`whoami`/.cache
export XDG_CACHE_DIR=/cluster/`whoami`/.cache
export PYTHONUSERBASE=/cluster/`whoami`/.python_packages
export PATH=/cluster/folle/miniconda/bin:$PATH

echo "Your job is running on" $(hostname)

conda init bash
source /home/folle/.bashrc
conda activate sr
pip install dev-requirements.txt
python train_varnet_demo.py