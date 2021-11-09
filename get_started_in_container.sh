#!/bin/sh
export PATH="/home/user/miniconda/bin/conda:$PATH"
conda create --name fast python=3.8 -y
./home/user/miniconda/bin/source fast
conda activate fast
pip install -r /home/user/fastMRI/dev-requirements.txt