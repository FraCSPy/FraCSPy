#!/bin/bash
# 
# Installer for pyfrac
# 
# Run: ./install_pyfrac.sh
# 
# C. Birnie, 13/04/2023

echo 'Creating pyfrac environment'

# create conda env
conda env create -f environment.yml
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pyfrac
pip install -e .
conda env list
echo 'Created and activated environment:' $(which python)

# install cusignal (optional)
conda install -c rapidsai -c nvidia -c conda-forge cusignal=21.08 -y

# check cupy works as expected
echo 'Checking cupy version and running a command...'
python -c 'import pyfrac; import cupy as cp; print(cp.__version__); import torch; print(torch.__version__);  print(torch.cuda.get_device_name(torch.cuda.current_device())); print(torch.ones(10).to("cuda:0"))'

echo 'Done!'


