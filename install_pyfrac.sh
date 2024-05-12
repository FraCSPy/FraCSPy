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
source $CONDA_PREFIX/etc/profile.d/conda.sh
conda activate pyfrac
pip install -e .
conda env list
echo 'Created and activated environment:' $(which python)

# check cupy works as expected
echo 'Checking pyrfrac version...'
python -c 'import pyfrac; print(pyfrac.__version__)'

echo 'Done!'


