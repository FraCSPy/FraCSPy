#!/bin/bash
# 
# Installer for fracspy
# 
# Run: ./install.sh
# 
# C. Birnie, 13/04/2023
# Updated by D. Anikiev 30/05/2024

ENV_YAML=environment.yml
ENV_NAME=fracspy
PACKAGE_NAME=fracspy

echo 'Creating $(ENV_NAME) environment'

# create conda env
conda env create -f $ENV_YAML
source $CONDA_PREFIX/etc/profile.d/conda.sh
conda activate $ENV_NAME
pip install -e .
conda env list
echo 'Created and activated environment $(ENV_YAML):' $(which python)

# Check
echo 'Checking $(PACKAGE_NAME) version...'
python -c 'import fracspy as fp; print(fp.__version__)'

echo 'Done!'


