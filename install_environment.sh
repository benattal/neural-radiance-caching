#!/bin/bash

# Install main environment
source ~/.bashrc
conda env create -f environment.yml

eval "$(conda shell.bash hook)"
conda activate yobo

pip install -r requirements.txt

# Install eval environment
conda deactivate
cd evaluation
conda env create -f environment.yml
pip install .
conda deactivate

# Activate
conda activate yobo