#!/bin/bash
# Run this at the start of every Devon session:
#   source /storage/eg99/neuroprobe/devon_init.sh

export HOME=/storage/eg99
export CONDARC=/storage/eg99/.condarc
source /storage/eg99/anaconda3/etc/profile.d/conda.sh
conda activate neuroprobe

echo "Environment ready: $(which python)"
