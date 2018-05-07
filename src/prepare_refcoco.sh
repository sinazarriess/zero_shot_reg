#!/usr/bin/env bash
export PATH=/media/compute/vol/dsg/lilian/anaconda2/bin${PATH:+:${PATH}}
export PATH=/media/compute/vol/cuda/9.0/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/media/compute/vol/cuda/9.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export CUDA_HOME=/media/compute/vol/cuda/9.0
# activate conda env with tensorflow:
source /media/compute/vol/dsg/lilian/anaconda2/bin/activate tf-lilian
python prepare_refcoco.py
