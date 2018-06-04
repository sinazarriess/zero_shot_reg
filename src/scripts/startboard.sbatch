#!/usr/bin/env bash

export PATH=/media/compute/vol/dsg/lilian/anaconda2/bin${PATH:+:${PATH}}
export PATH=/media/compute/vol/cuda/9.0/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/media/compute/vol/cuda/9.0/lib64${LD_LIBRARY_PATH:+:${LD
_LIBRARY_PATH}}
export CUDA_HOME=/media/compute/vol/cuda/9.0
# activate conda env with tensorflow:
source /media/compute/vol/dsg/lilian/anaconda2/bin/activate tf-lilian

#SBATCH --mail-user=lschroed@techfak.uni-bielefeld.de
#SBATCH --mail-type=END
# commands that should be scheduled
srun tensorboard --logdir="../exp/first/results/inject_refcoco_refrnn_compositional_3_512_1/logs/"
