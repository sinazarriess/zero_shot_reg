#!/usr/bin/env bash

#
#SBATCH --mail-user=lschroed@techfak.uni-bielefeld.de
#SBATCH --mail-type=END
# commands that should be scheduled
srun tensorboard --logdir="../logs/"

