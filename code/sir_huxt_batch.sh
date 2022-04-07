#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --threads-per-core=1

#SBATCH --job-name=sir_huxt_random_background
#SBATCH --output=sir_huxt_rndbkg_out.txt

#SBATCH --time=48:00:00

module load python3/anaconda/5.1.0
source activate myenv
python SIR_HUXt_dev.py
