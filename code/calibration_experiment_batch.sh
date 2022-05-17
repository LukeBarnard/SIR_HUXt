#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --threads-per-core=1

#SBATCH --job-name=sir_huxt_calibration
#SBATCH --output=calibration_experiment_report.txt

#SBATCH --time=24:00:00

module load anaconda
source activate sir_huxt
python calibration_experiments.py
