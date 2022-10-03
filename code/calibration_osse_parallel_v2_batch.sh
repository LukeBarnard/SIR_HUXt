#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --threads-per-core=1
#SBATCH --partition=long

#SBATCH --job-name=calibration_osse
#SBATCH --output=calibration_osse_report.txt

#SBATCH --time=50:00:00

module load anaconda
source activate sir_huxt
python calibration_osse_parallel_v2.py
