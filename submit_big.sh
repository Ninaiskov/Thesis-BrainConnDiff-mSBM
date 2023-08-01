#!/bin/sh
#BSUB -J big_job
#BSUB -q computebigbigmem
#BSUB -R "rusage[mem=10GB]"
#BSUB -B
#BSUB -N
#BSUB -o out_big_%J.txt
#BSUB -e err_big_%J.txt
#BSUB -W 168:00
#BSUB -n 2
#BSUB -R "span[hosts=1]"

# -- commands you want to execute -- 
source ~/miniconda3/bin/activate
conda activate speciale

python3 finalresults.py
