#!/bin/sh 
### General options 
### -- specify queue -- 
#BSUB -q gpuv100
### -- set the job Name -- 
#BSUB -J hpc_job
### -- ask for number of cores (default: 1) -- 
#BSUB -n 2
### -- specify that the cores must be on the same host -- 
#BSUB -R "span[hosts=1]"
### -- For requesting a GPU with 32GB of memory, then please add a
#BSUB -R "select[gpu32gb]"
### -- specify that we need 2GB of memory per core/slot -- 
#BSUB -R "rusage[mem=32GB]"
### -- set walltime limit: hh:mm -- 
#BSUB -W 24:00
#BSUB -gpu "num=1"
### -- set the email address -- 
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
##BSUB -u
### -- send notification at start -- 
#BSUB -B
### -- send notification at completion -- 
#BSUB -N 
### -- Specify the output and error file. %J is the job-id -- 
### -- -o and -e mean append, -oo and -eo mean overwrite -- 
#BSUB -o out_hpc_%J.out 
#BSUB -e err_hpc_%J.err 

# here follow the commands you want to execute 
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/zhome/8a/1/127380/miniconda3/lib/

source ~/miniconda3/bin/activate
conda activate speciale

python3 main.py --dataset synthetic --maxiter_gibbs 100 --Nc 100 --K 10 --model_type nonparametric --noc 50
