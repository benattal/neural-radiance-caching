#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --partition=gpunodes
#SBATCH --nodelist=calypso
#SBATCH --gres=gpu:rtx_a6000:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=100G
#SBATCH --output=/scratch/ondemand28/battal/active-yobo/logs/slurm_outputs/output/job_%A_%a.out            # Standard output log (%j is replaced with the job ID)
#SBATCH --error=/scratch/ondemand28/battal/active-yobo/logs/slurm_outputs/error/job_%A_%a.err             # Standard error log (%j is replaced with the job ID)
#SBATCH --qos=priority
#SBATCH --time=12:00:00
#SBATCH --array=1-1%10

# prepare your environment here

# Activate virtual environment (if you have one)
cd /scratch/ondemand28/battal/active-yobo

eval $(sed -n "${SLURM_ARRAY_TASK_ID}p" < scripts/run_eval_material.txt)
