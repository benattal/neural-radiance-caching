#!/bin/bash
#SBATCH --job-name=training
#SBATCH --partition=gpunodes
#SBATCH --gres=gpu:rtx_6000_ada:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=100G
#SBATCH --output=/scratch/ondemand28/battal/active-yobo/checkpoints/yobo_results/synthetic/final_res_sim/slurm_outputs/output/job_%A_%a.out            # Standard output log (%j is replaced with the job ID)
#SBATCH --error=/scratch/ondemand28/battal/active-yobo/checkpoints/yobo_results/synthetic/final_res_sim/slurm_outputs/error/job_%A_%a.err             # Standard error log (%j is replaced with the job ID)
#SBATCH --time=72:00:00
#SBATCH --array=3-15%10

# prepare your environment here

# Activate virtual environment (if you have one)
cd /scratch/ondemand28/battal/active-yobo

eval $(sed -n "${SLURM_ARRAY_TASK_ID}p" < scripts/run_training.txt)


