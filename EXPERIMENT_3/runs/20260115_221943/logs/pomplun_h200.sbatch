#!/bin/bash
#SBATCH --job-name=pomplun_train
#SBATCH --partition=pomplun
#SBATCH --account=cs_funda.durupinarbabur

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --time=00-12:00:00
#SBATCH --gres=gpu:1

#SBATCH --output=pomplun_%j.out
#SBATCH --error=pomplun_%j.err

echo "===== JOB INFO ====="
echo "JobID:   $SLURM_JOB_ID"
echo "Node:    $SLURMD_NODENAME"
echo "CPUs:    $SLURM_CPUS_PER_TASK"
echo "GPUs:    $SLURM_GPUS"
echo "Account: $SLURM_JOB_ACCOUNT"
echo "===================="

echo
echo "===== NVIDIA-SMI ====="
nvidia-smi
echo "======================"

# --- Activate environment ---
source ~/.bashrc
conda activate RL

# --- Go to project ---
cd $HOME/New_Projects/DSA-RL-Tracker/EXPERIMENTS/Old_Generator/EXPERIMENT_3/src/

# --- Run training ---
python train_deeper_model.py

