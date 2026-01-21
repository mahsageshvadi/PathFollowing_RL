#!/bin/bash
#SBATCH --job-name=pomplun_h200_test
#SBATCH --partition=pomplun
#SBATCH --account=cs_funda.durupinarbabur

#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --mem=100G
#SBATCH --time=00-10:00:00
#SBATCH --gres=gpu:h200:1

#SBATCH --output=pomplun_%j.out
#SBATCH --error=pomplun_%j.err

echo "===== JOB INFO ====="
echo "JobID:   $SLURM_JOB_ID"
echo "Node:    $SLURMD_NODENAME"
echo "Tasks:   $SLURM_NTASKS"
echo "GPUs:    $SLURM_GPUS"
echo "Account: $SLURM_JOB_ACCOUNT"
echo "===================="

echo
echo "===== NVIDIA-SMI ====="
nvidia-smi
echo "======================"

source ~/.bashrc
conda activate RL

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# Go up one level from src/ to EXPERIMENT_3/
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo "Current directory: $(pwd)"
echo "Running training from: $PROJECT_DIR"

cd ~/New_Projects/DSA-RL-Tracker/EXPERIMENTS/Old_Generator/EXPERIMENT_5/

python src/train_deeper_model.py \
  --curve_config config/curve_config.json