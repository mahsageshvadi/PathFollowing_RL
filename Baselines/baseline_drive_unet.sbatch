#!/bin/bash
#SBATCH --job-name=scavenger_gpu_test
#SBATCH --partition=DGXA100
#SBATCH --account=pi_funda.durupinarbabur
#SBATCH --qos=scavenger

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=02:00:00

#SBATCH --output=scavenger_gpu_%j.out
#SBATCH --error=scavenger_gpu_%j.err

# ----------------------------
# Paths (EDIT THESE)
# ----------------------------

source ~/.bashrc
conda activate RL

# Base paths
BASE=/home/mahsa.geshvadi001/New_Projects/PathFollowing_RL/Baselines
PROJECT=$BASE/seg_baselines
DATA=$BASE/Dataset
OUT_ROOT=$BASE/results

# ----------------------------
# Experiment config
# ----------------------------

DATASET=drive          # drive | isbi12_train | crack_train
MODEL=unet             # unet | unetpp

# One RUN_ID per dataset (contains both unet + unetpp)
# For drive, use "drive" as RUN_ID (no _train suffix)
RUN_ID=drive_unet
OUT=$OUT_ROOT/$RUN_ID

mkdir -p $OUT/drive/$MODEL
mkdir -p logs

# ----------------------------
# Train
# ----------------------------

echo
echo "===== TRAINING ====="
echo "Dataset: drive (train set)"
echo "Model:   $MODEL"
echo "Output:  $OUT/drive/$MODEL"

cd $PROJECT

python train.py \
  --dataset_root $DATA/drive \
  --model $MODEL \
  --outdir $OUT/drive/$MODEL \
  --epochs 80 \
  --batch 8 \
  --lr 1e-3 \
  --resize 512 512 \
  --seed 0

echo "===== TRAINING DONE ====="