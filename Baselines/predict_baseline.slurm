#!/bin/bash
#SBATCH --job-name=baseline_predict
#SBATCH --partition=DGXA100
#SBATCH --account=pi_funda.durupinarbabur
#SBATCH --qos=scavenger

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=02:00:00

#SBATCH --output=logs/predict_%j.out
#SBATCH --error=logs/predict_%j.err

# ----------------------------
# Paths (EDIT THESE)
# ----------------------------

source ~/.bashrc
conda activate RL

PROJECT=/home/mahsa.geshvadi001/New_Projects/PathFollowing_RL/Baselines
DATA=$PROJECT/Dataset
RESULTS=$PROJECT/results
PREDICTIONS=$PROJECT/predictions

mkdir -p $PREDICTIONS
mkdir -p logs

# ----------------------------
# Run predictions for all combinations
# ----------------------------

cd $PROJECT/seg_baselines

# Datasets: drive (train), drive_test (test), isbi12, crack
# Models: unet, unetpp
# Checkpoint structure: results/{dataset}_unet/{dataset}/{model}/best.pth
# Note: RUN_ID is {dataset}_unet for both unet and unetpp models
# For drive_test, we use the checkpoint trained on drive

# Predict on TEST sets (for final evaluation)
# Mapping: drive -> drive_test, isbi12 -> isbi12_test, crack -> crack_test
echo ""
echo "===== PREDICTING ON TEST SETS ====="

# DRIVE: train on drive, test on drive_test
for MODEL in unet unetpp; do
    echo ""
    echo "===== PREDICTING (TEST) ====="
    echo "Dataset: drive_test"
    echo "Model:   $MODEL"
    
    RUN_ID=drive_unet
    CKPT_PATH=$RESULTS/$RUN_ID/drive/$MODEL/best.pth
    OUT_MASKS=$PREDICTIONS/drive_test/$MODEL
    
    if [ ! -f "$CKPT_PATH" ]; then
        echo "WARNING: Checkpoint not found: $CKPT_PATH"
        echo "Skipping drive_test/$MODEL"
        continue
    fi
    
    if [ ! -d "$DATA/drive_test" ]; then
        echo "WARNING: Test dataset not found: $DATA/drive_test"
        echo "Skipping drive_test/$MODEL"
        continue
    fi
    
    mkdir -p $OUT_MASKS
    
    python predict.py \
      --dataset_root $DATA/drive_test \
      --model $MODEL \
      --ckpt $CKPT_PATH \
      --out_masks $OUT_MASKS \
      --resize 512 512 \
      --batch 8
    
    echo "Saved predictions to: $OUT_MASKS"
    echo "===== DONE: drive_test/$MODEL ====="
done

# ISBI12: train on isbi12_train, test on isbi12_test
for MODEL in unet unetpp; do
    echo ""
    echo "===== PREDICTING (TEST) ====="
    echo "Dataset: isbi12_test"
    echo "Model:   $MODEL"
    
    RUN_ID=isbi12_unet
    CKPT_PATH=$RESULTS/$RUN_ID/isbi12/$MODEL/best.pth
    OUT_MASKS=$PREDICTIONS/isbi12_test/$MODEL
    
    if [ ! -f "$CKPT_PATH" ]; then
        echo "WARNING: Checkpoint not found: $CKPT_PATH"
        echo "Skipping isbi12_test/$MODEL"
        continue
    fi
    
    if [ ! -d "$DATA/isbi12_test" ]; then
        echo "WARNING: Test dataset not found: $DATA/isbi12_test"
        echo "Skipping isbi12_test/$MODEL"
        continue
    fi
    
    mkdir -p $OUT_MASKS
    
    python predict.py \
      --dataset_root $DATA/isbi12_test \
      --model $MODEL \
      --ckpt $CKPT_PATH \
      --out_masks $OUT_MASKS \
      --resize 512 512 \
      --batch 8
    
    echo "Saved predictions to: $OUT_MASKS"
    echo "===== DONE: isbi12_test/$MODEL ====="
done

# CRACK: train on crack_train, test on crack_test
for MODEL in unet unetpp; do
    echo ""
    echo "===== PREDICTING (TEST) ====="
    echo "Dataset: crack_test"
    echo "Model:   $MODEL"
    
    RUN_ID=crack_unet
    CKPT_PATH=$RESULTS/$RUN_ID/crack/$MODEL/best.pth
    OUT_MASKS=$PREDICTIONS/crack_test/$MODEL
    
    if [ ! -f "$CKPT_PATH" ]; then
        echo "WARNING: Checkpoint not found: $CKPT_PATH"
        echo "Skipping crack_test/$MODEL"
        continue
    fi
    
    if [ ! -d "$DATA/crack_test" ]; then
        echo "WARNING: Test dataset not found: $DATA/crack_test"
        echo "Skipping crack_test/$MODEL"
        continue
    fi
    
    mkdir -p $OUT_MASKS
    
    python predict.py \
      --dataset_root $DATA/crack_test \
      --model $MODEL \
      --ckpt $CKPT_PATH \
      --out_masks $OUT_MASKS \
      --resize 512 512 \
      --batch 8
    
    echo "Saved predictions to: $OUT_MASKS"
    echo "===== DONE: crack_test/$MODEL ====="
done

echo ""
echo "===== ALL PREDICTIONS COMPLETE ====="
echo "Predictions saved to: $PREDICTIONS"
echo "Structure (TEST SETS ONLY):"
echo "  $PREDICTIONS/"
echo "    drive_test/unet/"
echo "    drive_test/unetpp/"
echo "    isbi12_test/unet/"
echo "    isbi12_test/unetpp/"
echo "    crack_test/unet/"
echo "    crack_test/unetpp/"