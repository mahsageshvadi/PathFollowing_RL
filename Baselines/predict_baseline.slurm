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

# Datasets: drive, isbi12, crack
# Models: unet, unetpp
# Checkpoint structure: results/{dataset}_unet/{dataset}/{model}/best.pth
# Note: RUN_ID is {dataset}_unet for both unet and unetpp models

for DATASET in drive isbi12 crack; do
    for MODEL in unet unetpp; do
        echo ""
        echo "===== PREDICTING ====="
        echo "Dataset: $DATASET"
        echo "Model:   $MODEL"
        
        RUN_ID=${DATASET}_unet
        CKPT_PATH=$RESULTS/$RUN_ID/$DATASET/$MODEL/best.pth
        OUT_MASKS=$PREDICTIONS/$DATASET/$MODEL
        
        if [ ! -f "$CKPT_PATH" ]; then
            echo "WARNING: Checkpoint not found: $CKPT_PATH"
            echo "Skipping $DATASET/$MODEL"
            continue
        fi
        
        mkdir -p $OUT_MASKS
        
        python predict.py \
          --dataset_root $DATA/$DATASET \
          --model $MODEL \
          --ckpt $CKPT_PATH \
          --out_masks $OUT_MASKS \
          --resize 512 512 \
          --batch 8
        
        echo "Saved predictions to: $OUT_MASKS"
        echo "===== DONE: $DATASET/$MODEL ====="
    done
done

echo ""
echo "===== ALL PREDICTIONS COMPLETE ====="
echo "Predictions saved to: $PREDICTIONS"
echo "Structure:"
echo "  $PREDICTIONS/"
echo "    drive/unet/"
echo "    drive/unetpp/"
echo "    isbi12/unet/"
echo "    isbi12/unetpp/"
echo "    crack/unet/"
echo "    crack/unetpp/"