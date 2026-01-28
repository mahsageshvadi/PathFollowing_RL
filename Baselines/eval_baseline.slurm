#!/bin/bash
#SBATCH --job-name=baseline_eval
#SBATCH --partition=DGXA100
#SBATCH --account=pi_funda.durupinarbabur
#SBATCH --qos=scavenger

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00

#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err

# ----------------------------
# Paths (EDIT THESE)
# ----------------------------

source ~/.bashrc
conda activate RL

PROJECT=/home/mahsa.geshvadi001/New_Projects/PathFollowing_RL/Baselines
DATA=$PROJECT/Dataset
PREDICTIONS=$PROJECT/predictions
METRICS=$PROJECT/metrics

mkdir -p $METRICS
mkdir -p logs

# ----------------------------
# Run evaluation for all combinations
# ----------------------------

cd $PROJECT/seg_baselines

# Datasets: drive, isbi12, crack
# Models: unet, unetpp
# Predictions structure: predictions/{dataset}/{model}/
# Metrics output: metrics/{dataset}/{model}/metrics.csv

for DATASET in drive isbi12 crack; do
    for MODEL in unet unetpp; do
        echo ""
        echo "===== EVALUATING ====="
        echo "Dataset: $DATASET"
        echo "Model:   $MODEL"
        
        PRED_DIR=$PREDICTIONS/$DATASET/$MODEL
        OUT_CSV=$METRICS/$DATASET/$MODEL/metrics.csv
        
        if [ ! -d "$PRED_DIR" ]; then
            echo "WARNING: Predictions directory not found: $PRED_DIR"
            echo "Skipping $DATASET/$MODEL"
            continue
        fi
        
        if [ -z "$(ls -A $PRED_DIR 2>/dev/null)" ]; then
            echo "WARNING: Predictions directory is empty: $PRED_DIR"
            echo "Skipping $DATASET/$MODEL"
            continue
        fi
        
        mkdir -p $(dirname $OUT_CSV)
        
        python eval_masks.py \
          --gt_root $DATA/$DATASET \
          --pred_dir $PRED_DIR \
          --out_csv $OUT_CSV
        
        echo "Saved metrics to: $OUT_CSV"
        echo "===== DONE: $DATASET/$MODEL ====="
    done
done

echo ""
echo "===== ALL EVALUATIONS COMPLETE ====="
echo "Metrics saved to: $METRICS"
echo "Structure:"
echo "  $METRICS/"
echo "    drive/unet/metrics.csv"
echo "    drive/unetpp/metrics.csv"
echo "    isbi12/unet/metrics.csv"
echo "    isbi12/unetpp/metrics.csv"
echo "    crack/unet/metrics.csv"
echo "    crack/unetpp/metrics.csv"
