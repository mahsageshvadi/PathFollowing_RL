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

# Evaluate on TEST sets (for paper reporting)
# Datasets: drive_test, isbi12_test, crack_test
# Models: unet, unetpp
# Predictions structure: predictions/{dataset}_test/{model}/
# Metrics output: metrics/{dataset}_test/{model}/metrics.csv

for DATASET in drive_test isbi12_test crack_test; do
    for MODEL in unet unetpp; do
        echo ""
        echo "===== EVALUATING (TEST) ====="
        echo "Dataset: $DATASET"
        echo "Model:   $MODEL"
        
        PRED_DIR=$PREDICTIONS/$DATASET/$MODEL
        GT_ROOT=$DATA/$DATASET
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
        
        if [ ! -d "$GT_ROOT" ]; then
            echo "WARNING: Ground truth directory not found: $GT_ROOT"
            echo "Skipping $DATASET/$MODEL"
            continue
        fi
        
        mkdir -p $(dirname $OUT_CSV)
        
        python eval_masks.py \
          --gt_root $GT_ROOT \
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
echo "    drive_test/unet/metrics.csv"
echo "    drive_test/unetpp/metrics.csv"
echo "    isbi12_test/unet/metrics.csv"
echo "    isbi12_test/unetpp/metrics.csv"
echo "    crack_test/unet/metrics.csv"
echo "    crack_test/unetpp/metrics.csv"
echo ""
echo "Run generate_baseline_table.py to create summary table"
