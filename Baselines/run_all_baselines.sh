#!/bin/bash
# Master script to run complete baseline pipeline:
# 1. Create train/test splits
# 2. Train all models
# 3. Predict on test sets
# 4. Evaluate on test sets
# 5. Generate summary tables

set -e

PROJECT=/home/mahsa.geshvadi001/New_Projects/PathFollowing_RL/Baselines
cd $PROJECT

echo "=========================================="
echo "BASELINE EXPERIMENT PIPELINE"
echo "=========================================="
echo ""

# Step 1: Create train/test splits
echo "STEP 1: Creating train/test splits..."
echo "----------------------------------------"
python create_train_test_splits.py
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to create splits"
    exit 1
fi
echo ""

# Step 2: Train all models
echo "STEP 2: Training all models..."
echo "----------------------------------------"
echo "Submitting training jobs..."
echo ""

# Submit all training jobs
sbatch baseline_drive_unet.sbatch
sbatch baseline_drive_unetpp.sbatch
sbatch baseline_isbi12_unet.sbatch
sbatch baseline_isbi12_unetpp.sbatch
sbatch baseline_crack_unet.sbatch
sbatch baseline_crack_unetpp.sbatch

echo ""
echo "Training jobs submitted. Check status with: squeue -u \$USER"
echo "Wait for training to complete before proceeding to Step 3."
echo ""
echo "To continue after training completes:"
echo "  sbatch predict_baseline.slurm    # Step 3: Predictions"
echo "  sbatch eval_baseline.slurm       # Step 4: Evaluation"
echo "  python generate_baseline_table.py  # Step 5: Generate tables"
echo ""
echo "Or run steps 3-5 manually after training completes."
