# Baseline Experiments - Complete Pipeline

This directory contains scripts to train, evaluate, and report baseline segmentation models (U-Net and U-Net++) on three datasets: DRIVE, ISBI12, and CrackForest.

## Dataset Structure

After running the preparation scripts, you should have:

```
Dataset/
  drive/              # DRIVE training set (20 images)
    images/
    masks/
  drive_test/        # DRIVE test set (20 images)
    images/
    masks/
  isbi12_train/      # ISBI12 training set (80% split)
    images/
    masks/
  isbi12_test/       # ISBI12 test set (20% split)
    images/
    masks/
  crack_train/       # CrackForest training set (80% split)
    images/
    masks/
  crack_test/        # CrackForest test set (20% split)
    images/
    masks/
```

## Complete Workflow

### Step 1: Prepare Datasets

**On cluster:**
```bash
cd /home/mahsa.geshvadi001/New_Projects/PathFollowing_RL/Baselines

# Prepare DRIVE dataset (creates drive/ and drive_test/)
python prepare_drive_dataset.py

# Create train/test splits for isbi12 and crack
python create_train_test_splits.py
```

### Step 2: Train Models

Train all 6 model-dataset combinations:

```bash
sbatch baseline_drive_unet.sbatch
sbatch baseline_drive_unetpp.sbatch
sbatch baseline_isbi12_unet.sbatch
sbatch baseline_isbi12_unetpp.sbatch
sbatch baseline_crack_unet.sbatch
sbatch baseline_crack_unetpp.sbatch
```

**Checkpoints saved to:**
- `results/drive_unet/drive/unet/best.pth`
- `results/drive_unet/drive/unetpp/best.pth`
- `results/isbi12_unet/isbi12/unet/best.pth`
- `results/isbi12_unet/isbi12/unetpp/best.pth`
- `results/crack_unet/crack/unet/best.pth`
- `results/crack_unet/crack/unetpp/best.pth`

### Step 3: Generate Predictions (on Test Sets)

```bash
sbatch predict_baseline.slurm
```

**Predictions saved to:**
- `predictions/drive_test/unet/`
- `predictions/drive_test/unetpp/`
- `predictions/isbi12_test/unet/`
- `predictions/isbi12_test/unetpp/`
- `predictions/crack_test/unet/`
- `predictions/crack_test/unetpp/`

### Step 4: Evaluate (Compute Metrics)

```bash
sbatch eval_baseline.slurm
```

**Metrics saved to:**
- `metrics/drive_test/unet/metrics.csv`
- `metrics/drive_test/unetpp/metrics.csv`
- `metrics/isbi12_test/unet/metrics.csv`
- `metrics/isbi12_test/unetpp/metrics.csv`
- `metrics/crack_test/unet/metrics.csv`
- `metrics/crack_test/unetpp/metrics.csv`

Each CSV contains per-image metrics: dice, iou, cldice, bne, pred_components, gt_components, pred_holes, gt_holes, largest_comp_ratio

### Step 5: Generate Summary Tables

```bash
python generate_baseline_table.py
```

**Outputs:**
- Console: Formatted tables with all metrics
- `baseline_results_summary.csv`: CSV with mean metrics
- `baseline_results_table.tex`: LaTeX table (Dice + BNE) for paper
- `baseline_results_detailed.tex`: LaTeX table with all metrics

## Metrics Computed

### Primary Metrics (for paper):
1. **Dice Similarity Coefficient (DSC)** ↑
   - Pixel-wise segmentation accuracy
   - Range: [0, 1], higher is better

2. **Betti Number Error (BNE)** ↓
   - Topological correctness metric
   - BNE = |β₀_pred - β₀_gt| + |β₁_pred - β₁_gt|
   - β₀ = number of connected components
   - β₁ = number of holes/loops
   - Lower is better

### Additional Metrics:
3. **IoU (Intersection over Union)** ↑
   - Standard segmentation metric

4. **clDice (Centerline Dice)** ↑
   - Skeleton-based metric
   - Measures overlap of skeletonized predictions

5. **Component Statistics**
   - Number of connected components (pred vs GT)
   - Number of holes (pred vs GT)
   - Largest component ratio

## Expected Output Format

```
BASELINE RESULTS SUMMARY (TEST SETS)
================================================================================
METHOD                              | DICE ↑       | BNE ↓        | SAMPLES
--------------------------------------------------------------------------------
U-Net (DRIVE)                       |     0.8234   |     2.1000   |       20
U-Net++ (DRIVE)                     |     0.8456   |     1.8000   |       20
U-Net (ISBI12)                      |     0.7890   |     1.5000   |       12
U-Net++ (ISBI12)                    |     0.8123   |     1.2000   |       12
U-Net (CrackForest)                 |     0.7654   |     3.4000   |       43
U-Net++ (CrackForest)               |     0.7889   |     2.9000   |       43
================================================================================
```

## Quick Start (All Steps)

```bash
# Step 1: Prepare datasets
python prepare_drive_dataset.py
python create_train_test_splits.py

# Step 2: Train (submit all jobs)
sbatch baseline_drive_unet.sbatch
sbatch baseline_drive_unetpp.sbatch
sbatch baseline_isbi12_unet.sbatch
sbatch baseline_isbi12_unetpp.sbatch
sbatch baseline_crack_unet.sbatch
sbatch baseline_crack_unetpp.sbatch

# Wait for training to complete, then:

# Step 3: Predict
sbatch predict_baseline.slurm

# Step 4: Evaluate
sbatch eval_baseline.slurm

# Step 5: Generate tables
python generate_baseline_table.py
```

## Notes

- All models train on **train sets** only
- All predictions and evaluations are on **test sets** only
- Train/test splits are created with seed=42 for reproducibility
- DRIVE uses official train/test split (20 train / 20 test)
- ISBI12 and CrackForest use 80/20 split
