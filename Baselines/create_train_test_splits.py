#!/usr/bin/env python3
"""
Create train/test splits for datasets that don't have separate test sets.
Splits isbi12 and crack into train/test (80/20) while preserving 1-to-1 correspondence.
"""
import os
import shutil
import random
from pathlib import Path

def create_split(dataset_name, train_ratio=0.8, seed=42):
    """Create train/test split for a dataset."""
    script_dir = Path(__file__).parent.absolute()
    dataset_dir = script_dir / "Dataset" / dataset_name
    
    train_dir = script_dir / "Dataset" / f"{dataset_name}_train"
    test_dir = script_dir / "Dataset" / f"{dataset_name}_test"
    
    images_dir = dataset_dir / "images"
    masks_dir = dataset_dir / "masks"
    
    if not images_dir.exists() or not masks_dir.exists():
        print(f"ERROR: Dataset {dataset_name} not found at {dataset_dir}")
        return False
    
    # Get all image files
    image_files = sorted(images_dir.glob("*.png"))
    mask_files = sorted(masks_dir.glob("*.png"))
    
    # Verify 1-to-1 correspondence
    img_stems = {f.stem for f in image_files}
    mask_stems = {f.stem for f in mask_files}
    
    if img_stems != mask_stems:
        print(f"ERROR: Mismatch in {dataset_name} - {len(img_stems)} images vs {len(mask_stems)} masks")
        return False
    
    # Create pairs
    pairs = sorted(img_stems)
    n_total = len(pairs)
    n_train = int(n_total * train_ratio)
    n_test = n_total - n_train
    
    # Shuffle with seed for reproducibility
    random.seed(seed)
    shuffled = pairs.copy()
    random.shuffle(shuffled)
    
    train_pairs = set(shuffled[:n_train])
    test_pairs = set(shuffled[n_train:])
    
    # Create directories
    train_images = train_dir / "images"
    train_masks = train_dir / "masks"
    test_images = test_dir / "images"
    test_masks = test_dir / "masks"
    
    for d in [train_images, train_masks, test_images, test_masks]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Copy files
    train_count = 0
    test_count = 0
    
    for stem in pairs:
        img_src = images_dir / f"{stem}.png"
        mask_src = masks_dir / f"{stem}.png"
        
        if stem in train_pairs:
            shutil.copy2(img_src, train_images / f"{stem}.png")
            shutil.copy2(mask_src, train_masks / f"{stem}.png")
            train_count += 1
        else:
            shutil.copy2(img_src, test_images / f"{stem}.png")
            shutil.copy2(mask_src, test_masks / f"{stem}.png")
            test_count += 1
    
    print(f"  {dataset_name}:")
    print(f"    Train: {train_count} pairs -> {train_dir}")
    print(f"    Test:  {test_count} pairs -> {test_dir}")
    
    return True

def main():
    print("Creating train/test splits...")
    print("=" * 60)
    
    # DRIVE already has drive_test, so skip
    # Create splits for isbi12 and crack
    success = True
    success &= create_split("isbi12", train_ratio=0.8, seed=42)
    success &= create_split("crack", train_ratio=0.8, seed=42)
    
    print("=" * 60)
    if success:
        print("✓ All splits created successfully!")
        print("\nDataset structure:")
        print("  drive/          -> train set (use drive)")
        print("  drive_test/     -> test set (already exists)")
        print("  isbi12_train/   -> train set (new)")
        print("  isbi12_test/    -> test set (new)")
        print("  crack_train/    -> train set (new)")
        print("  crack_test/     -> test set (new)")
    else:
        print("✗ Some splits failed. Check errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
