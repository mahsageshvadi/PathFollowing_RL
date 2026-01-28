#!/usr/bin/env python3
"""
Prepare DRIVE dataset into standard structure with train and test sets.
Works on both macOS and Linux clusters.

Input (original DRIVE layout, untouched):
  Dataset/DRIVE/
    training/images/*.tif
    training/1st_manual/*_manual1.gif   (GT for train)
    training/mask/*_training_mask.gif   (alternative masks)
    test/images/*_test.tif
    test/mask/*_test_mask.gif           (GT for test)

Output (new, used by baselines):
  Dataset/drive/
    images/*.png        (TRAIN images)
    masks/*.png         (TRAIN masks)
  Dataset/drive_test/
    images/*.png        (TEST images)
    masks/*.png         (TEST masks)
"""
import os
import sys
from pathlib import Path

try:
    import cv2
    import numpy as np
    HAS_CV2 = True
except ImportError:
    try:
        from PIL import Image
        import numpy as np
        HAS_CV2 = False
        HAS_PIL = True
    except ImportError:
        print("ERROR: Need either cv2 or PIL. Install with: pip install opencv-python or pip install Pillow")
        sys.exit(1)

def convert_to_png_cv2(src_path, dst_path):
    """Convert image to PNG using cv2."""
    img = cv2.imread(str(src_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        # Try color
        img = cv2.imread(str(src_path))
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img is None:
        return False
    cv2.imwrite(str(dst_path), img)
    return True

def convert_to_png_pil(src_path, dst_path):
    """Convert image to PNG using PIL."""
    try:
        img = Image.open(src_path).convert('L')
        img.save(dst_path)
        return True
    except Exception as e:
        print(f"Error converting {src_path}: {e}")
        return False

def convert_to_png(src_path, dst_path):
    """Convert image to PNG using available library."""
    if HAS_CV2:
        return convert_to_png_cv2(src_path, dst_path)
    elif HAS_PIL:
        return convert_to_png_pil(src_path, dst_path)
    return False

def ensure_binary_mask(mask_path, output_path):
    """Convert mask to binary PNG (0/255)."""
    if HAS_CV2:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return False
        mask_binary = (mask > 0).astype(np.uint8) * 255
        cv2.imwrite(str(output_path), mask_binary)
        return True
    elif HAS_PIL:
        try:
            img = Image.open(mask_path).convert('L')
            arr = np.array(img)
            arr_binary = (arr > 0).astype(np.uint8) * 255
            img_binary = Image.fromarray(arr_binary)
            img_binary.save(output_path)
            return True
        except Exception as e:
            print(f"Error processing {mask_path}: {e}")
            return False
    return False

def main():
    script_dir = Path(__file__).parent.absolute()
    dataset_dir = script_dir / "Dataset"
    src_drive = dataset_dir / "DRIVE"
    
    train_out = dataset_dir / "drive"
    test_out = dataset_dir / "drive_test"
    
    train_out.mkdir(parents=True, exist_ok=True)
    (train_out / "images").mkdir(exist_ok=True)
    (train_out / "masks").mkdir(exist_ok=True)
    
    test_out.mkdir(parents=True, exist_ok=True)
    (test_out / "images").mkdir(exist_ok=True)
    (test_out / "masks").mkdir(exist_ok=True)
    
    print("Preparing DRIVE dataset...")
    print(f"Source:   {src_drive}")
    print(f"Train ->  {train_out}")
    print(f"Test  ->  {test_out}")
    print()
    
    if not src_drive.exists():
        print(f"ERROR: Source directory not found: {src_drive}")
        print("Make sure Dataset/DRIVE/ exists with training/ and test/ subdirectories")
        sys.exit(1)
    
    # Process TRAIN split
    print("Processing TRAIN split...")
    train_img_dir = src_drive / "training" / "images"
    train_manual_dir = src_drive / "training" / "1st_manual"
    train_mask_dir = src_drive / "training" / "mask"
    
    count_train = 0
    for img in sorted(train_img_dir.glob("*_training.tif")):
        num = img.stem.replace("_training", "")
        
        # Prefer 1st_manual if available, otherwise fall back to training/mask
        gt = train_manual_dir / f"{num}_manual1.gif"
        if not gt.exists():
            gt = train_mask_dir / f"{num}_training_mask.gif"
        
        if not gt.exists():
            print(f"WARNING: No mask found for {img.name}")
            continue
        
        out_img = train_out / "images" / f"{num}.png"
        out_msk = train_out / "masks" / f"{num}.png"
        
        if convert_to_png(img, out_img) and ensure_binary_mask(gt, out_msk):
            count_train += 1
        else:
            print(f"WARNING: Failed to convert {img.name}")
    
    print(f"  Created {count_train} train image-mask pairs.")
    print()
    
    # Process TEST split
    print("Processing TEST split...")
    test_img_dir = src_drive / "test" / "images"
    test_mask_dir = src_drive / "test" / "mask"
    
    count_test = 0
    for img in sorted(test_img_dir.glob("*_test.tif")):
        num = img.stem.replace("_test", "")
        gt = test_mask_dir / f"{num}_test_mask.gif"
        
        if not gt.exists():
            print(f"WARNING: No mask found for {img.name}")
            continue
        
        out_img = test_out / "images" / f"{num}.png"
        out_msk = test_out / "masks" / f"{num}.png"
        
        if convert_to_png(img, out_img) and ensure_binary_mask(gt, out_msk):
            count_test += 1
        else:
            print(f"WARNING: Failed to convert {img.name}")
    
    print(f"  Created {count_test} test image-mask pairs.")
    print()
    
    # Verify
    print("Verifying 1-to-1 correspondence...")
    for split_name, split_path in [("drive", train_out), ("drive_test", test_out)]:
        imgs = len(list((split_path / "images").glob("*.png")))
        msks = len(list((split_path / "masks").glob("*.png")))
        print(f"  {split_name}: {imgs} images, {msks} masks")
    
    print()
    print("DRIVE preparation complete.")

if __name__ == "__main__":
    main()
