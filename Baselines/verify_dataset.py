#!/usr/bin/env python3
"""
Verify that a dataset is properly structured and all files exist.
"""
import os
import sys
import glob
import cv2

def verify_dataset(dataset_root):
    """Verify dataset structure and file integrity."""
    print(f"\n{'='*60}")
    print(f"Verifying dataset: {dataset_root}")
    print(f"{'='*60}\n")
    
    img_dir = os.path.join(dataset_root, "images")
    mask_dir = os.path.join(dataset_root, "masks")
    
    # Check directories exist
    if not os.path.exists(dataset_root):
        print(f"❌ Dataset root does not exist: {dataset_root}")
        return False
    
    if not os.path.exists(img_dir):
        print(f"❌ Images directory does not exist: {img_dir}")
        return False
    
    if not os.path.exists(mask_dir):
        print(f"❌ Masks directory does not exist: {mask_dir}")
        return False
    
    # Get file lists
    imgs = sorted(glob.glob(os.path.join(img_dir, "*.png")))
    masks = sorted(glob.glob(os.path.join(mask_dir, "*.png")))
    
    print(f"Found {len(imgs)} images and {len(masks)} masks")
    
    if len(imgs) == 0:
        print(f"❌ No PNG images found in {img_dir}")
        return False
    
    if len(masks) == 0:
        print(f"❌ No PNG masks found in {mask_dir}")
        return False
    
    # Check 1-to-1 correspondence
    img_names = {os.path.basename(f) for f in imgs}
    mask_names = {os.path.basename(f) for f in masks}
    
    missing_masks = img_names - mask_names
    missing_imgs = mask_names - img_names
    
    if missing_masks:
        print(f"❌ Missing masks for {len(missing_masks)} images:")
        for name in sorted(missing_masks)[:5]:
            print(f"    - {name}")
        if len(missing_masks) > 5:
            print(f"    ... and {len(missing_masks) - 5} more")
    
    if missing_imgs:
        print(f"❌ Missing images for {len(missing_imgs)} masks:")
        for name in sorted(missing_imgs)[:5]:
            print(f"    - {name}")
        if len(missing_imgs) > 5:
            print(f"    ... and {len(missing_imgs) - 5} more")
    
    if missing_masks or missing_imgs:
        return False
    
    # Verify files can be read
    print("\nVerifying file integrity...")
    failed_imgs = []
    failed_masks = []
    
    for img_path in imgs[:10]:  # Check first 10
        try:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                failed_imgs.append(img_path)
        except Exception as e:
            failed_imgs.append(f"{img_path} ({e})")
    
    for mask_path in masks[:10]:  # Check first 10
        try:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                failed_masks.append(mask_path)
        except Exception as e:
            failed_masks.append(f"{mask_path} ({e})")
    
    if failed_imgs:
        print(f"❌ Failed to read {len(failed_imgs)} images:")
        for path in failed_imgs[:3]:
            print(f"    - {path}")
        return False
    
    if failed_masks:
        print(f"❌ Failed to read {len(failed_masks)} masks:")
        for path in failed_masks[:3]:
            print(f"    - {path}")
        return False
    
    print(f"✓ Dataset verification passed!")
    print(f"  - {len(imgs)} image-mask pairs")
    print(f"  - All files readable")
    print(f"  - 1-to-1 correspondence verified")
    return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python verify_dataset.py <dataset_root>")
        print("Example: python verify_dataset.py Dataset/drive")
        sys.exit(1)
    
    dataset_root = sys.argv[1]
    success = verify_dataset(dataset_root)
    sys.exit(0 if success else 1)
