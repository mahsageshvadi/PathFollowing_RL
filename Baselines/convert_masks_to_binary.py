#!/usr/bin/env python3
"""
Convert all mask files to binary PNG (0/255).
Uses only standard library + numpy (if available) or basic image processing.
"""
import os
import sys
from pathlib import Path

# Try to import image processing libraries
try:
    import cv2
    import numpy as np
    USE_CV2 = True
except ImportError:
    USE_CV2 = False
    try:
        from PIL import Image
        import numpy as np
        USE_PIL = True
    except ImportError:
        USE_PIL = False
        print("Warning: Neither cv2 nor PIL available. Trying basic approach...")

def convert_mask_to_binary_cv2(mask_path, output_path):
    """Convert mask to binary using cv2."""
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return False
    # Convert to binary: any non-zero becomes 255
    mask_binary = (mask > 0).astype(np.uint8) * 255
    cv2.imwrite(str(output_path), mask_binary)
    return True

def convert_mask_to_binary_pil(mask_path, output_path):
    """Convert mask to binary using PIL."""
    try:
        img = Image.open(mask_path).convert('L')  # Convert to grayscale
        # Convert to numpy array
        arr = np.array(img)
        # Binary threshold: any non-zero becomes 255
        arr_binary = (arr > 0).astype(np.uint8) * 255
        # Convert back to PIL Image and save
        img_binary = Image.fromarray(arr_binary)
        img_binary.save(output_path)
        return True
    except Exception as e:
        print(f"Error processing {mask_path}: {e}")
        return False

def convert_mask_to_binary(mask_path, output_path):
    """Convert mask to binary using available library."""
    if USE_CV2:
        return convert_mask_to_binary_cv2(mask_path, output_path)
    elif USE_PIL:
        return convert_mask_to_binary_pil(mask_path, output_path)
    else:
        print(f"Error: No image processing library available for {mask_path}")
        return False

def process_dataset(dataset_name):
    """Process all masks in a dataset."""
    base = Path("Dataset") / dataset_name
    mask_dir = base / "masks"
    
    if not mask_dir.exists():
        print(f"  {dataset_name}: masks directory not found")
        return 0
    
    mask_files = sorted(mask_dir.glob("*.png"))
    if not mask_files:
        print(f"  {dataset_name}: No PNG masks found")
        return 0
    
    converted = 0
    for mask_file in mask_files:
        # Create temporary name, then replace
        temp_path = mask_file.with_suffix('.tmp.png')
        if convert_mask_to_binary(mask_file, temp_path):
            # Replace original with binary version
            temp_path.replace(mask_file)
            converted += 1
    
    print(f"  {dataset_name}: Converted {converted}/{len(mask_files)} masks to binary")
    return converted

if __name__ == "__main__":
    print("Converting masks to binary (0/255)...")
    print("=" * 50)
    
    if not USE_CV2 and not USE_PIL:
        print("ERROR: No image processing library available!")
        print("Please install one of:")
        print("  - opencv-python: pip install opencv-python")
        print("  - Pillow: pip install Pillow")
        sys.exit(1)
    
    total = 0
    total += process_dataset("drive")
    total += process_dataset("drive_test")
    total += process_dataset("isbi12")
    total += process_dataset("crack")
    
    print("=" * 50)
    print(f"Total: Converted {total} masks to binary (0/255)")
