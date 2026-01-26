#!/usr/bin/env python3
"""
Generate datasets sorted into bins based on curve complexity.
"""
import os
import sys
import cv2
import numpy as np
import argparse
import json
from tqdm import tqdm

# Import existing classes
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

from train_deeper_model import CurveMakerFlexible, load_curve_config

def generate_dataset_bins(output_dir="data_bins", n_images=100, config_path="../config/curve_bins.json"):
    # Load the bins config
    config, _ = load_curve_config(config_path)
    stages = config.get('training_stages', [])
    
    h = config.get('image', {}).get('height', 128)
    w = config.get('image', {}).get('width', 128)

    print(f"Generating {n_images} images per bin into '{output_dir}'...")

    for stage in stages:
        bin_name = stage['name']
        bin_id = stage['stage_id']
        params = stage['config'] # Note: train_deeper_model puts this in 'config' key inside the stage dict in code
        
        # Handle JSON structure vs Internal Python structure difference
        # The JSON has 'curve_generation', the Env code flattens it. 
        # Here we manually extract what we need from JSON.
        gen_cfg = stage.get('curve_generation', {})
        
        save_path = os.path.join(output_dir, bin_name)
        os.makedirs(save_path, exist_ok=True)
        
        print(f"  > Generating Bin {bin_id}: {bin_name}")
        
        curve_maker = CurveMakerFlexible(h=h, w=w, seed=bin_id*999, config=config)
        
        for i in tqdm(range(n_images)):
            # Determine specific parameters for this image
            # If ranges are provided, sample from them
            
            # Curvature
            c_range = gen_cfg.get('curvature_range', [1.0, 1.0])
            curv = np.random.uniform(c_range[0], c_range[1])
            
            # Width
            w_range = gen_cfg.get('width_range', [3, 3])
            
            # Self Cross
            allow_cross = gen_cfg.get('allow_self_cross', False)
            cross_prob = gen_cfg.get('self_cross_prob', 0.0)

                        # --- NEW: EXTRACT INTENSITY SETTINGS ---
            min_int = gen_cfg.get('min_intensity', 0.6)
            max_int = gen_cfg.get('max_intensity', 1.0)
            bg_int = gen_cfg.get('background_intensity', 0.0)
            
            # Gradient settings
            int_var = gen_cfg.get('intensity_variation', 'none')
            start_i = gen_cfg.get('start_intensity', None)
            end_i = gen_cfg.get('end_intensity', None)

            img, mask, _ = curve_maker.sample_curve(
                width_range=tuple(gen_cfg.get('width_range', [3, 3])),
                curvature_factor=np.random.uniform(*gen_cfg.get('curvature_range', [1.0, 1.0])),
                allow_self_cross=gen_cfg.get('allow_self_cross', False),
                self_cross_prob=gen_cfg.get('self_cross_prob', 0.0),
                branches=False,
                noise_prob=gen_cfg.get('noise_prob', 0.0),
                
                # PASS INTENSITY ARGUMENTS HERE
                min_intensity=min_int,
                max_intensity=max_int,
                background_intensity=bg_int,
                intensity_variation=int_var,
                start_intensity=start_i,
                end_intensity=end_i
            )
            # Save
            filename = f"{bin_name}_{i:05d}.png"
            cv2.imwrite(os.path.join(save_path, filename), (img * 255).astype(np.uint8))
            
            # Optional: Save mask
            # cv2.imwrite(os.path.join(save_path, f"mask_{filename}"), (mask * 255).astype(np.uint8))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="data_generated")
    parser.add_argument("--num", type=int, default=50)
    parser.add_argument("--config", type=str, default="../config/curve_bins.json")
    args = parser.parse_args()
    
    generate_dataset_bins(args.out, args.num, args.config)