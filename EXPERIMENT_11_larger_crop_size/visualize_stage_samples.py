#!/usr/bin/env python3
"""
Visualize sample curves for each training stage in Experiment 9.
Shows what the model sees during training for each stage.
Updated to handle Junctions and Structured Noise.
"""
import argparse
import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directory to path
_script_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_script_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

# Import from train_deeper_model.py where CurveMakerFlexible is defined
from src.train_deeper_model import CurveMakerFlexible, load_curve_config

def format_params(cfg):
    """Format curve generation parameters for display."""
    params = []
    params.append(f"Width: {cfg.get('width_range', [])}")
    params.append(f"Int: {cfg.get('min_intensity', 0):.2f}-{cfg.get('max_intensity', 1.0):.2f}")
    
    if cfg.get('num_distractors_range'):
        params.append(f"Distractors: {cfg.get('num_distractors_range')}")
    
    if cfg.get('roughness'):
        params.append(f"Rough: {cfg.get('roughness'):.2f}")

    if cfg.get('background_intensity') is not None:
        params.append(f"BG: {cfg.get('background_intensity'):.2f}")
    
    return ", ".join(params)

def visualize_stage_samples(config_path, samples_per_stage=6, seed=4269, save_dir=None):
    """
    Generate and visualize sample curves for each training stage.
    """
    # Load config
    curve_config, config_file_path = load_curve_config(config_path)
    if curve_config is None:
        print(f"Error: Could not load config from {config_path}")
        return
    
    stages = curve_config.get('training_stages', [])
    if not stages:
        print("Error: No training stages found in config")
        return
    
    print(f"✅ Loaded config: {config_file_path}")
    print(f"✅ Found {len(stages)} training stages")
    print(f"✅ Generating {samples_per_stage} samples per stage\n")
    
    h = curve_config.get('image', {}).get('height', 128)
    w = curve_config.get('image', {}).get('width', 128)
    
    for stage_idx, stage in enumerate(stages):
        stage_id = stage.get('stage_id', stage_idx + 1)
        stage_name = stage.get('name', f'Stage{stage_id}')
        curve_cfg = stage.get('curve_generation', {})
        
        print(f"📊 Stage {stage_id}: {stage_name}")
        print(f"   {format_params(curve_cfg)}")
        
        # Create figure
        n_cols = 3
        n_rows = (samples_per_stage + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()
        
        for sample_idx in range(samples_per_stage):
            sample_seed = seed + stage_id * 1000 + sample_idx
            curve_maker = CurveMakerFlexible(h=h, w=w, seed=sample_seed, config=curve_config)
            
            # Helper to resolve ranges
            def resolve(key_base, default_val):
                range_key = f"{key_base}_range"
                if range_key in curve_cfg:
                    val_range = curve_cfg[range_key]
                    return np.random.uniform(val_range[0], val_range[1])
                elif key_base in curve_cfg:
                    return curve_cfg[key_base]
                return default_val

            # Extract basic params
            width_range = tuple(curve_cfg.get('width_range', [3, 5]))
            noise_prob = resolve('noise', 0.0) # handles noise_prob or noise_range
            invert_prob = curve_cfg.get('invert_prob', 0.0)
            
            # Resolve Intensity
            min_int = resolve('min_intensity', 0.6)
            max_int = resolve('max_intensity', 1.0)
            bg_int = resolve('background_intensity', 0.0)
            
            # Resolve Curvature
            curvature = resolve('curvature', 1.0)

            # Resolve New Params (Distractors & Roughness)
            n_dist = resolve('num_distractors', 0)
            rough_p = resolve('roughness', 0.0)

            # Generate curve (FIXED UNPACKING HERE)
            img, mask, pts_all, junction_pts = curve_maker.sample_curve(
                width_range=width_range,
                noise_prob=noise_prob,
                invert_prob=invert_prob,
                min_intensity=min_int,
                max_intensity=max_int,
                background_intensity=bg_int,
                branches=curve_cfg.get('branches', False),
                curvature_factor=curvature,
                width_variation=curve_cfg.get('width_variation', 'none'),
                start_width=curve_cfg.get('start_width', None),
                end_width=curve_cfg.get('end_width', None),
                intensity_variation=curve_cfg.get('intensity_variation', 'none'),
                start_intensity=curve_cfg.get('start_intensity', None),
                end_intensity=curve_cfg.get('end_intensity', None),
                allow_self_cross=curve_cfg.get('allow_self_cross', False),
                self_cross_prob=curve_cfg.get('self_cross_prob', 0.0),
                # New Arguments
                num_distractors=int(n_dist),
                roughness_prob=rough_p
            )
            
            # Mark Junctions on Visualization
            display_img = np.dstack([img, img, img])
            if junction_pts:
                for jp in junction_pts:
                    # Draw red circle at junctions
                    cv2.circle(display_img, (int(jp[1]), int(jp[0])), 3, (1.0, 0, 0), -1)

            ax = axes[sample_idx]
            ax.imshow(display_img, vmin=0, vmax=1)
            ax.set_title(f'Sample {sample_idx + 1}', fontsize=10)
            ax.axis('off')
        
        for idx in range(samples_per_stage, len(axes)):
            axes[idx].axis('off')
        
        fig.suptitle(
            f'Stage {stage_id}: {stage_name}\n{format_params(curve_cfg)}',
            fontsize=14,
            y=0.98
        )
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f'stage_{stage_id:02d}_{stage_name}.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"   💾 Saved: {save_path}")
        else:
            plt.show()
        
        plt.close()
        print()
    
    print("✅ Visualization complete!")

import cv2 # Ensure cv2 is imported for drawing junction circles

def main():
    parser = argparse.ArgumentParser(description="Visualize sample curves for each training stage")
    parser.add_argument('--config', type=str, default='config/curve_config.json')
    parser.add_argument('--samples', type=int, default=6)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_dir', type=str, default=None)
    args = parser.parse_args()
    
    visualize_stage_samples(args.config, args.samples, args.seed, args.save_dir)

if __name__ == '__main__':
    main()