#!/usr/bin/env python3
"""
Visualize sample curves for each training stage in Experiment3_Refine.
Shows what the model sees during training for each stage.
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
    params.append(f"Intensity: {cfg.get('min_intensity', 0):.2f}-{cfg.get('max_intensity', 1.0):.2f}")
    params.append(f"Curvature: {cfg.get('curvature_factor', 1.0):.2f}")
    params.append(f"Noise: {cfg.get('noise_prob', 0.0):.2f}")
    
    if cfg.get('background_intensity') is not None:
        params.append(f"BG: {cfg.get('background_intensity'):.2f}")
    
    if cfg.get('allow_self_cross', False):
        params.append(f"Self-cross: {cfg.get('self_cross_prob', 0.0):.2f}")
    
    width_var = cfg.get('width_variation', 'none')
    if width_var != 'none':
        params.append(f"Width var: {width_var}")
    
    intensity_var = cfg.get('intensity_variation', 'none')
    if intensity_var != 'none':
        params.append(f"Intensity var: {intensity_var}")
    
    return ", ".join(params)

def visualize_stage_samples(config_path, samples_per_stage=6, seed=42, save_dir=None):
    """
    Generate and visualize sample curves for each training stage.
    
    Args:
        config_path: Path to curve_config.json
        samples_per_stage: Number of sample images to generate per stage
        seed: Random seed for reproducibility
        save_dir: Directory to save images (if None, just displays)
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
    
    # Create curve generator
    h = curve_config.get('image', {}).get('height', 128)
    w = curve_config.get('image', {}).get('width', 128)
    
    # Visualize each stage
    for stage_idx, stage in enumerate(stages):
        stage_id = stage.get('stage_id', stage_idx + 1)
        stage_name = stage.get('name', f'Stage{stage_id}')
        curve_cfg = stage.get('curve_generation', {})
        
        print(f"📊 Stage {stage_id}: {stage_name}")
        print(f"   {format_params(curve_cfg)}")
        
        # Create figure for this stage
        n_cols = 3
        n_rows = (samples_per_stage + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()
        
        # Generate samples
        for sample_idx in range(samples_per_stage):
            # Use different seed for each sample
            sample_seed = seed + stage_id * 1000 + sample_idx
            curve_maker = CurveMakerFlexible(h=h, w=w, seed=sample_seed, config=curve_config)
            
            # Extract parameters from stage config
            width_range = tuple(curve_cfg.get('width_range', [3, 5]))
            noise_prob = curve_cfg.get('noise_prob', 0.0)
            invert_prob = curve_cfg.get('invert_prob', 0.0)
            min_intensity = curve_cfg.get('min_intensity', 0.6)
            max_intensity = curve_cfg.get('max_intensity', None)
            branches = curve_cfg.get('branches', False)
            curvature_factor = curve_cfg.get('curvature_factor', 1.0)
            width_variation = curve_cfg.get('width_variation', 'none')
            start_width = curve_cfg.get('start_width', None)
            end_width = curve_cfg.get('end_width', None)
            intensity_variation = curve_cfg.get('intensity_variation', 'none')
            start_intensity = curve_cfg.get('start_intensity', None)
            end_intensity = curve_cfg.get('end_intensity', None)
            background_intensity = curve_cfg.get('background_intensity', None)
            allow_self_cross = curve_cfg.get('allow_self_cross', False)
            self_cross_prob = curve_cfg.get('self_cross_prob', 0.0)
            
            # Generate curve
            img, mask, pts_all = curve_maker.sample_curve(
                width_range=width_range,
                noise_prob=noise_prob,
                invert_prob=invert_prob,
                min_intensity=min_intensity,
                max_intensity=max_intensity,
                branches=branches,
                curvature_factor=curvature_factor,
                width_variation=width_variation,
                start_width=start_width,
                end_width=end_width,
                intensity_variation=intensity_variation,
                start_intensity=start_intensity,
                end_intensity=end_intensity,
                background_intensity=background_intensity,
                allow_self_cross=allow_self_cross,
                self_cross_prob=self_cross_prob
            )
            
            # Display
            ax = axes[sample_idx]
            ax.imshow(img, cmap='gray', vmin=0, vmax=1)
            ax.set_title(f'Sample {sample_idx + 1}', fontsize=10)
            ax.axis('off')
        
        # Hide unused subplots
        for idx in range(samples_per_stage, len(axes)):
            axes[idx].axis('off')
        
        # Add main title
        fig.suptitle(
            f'Stage {stage_id}: {stage_name}\n{format_params(curve_cfg)}',
            fontsize=14,
            y=0.98
        )
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Save or show
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

def main():
    parser = argparse.ArgumentParser(
        description="Visualize sample curves for each training stage"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='../config/curve_config.json',
        help='Path to curve_config.json (default: ../config/curve_config.json)'
    )
    parser.add_argument(
        '--samples',
        type=int,
        default=6,
        help='Number of samples per stage (default: 6)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    parser.add_argument(
        '--save_dir',
        type=str,
        default=None,
        help='Directory to save images (default: None, displays instead)'
    )
    args = parser.parse_args()
    
    visualize_stage_samples(
        config_path=args.config,
        samples_per_stage=args.samples,
        seed=args.seed,
        save_dir=args.save_dir
    )

if __name__ == '__main__':
    main()

