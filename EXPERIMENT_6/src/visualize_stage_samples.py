#!/usr/bin/env python3
"""
Visualize sample curves for each training stage.
Ensures parameters (ranges, intensities, noise) are resolved exactly like the trainer.
"""
import argparse
import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Ensure local imports work
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

# Try importing from local directory first, then src if needed
try:
    from train_deeper_model import CurveMakerFlexible, load_curve_config
except ImportError:
    try:
        from src.train_deeper_model import CurveMakerFlexible, load_curve_config
    except ImportError:
        print("❌ Error: Could not import 'train_deeper_model'. Make sure it is in the same directory or src/.")
        sys.exit(1)

def resolve_param(config, key_base, default_val):
    """
    Resolve a parameter from the config using the same priority as the trainer:
    1. *_range (sample uniform)
    2. Exact key
    3. *_prob key
    4. Default
    """
    range_key = f"{key_base}_range"
    if range_key in config:
        val_range = config[range_key]
        return np.random.uniform(val_range[0], val_range[1])
    elif key_base in config:
        return config[key_base]
    elif f"{key_base}_prob" in config:
        return config[f"{key_base}_prob"]
    return default_val

def generate_stage_sample(maker, stage_config):
    """
    Generate a single sample using the exact logic from CurveEnvUnified.reset()
    """
    # 1. Resolve Parameters
    w_range = tuple(stage_config.get('width_range', [2, 4]))
    
    # Curvature
    if 'curvature_range' in stage_config:
        cr = stage_config['curvature_range']
        curvature = np.random.uniform(cr[0], cr[1])
    else:
        curvature = stage_config.get('curvature_factor', 1.0)

    # Noise
    if 'noise_range' in stage_config:
        nr = stage_config['noise_range']
        noise_prob = np.random.uniform(nr[0], nr[1])
    else:
        noise_prob = stage_config.get('noise_prob', 0.0)

    # Background
    if 'background_intensity_range' in stage_config:
        br = stage_config['background_intensity_range']
        bg_int = np.random.uniform(br[0], br[1])
    else:
        bg_int = stage_config.get('background_intensity', 0.0)

    # Intensity Logic (Exact match to trainer)
    min_int = resolve_param(stage_config, 'min_intensity', 0.1)
    max_int = resolve_param(stage_config, 'max_intensity', 1.0)
    
    # Enforce Visibility Constraints (Reduced margin)
    if min_int < bg_int + 0.02: min_int = bg_int + 0.02
    if max_int < min_int: max_int = min_int + 0.05
    if max_int > 1.0: max_int = 1.0
    if min_int > 1.0: min_int = 1.0

    # 2. Generate Curve
    img, mask, _ = maker.sample_curve(
        width_range=w_range,
        noise_prob=noise_prob,
        invert_prob=stage_config.get('invert_prob', 0.0),
        min_intensity=min_int,
        max_intensity=max_int,
        background_intensity=bg_int,
        branches=stage_config.get('branches', False),
        curvature_factor=curvature,
        allow_self_cross=stage_config.get('allow_self_cross', False),
        self_cross_prob=stage_config.get('self_cross_prob', 0.0),
        width_variation=stage_config.get('width_variation', 'none'),
        start_width=stage_config.get('start_width', None),
        end_width=stage_config.get('end_width', None),
        intensity_variation=stage_config.get('intensity_variation', 'none'),
        start_intensity=stage_config.get('start_intensity', None),
        end_intensity=stage_config.get('end_intensity', None)
    )
    
    # 3. Apply Tissue Noise (if enabled)
    if stage_config.get('tissue', False):
        sigma_range = [2.0, 5.0] 
        intensity_range = [0.2, 0.4] 
        
        from scipy.ndimage import gaussian_filter
        noise = np.random.randn(*img.shape)
        tissue = gaussian_filter(noise, sigma=np.random.uniform(*sigma_range))
        tissue = (tissue - tissue.min()) / (tissue.max() - tissue.min())
        tissue = tissue * np.random.uniform(*intensity_range)
        
        is_white_bg = np.mean([img[0,0], img[0,-1]]) > 0.5
        if is_white_bg:
            img = np.clip(img - tissue, 0.0, 1.0)
        else:
            img = np.clip(img + tissue, 0.0, 1.0)

    # Store params for display
    params_used = {
        "curv": f"{curvature:.2f}",
        "int": f"{min_int:.2f}-{max_int:.2f}",
        "bg": f"{bg_int:.2f}",
        "noise": f"{noise_prob:.2f}",
        "inv": stage_config.get('invert_prob', 0.0)
    }
    return img, params_used

def visualize_stage_samples(config_path, samples_per_stage=6, seed=42, save_dir=None):

    # Load config
    curve_config, config_file_path = load_curve_config(config_path)
    
    # Get image dimensions
    h = curve_config.get('image', {}).get('height', 128)
    w = curve_config.get('image', {}).get('width', 128)
    
    stages = curve_config.get('training_stages', [])
    if not stages:
        print("❌ No training stages found in config.")
        return

    print(f"✅ Loaded Config: {config_file_path}")
    print(f"✅ Found {len(stages)} stages. Generating {samples_per_stage} samples each.\n")

    for i, stage in enumerate(stages):
        stage_name = stage.get('name', f"Stage {i+1}")
        print(f"🎨 Visualizing: {stage_name}")
        
        # Merge config just like the trainer
        merged_config = stage.get('curve_generation', {}).copy()
        merged_config.update(stage.get('training', {}))
        
        # Setup Plot
        cols = 3
        rows = (samples_per_stage + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
        axes = axes.flatten()
        
        fig.suptitle(f"{stage_name}\nConfig: {json.dumps(merged_config, indent=1)}", fontsize=8, y=0.99)
        
        # Generate Samples
        for j in range(samples_per_stage):
            # Unique seed per sample
            sample_seed = seed + (i * 1000) + j
            maker = CurveMakerFlexible(h=h, w=w, seed=sample_seed, config=curve_config)
            
            img, p = generate_stage_sample(maker, merged_config)
            
            ax = axes[j]
            ax.imshow(img, cmap='gray', vmin=0, vmax=1)
            ax.set_title(f"C:{p['curv']} I:{p['int']}\nBG:{p['bg']} N:{p['noise']}", fontsize=9)
            ax.axis('off')

        # Hide unused subplots
        for j in range(samples_per_stage, len(axes)):
            axes[j].axis('off')
            
        plt.tight_layout(rect=[0, 0, 1, 0.90]) # Make room for suptitle
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            fname = f"{stage_name.replace(' ', '_')}.png"
            path = os.path.join(save_dir, fname)
            plt.savefig(path, dpi=150)
            print(f"   💾 Saved to {path}")
            plt.close()
        else:
            plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="config/curve_config.json", help="Path to curve_config.json")
    parser.add_argument('--samples', type=int, default=6)
    parser.add_argument('--seed', type=int, default=87)
    parser.add_argument('--save_dir', type=str, default=None, help="Save images to directory instead of showing")
    
    args = parser.parse_args()
    
    visualize_stage_samples(args.config, args.samples, args.seed, args.save_dir)