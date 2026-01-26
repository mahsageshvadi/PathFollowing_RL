import os
import sys
import cv2
import numpy as np
import random
from tqdm import tqdm

# Boilerplate to load your local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from train_deeper_model import CurveMakerFlexible, load_curve_config, crop32
except ImportError:
    from src.train_deeper_model import CurveMakerFlexible, load_curve_config, crop32

def get_extreme_params():
    """
    Generates parameters to cover ANY possible vessel appearance.
    """
    # 1. ANY WIDTH: From 1px (hairline) to 8px (thick artery)
    w_min = random.randint(1, 4)
    w_max = random.randint(w_min, 8)
    
    # 2. ANY CONTRAST:
    bg = random.uniform(0.0, 0.4)
    v_min = bg + random.uniform(0.05, 0.4) 
    v_max = random.uniform(v_min, 1.0)
    
    # 3. ANY NOISE:
    noise_roll = random.random()
    if noise_roll < 0.2:
        noise_p = 0.0
    elif noise_roll < 0.6:
        noise_p = random.uniform(0.1, 0.4)
    else:
        noise_p = random.uniform(0.4, 1.0) # Extreme noise

    # Return parameters compatible with sample_curve, plus 'noise_level_range' to be handled separately
    return {
        "width_range": [w_min, w_max],
        "curvature_factor": random.uniform(0.0, 1.8),
        "min_intensity": min(1.0, v_min),
        "max_intensity": min(1.0, v_max),
        "background_intensity": bg,
        "noise_prob": noise_p,
        "noise_level_range": [0.05, 0.3] if noise_p > 0 else [0.0, 0.0]
    }

def generate_strict_data(out_dir="data_strict", n_pairs=5000):
    # Load config context
    if os.path.exists("config/curve_config.json"):
        config, _ = load_curve_config("config/curve_config.json")
    elif os.path.exists("curve_config.json"):
        config, _ = load_curve_config("curve_config.json")
    else:
        config = {'image': {'height': 128, 'width': 128}, 'bezier': {'margin': 10}}

    h = config.get('image', {}).get('height', 128)
    w = config.get('image', {}).get('width', 128)
    
    # Setup Dirs
    os.makedirs(os.path.join(out_dir, "0_normal"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "1_junction"), exist_ok=True)
    
    maker = CurveMakerFlexible(h=h, w=w, seed=999, config=config)
    
    # Fix branching range
    maker.branch_num_range = (1, 2)
    
    print(f"Generating {n_pairs*2} Strictly Centered Samples...")
    
    for i in tqdm(range(n_pairs)):
        
        # ==========================================
        # CLASS 1: THE CENTERED JUNCTION
        # ==========================================
        params = get_extreme_params()
        
        # EXTRACT and REMOVE 'noise_level_range' so it doesn't crash sample_curve
        noise_lvl = params.pop('noise_level_range')
        
        # Apply the noise level to the maker object directly
        maker.noise_level_range = tuple(noise_lvl)
        
        # Generate Curve
        img_j, _, pts_j = maker.sample_curve(
            **params,
            branches=True # Force a split
        )
        
        if len(pts_j) > 1:
            # The junction is strictly at the start of the branch line
            junction_xy = pts_j[1][0]
            crop_j = crop32(img_j, int(junction_xy[0]), int(junction_xy[1]))
            cv2.imwrite(f"{out_dir}/1_junction/j_{i}.png", (crop_j * 255).astype(np.uint8))
        
        # ==========================================
        # CLASS 0: THE PATH (Negative Sample)
        # ==========================================
        params_n = get_extreme_params()
        
        # Handle noise level manually again
        noise_lvl_n = params_n.pop('noise_level_range')
        maker.noise_level_range = tuple(noise_lvl_n)
        
        img_n, _, pts_n = maker.sample_curve(
            **params_n,
            branches=False # Force simple path
        )
        
        main_curve = pts_n[0]
        if len(main_curve) > 20:
            # Pick a random point away from the edges
            idx = random.randint(10, len(main_curve)-10)
            center_xy = main_curve[idx]
            
            crop_n = crop32(img_n, int(center_xy[0]), int(center_xy[1]))
            cv2.imwrite(f"{out_dir}/0_normal/n_{i}.png", (crop_n * 255).astype(np.uint8))

    print(f"✅ Done. Data in '{out_dir}/'.")

if __name__ == "__main__":
    generate_strict_data()