#!/usr/bin/env python3
"""
Model Generalization Tester.
Tests a specific model checkpoint against a specific Stage configuration.
Saves metrics (Success/Failure rates) to JSON and failure images to disk.
"""
import argparse
import numpy as np
import torch
import os
import sys
import json
import cv2
from tqdm import tqdm

# Add parent directory
_script_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_script_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

# Import shared modules
try:
    from src.models_deeper import ActorOnly
    from src.train_deeper_model import CurveEnvUnified, load_curve_config, fixed_window_history
except ImportError:
    try:
        from train_deeper_model import CurveEnvUnified, load_curve_config, fixed_window_history
        from models_deeper import ActorOnly
    except ImportError:
        print("❌ Error: Could not import modules. Ensure 'src' is in python path.")
        sys.exit(1)

# Constants
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_ACTIONS = 8 

K = 16 # History window size

# --- JSON Encoder for NumPy Types ---
class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def load_model(path, n_actions=N_ACTIONS):
    print(f"🔄 Loading weights from: {path}")
    model = ActorOnly(n_actions=n_actions, K=K).to(DEVICE)
    
    try:
        checkpoint = torch.load(path, map_location=DEVICE)
        
        # Clean weights: handle 'actor_' prefix and 'critic' keys
        clean_state = {}
        for k, v in checkpoint.items():
            if k.startswith('actor_'):
                clean_state[k] = v
            elif not k.startswith('critic'):
                clean_state[f'actor_{k}'] = v
        
        model.load_state_dict(clean_state, strict=False)
        model.eval()
        print("✅ Model loaded successfully.")
        return model
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        sys.exit(1)

def run_evaluation(model, config, target_stage_id, episodes=100, save_failures=False, out_dir="test_results", model_path="unknown"):
    # 1. Setup Environment
    h = config.get('image', {}).get('height', 128)
    w = config.get('image', {}).get('width', 128)
    
    stages = config.get('training_stages', [])
    target_stage = next((s for s in stages if s['stage_id'] == target_stage_id), None)
    
    if not target_stage:
        print(f"❌ Stage ID {target_stage_id} not found in config.")
        return

    print(f"\n🎯 Testing against: {target_stage['name']} (ID: {target_stage_id})")
    
    # Merge configs
    env_config = target_stage.get('curve_generation', {}).copy()
    env_config.update(target_stage.get('training', {}))
    
    print("   Parameters:")
    print(json.dumps(env_config, indent=4))

    env = CurveEnvUnified(h=h, w=w, max_steps=400, base_seed=42, stage_id=target_stage_id, curve_config=config)
    env.set_stage(env_config)

    os.makedirs(out_dir, exist_ok=True)

    # 2. Metrics Setup
    success_count = 0
    total_reward = 0.0
    strict_stop = env_config.get('strict_stop', False)
    
    fail_dir = os.path.join(out_dir, f"failures_stage{target_stage_id}")
    if save_failures:
        os.makedirs(fail_dir, exist_ok=True)
        print(f"   📂 Saving failures to: {fail_dir}")

    # 3. Evaluation Loop
    pbar = tqdm(range(episodes), desc="Testing")
    for i in pbar:
        obs_dict = env.reset(episode_number=100000 + i)
        
        done = False
        ep_rew = 0
        ahist = [np.zeros(N_ACTIONS)] * K
        
        while not done:
            obs_t = torch.tensor(obs_dict['actor'][None], dtype=torch.float32, device=DEVICE)
            ahist_arr = fixed_window_history(ahist, K, N_ACTIONS)
            hist_t = torch.tensor(ahist_arr[None], dtype=torch.float32, device=DEVICE)
            
            with torch.no_grad():
                logits, _ = model(obs_t, hist_t)
                action = torch.argmax(logits, dim=1).item()
            
            obs_dict, r, done, info = env.step(action)
            ep_rew += r
            
            oh = np.zeros(N_ACTIONS)
            oh[action] = 1.0
            ahist.append(oh)
        
        total_reward += ep_rew
        
        # Check Success
        is_success = False
        if strict_stop:
            is_success = info.get('stopped_correctly', False)
        else:
            is_success = info.get('reached_end', False)
            
        if is_success:
            success_count += 1
        elif save_failures:
            # Visualization
            img_vis = (env.ep.img * 255).astype(np.uint8)
            img_vis = cv2.cvtColor(img_vis, cv2.COLOR_GRAY2BGR)
            
            path = np.array(env.path_points).astype(np.int32)
            cv2.polylines(img_vis, [path[:, ::-1]], isClosed=False, color=(0, 255, 255), thickness=1)
            
            if len(path) > 0:
                cv2.circle(img_vis, (path[0][1], path[0][0]), 2, (0, 255, 0), -1)
                cv2.circle(img_vis, (path[-1][1], path[-1][0]), 2, (0, 0, 255), -1)

            gt = env.ep.gt_poly.astype(np.int32)
            cv2.polylines(img_vis, [gt[:, ::-1]], isClosed=False, color=(0, 255, 0), thickness=1)
            
            filename = f"fail_ep{i}_R{int(ep_rew)}.png"
            cv2.imwrite(os.path.join(fail_dir, filename), img_vis)

        pbar.set_postfix({'SR': f"{success_count/(i+1):.2%}", 'AvgR': f"{total_reward/(i+1):.1f}"})

    # 4. Final Stats
    success_rate = (success_count / episodes) * 100
    failure_count = episodes - success_count
    failure_rate = (failure_count / episodes) * 100
    avg_reward = total_reward / episodes
    
    # 5. Save Results (Using NumpyEncoder to fix float32 error)
    results_data = {
        "target_stage_id": target_stage_id,
        "target_stage_name": target_stage['name'],
        "model_path": model_path,
        "total_episodes": episodes,
        "success_count": success_count,
        "success_rate_percent": float(round(success_rate, 2)),
        "failure_count": failure_count,
        "failure_rate_percent": float(round(failure_rate, 2)),
        "average_reward": float(round(avg_reward, 4)),
        "config_used": env_config
    }
    
    json_path = os.path.join(out_dir, f"results_stage{target_stage_id}.json")
    with open(json_path, 'w') as f:
        json.dump(results_data, f, indent=4, cls=NumpyEncoder)

    # 6. Output
    print("\n" + "="*40)
    print(f"📊 RESULTS: {target_stage['name']}")
    print("="*40)
    print(f"   Success Rate:   {success_rate:.2f}%")
    print(f"   Failure Rate:   {failure_rate:.2f}%")
    print(f"   Avg Reward:     {avg_reward:.2f}")
    print(f"   💾 Detailed results saved to: {json_path}")
    print("="*40 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test model generalization across stages")
    parser.add_argument("--weights", type=str, required=True, help="Path to model weights (.pth)")
    parser.add_argument("--config", type=str, default="config/curve_config.json", help="Path to config file")
    parser.add_argument("--test_stage", type=int, required=True, help="Stage ID to test AGAINST (e.g., 2)")
    parser.add_argument("--episodes", type=int, default=100, help="Number of test episodes")
    parser.add_argument("--save_failures", action="store_true", help="Save images of failed episodes")
    parser.add_argument("--out_dir", type=str, default="test_output", help="Directory to save failures")
    
    args = parser.parse_args()
    
    cfg, _ = load_curve_config(args.config)
    model = load_model(args.weights)
    
    run_evaluation(
        model=model,
        config=cfg,
        target_stage_id=args.test_stage,
        episodes=args.episodes,
        save_failures=args.save_failures,
        out_dir=args.out_dir,
        model_path=args.weights
    )