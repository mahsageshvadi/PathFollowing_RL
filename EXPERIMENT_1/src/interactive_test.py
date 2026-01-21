#!/usr/bin/env python3
"""
Interactive testing script for DSA RL model.
Generates synthetic curves and visualizes tracking step-by-step
using the updated deep + long-memory actor.
"""
import argparse
import numpy as np
import torch
import cv2
import os
import sys
import matplotlib.pyplot as plt

# Add parent directory to path
_script_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_script_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from src.models_deeper import ActorOnly
from src.curve_generator import CurveMakerFlexible, load_curve_config

# ---------- CONSTANTS ----------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ACTIONS_MOVEMENT = [
    (-1, 0), (1, 0), (0, -1), (0, 1),
    (-1, -1), (-1, 1), (1, -1), (1, 1)
]
ACTION_STOP_IDX = 8
N_ACTIONS = 9

K = 16                      # ✅ must match training
STEP_ALPHA = 1.0
CROP = 33

MIN_STEPS_BEFORE_STOP = 10
TEMPERATURE = 0.5

ACTION_NAMES = [
    "UP", "DOWN", "LEFT", "RIGHT",
    "UP-LEFT", "UP-RIGHT", "DOWN-LEFT", "DOWN-RIGHT", "STOP"
]

# ---------- HELPERS ----------
def clamp(v, lo, hi):
    return max(lo, min(v, hi))

def crop32(img, cy, cx):
    h, w = img.shape
    y0, y1 = cy - CROP // 2, cy + CROP // 2 + 1
    x0, x1 = cx - CROP // 2, cx + CROP // 2 + 1

    out = np.zeros((CROP, CROP), dtype=img.dtype)
    sy0, sy1 = clamp(y0, 0, h), clamp(y1, 0, h)
    sx0, sx1 = clamp(x0, 0, w), clamp(x1, 0, w)

    oy0, ox0 = sy0 - y0, sx0 - x0
    sh, sw = sy1 - sy0, sx1 - sx0

    if sh > 0 and sw > 0:
        out[oy0:oy0+sh, ox0:ox0+sw] = img[sy0:sy1, sx0:sx1]

    return out

def fixed_window_history(ahist, K, n_actions):
    out = np.zeros((K, n_actions), dtype=np.float32)
    tail = ahist[-K:]
    if len(tail) > 0:
        out[-len(tail):] = np.stack(tail, axis=0)
    return out

# ---------- INTERACTIVE ENV ----------
class InteractiveEnv:
    def __init__(self, img, gt_poly, start_idx=5):
        self.img = img
        self.h, self.w = img.shape
        self.gt_poly = gt_poly

        start_idx = min(start_idx, len(gt_poly) - 1)
        self.agent = tuple(map(float, gt_poly[start_idx]))

        if start_idx >= 2:
            p2, p1 = gt_poly[start_idx-2], gt_poly[start_idx-1]
        else:
            p2 = p1 = gt_poly[0]

        self.history_pos = [tuple(p2), tuple(p1), self.agent]
        self.path_points = [self.agent]
        self.path_mask = np.zeros_like(img, dtype=np.float32)
        self.path_mask[int(self.agent[0]), int(self.agent[1])] = 1.0

        self.steps = 0
        self.max_steps = 1000

    def obs(self):
        curr, p1, p2 = self.history_pos[-1], self.history_pos[-2], self.history_pos[-3]
        ch0 = crop32(self.img, int(curr[0]), int(curr[1]))
        ch1 = crop32(self.img, int(p1[0]), int(p1[1]))
        ch2 = crop32(self.img, int(p2[0]), int(p2[1]))
        ch3 = crop32(self.path_mask, int(curr[0]), int(curr[1]))
        return np.stack([ch0, ch1, ch2, ch3], axis=0).astype(np.float32)

    def step(self, action):
        if action == ACTION_STOP_IDX:
            return True, "Stopped"

        dy, dx = ACTIONS_MOVEMENT[action]
        ny = clamp(self.agent[0] + dy * STEP_ALPHA, 0, self.h-1)
        nx = clamp(self.agent[1] + dx * STEP_ALPHA, 0, self.w-1)

        self.agent = (ny, nx)
        self.history_pos.append(self.agent)
        self.path_points.append(self.agent)
        self.path_mask[int(ny), int(nx)] = 1.0
        self.steps += 1

        dist = np.linalg.norm(np.array(self.agent) - self.gt_poly[-1])
        if dist < 5.0:
            return True, "Reached End"
        if self.steps >= self.max_steps:
            return True, "Max Steps"

        return False, ""

# ---------- MAIN ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--actor_weights", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--seed", type=int, default=65)
    parser.add_argument("--start_idx", type=int, default=5)
    args = parser.parse_args()

    print("Loading model...")
    model = ActorOnly(n_actions=N_ACTIONS, K=K).to(DEVICE)
    
    try:
        loaded_weights = torch.load(args.actor_weights, map_location=DEVICE)
        
        # Filter out critic weights if present (in case full model weights are loaded)
        actor_weights = {k: v for k, v in loaded_weights.items() if k.startswith('actor_')}
        
        if len(actor_weights) == 0:
            # No actor_ prefix - might be actor-only weights without prefix
            actor_weights = loaded_weights
        
        # Check for action count mismatch (8 vs 9 actions)
        # Old models had 8 actions (no STOP), new models have 9 actions (with STOP)
        checkpoint_actions = None
        for key in actor_weights.keys():
            if 'actor_head.2.weight' in key:
                checkpoint_actions = actor_weights[key].shape[0]
                break
            elif 'actor_head.2.bias' in key:
                checkpoint_actions = actor_weights[key].shape[0]
                break
        
        if checkpoint_actions is not None and checkpoint_actions == 8 and N_ACTIONS == 9:
            print(f"⚠️  Warning: Checkpoint has {checkpoint_actions} actions, model expects {N_ACTIONS}")
            print("   This checkpoint was trained without STOP action.")
            print("   Attempting to adapt weights...")
            
            # Adapt weights: copy last action weights to STOP action
            adapted_weights = {}
            for k, v in actor_weights.items():
                if 'actor_head.2.weight' in k:
                    # Expand from 8 to 9 actions by duplicating last action
                    new_weight = torch.zeros(9, v.shape[1], device=v.device, dtype=v.dtype)
                    new_weight[:8] = v
                    new_weight[8] = v[-1]  # Copy last action for STOP
                    adapted_weights[k] = new_weight
                elif 'actor_head.2.bias' in k:
                    # Expand bias from 8 to 9
                    new_bias = torch.zeros(9, device=v.device, dtype=v.dtype)
                    new_bias[:8] = v
                    new_bias[8] = v[-1]  # Copy last action bias for STOP
                    adapted_weights[k] = new_bias
                elif 'actor_lstm.weight_ih_l' in k:
                    # LSTM input-to-hidden weights: [4*hidden, input_size]
                    # Layer 0: input_size = n_actions (8 in checkpoint, 9 in model)
                    # Layer 1+: input_size = hidden_size from previous layer (should be 128)
                    layer_num = int(k.split('_l')[-1])  # Extract layer number (0, 1, etc.)
                    checkpoint_input_size = v.shape[1]
                    
                    if layer_num == 0:
                        # First layer: input is n_actions (one-hot action vector)
                        if checkpoint_input_size == 8:
                            # Expand from 8 to 9 actions
                            new_weight = torch.zeros(v.shape[0], 9, device=v.device, dtype=v.dtype)
                            new_weight[:, :8] = v
                            new_weight[:, 8] = v[:, -1]  # Copy last action for STOP
                            adapted_weights[k] = new_weight
                        else:
                            # Unexpected input size, keep as-is (will be skipped if mismatch)
                            adapted_weights[k] = v
                    else:
                        # Later layers: input should be hidden_size from previous layer
                        # For PyTorch LSTM, layer 1+ input is the hidden state (size = hidden_size)
                        model_expected_size = model.actor_lstm.hidden_size  # Should be 128
                        
                        if checkpoint_input_size == model_expected_size:
                            adapted_weights[k] = v
                        else:
                            # Size mismatch - architecture difference, skip this weight
                            print(f"   ⚠️  Skipping {k}: architecture mismatch (checkpoint: {checkpoint_input_size}, model expects: {model_expected_size})")
                            # Don't add to adapted_weights - model will use random init for this layer
                elif 'actor_lstm.weight_hh_l' in k or 'actor_lstm.bias_ih_l' in k or 'actor_lstm.bias_hh_l' in k:
                    # LSTM hidden-to-hidden weights and biases don't depend on input_size
                    # They only depend on hidden_size, so they should match if architectures are compatible
                    # Just copy them - if there's a mismatch, strict=False will handle it
                    adapted_weights[k] = v
                else:
                    adapted_weights[k] = v
            
            actor_weights = adapted_weights
            print("   ✓ Adapted weights from 8 to 9 actions")
        
        # Load with strict=False to allow architecture mismatches
        missing_keys, unexpected_keys = model.load_state_dict(actor_weights, strict=False)
        if missing_keys:
            print(f"   ⚠️  Missing keys (using random init): {len(missing_keys)} layers")
        if unexpected_keys:
            print(f"   ⚠️  Unexpected keys (ignored): {len(unexpected_keys)} layers")
        print(f"✓ Loaded actor weights from: {args.actor_weights}")
        
    except Exception as e:
        print(f"Error loading weights: {e}")
        print("\nTroubleshooting:")
        print("  1. Make sure you're using actor-only weights (actor_*.pth)")
        print("  2. Or use full model weights - critic weights will be filtered automatically")
        print("  3. Check that the model architecture matches (K=16, 9 actions)")
        return
    
    model.eval()

    curve_config, _ = load_curve_config(args.config)
    curve_maker = CurveMakerFlexible(h=128, w=128, seed=args.seed, config=curve_config)

    img, _, pts_all = curve_maker.sample_curve()
    gt_poly = pts_all[0].astype(np.float32)

    env = InteractiveEnv(img, gt_poly, start_idx=args.start_idx)

    ahist = [np.zeros(N_ACTIONS, dtype=np.float32) for _ in range(K)]
    step = 0
    done = False

    print("\nINTERACTIVE MODE — press any key to advance, 'q' to quit\n")

    while not done:
        obs = env.obs()
        obs_t = torch.tensor(obs[None], device=DEVICE)
        A = fixed_window_history(ahist, K, N_ACTIONS)[None]
        A_t = torch.tensor(A, device=DEVICE)

        with torch.no_grad():
            logits, _ = model(obs_t, A_t)
            logits = torch.clamp(logits, -20, 20)
            probs = torch.softmax(logits / TEMPERATURE, dim=1)

            if step < MIN_STEPS_BEFORE_STOP:
                probs[0, ACTION_STOP_IDX] = 0.0
                probs /= probs.sum()

            action = torch.multinomial(probs, 1).item()

        done, reason = env.step(action)

        ah = np.zeros(N_ACTIONS, dtype=np.float32)
        ah[action] = 1.0
        ahist.append(ah)

        print(f"Step {step}: {ACTION_NAMES[action]}")
        step += 1

        plt.imshow(img, cmap="gray")
        px = [p[1] for p in env.path_points]
        py = [p[0] for p in env.path_points]
        plt.plot(px, py, "c-")
        plt.scatter(px[-1], py[-1], c="r")
        plt.title(f"Step {step} | {ACTION_NAMES[action]}")
        plt.show(block=True)

        if done:
            print(f"\nFinished: {reason}")

if __name__ == "__main__":
    main()
