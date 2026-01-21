#!/usr/bin/env python3
"""
Inference script for DSA RL Experiment.
Uses ActorOnly model (no critic) for efficient inference.
Refined rollout for the newer deeper + longer-memory actor (K=16, multi-layer LSTM).
"""
import argparse
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
import os
import sys

# Add parent directory to path so 'src' package can be imported
_script_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_script_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

# Use absolute import - works both as module and script
from src.models_deeper import ActorOnly

# ---------- CONSTANTS ----------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 8 movement actions + 1 stop action = 9
ACTIONS_MOVEMENT = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
ACTION_STOP_IDX = 8
N_ACTIONS = 8
CROP = 33

def clamp(v, lo, hi):
    return max(lo, min(v, hi))

def fixed_window_history(ahist_list, K, n_actions):
    """Create fixed-size window of action history."""
    out = np.zeros((K, n_actions), dtype=np.float32)
    if len(ahist_list) == 0:
        return out
    tail = ahist_list[-K:]
    out[-len(tail):] = np.stack(tail, axis=0)
    return out

# ---------- UTILITIES ----------
def preprocess_full_image(path):
    """Load and preprocess DSA image for inference."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    # CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
    img = clahe.apply(img)

    # Normalize
    img = img.astype(np.float32) / 255.0

    # Auto-invert if light background
    if np.median(img) > 0.5:
        print("Detected light background. Inverting image...")
        img = 1.0 - img

    return img

def crop32_inference(img: np.ndarray, cy: int, cx: int, size=CROP):
    """Crop 33x33 patch from full image with smart padding."""
    h, w = img.shape
    corners = [img[0, 0], img[0, w - 1], img[h - 1, 0], img[h - 1, w - 1]]
    bg_estimate = np.median(corners)
    pad_val = 1.0 if bg_estimate > 0.5 else 0.0

    r = size // 2
    y0, y1 = cy - r, cy + r + 1
    x0, x1 = cx - r, cx + r + 1

    out = np.full((size, size), pad_val, dtype=img.dtype)
    sy0, sy1 = clamp(y0, 0, h), clamp(y1, 0, h)
    sx0, sx1 = clamp(x0, 0, w), clamp(x1, 0, w)

    oy0 = sy0 - y0
    ox0 = sx0 - x0
    sh = sy1 - sy0
    sw = sx1 - sx0

    if sh > 0 and sw > 0:
        out[oy0:oy0 + sh, ox0:ox0 + sw] = img[sy0:sy1, sx0:sx1]

    return out

def get_closest_action(dy, dx):
    """Match click vector to discrete movement action (0-7)."""
    best_idx = -1
    max_dot = -float("inf")
    mag = np.sqrt(dy**2 + dx**2) + 1e-6
    uy, ux = dy / mag, dx / mag

    for i, (ay, ax) in enumerate(ACTIONS_MOVEMENT):
        amag = np.sqrt(ay**2 + ax**2) + 1e-6
        ny, nx = ay / amag, ax / amag
        dot = uy * ny + ux * nx
        if dot > max_dot:
            max_dot = dot
            best_idx = i
    return best_idx

# ---------- INFERENCE ENVIRONMENT ----------
class InferenceEnv:
    """Environment for inference on full-size DSA images."""
    def __init__(self, full_img, start_pt, start_vector, max_steps=1000, step_alpha=1.0):
        self.img = full_img
        self.h, self.w = full_img.shape
        self.max_steps = max_steps
        self.step_alpha = float(step_alpha)

        # Initialize agent position (y, x)
        self.agent = (float(start_pt[0]), float(start_pt[1]))

        # Setup momentum from start vector
        dy, dx = float(start_vector[0]), float(start_vector[1])
        mag = np.sqrt(dy**2 + dx**2) + 1e-6
        dy, dx = (dy / mag) * self.step_alpha, (dx / mag) * self.step_alpha

        # History for momentum (3 positions)
        p_prev1 = (self.agent[0] - dy, self.agent[1] - dx)
        p_prev2 = (self.agent[0] - 2 * dy, self.agent[1] - 2 * dx)
        self.history_pos = [p_prev2, p_prev1, self.agent]

        self.path_points = [self.agent]
        self.steps = 0
        self.path_mask = np.zeros_like(full_img, dtype=np.float32)
        self.path_mask[int(self.agent[0]), int(self.agent[1])] = 1.0

        self.visited = set()
        self.visited.add((int(self.agent[0]), int(self.agent[1])))

        # Softer loop detection
        self.loop_count = 0

    def obs(self):
        """Get observation for current position."""
        curr = self.history_pos[-1]
        p1 = self.history_pos[-2]
        p2 = self.history_pos[-3]

        ch0 = crop32_inference(self.img, int(curr[0]), int(curr[1]))
        ch1 = crop32_inference(self.img, int(p1[0]), int(p1[1]))
        ch2 = crop32_inference(self.img, int(p2[0]), int(p2[1]))
        ch3 = crop32_inference(self.path_mask, int(curr[0]), int(curr[1]))

        actor_obs = np.stack([ch0, ch1, ch2, ch3], axis=0).astype(np.float32)
        return actor_obs

    def step(self, a_idx):
        """Execute action and return (done, reason)."""
        self.steps += 1

        # STOP action (only if STOP action exists in action space)
        if ACTION_STOP_IDX < N_ACTIONS and a_idx == ACTION_STOP_IDX:
            return True, "Stopped by Agent"

        # Movement action
        dy, dx = ACTIONS_MOVEMENT[a_idx]
        ny = self.agent[0] + dy * self.step_alpha
        nx = self.agent[1] + dx * self.step_alpha

        # Bounds check
        if not (0 <= ny < self.h and 0 <= nx < self.w):
            return True, "Hit Image Border"

        iy, ix = int(ny), int(nx)

        # Softer loop detection: allow brief revisits, stop only if persistent
        if (iy, ix) in self.visited:
            self.loop_count += 1
        else:
            self.loop_count = 0
            self.visited.add((iy, ix))

        if self.steps > 20 and self.loop_count > 5:
            return True, "Loop Detected"

        # Update state
        self.agent = (ny, nx)
        self.history_pos.append(self.agent)
        self.path_points.append(self.agent)
        self.path_mask[iy, ix] = 1.0

        if self.steps >= self.max_steps:
            return True, "Max Steps Reached"

        return False, "Running"

# ---------- INTERACTIVE CLICK HANDLER ----------
coords = []
def onclick(event):
    """Handle mouse clicks for start and direction selection."""
    global coords
    if event.xdata is None or event.ydata is None:
        return
    ix, iy = int(event.xdata), int(event.ydata)
    coords.append((ix, iy))

    plt.plot(ix, iy, "ro", markersize=5)
    if len(coords) > 1:
        plt.plot([coords[-2][0], coords[-1][0]],
                 [coords[-2][1], coords[-1][1]], "r-", linewidth=2)
    plt.draw()

    if len(coords) == 2:
        print("Direction set. Starting tracking...")
        plt.pause(0.5)
        plt.close()

# ---------- MAIN ----------
def main():
    parser = argparse.ArgumentParser(description="DSA RL Inference - Track curves in real DSA images")
    parser.add_argument("--image_path", type=str, required=True, help="Path to DSA image file")
    parser.add_argument("--actor_weights", type=str, required=True,
                        help="Path to actor-only weights (e.g., actor_Stage3_Realism_FINAL.pth)")
    parser.add_argument("--max_steps", type=int, default=1000, help="Maximum steps before stopping")
    parser.add_argument("--temperature", type=float, default=0.5,
                        help="Sampling temperature (lower=more deterministic). Recommended ~0.3-0.8")
    parser.add_argument("--min_steps_before_stop", type=int, default=20,
                        help="Prevent STOP action before this many steps")
    parser.add_argument("--step_alpha", type=float, default=2.0,
                        help="Step size per move action (pixels). Must match training/inference behavior you want.")
    args = parser.parse_args()

    # Load raw image for display
    raw_img = cv2.imread(args.image_path, cv2.IMREAD_GRAYSCALE)
    if raw_img is None:
        print(f"Error: Could not load image from {args.image_path}")
        return

    # Preprocess for the agent
    processed_img = preprocess_full_image(args.image_path)
    if processed_img is None:
        print(f"Error: preprocess failed for {args.image_path}")
        return

    # Load ActorOnly model (NEW STRUCTURE expects longer memory)
    K = 16  # <-- IMPORTANT: must match training
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

    # Get start point and direction from user clicks
    print("\n--- INSTRUCTIONS ---")
    print("1. Click START point")
    print("2. Click DIRECTION point")
    print("--------------------\n")

    plt.figure(figsize=(12, 12))
    plt.imshow(raw_img, cmap="gray")
    plt.title("Select Start & Direction")
    plt.connect("button_press_event", onclick)
    plt.show()

    if len(coords) < 2:
        print("Error: Need 2 clicks (start and direction)")
        return

    # Convert matplotlib (x,y) to numpy (y,x)
    p1_x, p1_y = coords[0]
    p2_x, p2_y = coords[1]
    vec_y = p2_y - p1_y
    vec_x = p2_x - p1_x

    # Init env
    env = InferenceEnv(
        processed_img,
        start_pt=(p1_y, p1_x),
        start_vector=(vec_y, vec_x),
        max_steps=args.max_steps,
        step_alpha=args.step_alpha
    )

    # Prime action history with the start direction
    start_action = get_closest_action(vec_y, vec_x)
    a_onehot = np.zeros(N_ACTIONS, dtype=np.float32)
    a_onehot[start_action] = 1.0
    ahist = [a_onehot] * K

    # --------- REFINED ROLLOUT (NEW STRUCTURE) ----------
    done = False
    print(f"Tracking started... (Max steps: {args.max_steps})")

    temperature = max(1e-6, float(args.temperature))
    min_steps_before_stop = int(args.min_steps_before_stop)

    while not done:
        obs = env.obs()
        obs_t = torch.tensor(obs[None], dtype=torch.float32, device=DEVICE)

        A = fixed_window_history(ahist, K, N_ACTIONS)[None, ...]
        A_t = torch.tensor(A, dtype=torch.float32, device=DEVICE)

        with torch.no_grad():
            logits, _hc = model(obs_t, A_t)
            logits = torch.clamp(logits, -20, 20)

            probs = torch.softmax(logits / temperature, dim=1)

            # Prevent early STOP (only if STOP action exists in action space)
            if env.steps < min_steps_before_stop and ACTION_STOP_IDX < N_ACTIONS:
                probs[0, ACTION_STOP_IDX] = 0.0
                probs = probs / (probs.sum(dim=1, keepdim=True) + 1e-8)

            # Sample (better than argmax for memory-based PPO policies)
            action = torch.multinomial(probs, 1).item()

        done, reason = env.step(action)

        # Update action history
        new_onehot = np.zeros(N_ACTIONS, dtype=np.float32)
        new_onehot[action] = 1.0
        ahist.append(new_onehot)
        
        # Print stop reason immediately when done
        if done:
            print(f"\n{'='*60}")
            print(f"🛑 STOPPED: {reason}")
            print(f"   Total steps: {env.steps}")
            print(f"   Path length: {len(env.path_points)} points")
            print(f"{'='*60}\n")

    print(f"Finished: {reason} ({env.steps} steps)")

    # Visualize result
    path = env.path_points
    try:
        y = [p[0] for p in path]
        x = [p[1] for p in path]
        tck, u = splprep([y, x], s=20.0)
        new = splev(np.linspace(0, 1, len(path) * 3), tck)
        sy, sx = new[0], new[1]
    except Exception:
        sy, sx = [p[0] for p in path], [p[1] for p in path]

    plt.figure(figsize=(12, 12))
    plt.imshow(raw_img, cmap="gray")
    plt.plot(sx, sy, "cyan", linewidth=2, label="Tracked Path")
    plt.plot(p1_x, p1_y, "go", markersize=8, label="Start")
    if "Stopped" in reason:
        plt.plot(path[-1][1], path[-1][0], "rx", markersize=10, markeredgewidth=3, label="Stop")
    plt.legend()
    plt.title(f"Result: {reason}")
    plt.show()

if __name__ == "__main__":
    main()
