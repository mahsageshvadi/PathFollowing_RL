#!/usr/bin/env python3
"""
Interactive Inference on Real Images.
1. Loads a real image.
2. User clicks Start + Direction.
3. Step-by-step inference with visualization.
"""
import argparse
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import sys

# Add parent directory
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

from src.models_deeper import ActorOnly

# --- CONSTANTS ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ACTIONS_MOVEMENT = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
ACTION_NAMES = ["U", "D", "L", "R", "UL", "UR", "DL", "DR", "STOP"] 
ACTION_STOP_IDX = 8  # <--- Added this missing constant
N_ACTIONS = 8
K = 16
CROP = 33

# --- HELPERS ---
def clamp(v, lo, hi): return max(lo, min(v, hi))

def crop32_inference(img, cy, cx, size=CROP):
    """Robust cropping for real images with padding."""
    h, w = img.shape
    corners = [img[0,0], img[0,w-1], img[h-1,0], img[h-1,w-1]]
    bg_estimate = np.median(corners)
    pad_val = 1.0 if bg_estimate > 0.5 else 0.0

    r = size // 2
    y0, y1 = cy - r, cy + r + 1
    x0, x1 = cx - r, cx + r + 1

    out = np.full((size, size), pad_val, dtype=np.float32)
    sy0, sy1 = clamp(y0, 0, h), clamp(y1, 0, h)
    sx0, sx1 = clamp(x0, 0, w), clamp(x1, 0, w)

    oy0 = sy0 - y0
    ox0 = sx0 - x0
    sh = sy1 - sy0
    sw = sx1 - sx0

    if sh > 0 and sw > 0:
        out[oy0:oy0+sh, ox0:ox0+sw] = img[sy0:sy1, sx0:sx1]

    return out

def get_closest_action(dy, dx):
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

def fixed_window_history(ahist_list, K, n_actions):
    out = np.zeros((K, n_actions), dtype=np.float32)
    if len(ahist_list) == 0: return out
    tail = ahist_list[-K:]
    out[-len(tail):] = np.stack(tail, axis=0)
    return out

def preprocess_image(path, invert=False):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image: {path}")
    
    # 1. Resize if too large (helps with the "Vessel Width" problem)
    # If the image is massive (e.g. 1024x1024), the vessels are too thick for a 33x33 crop.
    # We resize it so the main vessels are closer to 5-8 pixels wide.
    h, w = img.shape
    target_dim = 512  # Try 256 or 512 depending on vessel thickness
    scale = target_dim / max(h, w)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        print(f"ℹ️ Resized image to {img.shape} for better vessel-to-crop ratio")

    # 2. Invert if requested (Make vessels LIGHT on DARK background)
    # DSA images are usually Dark vessels on Light BG. The model prefers Light on Dark.
    # If we don't invert via flag, we try to auto-detect.
    if invert or np.mean(img) > 127: 
        img = 255 - img
        print("ℹ️ Image Inverted (Vessels are now light)")

    # 3. CLAHE (Contrast Limited Adaptive Histogram Equalization)
    # This pulls the faint vessels out of the grey background.
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    img = clahe.apply(img)

    # 4. Normalize 0-1
    img = img.astype(np.float32) / 255.0
    
    # 5. Thresholding (Optional but helpful)
    # Kills background noise, leaving only vessels
    # img = np.maximum(0, img - 0.2) # Remove gray background
    # img = img / (img.max() + 1e-6) # Renormalize
    
    return img

# --- INTERACTIVE CLASS ---
class RealImageStepper:
    def __init__(self, model, image_path, invert=False):
        self.model = model
        self.raw_img = preprocess_image(image_path, invert)
        self.h, self.w = self.raw_img.shape
        
        self.step_alpha = 1.0 
        self.clicks = []
        self.initialized = False
        self.done = False
        
        # Setup Plots
        self.fig = plt.figure(figsize=(14, 8))
        gs = gridspec.GridSpec(2, 3, width_ratios=[3, 1, 1])
        
        self.ax_main = self.fig.add_subplot(gs[:, 0])
        self.ax_crop0 = self.fig.add_subplot(gs[0, 1])
        self.ax_crop1 = self.fig.add_subplot(gs[0, 2])
        self.ax_probs = self.fig.add_subplot(gs[1, 1:])
        
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        
        # Initial display
        self.ax_main.imshow(self.raw_img, cmap='gray', vmin=0, vmax=1)
        self.ax_main.set_title("CLICK 1: Start Point | CLICK 2: Direction")
        
        print("\n--- INSTRUCTIONS ---")
        print("1. Click on the vessel to START.")
        print("2. Click slightly further along the vessel to set DIRECTION.")
        print("3. Press [SPACE] to step, [R] to reset clicks.")
        print("--------------------")
        
        plt.show()

    def init_agent(self):
        if len(self.clicks) < 2: return
        
        p1 = self.clicks[0] # (y, x)
        p2 = self.clicks[1]
        
        # Vector
        vy, vx = p2[0] - p1[0], p2[1] - p1[1]
        mag = np.sqrt(vy**2 + vx**2) + 1e-6
        norm_y, norm_x = vy/mag, vx/mag
        
        # Initialize State
        self.agent = (float(p1[0]), float(p1[1]))
        
        # Create history based on momentum
        prev1 = (self.agent[0] - norm_y, self.agent[1] - norm_x)
        prev2 = (self.agent[0] - 2*norm_y, self.agent[1] - 2*norm_x)
        self.history_pos = [prev2, prev1, self.agent]
        
        self.path = [self.agent]
        self.path_mask = np.zeros_like(self.raw_img)
        self.path_mask[int(self.agent[0]), int(self.agent[1])] = 1.0
        
        # Prime Action History
        start_action = get_closest_action(norm_y, norm_x)
        self.ahist = [np.zeros(N_ACTIONS)] * K
        self.ahist[-1][start_action] = 1.0
        
        self.last_probs = np.zeros(N_ACTIONS)
        self.last_action = start_action
        self.step_count = 0
        self.done = False
        self.initialized = True
        
        self.update_plot()
        self.ax_main.set_title("Press [SPACE] to step | [R] Reset")

    def on_click(self, event):
        if event.inaxes != self.ax_main: return
        if self.initialized: return # Ignore clicks after start
        
        ix, iy = float(event.xdata), float(event.ydata)
        self.clicks.append((iy, ix))
        
        self.ax_main.plot(ix, iy, 'ro', markersize=5)
        if len(self.clicks) == 2:
            self.ax_main.plot([self.clicks[0][1], self.clicks[1][1]], 
                              [self.clicks[0][0], self.clicks[1][0]], 'r-')
            self.init_agent()
        
        self.fig.canvas.draw()

    def step(self):
        if not self.initialized or self.done: return

        curr = self.history_pos[-1]
        p1 = self.history_pos[-2]
        p2 = self.history_pos[-3]
        
        # Prepare Obs
        ch0 = crop32_inference(self.raw_img, int(curr[0]), int(curr[1]))
        ch1 = crop32_inference(self.raw_img, int(p1[0]), int(p1[1]))
        ch2 = crop32_inference(self.raw_img, int(p2[0]), int(p2[1]))
        ch3 = crop32_inference(self.path_mask, int(curr[0]), int(curr[1]))
        
        obs = np.stack([ch0, ch1, ch2, ch3], axis=0).astype(np.float32)
        obs_t = torch.tensor(obs[None], dtype=torch.float32, device=DEVICE)
        
        ahist_arr = fixed_window_history(self.ahist, K, N_ACTIONS)
        hist_t = torch.tensor(ahist_arr[None], dtype=torch.float32, device=DEVICE)
        
        # Inference
        with torch.no_grad():
            logits, _ = self.model(obs_t, hist_t)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        
        self.last_probs = probs
        action = np.argmax(probs)
        self.last_action = action
        
        # Stop Check
        if action == ACTION_STOP_IDX:
            print("🛑 Model predicted STOP")
            self.done = True
            self.update_plot()
            return

        # Move
        dy, dx = ACTIONS_MOVEMENT[action]
        ny = self.agent[0] + dy * self.step_alpha
        nx = self.agent[1] + dx * self.step_alpha
        
        # Bounds Check
        if not (0 <= ny < self.h and 0 <= nx < self.w):
            print("🛑 Hit Image Border")
            self.done = True
            return

        self.agent = (ny, nx)
        self.history_pos.append(self.agent)
        self.path.append(self.agent)
        self.path_mask[int(ny), int(nx)] = 1.0
        
        # Update History
        oh = np.zeros(N_ACTIONS)
        oh[action] = 1.0
        self.ahist.append(oh)
        
        self.step_count += 1
        self.update_plot()

    def update_plot(self):
        self.ax_main.clear()
        self.ax_main.imshow(self.raw_img, cmap='gray', vmin=0, vmax=1)
        self.ax_main.set_title(f"Step: {self.step_count}")
        
        # Draw Path
        path_arr = np.array(self.path)
        if len(path_arr) > 0:
            self.ax_main.plot(path_arr[:, 1], path_arr[:, 0], 'c-', linewidth=2)
            self.ax_main.plot(path_arr[-1, 1], path_arr[-1, 0], 'ro', markersize=4)

        # Draw Crops
        if self.history_pos:
            curr = self.history_pos[-1]
            ch0 = crop32_inference(self.raw_img, int(curr[0]), int(curr[1]))
            ch3 = crop32_inference(self.path_mask, int(curr[0]), int(curr[1]))
            
            self.ax_crop0.clear()
            self.ax_crop0.imshow(ch0, cmap='gray', vmin=0, vmax=1)
            self.ax_crop0.set_title("Agent View")
            self.ax_crop0.axis('off')
            
            self.ax_crop1.clear()
            self.ax_crop1.imshow(ch3, cmap='bone')
            self.ax_crop1.set_title("Memory")
            self.ax_crop1.axis('off')

        # Draw Probs
        self.ax_probs.clear()
        display_names = ACTION_NAMES if len(self.last_probs) == 9 else ACTION_NAMES[:8]
        bars = self.ax_probs.bar(display_names, self.last_probs, color='skyblue')
        if self.last_action is not None and self.last_action < len(bars):
            bars[self.last_action].set_color('orange')
        self.ax_probs.set_ylim(0, 1.0)
        
        self.fig.canvas.draw()

    def on_key(self, event):
        if event.key in [' ', 'n', 'N']: self.step()
        elif event.key in ['r', 'R']:
            # Reset everything
            self.clicks = []
            self.initialized = False
            self.done = False
            self.path = []
            self.ax_main.clear()
            self.ax_main.imshow(self.raw_img, cmap='gray', vmin=0, vmax=1)
            self.ax_main.set_title("CLICK 1: Start Point | CLICK 2: Direction")
            self.fig.canvas.draw()
        elif event.key in ['q', 'Q']: plt.close()

# --- ENTRY POINT ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to real image")
    parser.add_argument("--weights", type=str, required=True, help="Path to .pth model")
    parser.add_argument("--invert", action="store_true", help="Invert image colors (use if background is white)")
    args = parser.parse_args()

    # Load Model
    print(f"Initializing Model (N_ACTIONS={N_ACTIONS})...")
    model = ActorOnly(n_actions=N_ACTIONS, K=K).to(DEVICE)
    
    try:
        loaded = torch.load(args.weights, map_location=DEVICE)
        clean_weights = {}
        for k, v in loaded.items():
            if k.startswith('actor_'): clean_weights[k] = v
            elif 'critic' not in k: clean_weights[f"actor_{k}"] = v
        
        model.load_state_dict(clean_weights, strict=False)
        model.eval()
        print("✅ Model loaded!")
    except Exception as e:
        print(f"❌ Error loading weights: {e}")
        return

    # Start Interface
    RealImageStepper(model, args.image, args.invert)

if __name__ == "__main__":
    main()