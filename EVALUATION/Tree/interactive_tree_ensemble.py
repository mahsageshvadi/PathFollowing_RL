#!/usr/bin/env python3
import argparse
import os
import sys
import copy
import numpy as np

import matplotlib
import platform
if platform.system() == "Darwin": matplotlib.use('MacOSX')
else:
    try: matplotlib.use('TkAgg')
    except: pass

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import cv2

from models_deeper import AsymmetricActorCritic

# -------------------------
# Constants
# -------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ACTIONS_MOVEMENT = [(-1, 0), (1, 0), (0, -1), (0, 1),
                    (-1, -1), (-1, 1), (1, -1), (1, 1)]
ACTION_NAMES = ["U", "D", "L", "R", "UL", "UR", "DL", "DR"]
N_ACTIONS = 8
CROP = 33
STEP_SIZE = 2.0

def clamp(v, lo, hi): return max(lo, min(v, hi))

def window_crop(crop, p_low=2, p_high=98):
    lo, hi = np.percentile(crop, p_low), np.percentile(crop, p_high)
    if hi - lo < 1e-6:
        return crop.astype(np.float32)
    return np.clip((crop - lo) / (hi - lo), 0, 1).astype(np.float32)

def crop32_inference(img, cy, cx):
    h, w = img.shape
    pad_val = 1.0 if np.median([img[0,0], img[0,-1],
                               img[-1,0], img[-1,-1]]) > 0.5 else 0.0
    r = CROP // 2
    y0, y1 = int(cy - r), int(cy + r + 1)
    x0, x1 = int(cx - r), int(cx + r + 1)

    out = np.full((CROP, CROP), pad_val, dtype=np.float32)
    sy0, sy1 = clamp(y0, 0, h), clamp(y1, 0, h)
    sx0, sx1 = clamp(x0, 0, w), clamp(x1, 0, w)

    out[sy0-y0:sy0-y0+(sy1-sy0),
        sx0-x0:sx0-x0+(sx1-sx0)] = img[sy0:sy1, sx0:sx1]
    return out

def fixed_window_history(ahist_list, K, n_actions):
    out = np.zeros((K, n_actions), dtype=np.float32)
    if not ahist_list:
        return out
    tail = ahist_list[-K:]
    out[-len(tail):] = np.stack(tail, axis=0)
    return out

# ======================================================
# Tree Builder
# ======================================================

class ManualTreeBuilder:
    def __init__(self, model_junc, image_path):
        self.model = model_junc

        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if np.mean(img) > 127:
            img = 255 - img
        self.raw_img = cv2.createCLAHE(
            clipLimit=2.0, tileGridSize=(8, 8)
        ).apply(img).astype(np.float32) / 255.0
        self.h, self.w = self.raw_img.shape

        # --- Tree state ---
        self.stack = []
        self.all_segments = []
        self.current_segment = []
        self.global_visited = np.zeros_like(self.raw_img)

        self.junction_cooldown = 0
        self.min_branch_dist = 15.0

        self.running = False
        self.initialized = False
        self.done = False
        self.clicks = []

        # -------------------------
        # Visualization setup
        # -------------------------
        self.fig = plt.figure(figsize=(16, 9))
        self.timer = self.fig.canvas.new_timer(interval=30)
        self.timer.add_callback(self.auto_step)

        gs = gridspec.GridSpec(2, 3, width_ratios=[3, 1, 1])
        self.ax_main = self.fig.add_subplot(gs[:, 0])
        self.ax_crop = self.fig.add_subplot(gs[0, 1])
        self.ax_mask = self.fig.add_subplot(gs[0, 2])
        self.ax_probs = self.fig.add_subplot(gs[1, 1:])

        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

        # --- Live visualization buffers ---
        self.vis_c0 = np.zeros((CROP, CROP), dtype=np.float32)
        self.vis_c3 = np.zeros((CROP, CROP), dtype=np.float32)

        self.last_p_junc = 0.0
        self.last_probs = np.zeros(N_ACTIONS)
        self.step_count = 0

        self.update_plot()
        print("\n--- 🌳 TREE DEBUGGER ---")
        print("Click Start -> Click Direction. [SPACE] to Build/Jump. [R] to Reset.")

    # --------------------------------------------------

    def init_agent(self, p1, p2):
        self.agent = (float(p1[0]), float(p1[1]))
        dy, dx = p2[0]-p1[0], p2[1]-p1[1]
        mag = np.sqrt(dy**2 + dx**2) + 1e-6
        uy, ux = dy/mag, dx/mag

        self.history_pos = [
            (self.agent[0] - 2*uy, self.agent[1] - 2*ux),
            (self.agent[0] - uy, self.agent[1] - ux),
            self.agent
        ]

        idx = np.argmax([uy*ay + ux*ax for ay, ax in ACTIONS_MOVEMENT])
        self.ahist = [np.zeros(N_ACTIONS) for _ in range(16)]
        self.ahist[-1][idx] = 1.0

        self.current_segment = [self.agent]
        self.path_mask = np.zeros_like(self.raw_img)

        self.done = False
        self.initialized = True
        self.step_count = 0

        self.running = True
        self.timer.start()
        self.update_plot()

    # --------------------------------------------------

    def step(self):
        if not self.initialized or self.done:
            return

        curr, p1, p2 = self.history_pos[-1], self.history_pos[-2], self.history_pos[-3]

        # --------------------------------------------------
        # Agent crops (SAVE EXACTLY WHAT AGENT SEES)
        # --------------------------------------------------
        self.vis_c0 = window_crop(
            crop32_inference(self.raw_img, curr[0], curr[1])
        )
        self.vis_c3 = crop32_inference(
            self.path_mask, curr[0], curr[1]
        )

        c0 = self.vis_c0
        c1 = window_crop(crop32_inference(self.raw_img, p1[0], p1[1]))
        c2 = window_crop(crop32_inference(self.raw_img, p2[0], p2[1]))
        c3 = self.vis_c3

        obs_t = torch.tensor(
            np.stack([c0, c1, c2, c3])[None],
            dtype=torch.float32,
            device=DEVICE
        )
        h16_t = torch.tensor(
            fixed_window_history(self.ahist, 16, N_ACTIONS)[None],
            device=DEVICE
        )

        with torch.no_grad():
            move_logits, stop_logit, _, j_logit, _ = self.model(obs_t, ...)
            
            stop_prob = torch.sigmoid(stop_logit).item()
            move_probs = torch.softmax(move_logits, dim=1).cpu().numpy()[0]
        # --------------------------------------------------
        # Junction Logic
        # --------------------------------------------------
        if self.junction_cooldown > 0:
            self.junction_cooldown -= 1

        if self.last_p_junc > 0.7 and self.junction_cooldown == 0:
            too_close = False
            for b in self.stack:
                d = np.hypot(
                    self.agent[0] - b['pos'][0],
                    self.agent[1] - b['pos'][1]
                )
                if d < self.min_branch_dist:
                    too_close = True
                    break

            if not too_close:
                top_idx = np.argsort(self.last_probs)[-2:]
                self.stack.append({
                    'pos': self.agent,
                    'ahist': copy.deepcopy(self.ahist),
                    'act': top_idx[0]
                })
                self.junction_cooldown = 20
                print(f"🌿 Junction Saved at step {self.step_count}")
                action = top_idx[1]
            else:
                action = np.argmax(self.last_probs)
        else:
            action = np.argmax(self.last_probs)

        # --------------------------------------------------
        # Movement
        # --------------------------------------------------
        dy, dx = ACTIONS_MOVEMENT[action]
        ny = self.agent[0] + dy * STEP_SIZE
        nx = self.agent[1] + dx * STEP_SIZE

        # Stop conditions
        if not (0 <= ny < self.h and 0 <= nx < self.w):
            self.finish_segment("IMAGE BOUNDARY")
            return

        if self.step_count > 15 and self.global_visited[int(ny), int(nx)] > 0.5:
            self.finish_segment("VIRTUAL COLLISION / LOOP")
            return

        self.agent = (ny, nx)
        self.history_pos.append(self.agent)
        self.current_segment.append(self.agent)

        self.path_mask[int(ny), int(nx)] = 1.0
        self.global_visited[int(ny), int(nx)] = 1.0

        self.ahist.append(np.eye(N_ACTIONS)[action])
        self.step_count += 1
        self.update_plot()

    # --------------------------------------------------

    def finish_segment(self, reason):
        print(f"⏹️ Segment Stop: {reason} at step {self.step_count}")
        self.all_segments.append(self.current_segment)
        self.done = True
        self.update_plot()

    # --------------------------------------------------

    def auto_step(self):
        if not self.running:
            return

        if self.done:
            if self.stack:
                b = self.stack.pop()
                print(f"🔄 Popping Branch... ({len(self.stack)} remaining)")
                dy, dx = ACTIONS_MOVEMENT[b['act']]
                self.init_agent(
                    b['pos'],
                    (b['pos'][0] + dy, b['pos'][1] + dx)
                )
            else:
                print("🏁 Tree Complete.")
                self.running = False
                self.timer.stop()
            return

        self.step()

    # --------------------------------------------------

    def update_plot(self):
        # Main view
        self.ax_main.clear()
        self.ax_main.imshow(self.raw_img, cmap="gray", vmin=0, vmax=1)

        for seg in self.all_segments:
            pts = np.array(seg)
            self.ax_main.plot(pts[:,1], pts[:,0], color='yellow', alpha=0.3)

        if len(self.current_segment) > 0:
            pts = np.array(self.current_segment)
            self.ax_main.plot(pts[:,1], pts[:,0], color='cyan', linewidth=2)
            self.ax_main.plot(pts[-1,1], pts[-1,0], 'ro', markersize=4)

        for b in self.stack:
            self.ax_main.plot(b['pos'][1], b['pos'][0], 'go', markersize=4)

        self.ax_main.set_title(
            f"Junc Prob: {self.last_p_junc:.2f} | Step: {self.step_count}"
        )

        # --------------------------
        # LIVE AGENT PATCH
        # --------------------------
        self.ax_crop.clear()
        self.ax_crop.imshow(self.vis_c0, cmap="gray", vmin=0, vmax=1)
        self.ax_crop.set_title("Agent View (c0)")
        self.ax_crop.axis("off")

        self.ax_mask.clear()
        self.ax_mask.imshow(self.vis_c3, cmap="gray", vmin=0, vmax=1)
        self.ax_mask.set_title("Path Memory (c3)")
        self.ax_mask.axis("off")

        self.fig.canvas.draw_idle()

    # --------------------------------------------------

    def on_click(self, event):
        if event.inaxes != self.ax_main or self.initialized:
            return
        self.clicks.append((event.ydata, event.xdata))
        self.ax_main.plot(event.xdata, event.ydata, 'ro')
        if len(self.clicks) == 2:
            self.init_agent(self.clicks[0], self.clicks[1])
        self.fig.canvas.draw()

    def on_key(self, event):
        if event.key == " ":
            print("⌨️ Space hit: Manual Segment Cut.")
            if self.initialized and not self.done:
                self.finish_segment("USER MANUAL CUT")
            if not self.running:
                self.running = True
                self.timer.start()

        elif event.key == "r":
            print("⌨️ R hit: Resetting.")
            self.timer.stop()
            self.running = False
            self.initialized = False
            self.clicks = []
            self.all_segments = []
            self.stack = []
            self.global_visited[:] = 0
            self.update_plot()

        elif event.key == "q":
            plt.close()

# ======================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", default="../Retinal_DRIVE_pngs/21_training.png")
    parser.add_argument("--weights",
        default=os.path.join(os.path.dirname(__file__), "model_Stage7_Branching_Full_Realism_FINAL.pth"),
        type=str)
    args = parser.parse_args()

    model = AsymmetricActorCritic(n_actions=8).to(DEVICE)
    weights_path = os.path.expanduser(args.weights)
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE), strict=False)

    debugger = ManualTreeBuilder(model.eval(), args.img)
    plt.show()
