#!/usr/bin/env python3
"""
Training script for DSA RL Experiment
Trains an agent to follow curves using PPO with curriculum learning.
Curves are generated on-the-fly based strictly on JSON configuration.
"""
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from dataclasses import dataclass
from scipy.ndimage import gaussian_filter
import scipy.ndimage
import os
import sys
import cv2
import shutil
import json
from datetime import datetime

# Add parent directory to path so 'src' package can be imported
_script_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_script_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

# ---------- GLOBALS ----------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    print(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
else:
    print("⚠️  GPU not available, using CPU (training will be slower)")

# 8 Movement Actions + 1 Stop Action = 9 Total
ACTIONS_MOVEMENT = [(-1, 0), (1, 0), (0,-1), (0, 1), (-1,-1), (-1,1), (1,-1), (1,1)]
ACTION_STOP_IDX = 8
N_ACTIONS = 8

STEP_ALPHA = 1.0
CROP = 33
EPSILON = 1e-6
BASE_SEED = 50

# ---------- CONFIG LOADING ----------
def load_curve_config(config_path=None):
    """Load curve generation configuration from JSON file."""
    if config_path is None:
        # Default paths
        paths = [
            os.path.join(_parent_dir, "config", "curve_config.json"),
            os.path.join(_parent_dir, "curve_config.json"),
            os.path.join(_script_dir, "curve_config.json")
        ]
        for p in paths:
            if os.path.exists(p):
                config_path = p
                break
    
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"✓ Loaded curve configuration from: {config_path}")
        return config, config_path
    else:
        print(f"❌ Config file not found: {config_path}")
        sys.exit(1)

# ---------- CURVE GENERATION (On-The-Fly) ----------
def _rng(seed=None):
    return np.random.default_rng(seed)

def _cubic_bezier(p0, p1, p2, p3, t):
    omt = 1.0 - t
    return (omt**3)*p0 + 3*omt*omt*t*p1 + 3*omt*t*t*p2 + (t**3)*p3

class CurveMakerFlexible:
    def __init__(self, h=128, w=128, seed=None, config=None):
        self.h = h
        self.w = w
        self.rng = _rng(seed)
        self.config = config or {}
        
        bezier_cfg = self.config.get('bezier', {})
        self.bezier_n_samples = bezier_cfg.get('n_samples', 1000)
        self.bezier_margin = bezier_cfg.get('margin', 10)
        self.bezier_min_distance = bezier_cfg.get('min_distance', 40.0)
        self.bezier_spread = bezier_cfg.get('control_point_spread', 0.3)
        self.bezier_factor = bezier_cfg.get('control_point_factor', 0.6)
        
        branch_cfg = self.config.get('branches', {})
        self.branch_num_range = tuple(branch_cfg.get('num_branches_range', [1, 3]))
        self.branch_start_range = tuple(branch_cfg.get('start_range', [0.2, 0.8]))
        self.branch_thickness_factor = branch_cfg.get('thickness_factor', 0.7)
        
        noise_cfg = self.config.get('noise', {})
        self.noise_num_blobs_range = tuple(noise_cfg.get('num_blobs_range', [1, 4]))
        self.noise_blob_sigma_range = tuple(noise_cfg.get('blob_sigma_range', [2.0, 8.0]))
        self.noise_blob_intensity_range = tuple(noise_cfg.get('blob_intensity_range', [0.05, 0.2]))
        self.noise_level_range = tuple(noise_cfg.get('noise_level_range', [0.05, 0.15]))
        self.noise_gaussian_blur_prob = noise_cfg.get('gaussian_blur_prob', 0.5)
        self.noise_gaussian_blur_sigma_range = tuple(noise_cfg.get('gaussian_blur_sigma_range', [0.5, 1.0]))

    def _random_point(self, margin=None):
        margin = margin if margin is not None else self.bezier_margin
        y = self.rng.integers(margin, self.h - margin)
        x = self.rng.integers(margin, self.w - margin)
        return np.array([y, x], dtype=np.float32)

    def _generate_bezier_points(self, p0=None, n_samples=None, curvature_factor=1.0,
                                allow_self_cross=False, self_cross_prob=0.0):
        n_samples = n_samples if n_samples is not None else self.bezier_n_samples
        
        if p0 is None: p0 = self._random_point()
        
        for _ in range(20): 
            p3 = self._random_point()
            dist = np.linalg.norm(p0 - p3)
            if curvature_factor < 0.3:
                if dist > 60.0: break
            elif curvature_factor > 1.2:
                if 20.0 < dist < 80.0: break
            else:
                if dist > self.bezier_min_distance: break
        else:
            p3 = np.array([self.h - p0[0], self.w - p0[1]], dtype=np.float32)

        center = (p0 + p3) / 2.0
        
        if curvature_factor < 0.3:
            p1 = p0 + (p3 - p0) * 0.33 + self.rng.normal(0, 1, 2) * curvature_factor * 2.0
            p2 = p0 + (p3 - p0) * 0.66 + self.rng.normal(0, 1, 2) * curvature_factor * 2.0
        elif curvature_factor > 1.2:
            vec = p3 - p0
            norm = np.linalg.norm(vec) + 1e-8
            perp = np.array([-vec[1], vec[0]]) / norm
            spread_mag = self.h * 0.6 * self.rng.uniform(0.8, 1.2)
            
            if allow_self_cross and self.rng.random() < self_cross_prob:
                p1 = center + perp * spread_mag
                p2 = center - perp * spread_mag
            else:
                direction = 1 if self.rng.random() < 0.5 else -1
                p1 = center + perp * spread_mag * direction
                p2 = center + perp * spread_mag * direction
                p1 += self.rng.normal(0, 10, 2)
                p2 += self.rng.normal(0, 10, 2)
        else:
            spread = np.array([self.h, self.w], dtype=np.float32) * self.bezier_spread * curvature_factor
            p1 = center + self.rng.normal(0, 1, 2) * spread * self.bezier_factor
            p2 = center + self.rng.normal(0, 1, 2) * spread * self.bezier_factor
        
        ts = np.linspace(0, 1, n_samples, dtype=np.float32)
        pts = np.stack([_cubic_bezier(p0, p1, p2, p3, t) for t in ts], axis=0)
        margin = self.bezier_margin
        pts[:, 0] = np.clip(pts[:, 0], margin, self.h - margin - 1)
        pts[:, 1] = np.clip(pts[:, 1], margin, self.w - margin - 1)
        return pts

    def _draw_aa_curve(self, img, pts, thickness, intensity, width_variation="none", start_width=None, end_width=None,
                       intensity_variation="none", start_intensity=None, end_intensity=None):
        
        # Determine actual Intensity values
        if intensity_variation == "none":
            start_i = end_i = intensity
        elif intensity_variation == "bright_to_dim":
            start_i = start_intensity if start_intensity is not None else intensity * 1.5
            end_i = end_intensity if end_intensity is not None else intensity * 0.5
        elif intensity_variation == "dim_to_bright":
            start_i = start_intensity if start_intensity is not None else intensity * 0.5
            end_i = end_intensity if end_intensity is not None else intensity * 1.5
        elif intensity_variation == "custom":
            start_i = start_intensity if start_intensity is not None else intensity
            end_i = end_intensity if end_intensity is not None else intensity
        else:
            start_i = end_i = intensity
        
        # Clamp intensities to 0.0-1.0
        start_i = np.clip(start_i, 0.0, 1.0)
        end_i = np.clip(end_i, 0.0, 1.0)

        def draw_segment(mask, pts_seg, th, val):
            pts_xy = pts_seg[:, ::-1] * 16
            pts_int = pts_xy.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(mask, [pts_int], isClosed=False, color=float(val), 
                          thickness=max(1, int(th)), lineType=cv2.LINE_AA, shift=4)

        if width_variation == "none" and intensity_variation == "none":
            canvas = np.zeros_like(img)
            draw_segment(canvas, pts, thickness, 1.0)
            img[:] = np.maximum(img, canvas * intensity)
        else:
            if width_variation == "wide_to_narrow":
                start_w = start_width if start_width is not None else thickness * 2.0
                end_w = end_width if end_width is not None else thickness
            elif width_variation == "narrow_to_wide":
                start_w = start_width if start_width is not None else thickness
                end_w = end_width if end_width is not None else thickness * 2.0
            elif width_variation == "custom":
                start_w = start_width if start_width is not None else thickness
                end_w = end_width if end_width is not None else thickness
            else:
                start_w = end_w = thickness

            n_pts = len(pts)
            steps = 50
            segment_len = max(1, n_pts // steps)
            
            for i in range(0, n_pts - 1, segment_len):
                end_idx = min(i + segment_len, n_pts - 1)
                t_start = i / max(1, n_pts - 1)
                t_end = end_idx / max(1, n_pts - 1)
                t_mid = (t_start + t_end) / 2.0
                
                w_curr = start_w + (end_w - start_w) * t_mid
                i_curr = start_i + (end_i - start_i) * t_mid
                
                seg_mask = np.zeros_like(img)
                draw_segment(seg_mask, pts[i:end_idx+1], w_curr, 1.0)
                img[:] = np.maximum(img, seg_mask * i_curr)

    def sample_curve(self, 
                     width_range=(2, 2),    
                     noise_prob=0.0,        
                     invert_prob=0.0,
                     min_intensity=0.1,  
                     max_intensity=1.0,
                     branches=False,
                     curvature_factor=1.0,
                     width_variation="none",
                     start_width=None,
                     end_width=None,
                     intensity_variation="none",
                     start_intensity=None,
                     end_intensity=None,
                     background_intensity=None,
                     allow_self_cross=False,
                     self_cross_prob=0.0):  
        
        bg_val = background_intensity if background_intensity is not None else 0.0
        img = np.full((self.h, self.w), bg_val, dtype=np.float32)
        mask = np.zeros_like(img) 
        
        thickness = self.rng.integers(width_range[0], width_range[1] + 1) if width_range[1] >= width_range[0] else width_range[0]
        thickness = max(1, int(thickness))

        # Range Logic
        max_int = max_intensity if max_intensity is not None else 1.0
        min_int = min_intensity if min_intensity is not None else 0.1
        if max_int < min_int: max_int = min_int + 0.01

        intensity = self.rng.uniform(min_int, max_int)
        
        pts_main = self._generate_bezier_points(
            curvature_factor=curvature_factor,
            allow_self_cross=allow_self_cross,
            self_cross_prob=self_cross_prob
        )
        
        self._draw_aa_curve(img, pts_main, thickness, intensity, width_variation, start_width, end_width,
                           intensity_variation, start_intensity, end_intensity)
        self._draw_aa_curve(mask, pts_main, thickness, 1.0, width_variation, start_width, end_width)
        pts_all = [pts_main]

        if branches:
            num_branches = self.rng.integers(self.branch_num_range[0], self.branch_num_range[1] + 1)
            for _ in range(num_branches):
                start_min = int(len(pts_main) * self.branch_start_range[0])
                start_max = int(len(pts_main) * self.branch_start_range[1])
                idx = self.rng.integers(start_min, start_max)
                p0 = pts_main[idx]
                pts_branch = self._generate_bezier_points(p0=p0, curvature_factor=curvature_factor)
                b_thick = max(1, int(thickness * self.branch_thickness_factor))
                
                self._draw_aa_curve(img, pts_branch, b_thick, intensity, width_variation)
                self._draw_aa_curve(mask, pts_branch, b_thick, 1.0, width_variation)
                pts_all.append(pts_branch)

        if self.rng.random() < noise_prob:
            self._apply_dsa_noise(img)

        if self.rng.random() < invert_prob:
            img = 1.0 - img

        return np.clip(img, 0.0, 1.0), (mask > 0.1).astype(np.uint8), pts_all

    def _apply_dsa_noise(self, img):
        num_blobs = self.rng.integers(*self.noise_num_blobs_range)
        for _ in range(num_blobs):
            y, x = self._random_point(margin=0)
            sigma = self.rng.uniform(*self.noise_blob_sigma_range)
            yy, xx = np.ogrid[:self.h, :self.w]
            dist_sq = (yy - y)**2 + (xx - x)**2
            blob = np.exp(-dist_sq / (2 * sigma**2))
            blob_int = self.rng.uniform(*self.noise_blob_intensity_range)
            img[:] = np.maximum(img, blob * blob_int)

        noise_level = self.rng.uniform(*self.noise_level_range)
        noise = self.rng.normal(0, noise_level, img.shape)
        img[:] += noise

        if self.rng.random() < self.noise_gaussian_blur_prob:
            sigma = self.rng.uniform(*self.noise_gaussian_blur_sigma_range)
            img[:] = scipy.ndimage.gaussian_filter(img, sigma=sigma)

# ---------- HELPERS ----------
def clamp(v, lo, hi): return max(lo, min(v, hi))

def crop32(img, cy, cx, size=CROP):
    h, w = img.shape
    corners = [img[0,0], img[0, w-1], img[h-1, 0], img[h-1, w-1]]
    bg_estimate = np.median(corners)
    pad_val = 1.0 if bg_estimate > 0.5 else 0.0
    
    r = size // 2
    y0, y1 = cy - r, cy + r + 1
    x0, x1 = cx - r, cx + r + 1
    
    out = np.full((size, size), pad_val, dtype=img.dtype)
    sy0, sy1 = clamp(y0, 0, h), clamp(y1, 0, h)
    sx0, sx1 = clamp(x0, 0, w), clamp(x1, 0, w)
    
    oy0, ox0 = sy0 - y0, sx0 - x0
    sh, sw = sy1 - sy0, sx1 - sx0
    
    if sh > 0 and sw > 0:
        out[oy0:oy0+sh, ox0:ox0+sw] = img[sy0:sy1, sx0:sx1]
    return out

def fixed_window_history(ahist_list, K, n_actions):
    out = np.zeros((K, n_actions), dtype=np.float32)
    if len(ahist_list) == 0: return out
    tail = ahist_list[-K:]
    out[-len(tail):] = np.stack(tail, axis=0)
    return out

def get_distance_to_poly(pt, poly):
    dif = poly - np.array(pt, dtype=np.float32)
    d2 = np.sum(dif * dif, axis=1)
    return np.sqrt(np.min(d2))

def nearest_gt_index(pt, poly):
    dif = poly - np.array(pt, dtype=np.float32)
    d2 = np.sum(dif * dif, axis=1)
    return int(np.argmin(d2))

@dataclass
class CurveEpisode:
    img: np.ndarray
    mask: np.ndarray
    gt_poly: np.ndarray

# ---------- ENVIRONMENT ----------
class CurveEnvUnified:
    def __init__(self, h=128, w=128, max_steps=200, base_seed=BASE_SEED, stage_id=1, curve_config=None):
        self.h, self.w = h, w
        self.max_steps = max_steps
        self.base_seed = base_seed
        self.current_episode = 0
        self.curve_config = curve_config or {}
        self.stage_config = {"stage_id": stage_id}
        self.tissue_cfg = self.curve_config.get('tissue_noise', {})
        self.steps = 0 # FIXED: Initialize steps in init to avoid AttributeError

    def set_stage(self, config: dict):
        self.stage_config = config.copy()

    def generate_tissue_noise(self):
        sigma_range = tuple(self.tissue_cfg.get('sigma_range', [2.0, 5.0]))
        intensity_range = tuple(self.tissue_cfg.get('intensity_range', [0.2, 0.4]))
        
        noise = np.random.randn(self.h, self.w)
        tissue = gaussian_filter(noise, sigma=np.random.uniform(*sigma_range))
        tissue = (tissue - tissue.min()) / (tissue.max() - tissue.min())
        return tissue * np.random.uniform(*intensity_range)

    def reset(self, episode_number=None):
        if episode_number is not None:
            self.current_episode = episode_number
        
        episode_seed = self.base_seed + self.current_episode
        curve_maker = CurveMakerFlexible(h=self.h, w=self.w, seed=episode_seed, config=self.curve_config)
        
        def resolve(key_base, default_val):
            range_key = f"{key_base}_range"
            if range_key in self.stage_config:
                val_range = self.stage_config[range_key]
                return np.random.uniform(val_range[0], val_range[1])
            elif key_base in self.stage_config:
                return self.stage_config[key_base]
            elif f"{key_base}_prob" in self.stage_config:
                return self.stage_config[f"{key_base}_prob"]
            return default_val

        # --- PARAMETER RESOLUTION ---
        w_range = tuple(self.stage_config.get('width_range', [2, 4]))
        
        if 'curvature_range' in self.stage_config:
            cr = self.stage_config['curvature_range']
            curvature = np.random.uniform(cr[0], cr[1])
        else:
            curvature = self.stage_config.get('curvature_factor', 1.0)

        if 'noise_range' in self.stage_config:
            nr = self.stage_config['noise_range']
            noise_prob = np.random.uniform(nr[0], nr[1])
        else:
            noise_prob = self.stage_config.get('noise_prob', 0.0)

        if 'background_intensity_range' in self.stage_config:
            br = self.stage_config['background_intensity_range']
            bg_int = np.random.uniform(br[0], br[1])
        else:
            bg_int = self.stage_config.get('background_intensity', 0.0)

        # Intensity Logic
        min_int = resolve('min_intensity', 0.1)
        max_int = resolve('max_intensity', 1.0)
        
        if min_int < bg_int + 0.02: min_int = bg_int + 0.02
        if max_int < min_int: max_int = min_int + 0.05
        if max_int > 1.0: max_int = 1.0
        if min_int > 1.0: min_int = 1.0

        # Debug Print for first episode of a stage
        if self.current_episode % 100 == 0:
            pass

        img, mask, pts_all = curve_maker.sample_curve(
            width_range=w_range,
            noise_prob=noise_prob,
            invert_prob=self.stage_config.get('invert_prob', 0.0),
            min_intensity=min_int,
            max_intensity=max_int,
            background_intensity=bg_int,
            branches=self.stage_config.get('branches', False),
            curvature_factor=curvature,
            allow_self_cross=self.stage_config.get('allow_self_cross', False),
            self_cross_prob=self.stage_config.get('self_cross_prob', 0.0),
            width_variation=self.stage_config.get('width_variation', 'none'),
            start_width=self.stage_config.get('start_width', None),
            end_width=self.stage_config.get('end_width', None),
            intensity_variation=self.stage_config.get('intensity_variation', 'none'),
            start_intensity=self.stage_config.get('start_intensity', None),
            end_intensity=self.stage_config.get('end_intensity', None)
        )
        
        if self.stage_config.get('tissue', False):
            np.random.seed(episode_seed + 10000)
            tissue = self.generate_tissue_noise()
            is_white_bg = np.mean([img[0,0], img[0,-1]]) > 0.5
            if is_white_bg:
                img = np.clip(img - tissue, 0.0, 1.0)
            else:
                img = np.clip(img + tissue, 0.0, 1.0)
            np.random.seed()

        self.gt_map = np.zeros_like(img)
        gt_poly = pts_all[0].astype(np.float32)
        for pt in gt_poly:
            r, c = int(pt[0]), int(pt[1])
            if 0<=r<self.h and 0<=c<self.w:
                self.gt_map[r,c] = 1.0
        
        self.ep = CurveEpisode(img=img, mask=mask, gt_poly=gt_poly)

        np.random.seed(episode_seed + 20000)
        use_cold_start = False
        if self.stage_config.get('mixed_start', False):
            use_cold_start = (np.random.rand() < 0.5)
        np.random.seed()

        if use_cold_start:
            curr = gt_poly[0]
            self.history_pos = [tuple(curr)] * 3
            self.prev_idx = 0
            self.agent = (float(curr[0]), float(curr[1]))
        else:
            start_idx = 5 if len(gt_poly) > 10 else 0
            curr = gt_poly[start_idx]
            p1 = gt_poly[max(0, start_idx-1)]
            p2 = gt_poly[max(0, start_idx-2)]
            self.history_pos = [tuple(p2), tuple(p1), tuple(curr)]
            self.prev_idx = start_idx
            self.agent = (float(curr[0]), float(curr[1]))

        self.steps = 0
        self.prev_action = -1
        self.path_mask = np.zeros_like(mask, dtype=np.float32)
        self.path_points = [self.agent]
        self.path_mask[int(self.agent[0]), int(self.agent[1])] = 1.0
        
        self.L_prev = get_distance_to_poly(self.agent, self.ep.gt_poly)
        self.current_episode += 1
        return self.obs()

    def obs(self):
        curr = self.history_pos[-1]
        p1 = self.history_pos[-2]
        p2 = self.history_pos[-3]
        
        ch0 = crop32(self.ep.img, int(curr[0]), int(curr[1]))
        ch1 = crop32(self.ep.img, int(p1[0]), int(p1[1]))
        ch2 = crop32(self.ep.img, int(p2[0]), int(p2[1]))
        ch3 = crop32(self.path_mask, int(curr[0]), int(curr[1]))
        
        actor_obs = np.stack([ch0, ch1, ch2, ch3], axis=0).astype(np.float32)
        gt_crop = crop32(self.gt_map, int(curr[0]), int(curr[1]))
        gt_obs = gt_crop[None, ...]
        
        return {"actor": actor_obs, "critic_gt": gt_obs}

    def step(self, a_idx):
        self.steps += 1
        dist_to_end = np.sqrt(
            (self.agent[0] - self.ep.gt_poly[-1][0])**2 + 
            (self.agent[1] - self.ep.gt_poly[-1][1])**2
        )

        if a_idx == ACTION_STOP_IDX:
            is_strict = self.stage_config.get('strict_stop', False)
            if dist_to_end < 5.0:
                stop_bonus = 30.0 if is_strict else 20.0
                return self.obs(), stop_bonus, True, {"reached_end": True, "stopped_correctly": True}
            else:
                return self.obs(), -2.0, False, {"reached_end": False, "stopped_correctly": False}

        dy, dx = ACTIONS_MOVEMENT[a_idx]
        ny = clamp(self.agent[0] + dy * STEP_ALPHA, 0, self.h-1)
        nx = clamp(self.agent[1] + dx * STEP_ALPHA, 0, self.w-1)
        self.agent = (ny, nx)
        
        self.history_pos.append(self.agent)
        self.path_points.append(self.agent)
        self.path_mask[int(ny), int(nx)] = 1.0

        L_t = get_distance_to_poly(self.agent, self.ep.gt_poly)
        dist_diff = abs(L_t - self.L_prev)
        best_idx = nearest_gt_index(self.agent, self.ep.gt_poly)
        progress_delta = best_idx - self.prev_idx
        
        sigma = 1.5 if self.stage_config.get('stage_id') == 1 else 1.0
        precision_score = np.exp(-(L_t**2) / (2 * sigma**2))
        
        if L_t < self.L_prev:
            r = np.log(EPSILON + dist_diff)
        else:
            r = -np.log(EPSILON + dist_diff)
        r = float(np.clip(r, -2.0, 2.0))

        if progress_delta > 0:
            r += precision_score * 2.0
        elif progress_delta <= 0:
            r -= 0.1

        if self.stage_config.get('stage_id', 1) >= 2:
            lookahead = min(best_idx + 4, len(self.ep.gt_poly) - 1)
            gt_vec = self.ep.gt_poly[lookahead] - self.ep.gt_poly[best_idx]
            act_vec = np.array([dy, dx])
            ng, na = np.linalg.norm(gt_vec), np.linalg.norm(act_vec)
            if ng > 1e-6 and na > 1e-6:
                cos_sim = np.dot(gt_vec, act_vec) / (ng * na)
                if cos_sim > 0: r += cos_sim * 0.5

        if self.prev_action != -1 and self.prev_action != a_idx:
            r -= 0.05
        self.prev_action = a_idx
        r -= 0.05

        self.L_prev = L_t
        self.prev_idx = max(self.prev_idx, best_idx)

        done = False
        reached_end = (dist_to_end < 5.0)
        
        off_track_lim = 10.0 if self.stage_config.get('stage_id', 1) == 1 else 8.0
        if L_t > off_track_lim:
            r -= 5.0
            done = True
        
        if self.steps >= self.max_steps:
            done = True

        if not self.stage_config.get('strict_stop', False):
            if reached_end:
                r += 20.0
                done = True
        else:
            if reached_end: r += 5.0

        return self.obs(), r, done, {"reached_end": reached_end, "stopped_correctly": False}

# ---------- NETWORK & PPO ----------
from src.models_deeper import AsymmetricActorCritic

def update_ppo(ppo_opt, model, buf_list, clip=0.2, epochs=4, minibatch=32):
    obs_a = torch.tensor(np.concatenate([x['obs']['actor'] for x in buf_list]), dtype=torch.float32, device=DEVICE)
    obs_c = torch.tensor(np.concatenate([x['obs']['critic_gt'] for x in buf_list]), dtype=torch.float32, device=DEVICE)
    ahist = torch.tensor(np.concatenate([x['ahist'] for x in buf_list]), dtype=torch.float32, device=DEVICE)
    act   = torch.tensor(np.concatenate([x['act'] for x in buf_list]), dtype=torch.long, device=DEVICE)
    logp  = torch.tensor(np.concatenate([x['logp'] for x in buf_list]), dtype=torch.float32, device=DEVICE)
    adv   = torch.tensor(np.concatenate([x['adv'] for x in buf_list]), dtype=torch.float32, device=DEVICE)
    ret   = torch.tensor(np.concatenate([x['ret'] for x in buf_list]), dtype=torch.float32, device=DEVICE)

    if adv.numel() > 1: adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    
    N = obs_a.shape[0]
    idxs = np.arange(N)
    for _ in range(epochs):
        np.random.shuffle(idxs)
        for s in range(0, N, minibatch):
            mb = idxs[s:s+minibatch]
            if len(mb) == 0: continue
            
            logits, val, _, _ = model(obs_a[mb], obs_c[mb], ahist[mb])
            logits = torch.clamp(logits, -20, 20)
            dist = Categorical(logits=logits)
            new_logp = dist.log_prob(act[mb])
            entropy = dist.entropy().mean()
            
            ratio = torch.exp(new_logp - logp[mb])
            surr1 = ratio * adv[mb]
            surr2 = torch.clamp(ratio, 1.0-clip, 1.0+clip) * adv[mb]
            p_loss = -torch.min(surr1, surr2).mean()
            v_loss = F.mse_loss(val, ret[mb])
            
            loss = p_loss + 0.5 * v_loss - 0.005 * entropy
            
            ppo_opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            ppo_opt.step()

# ---------- MAIN ----------
def run_unified_training(run_dir, base_seed=BASE_SEED, clean_previous=False, resume_from=None, curve_config_path=None):
    curve_config, config_path = load_curve_config(curve_config_path)
    
    # Print Config
    print("\n" + "="*60)
    print(f"FULL CONFIGURATION LOADED FROM: {config_path}")
    print("="*60)
    print(json.dumps(curve_config, indent=2))
    print("="*60 + "\n")

    img_h = curve_config.get('image', {}).get('height', 128)
    img_w = curve_config.get('image', {}).get('width', 128)

    parent_dir = os.path.dirname(_script_dir)
    runs_base = os.path.join(parent_dir, "runs")
    
    if run_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(runs_base, timestamp)
    else:
        if not os.path.isabs(run_dir):
            run_dir = os.path.join(runs_base, run_dir)
            
    if clean_previous and os.path.exists(runs_base):
        shutil.rmtree(runs_base)

    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "weights"), exist_ok=True)
    shutil.copy2(config_path, os.path.join(run_dir, "curve_config.json"))

    model = AsymmetricActorCritic(n_actions=N_ACTIONS).to(DEVICE)
    K = 16
    
    if resume_from:
        print(f"Loading checkpoint: {resume_from}")
        model.load_state_dict(torch.load(resume_from, map_location=DEVICE))

    raw_stages = curve_config.get('training_stages', [])
    if not raw_stages:
        print("❌ No 'training_stages' found in JSON. Exiting.")
        return

    stages = []
    for s in raw_stages:
        merged_config = s.get('curve_generation', {}).copy()
        merged_config.update(s.get('training', {}))
        merged_config['stage_id'] = s.get('stage_id')
        
        stages.append({
            'name': s.get('name', f"Stage{s['stage_id']}"),
            'episodes': s.get('episodes', 5000),
            'lr': s.get('learning_rate', 1e-4),
            'config': merged_config
        })

    print(f"=== Starting Training with {len(stages)} Stages ===")
    global_episode_offset = 0
    previous_stages = [] 
    
    for stage_idx, stage in enumerate(stages):
        print(f"\n⚡ STARTING {stage['name']} (Eps: {stage['episodes']}, LR: {stage['lr']})")
        
        env = CurveEnvUnified(h=img_h, w=img_w, base_seed=base_seed, 
                              stage_id=stage['config']['stage_id'], curve_config=curve_config)
        env.set_stage(stage['config'])
        
        opt = torch.optim.Adam(model.parameters(), lr=stage['lr'])
        batch_buffer = []
        ep_returns = []
        
        for ep in range(1, stage['episodes'] + 1):
            global_ep = global_episode_offset + ep
            
            is_prev = False
            if previous_stages and np.random.rand() < 0.15:
                prev_idx = np.random.randint(0, len(previous_stages))
                env.set_stage(previous_stages[prev_idx])
                is_prev = True
            else:
                env.set_stage(stage['config'])
            
            obs_dict = env.reset(episode_number=global_ep)
            done = False
            
            ahist = []
            ep_traj = {"obs":{'actor':[], 'critic_gt':[]}, "ahist":[], "act":[], "logp":[], "val":[], "rew":[]}
            
            while not done:
                obs_a = torch.tensor(obs_dict['actor'][None], dtype=torch.float32, device=DEVICE)
                obs_c = torch.tensor(obs_dict['critic_gt'][None], dtype=torch.float32, device=DEVICE)
                A = fixed_window_history(ahist, K, N_ACTIONS)[None, ...]
                A_t = torch.tensor(A, dtype=torch.float32, device=DEVICE)

                with torch.no_grad():
                    logits, value, _, _ = model(obs_a, obs_c, A_t)
                    logits = torch.clamp(logits, -20, 20)
                    dist = Categorical(logits=logits)
                    action = dist.sample().item()
                    logp = dist.log_prob(torch.tensor(action, device=DEVICE)).item()
                    val = value.item()

                next_obs, r, done, info = env.step(action)

                ep_traj["obs"]['actor'].append(obs_dict['actor'])
                ep_traj["obs"]['critic_gt'].append(obs_dict['critic_gt'])
                ep_traj["ahist"].append(A[0])
                ep_traj["act"].append(action)
                ep_traj["logp"].append(logp)
                ep_traj["val"].append(val)
                ep_traj["rew"].append(r)
                
                a_onehot = np.zeros(N_ACTIONS); a_onehot[action] = 1.0
                ahist.append(a_onehot)
                obs_dict = next_obs
            
            if len(ep_traj["rew"]) > 2:
                rews = np.array(ep_traj["rew"])
                vals = np.array(ep_traj["val"] + [0.0])
                delta = rews + 0.9 * vals[1:] - vals[:-1]
                adv = np.zeros_like(rews)
                acc = 0
                for t in reversed(range(len(rews))):
                    acc = delta[t] + 0.9 * 0.95 * acc
                    adv[t] = acc
                ret = adv + vals[:-1]
                
                batch_buffer.append({
                    "obs": {"actor": np.array(ep_traj["obs"]['actor']), "critic_gt": np.array(ep_traj["obs"]['critic_gt'])},
                    "ahist": np.array(ep_traj["ahist"]),
                    "act": np.array(ep_traj["act"]),
                    "logp": np.array(ep_traj["logp"]),
                    "adv": adv, "ret": ret
                })
                ep_returns.append(sum(rews))

            if len(batch_buffer) >= 64:
                update_ppo(opt, model, batch_buffer)
                batch_buffer = []

            if ep % 100 == 0:
                avg_r = np.mean(ep_returns[-100:])
                print(f"[{stage['name']}] Ep {ep} | AvgRew: {avg_r:.2f} {'(PrevStage)' if is_prev else ''}")

            if ep % 2000 == 0:
                p = os.path.join(run_dir, "checkpoints", f"ckpt_{stage['name']}_ep{ep}.pth")
                torch.save(model.state_dict(), p)
        
        final_path = os.path.join(run_dir, "weights", f"model_{stage['name']}_FINAL.pth")
        torch.save(model.state_dict(), final_path)
        previous_stages.append(stage['config'].copy())
        global_episode_offset += stage['episodes']

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, default=None)
    parser.add_argument("--base_seed", type=int, default=BASE_SEED)
    parser.add_argument("--curve_config", type=str, default=None)
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--clean_previous", action="store_true")
    
    args = parser.parse_args()
    run_unified_training(args.run_dir, args.base_seed, args.clean_previous, args.resume_from, args.curve_config)