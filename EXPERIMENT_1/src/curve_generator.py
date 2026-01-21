
import numpy as np
import scipy.ndimage


def _rng(seed=None):
    return np.random.default_rng(seed)


def _cubic_bezier(p0, p1, p2, p3, t):
    omt = 1.0 - t
    return (omt**3)*p0 + 3*omt*omt*t*p1 + 3*omt*t*t*p2 + (t**3)*p3


class CurveMakerLengthSafe:
    """
    Curve generator that is:
    - Geometry safe
    - Stroke safe
    - LENGTH safe (arc-length truncated)
    """

    def __init__(self, h=128, w=128, seed=None):
        self.h = h
        self.w = w
        self.rng = _rng(seed)

        # ===== Absolute safety =====
        self.margin = 22
        self.max_thickness = 10

        # ===== Bezier =====
        self.n_samples = 1200
        self.min_end_dist = 50
        self.base_spread = 0.35
        self.ctrl_factor = 0.6

        # ===== Length control =====
        self.max_arc_fraction = 0.80  # % of image diagonal

    # --------------------------------------------------
    def _effective_margin(self):
        return self.margin + self.max_thickness

    # --------------------------------------------------
    def _random_point(self):
        m = self._effective_margin()
        # Ensure we don't swap bounds if margin is too large for image size
        y_max = max(m + 1, self.h - m)
        x_max = max(m + 1, self.w - m)
        
        y = self.rng.integers(m, y_max)
        x = self.rng.integers(m, x_max)
        return np.array([y, x], dtype=np.float32)

    # --------------------------------------------------
    def _inside(self, p):
        m = self._effective_margin()
        return (
            m <= p[0] < self.h - m and
            m <= p[1] < self.w - m
        )

    # --------------------------------------------------
    def _clip_points_to_bounds(self, pts):
        """Clip all points to stay within image bounds with margin."""
        m = self._effective_margin()
        pts[:, 0] = np.clip(pts[:, 0], m, self.h - m - 1)
        pts[:, 1] = np.clip(pts[:, 1], m, self.w - m - 1)
        return pts

    # --------------------------------------------------
    def _filter_points_inside(self, pts):
        """Filter out points that are outside bounds, keeping only valid segments."""
        m = self._effective_margin()
        valid = (
            (pts[:, 0] >= m) & (pts[:, 0] < self.h - m) &
            (pts[:, 1] >= m) & (pts[:, 1] < self.w - m)
        )
        return pts[valid]

    # --------------------------------------------------
    def _generate_bezier_points(self, p0=None, n_samples=None, curvature_factor=1.0,
                                allow_self_cross=False, self_cross_prob=0.0):
        """Generates a curve centered in the image. All points guaranteed to stay within bounds."""
        n_samples = n_samples if n_samples is not None else self.n_samples
        m = self._effective_margin()
        
        # Image Center
        cy, cx = self.h / 2.0, self.w / 2.0
        center_img = np.array([cy, cx], dtype=np.float32)

        # Maximum radius to stay within margins
        max_radius = min(self.h - 2*m, self.w - 2*m) / 2.0
        max_radius = max(10.0, max_radius)  # Ensure minimum radius

        # Use rejection sampling to ensure curve stays within bounds
        for attempt in range(50):
            # --- 1. GENERATE P0 and P3 AROUND CENTER ---
            if p0 is None:
                # Random angle for Start Point
                theta1 = self.rng.uniform(0, 2 * np.pi)
                # Random distance from center
                r1 = self.rng.uniform(max_radius * 0.3, max_radius * 0.8)
                p0 = center_img + np.array([r1 * np.sin(theta1), r1 * np.cos(theta1)], dtype=np.float32)
                p0 = np.clip(p0, m, np.array([self.h - m, self.w - m], dtype=np.float32))
            else:
                # Calculate angle from center to p0
                offset = p0 - center_img
                theta1 = np.arctan2(offset[0], offset[1])

            # Generate P3 roughly opposite to P0
            theta2 = theta1 + np.pi + self.rng.uniform(-0.6, 0.6)  # +/- jitter
            r2 = self.rng.uniform(max_radius * 0.3, max_radius * 0.8)
            p3 = center_img + np.array([r2 * np.sin(theta2), r2 * np.cos(theta2)], dtype=np.float32)
            p3 = np.clip(p3, m, np.array([self.h - m, self.w - m], dtype=np.float32))

            # Check minimum distance
            if np.linalg.norm(p0 - p3) < self.min_end_dist:
                continue

            # --- 2. CALCULATE CONTROL POINTS (P1, P2) ---
            center_curve = (p0 + p3) / 2.0
            
            # Limit spread based on available space to prevent curves from going outside
            max_spread_y = min(center_curve[0] - m, self.h - m - center_curve[0])
            max_spread_x = min(center_curve[1] - m, self.w - m - center_curve[1])
            max_spread = np.array([max_spread_y, max_spread_x], dtype=np.float32)
            
            # Calculate desired spread
            desired_spread = np.array([self.h, self.w], dtype=np.float32) * self.base_spread * curvature_factor
            # Use the smaller of desired spread and available space
            spread = np.minimum(desired_spread, max_spread * 0.7)  # 70% of max to be safe

            if allow_self_cross and self.rng.random() < self_cross_prob:
                dir_vec = self.rng.normal(0, 1, 2)
                norm = np.linalg.norm(dir_vec) + 1e-8
                dir_unit = dir_vec / norm
                p1 = center_curve + dir_unit * spread * self.ctrl_factor
                p2 = center_curve - dir_unit * spread * self.ctrl_factor
            else:
                p1 = center_curve + self.rng.normal(0, 1, 2) * spread * self.ctrl_factor
                p2 = center_curve + self.rng.normal(0, 1, 2) * spread * self.ctrl_factor
            
            # --- 3. CLAMP CONTROL POINTS ---
            p1 = np.clip(p1, m, np.array([self.h - m, self.w - m], dtype=np.float32))
            p2 = np.clip(p2, m, np.array([self.h - m, self.w - m], dtype=np.float32))

            # --- 4. GENERATE POINTS ---
            ts = np.linspace(0, 1, n_samples, dtype=np.float32)
            pts = np.stack([_cubic_bezier(p0, p1, p2, p3, t) for t in ts], axis=0)
            
            # --- 5. CLIP ALL POINTS TO BOUNDS ---
            # This is critical: bezier curves can extend beyond control points
            pts = self._clip_points_to_bounds(pts)
            
            # --- 6. VERIFY ALL POINTS ARE WITHIN BOUNDS ---
            if np.all(pts[:, 0] >= m) and np.all(pts[:, 0] < self.h - m) and \
               np.all(pts[:, 1] >= m) and np.all(pts[:, 1] < self.w - m):
                return pts
        
        # Fallback: if rejection sampling fails, generate a simple centered line
        # and clip it
        p0 = center_img - np.array([max_radius * 0.5, 0], dtype=np.float32)
        p3 = center_img + np.array([max_radius * 0.5, 0], dtype=np.float32)
        p0 = np.clip(p0, m, np.array([self.h - m, self.w - m], dtype=np.float32))
        p3 = np.clip(p3, m, np.array([self.h - m, self.w - m], dtype=np.float32))
        pts = np.linspace(p0, p3, n_samples)
        return self._clip_points_to_bounds(pts)

    # --------------------------------------------------
    def _truncate_by_length(self, pts):
        """
        Stop drawing once arc length exceeds safe limit
        """
        diffs = np.diff(pts, axis=0)
        seg_lens = np.linalg.norm(diffs, axis=1)
        cum_len = np.cumsum(seg_lens)

        max_len = self.max_arc_fraction * np.sqrt(self.h**2 + self.w**2)
        idx = np.searchsorted(cum_len, max_len)

        return pts[:idx + 1]

    # --------------------------------------------------
    def _draw_safe(self, img, pts, thickness, intensity):
        r = max(1, thickness // 2)

        for p in pts:
            y, x = int(round(p[0])), int(round(p[1]))

            y0 = max(0, y - r)
            y1 = min(self.h, y + r + 1)
            x0 = max(0, x - r)
            x1 = min(self.w, x + r + 1)

            yy, xx = np.ogrid[y0:y1, x0:x1]
            mask = (yy - y)**2 + (xx - x)**2 <= r*r

            img[y0:y1, x0:x1][mask] = np.maximum(
                img[y0:y1, x0:x1][mask], intensity
            )

    # --------------------------------------------------
    def sample(self, thickness_range=(2, 5), curvature=1.0, noise=False, 
               allow_self_cross=False, self_cross_prob=0.0):

        img = np.zeros((self.h, self.w), dtype=np.float32)
        mask = np.zeros_like(img)

        thickness = int(self.rng.integers(*thickness_range))
        intensity = self.rng.uniform(0.4, 0.8)

        # Generate curve with bounds checking
        pts = self._generate_bezier_points(
            curvature_factor=curvature,
            allow_self_cross=allow_self_cross,
            self_cross_prob=self_cross_prob
        )
        
        # Truncate by length (after ensuring bounds)
        pts = self._truncate_by_length(pts)
        
        # Final safety check: ensure all points are still within bounds
        pts = self._clip_points_to_bounds(pts)

        self._draw_safe(img, pts, thickness, intensity)
        self._draw_safe(mask, pts, thickness, 1.0)

        if noise:
            img += self.rng.normal(0, 0.02, img.shape)
            img = scipy.ndimage.gaussian_filter(img, sigma=0.6)

        img = np.clip(img, 0.0, 1.0)
        mask = (mask > 0).astype(np.uint8)

        return img, mask, [pts]  # Return as list for compatibility