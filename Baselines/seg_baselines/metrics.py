import numpy as np
from skimage.morphology import skeletonize, binary_closing, binary_opening
from skimage.measure import label
from scipy.ndimage import binary_fill_holes

def dice(pred, gt, eps=1e-6):
    pred = pred.astype(bool); gt = gt.astype(bool)
    inter = (pred & gt).sum()
    return (2*inter + eps) / (pred.sum() + gt.sum() + eps)

def iou(pred, gt, eps=1e-6):
    pred = pred.astype(bool); gt = gt.astype(bool)
    inter = (pred & gt).sum()
    union = (pred | gt).sum()
    return (inter + eps) / (union + eps)

def cldice(pred, gt, eps=1e-6):
    # Standard clDice: overlap of skeletons
    pred = pred.astype(bool); gt = gt.astype(bool)
    sk_pred = skeletonize(pred)
    sk_gt   = skeletonize(gt)

    tprec = (sk_pred & gt).sum() / (sk_pred.sum() + eps)   # skeleton precision
    tsens = (sk_gt & pred).sum() / (sk_gt.sum() + eps)     # skeleton sensitivity
    return (2*tprec*tsens + eps) / (tprec + tsens + eps)

def n_components(mask):
    """Count number of connected components (β₀)."""
    return label(mask.astype(bool), connectivity=2).max()

def count_holes(mask):
    """
    Count number of holes (1D loops, β₁).
    A hole is a background region completely surrounded by foreground.
    
    Method: Fill holes, then count background components that were filled.
    """
    mask_bool = mask.astype(bool)
    if mask_bool.sum() == 0:
        return 0
    
    # Create a padded version to handle edge cases
    padded = np.pad(mask_bool, 1, mode='constant', constant_values=False)
    
    # Fill holes
    filled = binary_fill_holes(padded)
    
    # Remove padding
    filled = filled[1:-1, 1:-1]
    
    # Holes are the filled regions that weren't in the original
    holes_mask = filled & (~mask_bool)
    
    if holes_mask.sum() == 0:
        return 0
    
    # Count connected components in holes
    # Each hole is a separate background component surrounded by foreground
    holes_labeled = label(holes_mask, connectivity=2)
    n_holes = holes_labeled.max()
    
    return n_holes

def betti_number_error(pred, gt):
    """
    Betti Number Error (BNE).
    Measures topological difference:
    - β₀: number of connected components
    - β₁: number of holes/loops
    
    BNE = |β₀_pred - β₀_gt| + |β₁_pred - β₁_gt|
    Lower is better.
    """
    pred_bool = pred.astype(bool)
    gt_bool = gt.astype(bool)
    
    # β₀: connected components
    beta0_pred = n_components(pred_bool)
    beta0_gt = n_components(gt_bool)
    
    # β₁: holes/loops
    beta1_pred = count_holes(pred_bool)
    beta1_gt = count_holes(gt_bool)
    
    bne = abs(beta0_pred - beta0_gt) + abs(beta1_pred - beta1_gt)
    return bne

def largest_component_ratio(mask, eps=1e-6):
    lab = label(mask.astype(bool), connectivity=2)
    if lab.max() == 0:
        return 0.0
    sizes = [(lab == i).sum() for i in range(1, lab.max()+1)]
    return max(sizes) / (sum(sizes) + eps)
