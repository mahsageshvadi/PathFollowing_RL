import numpy as np
from skimage.morphology import skeletonize
from skimage.measure import label

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
    return label(mask.astype(bool), connectivity=2).max()

def largest_component_ratio(mask, eps=1e-6):
    lab = label(mask.astype(bool), connectivity=2)
    if lab.max() == 0:
        return 0.0
    sizes = [(lab == i).sum() for i in range(1, lab.max()+1)]
    return max(sizes) / (sum(sizes) + eps)
