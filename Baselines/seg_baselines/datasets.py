import glob, os, cv2
import numpy as np
import torch
from torch.utils.data import Dataset

def _read_gray(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img

class SegDataset(Dataset):
    def __init__(self, root, resize=None):
        self.root = root
        self.resize = resize
        img_dir = os.path.join(root, "images")
        mask_dir = os.path.join(root, "masks")
        
        self.imgs = sorted(glob.glob(os.path.join(img_dir, "*.png")))
        self.masks = sorted(glob.glob(os.path.join(mask_dir, "*.png")))
        
        # Better error messages
        if not os.path.exists(img_dir):
            raise FileNotFoundError(f"Images directory not found: {img_dir}")
        if not os.path.exists(mask_dir):
            raise FileNotFoundError(f"Masks directory not found: {mask_dir}")
        if len(self.imgs) == 0:
            raise ValueError(f"No PNG images found in {img_dir}")
        if len(self.masks) == 0:
            raise ValueError(f"No PNG masks found in {mask_dir}")
        if len(self.imgs) != len(self.masks):
            img_names = [os.path.basename(f) for f in self.imgs]
            mask_names = [os.path.basename(f) for f in self.masks]
            missing_in_masks = set(img_names) - set(mask_names)
            missing_in_imgs = set(mask_names) - set(img_names)
            error_msg = (
                f"Mismatch: {len(self.imgs)} images vs {len(self.masks)} masks at {root}\n"
                f"  Images directory: {img_dir}\n"
                f"  Masks directory: {mask_dir}\n"
            )
            if missing_in_masks:
                error_msg += f"  Missing masks for images: {sorted(missing_in_masks)[:5]}\n"
            if missing_in_imgs:
                error_msg += f"  Missing images for masks: {sorted(missing_in_imgs)[:5]}\n"
            raise AssertionError(error_msg)

    def __len__(self): return len(self.imgs)

    def __getitem__(self, idx):
        img = _read_gray(self.imgs[idx]).astype(np.float32) / 255.0
        msk = _read_gray(self.masks[idx]).astype(np.float32) / 255.0
        msk = (msk > 0.5).astype(np.float32)

        if self.resize is not None:
            h, w = self.resize
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
            msk = cv2.resize(msk, (w, h), interpolation=cv2.INTER_NEAREST)

        img = torch.from_numpy(img).unsqueeze(0)  # (1,H,W)
        msk = torch.from_numpy(msk).unsqueeze(0)
        return img, msk, os.path.basename(self.imgs[idx])
