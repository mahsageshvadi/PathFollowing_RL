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
        self.imgs = sorted(glob.glob(os.path.join(root, "images", "*.png")))
        self.masks = sorted(glob.glob(os.path.join(root, "masks", "*.png")))
        assert len(self.imgs) == len(self.masks) and len(self.imgs) > 0, f"Bad dataset at {root}"

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
