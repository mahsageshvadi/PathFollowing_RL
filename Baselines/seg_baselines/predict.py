import argparse, os
import numpy as np
import torch
import cv2
from torch.utils.data import DataLoader

from datasets import SegDataset
from models import UNet, UNetPP

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", required=True)
    ap.add_argument("--model", choices=["unet","unetpp"], default="unet")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_masks", required=True)
    ap.add_argument("--resize", type=int, nargs=2, default=[512,512])
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_masks, exist_ok=True)

    ds = SegDataset(args.dataset_root, resize=tuple(args.resize))
    dl = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=2)

    if args.model == "unet":
        net = UNet(in_channels=1, out_channels=1, base=64)
    else:
        net = UNetPP(in_channels=1, out_channels=1, base=32)

    net.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    net.to(device).eval()

    with torch.no_grad():
        for imgs, _, names in dl:
            imgs = imgs.to(device)
            logits = net(imgs)
            prob = torch.sigmoid(logits).cpu().numpy()[:,0]
            pred = (prob > 0.5).astype(np.uint8) * 255

            for i, nm in enumerate(names):
                cv2.imwrite(os.path.join(args.out_masks, nm), pred[i])

    print("Saved masks to:", args.out_masks)

if __name__ == "__main__":
    main()
