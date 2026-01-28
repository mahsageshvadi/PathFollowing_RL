import argparse, os, random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from datasets import SegDataset
from models import UNet, UNetPP

def set_seed(s=0):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def bce_dice_loss(logits, target):
    bce = F.binary_cross_entropy_with_logits(logits, target)
    prob = torch.sigmoid(logits)
    inter = (prob * target).sum(dim=(1,2,3))
    den = prob.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3)) + 1e-6
    d = 1 - (2*inter / den).mean()
    return bce + d

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", required=True)
    ap.add_argument("--model", choices=["unet","unetpp"], default="unet")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--resize", type=int, nargs=2, default=[512,512])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.outdir, exist_ok=True)

    # Verify dataset before training
    print(f"Loading dataset from: {args.dataset_root}")
    try:
        ds = SegDataset(args.dataset_root, resize=tuple(args.resize))
        print(f"✓ Dataset loaded: {len(ds)} samples")
    except (FileNotFoundError, ValueError, AssertionError) as e:
        print(f"ERROR: Dataset verification failed!")
        print(f"  Dataset root: {args.dataset_root}")
        print(f"  Error: {e}")
        print("\nPlease ensure:")
        print(f"  1. Dataset directory exists: {args.dataset_root}")
        print(f"  2. Contains 'images/' and 'masks/' subdirectories")
        print(f"  3. Both contain matching PNG files")
        print(f"\nTo prepare DRIVE dataset, run:")
        print(f"  python prepare_drive_dataset.py")
        raise
    n = len(ds)
    n_val = max(1, int(0.2*n))
    n_train = n - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed))

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=2)
    val_loader   = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=2)

    if args.model == "unet":
        net = UNet(in_channels=1, out_channels=1, base=64)
    else:
        net = UNetPP(in_channels=1, out_channels=1, base=32)

    net.to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)

    best = 1e9
    for ep in range(args.epochs):
        net.train()
        tr_loss = 0.0
        for img, msk, _ in train_loader:
            img, msk = img.to(device), msk.to(device)
            logits = net(img)
            loss = bce_dice_loss(logits, msk)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tr_loss += loss.item()

        net.eval()
        va_loss = 0.0
        with torch.no_grad():
            for img, msk, _ in val_loader:
                img, msk = img.to(device), msk.to(device)
                logits = net(img)
                va_loss += bce_dice_loss(logits, msk).item()

        tr_loss /= max(1, len(train_loader))
        va_loss /= max(1, len(val_loader))
        print(f"[{args.model}] ep={ep:03d} train={tr_loss:.4f} val={va_loss:.4f}")

        if va_loss < best:
            best = va_loss
            torch.save(net.state_dict(), os.path.join(args.outdir, "best.pth"))

    print("Saved:", os.path.join(args.outdir, "best.pth"))

if __name__ == "__main__":
    main()

