import argparse, os, glob
import numpy as np
import cv2
import pandas as pd

from metrics import dice, iou, cldice, n_components, largest_component_ratio

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt_root", required=True, help="dataset_root containing masks/")
    ap.add_argument("--pred_dir", required=True, help="folder with predicted png masks")
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    gt_paths = sorted(glob.glob(os.path.join(args.gt_root, "masks", "*.png")))
    rows = []

    for gt_path in gt_paths:
        name = os.path.basename(gt_path)
        pred_path = os.path.join(args.pred_dir, name)
        if not os.path.exists(pred_path):
            continue

        gt = cv2.imread(gt_path, 0) > 127
        pr = cv2.imread(pred_path, 0) > 127

        rows.append({
            "name": name,
            "dice": dice(pr, gt),
            "iou": iou(pr, gt),
            "cldice": cldice(pr, gt),
            "pred_components": n_components(pr),
            "gt_components": n_components(gt),
            "largest_comp_ratio": largest_component_ratio(pr),
        })

    df = pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False)

    mean = df.mean(numeric_only=True)
    print("Saved:", args.out_csv)
    print("\nMEAN METRICS:")
    print(mean)

if __name__ == "__main__":
    main()
