import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import glob
import cv2
import numpy as np
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class JunctionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # Input: 1x33x33
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1) # Output logit
        )
    def forward(self, x):
        return self.net(x)

class PatchDataset(Dataset):
    def __init__(self, root_dir):
        self.files = []
        # Normal (0)
        for f in glob.glob(f"{root_dir}/0_normal/*.png"):
            self.files.append((f, 0.0))
        # Junction (1)
        for f in glob.glob(f"{root_dir}/1_junction/*.png"):
            self.files.append((f, 1.0))
        print(f"Loaded {len(self.files)} samples.")

    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        path, label = self.files[idx]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = img.astype(np.float32) / 255.0
        return torch.tensor(img[None], dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

def train():
    if not os.path.exists("data_junctions"):
        print("Run generate_junction_data.py first!")
        return

    ds = PatchDataset("data_junctions")
    loader = DataLoader(ds, batch_size=64, shuffle=True)
    
    model = JunctionNet().to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.BCEWithLogitsLoss()
    
    print("Training Classifier...")
    for epoch in range(5):
        total_loss = 0
        correct = 0
        total = 0
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x).squeeze(1)
            loss = crit(logits, y)
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            total_loss += loss.item()
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == y).sum().item()
            total += y.size(0)
            
        acc = correct / total
        print(f"Epoch {epoch+1}: Loss {total_loss/len(loader):.4f} | Acc {acc*100:.1f}%")
        
    torch.save(model.state_dict(), "junction_model.pth")
    print("Saved junction_model.pth")

if __name__ == "__main__":
    train()