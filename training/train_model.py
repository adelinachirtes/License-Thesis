import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from models.shot_classifier import ShotClassifier
import os

class ShotTrajectoryDataset(Dataset):
    def __init__(self, path="../data/trajectories.json", max_len=50):
        with open(path) as f:
            self.samples = [s for s in json.load(f) if s['label'] is not None]
        self.max_len = max_len

    def pad_sequence(self, seq):
        seq = torch.tensor(seq, dtype=torch.float32)
        if len(seq) < self.max_len:
            padding = torch.zeros((self.max_len - len(seq), 2))
            seq = torch.cat((seq, padding), dim=0)
        else:
            seq = seq[:self.max_len]
        return seq

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        traj = self.pad_sequence(self.samples[idx]['trajectory'])
        label = torch.tensor([self.samples[idx]['label']], dtype=torch.float32)
        return traj, label

def train():
    dataset = ShotTrajectoryDataset()
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = ShotClassifier()
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(20):
        model.train()
        epoch_loss = 0
        correct = 0
        total = 0

        for x, y in loader:
            preds = model(x).squeeze()
            loss = criterion(preds, y.squeeze())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pred_class = (preds > 0.5).float()
            correct += (pred_class == y).sum().item()
            total += y.size(0)

        acc = correct / total
        print(f"Epoch {epoch+1}: Loss={epoch_loss:.4f} | Acc={acc:.4f}")

    torch.save(model.state_dict(), "../models/shot_classifier.pt")
    print("[✓] Model salvat în models/shot_classifier.pt")

if __name__ == "__main__":
    train()
