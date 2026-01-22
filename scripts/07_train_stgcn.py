"""Quick ST-GCN training script (smoke-test).

Usage:
    python scripts/07_train_stgcn.py --epochs 2
"""
import argparse
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ensure project root is on sys.path so `import src.*` works
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.models.stgcn import STGCNModel


def create_temporal_windows(X, window_size=7, horizon=1):
    windows = []
    targets = []
    T = X.shape[0]
    for t in range(T - window_size - horizon + 1):
        windows.append(X[t:t+window_size])
        targets.append(X[t+window_size+horizon-1])
    return np.array(windows), np.array(targets)


class WindowDataset(Dataset):
    def __init__(self, X_windows, y_targets):
        self.X = X_windows
        self.y = y_targets

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # return (T, N, F), (N, F)
        return torch.from_numpy(self.X[idx]).float(), torch.from_numpy(self.y[idx]).float()


def main(args):
    device = torch.device('cpu')
    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')

    data_path = os.path.join('data', 'processed', 'node_feature_tensor.npy')
    adj_path = os.path.join('data', 'processed', 'adjacency_matrix.npy')

    print('Loading data...')
    X = np.load(data_path)  # (T, N, F)
    A = np.load(adj_path)   # (N, N)

    print(f"Loaded X shape: {X.shape}, A shape: {A.shape}")

    X_windows, y_targets = create_temporal_windows(X, window_size=args.window_size, horizon=1)

    # limit for smoke-test
    limit = min(len(X_windows), args.max_samples)
    X_windows = X_windows[:limit]
    y_targets = y_targets[:limit]

    # temporal -> (samples, T, N, F); targets -> (samples, N, F)
    split = int(len(X_windows) * 0.8)
    train_ds = WindowDataset(X_windows[:split], y_targets[:split])
    val_ds = WindowDataset(X_windows[split:], y_targets[split:])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    T, N, F = X.shape[0], X.shape[1], X.shape[2]
    in_channels = F
    out_channels = F

    model = STGCNModel(in_channels=in_channels, hidden_channels=args.hidden, out_channels=out_channels, A=A)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    print('Starting training (smoke-test)...')
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)  # (B, T, N, F)
            yb = yb.to(device)  # (B, N, F)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        avg_loss = total_loss / len(train_loader.dataset)

        # validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                out = model(xb)
                loss = criterion(out, yb)
                val_loss += loss.item() * xb.size(0)
        val_loss = val_loss / len(val_loader.dataset) if len(val_loader.dataset) > 0 else 0.0

        print(f"Epoch {epoch}/{args.epochs} — train_loss: {avg_loss:.6f} — val_loss: {val_loss:.6f}")

    print('Smoke-test complete.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--window-size', type=int, default=7)
    parser.add_argument('--max-samples', type=int, default=128)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    args = parser.parse_args()
    main(args)
