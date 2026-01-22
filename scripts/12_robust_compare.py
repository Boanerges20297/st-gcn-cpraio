"""Robust comparative ST-GCN experiment:
- Train baseline and augmented (procedimentos) models for more epochs
- Compute validation RMSE per neighborhood and overall
- Save `outputs/robust_compare_summary.csv` and `outputs/robust_per_bairro_rmse.csv`
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

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
        return torch.from_numpy(self.X[idx]).float(), torch.from_numpy(self.y[idx]).float()


def run_epoch_train(model, loader, opt, crit, device):
    model.train()
    total = 0.0
    for xb, yb in loader:
        xb = xb.to(device); yb = yb.to(device)
        opt.zero_grad()
        out = model(xb)
        loss = crit(out, yb)
        loss.backward()
        opt.step()
        total += loss.item() * xb.size(0)
    return total / len(loader.dataset)


def eval_and_collect(model, loader, crit, device):
    model.eval()
    total = 0.0
    preds = []
    targs = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device); yb = yb.to(device)
            out = model(xb)
            total += crit(out, yb).item() * xb.size(0)
            preds.append(out.cpu().numpy())
            targs.append(yb.cpu().numpy())
    val_loss = total / len(loader.dataset) if len(loader.dataset) > 0 else np.nan
    preds = np.vstack(preds) if preds else np.zeros((0,))
    targs = np.vstack(targs) if targs else np.zeros((0,))
    return val_loss, preds, targs


def train_and_evaluate(X, A, epochs=10, max_samples=None):
    Xw, yt = create_temporal_windows(X, window_size=7, horizon=1)
    if max_samples:
        limit = min(len(Xw), max_samples)
        Xw = Xw[:limit]; yt = yt[:limit]

    split = int(len(Xw) * 0.8)
    train_ds = WindowDataset(Xw[:split], yt[:split])
    val_ds = WindowDataset(Xw[split:], yt[split:])
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    T, N, F = X.shape[0], X.shape[1], X.shape[2]
    model = STGCNModel(in_channels=F, hidden_channels=64, out_channels=F, A=A)
    device = torch.device('cpu')
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.MSELoss()

    best_val = float('inf')
    best_preds = None
    best_targs = None

    for ep in range(1, epochs + 1):
        tr = run_epoch_train(model, train_loader, opt, crit, device)
        val_loss, preds, targs = eval_and_collect(model, val_loader, crit, device)
        if val_loss < best_val:
            best_val = val_loss
            best_preds = preds
            best_targs = targs
        print(f"Epoch {ep}/{epochs} — train_loss: {tr:.6f} — val_loss: {val_loss:.6f}")

    # compute per-node RMSE across best_preds/best_targs
    if best_preds is None or best_targs is None:
        return best_val, None

    # preds shape: (samples, N, F)
    preds = best_preds
    targs = best_targs
    samples = preds.shape[0]
    # per-node per-feature MSE
    mse_per_node_feat = ((preds - targs) ** 2).mean(axis=0)  # (N, F)
    rmse_per_node = np.sqrt(mse_per_node_feat.mean(axis=1))  # (N,)
    overall_rmse = np.sqrt(((preds - targs) ** 2).mean())
    return overall_rmse, rmse_per_node


def main():
    tensor_path = 'data/processed/node_feature_tensor.npy'
    adj_path = 'data/processed/adjacency_matrix.npy'
    proc_path = 'outputs/procedures_daily_node_matrix.npy'

    X = np.load(tensor_path)  # (T, N, F)
    A = np.load(adj_path)
    proc = np.load(proc_path)  # (T, N)

    # normalize procedures by global 99th
    p99 = np.percentile(proc, 99) if proc.size else 1.0
    proc_norm = (proc / max(p99, 1.0)).clip(0, 1)

    # baseline
    print('Running baseline (10 epochs)...')
    base_rmse, base_per_node = train_and_evaluate(X, A, epochs=10, max_samples=None)

    # augmented
    X_aug = np.concatenate([X, proc_norm[:, :, None]], axis=2)
    print('\nRunning augmented with procedimentos (10 epochs)...')
    aug_rmse, aug_per_node = train_and_evaluate(X_aug, A, epochs=10, max_samples=None)

    # save summary
    out_dir = 'outputs'
    os.makedirs(out_dir, exist_ok=True)
    summary = {
        'baseline_overall_rmse': float(base_rmse) if base_rmse is not None else None,
        'augmented_overall_rmse': float(aug_rmse) if aug_rmse is not None else None,
        'delta': float(base_rmse - aug_rmse) if (base_rmse is not None and aug_rmse is not None) else None
    }
    pd.DataFrame([summary]).to_csv(os.path.join(out_dir, 'robust_compare_summary.csv'), index=False)

    if base_per_node is not None and aug_per_node is not None:
        df_nodes = pd.DataFrame({'bairro_id': list(range(len(base_per_node))),
                                 'rmse_baseline': base_per_node,
                                 'rmse_augmented': aug_per_node,
                                 'rmse_delta': base_per_node - aug_per_node})
        df_nodes.to_csv(os.path.join(out_dir, 'robust_per_bairro_rmse.csv'), index=False)

    print('\nSummary saved to outputs/robust_compare_summary.csv and outputs/robust_per_bairro_rmse.csv')
    print('Overall: baseline_rmse=%.6f, augmented_rmse=%.6f' % (base_rmse, aug_rmse))


if __name__ == '__main__':
    main()
