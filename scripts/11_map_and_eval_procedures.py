"""Map raw procedures to `bairro_id`, build daily node series, augment tensor,
and run a quick comparative ST-GCN smoke-test (with and without procedures).

Outputs:
 - outputs/procedures_daily_node_matrix.npy
 - printed baseline vs augmented validation losses
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.models.stgcn import STGCNModel
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


def load_mapping_cache(report_path):
    with open(report_path, 'r', encoding='utf-8') as f:
        rep = json.load(f)
    return rep.get('mapping_cache', {})


def load_neighborhood_map(params_path):
    with open(params_path, 'r', encoding='utf-8') as f:
        p = json.load(f)
    return p['neighborhood_mapping'], p['date_range']


def load_raw_ops(path):
    with open(path, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    for item in raw:
        if isinstance(item, dict) and item.get('type') == 'table' and 'data' in item:
            return pd.DataFrame(item['data'])
    return pd.DataFrame(raw)


def build_daily_node_matrix(df_ops, mapping_cache, neighborhood_map, date_range):
    # Map raw BairroOcor -> official name using mapping_cache
    def map_raw(raw):
        if raw in mapping_cache and mapping_cache[raw]:
            return mapping_cache[raw]['matched']
        # best-effort uppercase
        try:
            return str(raw).upper()
        except Exception:
            return None

    df = df_ops.copy()
    df['BairroOcor_mapped'] = df['BairroOcor'].apply(map_raw)

    # Parse date
    df['Data'] = pd.to_datetime(df['Data'], errors='coerce')
    df = df[df['Data'].notna()]

    # Map to bairro_id
    neighborhood_map_upper = {k.upper(): v for k, v in neighborhood_map.items()}
    df['bairro_id'] = df['BairroOcor_mapped'].map(lambda x: neighborhood_map_upper.get(x))

    # Keep only mapped and Fortaleza entries (optional)
    # Many operations include other cities; include only bairro_id not null
    df = df[df['bairro_id'].notna()].copy()
    df['bairro_id'] = df['bairro_id'].astype(int)

    # build date index
    start = pd.to_datetime(date_range['start']).normalize()
    end = pd.to_datetime(date_range['end']).normalize()
    dates = pd.date_range(start, end, freq='D')

    num_days = len(dates)
    num_nodes = max(neighborhood_map.values()) + 1

    mat = np.zeros((num_days, num_nodes), dtype=float)

    # count procedimentos per day per bairro
    grp = df.groupby([df['Data'].dt.normalize(), 'bairro_id']).size().reset_index(name='procedimentos')
    date_to_idx = {d: i for i, d in enumerate(dates)}
    for _, row in grp.iterrows():
        d = pd.to_datetime(row['Data']).normalize()
        if d in date_to_idx:
            i = date_to_idx[d]
            j = int(row['bairro_id'])
            mat[i, j] = row['procedimentos']

    return dates, mat


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


def run_train(X, A, epochs=3, max_samples=128):
    X_windows, y_targets = create_temporal_windows(X, window_size=7, horizon=1)
    limit = min(len(X_windows), max_samples)
    X_windows = X_windows[:limit]
    y_targets = y_targets[:limit]

    split = int(len(X_windows) * 0.8)
    train_ds = WindowDataset(X_windows[:split], y_targets[:split])
    val_ds = WindowDataset(X_windows[split:], y_targets[split:])

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

    T, N, F = X.shape[0], X.shape[1], X.shape[2]
    model = STGCNModel(in_channels=F, hidden_channels=64, out_channels=F, A=A)
    device = torch.device('cpu')
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.MSELoss()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            out = model(xb)
            loss = crit(out, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * xb.size(0)
        avg = total_loss / len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                out = model(xb)
                val_loss += crit(out, yb).item() * xb.size(0)
        val = val_loss / len(val_loader.dataset) if len(val_loader.dataset) > 0 else 0.0
        print(f"Epoch {epoch}/{epochs} — train_loss: {avg:.6f} — val_loss: {val:.6f}")

    return val


def main():
    report_path = 'outputs/neighborhood_mapping_report.json'
    params_path = 'data/processed/normalization_params_deduplicated.json'
    raw_path = 'data/raw/ocorrencia_policial_operacional.json'
    tensor_path = 'data/processed/node_feature_tensor.npy'
    adj_path = 'data/processed/adjacency_matrix.npy'

    print('Loading resources...')
    mapping_cache = load_mapping_cache(report_path)
    neighborhood_map, date_range = load_neighborhood_map(params_path)
    df_ops = load_raw_ops(raw_path)

    dates, mat = build_daily_node_matrix(df_ops, mapping_cache, neighborhood_map, date_range)
    np.save('outputs/procedures_daily_node_matrix.npy', mat)
    print('Saved outputs/procedures_daily_node_matrix.npy:', mat.shape)

    # Load original tensor and adjacency
    X = np.load(tensor_path)  # (T, N, F)
    A = np.load(adj_path)
    print('Loaded original X shape:', X.shape)

    # normalize procedures to [0,1] using same percentile used earlier (drogas/armas/dinheiro used 99th pct in params)
    # here we simply scale by 99th percentile per-node to avoid outliers
    p99 = np.percentile(mat, 99)
    scale = p99 if p99 > 0 else 1.0
    mat_norm = (mat / scale).clip(0, 1)

    # append as a new feature channel
    X_aug = np.concatenate([X, mat_norm[:, :, None]], axis=2)
    print('Augmented X shape:', X_aug.shape)

    # Quick comparative training
    print('\nRunning baseline (original features) short train:')
    val_base = run_train(X, A, epochs=3, max_samples=256)

    print('\nRunning augmented (with procedimentos) short train:')
    val_aug = run_train(X_aug, A, epochs=3, max_samples=256)

    print('\nSummary:')
    print(f'  Baseline val_loss: {val_base:.6f}')
    print(f'  Augmented val_loss: {val_aug:.6f}')

    if val_aug < val_base:
        print('\nConclusion: adicionar `procedimentos` reduziu a perda de validação (melhora).')
    else:
        print('\nConclusão: adicionar `procedimentos` não melhorou a perda de validação no teste rápido; recomenda-se testar com mais epochs, validação por bairro e features adicionais.')


if __name__ == '__main__':
    main()
