"""
Treina ST-GCN por região (CVLI-centric) usando datasets gerados em
`data/tensors/dataset_<region>.pt`.
Uso: python scripts_ajuste/03_trainer_region.py CAPITAL
"""
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import json

repo = Path(__file__).parent.parent
sys.path.insert(0, str(repo))
from src import config
from src.model import STGCN_Cpraio

class CrimeSeriesDataset(Dataset):
    def __init__(self, X, window_size=180, target_window=30):
        self.X = X
        self.window_size = window_size
        self.target_window = target_window
    def __len__(self):
        return len(self.X) - self.window_size - self.target_window + 1
    def __getitem__(self, idx):
        x_seq = self.X[idx: idx + self.window_size]
        target_seq = self.X[idx + self.window_size: idx + self.window_size + self.target_window]
        y_target = target_seq.mean(dim=0)
        return x_seq, y_target

TRAIN_CONFIG = {
    'batch_size': int(config.HyperParams.get('batch_size', 32)),
    'epochs': int(config.HyperParams.get('epochs', 150)),
    'learning_rate': float(config.HyperParams.get('learning_rate', 0.001)),
    'weight_decay': float(config.HyperParams.get('weight_decay', 1e-4)),
    'patience': int(config.HyperParams.get('patience', 20)) if 'patience' in config.HyperParams else 20
}

def load_dataset_for_region(region):
    path = config.TENSOR_DIR / f"dataset_{region.lower()}.pt"
    if not path.exists():
        print('[X] Dataset not found for region:', region, path)
        return None, None, None
    data = torch.load(path)
    return data['X'], data.get('edge_index'), data


def train_region(region):
    print(f"\n=== TRAINING REGION: {region} ===")
    X, edge_index, meta = load_dataset_for_region(region)
    if X is None:
        return

    # normalize
    mean = X.mean(dim=(0,1), keepdim=True)
    std = X.std(dim=(0,1), keepdim=True) + 1e-6
    Xn = (X - mean) / std

    split_idx = int(len(Xn) * 0.8)
    train_data = Xn[:split_idx]
    val_data = Xn[split_idx:]

    ws = int(config.HyperParams.get('window_size_cvli', 180))
    tw = int(config.HyperParams.get('target_window_cvli', 30))

    tr_ds = CrimeSeriesDataset(train_data, ws, tw)
    va_ds = CrimeSeriesDataset(val_data, ws, tw)
    if len(va_ds) == 0:
        va_ds = tr_ds

    tr_loader = DataLoader(tr_ds, batch_size=TRAIN_CONFIG['batch_size'], shuffle=True)
    va_loader = DataLoader(va_ds, batch_size=TRAIN_CONFIG['batch_size'])

    num_nodes = X.shape[1]
    num_features = X.shape[2]

    model = STGCN_Cpraio(num_nodes=num_nodes, in_channels=num_features, hidden_channels=config.HyperParams.get('hidden_dim',32), out_channels=num_features, dropout=config.HyperParams.get('dropout',0.3))

    optimizer = optim.Adam(model.parameters(), lr=TRAIN_CONFIG['learning_rate'], weight_decay=TRAIN_CONFIG['weight_decay'])
    criterion = nn.MSELoss()

    MODELS_DIR = config.MODEL_DIR
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODELS_DIR / f"model_{region.lower()}_cvli.pth"
    stats_path = MODELS_DIR / f"stats_{region.lower()}_cvli.pt"

    best_val = float('inf')
    patience = 0

    for epoch in range(TRAIN_CONFIG['epochs']):
        model.train()
        tr_loss = 0
        for x_b, y_b in tr_loader:
            optimizer.zero_grad()
            out = model(x_b, edge_index)
            loss = criterion(out, y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_loss += loss.item()
        tr_loss /= max(1, len(tr_loader))

        model.eval()
        va_loss = 0
        with torch.no_grad():
            for x_b, y_b in va_loader:
                out = model(x_b, edge_index)
                va_loss += criterion(out, y_b).item()
        va_loss /= max(1, len(va_loader))

        print(f"Epoch {epoch+1} | Tr {tr_loss:.6f} | Va {va_loss:.6f}")

        if va_loss < best_val:
            best_val = va_loss
            patience = 0
            torch.save(model.state_dict(), model_path)
            torch.save({'mean': mean, 'std': std}, stats_path)
        else:
            patience += 1
            if patience >= TRAIN_CONFIG['patience']:
                print('Early stopping')
                break

    print('Saved model:', model_path)
    print('Saved stats:', stats_path)

if __name__ == '__main__':
    regions = sys.argv[1:] or ['CAPITAL']
    for r in regions:
        train_region(r)
