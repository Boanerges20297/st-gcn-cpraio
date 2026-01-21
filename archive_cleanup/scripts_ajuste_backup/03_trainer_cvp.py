"""
Treina ST-GCN focalizado em CVP.
- janela histórica: 30 dias
- horizonte alvo: 7 dias
- otimiza MSE apenas na feature CVP (índice 1)
"""
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import json
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from src import config
from src.model import STGCN_Cpraio

class CrimeSeriesDataset(Dataset):
    def __init__(self, X, window_size=30, target_window=7):
        self.X = X
        self.window_size = window_size
        self.target_window = target_window
    def __len__(self):
        return len(self.X) - self.window_size - self.target_window + 1
    def __getitem__(self, idx):
        x_seq = self.X[idx: idx + self.window_size]
        target_seq = self.X[idx + self.window_size: idx + self.window_size + self.target_window]
        y_target = target_seq.mean(dim=0)  # (nodes, features)
        return x_seq, y_target

TRAIN_CONFIG = {
    'batch_size': 32,
    'epochs': 150,
    'learning_rate': 0.001,
    'weight_decay': 1e-4,
    'patience': 20
}

FEATURE_IDX = 1  # CVP


def load_dataset(path=None):
    if path is None:
        path = config.TENSOR_DIR / 'dataset_cvli_novo_criterio.pt'
    if not Path(path).exists():
        print('[X] Dataset not found:', path)
        return None, None, None
    data = torch.load(path)
    X = data['X']
    edge_index = data.get('edge_index')
    metadata = data
    print('Loaded', path, 'shape', X.shape)
    return X, edge_index, metadata


def train(dataset_path=None, model_out=None, stats_out=None):
    X, edge_index, metadata = load_dataset(dataset_path)
    if X is None:
        return

    # normalize
    mean = X.mean(dim=(0,1), keepdim=True)
    std = X.std(dim=(0,1), keepdim=True) + 1e-6
    Xn = (X - mean) / std

    # split temporal
    split_idx = int(len(Xn) * 0.8)
    train_data = Xn[:split_idx]
    val_data = Xn[split_idx:]

    ws = int(config.HyperParams.get('window_size_cvp', 30))
    tw = int(config.HyperParams.get('target_window_cvp', 7))

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

    model_path = MODELS_DIR / (model_out or 'model_cvp.pth')
    stats_path = MODELS_DIR / (stats_out or 'stats_cvp.pt')

    best_val = float('inf')
    patience = 0

    for epoch in range(TRAIN_CONFIG['epochs']):
        model.train()
        tr_loss = 0
        for x_b, y_b in tr_loader:
            optimizer.zero_grad()
            out = model(x_b, edge_index)
            # out, y_b shapes: (batch, nodes, features)
            loss = criterion(out[:,:,FEATURE_IDX], y_b[:,:,FEATURE_IDX])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_loss += loss.item()
        tr_loss /= max(1, len(tr_loader))

        # val
        model.eval()
        va_loss = 0
        with torch.no_grad():
            for x_b, y_b in va_loader:
                out = model(x_b, edge_index)
                va_loss += criterion(out[:,:,FEATURE_IDX], y_b[:,:,FEATURE_IDX]).item()
        va_loss /= max(1, len(va_loader))

        print(f'Epoch {epoch+1} | Tr {tr_loss:.6f} | Va {va_loss:.6f}')

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
    train()
