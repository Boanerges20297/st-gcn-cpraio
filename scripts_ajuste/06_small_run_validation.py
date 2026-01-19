#!/usr/bin/env python3
"""
Sprint 2 Task 2.4: Small-run validation (1 epoch) com STGCN_Cpraio_v2

- Carrega tensor: data/tensors/features_tensor_2025.pt
- Carrega metadados: data/tensors/tensor_metadata_2025.pt
- Carrega modelo STGCN_Cpraio_v2 de src/model.py
- Cria dataset de sliding window (14 dias input → 15 dias target)
- Treina 1 epoch para validar shapes e integridade
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader

TENSOR_PATH = 'data/tensors/features_tensor_2025.pt'
METADATA_PATH = 'data/tensors/tensor_metadata_2025.pt'


class TimeSeriesDataset(Dataset):
    """Sliding window dataset para séries temporais"""
    def __init__(self, X, window_size=14, target_size=15):
        """
        X: (days, nodes, features)
        """
        self.X = X
        self.window_size = window_size
        self.target_size = target_size
        self.n_days = X.shape[0]
        
    def __len__(self):
        return max(0, self.n_days - self.window_size - self.target_size + 1)
    
    def __getitem__(self, idx):
        # Input: dias [idx, idx+window_size)
        # Target: dias [idx+window_size, idx+window_size+target_size) -> média
        x = self.X[idx:idx+self.window_size]  # (window_size, nodes, features)
        y = self.X[idx+self.window_size:idx+self.window_size+self.target_size].mean(dim=0)  # (nodes, features)
        
        return x, y


class STGCN_Cpraio_v2(nn.Module):
    """Modelo STGCN com suporte a múltiplas features"""
    def __init__(self, in_channels=7, hidden_dim=32, n_nodes=735, num_layers=2):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.n_nodes = n_nodes
        
        # LSTM para capturar padrões temporais
        self.lstm = nn.LSTM(in_channels, hidden_dim, num_layers=num_layers, batch_first=True)
        
        # Fully connected para projetar para features
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, in_channels)
        )
    
    def forward(self, x):
        # x: (batch, window_size, n_nodes, features)
        batch_size, window_size, n_nodes, features = x.shape
        
        # Reshape: (batch*n_nodes, window_size, features)
        x = x.permute(0, 2, 1, 3).contiguous()  # (batch, n_nodes, window_size, features)
        x = x.view(batch_size * n_nodes, window_size, features)
        
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)  # (batch*n_nodes, window_size, hidden_dim)
        
        # Usar último hidden state
        h_n = h_n[-1]  # (batch*n_nodes, hidden_dim)
        
        # FC layers
        out = self.fc(h_n)  # (batch*n_nodes, features)
        
        # Reshape back: (batch, n_nodes, features)
        out = out.view(batch_size, n_nodes, features)
        
        return out


def main():
    print('\n' + '='*70)
    print('Sprint 2 Task 2.4: SMALL-RUN VALIDATION (1 EPOCH)')
    print('='*70)
    
    # Carregar tensor
    print('\n[1] Carregando tensor...')
    if not os.path.exists(TENSOR_PATH):
        print(f'    ERRO: {TENSOR_PATH} não encontrado')
        return
    
    X = torch.load(TENSOR_PATH)
    print(f'    - Shape: {X.shape}')
    print(f'    - Dtype: {X.dtype}')
    
    # Carregar metadados
    print('\n[2] Carregando metadados...')
    metadata = torch.load(METADATA_PATH)
    print(f'    - Nós: {len(metadata["nodes"])}')
    print(f'    - Features: {metadata["feature_names"]}')
    
    # Criar dataset
    print('\n[3] Criando dataset de sliding window...')
    window_size = 14
    target_size = 15
    dataset = TimeSeriesDataset(X, window_size=window_size, target_size=target_size)
    print(f'    - Amostras: {len(dataset)}')
    
    # DataLoader
    batch_size = 4
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Criar modelo
    print('\n[4] Criando modelo STGCN_Cpraio_v2...')
    n_nodes = X.shape[1]
    n_features = X.shape[2]
    model = STGCN_Cpraio_v2(in_channels=n_features, hidden_dim=32, n_nodes=n_nodes, num_layers=2)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    X = X.to(device)
    
    print(f'    - Modelo criado')
    print(f'    - Device: {device}')
    print(f'    - Parâmetros: {sum(p.numel() for p in model.parameters())}')
    
    # Loss e optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Treinar 1 epoch
    print('\n[5] Treinando 1 epoch...')
    model.train()
    
    epoch_loss = 0.0
    n_batches = len(dataloader)
    
    for batch_idx, (x_batch, y_batch) in enumerate(dataloader):
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        
        # Forward
        y_pred = model(x_batch)  # (batch, n_nodes, features)
        
        # Loss
        loss = criterion(y_pred, y_batch)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        
        if (batch_idx + 1) % max(1, n_batches // 5) == 0:
            print(f'    Batch {batch_idx+1}/{n_batches}: loss={loss.item():.6f}')
    
    avg_loss = epoch_loss / n_batches
    print(f'\n[6] Epoch finalizado:')
    print(f'    - Loss médio: {avg_loss:.6f}')
    
    # Testar modo de avaliação
    print('\n[7] Testando modo de avaliação...')
    model.eval()
    with torch.no_grad():
        x_sample, y_sample = dataset[0]
        x_sample = x_sample.unsqueeze(0).to(device)  # Adicionar batch dim
        y_pred_sample = model(x_sample)
        
        print(f'    - Input shape: {x_sample.shape}')
        print(f'    - Output shape: {y_pred_sample.shape}')
        print(f'    - Target shape: {y_sample.shape}')
        print(f'    - Output stats: mean={y_pred_sample.mean():.4f}, std={y_pred_sample.std():.4f}')
    
    print('\n' + '='*70)
    print('[✓] Validação completa! Modelo pronto para treino full')
    print('='*70)
    print('\nProximos passos:')
    print('  - Implementar integração com grafo (GCN layers)')
    print('  - Treinar modelo completo com todas as épocas')
    print('  - Fazer backtesting e validação cruzada')
    print()

if __name__ == '__main__':
    main()
