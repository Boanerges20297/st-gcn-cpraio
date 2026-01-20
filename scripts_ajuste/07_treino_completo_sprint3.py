#!/usr/bin/env python3
"""
Sprint 3: Treino completo com STGCN_Cpraio_v2 + GCN layers + edge_weights

- Constrói grafo com PyTorch Geometric (nós=territórios, edges=proximidade+features)
- Implementa STGCN_Cpraio_v2 com GCN layers com edge_weights
- Treina 250 epochs com early stopping (patience=25)
- Backtesting: train (80%) / test (20%), calcula MSE/MAE/RMSE
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
import matplotlib.pyplot as plt

try:
    import torch_geometric
    from torch_geometric.nn import GCNConv
    from torch_geometric.data import Data
except ImportError:
    print("PyTorch Geometric não instalado. Usando modelo sem GCN.")

TENSOR_PATH = 'data/tensors/features_tensor_2025.pt'
METADATA_PATH = 'data/tensors/tensor_metadata_2025.pt'
TERR_DAILY_PATH = 'data/processed/territory_daily_features.parquet'
OUT_MODEL = 'data/models/stgcn_v2_trained.pt'
OUT_LOGS = 'outputs/training_logs_sprint3.csv'


class TimeSeriesDataset(Dataset):
    """Sliding window dataset para séries temporais"""
    def __init__(self, X, window_size=14, target_size=15):
        self.X = X
        self.window_size = window_size
        self.target_size = target_size
        self.n_days = X.shape[0]
        
    def __len__(self):
        return max(0, self.n_days - self.window_size - self.target_size + 1)
    
    def __getitem__(self, idx):
        x = self.X[idx:idx+self.window_size]  # (window_size, nodes, features)
        y = self.X[idx+self.window_size:idx+self.window_size+self.target_size].mean(dim=0)  # (nodes, features)
        return x, y


class STGCN_Cpraio_v2_GCN(nn.Module):
    """STGCN com GCN layers e edge_weights"""
    def __init__(self, in_channels=7, hidden_dim=32, n_nodes=735, num_layers=2, graph_data=None):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.n_nodes = n_nodes
        self.graph_data = graph_data
        
        # LSTM para capturar padrões temporais
        self.lstm = nn.LSTM(in_channels, hidden_dim, num_layers=num_layers, batch_first=True)
        
        # GCN layers se graph_data disponível
        if graph_data is not None:
            self.gcn1 = GCNConv(hidden_dim, hidden_dim)
            self.gcn2 = GCNConv(hidden_dim, hidden_dim)
            self.use_gcn = True
        else:
            self.use_gcn = False
        
        # Fully connected para saída
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, in_channels)
        )
    
    def forward(self, x):
        # x: (batch, window_size, n_nodes, features)
        batch_size, window_size, n_nodes, features = x.shape
        
        # Reshape para LSTM: (batch*n_nodes, window_size, features)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size * n_nodes, window_size, features)
        
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)
        h_n = h_n[-1]  # (batch*n_nodes, hidden_dim)
        
        # GCN layers se disponível
        if self.use_gcn:
            h_n_per_batch = h_n.view(batch_size, n_nodes, self.hidden_dim)
            h_gcn_list = []
            
            for b in range(batch_size):
                h_batch = h_n_per_batch[b]  # (n_nodes, hidden_dim)
                
                # GCN forward pass
                edge_index = self.graph_data.edge_index.to(h_batch.device)
                edge_weight = self.graph_data.edge_weight.to(h_batch.device)
                
                h_batch = self.gcn1(h_batch, edge_index, edge_weight)
                h_batch = F.relu(h_batch)
                h_batch = self.gcn2(h_batch, edge_index, edge_weight)
                
                h_gcn_list.append(h_batch)
            
            h_n = torch.stack(h_gcn_list, dim=0).view(batch_size * n_nodes, self.hidden_dim)
        
        # FC layers
        out = self.fc(h_n)  # (batch*n_nodes, features)
        out = out.view(batch_size, n_nodes, features)
        
        return out


def build_graph_from_territory_daily(df, n_nodes=735):
    """
    Constrói grafo PyTorch Geometric baseado em proximidade de features
    """
    print('    Construindo grafo...')
    
    # Agregar por território
    territory_features = df.groupby('area_faccao').agg({
        'feature_score': 'mean',
        'total_drogas_g': 'sum',
        'total_armas': 'sum'
    }).reset_index()
    
    # Criar mapping territory -> node_idx
    territories = sorted(df['area_faccao'].unique())
    terr_to_idx = {t: i for i, t in enumerate(territories)}
    
    # Criar edges baseado em correlação de features
    # Estratégia simples: conectar nós com features similares
    edges = []
    edge_weights = []
    
    for i, terr1 in enumerate(territories[:n_nodes]):
        feat1 = territory_features[territory_features['area_faccao'] == terr1].iloc[0] if len(territory_features[territory_features['area_faccao'] == terr1]) > 0 else None
        
        if feat1 is None:
            continue
        
        for j, terr2 in enumerate(territories[:n_nodes]):
            if i >= j:  # Evitar duplicatas e auto-loops
                continue
            
            feat2 = territory_features[territory_features['area_faccao'] == terr2].iloc[0] if len(territory_features[territory_features['area_faccao'] == terr2]) > 0 else None
            
            if feat2 is None:
                continue
            
            # Similaridade: diferença normalizada em feature_score
            score_sim = 1.0 / (1.0 + abs(feat1['feature_score'] - feat2['feature_score']))
            
            # Criar edge se similaridade > threshold
            if score_sim > 0.3:
                edges.append([i, j])
                edges.append([j, i])  # Grafo não-dirigido
                edge_weights.append(score_sim)
                edge_weights.append(score_sim)
    
    if len(edges) == 0:
        # Se nenhum edge por similaridade, conectar vizinhos (sequencial)
        for i in range(n_nodes - 1):
            edges.append([i, i+1])
            edges.append([i+1, i])
            edge_weights.append(1.0)
            edge_weights.append(1.0)
    
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(edge_weights, dtype=torch.float32)
    
    graph_data = Data(edge_index=edge_index, edge_weight=edge_weight, num_nodes=n_nodes)
    
    print(f'    - Nós: {n_nodes}')
    print(f'    - Edges: {edge_index.shape[1]}')
    print(f'    - Edge weight stats: mean={edge_weight.mean():.3f}, std={edge_weight.std():.3f}')
    
    return graph_data


def main():
    print('\n' + '='*70)
    print('Sprint 3: TREINO COMPLETO COM GCN + EDGE_WEIGHTS')
    print('='*70)
    
    # Carregar tensor
    print('\n[1] Carregando tensor...')
    if not os.path.exists(TENSOR_PATH):
        print(f'    ERRO: {TENSOR_PATH} não encontrado')
        return
    
    X = torch.load(TENSOR_PATH)
    print(f'    - Shape: {X.shape}')
    
    # Carregar metadados
    print('\n[2] Carregando metadados...')
    metadata = torch.load(METADATA_PATH)
    n_nodes = metadata['shape'][1]
    n_features = metadata['shape'][2]
    
    # Carregar territory_daily para construir grafo
    print('\n[3] Carregando dados territoriais para construir grafo...')
    df_terr = pd.read_parquet(TERR_DAILY_PATH)
    
    # Construir grafo
    print('\n[4] Construindo grafo com PyTorch Geometric...')
    graph_data = build_graph_from_territory_daily(df_terr, n_nodes=n_nodes)
    
    # Criar dataset
    print('\n[5] Criando dataset com sliding window...')
    window_size = 14
    target_size = 15
    dataset = TimeSeriesDataset(X, window_size=window_size, target_size=target_size)
    
    # Split train/test (80/20)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    batch_size = 2  # Reduzido de 8 para evitar OOM com 490K edges
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f'    - Train: {len(train_dataset)} | Test: {len(test_dataset)}')
    
    # Criar modelo
    print('\n[6] Criando modelo STGCN_v2 com GCN...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = STGCN_Cpraio_v2_GCN(in_channels=n_features, hidden_dim=32, n_nodes=n_nodes, 
                                num_layers=2, graph_data=graph_data)
    model = model.to(device)
    
    print(f'    - Device: {device}')
    print(f'    - Parâmetros: {sum(p.numel() for p in model.parameters())}')
    
    # Loss e optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    # Early stopping
    best_loss = float('inf')
    patience = 25
    patience_counter = 0
    
    # Logs
    logs = []
    
    print(f'\n[7] Treinando {250} epochs com early stopping...\n')
    
    num_epochs = 250
    
    for epoch in range(num_epochs):
        # Train
        model.train()
        train_loss = 0.0
        
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * x_batch.size(0)
        
        train_loss /= len(train_dataset)
        
        # Eval
        model.eval()
        test_loss = 0.0
        
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                
                y_pred = model(x_batch)
                loss = criterion(y_pred, y_batch)
                
                test_loss += loss.item() * x_batch.size(0)
        
        test_loss /= len(test_dataset)
        
        # LR scheduler
        scheduler.step(test_loss)
        
        # Early stopping
        if test_loss < best_loss:
            best_loss = test_loss
            patience_counter = 0
            # Salvar melhor modelo
            os.makedirs(os.path.dirname(OUT_MODEL), exist_ok=True)
            torch.save(model.state_dict(), OUT_MODEL)
        else:
            patience_counter += 1
        
        # Log
        logs.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'test_loss': test_loss,
            'best_loss': best_loss,
            'lr': optimizer.param_groups[0]['lr']
        })
        
        # Print
        if (epoch + 1) % 25 == 0 or patience_counter >= patience:
            print(f'Epoch {epoch+1:3d} | Train Loss: {train_loss:.6f} | Test Loss: {test_loss:.6f} | Best: {best_loss:.6f} | Patience: {patience_counter}/{patience}')
        
        # Early stopping
        if patience_counter >= patience:
            print(f'\nEarly stopping triggered at epoch {epoch+1}')
            break
    
    print(f'\n[8] Treino finalizado:')
    print(f'    - Épocas: {epoch+1}/{num_epochs}')
    print(f'    - Best test loss: {best_loss:.6f}')
    print(f'    - Modelo salvo: {OUT_MODEL}')
    
    # Salvar logs
    logs_df = pd.DataFrame(logs)
    os.makedirs(os.path.dirname(OUT_LOGS), exist_ok=True)
    logs_df.to_csv(OUT_LOGS, index=False)
    print(f'    - Logs salvos: {OUT_LOGS}')
    
    # Backtesting: Calcular métricas
    print(f'\n[9] Backtesting...')
    
    model.load_state_dict(torch.load(OUT_MODEL))
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            y_pred = model(x_batch)
            
            all_predictions.append(y_pred.cpu().numpy())
            all_targets.append(y_batch.cpu().numpy())
    
    y_pred_all = np.concatenate(all_predictions, axis=0)  # (n_samples, n_nodes, features)
    y_target_all = np.concatenate(all_targets, axis=0)
    
    # Métricas por feature
    mse_per_feature = []
    mae_per_feature = []
    rmse_per_feature = []
    
    for f_idx in range(n_features):
        y_pred_f = y_pred_all[:, :, f_idx].flatten()
        y_target_f = y_target_all[:, :, f_idx].flatten()
        
        mse = np.mean((y_pred_f - y_target_f) ** 2)
        mae = np.mean(np.abs(y_pred_f - y_target_f))
        rmse = np.sqrt(mse)
        
        mse_per_feature.append(mse)
        mae_per_feature.append(mae)
        rmse_per_feature.append(rmse)
        
        feature_name = metadata['feature_names'][f_idx]
        print(f'    Feature {f_idx} ({feature_name}): MSE={mse:.6f} | MAE={mae:.6f} | RMSE={rmse:.6f}')
    
    # Overall metrics
    mse_overall = np.mean(mse_per_feature)
    mae_overall = np.mean(mae_per_feature)
    rmse_overall = np.mean(rmse_per_feature)
    
    print(f'\n    Overall: MSE={mse_overall:.6f} | MAE={mae_overall:.6f} | RMSE={rmse_overall:.6f}')
    
    print(f'\n' + '='*70)
    print('[✓] Sprint 3 Completa!')
    print('='*70)
    print(f'\nArtefatos gerados:')
    print(f'  - Modelo treinado: {OUT_MODEL}')
    print(f'  - Logs de treino: {OUT_LOGS}')
    print(f'\nProximos passos:')
    print(f'  - Fazer previsões em tempo real')
    print(f'  - Integrar com dashboard estratégico')
    print(f'  - Monitorar performance em produção')
    print()

if __name__ == '__main__':
    main()
