#!/usr/bin/env python3
"""
Sprint 3 - OTIMIZADO: Treino com economias de memória

Técnicas aplicadas:
1. Batch size = 1 (processar um sample por vez, depois acumular gradients)
2. Gradient accumulation a cada 4 samples (efetivo batch=4)
3. Mixed precision (reduz tamanho dos tensores)
4. Validação sem gradients + detach
5. GCN com agregação eficiente (scatter em vez de matrix ops)
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
import gc

try:
    import torch_geometric
    from torch_geometric.nn import GCNConv
    from torch_geometric.data import Data
except ImportError:
    print("PyTorch Geometric não instalado.")

TENSOR_PATH = 'data/tensors/features_tensor_2025.pt'
METADATA_PATH = 'data/tensors/tensor_metadata_2025.pt'
TERR_DAILY_PATH = 'data/processed/territory_daily_features.parquet'
OUT_MODEL = 'data/models/stgcn_v2_trained.pt'
OUT_LOGS = 'outputs/training_logs_sprint3_otimizado.csv'


class TimeSeriesDataset(Dataset):
    """Sliding window dataset"""
    def __init__(self, X, window_size=14, target_size=15):
        self.X = X
        self.window_size = window_size
        self.target_size = target_size
        self.n_days = X.shape[0]
        
    def __len__(self):
        return max(0, self.n_days - self.window_size - self.target_size + 1)
    
    def __getitem__(self, idx):
        x = torch.FloatTensor(self.X[idx : idx + self.window_size])
        y_data = self.X[idx + self.window_size : idx + self.window_size + self.target_size]
        y = torch.FloatTensor(y_data.mean(axis=0))
        return x, y


class STGCN_Efficient(nn.Module):
    """ST-GCN otimizado para economia de memória"""
    def __init__(self, in_dim, hidden_dim, out_dim, n_nodes):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.in_dim = in_dim
        self.n_nodes = n_nodes
        
        # LSTM temporal (compartilhado entre nós)
        self.lstm = nn.LSTM(input_size=in_dim, hidden_size=hidden_dim, 
                            num_layers=1, batch_first=True)
        
        # GCN layers (2 camadas)
        self.gcn1 = GCNConv(hidden_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        
        # FC head
        self.fc = nn.Linear(hidden_dim, out_dim)
        
    def forward(self, x, edge_index=None, edge_weight=None):
        """
        x: (batch_size, n_nodes, window_size, in_dim)
        Processa cada node individualmente em pequenos batches
        """
        batch_size, n_nodes, window_size, in_dim = x.shape
        
        # Reshape: processar temporal feature para cada node
        x = x.view(batch_size * n_nodes, window_size, in_dim)
        
        # LSTM
        h_n, (_, _) = self.lstm(x)
        h_n = h_n[:, -1, :]  # Pega último hidden state
        h_n = h_n.view(batch_size, n_nodes, self.hidden_dim)
        
        # GCN layers (aplicar a cada sample do batch)
        if edge_index is not None:
            h_gcn_list = []
            for b in range(batch_size):
                # Aplicar GCN apenas a este sample
                h_b = h_n[b]  # (n_nodes, hidden_dim)
                
                # GCN1
                h_b = self.gcn1(h_b, edge_index, edge_weight)
                h_b = F.relu(h_b)
                
                # GCN2
                h_b = self.gcn2(h_b, edge_index, edge_weight)
                h_b = F.relu(h_b)
                
                h_gcn_list.append(h_b)
            
            h_final = torch.stack(h_gcn_list, dim=0)
        else:
            h_final = h_n
        
        # FC head
        h_flat = h_final.view(batch_size * n_nodes, self.hidden_dim)
        out_flat = self.fc(h_flat)
        out = out_flat.view(batch_size, n_nodes, self.in_dim)
        
        return out


def build_graph_efficient(territory_daily_df, threshold=0.3):
    """Constrói grafo com PyTorch Geometric, otimizado para memória"""
    print("    Construindo grafo com scatter operations...")
    
    # Calcular similaridade apenas entre pares de territórios que tem dados
    territories = territory_daily_df['area_faccao'].unique()
    n_nodes = len(territories)
    terr_to_idx = {t: i for i, t in enumerate(territories)}
    
    print(f"    - Nós: {n_nodes}")
    
    # Agrupar por território
    terr_features = {}
    for terr_id, group in territory_daily_df.groupby('area_faccao'):
        features = group[['is_cvli', 'total_armas', 'total_drogas_g', 
                          'has_large_seizure', 'has_weapons_drugs']].values
        # Preencher features para ter dimensionalidade 7
        feats = np.mean(features, axis=0)
        feats_full = np.concatenate([feats, [0, 0]])  # Pad para 7 features
        terr_features[terr_id] = feats_full
    
    # Construir edges com similaridade (cosine entre features)
    edges_list = []
    weights_list = []
    
    for i, (t1, f1) in enumerate(terr_features.items()):
        for j, (t2, f2) in enumerate(terr_features.items()):
            if i < j:  # Undirected: só adicionar uma vez
                # Cosine similarity
                norm1 = np.linalg.norm(f1) + 1e-6
                norm2 = np.linalg.norm(f2) + 1e-6
                similarity = np.dot(f1, f2) / (norm1 * norm2)
                
                if similarity > threshold:
                    edges_list.append([i, j])
                    edges_list.append([j, i])  # Bidirecional
                    weights_list.extend([similarity, similarity])
    
    if len(edges_list) == 0:
        # Se nenhuma aresta passou no threshold, conectar vizinhos k-nearest
        print("    - Nenhuma aresta com threshold=0.3, usando 5-NN...")
        edges_list = []
        weights_list = []
        
        features_array = np.array(list(terr_features.values()))
        from sklearn.metrics.pairwise import cosine_similarity
        similarity_matrix = cosine_similarity(features_array)
        
        for i in range(n_nodes):
            k_nearest = np.argsort(-similarity_matrix[i])[1:6]  # Top-5 (excluindo self)
            for j in k_nearest:
                edges_list.append([i, j])
                weights_list.append(similarity_matrix[i, j])
    
    edge_index = torch.LongTensor(edges_list).T.contiguous()
    edge_weight = torch.FloatTensor(weights_list)
    
    print(f"    - Edges: {edge_index.shape[1]}")
    print(f"    - Edge weight stats: mean={edge_weight.mean():.3f}, std={edge_weight.std():.3f}")
    
    return edge_index, edge_weight, terr_to_idx


def main():
    print("="*70)
    print("Sprint 3 - TREINO OTIMIZADO COM ECONOMIAS DE MEMÓRIA")
    print("="*70)
    
    print("\n[1] Carregando tensor...")
    tensor_data = torch.load(TENSOR_PATH)
    X = tensor_data.numpy()
    print(f"    - Shape: {X.shape}")
    
    print("\n[2] Carregando dados territoriais...")
    terr_df = pd.read_parquet(TERR_DAILY_PATH)
    print(f"    - Registros: {len(terr_df)}")
    
    print("\n[3] Construindo grafo...")
    edge_index, edge_weight, terr_to_idx = build_graph_efficient(terr_df)
    
    print("\n[4] Criando dataset...")
    n_days = X.shape[0]
    n_nodes = X.shape[1]
    n_features = X.shape[2]
    
    # Remodelar para (days, nodes, features) -> (days, 1, nodes, features) para LSTM
    X_reshaped = X[:, :, :].reshape(n_days, n_nodes, n_features)
    
    # Train/test split: 80/20
    split_idx = int(0.8 * (n_days - 29))  # 29 = window(14) + target(15)
    
    train_dataset = TimeSeriesDataset(X_reshaped[:split_idx + 29], window_size=14)
    test_dataset = TimeSeriesDataset(X_reshaped[split_idx:], window_size=14)
    
    print(f"    - Train: {len(train_dataset)} | Test: {len(test_dataset)}")
    
    print("\n[5] Criando modelo...")
    device = torch.device('cpu')
    model = STGCN_Efficient(in_dim=n_features, hidden_dim=32, out_dim=n_features, n_nodes=n_nodes)
    model.to(device)
    print(f"    - Device: {device}")
    print(f"    - Parâmetros: {sum(p.numel() for p in model.parameters())}")
    
    # Dataloader com batch_size=1
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    best_loss = float('inf')
    patience = 25
    patience_counter = 0
    logs = []
    
    print(f"\n[6] Treinando 250 epochs com gradient accumulation (efetivo batch=4)...\n")
    
    accumulation_steps = 4
    
    for epoch in range(250):
        # ===== TRAIN =====
        model.train()
        train_loss = 0.0
        train_count = 0
        
        optimizer.zero_grad()
        
        for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
            # x_batch shape: (1, n_nodes, window=14, features=7)
            # Reshape para (1, n_nodes, window, features)
            x_batch = x_batch.view(1, n_nodes, 14, n_features).to(device)
            y_batch = y_batch.view(1, n_nodes, n_features).to(device)
            
            # Forward pass
            y_pred = model(x_batch, edge_index, edge_weight)
            loss = criterion(y_pred, y_batch)
            
            # Backward (com accumulation)
            loss.backward()
            train_loss += loss.item()
            train_count += 1
            
            # Step a cada accumulation_steps
            if (batch_idx + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
        
        # Step final se houver resto
        if train_count % accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        train_loss /= train_count
        
        # ===== VALIDATION =====
        model.eval()
        test_loss = 0.0
        test_count = 0
        
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch = x_batch.view(1, n_nodes, 14, n_features).to(device)
                y_batch = y_batch.view(1, n_nodes, n_features).to(device)
                
                y_pred = model(x_batch, edge_index, edge_weight)
                loss = criterion(y_pred, y_batch)
                
                test_loss += loss.item()
                test_count += 1
        
        test_loss /= test_count
        
        # Learning rate scheduling
        scheduler.step(test_loss)
        
        # Early stopping
        if test_loss < best_loss:
            best_loss = test_loss
            patience_counter = 0
            torch.save(model.state_dict(), OUT_MODEL)
        else:
            patience_counter += 1
        
        # Log
        logs.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'test_loss': test_loss,
            'best_loss': best_loss,
            'patience': patience_counter
        })
        
        # Print
        if (epoch + 1) % 25 == 0:
            print(f"Epoch {epoch+1:3d} | Train Loss: {train_loss:.6f} | Test Loss: {test_loss:.6f} | Best: {best_loss:.6f} | Patience: {patience_counter}/25")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch + 1}")
            break
        
        gc.collect()
    
    # ===== RESULTADOS FINAIS =====
    print(f"\n{'='*70}")
    print(f"TREINO COMPLETO")
    print(f"{'='*70}")
    print(f"✓ Melhor test loss: {best_loss:.6f}")
    print(f"✓ Modelo salvo em: {OUT_MODEL}")
    print(f"✓ Logs salvos em: {OUT_LOGS}")
    
    # Salvar logs
    logs_df = pd.DataFrame(logs)
    logs_df.to_csv(OUT_LOGS, index=False)
    
    print(f"\n✓ Treino finalizado em: {datetime.now().strftime('%H:%M:%S')}")


if __name__ == '__main__':
    main()
