"""
TRAINER - NOVO CRITÉRIO CVLI-CENTRIC
====================================
Treina ST-GCN com:
1. Dados CVLI 2022-2024 para treino
2. Dados CVLI 2025 para validação
3. CVP apenas como contexto
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import config
from src.model import STGCN_Cpraio

class CrimeSeriesDataset(Dataset):
    """
    Dataset para série temporal de crimes
    Input: últimos 14 dias
    Target: média dos próximos 15 dias
    """
    def __init__(self, X, window_size=14, target_window=15):
        self.X = X
        self.window_size = window_size
        self.target_window = target_window
        
    def __len__(self):
        return len(self.X) - self.window_size - self.target_window + 1
    
    def __getitem__(self, idx):
        x_seq = self.X[idx : idx + self.window_size]
        target_seq = self.X[idx + self.window_size : idx + self.window_size + self.target_window]
        y_target = target_seq.mean(dim=0)
        return x_seq, y_target

TRAIN_CONFIG = {
    'batch_size': 32,
    'epochs': 150,
    'learning_rate': 0.001,
    'weight_decay': 1e-4,
    'patience': 20,
    'cvli_weight': 1.0  # Feature 0 é CVLI (mais importante)
}

def load_dataset():
    """Carrega dataset construído pelo graph_builder"""
    dataset_path = config.TENSOR_DIR / "dataset_cvli_novo_criterio.pt"
    
    if not dataset_path.exists():
        print(f"[X] Dataset não encontrado: {dataset_path}")
        print("    Execute: python scripts_ajuste/02_graph_builder_novo.py")
        return None, None, None
    
    print("[-] Carregando dataset...")
    data = torch.load(dataset_path, weights_only=False)
    
    X = data['X']
    edge_index = data['edge_index']
    metadata = data
    
    print(f"  [V] Shape X: {X.shape}")
    print(f"  [V] Edge index: {edge_index.shape}")
    
    return X, edge_index, metadata

def load_validation_data(bairro_to_idx):
    """
    Carrega dados de validação de 2025
    """
    val_file = config.DATA_PROCESSED / "dataset_validacao_cvli_2025.parquet"
    
    if not val_file.exists():
        print(f"[!] Dados de validação não encontrados: {val_file}")
        return None
    
    print("[-] Carregando dados de validação 2025...")
    df_val = pd.read_parquet(val_file)
    
    # Criar série temporal para validação
    date_range_val = pd.date_range(
        start=df_val['data'].min(),
        end=df_val['data'].max(),
        freq='D'
    )
    
    num_days = len(date_range_val)
    num_nodes = len(bairro_to_idx)
    num_features = 6  # Mesmo que treino
    
    X_val = np.zeros((num_days, num_nodes, num_features))
    
    # Preencher
    for day_idx, date in enumerate(date_range_val):
        day_data = df_val[df_val['data'].dt.date == date.date()]
        
        for _, row in day_data.iterrows():
            bairro = row['bairro'].upper() if pd.notna(row['bairro']) else 'DESCONHECIDO'
            
            if bairro not in bairro_to_idx:
                continue
            
            node_idx = bairro_to_idx[bairro]
            
            if row['tipo_crime'].lower() == 'cvli':
                X_val[day_idx, node_idx, 0] += 1
            elif row['tipo_crime'].lower() == 'cvp':
                X_val[day_idx, node_idx, 1] += 1
            
            faccao = str(row['faccao']).upper() if pd.notna(row['faccao']) else 'SEM_FACCAO'
            if 'CV' in faccao:
                X_val[day_idx, node_idx, 2] += 1
            elif 'PCC' in faccao:
                X_val[day_idx, node_idx, 3] += 1
            elif 'GDE' in faccao:
                X_val[day_idx, node_idx, 4] += 1
            else:
                X_val[day_idx, node_idx, 5] += 1
    
    X_val_torch = torch.tensor(X_val, dtype=torch.float32)
    
    print(f"  [V] Validação shape: {X_val_torch.shape}")
    
    return X_val_torch

def train():
    """Orquestração principal de treinamento"""
    print("=" * 60)
    print(" TRAINER - NOVO CRITÉRIO CVLI-CENTRIC")
    print("=" * 60)
    
    # 1. Carregar dados
    X, edge_index, metadata = load_dataset()
    if X is None:
        return
    
    bairro_to_idx = metadata.get('bairro_to_idx', {})
    
    # 2. Normalização Z-Score
    mean = X.mean(dim=(0, 1), keepdim=True)
    std = X.std(dim=(0, 1), keepdim=True) + 1e-5
    X_norm = (X - mean) / std
    
    print(f"[-] Normalização: mean={mean.mean():.4f}, std={std.mean():.4f}")
    
    # 3. Split treino (80%) / validação-treino (20%)
    split_idx = int(len(X_norm) * 0.8)
    train_data = X_norm[:split_idx]
    val_data = X_norm[split_idx:]
    
    # 4. Criar datasets
    ws = 14
    tw = 15
    
    train_ds = CrimeSeriesDataset(train_data, ws, tw)
    val_ds = CrimeSeriesDataset(val_data, ws, tw)
    
    if len(val_ds) == 0:
        val_ds = train_ds
        print("[!] Dados insuficientes para validação. Usando treino.")
    
    train_loader = DataLoader(train_ds, batch_size=TRAIN_CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=TRAIN_CONFIG['batch_size'], shuffle=False)
    
    print(f"[-] Datasets: Train={len(train_ds)}, Val={len(val_ds)}")
    
    # 5. Modelo
    num_nodes = X.shape[1]
    num_features = X.shape[2]
    
    model = STGCN_Cpraio(
        num_nodes=num_nodes,
        in_channels=num_features,
        hidden_channels=32,
        out_channels=num_features,
        dropout=0.4
    )
    
    optimizer = optim.Adam(model.parameters(), lr=TRAIN_CONFIG['learning_rate'], 
                          weight_decay=TRAIN_CONFIG['weight_decay'])
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    print(f"[-] Modelo: {num_nodes} nós, {num_features} features")
    
    # 6. Loop de treinamento
    best_loss = float('inf')
    patience_counter = 0
    
    model_path = config.MODEL_DIR / "model_cvli_novo_criterio.pth"
    stats_path = config.MODEL_DIR / "stats_cvli_novo_criterio.pt"
    
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    pbar = tqdm(range(TRAIN_CONFIG['epochs']), desc="Treinando CVLI-centric")
    
    for epoch in pbar:
        # Treino
        model.train()
        train_loss = 0
        
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            out = model(x_batch, edge_index)
            loss = criterion(out, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validação
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                out = model(x_batch, edge_index)
                loss = criterion(out, y_batch)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)
        
        pbar.set_postfix({
            'TLoss': f'{avg_train_loss:.4f}',
            'VLoss': f'{avg_val_loss:.4f}'
        })
        
        # Early stopping
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), model_path)
            torch.save({'mean': mean, 'std': std}, stats_path)
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= TRAIN_CONFIG['patience']:
            print(f"\n[!] Early Stopping na época {epoch+1}")
            break
    
    print(f"\n[V] Modelo salvo: {model_path}")
    print(f"[V] Stats salvas: {stats_path}")
    
    # 7. Teste em dados 2025
    print("\n[-] Teste em dados de validação 2025...")
    X_val_2025 = load_validation_data(bairro_to_idx)
    
    if X_val_2025 is not None:
        X_val_2025_norm = (X_val_2025 - mean) / std
        
        model.eval()
        with torch.no_grad():
            # Fazer predições
            val_2025_ds = CrimeSeriesDataset(X_val_2025_norm, ws, tw)
            
            if len(val_2025_ds) > 0:
                val_2025_loader = DataLoader(val_2025_ds, batch_size=TRAIN_CONFIG['batch_size'])
                
                test_loss = 0
                for x_batch, y_batch in val_2025_loader:
                    out = model(x_batch, edge_index)
                    loss = criterion(out, y_batch)
                    test_loss += loss.item()
                
                avg_test_loss = test_loss / len(val_2025_loader)
                print(f"[V] Loss em 2025: {avg_test_loss:.4f}")
    
    print("\n" + "=" * 60)
    print(" TREINAMENTO CONCLUÍDO COM SUCESSO")
    print("=" * 60)

if __name__ == "__main__":
    train()
