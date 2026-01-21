#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Treinador ST-GCN - Criticidade Janela 180 Dias
"""

import os
import sys
import torch
import torch.nn as nn
from pathlib import Path
import json
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from model import STGCN_Cpraio

def main():
    print("\n" + "="*80)
    print("ST-GCN - CRITICIDADE JANELA 180 DIAS")
    print("="*80)
    
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / 'data'
    OUTPUT_DIR = BASE_DIR / 'outputs'
    MODELS_DIR = OUTPUT_DIR / 'models'
    
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[DEVICE] {device}")
    
    # CONFIG
    WINDOW = 14
    BATCH_SIZE = 32
    EPOCHS = 150
    LR = 0.001
    
    try:
        # LOAD DATA
        print("\n[1] LOADING DATA...")
        data = torch.load(DATA_DIR / 'tensors' / 'dataset_criticidade_janela180d.pt')
        dataset = data
        
        print(f"    Shape: {dataset.shape}")
        
        with open(DATA_DIR / 'tensors' / 'metadata_janela180d.json', 'r', encoding='utf-8') as f:
            meta = json.load(f)
        
        adj = np.load(DATA_DIR / 'tensors' / 'adjacency_matrix.npy')
        edge_index = torch.tensor(np.argwhere(adj > 0).T, dtype=torch.long)
        
        num_nodes = meta['num_nodes']
        num_features = meta['num_features']
        print(f"    Nodes: {num_nodes}, Features: {num_features}")
        print(f"    Edges: {edge_index.shape[1]}")
        
        # CREATE SEQUENCES
        print("\n[2] CREATING SEQUENCES...")
        X, y = [], []
        for i in range(len(dataset) - WINDOW):
            X.append(dataset[i:i+WINDOW])  # Window of 14 days
            y.append(dataset[i+WINDOW])     # Next day
        
        X = torch.stack(X)  # [time_steps, window, nodes, features]
        y = torch.stack(y)  # [time_steps, nodes, features]
        print(f"    X: {X.shape}, y: {y.shape}")
        
        # Normalize
        X_mean, X_std = X.mean(dim=(0, 1, 2), keepdim=True), X.std(dim=(0, 1, 2), keepdim=True)
        X_std[X_std == 0] = 1
        X = (X - X_mean) / X_std
        
        y_mean, y_std = y.mean(dim=(0, 1, 2), keepdim=True), y.std(dim=(0, 1, 2), keepdim=True)
        y_std[y_std == 0] = 1
        y = (y - y_mean) / y_std
        
        # Train/val split
        split = int(0.8 * len(X))
        X_tr, X_va = X[:split], X[split:]
        y_tr, y_va = y[:split], y[split:]
        
        print(f"    Train: {X_tr.shape[0]}, Val: {X_va.shape[0]}")
        
        # DATALOADERS
        print("\n[3] CREATING DATALOADERS...")
        tr_data = torch.utils.data.TensorDataset(X_tr, y_tr)
        va_data = torch.utils.data.TensorDataset(X_va, y_va)
        
        tr_loader = torch.utils.data.DataLoader(tr_data, batch_size=BATCH_SIZE, shuffle=True)
        va_loader = torch.utils.data.DataLoader(va_data, batch_size=BATCH_SIZE)
        
        print(f"    Batches - Train: {len(tr_loader)}, Val: {len(va_loader)}")
        
        # MODEL
        print("\n[4] CREATING MODEL...")
        model = STGCN_Cpraio(
            num_nodes=num_nodes,
            in_channels=num_features,
            hidden_channels=64,
            out_channels=num_features,
            dropout=0.3
        )
        model.edge_index = edge_index
        model.to(device)
        print(f"    Params: {sum(p.numel() for p in model.parameters()):,}")
        
        # OPTIMIZER
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        criterion = nn.MSELoss()
        
        # TRAINING
        print("\n[5] TRAINING...")
        print("="*80)
        
        best_val = float('inf')
        patience = 0
        history = []
        
        for epoch in range(EPOCHS):
            # TRAIN
            model.train()
            tr_loss = 0
            for X_b, y_b in tr_loader:
                X_b = X_b.to(device)
                y_b = y_b.to(device)
                
                # Squeeze window dimension for model
                X_b_flat = X_b.squeeze(-1)  # [batch, window, nodes] - remove feature dim (1)
                X_b_flat = X_b_flat.mean(dim=1)  # [batch, nodes] - average window
                X_b_flat = X_b_flat.unsqueeze(-1)  # [batch, nodes, 1]
                
                optimizer.zero_grad()
                pred = model(X_b_flat, edge_index.to(device))
                
                # Reshape for loss
                pred = pred.view(y_b.shape[0], num_nodes, num_features)
                
                loss = criterion(pred, y_b)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                tr_loss += loss.item()
            
            tr_loss /= len(tr_loader)
            
            # VALIDATE
            model.eval()
            va_loss = 0
            with torch.no_grad():
                for X_b, y_b in va_loader:
                    X_b = X_b.to(device)
                    y_b = y_b.to(device)
                    
                    X_b_flat = X_b.squeeze(-1).mean(dim=1).unsqueeze(-1)
                    
                    pred = model(X_b_flat, edge_index.to(device))
                    pred = pred.view(y_b.shape[0], num_nodes, num_features)
                    va_loss += criterion(pred, y_b).item()
            
            va_loss /= len(va_loader)
            
            history.append({'epoch': epoch+1, 'train': tr_loss, 'val': va_loss})
            
            if va_loss < best_val:
                best_val = va_loss
                patience = 0
                torch.save(model.state_dict(), MODELS_DIR / 'model_janela180d.pth')
                torch.save({'X_mean': X_mean, 'X_std': X_std, 'y_mean': y_mean, 'y_std': y_std},
                          MODELS_DIR / 'stats_janela180d.pt')
                print(f"Ep {epoch+1:3d} | Tr: {tr_loss:.6f} | Va: {va_loss:.6f} ✓")
            else:
                patience += 1
                print(f"Ep {epoch+1:3d} | Tr: {tr_loss:.6f} | Va: {va_loss:.6f}")
                
                if patience >= 20:
                    print(f"\nEarly stop (patience={patience})")
                    break
        
        print("\n" + "="*80)
        print("✅ TRAINING COMPLETE")
        print("="*80)
        
        with open(MODELS_DIR / 'training_history_janela180d.json', 'w') as f:
            json.dump(history, f, indent=2)
        
        print(f"Best val loss: {best_val:.6f}")
        print(f"Model saved to {MODELS_DIR}")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
