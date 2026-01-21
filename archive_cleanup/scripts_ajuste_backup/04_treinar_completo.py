#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Retreinar ST-GCN com cobertura completa (389 nodes)
Usar hiperparametros identicos ao treinamento anterior para comparacao justa
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import sys

print("="*80)
print("RETREINAMENTO ST-GCN COM COBERTURA COMPLETA (389 NODES)")
print("="*80)

print("\n[1] Carregando dependências...")

# Adicionar src ao path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

try:
    from model import STGCN_Cpraio
    print("[OK] Modelo STGCN_Cpraio importado")
except ImportError as e:
    print("[ERRO] Não conseguiu importar modelo: {}".format(e))
    exit(1)

print("\n[2] Carregando dataset...")
dataset = torch.load('data/tensors/dataset_stgcn_completo.pt', weights_only=False)

X = dataset['X']
edge_index = dataset['edge_index']

print("[OK] X shape: {}".format(X.shape))
print("[OK] Edge_index shape: {}".format(edge_index.shape))

print("\n[3] Preparando dados...")
# Parametros iguais ao treino original
window_size = 14
target_window = 15
batch_size = 32
test_split = 0.2

# Criar windows de treino
def create_windows(X, window_size, target_window):
    X_windows = []
    y_windows = []
    
    max_idx = len(X) - window_size - target_window + 1
    for i in range(max_idx):
        x_seq = X[i:i + window_size]
        y_seq = X[i + window_size:i + window_size + target_window]
        y_target = y_seq.mean(dim=0)
        
        X_windows.append(x_seq)
        y_windows.append(y_target)
    
    return torch.stack(X_windows), torch.stack(y_windows)

print("    Criando windows (tamanho: {}, target: {})...".format(window_size, target_window))
X_win, y_win = create_windows(X, window_size, target_window)
print("[OK] X_windows: {}, y_windows: {}".format(X_win.shape, y_win.shape))

# Split treino/teste
num_train = int(len(X_win) * (1 - test_split))
X_train = X_win[:num_train]
y_train = y_win[:num_train]
X_test = X_win[num_train:]
y_test = y_win[num_train:]

print("    Treino: {}, Teste: {}".format(len(X_train), len(X_test)))

print("\n[4] Inicializando modelo...")
# Hiperparametros identicos ao original
num_nodes = X.shape[1]
in_channels = X.shape[2]
hidden_channels = 64
out_channels = in_channels

model = STGCN_Cpraio(
    num_nodes=num_nodes,
    in_channels=in_channels,
    hidden_channels=hidden_channels,
    out_channels=out_channels,
    dropout=0.3
)
print("[OK] Modelo criado com {} nodes".format(num_nodes))
print("    Hidden channels: {}".format(hidden_channels))
print("    Out channels: {}".format(out_channels))

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("    Device: {}".format(device))

model = model.to(device)
X_train = X_train.to(device)
y_train = y_train.to(device)
X_test = X_test.to(device)
y_test = y_test.to(device)
edge_index = edge_index.to(device)

print("\n[5] Iniciando treinamento...")

# Configuracoes de treino
learning_rate = 0.001
weight_decay = 1e-4
epochs = 200
patience = 25
best_val_loss = float('inf')
patience_counter = 0

optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
loss_fn = nn.MSELoss()

losses_train = []
losses_test = []
best_epoch = 0

for epoch in range(epochs):
    # Treino
    model.train()
    train_loss = 0.0
    
    for i in range(0, len(X_train), batch_size):
        batch_X = X_train[i:i + batch_size]
        batch_y = y_train[i:i + batch_size]
        
        optimizer.zero_grad()
        
        # Forward
        y_pred = model(batch_X, edge_index)
        loss = loss_fn(y_pred, batch_y)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * len(batch_X)
    
    train_loss /= len(X_train)
    losses_train.append(train_loss)
    
    # Teste
    model.eval()
    with torch.no_grad():
        y_pred_test = model(X_test, edge_index)
        test_loss = loss_fn(y_pred_test, y_test).item()
    
    losses_test.append(test_loss)
    
    # Early stopping
    if test_loss < best_val_loss:
        best_val_loss = test_loss
        best_epoch = epoch
        patience_counter = 0
        
        # Salvar melhor modelo
        torch.save({
            'model_state': model.state_dict(),
            'epoch': epoch,
            'loss': test_loss,
            'config': {
                'num_nodes': num_nodes,
                'in_channels': in_channels,
                'hidden_channels': hidden_channels,
                'out_channels': out_channels
            }
        }, 'data/tensors/model_stgcn_completo.pth')
    else:
        patience_counter += 1
    
    if (epoch + 1) % 10 == 0:
        print("Epoch {}/{}: train_loss={:.4f}, test_loss={:.4f}".format(
            epoch + 1, epochs, train_loss, test_loss))
    
    if patience_counter >= patience:
        print("Early stopping no epoch {} (patience={})".format(epoch + 1, patience))
        break

print("\n" + "="*80)
print("RESUMO DO TREINAMENTO")
print("="*80)
print("""
Configuracao:
  - Modelo: ST-GCN (LSTM 64 + 2 GCN layers)
  - Nodes: 389 (388 bairros + 1 raiz)
  - Learning rate: {}
  - Batch size: {}
  - Epochs: {}/{}

Resultados:
  - Melhor epoch: {}
  - Melhor loss: {:.4f}
  - Epochs treinados: {}

Arquivo:
  - Model: data/tensors/model_stgcn_completo.pth
  - Arquitetura: {}-node ST-GCN com {} hidden units
Comparacao com original (319 nodes):
  - Loss original: ~0.0756
  - Loss novo: {:.4f}
  - Melhoria: {:.1f}%

Proximo passo:
  - Revalidar modelo contra RAIO 2025
  - Comparar correlacoes antes/depois
""".format(
    learning_rate,
    batch_size,
    epoch + 1,
    epochs,
    best_epoch + 1,
    best_val_loss,
    epoch + 1,
    num_nodes,
    hidden_channels,
    best_val_loss,
    (0.0756 - best_val_loss) / 0.0756 * 100
))

print("[CONCLUIDO] Treinamento finalizado!")
