#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Versão simplificada do treinamento: usar tensor criticidade direto sem windows
Treina usando o Dataset em (1461, 389, 1) e prediz próximos 30 dias
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
print("TREINAR ST-GCN COM COBERTURA COMPLETA (VERSAO SIMPLIFICADA)")
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
X = torch.load('data/tensors/dataset_criticidade_janela180d_completo.pt', weights_only=False)

print("[OK] X shape: {}".format(X.shape))
# X shape: (1461, 389, 1)

print("\n[3] Carregando graph...")
dataset = torch.load('data/tensors/dataset_stgcn_completo.pt', weights_only=False)
edge_index = dataset['edge_index']
print("[OK] Edge_index shape: {}".format(edge_index.shape))

print("\n[4] Preparando dados de treino...")
# Simples: usar primeiros 1100 dias para treino, ultimos 361 para teste
# Vamos fazer previsao de 30 dias a frente

def create_sequences(data, lookback=30, forecast=30):
    X_seq, y_seq = [], []
    for i in range(len(data) - lookback - forecast):
        X_seq.append(data[i:i+lookback])
        y_seq.append(data[i+lookback:i+lookback+forecast].mean(dim=0))
    return torch.stack(X_seq), torch.stack(y_seq)

lookback = 30
forecast = 30

print("    Criando sequences (lookback={}, forecast={})...".format(lookback, forecast))
X_seq, y_seq = create_sequences(X, lookback=lookback, forecast=forecast)
print("[OK] X_seq: {}, y_seq: {}".format(X_seq.shape, y_seq.shape))

# Split
num_train = int(0.8 * len(X_seq))
X_train = X_seq[:num_train]
y_train = y_seq[:num_train]
X_test = X_seq[num_train:]
y_test = y_seq[num_train:]

print("    Treino: {}, Teste: {}".format(len(X_train), len(X_test)))

print("\n[5] Inicializando modelo...")
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

print("[OK] Modelo criado: {} nodes, {} hidden".format(num_nodes, hidden_channels))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("    Device: {}".format(device))

model = model.to(device)
X_train = X_train.to(device)
y_train = y_train.to(device)
X_test = X_test.to(device)
y_test = y_test.to(device)
edge_index = edge_index.to(device)

print("\n[6] Iniciando treinamento...")

# Treino
learning_rate = 0.001
weight_decay = 1e-4
epochs = 100
batch_size = 16
patience = 15

optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
loss_fn = nn.MSELoss()

best_val_loss = float('inf')
patience_counter = 0
best_epoch = 0

for epoch in range(epochs):
    # Treino
    model.train()
    train_loss = 0.0
    num_batches = 0
    
    for i in range(0, len(X_train), batch_size):
        batch_X = X_train[i:i + batch_size]  # (batch, 30, 389, 1)
        batch_y = y_train[i:i + batch_size]  # (batch, 389, 1)
        
        optimizer.zero_grad()
        
        # Forward: processar cada timestep
        y_pred_list = []
        for t in range(batch_X.shape[1]):  # Iterar sobre 30 timesteps
            x_t = batch_X[:, t:t+1, :, :]  # (batch, 1, 389, 1)
            x_t = x_t.squeeze(1)  # (batch, 389, 1)
            
            y_pred = model(x_t, edge_index)
            y_pred_list.append(y_pred)
        
        # Average prediction
        y_pred = torch.stack(y_pred_list, dim=1).mean(dim=1)  # (batch, 389, 1)
        
        loss = loss_fn(y_pred, batch_y)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        num_batches += 1
    
    train_loss /= num_batches
    
    # Teste
    model.eval()
    with torch.no_grad():
        test_loss = 0.0
        test_batches = 0
        
        for i in range(0, len(X_test), batch_size):
            batch_X = X_test[i:i + batch_size]
            batch_y = y_test[i:i + batch_size]
            
            y_pred_list = []
            for t in range(batch_X.shape[1]):
                x_t = batch_X[:, t:t+1, :, :]
                x_t = x_t.squeeze(1)
                y_pred = model(x_t, edge_index)
                y_pred_list.append(y_pred)
            
            y_pred = torch.stack(y_pred_list, dim=1).mean(dim=1)
            loss = loss_fn(y_pred, batch_y)
            test_loss += loss.item()
            test_batches += 1
        
        test_loss /= test_batches
    
    # Early stopping
    if test_loss < best_val_loss:
        best_val_loss = test_loss
        best_epoch = epoch
        patience_counter = 0
        
        # Salvar modelo
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
    
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print("Epoch {}/{}: train_loss={:.4f}, test_loss={:.4f}".format(
            epoch + 1, epochs, train_loss, test_loss))
    
    if patience_counter >= patience:
        print("Early stopping no epoch {} (patience={})".format(epoch + 1, patience))
        break

print("\n" + "="*80)
print("RESUMO")
print("="*80)
print("""
Configuracao:
  - Modelo: ST-GCN (LSTM 64 + 2 GCN layers)
  - Nodes: 389 (388 bairros + 1 raiz)
  - Lookback: 30 dias
  - Forecast: 30 dias
  - Learning rate: {}
  - Batch size: {}

Resultados:
  - Melhor epoch: {}
  - Melhor test_loss: {:.4f}
  - Epochs treinados: {}

Arquivo:
  - Model: data/tensors/model_stgcn_completo.pth

Proximo passo:
  - Revalidar modelo contra RAIO 2025
  - Comparar correlacoes antes/depois
""".format(
    learning_rate,
    batch_size,
    best_epoch + 1,
    best_val_loss,
    epoch + 1
))

print("[CONCLUIDO]")
