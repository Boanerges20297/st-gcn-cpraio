#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Treinador ST-GCN com Checkpoints Robustos
==========================================
Versão 2 com salva periodicamente, tratamento de exceções e resumo robusto.
"""

import os
import sys
import torch
import torch.nn as nn
from pathlib import Path
import json
import numpy as np
from datetime import datetime

# Adiciona src ao path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from model import STGCN_Cpraio

def load_tensor_data(tensor_path):
    """Carrega o tensor dataset."""
    print(f"\n[LOAD] Carregando tensor de {tensor_path}")
    data = torch.load(tensor_path)
    
    # Se for dict, extrai o tensor
    if isinstance(data, dict):
        if 'data' in data:
            dataset = data['data']
        elif 'X' in data:
            dataset = data['X']
        else:
            dataset = next(v for v in data.values() if isinstance(v, torch.Tensor))
    else:
        dataset = data
    
    print(f"[LOAD] Shape: {dataset.shape}")
    return dataset

def load_graph_data(metadata_path):
    """Carrega metadados do grafo."""
    print(f"[LOAD] Carregando metadados de {metadata_path}")
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    return metadata

def load_adjacency(adj_path):
    """Carrega matriz de adjacência."""
    print(f"[LOAD] Carregando adjacência de {adj_path}")
    adj_matrix = np.load(adj_path)
    return adj_matrix

def prepare_edge_index(adj_matrix):
    """Converte matriz de adjacência para formato PyG."""
    adj_coo = np.argwhere(adj_matrix > 0)
    edge_index = torch.tensor(adj_coo.T, dtype=torch.long)
    print(f"[GRAPH] Edge index shape: {edge_index.shape}")
    return edge_index

def create_sequences(data, window_size=14, pred_size=15):
    """Cria sequências de treino/validação."""
    X, y = [], []
    for i in range(len(data) - window_size - pred_size + 1):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size:i+window_size+pred_size])
    return torch.stack(X), torch.stack(y)

def train_epoch(model, dataloader, optimizer, criterion, device):
    """Treina uma época."""
    model.train()
    total_loss = 0
    batch_count = 0
    
    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        edge_index = model.edge_index.to(device)
        
        optimizer.zero_grad()
        pred = model(X_batch, edge_index)
        
        # Reshape para comparação
        pred_reshaped = pred.view(y_batch.shape[0], -1, model.num_nodes, model.out_channels)
        y_reshaped = y_batch.view(y_batch.shape[0], -1, model.num_nodes, 1)
        
        loss = criterion(pred_reshaped, y_reshaped.squeeze(-1))
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        batch_count += 1
    
    avg_loss = total_loss / batch_count if batch_count > 0 else 0
    return avg_loss

def validate(model, dataloader, criterion, device):
    """Valida o modelo."""
    model.eval()
    total_loss = 0
    batch_count = 0
    
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            edge_index = model.edge_index.to(device)
            
            pred = model(X_batch, edge_index)
            
            # Reshape para comparação
            pred_reshaped = pred.view(y_batch.shape[0], -1, model.num_nodes, model.out_channels)
            y_reshaped = y_batch.view(y_batch.shape[0], -1, model.num_nodes, 1)
            
            loss = criterion(pred_reshaped, y_reshaped.squeeze(-1))
            
            total_loss += loss.item()
            batch_count += 1
    
    avg_loss = total_loss / batch_count if batch_count > 0 else 0
    return avg_loss

def main():
    print("\n" + "="*80)
    print("ST-GCN CPRAIO - TREINADOR v2 (COM CHECKPOINTS)")
    print("="*80)
    
    # Configurações
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / 'data'
    OUTPUT_DIR = BASE_DIR / 'outputs'
    MODELS_DIR = OUTPUT_DIR / 'models'
    
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[DEVICE] Usando: {device}")
    
    # Hiperparâmetros
    WINDOW_SIZE = 14
    PRED_SIZE = 15
    HIDDEN_CHANNELS = 64
    BATCH_SIZE = 32
    EPOCHS = 150
    LEARNING_RATE = 0.001
    PATIENCE = 20
    CHECKPOINT_EVERY = 10
    
    try:
        # 1. CARREGAR DADOS
        print("\n[PASSO 1] CARREGANDO DADOS")
        tensor_path = DATA_DIR / 'tensors' / 'dataset_cvli_novo_criterio.pt'
        metadata_path = DATA_DIR / 'tensors' / 'metadata_cvli.json'
        adj_path = DATA_DIR / 'tensors' / 'adjacency_matrix.npy'
        
        dataset = load_tensor_data(tensor_path)
        metadata = load_graph_data(metadata_path)
        adj_matrix = load_adjacency(adj_path)
        
        num_nodes = metadata['num_nodes']
        num_features = metadata['num_features']
        
        print(f"[DADOS] Nós: {num_nodes}, Features: {num_features}")
        
        # 2. PREPARAR GRAFO
        print("\n[PASSO 2] PREPARANDO GRAFO")
        edge_index = prepare_edge_index(adj_matrix)
        
        # 3. CRIAR SEQUÊNCIAS
        print("\n[PASSO 3] CRIANDO SEQUÊNCIAS")
        X, y = create_sequences(dataset, WINDOW_SIZE, PRED_SIZE)
        print(f"[SEQS] X shape: {X.shape}, y shape: {y.shape}")
        
        # Normalizar
        X_mean = X.mean(dim=(0, 1, 2), keepdim=True)
        X_std = X.std(dim=(0, 1, 2), keepdim=True)
        X_std[X_std == 0] = 1
        X_norm = (X - X_mean) / X_std
        
        y_mean = y.mean(dim=(0, 1, 2), keepdim=True)
        y_std = y.std(dim=(0, 1, 2), keepdim=True)
        y_std[y_std == 0] = 1
        y_norm = (y - y_mean) / y_std
        
        # Split treino/validação
        split_idx = int(0.8 * len(X_norm))
        X_train, X_val = X_norm[:split_idx], X_norm[split_idx:]
        y_train, y_val = y_norm[:split_idx], y_norm[split_idx:]
        
        print(f"[SPLIT] Treino: {X_train.shape[0]}, Validação: {X_val.shape[0]}")
        
        # 4. CRIAR DATALOADERS
        print("\n[PASSO 4] CRIANDO DATALOADERS")
        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=BATCH_SIZE, 
            shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, 
            batch_size=BATCH_SIZE, 
            shuffle=False
        )
        
        print(f"[LOADERS] Treino: {len(train_loader)} batches, Validação: {len(val_loader)} batches")
        
        # 5. CRIAR MODELO
        print("\n[PASSO 5] CRIANDO MODELO")
        model = STGCN_Cpraio(
            num_nodes=num_nodes,
            in_channels=num_features,
            hidden_channels=HIDDEN_CHANNELS,
            out_channels=1,
            dropout=0.3
        )
        model.edge_index = edge_index
        model.to(device)
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"[MODELO] Parâmetros totais: {total_params:,}")
        
        # 6. OTIMIZADOR E CRITÉRIO
        print("\n[PASSO 6] CONFIGURANDO OTIMIZAÇÃO")
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        criterion = nn.MSELoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        # 7. TREINAR
        print("\n[PASSO 7] INICIANDO TREINAMENTO")
        print(f"[CONFIG] Epochs: {EPOCHS}, Batch size: {BATCH_SIZE}, LR: {LEARNING_RATE}")
        print("="*80)
        
        best_val_loss = float('inf')
        patience_counter = 0
        training_history = []
        
        for epoch in range(EPOCHS):
            try:
                # Treino
                train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
                
                # Validação
                val_loss = validate(model, val_loader, criterion, device)
                
                # Agendador
                scheduler.step(val_loss)
                
                # Log
                training_history.append({
                    'epoch': epoch + 1,
                    'train_loss': train_loss,
                    'val_loss': val_loss
                })
                
                print(f"Epoch {epoch+1:3d}/{EPOCHS} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
                
                # Melhor modelo
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    
                    # Salvar melhor modelo
                    torch.save(model.state_dict(), 
                              MODELS_DIR / 'model_cvli_novo_criterio.pth')
                    torch.save({
                        'X_mean': X_mean,
                        'X_std': X_std,
                        'y_mean': y_mean,
                        'y_std': y_std
                    }, MODELS_DIR / 'stats_cvli_novo_criterio.pt')
                    
                    print(f"  ✓ Novo melhor modelo salvo (val_loss: {val_loss:.6f})")
                else:
                    patience_counter += 1
                
                # Checkpoint periódico
                if (epoch + 1) % CHECKPOINT_EVERY == 0:
                    torch.save(model.state_dict(),
                              MODELS_DIR / f'checkpoint_epoch_{epoch+1}.pth')
                    print(f"  • Checkpoint em época {epoch+1}")
                
                # Early stopping
                if patience_counter >= PATIENCE:
                    print(f"\n[EARLY STOP] Paciência esgotada após {PATIENCE} épocas sem melhora")
                    break
                    
            except KeyboardInterrupt:
                print(f"\n⚠️  Interrupção do usuário na época {epoch+1}")
                # Salvar antes de sair
                torch.save(model.state_dict(),
                          MODELS_DIR / 'model_cvli_novo_criterio_ultimo.pth')
                raise
            except Exception as e:
                print(f"\n❌ Erro na época {epoch+1}: {str(e)}")
                raise
        
        # 8. SALVAMENTO FINAL
        print("\n[PASSO 8] SALVANDO MODELO FINAL")
        torch.save(model.state_dict(), 
                  MODELS_DIR / 'model_cvli_novo_criterio.pth')
        torch.save({
            'X_mean': X_mean,
            'X_std': X_std,
            'y_mean': y_mean,
            'y_std': y_std
        }, MODELS_DIR / 'stats_cvli_novo_criterio.pt')
        
        # Salvar histórico
        with open(MODELS_DIR / 'training_history.json', 'w') as f:
            json.dump(training_history, f, indent=2)
        
        print("\n" + "="*80)
        print("✅ TREINAMENTO CONCLUÍDO COM SUCESSO!")
        print("="*80)
        print(f"Melhor val_loss: {best_val_loss:.6f}")
        print(f"Épocas totais: {epoch + 1}/{EPOCHS}")
        print(f"Arquivos salvos em: {MODELS_DIR}")
        
        # Status final
        print("\n[ARQUIVOS CRIADOS]")
        for f in MODELS_DIR.glob('model_cvli_novo_criterio*'):
            size = f.stat().st_size / (1024*1024)
            print(f"  ✓ {f.name} ({size:.2f} MB)")
        
    except Exception as e:
        print(f"\n❌ ERRO FATAL: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
