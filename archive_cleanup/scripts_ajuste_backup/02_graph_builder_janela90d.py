#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Graph Builder - Janela 90 dias
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import json

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'
TENSORS_DIR = DATA_DIR / 'tensors'

print("\n" + "="*80)
print("GRAPH BUILDER - JANELA 90 DIAS")
print("="*80)

# LOAD
print("\n[1] Carregando datasets...")
df_treino = pd.read_parquet(DATA_DIR / 'processed' / 'dataset_treino_janela90d.parquet')
df_validacao = pd.read_parquet(DATA_DIR / 'processed' / 'dataset_validacao_janela90d.parquet')

print(f"[OK] Treino: {len(df_treino):,}")
print(f"[OK] Validacao: {len(df_validacao):,}")

df_combined = pd.concat([df_treino, df_validacao]).reset_index(drop=True)

# LOAD GRAPH
print("\n[2] Carregando grafo...")
adj = np.load(DATA_DIR / 'tensors' / 'adjacency_matrix.npy')
edge_index = torch.tensor(np.argwhere(adj > 0).T, dtype=torch.long)

num_nodes = adj.shape[0]
print(f"[OK] Nos: {num_nodes}, Arestas: {edge_index.shape[1]}")

# MAPA
print("\n[3] Mapeando bairros...")
bairros_unicos = sorted(df_combined['bairro'].unique())
bairro_to_node = {bairro: idx for idx, bairro in enumerate(bairros_unicos[:num_nodes])}
print(f"[OK] {len(bairro_to_node)} bairros")

# TENSOR
print("\n[4] Construindo tensor...")
dates = sorted(df_combined['data'].unique())

tensor = np.zeros((len(dates), num_nodes, 1), dtype=np.float32)

for d_idx, date in enumerate(dates):
    if d_idx % 200 == 0:
        print(f"    Dia {d_idx}/{len(dates)}...")
    
    df_day = df_combined[df_combined['data'] == date]
    
    for _, row in df_day.iterrows():
        bairro = row['bairro']
        node_id = bairro_to_node.get(bairro)
        
        if node_id is not None and 0 <= node_id < num_nodes:
            crit = float(row['criticidade'])
            tensor[d_idx, node_id, 0] = crit

print(f"[OK] Shape: {tensor.shape}")

# SALVAR
print("\n[5] Salvando tensor...")
TENSORS_DIR.mkdir(parents=True, exist_ok=True)

tensor_path = TENSORS_DIR / 'dataset_criticidade_janela90d.pt'
torch.save(torch.from_numpy(tensor), tensor_path)
print(f"[OK] {tensor_path}")

# METADATA
print("\n[6] Salvando metadata...")

metadata = {
    'metodo': 'Janela 90 Dias',
    'tensor_shape': list(tensor.shape),
    'num_nodes': num_nodes,
    'num_features': 1,
    'num_days': len(dates),
    'valor_min': float(tensor.min()),
    'valor_max': float(tensor.max()),
    'valor_mean': float(tensor.mean()),
    'valor_std': float(tensor.std())
}

with open(TENSORS_DIR / 'metadata_janela90d.json', 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=2)

print(f"[OK] Metadata salvo")

# STATS
print("\n[7] Estatisticas...")
print(f"   Min: {tensor.min():.4f}")
print(f"   Max: {tensor.max():.4f}")
print(f"   Mean: {tensor.mean():.4f}")
print(f"   Std: {tensor.std():.4f}")

print("\n" + "="*80)
print("[SUCCESS] GRAPH BUILDER CONCLUIDO (90D)")
print("="*80)
