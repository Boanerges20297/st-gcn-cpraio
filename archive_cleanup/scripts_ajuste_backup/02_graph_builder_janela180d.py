#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Graph Builder - Criticidade Janela 180 dias
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import json
from datetime import datetime

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'
TENSORS_DIR = DATA_DIR / 'tensors'

print("\n" + "="*80)
print("GRAPH BUILDER - JANELA 180 DIAS")
print("="*80)

# 1. LOAD DATA
print("\n[1] CARREGANDO DATASETS...")
df_treino = pd.read_parquet(DATA_DIR / 'processed' / 'dataset_treino_janela180d.parquet')
df_validacao = pd.read_parquet(DATA_DIR / 'processed' / 'dataset_validacao_janela180d.parquet')

print(f"✓ Treino: {len(df_treino):,}")
print(f"✓ Validação: {len(df_validacao):,}")

# Combinar para criar grafo completo
df_combined = pd.concat([df_treino, df_validacao]).reset_index(drop=True)

# 2. LOAD GRAPH STRUCTURE
print("\n[2] CARREGANDO ESTRUTURA DO GRAFO...")
geojson_capital = DATA_DIR / 'graph' / 'Fortaleza_Capital.geojson'
adj_path = DATA_DIR / 'tensors' / 'adjacency_matrix.npy'

# Usar adjacência existente
adj_matrix = np.load(adj_path)
num_nodes = adj_matrix.shape[0]

print(f"✓ Nós: {num_nodes}")
print(f"✓ Arestas: {adj_matrix.sum() // 2:.0f}")

# 3. CRIAR MAPA BAIRRO -> NODE
print("\n[3] MAPEANDO BAIRROS PARA NÓS...")

# Carregar mapping ou criar novo
meta_path = DATA_DIR / 'tensors' / 'metadata_cvli.json'
try:
    with open(meta_path, 'r', encoding='utf-8') as f:
        meta_old = json.load(f)
    bairro_to_node = meta_old.get('bairro_mapping', {})
except:
    bairro_to_node = {}

# Se vazio, criar novo mapeamento
if not bairro_to_node:
    print("  Criando novo mapeamento...")
    bairros_unicos = sorted(df_combined['bairro'].unique())
    bairro_to_node = {bairro: idx for idx, bairro in enumerate(bairros_unicos[:num_nodes])}

print(f"[OK] Mapeamento: {len(bairro_to_node)} bairros -> {num_nodes} nós")

# 4. CRIAR TENSOR
print("\n[4] CONSTRUINDO TENSOR...")

# Datas únicas ordenadas
dates = sorted(df_combined['data'].unique())
bairros = sorted(df_combined['bairro'].unique())

print(f"   {len(dates)} dias x {len(bairros)} bairros")

# Tensor: (dias, nós, features)
# Feature: criticidade_180d
tensor = np.zeros((len(dates), num_nodes, 1), dtype=np.float32)

for d_idx, date in enumerate(dates):
    if d_idx % 100 == 0:
        print(f"   Dia {d_idx+1}/{len(dates)}...")
    
    df_day = df_combined[df_combined['data'] == date]
    
    for _, row in df_day.iterrows():
        bairro = row['bairro']
        node_id = bairro_to_node.get(bairro)
        
        if node_id is not None and 0 <= node_id < num_nodes:
            # Criticidade normalizada (0-1)
            crit = float(row['criticidade'])
            tensor[d_idx, node_id, 0] = crit

print(f"✓ Tensor shape: {tensor.shape}")

# 5. SALVAR TENSOR
print("\n[5] SALVANDO TENSOR...")

TENSORS_DIR.mkdir(parents=True, exist_ok=True)

tensor_path = TENSORS_DIR / 'dataset_criticidade_janela180d.pt'
torch.save(torch.from_numpy(tensor), tensor_path)

print(f"✓ {tensor_path}")

# 6. SALVAR METADATA
print("\n[6] SALVANDO METADATA...")

metadata = {
    'criacao': datetime.now().isoformat(),
    'metodo': 'Criticidade Janela 180 Dias',
    'tensor_shape': list(tensor.shape),
    'num_nodes': num_nodes,
    'num_features': 1,
    'num_days': len(dates),
    'date_range': f"{dates[0].strftime('%Y-%m-%d')} a {dates[-1].strftime('%Y-%m-%d')}",
    'bairro_mapping': bairro_to_node,
    'valor_min': float(tensor.min()),
    'valor_max': float(tensor.max()),
    'valor_mean': float(tensor.mean()),
    'valor_std': float(tensor.std())
}

meta_path = TENSORS_DIR / 'metadata_janela180d.json'
with open(meta_path, 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

print(f"✓ {meta_path}")

# 7. ESTATÍSTICAS
print("\n[7] ESTATÍSTICAS DO TENSOR...")
print(f"   Min: {tensor.min():.4f}")
print(f"   Max: {tensor.max():.4f}")
print(f"   Mean: {tensor.mean():.4f}")
print(f"   Std: {tensor.std():.4f}")

# Nós com maior variância
node_vars = tensor.var(axis=0).squeeze()
top_var = np.argsort(node_vars)[-10:][::-1]

print(f"\n   Top 10 nós por variância:")
for rank, node_id in enumerate(top_var, 1):
    # Encontrar bairro
    bairro_name = [k for k, v in bairro_to_node.items() if v == node_id]
    bairro_name = bairro_name[0] if bairro_name else f"Node_{node_id}"
    print(f"   {rank}. {bairro_name:20s} (var={node_vars[node_id]:.6f})")

print("\n" + "="*80)
print("✅ GRAPH BUILDER CONCLUÍDO")
print("="*80)
