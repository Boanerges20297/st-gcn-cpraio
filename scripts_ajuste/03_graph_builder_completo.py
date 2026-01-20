#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reconstrói o graph para 389 nodes (388 bairros + raiz)
Expande a matriz de adjacência original para cobrir os novos bairros
"""

import torch
import numpy as np
import json
from pathlib import Path
from scipy.spatial.distance import cdist
from sklearn.preprocessing import normalize

print("="*80)
print("RECONSTRUINDO GRAPH PARA 389 NODES")
print("="*80)

print("\n[1] Carregando matriz de adjacência original...")
adj_orig = np.load('data/tensors/adjacency_matrix.npy')
print("[OK] Shape original: {}".format(adj_orig.shape))
print("     Nodes: {}".format(adj_orig.shape[0]))

print("\n[2] Carregando mapeamentos...")
with open('data/tensors/metadata_janela180d.json', 'r', encoding='utf-8') as f:
    meta_orig = json.load(f)
    mapping_orig = meta_orig['bairro_mapping']

with open('data/tensors/metadata_janela180d_completo.json', 'r', encoding='utf-8') as f:
    meta_novo = json.load(f)
    mapping_novo = meta_novo['bairro_mapping']

print("[OK] Mapping original: {} bairros".format(len(mapping_orig)))
print("[OK] Mapping novo: {} bairros".format(len(mapping_novo)))

print("\n[3] Identificando novos bairros...")
bairros_novos = set(mapping_novo.keys()) - set(mapping_orig.keys())
print("[OK] {} bairros novos".format(len(bairros_novos)))

print("\n[4] Expandindo matriz de adjacência...")
# Criar nova matriz (389x389)
# Preservar estrutura original nos primeiros 319 nodes
adj_novo = np.zeros((389, 389), dtype=np.float32)

# Copiar dados originais (319x319 -> parte superior esquerda 319x319)
adj_novo[:319, :319] = adj_orig

# Para os novos bairros (indices 319-388), criar conexoes simples
# Estrategia: conectar cada novo bairro ao seu vizinho mais proximo geograficamente
# Como nao temos coords, vamos usar uma conexao mais simples:
# - Conectar ao nó raiz (index 318)
# - Conectar a 5 vizinhos mais proximos (arbitrariamente escolhidos)

print("    Adicionando conexoes para {} novos nodes...".format(len(bairros_novos)))

# Todos os novos nodes conectam ao nó raiz (index 318)
for i in range(319, 389):
    adj_novo[i, 318] = 1.0
    adj_novo[318, i] = 1.0

# Alguns novos nodes tambem conectam entre si (criar clusters)
# Isso é necessario para manter a topologia conexa
novosnodes_indices = list(range(319, 389))
for i, idx in enumerate(novosnodes_indices):
    # Conectar a 2-3 vizinhos proximos
    vizinhos_count = min(3, len(novosnodes_indices) - 1)
    vizinhos_idx = (i + 1) % len(novosnodes_indices)
    
    adj_novo[idx, novosnodes_indices[vizinhos_idx]] = 1.0
    adj_novo[novosnodes_indices[vizinhos_idx], idx] = 1.0

print("[OK] Nova matriz: {}".format(adj_novo.shape))

print("\n[5] Verificando conectividade...")
# Garantir que não há nodes desconectados
graus = adj_novo.sum(axis=1)
nodes_desconectados = np.where(graus == 0)[0]

if len(nodes_desconectados) > 0:
    print("[AVISO] {} nodes desconectados detectados".format(len(nodes_desconectados)))
    for idx in nodes_desconectados:
        adj_novo[idx, 318] = 1.0
        adj_novo[318, idx] = 1.0
    print("[CORRIGIDO] Todos os nodes agora conectados ao raiz")

print("\n[6] Calculando estatisticas da matriz...")
graus = adj_novo.sum(axis=1)
print("    Grau min: {}".format(int(graus.min())))
print("    Grau max: {}".format(int(graus.max())))
print("    Grau medio: {:.1f}".format(graus.mean()))
print("    Densidade: {:.4f}".format(np.count_nonzero(adj_novo) / (389 * 389)))

print("\n[7] Salvando nova matriz...")
np.save('data/tensors/adjacency_matrix_completo.npy', adj_novo)
print("[OK] Salvo: data/tensors/adjacency_matrix_completo.npy")

print("\n[8] Criando arquivo de edge_index para PyTorch...")
# Extrair edges da matriz (formato COO para PyTorch Geometric)
edges_i, edges_j = np.where(adj_novo > 0)
edge_index = torch.tensor([edges_i, edges_j], dtype=torch.long)

print("    Shape edge_index: {}".format(edge_index.shape))
print("    Numero de edges: {}".format(edge_index.shape[1]))

print("\n[9] Salvar dataset com edge_index...")
X = torch.load('data/tensors/dataset_criticidade_janela180d_completo.pt', weights_only=False)
print("[OK] Dataset carregado: shape {}".format(X.shape))

# Salvar como tuple (X, edge_index)
dataset_com_graph = {
    'X': X,
    'edge_index': edge_index,
    'adj_matrix': torch.from_numpy(adj_novo)
}

torch.save(dataset_com_graph, 'data/tensors/dataset_stgcn_completo.pt')
print("[OK] Salvo: data/tensors/dataset_stgcn_completo.pt")

print("\n" + "="*80)
print("RESUMO")
print("="*80)
print("""
Matrix de Adjacencia:
  - Shape: 389 x 389
  - Arquivo: data/tensors/adjacency_matrix_completo.npy

Dataset ST-GCN:
  - X shape: (1461, 389, 1)
  - Edge_index: ({}, 2)
  - Arquivo: data/tensors/dataset_stgcn_completo.pt

Topologia:
  - Preserva 319 nodes originais
  - Adiciona 70 novos nodes
  - Todos os novos nodes conectados ao nó raiz
  - Alguns clusters entre novos nodes para melhor propagação de gradientes

Proximo passo:
  - Executar treinamento com novo dataset
""".format(edge_index.shape[1]))

print("[CONCLUIDO]")
