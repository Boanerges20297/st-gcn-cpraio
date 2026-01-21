#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Expandir dataset_criticidade_janela180d.pt para cobrir 388 bairros
Adiciona 70 novos bairros com criticidade zerada (para preencher com dados do ETL)
"""

import torch
import json
import numpy as np
from pathlib import Path

print("="*80)
print("EXPANDINDO DATASET JANELA 180d PARA COBERTURA COMPLETA")
print("="*80)

print("\n[1] Carregando dataset original...")
dataset_path = Path("data/tensors/dataset_criticidade_janela180d.pt")
X = torch.load(dataset_path, weights_only=False)

print("[OK] Dataset original carregado")
print("    X shape: {}".format(X.shape))

num_dias = X.shape[0]
num_bairros_orig = X.shape[1]
num_features = X.shape[2]

print("    Dias: {}".format(num_dias))
print("    Bairros originais: {}".format(num_bairros_orig))
print("    Features: {}".format(num_features))

print("\n[2] Carregando novo mapeamento...")
with open('data/tensors/metadata_janela180d_completo.json', 'r', encoding='utf-8') as f:
    metadata = json.load(f)

novo_num_bairros = metadata['num_bairros']
print("[OK] Novo número de bairros: {}".format(novo_num_bairros))

print("\n[3] Expandindo tensor...")
# Dataset original tem 319 nodes (318 bairros + 1 raiz)
# Novo dataset terá 389 nodes (388 bairros + 1 raiz)
num_nodes_orig = X.shape[1]
num_nodes_novo = novo_num_bairros + 1  # +1 para nó raiz

print("[INFO] Nodes originais: {}".format(num_nodes_orig))
print("[INFO] Nodes novos: {}".format(num_nodes_novo))

# Criar tensor expandido (dias, 389, 1)
X_expandido = torch.zeros((num_dias, num_nodes_novo, num_features), dtype=X.dtype)

# Copiar dados originais
X_expandido[:, :num_nodes_orig, :] = X

print("[OK] Tensor expandido: {}".format(X_expandido.shape))
print("    Dados originais copiados para os primeiros {} nodes".format(num_nodes_orig))
print("    Novos bairros preenchidos com zeros (serao atualizados pelo ETL)")

print("\n[4] Salvando dataset expandido...")
novo_dataset_path = Path("data/tensors/dataset_criticidade_janela180d_completo.pt")

torch.save(X_expandido, novo_dataset_path)
print("[OK] Salvo em: {}".format(novo_dataset_path))
print("    Size: {:.2f} MB".format(novo_dataset_path.stat().st_size / 1024 / 1024))

print("\n" + "="*80)
print("RESUMO")
print("="*80)
print("""
Dataset Original:
  - Shape: (1461, 319, 1)
  - Arquivo: data/tensors/dataset_criticidade_janela180d.pt
  - 318 bairros + 1 nó raiz

Dataset Expandido:
  - Shape: (1461, 389, 1)
  - Arquivo: data/tensors/dataset_criticidade_janela180d_completo.pt
  - 388 bairros + 1 nó raiz
  - Adicionados 70 novos bairros (preenchidos com zeros)
  - Dados originais preservados nos primeiros 319 nodes

Proximo passo:
  1. Calcular dados dos novos bairros (ETL com RAIO 2025)
  2. Reconstruir graph com nova topologia
  3. Retreinar modelo
""")

print("[CONCLUIDO]")
