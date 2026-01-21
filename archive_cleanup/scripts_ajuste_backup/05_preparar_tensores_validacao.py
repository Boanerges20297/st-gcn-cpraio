#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calcular criticidade final com cobertura completa
para revalidacao contra RAIO 2025
Usar tensor expandido com dados dos 70 novos bairros
"""

import torch
import json
import numpy as np
from pathlib import Path

print("="*80)
print("PREPARAR TENSOR DE CRITICIDADE FINAL (389 NODES)")
print("="*80)

print("\n[1] Carregando tensor expandido...")
X = torch.load('data/tensors/dataset_criticidade_janela180d_completo.pt', weights_only=False)
print("[OK] Shape: {}".format(X.shape))

print("\n[2] Carregando metadata completo...")
with open('data/tensors/metadata_janela180d_completo.json', 'r', encoding='utf-8') as f:
    metadata = json.load(f)

print("[OK] {} bairros no mapeamento".format(len(metadata['bairro_mapping'])))

print("\n[3] Calculando estatisticas...")
print("    Min: {:.4f}".format(X.min().item()))
print("    Max: {:.4f}".format(X.max().item()))
print("    Mean: {:.4f}".format(X.mean().item()))
print("    Std: {:.4f}".format(X.std().item()))

# Verificar sparsidade
non_zero = (X > 0).sum().item()
total = X.numel()
sparsidade = (1 - non_zero / total) * 100
print("    Sparsidade: {:.1f}%".format(sparsidade))

print("\n[4] Verificar cobertura...")
# Verificar quais bairros tem dados
bairros_com_dados = []
bairros_sem_dados = []

for bairro, idx in metadata['bairro_mapping'].items():
    dados = X[:, idx, :].sum().item()
    if dados > 0:
        bairros_com_dados.append(bairro)
    else:
        bairros_sem_dados.append(bairro)

print("    Bairros com dados: {}".format(len(bairros_com_dados)))
print("    Bairros sem dados: {}".format(len(bairros_sem_dados)))

if len(bairros_sem_dados) > 0:
    print("    Amostra de bairros sem dados (primeiros 5):")
    for b in bairros_sem_dados[:5]:
        print("      - {}".format(b))

print("\n[5] Salvar tensor de criticidade final...")
output_path = Path('data/tensors/criticidade_final_completo.pt')
torch.save(X, output_path)
print("[OK] Salvo: {}".format(output_path))

print("\n[6] Gerar arquivo de validacao...")
val_info = {
    'tensor_shape': list(X.shape),
    'data_range': ['2022-01-01', '2025-12-31'],
    'num_dias': X.shape[0],
    'num_nodes': X.shape[1],
    'num_features': X.shape[2],
    'bairros_com_dados': len(bairros_com_dados),
    'bairros_sem_dados': len(bairros_sem_dados),
    'sparsidade_percent': sparsidade,
    'min': X.min().item(),
    'max': X.max().item(),
    'mean': X.mean().item(),
    'std': X.std().item()
}

info_path = Path('outputs/info_tensor_completo.json')
with open(info_path, 'w', encoding='utf-8') as f:
    json.dump(val_info, f, indent=2, ensure_ascii=False)

print("[OK] Info salva: {}".format(info_path))

print("\n" + "="*80)
print("RESUMO")
print("="*80)
print("""
Tensor de Criticidade Final:
  - Shape: 1461 dias x 389 nodes x 1 feature
  - Arquivo: data/tensors/criticidade_final_completo.pt
  - Metadata: data/tensors/metadata_janela180d_completo.json

Cobertura:
  - {} bairros com dados
  - {} bairros sem dados
  - Sparsidade: {:.1f}%

Valores:
  - Min: {:.4f}
  - Max: {:.4f}
  - Mean: {:.4f}
  - Std: {:.4f}

Este tensor pode ser usado para:
  1. Revalidacao contra RAIO 2025
  2. Analise de correlacao antes/depois
  3. Comparacao de cobertura geografica

Proximo passo:
  - Executar 09_validacao_raio_final_completo.py
  - Comparar r-value antes (0.348) vs depois
""".format(
    len(bairros_com_dados),
    len(bairros_sem_dados),
    sparsidade,
    X.min().item(),
    X.max().item(),
    X.mean().item(),
    X.std().item()
))

print("[CONCLUIDO]")
