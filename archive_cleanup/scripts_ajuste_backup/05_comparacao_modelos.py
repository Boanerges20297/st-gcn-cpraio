#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Comparacao: Resultados dos modelos
"""

import pandas as pd
import json
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'

print("\n" + "="*80)
print("RESUMO: MODELO COM JANELA 180 DIAS")
print("="*80)

# LOAD DADOS
print("\n[1] Carregando dados de validacao 2025...")
try:
    df_val = pd.read_parquet(DATA_DIR / 'processed' / 'dataset_validacao_janela180d.parquet')
    print("[OK] Dataset carregado")
except Exception as e:
    print(f"[ERROR] {e}")
    exit(1)

print("\n[2] Top 15 bairros criticos (2025 - ultimos 180 dias)...")

# Última data
data_final = df_val['data'].max()
df_final = df_val[df_val['data'] == data_final]

# Top 15
top_15 = df_final.nlargest(15, 'criticidade')

for i, (_, row) in enumerate(top_15.iterrows(), 1):
    cvli = row['cvli_180d']
    crit = row['criticidade']
    print(f"{i:2d}. {row['bairro']:25s} | CVLI 180d: {cvli:3.0f} | Criticidade: {crit:.4f}")

print("\n[3] Bairros com BAIXA criticidade...")
bottom_10 = df_final.nsmallest(10, 'criticidade')
print("\n   Bottom 10 (menor criticidade):")
for i, (_, row) in enumerate(bottom_10.iterrows(), 1):
    cvli = row['cvli_180d']
    crit = row['criticidade']
    print(f"   {i:2d}. {row['bairro']:25s} | CVLI 180d: {cvli:3.0f} | Criticidade: {crit:.4f}")

print("\n[4] Estatisticas...")

try:
    with open(DATA_DIR / 'processed' / 'metadata_janela180d.json', 'r', encoding='utf-8') as f:
        meta = json.load(f)
    
    print(f"\n   Metodo: {meta.get('metodo', 'N/A')}")
    print(f"   Total CVLI no periodo: {meta.get('validacao_cvli', 'N/A')}")
    print(f"   Bairros: {meta.get('bairros', 'N/A')}")
    print(f"   Criticidade Media Treino: {meta.get('criticidade_mean_treino', 'N/A'):.3f}")
    print(f"   Criticidade Media Validacao: {meta.get('criticidade_mean_val', 'N/A'):.3f}")
except Exception as e:
    print(f"   [ERROR] {e}")

# Verificar se Praia de Iracema está na lista
print("\n[5] Praia de Iracema especificamente...")
iracema = df_final[df_final['bairro'].str.contains('IRACEMA', case=False, na=False)]
if len(iracema) > 0:
    for _, row in iracema.iterrows():
        cvli = row['cvli_180d']
        crit = row['criticidade']
        rank = len(df_final[df_final['criticidade'] > crit])
        print(f"\n   Bairro: {row['bairro']}")
        print(f"   CVLI (180d): {cvli:.0f}")
        print(f"   Criticidade: {crit:.4f}")
        print(f"   Ranking: #{rank + 1} entre {len(df_final)}")
else:
    print("   [AVISO] Praia de Iracema nao encontrado")

print("\n" + "="*80)
print("[SUCCESS] ANALISE COMPLETA")
print("="*80)
