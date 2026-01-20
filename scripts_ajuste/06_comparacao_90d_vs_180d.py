#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Comparacao: 180d vs 90d
"""

import pandas as pd
import json
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'

print("\n" + "="*80)
print("COMPARACAO: JANELA 180d vs 90d")
print("="*80)

# LOAD
print("\n[1] Carregando datasets...")
df_180d = pd.read_parquet(DATA_DIR / 'processed' / 'dataset_validacao_janela180d.parquet')
df_90d = pd.read_parquet(DATA_DIR / 'processed' / 'dataset_validacao_janela90d.parquet')

print("[OK] Datasets carregados")

# ULTIMAS DATAS
data_180d = df_180d['data'].max()
data_90d = df_90d['data'].max()

df_180d_final = df_180d[df_180d['data'] == data_180d]
df_90d_final = df_90d[df_90d['data'] == data_90d]

print("\n[2] TOP 15 - JANELA 180 DIAS")
top_180d = df_180d_final.nlargest(15, 'criticidade')

for i, (_, row) in enumerate(top_180d.iterrows(), 1):
    print(f"{i:2d}. {row['bairro']:25s} | CVLI: {row['cvli_180d']:3.0f} | Crit: {row['criticidade']:.4f}")

print("\n[3] TOP 15 - JANELA 90 DIAS")
top_90d = df_90d_final.nlargest(15, 'criticidade')

for i, (_, row) in enumerate(top_90d.iterrows(), 1):
    print(f"{i:2d}. {row['bairro']:25s} | CVLI: {row['cvli_90d']:3.0f} | Crit: {row['criticidade']:.4f}")

print("\n[4] MUDANCAS ENTRE TOP 15")

set_180d = set(top_180d['bairro'].values)
set_90d = set(top_90d['bairro'].values)

novos = set_90d - set_180d
saidos = set_180d - set_90d

if novos:
    print(f"\n   Entraram no 90d (saiam do 180d):")
    for b in novos:
        r_90 = len(df_90d_final[df_90d_final['criticidade'] > df_90d_final[df_90d_final['bairro'] == b]['criticidade'].values[0]])
        print(f"   + {b} (rank 90d: #{r_90 + 1})")

if saidos:
    print(f"\n   Saíram no 90d (estavam no 180d):")
    for b in saidos:
        r_180 = len(df_180d_final[df_180d_final['criticidade'] > df_180d_final[df_180d_final['bairro'] == b]['criticidade'].values[0]])
        print(f"   - {b} (era rank 180d: #{r_180 + 1})")

print("\n[5] PRAIA DE IRACEMA")

iracema_180d = df_180d_final[df_180d_final['bairro'] == 'IRACEMA']
iracema_90d = df_90d_final[df_90d_final['bairro'] == 'IRACEMA']
praia_180d = df_180d_final[df_180d_final['bairro'] == 'PRAIA DE IRACEMA']
praia_90d = df_90d_final[df_90d_final['bairro'] == 'PRAIA DE IRACEMA']

if len(iracema_180d) > 0 and len(iracema_90d) > 0:
    c_180 = iracema_180d['criticidade'].values[0]
    c_90 = iracema_90d['criticidade'].values[0]
    print(f"\n   IRACEMA:")
    print(f"   180d: criticidade {c_180:.4f}")
    print(f"   90d:  criticidade {c_90:.4f}")
    if c_90 > c_180:
        print(f"   Aumento: +{((c_90/c_180 - 1)*100):.1f}% [MAIS CRITICO]")
    else:
        print(f"   Reducao: {((1 - c_90/c_180)*100):.1f}%")

if len(praia_180d) > 0 and len(praia_90d) > 0:
    c_180 = praia_180d['criticidade'].values[0]
    c_90 = praia_90d['criticidade'].values[0]
    print(f"\n   PRAIA DE IRACEMA:")
    print(f"   180d: criticidade {c_180:.4f}")
    print(f"   90d:  criticidade {c_90:.4f}")
    if c_90 > c_180:
        print(f"   Aumento: +{((c_90/c_180 - 1)*100):.1f}% [MAIS CRITICO]")
    else:
        print(f"   Reducao: {((1 - c_90/c_180)*100):.1f}%")

print("\n[6] MODELO PERFORMANCE")

try:
    with open(DATA_DIR / 'outputs' / 'models' / 'training_history_janela180d.json', 'r') as f:
        hist_180d = json.load(f)
    best_180d = min([h['val'] for h in hist_180d])
    print(f"\n   180d - Best val loss: {best_180d:.6f}")
except:
    print(f"\n   180d - Nao encontrado")

try:
    with open(DATA_DIR / 'outputs' / 'models' / 'training_history_janela90d.json', 'r') as f:
        hist_90d = json.load(f)
    best_90d = min([h['val'] for h in hist_90d])
    print(f"   90d  - Best val loss: {best_90d:.6f}")
    
    if best_90d < best_180d:
        print(f"   Melhora: {((1 - best_90d/best_180d)*100):.1f}% [MELHOR]")
    else:
        print(f"   Piora: {((best_90d/best_180d - 1)*100):.1f}%")
except:
    print(f"   90d  - Nao encontrado")

print("\n" + "="*80)
print("[SUCCESS] COMPARACAO CONCLUIDA")
print("="*80)
