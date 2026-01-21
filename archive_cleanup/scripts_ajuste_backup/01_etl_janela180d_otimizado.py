#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ETL Otimizado - Criticidade Janela 180 dias
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import timedelta
import json

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'

print("\n" + "="*80)
print("ETL - CRITICIDADE JANELA 180D (OTIMIZADO)")
print("="*80)

# LOAD
print("\n[1] Carregando dados...")
enriquecido_path = DATA_DIR / 'processed' / 'dados_status_enriquecidos_com_bairros.parquet'
df = pd.read_parquet(enriquecido_path)
print(f"[OK] {len(df):,} registros")

# DATAS
print("\n[2] Processando datas...")
df['data'] = pd.to_datetime(df['data'], errors='coerce')
df = df.dropna(subset=['data', 'bairro'])

data_min, data_max = pd.Timestamp('2022-01-01'), pd.Timestamp('2025-12-31')
df = df[(df['data'] >= data_min) & (df['data'] <= data_max)]
print(f"[OK] {len(df):,} registros no período")

# CVLI/CVP
print("\n[3] Identificando CVLI...")
def is_cvli(row):
    for col in ['tipo_crime', 'tipo_evento', 'ais']:
        if pd.isna(row[col]):
            continue
        val = str(row[col]).upper().strip()
        if any(kw in val for kw in ['HOMICIDIO', 'LATROCINIO', 'LESAO', 'CVLI']):
            return True
    return False

df['is_cvli'] = df.apply(is_cvli, axis=1)
cvli_count = df['is_cvli'].sum()
print(f"[OK] CVLI: {cvli_count:,} | CVP: {len(df)-cvli_count:,}")

# CALCULAR COM ROLLING WINDOW
print("\n[4] Calculando criticidade (janela móvel 180d)...")

dates = pd.date_range('2022-01-01', '2025-12-31', freq='D')
bairros = sorted(df['bairro'].unique())

print(f"    {len(dates)} dias x {len(bairros)} bairros")

# Para cada dia e bairro, contar CVLI nos últimos 180 dias
results = []

for i, current_date in enumerate(dates):
    if i % 100 == 0:
        print(f"    Dia {i}/{len(dates)}...")
    
    window_start = current_date - timedelta(days=180)
    
    # Filtrar dados nesta janela
    window_df = df[(df['data'] >= window_start) & (df['data'] < current_date)]
    
    # Agrupar por bairro
    cvli_by_bairro = window_df[window_df['is_cvli']].groupby('bairro').size()
    cvp_by_bairro = window_df[~window_df['is_cvli']].groupby('bairro').size()
    
    for bairro in bairros:
        cvli = cvli_by_bairro.get(bairro, 0)
        cvp = cvp_by_bairro.get(bairro, 0)
        
        results.append({
            'data': current_date,
            'bairro': bairro,
            'cvli_180d': cvli,
            'cvp_180d': cvp
        })

df_ts = pd.DataFrame(results)
print(f"[OK] {len(df_ts):,} registros de série temporal")

# NORMALIZAR CRITICIDADE
print("\n[5] Normalizando criticidade...")
df_ts['criticidade'] = df_ts.groupby('data')['cvli_180d'].transform(lambda x: x.rank(pct=True))
print(f"[OK] Mean crit: {df_ts['criticidade'].mean():.3f}")

# TOP CRÍTICOS (últimos 180 dias)
print("\n[6] Top 10 críticos (últimos 180 dias)...")
data_final = df_ts['data'].max()
top_crit = df_ts[df_ts['data'] == data_final].nlargest(10, 'criticidade')
for _, row in top_crit.iterrows():
    print(f"    {row['bairro']:20s} | CVLI: {row['cvli_180d']:3.0f} | Crit: {row['criticidade']:.3f}")

# SPLIT
print("\n[7] Criando splits...")
split_treino = pd.Timestamp('2024-12-31')
split_validacao = pd.Timestamp('2025-12-31')

df_treino = df_ts[df_ts['data'] <= split_treino]
df_validacao = df_ts[(df_ts['data'] > split_treino) & (df_ts['data'] <= split_validacao)]

print(f"[OK] Treino: {len(df_treino):,} | Validacao: {len(df_validacao):,}")

# SALVAR
print("\n[8] Salvando datasets...")
OUTPUT_DIR = DATA_DIR / 'processed'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df_treino.to_parquet(OUTPUT_DIR / 'dataset_treino_janela180d.parquet')
df_validacao.to_parquet(OUTPUT_DIR / 'dataset_validacao_janela180d.parquet')
print(f"[OK] Datasets salvos")

# METADATA
print("\n[9] Salvando metadata...")
treino_cvli = df_treino['cvli_180d'].sum()
val_cvli = df_validacao['cvli_180d'].sum()

metadata = {
    'metodo': 'Janela Movel 180 dias',
    'total_registros': len(df),
    'treino_dias': len(df_treino),
    'validacao_dias': len(df_validacao),
    'bairros': len(bairros),
    'treino_cvli': int(treino_cvli),
    'validacao_cvli': int(val_cvli),
    'criticidade_mean_treino': float(df_treino['criticidade'].mean()),
    'criticidade_mean_val': float(df_validacao['criticidade'].mean())
}

with open(OUTPUT_DIR / 'metadata_janela180d.json', 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=2)

print(f"[OK] Metadata salvo")

print("\n" + "="*80)
print("[SUCCESS] ETL COMPLETO")
print("="*80)
