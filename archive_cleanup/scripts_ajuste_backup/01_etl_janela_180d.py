#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ETL com Criticidade Dinâmica - Janela Móvel 180 dias
====================================================
Calcula criticidade CVLI com base em eventos nos últimos 180 dias (6 meses).
Isso torna o modelo muito mais sensível a picos recentes.

Exemplo:
- Praia de Iracema: 5 homicídios em 4 anos = BAIXO
- Praia de Iracema: 5 homicídios em 6 meses = ALTO
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'

print("\n" + "="*80)
print("ETL - CRITICIDADE COM JANELA 180 DIAS")
print("="*80)

# 1. LOAD DATA
print("\n[1] CARREGANDO DADOS ENRIQUECIDOS...")
enriquecido_path = DATA_DIR / 'processed' / 'dados_status_enriquecidos_com_bairros.parquet'

if not enriquecido_path.exists():
    print(f"[ERROR] Arquivo não encontrado: {enriquecido_path}")
    sys.exit(1)

df = pd.read_parquet(enriquecido_path)
print(f"[OK] {len(df):,} registros carregados")

# 2. CONVERTER DATAS
print("\n[2] PROCESSANDO DATAS...")
df['data'] = pd.to_datetime(df['data'], errors='coerce')
df = df.dropna(subset=['data', 'bairro'])

# Filtrar para período de análise (2022-2025)
data_min = pd.Timestamp('2022-01-01')
data_max = pd.Timestamp('2025-12-31')
df = df[(df['data'] >= data_min) & (df['data'] <= data_max)]
print(f"[OK] {len(df):,} registros no período 2022-2025")

# 3. SEPARAR CVLI vs CVP
print("\n[3] SEPARANDO CVLI vs CVP...")

# Detectar se é CVLI - usar tipo_crime ou tipo_evento
def is_cvli(row):
    for col in ['tipo_crime', 'tipo_evento', 'ais']:
        if pd.isna(row[col]):
            continue
        val = str(row[col]).upper().strip()
        cvli_keywords = ['HOMICIDIO', 'LATROCINIO', 'LESAO CORPORAL SEGUIDA DE MORTE', 'CVLI']
        if any(kw in val for kw in cvli_keywords):
            return True
    return False

df['is_cvli'] = df.apply(is_cvli, axis=1)

cvli_count = df['is_cvli'].sum()
print(f"[OK] CVLI: {cvli_count:,} ({100*cvli_count/len(df):.1f}%)")
print(f"[OK] CVP: {len(df)-cvli_count:,} ({100*(len(df)-cvli_count)/len(df):.1f}%)")

# 4. CRIAR SÉRIE TEMPORAL COM CRITICIDADE 180D
print("\n[4] CALCULANDO CRITICIDADE COM JANELA 180 DIAS...")

# Datas únicas para a série
dates = pd.date_range(start='2022-01-01', end='2025-12-31', freq='D')
bairros = df['bairro'].unique()

print(f"   {len(dates)} dias x {len(bairros)} bairros = {len(dates)*len(bairros):,} células")

# Estrutura: para cada data e bairro, contar CVLI nos últimos 180 dias
timeseries_data = []

for i, current_date in enumerate(dates):
    if i % 100 == 0:
        print(f"   Processando dia {i+1}/{len(dates)}...")
    
    # Janela 180 dias
    window_start = current_date - timedelta(days=180)
    
    for bairro in bairros:
        # CVLI nesta janela
        cvli_in_window = len(df[
            (df['bairro'] == bairro) &
            (df['is_cvli'] == True) &
            (df['data'] >= window_start) &
            (df['data'] < current_date)
        ])
        
        # CVP nesta janela (contexto)
        cvp_in_window = len(df[
            (df['bairro'] == bairro) &
            (df['is_cvli'] == False) &
            (df['data'] >= window_start) &
            (df['data'] < current_date)
        ])
        
        timeseries_data.append({
            'data': current_date,
            'bairro': bairro,
            'cvli_180d': cvli_in_window,
            'cvp_180d': cvp_in_window,
            'total_180d': cvli_in_window + cvp_in_window
        })

df_ts = pd.DataFrame(timeseries_data)
print(f"[OK] {len(df_ts):,} células preenchidas")

# 5. CALCULAR CRITICIDADE NORMALIZADA
print("\n[5] NORMALIZANDO CRITICIDADE...")

# Para cada data, calcular quantis
df_ts['cvli_quantile'] = df_ts.groupby('data')['cvli_180d'].transform(
    lambda x: x.rank(pct=True)
)

# Criticidade: 0-1 baseado em quantis
df_ts['criticidade'] = df_ts['cvli_quantile']

print(f"[OK] Criticidade calculada")
print(f"  Min: {df_ts['criticidade'].min():.3f}")
print(f"  Mean: {df_ts['criticidade'].mean():.3f}")
print(f"  Max: {df_ts['criticidade'].max():.3f}")

# 6. IDENTIFICAR TOP CRÍTICOS
print("\n[6] TOP 10 BAIRROS CRÍTICOS (últimos 180 dias)...")

# Últimos 180 dias
data_final = df_ts['data'].max()
data_inicio_window = data_final - timedelta(days=180)

top_critical = df_ts[df_ts['data'] == data_final].nlargest(10, 'criticidade')[
    ['bairro', 'cvli_180d', 'criticidade']
]

for idx, row in top_critical.iterrows():
    print(f"  {row['bairro']:20s} | CVLI 180d: {row['cvli_180d']:3.0f} | Crit: {row['criticidade']:.3f}")

# 7. SPLITS TEMPORAIS
print("\n[7] CRIANDO SPLITS TEMPORAIS...")

split_treino = pd.Timestamp('2024-12-31')
split_validacao = pd.Timestamp('2025-12-31')

df_treino = df_ts[df_ts['data'] <= split_treino]
df_validacao = df_ts[(df_ts['data'] > split_treino) & (df_ts['data'] <= split_validacao)]

print(f"[OK] Treino (até 2024-12-31): {len(df_treino):,} registros")
print(f"[OK] Validação (2025): {len(df_validacao):,} registros")

# 8. SALVAR DATASETS
print("\n[8] SALVANDO DATASETS...")

output_path_treino = DATA_DIR / 'processed' / 'dataset_treino_janela180d.parquet'
output_path_val = DATA_DIR / 'processed' / 'dataset_validacao_janela180d.parquet'

df_treino.to_parquet(output_path_treino)
df_validacao.to_parquet(output_path_val)

print(f"[OK] {output_path_treino}")
print(f"[OK] {output_path_val}")

# 9. ESTATÍSTICAS
print("\n[9] ESTATÍSTICAS...")

print("\n=== TREINO 2022-2024 ===")
treino_cvli_total = df_treino['cvli_180d'].sum()
print(f"Total CVLI (histórico): {treino_cvli_total:,}")
print(f"Média CVLI/dia: {treino_cvli_total/len(df_treino)*len(bairros):.1f}")

print("\n=== VALIDAÇÃO 2025 ===")
val_cvli_total = df_validacao['cvli_180d'].sum()
print(f"Total CVLI (histórico): {val_cvli_total:,}")
print(f"Média CVLI/dia: {val_cvli_total/len(df_validacao)*len(bairros):.1f}")

print("\n=== COMPARAÇÃO ===")
mudanca = ((val_cvli_total / (len(df_validacao)*len(bairros))) - 
           (treino_cvli_total / (len(df_treino)*len(bairros)))) / \
          (treino_cvli_total / (len(df_treino)*len(bairros))) * 100
print(f"Mudança: {mudanca:+.1f}%")

# 10. SALVAR METADATA
print("\n[10] SALVANDO METADATA...")

metadata = {
    'criacao': datetime.now().isoformat(),
    'metodo': 'Criticidade Janela Móvel 180 dias',
    'total_registros': len(df),
    'periodo_analise': '2022-01-01 a 2025-12-31',
    'treino_dias': len(df_treino),
    'validacao_dias': len(df_validacao),
    'bairros_unicos': len(bairros),
    'split_treino_data': split_treino.isoformat(),
    'split_validacao_data': split_validacao.isoformat(),
    'estatisticas': {
        'treino_cvli_total': int(treino_cvli_total),
        'validacao_cvli_total': int(val_cvli_total),
        'mudanca_percentual': float(mudanca),
        'criticidade_media_treino': float(df_treino['criticidade'].mean()),
        'criticidade_media_val': float(df_validacao['criticidade'].mean())
    }
}

meta_path = DATA_DIR / 'processed' / 'metadata_janela180d.json'
with open(meta_path, 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

print(f"[OK] {meta_path}")

print("\n" + "="*80)
print("[SUCCESS] ETL CONCLUÍDO COM SUCESSO")
print("="*80)
