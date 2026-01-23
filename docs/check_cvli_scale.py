#!/usr/bin/env python3
"""Verificar escala real do CVLI histórico"""

import pandas as pd
import numpy as np

# Ler dados
df = pd.read_csv('data/processed/cvli_producao.csv')
df['data'] = pd.to_datetime(df['data'])

# Agregar por bairro-dia
cvli_by_day_bairro = df.groupby(['data', 'bairro_assigned']).size().reset_index(name='count')

# Estatísticas por bairro
by_bairro = cvli_by_day_bairro.groupby('bairro_assigned')['count'].agg(['mean', 'max', 'std', 'sum']).sort_values('max', ascending=False)

print('=== TOP 10 BAIRROS (por CVLI total) ===')
print(by_bairro.head(10))

print('\n=== OVERALL STATISTICS ===')
print(f'Eventos totais: {len(df)}')
print(f'Dias únicos: {df["data"].nunique()}')
print(f'Bairros únicos: {df["bairro_assigned"].nunique()}')
print(f'CVLI médio por dia-bairro: {cvli_by_day_bairro["count"].mean():.3f}')
print(f'CVLI max um dia-bairro: {cvli_by_day_bairro["count"].max()}')
print(f'\nDistribuição:')
print(cvli_by_day_bairro['count'].describe())
