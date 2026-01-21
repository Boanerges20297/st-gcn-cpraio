#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.app import app
from src import config
import pandas as pd

# Carregar dados
df_crimes = pd.read_parquet(config.CONSOLIDATED_FILE)
print("\n=== ANÁLISE DOS DADOS ===")
print(f"Colunas no CSV: {df_crimes.columns.tolist()}")
print(f"\nPrimeiras linhas:")
print(df_crimes.head(2))

# Testar bairros
df_pred = pd.read_csv(config.ARTIFACTS['CAPITAL']['prediction'])
bairros = df_pred.nlargest(3, 'risco_previsto')['local_oficial'].unique()

print(f"\n=== ANÁLISE POR BAIRRO ===")
for bairro in bairros:
    df_b = df_crimes[df_crimes['local_oficial'] == bairro]
    print(f"\n{bairro}: {len(df_b)} registros")
    if 'faccao_predominante' in df_b.columns:
        print(f"  faccao_predominante: {df_b['faccao_predominante'].value_counts().to_dict()}")
    if 'faccao' in df_b.columns:
        print(f"  faccao: {df_b['faccao'].value_counts().to_dict()}")
