#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
import config
from datetime import datetime, timedelta

# Carregar dados
df_crimes = pd.read_parquet(config.CONSOLIDATED_FILE)

print(f"Total de registros: {len(df_crimes)}")
print(f"Colunas: {df_crimes.columns.tolist()}")

# Preparar data
if 'data_hora' in df_crimes.columns:
    df_crimes['data'] = pd.to_datetime(df_crimes['data_hora']).dt.date
else:
    print("Coluna 'data_hora' não encontrada!")

print(f"\nIntervalo de datas nos dados:")
print(f"  Min: {df_crimes['data'].min()}")
print(f"  Max: {df_crimes['data'].max()}")

# Verificar filtro padrão
hoje = datetime.now().date()
data_inicio = hoje - timedelta(days=30)

print(f"\nFiltro padrão (últimos 30 dias):")
print(f"  Data início: {data_inicio}")
print(f"  Data fim: {hoje}")

filtered = df_crimes[(df_crimes['data'] >= data_inicio) & (df_crimes['data'] <= hoje)]
print(f"  Crimes no período: {len(filtered)}")

if len(filtered) == 0:
    print(f"\n⚠️  Nenhum crime encontrado! Os dados são de antes de 30 dias atrás.")
    print(f"  Sugestão: Use o período completo do dataset ({df_crimes['data'].min()} a {df_crimes['data'].max()})")
