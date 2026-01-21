#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
from pathlib import Path
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Caminho do arquivo
consolidated = Path(__file__).parent.parent / 'data' / 'consolidated.parquet'
pred_file = Path(__file__).parent.parent / 'data' / 'processed' / 'predictions_capital.csv'

if consolidated.exists():
    df = pd.read_parquet(consolidated)
    print("=== ESTRUTURA DO ARQUIVO ===")
    print(f"Colunas: {df.columns.tolist()}")
    print(f"Shape: {df.shape}")
    print(f"\nPrimeiras linhas:")
    print(df.head(2))
    
    # Verificar facção
    if 'faccao_predominante' in df.columns:
        print(f"\nfaccao_predominante - valores únicos: {df['faccao_predominante'].unique()[:10]}")
        print(f"Nulos: {df['faccao_predominante'].isna().sum()}")
    
    if 'faccao' in df.columns:
        print(f"\nfaccao - valores únicos: {df['faccao'].unique()[:10]}")
        print(f"Nulos: {df['faccao'].isna().sum()}")
        
    # Verificar bairros
    if 'local_oficial' in df.columns:
        print(f"\nBairros únicos: {df['local_oficial'].nunique()}")
        print(f"Primeiros 5 bairros: {df['local_oficial'].unique()[:5]}")
else:
    print(f"Arquivo não encontrado: {consolidated}")
