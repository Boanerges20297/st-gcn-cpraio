#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Gera o mapa correto de facções por bairro a partir dos dados reais
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src import config
import pandas as pd
import json

# Carregar dados
df_crimes = pd.read_parquet(config.CONSOLIDATED_FILE)

print(f"Carregado: {config.CONSOLIDATED_FILE}")
print(f"Colunas: {df_crimes.columns.tolist()}")

# Obter facção dominante por bairro
bairro_faccao = {}

for bairro in df_crimes['local_oficial'].unique():
    df_bairro = df_crimes[df_crimes['local_oficial'] == bairro]
    
    # Tentar faccao_predominante primeiro
    if 'faccao_predominante' in df_bairro.columns:
        faccao_counts = df_bairro['faccao_predominante'].dropna().value_counts()
        if len(faccao_counts) > 0:
            faccao = faccao_counts.index[0]
        else:
            faccao = "DESCONHECIDA"
    elif 'faccao' in df_bairro.columns:
        faccao_counts = df_bairro['faccao'].dropna().value_counts()
        if len(faccao_counts) > 0:
            faccao = faccao_counts.index[0]
        else:
            faccao = "DESCONHECIDA"
    else:
        faccao = "DESCONHECIDA"
    
    bairro_faccao[bairro] = faccao
    print(f"{bairro}: {faccao}")

# Salvar
output_path = config.DATA_PROCESSED / 'bairro_faccoes_map.json'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(bairro_faccao, f, ensure_ascii=False, indent=2)

print(f"\n✓ Mapa salvo em: {output_path}")
print(f"Total de bairros: {len(bairro_faccao)}")
