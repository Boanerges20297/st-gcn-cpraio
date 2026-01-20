#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
investigacao_praia_iracema_profunda.py

Investigação detalhada sobre "Praia de Iracema"
- Volume histórico de crimes
- Tipos de crimes (CVP/CVLI)
- Tendência temporal
- Comparação com outros bairros
"""

import pandas as pd
import json
from pathlib import Path
from datetime import datetime
import numpy as np

print("="*80)
print("INVESTIGACAO DETALHADA: PRAIA DE IRACEMA")
print("="*80)

# ================== [1] DADOS BRUTOS ==================
print("\n[1] Carregando dados brutos...")

try:
    with open('data/raw/dados_status_ocorrencias_gerais.json', 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    if isinstance(raw_data, dict):
        records = raw_data.get('data', [])
    else:
        records = raw_data
    
    df_raw = pd.DataFrame(records)
    print(f"   Total de registros: {len(df_raw)}")
    print(f"   Colunas: {df_raw.columns.tolist()}")
    
except Exception as e:
    print(f"   Erro: {e}")
    exit(1)

# ================== [2] FILTRAR VARIAÇÕES DE IRACEMA ==================
print("\n[2] Analisando todas as variações de IRACEMA nos dados brutos...")

for variant in ['PRAIA DE IRACEMA', 'IRACEMA', 'JARDIM IRACEMA', 'PARQUE IRACEMA']:
    df_var = df_raw[df_raw['local_oficial'].str.contains(variant, case=False, na=False)] if 'local_oficial' in df_raw.columns else pd.DataFrame()
    
    if len(df_var) == 0:
        # Tentar no bairro
        if 'bairro' in df_raw.columns:
            df_var = df_raw[df_raw['bairro'].str.contains(variant, case=False, na=False)]
        elif 'local' in df_raw.columns:
            df_var = df_raw[df_raw['local'].str.contains(variant, case=False, na=False)]
    
    if len(df_var) > 0:
        print(f"\n   {variant}:")
        print(f"   - Total: {len(df_var)} registros")
        
        # Distribuição por tipo
        if 'tipo' in df_var.columns:
            print(f"   - Tipos:")
            for tipo, count in df_var['tipo'].value_counts().items():
                print(f"     • {tipo}: {count}")
        
        # Distribuição por data (primeiros e últimos)
        if 'data' in df_var.columns:
            df_var['data'] = pd.to_datetime(df_var['data'], errors='coerce')
            print(f"   - Data range: {df_var['data'].min()} a {df_var['data'].max()}")
            
            # Por ano
            df_var['ano'] = df_var['data'].dt.year
            print(f"   - Por ano:")
            for ano, count in sorted(df_var['ano'].value_counts().items()):
                print(f"     • {ano}: {count}")
    else:
        print(f"\n   {variant}: Nenhum registro encontrado")

# ================== [3] CONFERIR BASE CONSOLIDADA ==================
print("\n[3] Verificando base consolidada...")

try:
    base = pd.read_parquet('data/processed/base_consolidada_orcrim_v3.parquet')
    
    for variant in ['PRAIA DE IRACEMA', 'IRACEMA', 'JARDIM IRACEMA', 'PARQUE IRACEMA']:
        df_base = base[base['local_oficial'].str.contains(variant, case=False, na=False)] if 'local_oficial' in base.columns else pd.DataFrame()
        
        if len(df_base) > 0:
            print(f"\n   {variant}:")
            print(f"   - Total: {len(df_base)} registros")
            print(f"   - Região: {df_base['regiao_sistema'].unique() if 'regiao_sistema' in df_base.columns else 'N/A'}")
            print(f"   - Tipo: {df_base['tipo'].unique() if 'tipo' in df_base.columns else 'N/A'}")
        else:
            print(f"\n   {variant}: Nenhum registro na base consolidada")

except Exception as e:
    print(f"   Erro: {e}")

# ================== [4] COMPARACAO COM OUTROS BAIRROS DA CAPITAL ==================
print("\n[4] Comparação com outros bairros da capital...")

try:
    base = pd.read_parquet('data/processed/base_consolidada_orcrim_v3.parquet')
    base_capital = base[base['regiao_sistema'] == 'CAPITAL']
    
    # Contar crimes por bairro
    bairro_counts = base_capital['local_oficial'].value_counts()
    
    print(f"\n   Top 20 bairros da capital por volume de crimes:")
    for i, (bairro, count) in enumerate(bairro_counts.head(20).items(), 1):
        marker = "👈 PRAIA DE IRACEMA" if "PRAIA DE IRACEMA" in bairro else ""
        print(f"   {i:2d}. {bairro:30s}: {count:5d} {marker}")
    
    # Onde Praia de Iracema fica na lista?
    praia_rank = list(bairro_counts.index).index('PRAIA DE IRACEMA') + 1 if 'PRAIA DE IRACEMA' in bairro_counts.index else -1
    if praia_rank > 0:
        print(f"\n   📍 PRAIA DE IRACEMA está no #{praia_rank}º lugar entre os {len(bairro_counts)} bairros da capital")
        print(f"      Volume: {bairro_counts['PRAIA DE IRACEMA']} registros")
        print(f"      Percentual: {bairro_counts['PRAIA DE IRACEMA'] / len(base_capital) * 100:.2f}% dos crimes da capital")

except Exception as e:
    print(f"   Erro: {e}")

# ================== [5] ANALISE TEMPORAL ==================
print("\n[5] Análise temporal de Praia de Iracema...")

try:
    praia_records = base[base['local_oficial'] == 'PRAIA DE IRACEMA']
    
    if len(praia_records) > 0 and 'data_hora' in praia_records.columns:
        praia_records_copy = praia_records.copy()
        praia_records_copy['data_hora'] = pd.to_datetime(praia_records_copy['data_hora'], errors='coerce')
        praia_records_copy['ano_mes'] = praia_records_copy['data_hora'].dt.to_period('M')
        
        print(f"\n   Distribuição mensal:")
        for periodo, count in sorted(praia_records_copy['ano_mes'].value_counts().items()):
            print(f"   - {periodo}: {count}")
    
except Exception as e:
    print(f"   Erro: {e}")

# ================== [6] POSSIVEL CONFUSAO COM OUTROS LOCAIS ==================
print("\n[6] Verificando possíveis confusões de nomes...")

try:
    base = pd.read_parquet('data/processed/base_consolidada_orcrim_v3.parquet')
    
    # Bairros que contém "IRACEMA"
    bairros_iracema = base[base['local_oficial'].str.contains('IRACEMA', case=False, na=False)]['local_oficial'].unique()
    
    print(f"\n   Todos os bairros com 'IRACEMA' no nome:")
    for bairro in sorted(bairros_iracema):
        count = len(base[base['local_oficial'] == bairro])
        print(f"   - {bairro}: {count} registros")

except Exception as e:
    print(f"   Erro: {e}")

# ================== [7] CONCLUSAO ==================
print("\n" + "="*80)
print("CONCLUSAO")
print("="*80)

try:
    base = pd.read_parquet('data/processed/base_consolidada_orcrim_v3.parquet')
    praia_count = len(base[base['local_oficial'] == 'PRAIA DE IRACEMA'])
    
    if praia_count > 10:
        print(f"""
✅ "PRAIA DE IRACEMA" TEM {praia_count} REGISTROS HISTÓRICOS

ANÁLISE:
- Volume significativo de crimes registrados
- Pode ser ponto de confusão de operadores CIOPS
- Mas PODE ter ocorrências reais (assaltos a turistas, etc)

RECOMENDACAO:
1. NÃO REMOVER ainda - manter dados históricos
2. Documentar que é location ambíguo (praia vs bairro)
3. Considerar remover APENAS dados novos (após data X)
4. Ou manter mas com alerta no dashboard

PROXIMO PASSO:
- Verificar se criticidade está INFLADA por erro
- Ou se realmente há concentração de crimes lá
""")
    elif praia_count > 0 and praia_count <= 10:
        print(f"""
⚠️ "PRAIA DE IRACEMA" TEM APENAS {praia_count} REGISTROS

ANÁLISE:
- Muito poucos registros
- Provavelmente erro de digitação/confusão do operador
- Deveria ter sido normalizado para bairro adjacente

RECOMENDACAO:
- REMOVER este local
- Aplicar filtro no ETL
""")
    else:
        print(f"""
❌ "PRAIA DE IRACEMA" NÃO TEM REGISTROS

Provavelmente foi criada durante o processamento por erro de mapeamento.
""")

except Exception as e:
    print(f"Erro na conclusão: {e}")

print("="*80)
