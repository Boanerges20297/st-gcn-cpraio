#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
investigar_praia_iracema.py

Investiga por que "Praia de Iracema" aparece como bairro crítico
"""

import torch
import json
import pandas as pd
import numpy as np
from pathlib import Path

print("="*80)
print("INVESTIGACAO: PRAIA DE IRACEMA")
print("="*80)

# ================== [1] CARREGAR DADOS ==================
print("\n[1] Carregando dados...")

tensor_dir = Path('data/tensors')
criticidade = torch.load(tensor_dir / 'dataset_criticidade_janela180d.pt')

with open(tensor_dir / 'metadata_janela180d.json', 'r', encoding='utf-8') as f:
    metadata = json.load(f)

bairro_mapping = metadata['bairro_mapping']

print(f"   Tensor shape: {criticidade.shape}")
print(f"   Bairros mapeados: {len(bairro_mapping)}")

# ================== [2] PROCURAR "IRACEMA" ==================
print("\n[2] Procurando variações de 'IRACEMA' no mapeamento...")

iracema_variants = []
for bairro_nome, idx in bairro_mapping.items():
    if 'IRACEMA' in bairro_nome.upper():
        crit_atual = criticidade[-1, idx, 0].item()  # Último dia
        iracema_variants.append({
            'nome': bairro_nome,
            'indice': idx,
            'criticidade_ultima': crit_atual
        })
        print(f"   ✅ {bairro_nome} (idx={idx}) -> criticidade={crit_atual:.4f}")

if not iracema_variants:
    print("   ❌ Nenhuma variação de IRACEMA encontrada no mapeamento!")

# ================== [3] CHECKAR FONTE DE DADOS ==================
print("\n[3] Procurando 'PRAIA DE IRACEMA' na base consolidada...")

try:
    base = pd.read_parquet('data/processed/base_consolidada_orcrim_v3.parquet')
    
    # Procurar Praia de Iracema
    praia_records = base[base['local_oficial'].str.contains('PRAIA|IRACEMA', case=False, na=False)]
    
    print(f"   Registros encontrados: {len(praia_records)}")
    if len(praia_records) > 0:
        print(f"\n   Locais únicos:")
        for local in praia_records['local_oficial'].unique():
            count = len(praia_records[praia_records['local_oficial'] == local])
            print(f"   - {local}: {count} registros")
        
        print(f"\n   Exemplos:")
        for _, row in praia_records.head(3).iterrows():
            print(f"   - Data: {row['data_ocorrencia']}, Local: {row['local_oficial']}, Bairro: {row.get('bairro', 'N/A')}")
    
except Exception as e:
    print(f"   Erro: {e}")

# ================== [4] CONFERIR NORMALIZACAO ==================
print("\n[4] Verificando normalização nos dados brutos...")

try:
    # Carregar dados originais
    with open('data/raw/dados_status_ocorrencias_gerais.json', 'r', encoding='utf-8') as f:
        raw = json.load(f)
    
    if isinstance(raw, dict):
        records = raw.get('data', [])
    else:
        records = raw
    
    # Procurar por variações
    praia_in_raw = []
    for rec in records:
        local = rec.get('local_oficial', '')
        if 'IRACEMA' in local.upper():
            praia_in_raw.append(local)
    
    print(f"   Variações de IRACEMA nos dados brutos:")
    for variant in set(praia_in_raw):
        count = praia_in_raw.count(variant)
        print(f"   - '{variant}': {count} ocorrências")
    
except Exception as e:
    print(f"   Erro: {e}")

# ================== [5] PROCURAR EM ARQUIVOS PROCESSADOS ==================
print("\n[5] Procurando em arquivos processados...")

try:
    # Ver o arquivo atualizado pelo dashboard
    df_capital = pd.read_csv('outputs/reports/pred_capital_bairros.csv')
    df_rmf = pd.read_csv('outputs/reports/pred_rmf_bairros.csv')
    df_interior = pd.read_csv('outputs/reports/pred_interior_bairros.csv')
    
    # Concatenar
    df_all = pd.concat([df_capital, df_rmf, df_interior], ignore_index=True)
    
    # Procurar Iracema
    iracema_in_output = df_all[df_all['local_oficial'].str.contains('IRACEMA', case=False, na=False)]
    
    print(f"   Encontrado em outputs/reports/:")
    for _, row in iracema_in_output.iterrows():
        print(f"   - {row['local_oficial']}: risco={row['risco_previsto']:.4f}, nivel={row['nivel_risco']}")
    
except Exception as e:
    print(f"   Erro: {e}")

# ================== [6] RASTREAMENTO DE ÍNDICE ==================
print("\n[6] Rastreamento de índice para 'PRAIA DE IRACEMA'...")

for variant in iracema_variants:
    nome = variant['nome']
    idx = variant['indice']
    
    print(f"\n   Bairro: {nome}")
    print(f"   Índice: {idx}")
    
    # Últimos 30 dias
    ultimos_30 = criticidade[-30:, idx, 0]
    print(f"   Criticidade últimos 30 dias:")
    print(f"   - Min: {ultimos_30.min():.4f}")
    print(f"   - Max: {ultimos_30.max():.4f}")
    print(f"   - Mean: {ultimos_30.mean():.4f}")
    print(f"   - Tendência: {'AUMENTANDO' if ultimos_30[-1] > ultimos_30[0] else 'DIMINUINDO'}")

print("\n" + "="*80)
print("CONCLUSAO")
print("="*80)

if iracema_variants:
    print("""
Se "Praia de Iracema" está aparecendo como crítico:

1. POSSÍVEL CAUSA 1: Há um erro no mapeamento do tensor
   - "Praia de Iracema" foi mapeada com um índice errado
   - Deveria ter sido filtrada mas não foi

2. POSSÍVEL CAUSA 2: Erro na normalização dos dados
   - O bairro recebeu um nome genérico "IRACEMA" 
   - Está confundido com "JARDIM IRACEMA"

3. POSSÍVEL CAUSA 3: Dados brutos contêm "Praia de Iracema"
   - Precisa ser filtrado na origem

RECOMENDACAO:
- Verificar se "Praia de Iracema" deveria estar no mapeamento
- Se não deveria, remover da entrada de dados
- Se sim, documentar o motivo
""")
else:
    print("✅ 'Praia de Iracema' NÃO foi encontrada no mapeamento de bairros.")
    print("   Pode estar sendo referenciada por outro nome ou já foi filtrada.")

print("="*80)
