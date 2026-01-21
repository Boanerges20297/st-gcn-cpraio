#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
investigacao_criticidade_praia_iracema.py

Por que Praia de Iracema tem criticidade média/alta (0.3270)?
- 38 registros históricos totais
- Deveria ter criticidade BAIXA
"""

import torch
import pandas as pd
import json
import numpy as np
from pathlib import Path

print("="*80)
print("INVESTIGACAO: CRITICIDADE DE PRAIA DE IRACEMA")
print("="*80)

# ================== [1] CARREGAR DADOS ==================
print("\n[1] Carregando tensores e metadata...")

tensor_dir = Path('data/tensors')
criticidade = torch.load(tensor_dir / 'dataset_criticidade_janela180d.pt')

with open(tensor_dir / 'metadata_janela180d.json', 'r', encoding='utf-8') as f:
    metadata = json.load(f)

bairro_mapping = metadata['bairro_mapping']

# Índices de IRACEMA
praia_idx = bairro_mapping['PRAIA DE IRACEMA']
iracema_idx = bairro_mapping['IRACEMA']

print(f"   Tensor shape: {criticidade.shape}")
print(f"   PRAIA DE IRACEMA: índice {praia_idx}")
print(f"   IRACEMA: índice {iracema_idx}")

# ================== [2] ANALISAR CRITICIDADE ==================
print("\n[2] Análise de criticidade...")

praia_crit = criticidade[:, praia_idx, 0].numpy()
iracema_crit = criticidade[:, iracema_idx, 0].numpy()

print(f"\n   PRAIA DE IRACEMA:")
print(f"   - Min: {praia_crit.min():.4f}")
print(f"   - Max: {praia_crit.max():.4f}")
print(f"   - Mean: {praia_crit.mean():.4f}")
print(f"   - Std: {praia_crit.std():.4f}")
print(f"   - Mediana: {np.median(praia_crit):.4f}")
print(f"   - Últimos 30 dias mean: {praia_crit[-30:].mean():.4f}")

print(f"\n   IRACEMA (para comparação):")
print(f"   - Min: {iracema_crit.min():.4f}")
print(f"   - Max: {iracema_crit.max():.4f}")
print(f"   - Mean: {iracema_crit.mean():.4f}")
print(f"   - Std: {iracema_crit.std():.4f}")
print(f"   - Mediana: {np.median(iracema_crit):.4f}")
print(f"   - Últimos 30 dias mean: {iracema_crit[-30:].mean():.4f}")

# ================== [3] COMPARAR COM OUTROS BAIRROS ==================
print("\n[3] Ranking de criticidade (últimos 30 dias)...")

# Calcular criticidade média dos últimos 30 dias para todos
crit_30d_mean = criticidade[-30:, :, 0].mean(dim=0).numpy()

# Top 10 mais críticos
top_indices = np.argsort(crit_30d_mean)[::-1][:10]

print(f"\n   Top 10 bairros por criticidade (últimos 30 dias):")
for rank, idx in enumerate(top_indices, 1):
    # Procurar nome
    bairro_nome = [k for k, v in bairro_mapping.items() if v == idx][0] if idx in bairro_mapping.values() else f"Índice {idx}"
    crit_val = crit_30d_mean[idx]
    
    marker = "👈 PRAIA DE IRACEMA" if idx == praia_idx else ""
    marker = "👈 IRACEMA" if idx == iracema_idx else marker
    
    print(f"   {rank:2d}. {bairro_nome:30s}: {crit_val:.4f} {marker}")

# Onde eles ficam?
praia_rank = np.argsort(crit_30d_mean)[::-1].tolist().index(praia_idx) + 1
iracema_rank = np.argsort(crit_30d_mean)[::-1].tolist().index(iracema_idx) + 1

print(f"\n   PRAIA DE IRACEMA ranking: #{praia_rank} de {len(bairro_mapping)}")
print(f"   IRACEMA ranking: #{iracema_rank} de {len(bairro_mapping)}")

# ================== [4] VERIFICAR SE ESTÃO DUPLICADAS ==================
print("\n[4] Investigando possível duplicação...")

print(f"\n   Comparando índices dos dois:")
print(f"   - PRAIA DE IRACEMA (idx {praia_idx}) criticidade = {praia_crit[-1]:.4f}")
print(f"   - IRACEMA (idx {iracema_idx}) criticidade = {iracema_crit[-1]:.4f}")

if abs(praia_crit[-1] - iracema_crit[-1]) < 0.001:
    print("   ⚠️ ALERTA: Criticidades IDÊNTICAS (possível duplicação?)")
else:
    print("   ✓ Criticidades diferentes")

# ================== [5] CONFERIR DADOS BRUTOS ==================
print("\n[5] Contabilização de crimes na base consolidada...")

try:
    base = pd.read_parquet('data/processed/base_consolidada_orcrim_v3.parquet')
    
    praia_count = len(base[base['local_oficial'] == 'PRAIA DE IRACEMA'])
    iracema_count = len(base[base['local_oficial'] == 'IRACEMA'])
    
    print(f"\n   PRAIA DE IRACEMA: {praia_count} crimes no período completo")
    print(f"   IRACEMA: {iracema_count} crimes no período completo")
    
    # Distribuição por tipo
    praia_base = base[base['local_oficial'] == 'PRAIA DE IRACEMA']
    if 'tipo' in praia_base.columns:
        print(f"\n   Tipos de crime em PRAIA DE IRACEMA:")
        for tipo, count in praia_base['tipo'].value_counts().items():
            print(f"   - {tipo}: {count}")
    
    # Ao longo do tempo
    if 'data_hora' in praia_base.columns:
        praia_base['ano_mes'] = pd.to_datetime(praia_base['data_hora'], errors='coerce').dt.to_period('M')
        meses_com_crime = len(praia_base['ano_mes'].unique())
        print(f"\n   Distribuição: {meses_com_crime} meses diferentes com crimes")

except Exception as e:
    print(f"   Erro: {e}")

# ================== [6] CONCLUSAO ==================
print("\n" + "="*80)
print("CONCLUSAO")
print("="*80)

print(f"""
PRAIA DE IRACEMA vs IRACEMA:

Volume histórico:
- PRAIA DE IRACEMA: 38 crimes (período 2022-2025)
- IRACEMA: 57 crimes (período 2022-2025)

Criticidade atual (últimos 30 dias):
- PRAIA DE IRACEMA: {praia_crit[-30:].mean():.4f} (ranking #{praia_rank})
- IRACEMA: {iracema_crit[-30:].mean():.4f} (ranking #{iracema_rank})

POSSÍVEL PROBLEMA:
Com apenas 38-57 crimes em 4 anos, a criticidade {praia_crit[-30:].mean():.4f} parece ALTA.

HIPÓTESES:
1. Normalização está inflando valores baixos
2. Dados estão sendo contabilizados por bairro diferente
3. Concentração de crimes em períodos específicos causa picos
4. Métrica de criticidade não é proporcional ao volume

ACAO RECOMENDADA:
- Verificar fórmula de criticidade (se usa density, não apenas count)
- Se criticidade=crimes/população: Praia pode ter população baixa
- Se for erro, considerar APENAS remover dados após X data
""")

print("="*80)
