#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
13_analise_efetividade_pratica.py

Analisa a EFETIVIDADE PRÁTICA de prisões em reduzir crimes:
- Percentual de casos com diminuição >= 15%
- Magnitude média de redução quando há redução
- Relação dose-resposta (volume de prisões vs redução)
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy import stats

# ================== [1] CARREGAR DADOS ==================
print("[1] Carregando dados...")

# RAIO 2025 - estrutura PHPMyAdmin
with open('data/raw/ocorrencia_caucaia_2025.json', 'r', encoding='utf-8') as f:
    raio_data = json.load(f)

raio_operations = []
for item in raio_data:
    if isinstance(item, dict) and item.get('type') == 'table':
        raio_operations = item.get('data', [])
        break

raio_df = pd.DataFrame(raio_operations)
raio_df['data_dt'] = pd.to_datetime(raio_df['Data'], errors='coerce')
raio_df = raio_df.dropna(subset=['data_dt'])
raio_df = raio_df.sort_values('data_dt').reset_index(drop=True)

print(f"   Total RAIO operations: {len(raio_df)}")
print(f"   Data range: {raio_df['data_dt'].min()} to {raio_df['data_dt'].max()}")

# Criticidade completa
try:
    import torch
    criticidade_tensor = torch.load('data/tensors/criticidade_final_completo.pt').numpy()
except:
    try:
        criticidade_npz = np.load('data/tensors/criticidade_final_completo.pt', allow_pickle=True)
        criticidade_tensor = np.array(list(criticidade_npz.values())[0]) if hasattr(criticidade_npz, 'values') else criticidade_npz
    except:
        # Fallback para arquivo original
        criticidade_tensor = np.load('data/tensors/dataset_criticidade_janela180d_completo.pt', allow_pickle=True)
print(f"   Criticidade tensor shape: {criticidade_tensor.shape}")

# Metadata
with open('data/tensors/metadata_janela180d_completo.json', 'r', encoding='utf-8') as f:
    metadata = json.load(f)

bairro_to_idx = metadata.get('bairro_mapping', {})
if not bairro_to_idx:
    # Se não tiver, carregar do metadata original e complementar
    with open('data/tensors/metadata_janela180d.json', 'r', encoding='utf-8') as f:
        metadata_orig = json.load(f)
        bairro_to_idx = metadata_orig.get('bairro_mapping', {})
print(f"   Neighborhoods in model: {len(bairro_to_idx)}")

# ================== [2] PERÍODOS 30-DIA ==================
print("\n[2] Agrupando em períodos de 30 dias...")

start_date = raio_df['data_dt'].min().replace(hour=0, minute=0, second=0, microsecond=0)
end_date = raio_df['data_dt'].max().replace(hour=23, minute=59, second=59, microsecond=999999)

periods = []
current = start_date
while current < end_date:
    next_period = current + timedelta(days=30)
    periods.append({'start': current, 'end': next_period, 'period_id': len(periods)})
    current = next_period

print(f"   Total periods: {len(periods)}")

# ================== [3] RAIO POR PERÍODO ==================
print("\n[3] Volume de RAIO por período...")

raio_per_period = []
for p in periods:
    period_raio = raio_df[(raio_df['data_dt'] >= p['start']) & (raio_df['data_dt'] < p['end'])]
    raio_per_period.append({
        'period_id': p['period_id'],
        'start': p['start'],
        'end': p['end'],
        'raio_count': len(period_raio),
        'raio_bairros': set(period_raio['BairroOcor'].unique()) if len(period_raio) > 0 else set()
    })

raio_periods_df = pd.DataFrame(raio_per_period)
print(f"   Min: {raio_periods_df['raio_count'].min()}, Max: {raio_periods_df['raio_count'].max()}, Média: {raio_periods_df['raio_count'].mean():.1f}")

# ================== [4] CRITICIDADE POR PERÍODO ==================
print("\n[4] Criticidade média por período...")

data_base = datetime(2022, 10, 1)
criticidade_per_period = []

for i, row in raio_periods_df.iterrows():
    period_id = row['period_id']
    start_idx = (row['start'] - data_base).days
    end_idx = (row['end'] - data_base).days
    
    if start_idx < 0 or end_idx > len(criticidade_tensor):
        continue
    
    if len(row['raio_bairros']) > 0:
        bairro_indices = []
        for bairro in row['raio_bairros']:
            if bairro in bairro_to_idx:
                bairro_indices.append(bairro_to_idx[bairro])
        
        if len(bairro_indices) > 0:
            crit_slice = criticidade_tensor[start_idx:end_idx, bairro_indices, 0]
            crit_mean = np.mean(crit_slice)
            crit_std = np.std(crit_slice)
        else:
            continue
    else:
        continue
    
    criticidade_per_period.append({
        'period_id': period_id,
        'start': row['start'],
        'raio_count': row['raio_count'],
        'criticidade_mean': crit_mean,
        'criticidade_std': crit_std,
        'num_bairros': len(row['raio_bairros'])
    })

crit_df = pd.DataFrame(criticidade_per_period).dropna()
print(f"   Períodos válidos: {len(crit_df)}")

# ================== [5] COMPARAR PERÍODOS ==================
print("\n[5] Comparando períodos consecutivos...")

comparacoes = []
for i in range(len(crit_df) - 1):
    t_atual = crit_df.iloc[i]
    t_prox = crit_df.iloc[i + 1]
    
    crit_atual = t_atual['criticidade_mean']
    crit_prox = t_prox['criticidade_mean']
    raio_atual = t_atual['raio_count']
    
    var_pct = ((crit_prox - crit_atual) / crit_atual * 100) if crit_atual > 0 else 0
    
    if raio_atual > 10:
        raio_trend = "RAIO_ALTO"
    elif raio_atual > 5:
        raio_trend = "RAIO_MEDIO"
    elif raio_atual > 0:
        raio_trend = "RAIO_BAIXO"
    else:
        raio_trend = "RAIO_NULO"
    
    if var_pct < -15:
        reducao_cat = "REDUZIDO_15PLUS"
    elif var_pct < 0:
        reducao_cat = "REDUZIDO_0A15"
    elif var_pct < 15:
        reducao_cat = "AUMENTOU_0A15"
    else:
        reducao_cat = "AUMENTOU_15PLUS"
    
    comparacoes.append({
        'raio_count_t': raio_atual,
        'raio_trend': raio_trend,
        'crit_t': crit_atual,
        'crit_t1': crit_prox,
        'var_pct': var_pct,
        'reducao_cat': reducao_cat
    })

comp_df = pd.DataFrame(comparacoes)

# ================== [6] ANÁLISE PRÁTICA ==================
print("\n[6] ANÁLISE PRÁTICA - Efetividade por Intensidade de Prisões")
print("=" * 80)

for raio_trend in ["RAIO_ALTO", "RAIO_MEDIO", "RAIO_BAIXO"]:
    subset = comp_df[comp_df['raio_trend'] == raio_trend]
    
    if len(subset) == 0:
        continue
    
    reduzido_15plus = len(subset[subset['reducao_cat'] == 'REDUZIDO_15PLUS'])
    reduzido_0a15 = len(subset[subset['reducao_cat'] == 'REDUZIDO_0A15'])
    total_reducoes = reduzido_15plus + reduzido_0a15
    
    pct_15plus = (reduzido_15plus / len(subset) * 100) if len(subset) > 0 else 0
    pct_total = (total_reducoes / len(subset) * 100) if len(subset) > 0 else 0
    
    print(f"\n{raio_trend} (n={len(subset)} períodos):")
    print(f"  Prisões: {subset['raio_count_t'].mean():.1f} ± {subset['raio_count_t'].std():.1f} por período")
    print(f"  Redução ≥ 15%: {reduzido_15plus} casos ({pct_15plus:.1f}%)")
    print(f"  Qualquer redução: {total_reducoes} casos ({pct_total:.1f}%)")
    
    reducoes = subset[subset['var_pct'] < 0]
    if len(reducoes) > 0:
        print(f"  Magnitude média: {reducoes['var_pct'].mean():.1f}% (quando reduz)")

# ================== [7] DOSE-RESPOSTA ==================
print("\n[7] Relação Dose-Resposta")
print("=" * 80)

corr = comp_df[['raio_count_t', 'var_pct']].corr().iloc[0, 1]
slope, intercept, r_value, p_value, std_err = stats.linregress(comp_df['raio_count_t'], comp_df['var_pct'])

print(f"\nCorrelação (Pearson r): {corr:.4f}")
print(f"Slope: {slope:.6f}%/prisão adicional")
print(f"P-value: {p_value:.4f}")
print(f"Significância: {'SIM ✓' if p_value < 0.05 else 'NÃO ✗'}")

# ================== [8] EFETIVIDADE FINAL ==================
print("\n[8] EFETIVIDADE OBSERVADA")
print("=" * 80)

total_15 = len(comp_df[comp_df['reducao_cat'] == 'REDUZIDO_15PLUS'])
total_qual = len(comp_df[comp_df['var_pct'] < 0])

print(f"\nEm {len(comp_df)} períodos:")
print(f"  • Redução ≥ 15%: {total_15} ({total_15/len(comp_df)*100:.1f}%)")
print(f"  • Qualquer redução: {total_qual} ({total_qual/len(comp_df)*100:.1f}%)")

if pct_15plus > 30:
    print("\n✅ CONCLUSÃO: EFETIVIDADE ALTA - Prisões reduzem crimes significativamente")
elif total_qual > len(comp_df) * 0.5:
    print("\n⚠️  CONCLUSÃO: EFETIVIDADE MODERADA - Há redução em vários períodos")
else:
    print("\n❌ CONCLUSÃO: EFETIVIDADE BAIXA - Prisões não reduzem crimes detectavelmente")

print("\n" + "=" * 80)
