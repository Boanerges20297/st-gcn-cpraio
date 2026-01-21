#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
14_analise_efetividade_estado.py

Analisa EFETIVIDADE PRÁTICA em TODO O CEARÁ
- Carrega dados de operações RAIO de todas as cidades
- Compara com criticidade estadual
- Mesma metodologia: períodos 30-dias, redução >= 15%
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy import stats
import gc

# ================== [1] CARREGAR DADOS RAIO ESTADUAL ==================
print("[1] Carregando dados RAIO estadual...")

raio_operations = []
try:
    # Carregar arquivo operacional completo
    with open('data/raw/ocorrencia_policial_operacional.json', 'r', encoding='utf-8') as f:
        raio_data = json.load(f)
    
    for item in raio_data:
        if isinstance(item, dict) and item.get('type') == 'table':
            raio_operations = item.get('data', [])
            break
except Exception as e:
    print(f"   Erro ao carregar operacional, tentando Caucaia...")
    with open('data/raw/ocorrencia_caucaia_2025.json', 'r', encoding='utf-8') as f:
        raio_data = json.load(f)
    for item in raio_data:
        if isinstance(item, dict) and item.get('type') == 'table':
            raio_operations = item.get('data', [])
            break

raio_df = pd.DataFrame(raio_operations)
raio_df['data_dt'] = pd.to_datetime(raio_df['Data'], errors='coerce')
raio_df = raio_df.dropna(subset=['data_dt'])
raio_df = raio_df.sort_values('data_dt').reset_index(drop=True)

print(f"   Total operações: {len(raio_df)}")
print(f"   Período: {raio_df['data_dt'].min()} a {raio_df['data_dt'].max()}")

# Estatísticas geográficas
cidades = raio_df['CidadeOcor'].value_counts()
print(f"\n   Cidades representadas: {len(cidades)}")
print(f"   Top 10 cidades por operações:")
for cidade, cnt in cidades.head(10).items():
    print(f"     - {cidade}: {cnt}")

# ================== [2] CARREGAR CRITICIDADE ==================
print("\n[2] Carregando tensor de criticidade...")

try:
    import torch
    criticidade_tensor = torch.load('data/tensors/criticidade_final_completo.pt').numpy()
    metadata_file = 'data/tensors/metadata_janela180d_completo.json'
except:
    criticidade_tensor = np.load('data/tensors/dataset_criticidade_janela180d_completo.pt', allow_pickle=True)
    metadata_file = 'data/tensors/metadata_janela180d.json'

with open(metadata_file, 'r', encoding='utf-8') as f:
    metadata = json.load(f)

bairro_to_idx = metadata.get('bairro_mapping', {})
print(f"   Tensor shape: {criticidade_tensor.shape}")
print(f"   Bairros/locais no modelo: {len(bairro_to_idx)}")

# ================== [3] PERÍODOS 30-DIA ==================
print("\n[3] Agrupando em períodos de 30 dias...")

start_date = raio_df['data_dt'].min().replace(hour=0, minute=0, second=0, microsecond=0)
end_date = raio_df['data_dt'].max().replace(hour=23, minute=59, second=59, microsecond=999999)

periods = []
current = start_date
while current < end_date:
    next_period = current + timedelta(days=30)
    periods.append({'start': current, 'end': next_period, 'period_id': len(periods)})
    current = next_period

print(f"   Total períodos: {len(periods)}")

# ================== [4] RAIO POR PERÍODO ==================
print("\n[4] Volume de RAIO por período (ESTADUAL)...")

raio_per_period = []
for p in periods:
    period_raio = raio_df[(raio_df['data_dt'] >= p['start']) & (raio_df['data_dt'] < p['end'])]
    
    # Cidades com operações neste período
    cidades_periodo = set(period_raio['CidadeOcor'].unique()) if len(period_raio) > 0 else set()
    
    # Bairros com operações
    bairros_periodo = set(period_raio['BairroOcor'].unique()) if len(period_raio) > 0 else set()
    
    raio_per_period.append({
        'period_id': p['period_id'],
        'start': p['start'],
        'end': p['end'],
        'raio_count': len(period_raio),
        'cidades': cidades_periodo,
        'bairros': bairros_periodo
    })

raio_periods_df = pd.DataFrame(raio_per_period)
print(f"   Min: {raio_periods_df['raio_count'].min()}, Max: {raio_periods_df['raio_count'].max()}, Média: {raio_periods_df['raio_count'].mean():.1f}")

# ================== [5] CRITICIDADE ESTADUAL ==================
print("\n[5] Criticidade média estadual por período...")

data_base = datetime(2022, 10, 1)
criticidade_per_period = []

for i, row in raio_periods_df.iterrows():
    period_id = row['period_id']
    start_idx = (row['start'] - data_base).days
    end_idx = (row['end'] - data_base).days
    
    if start_idx < 0 or end_idx > len(criticidade_tensor):
        continue
    
    # Bairros presentes neste período que estão no modelo
    bairro_indices = []
    for bairro in row['bairros']:
        if bairro in bairro_to_idx:
            bairro_indices.append(bairro_to_idx[bairro])
    
    if len(bairro_indices) > 0:
        crit_slice = criticidade_tensor[start_idx:end_idx, bairro_indices, 0]
        crit_mean = np.mean(crit_slice)
        crit_std = np.std(crit_slice)
        cobertura = len(bairro_indices) / len(row['bairros'])
    else:
        continue
    
    criticidade_per_period.append({
        'period_id': period_id,
        'start': row['start'],
        'raio_count': row['raio_count'],
        'num_cidades': len(row['cidades']),
        'num_bairros_op': len(row['bairros']),
        'num_bairros_modelo': len(bairro_indices),
        'criticidade_mean': crit_mean,
        'criticidade_std': crit_std,
        'cobertura': cobertura
    })

crit_df = pd.DataFrame(criticidade_per_period).dropna()
print(f"   Períodos válidos: {len(crit_df)}")
print(f"   Cobertura média de bairros: {crit_df['cobertura'].mean()*100:.1f}%")

# ================== [6] COMPARAR PERÍODOS ==================
print("\n[6] Comparando períodos consecutivos (ESTADUAL)...")

comparacoes = []
for i in range(len(crit_df) - 1):
    t_atual = crit_df.iloc[i]
    t_prox = crit_df.iloc[i + 1]
    
    crit_atual = t_atual['criticidade_mean']
    crit_prox = t_prox['criticidade_mean']
    raio_atual = t_atual['raio_count']
    
    var_pct = ((crit_prox - crit_atual) / crit_atual * 100) if crit_atual > 0 else 0
    
    # Classificar por intensidade
    if raio_atual > 100:
        raio_trend = "RAIO_MUITO_ALTO"
    elif raio_atual > 75:
        raio_trend = "RAIO_ALTO"
    elif raio_atual > 50:
        raio_trend = "RAIO_MEDIO"
    else:
        raio_trend = "RAIO_BAIXO"
    
    # Classificar redução
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
        'cidades': t_atual['num_cidades'],
        'bairros': t_atual['num_bairros_modelo'],
        'crit_t': crit_atual,
        'crit_t1': crit_prox,
        'var_pct': var_pct,
        'reducao_cat': reducao_cat
    })

comp_df = pd.DataFrame(comparacoes)

# ================== [7] ANÁLISE PRÁTICA ESTADUAL ==================
print("\n[7] ANÁLISE PRÁTICA - EFETIVIDADE NO CEARÁ")
print("=" * 80)

for raio_trend in ["RAIO_MUITO_ALTO", "RAIO_ALTO", "RAIO_MEDIO", "RAIO_BAIXO"]:
    subset = comp_df[comp_df['raio_trend'] == raio_trend]
    
    if len(subset) == 0:
        continue
    
    reduzido_15plus = len(subset[subset['reducao_cat'] == 'REDUZIDO_15PLUS'])
    reduzido_0a15 = len(subset[subset['reducao_cat'] == 'REDUZIDO_0A15'])
    total_reducoes = reduzido_15plus + reduzido_0a15
    
    pct_15plus = (reduzido_15plus / len(subset) * 100) if len(subset) > 0 else 0
    pct_total = (total_reducoes / len(subset) * 100) if len(subset) > 0 else 0
    
    print(f"\n{raio_trend} (n={len(subset)} períodos):")
    print(f"  Operações: {subset['raio_count_t'].mean():.0f} ± {subset['raio_count_t'].std():.0f}/período")
    print(f"  Abrangência: {subset['cidades'].mean():.1f} cidades, {subset['bairros'].mean():.0f} bairros")
    print(f"  Redução ≥ 15%: {reduzido_15plus} casos ({pct_15plus:.1f}%)")
    print(f"  Qualquer redução: {total_reducoes} casos ({pct_total:.1f}%)")
    
    reducoes = subset[subset['var_pct'] < 0]
    if len(reducoes) > 0:
        print(f"  Magnitude: {reducoes['var_pct'].mean():.1f}% (quando reduz)")

# ================== [8] DOSE-RESPOSTA ESTADUAL ==================
print("\n[8] Relação Dose-Resposta (ESTADUAL)")
print("=" * 80)

corr = comp_df[['raio_count_t', 'var_pct']].corr().iloc[0, 1]
slope, intercept, r_value, p_value, std_err = stats.linregress(comp_df['raio_count_t'], comp_df['var_pct'])

print(f"\nCorrelação (Pearson r): {corr:.4f}")
print(f"Slope: {slope:.8f}%/operação")
print(f"P-value: {p_value:.4f}")
print(f"Significância: {'SIM ✓' if p_value < 0.05 else 'NÃO ✗'}")

# ================== [9] COMPARAÇÃO CAUCAIA vs ESTADO ==================
print("\n[9] COMPARAÇÃO: CAUCAIA vs CEARÁ")
print("=" * 80)

# Dados de Caucaia (do script anterior)
caucaia_reduzido_15 = 1  # 14.3% de 7
caucaia_total_periods = 7
caucaia_r = -0.2468

# Dados estaduais
estado_reduzido_15 = len(comp_df[comp_df['reducao_cat'] == 'REDUZIDO_15PLUS'])
estado_total = len(comp_df)
estado_r = corr

print(f"\nCaucaia (município):")
print(f"  Redução ≥ 15%: {caucaia_reduzido_15}/{caucaia_total_periods} ({caucaia_reduzido_15/caucaia_total_periods*100:.1f}%)")
print(f"  Correlação dose-resposta: r = {caucaia_r:.4f}")

print(f"\nCeará (estado):")
print(f"  Redução ≥ 15%: {estado_reduzido_15}/{estado_total} ({estado_reduzido_15/estado_total*100:.1f}%)")
print(f"  Correlação dose-resposta: r = {estado_r:.4f}")

print(f"\nDiferença:")
diff_pct = (estado_reduzido_15/estado_total*100) - (caucaia_reduzido_15/caucaia_total_periods*100)
print(f"  Efetividade estadual: {diff_pct:+.1f}% vs Caucaia")

# ================== [10] CONCLUSÃO ==================
print("\n[10] CONCLUSÃO ESTRATÉGICA")
print("=" * 80)

total_15 = len(comp_df[comp_df['reducao_cat'] == 'REDUZIDO_15PLUS'])
total_qual = len(comp_df[comp_df['var_pct'] < 0])

print(f"\nEm {len(comp_df)} períodos analisados NO CEARÁ:")
print(f"  • Redução ≥ 15%: {total_15} ({total_15/len(comp_df)*100:.1f}%)")
print(f"  • Qualquer redução: {total_qual} ({total_qual/len(comp_df)*100:.1f}%)")
print(f"  • Dose-resposta: r = {corr:.4f} (esperado: r < -0.3 para eficácia)")

if total_15/len(comp_df) > 0.3:
    print("\n✅ EFETIVIDADE ALTA - Operações reduzem crimes detectavelmente")
elif total_qual/len(comp_df) > 0.5:
    print("\n⚠️  EFETIVIDADE MODERADA - Há redução em muitos períodos")
else:
    print("\n❌ EFETIVIDADE BAIXA - Operações não correlacionam com redução de crimes")

print(f"\n📊 Comparação com Caucaia: Ceará mostra {'MELHOR' if estado_r > caucaia_r else 'PIOR'} efetividade")
print("\n" + "=" * 80)
