"""
ANÁLISE APROFUNDADA - EFETIVIDADE RAIO 2025 vs MODELO 180d
===========================================================
"""

import pandas as pd
import numpy as np
import torch
import json
from pathlib import Path
from datetime import datetime
import sys
import warnings

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import config

print("=" * 80)
print("ANÁLISE DETALHADA: CORRELAÇÃO ENTRE MODELO 180d E ATUAÇÕES RAIO 2025")
print("=" * 80)

# Carrega dados
print("\n[CARREGANDO DADOS]")
df_raio = pd.read_parquet(config.DATA_PROCESSED / "prisoes_raio_2025.parquet")

# Extrai records
records = []
for data_array in df_raio['data']:
    if data_array is not None and hasattr(data_array, '__iter__'):
        for item in data_array:
            if isinstance(item, dict):
                records.append(item)

df_raio = pd.DataFrame(records)
df_raio['data_operacao'] = pd.to_datetime(df_raio['Data'], errors='coerce')
df_raio['bairro'] = df_raio['BairroOcor'].fillna('DESCONHECIDO').str.upper()
df_raio = df_raio[df_raio['data_operacao'].dt.year == 2025]

print(f"[OK] {len(df_raio)} operações RAIO 2025")

# Carrega criticidade
criticidade_tensor = torch.load(config.TENSOR_DIR / "dataset_criticidade_janela180d.pt", weights_only=False)
criticidade_tensor = criticidade_tensor.squeeze(-1)  # (dias, nós)

with open(config.TENSOR_DIR / "metadata_janela180d.json", encoding='utf-8') as f:
    metadata = json.load(f)

bairro_mapping = metadata.get('bairro_mapping', {})
idx_to_bairro = {v: k for k, v in bairro_mapping.items()}

print(f"[OK] Criticidade 180d carregada: {criticidade_tensor.shape}")

# Gera predições 2025
start_date = datetime(2022, 1, 1)
dates = pd.date_range(start=start_date, periods=criticidade_tensor.shape[0], freq='D')
mask_2025 = (dates.year == 2025)

criticidade_2025 = criticidade_tensor[mask_2025].mean(dim=0)  # Média de 2025 (nós,)

# Cria dataframe de criticidade média 2025 por bairro
df_criticidade = pd.DataFrame({
    'bairro': [idx_to_bairro.get(i, f"NODE_{i}") for i in range(len(criticidade_2025))],
    'criticidade_media_2025': criticidade_2025.numpy()
})

# Conta prisões por bairro
df_prisoes_count = df_raio.groupby('bairro').size().reset_index(name='num_prisoes')

# Merge
df_merged = pd.merge(df_criticidade, df_prisoes_count, on='bairro', how='left')
df_merged['num_prisoes'] = df_merged['num_prisoes'].fillna(0)

# Sort por criticidade
df_merged = df_merged.sort_values('criticidade_media_2025', ascending=False)

print("\n[TOP 20 CRÍTICOS vs PRISÕES RAIO 2025]")
print(f"\n{'Rank':<5} {'Bairro':<30} {'Criticidade':<12} {'Prisões':<10} {'Padrão':<15}")
print("-" * 70)

for i, (_, row) in enumerate(df_merged.head(20).iterrows(), 1):
    bairro = row['bairro'][:28]
    criticidade = row['criticidade_media_2025']
    prisoes = int(row['num_prisoes'])
    
    # Classifica padrão
    if criticidade > 0.9 and prisoes < 5:
        pattern = "[ALTO/BAIXO RAIO]"
    elif criticidade < 0.3 and prisoes > 20:
        pattern = "[BAIXO/ALTO RAIO]"
    elif criticidade > 0.7 and prisoes > 10:
        pattern = "[ESPERADO]"
    else:
        pattern = "[MISTO]"
    
    print(f"{i:<5} {bairro:<30} {criticidade:<12.4f} {prisoes:<10} {pattern:<15}")

# Análise estratégica
print("\n[PADRÕES IDENTIFICADOS]")

alto_crit_baixa_acao = df_merged[(df_merged['criticidade_media_2025'] > 0.8) & (df_merged['num_prisoes'] < 5)]
print(f"\n1. CRÍTICOS COM BAIXA AÇÃO RAIO (Crit > 0.8, Prisões < 5):")
print(f"   Total: {len(alto_crit_baixa_acao)} bairros")
print(f"   Exemplos: {', '.join(alto_crit_baixa_acao.head(5)['bairro'].tolist())}")

baixo_crit_alta_acao = df_merged[(df_merged['criticidade_media_2025'] < 0.3) & (df_merged['num_prisoes'] > 10)]
print(f"\n2. BAIXO CRÍTICOS COM ALTA AÇÃO RAIO (Crit < 0.3, Prisões > 10):")
print(f"   Total: {len(baixo_crit_alta_acao)} bairros")
if len(baixo_crit_alta_acao) > 0:
    print(f"   Exemplos: {', '.join(baixo_crit_alta_acao.head(5)['bairro'].tolist())}")

esperado = df_merged[(df_merged['criticidade_media_2025'] > 0.7) & (df_merged['num_prisoes'] > 10)]
print(f"\n3. PADRÃO ESPERADO (Crit > 0.7, Prisões > 10):")
print(f"   Total: {len(esperado)} bairros")
if len(esperado) > 0:
    print(f"   Exemplos: {', '.join(esperado.head(5)['bairro'].tolist())}")

# Conclusão
print("\n[INTERPRETAÇÃO]")
print(f"Correlação Pearson: {df_merged['criticidade_media_2025'].corr(df_merged['num_prisoes']):.4f}")

if len(alto_crit_baixa_acao) > 20:
    print("\n[!] ACHADO CRÍTICO:")
    print("    Mais de 20 bairros com ALTA criticidade mas BAIXA ação RAIO")
    print("    Possíveis causas:")
    print("    - RAIO pode ter prioridades diferentes do modelo")
    print("    - Modelo pode estar superestimando criticidade em certas regiões")
    print("    - Recursos RAIO limitados para cobrir todas as áreas críticas")
elif len(esperado) > len(df_merged) * 0.5:
    print("\n[OK] ALINHAMENTO SATISFATÓRIO:")
    print("    Mais de 50% dos bairros seguem padrão esperado")
    print("    Correlação positiva entre críticos e atuações")
else:
    print("\n[!] DESALINHAMENTO PARCIAL:")
    print("    Recomendação: Revisar critérios de alocação de recursos RAIO")

print("\n" + "=" * 80)
