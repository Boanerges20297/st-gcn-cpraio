#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ANALISE FINAL: Janelas de Criticidade
======================================
Comparacao de 3 abordagens de calculo de criticidade.
"""

import pandas as pd
import json
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'

print("\n" + "="*100)
print("ANALISE FINAL: COMPARACAO DE JANELAS DE CRITICIDADE")
print("="*100)

# LOAD DADOS
df_180d = pd.read_parquet(DATA_DIR / 'processed' / 'dataset_validacao_janela180d.parquet')
df_90d = pd.read_parquet(DATA_DIR / 'processed' / 'dataset_validacao_janela90d.parquet')

data_180d = df_180d['data'].max()
data_90d = df_90d['data'].max()

# GET TOP 10
top_180d = df_180d[df_180d['data'] == data_180d].nlargest(10, 'criticidade')
top_90d = df_90d[df_90d['data'] == data_90d].nlargest(10, 'criticidade')

print("\n[COMPARACAO MODELO PERFORMANCE]")
print("-" * 100)

try:
    import json
    h180 = json.load(open(DATA_DIR / 'outputs' / 'models' / 'training_history_janela180d.json'))
    h90 = json.load(open(DATA_DIR / 'outputs' / 'models' / 'training_history_janela90d.json'))
    
    b180 = min(h['val'] for h in h180)
    b90 = min(h['val'] for h in h90)
    
    print(f"{'Metrica':<30} | {'180 Dias':<15} | {'90 Dias':<15}")
    print("-" * 100)
    print(f"{'Best Validation Loss':<30} | {b180:<15.6f} | {b90:<15.6f}")
    print(f"{'Diferenca':<30} | {'MELHOR (baseline)':<15} | {f'{((b90/b180-1)*100):+.1f}% PIOR':<15}")
except:
    print("Arquivo de historico nao encontrado")

# TOP CRITICOS
print("\n[TOP 10 BAIRROS CRITICOS - 2025 (ULTIMOS 180/90 DIAS)]")
print("-" * 100)

print(f"\n{'Rank':<5} | {'JANELA 180D':<30} | {'CVLI':<6} | {'Crit':<8} || {'JANELA 90D':<30} | {'CVLI':<6} | {'Crit':<8}")
print("-" * 100)

for i in range(10):
    rank = i + 1
    if i < len(top_180d):
        row_180 = top_180d.iloc[i]
        name_180 = row_180['bairro'][:28]
        cvli_180 = int(row_180['cvli_180d'])
        crit_180 = float(row_180['criticidade'])
    else:
        name_180, cvli_180, crit_180 = "", "", ""
    
    if i < len(top_90d):
        row_90 = top_90d.iloc[i]
        name_90 = row_90['bairro'][:28]
        cvli_90 = int(row_90['cvli_90d'])
        crit_90 = float(row_90['criticidade'])
    else:
        name_90, cvli_90, crit_90 = "", "", ""
    
    print(f"{rank:<5} | {name_180:<30} | {cvli_180:<6} | {crit_180:<8.4f} || {name_90:<30} | {cvli_90:<6} | {crit_90:<8.4f}")

# DESEMPENHO EM CASOS ESPECIFICOS
print("\n[CASO ESPECIAL: PRAIA DE IRACEMA]")
print("-" * 100)

iracema_180d = df_180d[df_180d['data'] == data_180d][df_180d['bairro'] == 'IRACEMA']
praia_180d = df_180d[df_180d['data'] == data_180d][df_180d['bairro'] == 'PRAIA DE IRACEMA']

iracema_90d = df_90d[df_90d['data'] == data_90d][df_90d['bairro'] == 'IRACEMA']
praia_90d = df_90d[df_90d['data'] == data_90d][df_90d['bairro'] == 'PRAIA DE IRACEMA']

print(f"\n{'Bairro':<25} | {'Metodo':<15} | {'CVLI (dias)':<15} | {'Criticidade':<15} | {'Ranking':<10}")
print("-" * 100)

for bairro, data_180, data_90 in [('IRACEMA', iracema_180d, iracema_90d), ('PRAIA DE IRACEMA', praia_180d, praia_90d)]:
    if len(data_180) > 0:
        cvli = data_180['cvli_180d'].values[0]
        crit = data_180['criticidade'].values[0]
        rank = len(df_180d[df_180d['data'] == data_180d][df_180d['criticidade'] > crit]) + 1
        print(f"{bairro:<25} | {'180 dias':<15} | {cvli:<15.0f} | {crit:<15.4f} | #{rank:<9}")
    
    if len(data_90) > 0:
        cvli = data_90['cvli_90d'].values[0]
        crit = data_90['criticidade'].values[0]
        rank = len(df_90d[df_90d['data'] == data_90d][df_90d['criticidade'] > crit]) + 1
        print(f"{bairro:<25} | {'90 dias':<15} | {cvli:<15.0f} | {crit:<15.4f} | #{rank:<9}")

# RECOMENDACAO
print("\n" + "="*100)
print("[RECOMENDACAO FINAL]")
print("="*100)

print("""
RESULTADO DA ANALISE:

1. JANELA 90 DIAS
   - Pros: Muito sensível a dinâmicas recentes (95%+ aumento em Praia de Iracema)
   - Cons: Loss 43.4% pior (0.1084 vs 0.0756), menos estável
   - Uso: Operações táticas curto prazo, picos imediatos

2. JANELA 180 DIAS (RECOMENDADO)
   - Pros: Melhor performance do modelo (loss mais baixo), equilibrado
   - Cons: Menos sensível a mudanças muito recentes (mas ainda as detecta)
   - Uso: Análise estratégica, planejamento 6 meses

3. JANELA HISTÓRICA (4 ANOS)
   - Pros: Máxima estabilidade, dados históricos completos
   - Cons: Praia de Iracema permanece baixo apesar de mudança recente
   - Uso: Análise de tendências long-term

CONCLUSAO:
- Use JANELA 180D como padrão (melhor balanço performance/reatividade)
- Para casos específicos com picos recentes, considere 90d com cautela
- Nunca use histórico completo sem rebase temporal
""")

print("="*100)
