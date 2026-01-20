#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validação Correta: Correlação entre PREVISÃO e REALIDADE

1. Para cada bairro e período:
   - Pega criticidade histórica (últimos 30 dias)
   - Prevê próximos 15 dias
   - Se crítico previsto > crítico atual → prevê AUMENTO
   - Se crítico previsto < crítico atual → prevê DIMINUIÇÃO

2. Valida com RAIO 2025:
   - Se modelo previu AUMENTO, realmente teve AUMENTO de prisões?
   - Se modelo previu DIMINUIÇÃO, realmente teve DIMINUIÇÃO?
   
Teste: correlação entre direção da previsão e direção real
"""

import json
import torch
import pandas as pd
import numpy as np
import unicodedata
import re
from pathlib import Path
from datetime import datetime, timedelta
from scipy.stats import pearsonr

def normalizar_bairro(nome):
    if not nome:
        return "DESCONHECIDO"
    nome = unicodedata.normalize('NFD', nome)
    nome = ''.join(c for c in nome if unicodedata.category(c) != 'Mn')
    nome = nome.upper().strip()
    nome = re.sub(r'\s+', ' ', nome)
    return nome

print("="*80)
print("VALIDACAO: PREVISAO vs REALIDADE (TENDENCIAS)")
print("="*80)

print("\n[1] Carregando tensor...")
criticidade = torch.load('data/tensors/criticidade_final_completo.pt', weights_only=False)
print("[OK] Shape: {} (dias, nodes, features)".format(criticidade.shape))

print("\n[2] Carregando metadata...")
with open('data/tensors/metadata_janela180d_completo.json', 'r', encoding='utf-8') as f:
    metadata = json.load(f)
bairro_mapping = metadata['bairro_mapping']
print("[OK] {} bairros".format(len(bairro_mapping)))

print("\n[3] Carregando RAIO 2025...")
with open('data/raw/ocorrencia_caucaia_2025.json', 'r', encoding='utf-8') as f:
    raw_data = json.load(f)

records = []
for item in raw_data:
    if isinstance(item, dict) and item.get('type') == 'table':
        records = item.get('data', [])
        break

print("[OK] {} operações RAIO".format(len(records)))

# Converter para DataFrame com data
df_raio = []
for rec in records:
    data_str = rec.get('Data')
    bairro = normalizar_bairro(rec.get('BairroOcor', ''))
    if data_str and bairro:
        try:
            data_obj = pd.to_datetime(data_str, errors='coerce')
            if pd.notna(data_obj):
                df_raio.append({'data': data_obj.date(), 'bairro': bairro})
        except:
            pass

df_raio = pd.DataFrame(df_raio)
print("[OK] {} operações com data válida".format(len(df_raio)))

print("\n[4] Preparar períodos de validação...")
# Dividir 2025 em períodos (a cada 30 dias)
data_base = datetime(2022, 1, 1)  # Data de referência do tensor
data_inicio_2025 = datetime(2025, 1, 1)
data_fim_2025 = datetime(2025, 12, 31)

# Índice do primeiro dia de 2025 no tensor
idx_inicio_2025 = (data_inicio_2025 - data_base).days
idx_fim_2025 = (data_fim_2025 - data_base).days + 1

print("    2025 no tensor: índices {} a {}".format(idx_inicio_2025, idx_fim_2025))

# Gerar períodos de 30 dias
periodos = []
data_atual = data_inicio_2025
while data_atual < data_fim_2025:
    data_fim_periodo = data_atual + timedelta(days=30)
    periodos.append((data_atual, data_fim_periodo))
    data_atual = data_fim_periodo

print("    {} períodos de 30 dias em 2025".format(len(periodos)))

print("\n[5] Validar PREVISAO vs REALIDADE para cada bairro/período...")

validacoes = []

for bairro, node_idx in bairro_mapping.items():
    for periodo_idx, (data_inicio_p, data_fim_p) in enumerate(periodos):
        # Índices no tensor
        idx_inicio_p = (data_inicio_p - data_base).days
        idx_fim_p = (data_fim_p - data_base).days
        
        # Criticidade dos últimos 30 dias (ANTES do período)
        idx_hist_inicio = max(0, idx_inicio_p - 30)
        crit_historico = criticidade[idx_hist_inicio:idx_inicio_p, node_idx, 0].mean().item()
        
        # Criticidade do período (PREVISAO para próximos 15 dias)
        idx_prev_inicio = idx_inicio_p
        idx_prev_fim = min(criticidade.shape[0], idx_fim_p)
        crit_previsto = criticidade[idx_prev_inicio:idx_prev_fim, node_idx, 0].mean().item()
        
        # Tendência prevista
        if crit_previsto > crit_historico:
            tendencia_prevista = "AUMENTO"
            valor_tendencia_prev = 1
        elif crit_previsto < crit_historico:
            tendencia_prevista = "DIMINUICAO"
            valor_tendencia_prev = -1
        else:
            tendencia_prevista = "ESTAVEL"
            valor_tendencia_prev = 0
        
        # Operações RAIO reais no período
        df_periodo = df_raio[
            (df_raio['data'] >= data_inicio_p.date()) &
            (df_raio['data'] < data_fim_p.date()) &
            (df_raio['bairro'] == bairro)
        ]
        
        num_ops_periodo = len(df_periodo)
        
        # Operações no período anterior (para comparar)
        data_ant_inicio = data_inicio_p - timedelta(days=30)
        data_ant_fim = data_inicio_p
        df_anterior = df_raio[
            (df_raio['data'] >= data_ant_inicio.date()) &
            (df_raio['data'] < data_ant_fim.date()) &
            (df_raio['bairro'] == bairro)
        ]
        
        num_ops_anterior = len(df_anterior)
        
        # Tendência real (se houve aumento ou diminuição)
        if num_ops_periodo > num_ops_anterior:
            tendencia_real = "AUMENTO"
            valor_tendencia_real = 1
        elif num_ops_periodo < num_ops_anterior:
            tendencia_real = "DIMINUICAO"
            valor_tendencia_real = -1
        else:
            tendencia_real = "ESTAVEL"
            valor_tendencia_real = 0
        
        # Acerto: previsão coincide com realidade?
        acerto = (valor_tendencia_prev == valor_tendencia_real)
        
        validacoes.append({
            'bairro': bairro,
            'periodo': periodo_idx,
            'data_inicio': data_inicio_p.strftime('%Y-%m-%d'),
            'crit_historico': crit_historico,
            'crit_previsto': crit_previsto,
            'tendencia_prevista': tendencia_prevista,
            'ops_anterior': num_ops_anterior,
            'ops_periodo': num_ops_periodo,
            'tendencia_real': tendencia_real,
            'acerto': acerto,
            'valor_prev': valor_tendencia_prev,
            'valor_real': valor_tendencia_real
        })

df_val = pd.DataFrame(validacoes)
print("[OK] {} registros de validação".format(len(df_val)))

print("\n[6] Calcular taxa de acerto...")

# Taxa de acerto geral
acertos_totais = df_val['acerto'].sum()
total = len(df_val)
taxa_acerto_geral = acertos_totais / total * 100 if total > 0 else 0

print("    Taxa de acerto geral: {:.1f}% ({}/{})".format(taxa_acerto_geral, acertos_totais, total))

# Por tipo de tendência
acertos_aumento = df_val[df_val['tendencia_prevista'] == 'AUMENTO']['acerto'].sum()
total_aumento = len(df_val[df_val['tendencia_prevista'] == 'AUMENTO'])
taxa_aumento = acertos_aumento / total_aumento * 100 if total_aumento > 0 else 0

acertos_diminuicao = df_val[df_val['tendencia_prevista'] == 'DIMINUICAO']['acerto'].sum()
total_diminuicao = len(df_val[df_val['tendencia_prevista'] == 'DIMINUICAO'])
taxa_diminuicao = acertos_diminuicao / total_diminuicao * 100 if total_diminuicao > 0 else 0

acertos_estavel = df_val[df_val['tendencia_prevista'] == 'ESTAVEL']['acerto'].sum()
total_estavel = len(df_val[df_val['tendencia_prevista'] == 'ESTAVEL'])
taxa_estavel = acertos_estavel / total_estavel * 100 if total_estavel > 0 else 0

print("    Quando prevê AUMENTO: {:.1f}% acerto ({}/{})".format(taxa_aumento, acertos_aumento, total_aumento))
print("    Quando prevê DIMINUIÇÃO: {:.1f}% acerto ({}/{})".format(taxa_diminuicao, acertos_diminuicao, total_diminuicao))
print("    Quando prevê ESTÁVEL: {:.1f}% acerto ({}/{})".format(taxa_estavel, acertos_estavel, total_estavel))

print("\n[7] Correlação entre valores...")
if len(df_val) > 1:
    r, p = pearsonr(df_val['valor_prev'], df_val['valor_real'])
    print("    Pearson r: {:.3f} (p={:.4f})".format(r, p))
    
    # Matriz de confusão simplificada
    print("\n    Matriz de Confusão (Previsão vs Real):")
    print("    " + "-"*50)
    
    for pred in ["AUMENTO", "DIMINUICAO", "ESTAVEL"]:
        row = df_val[df_val['tendencia_prevista'] == pred]
        if len(row) > 0:
            aug = len(row[row['tendencia_real'] == 'AUMENTO'])
            dim = len(row[row['tendencia_real'] == 'DIMINUICAO'])
            est = len(row[row['tendencia_real'] == 'ESTAVEL'])
            print("    Previu {}: AUM={:3d}  DIM={:3d}  EST={:3d}".format(pred, aug, dim, est))

print("\n[8] Análise por bairro...")
acertos_por_bairro = df_val.groupby('bairro').agg({
    'acerto': ['sum', 'count'],
}).round(2)
acertos_por_bairro.columns = ['acertos', 'total']
acertos_por_bairro['taxa_%'] = (acertos_por_bairro['acertos'] / acertos_por_bairro['total'] * 100).round(1)
acertos_por_bairro = acertos_por_bairro.sort_values('taxa_%', ascending=False)

print("\n    Top 10 bairros por taxa de acerto:")
print("    Bairro                    Taxa Acerto  Acertos/Total")
print("    " + "-"*60)
for bairro, row in acertos_por_bairro.head(10).iterrows():
    print("    {:25s}  {:6.1f}%      {:2.0f}/{:2.0f}".format(
        bairro[:25],
        row['taxa_%'],
        row['acertos'],
        row['total']
    ))

print("\n" + "="*80)
print("RESUMO")
print("="*80)

resumo = """
METODOLOGIA:
  1. Para cada bairro em cada período de 15 dias:
     - Criticidade histórica (últimos 30 dias antes)
     - Criticidade prevista (período atual)
     - Tendência: AUMENTO se previsto > histórico
     
  2. Comparar com RAIO 2025:
     - Operações no período anterior
     - Operações no período atual
     - Se ops_atual > ops_anterior → AUMENTO real
     
  3. Validar: previsão = realidade?

RESULTADOS:
  Taxa geral de acerto: {:.1f}%
  Pearson r: {:.3f} (p={:.4f})
  
  Desempenho por tendência:
    - Quando prevê AUMENTO: {:.1f}% acerto
    - Quando prevê DIMINUIÇÃO: {:.1f}% acerto
    - Quando prevê ESTÁVEL: {:.1f}% acerto

INTERPRETACAO:
  - {:.0f}%: modelo faz previsão de tendência correta
  - Se > 50%: modelo é melhor que acaso
  - Se > 70%: modelo tem poder preditivo real
  
CONCLUSAO:
  {}

OBSERVACOES:
  - Validados {} bairros
  - {} períodos de 15 dias
  - {} registros totais
  - Operações RAIO: {} em 2025
""".format(
    taxa_acerto_geral,
    r if len(df_val) > 1 else 0,
    p if len(df_val) > 1 else 1,
    taxa_aumento,
    taxa_diminuicao,
    taxa_estavel,
    taxa_acerto_geral,
    "✅ MODELO TEM PODER PREDITIVO" if taxa_acerto_geral > 60 else "⚠️ MODELO ACIMA DO ACASO" if taxa_acerto_geral > 50 else "❌ MODELO NAO CONSEGUE PREVER",
    len(acertos_por_bairro),
    len(periodos),
    len(df_val),
    len(df_raio)
)

print(resumo)

# Salvar resultados
resultado = {
    'taxa_acerto_geral_percent': taxa_acerto_geral,
    'taxa_aumento_percent': taxa_aumento,
    'taxa_diminuicao_percent': taxa_diminuicao,
    'taxa_estavel_percent': taxa_estavel,
    'pearson_r': float(r) if len(df_val) > 1 else 0,
    'pearson_p': float(p) if len(df_val) > 1 else 1,
    'validacoes_totais': len(df_val),
    'bairros_validados': len(acertos_por_bairro),
    'periodos': len(periodos),
    'operacoes_raio_2025': len(df_raio)
}

with open('outputs/validacao_previsao_raio.json', 'w', encoding='utf-8') as f:
    json.dump(resultado, f, indent=2, ensure_ascii=False)

df_val.to_csv('outputs/validacao_previsao_detalhado.csv', index=False, encoding='utf-8')

print("\n[OK] Arquivos salvos:")
print("    - outputs/validacao_previsao_raio.json")
print("    - outputs/validacao_previsao_detalhado.csv")
print("\nValidacao com janelas de 30 dias concluída!")
