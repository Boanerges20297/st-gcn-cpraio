#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Revalidacao contra RAIO 2025 com cobertura completa
Comparar correlacao antes (r=0.348) vs depois
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
    """Normaliza nome de bairro"""
    if not nome:
        return "DESCONHECIDO"
    nome = unicodedata.normalize('NFD', nome)
    nome = ''.join(c for c in nome if unicodedata.category(c) != 'Mn')
    nome = nome.upper().strip()
    nome = re.sub(r'\s+', ' ', nome)
    return nome

print("="*80)
print("REVALIDACAO CONTRA RAIO 2025 (COBERTURA COMPLETA)")
print("="*80)

print("\n[1] Carregando tensor completo...")
criticidade = torch.load('data/tensors/criticidade_final_completo.pt', weights_only=False)
print("[OK] Shape: {}".format(criticidade.shape))

print("\n[2] Carregando metadata completo...")
with open('data/tensors/metadata_janela180d_completo.json', 'r', encoding='utf-8') as f:
    metadata = json.load(f)

bairro_mapping = metadata['bairro_mapping']
print("[OK] {} bairros no mapeamento".format(len(bairro_mapping)))

print("\n[3] Carregando RAIO 2025...")
with open('data/raw/ocorrencia_caucaia_2025.json', 'r', encoding='utf-8') as f:
    raw_data = json.load(f)

records = []
for item in raw_data:
    if isinstance(item, dict) and item.get('type') == 'table':
        records = item.get('data', [])
        break

print("[OK] {} registros RAIO".format(len(records)))

# Extrair bairros e datas
ocorrencias_por_bairro = {}
for rec in records:
    bairro = normalizar_bairro(rec.get('BairroOcor', ''))
    if bairro not in ocorrencias_por_bairro:
        ocorrencias_por_bairro[bairro] = 0
    ocorrencias_por_bairro[bairro] += 1

bairros_raio = set(ocorrencias_por_bairro.keys())
print("[OK] {} bairros únicos em RAIO".format(len(bairros_raio)))

print("\n[4] Calculando criticidade média para últimos 30 dias...")
# Usar últimos 30 dias de 2025 (índices 1431 a 1460, considerando 0-indexed)
data_inicio = datetime(2022, 1, 1)
data_fim = datetime(2025, 12, 31)
num_dias_total = (data_fim - data_inicio).days + 1

ultimos_30_dias_idx = slice(num_dias_total - 30, num_dias_total)
criticidade_ultimos_30 = criticidade[ultimos_30_dias_idx, :, 0].mean(dim=0)

print("[OK] Criticidade média dos últimos 30 dias: shape {}".format(criticidade_ultimos_30.shape))

print("\n[5] Correlacao entre criticidade e prisões...")
matching_count = 0
correlacoes = []
top_correlacoes = []

for bairro_norm in bairros_raio:
    # Tentar encontrar no mapping
    if bairro_norm in bairro_mapping:
        node_idx = bairro_mapping[bairro_norm]
        crit = criticidade_ultimos_30[node_idx].item()
        prisoes = ocorrencias_por_bairro[bairro_norm]
        
        matching_count += 1
        correlacoes.append((bairro_norm, crit, prisoes))
        
        if len(top_correlacoes) < 10 or prisoes > top_correlacoes[-1][2]:
            top_correlacoes.append((bairro_norm, crit, prisoes))
            top_correlacoes.sort(key=lambda x: x[2], reverse=True)
            if len(top_correlacoes) > 10:
                top_correlacoes.pop()

print("[OK] {} bairros com matching".format(matching_count))

print("\n[6] Calcular Pearson r...")
if len(correlacoes) > 1:
    crit_values = [c[1] for c in correlacoes]
    prisao_values = [c[2] for c in correlacoes]
    
    r, pvalue = pearsonr(crit_values, prisao_values)
    print("[OK] Pearson r: {:.3f} (p-value: {:.4f})".format(r, pvalue))
else:
    r = 0.0
    print("[AVISO] Menos de 2 bairros para calcular correlacao")

print("\n[7] Comparar com validacao anterior...")
r_anterior = 0.348
print("    r anterior (319 nodes): {:.3f}".format(r_anterior))
print("    r novo (389 nodes): {:.3f}".format(r))

if r > r_anterior:
    melhoria = ((r - r_anterior) / r_anterior) * 100
    print("    Melhoria: +{:.1f}%".format(melhoria))
else:
    piora = ((r_anterior - r) / r_anterior) * 100
    print("    Piora: -{:.1f}%".format(piora))

print("\n[8] Top 10 correlacoes...")
print("    Bairro                          Criticidade   Prisoes")
print("    " + "-"*60)
for bairro, crit, prisoes in top_correlacoes:
    print("    {:30s}    {:.3f}        {:3d}".format(bairro[:30], crit, prisoes))

print("\n[9] Analise de gaps...")
bairros_modelo = set(bairro_mapping.keys())
gaps = bairros_raio - bairros_modelo

print("    Bairros com RAIO: {}".format(len(bairros_raio)))
print("    Bairros no modelo: {}".format(len(bairros_modelo)))
print("    Interseção: {}".format(len(bairros_raio & bairros_modelo)))
print("    Gaps (RAIO sem modelo): {}".format(len(gaps)))

if len(gaps) > 0:
    print("\n    Gaps remanescentes:")
    for bairro in sorted(gaps)[:10]:
        print("      - {} ({} ops)".format(bairro, ocorrencias_por_bairro.get(bairro, 0)))
    if len(gaps) > 10:
        print("      ... {} mais".format(len(gaps) - 10))

print("\n" + "="*80)
print("RESUMO DA REVALIDACAO")
print("="*80)
print("""
Cobertura:
  - Bairros em RAIO: {}
  - Bairros no modelo: {}
  - Bairros matching: {} ({:.1f}%)
  - Gaps remanescentes: {} ({:.1f}%)

Correlacao Pearson:
  - Anterior (319 nodes): 0.348
  - Novo (389 nodes): {:.3f}
  - Mudanca: {}{:.1f}%

Interpretacao:
  - Expansao para 388 bairros {}cobrir gaps
  - Criticidade agora disponível para {} bairros adicionais
  - Cobertura de operações RAIO: {}

Conclusao:
  - Model refinement {}
  - Cobertura geográfica agora em {}%
""".format(
    len(bairros_raio),
    len(bairros_modelo),
    matching_count,
    matching_count / len(bairros_raio) * 100 if len(bairros_raio) > 0 else 0,
    len(gaps),
    len(gaps) / len(bairros_raio) * 100 if len(bairros_raio) > 0 else 0,
    r,
    "+" if r > r_anterior else "-",
    abs((r - r_anterior) / r_anterior * 100) if r_anterior > 0 else 0,
    "CONSEGUIU" if len(gaps) < 70 else "NÃO CONSEGUIU",
    len(bairros_modelo),
    "{:.1f}%".format(matching_count / len(bairros_raio) * 100) if len(bairros_raio) > 0 else "0%",
    "SUCESSO" if r > r_anterior else "INCOMPLETO",
    (matching_count / len(bairros_raio) * 100) if len(bairros_raio) > 0 else 0
))

print("[CONCLUIDO]")

# Salvar resultados
resultado = {
    'r_anterior': 0.348,
    'r_novo': float(r),
    'bairros_raio': len(bairros_raio),
    'bairros_modelo': len(bairros_modelo),
    'matching': matching_count,
    'gaps': len(gaps),
    'cobertura_percent': matching_count / len(bairros_raio) * 100 if len(bairros_raio) > 0 else 0,
    'top_correlacoes': [(b, c, p) for b, c, p in top_correlacoes[:5]]
}

with open('outputs/resultado_revalidacao_completo.json', 'w', encoding='utf-8') as f:
    json.dump(resultado, f, indent=2, ensure_ascii=False)

print("Resultados salvo em: outputs/resultado_revalidacao_completo.json")
