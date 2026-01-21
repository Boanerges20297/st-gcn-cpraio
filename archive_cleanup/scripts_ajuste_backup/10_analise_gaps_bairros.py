#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Refinamento do Modelo 180d - Análise de Bairros Faltantes
Identifica gaps na cobertura e gera plano de ação
"""

import json
import pandas as pd
import unicodedata
import re
from collections import defaultdict

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
print("ANALISE DE GAPS: BAIRROS FALTANTES NO MODELO")
print("="*80 + "\n")

# Carregar prisões RAIO
print("[1] Carregando dados RAIO...")
with open('data/raw/ocorrencia_caucaia_2025.json', 'r', encoding='utf-8') as f:
    raw_data = json.load(f)

records = []
for item in raw_data:
    if isinstance(item, dict) and item.get('type') == 'table':
        records = item.get('data', [])
        break

bairros_raio_raw = [normalizar_bairro(rec.get('BairroOcor', '')) for rec in records]
bairros_raio = set(bairros_raio_raw)
print(f"[OK] {len(bairros_raio)} bairros únicos com operações RAIO\n")

# Carregar modelo metadata
print("[2] Carregando metadata do modelo...")
with open('data/tensors/metadata_janela180d.json', 'r', encoding='utf-8') as f:
    metadata = json.load(f)
    bairro_mapping = metadata.get('bairro_mapping', {})

bairros_modelo = set(normalizar_bairro(b) for b in bairro_mapping.keys())
print(f"[OK] {len(bairros_modelo)} bairros no modelo 180d\n")

# Análise de gaps
print("="*80)
print("ANALISE DE COBERTURA")
print("="*80)

interseção = bairros_raio & bairros_modelo
gaps = bairros_raio - bairros_modelo
extras = bairros_modelo - bairros_raio

print(f"\nBairros com prisões RAIO: {len(bairros_raio)}")
print(f"Bairros no modelo: {len(bairros_modelo)}")
print(f"Interseção: {len(interseção)}")
print(f"GAPS (prisões mas não no modelo): {len(gaps)}")
print(f"Extras (modelo mas sem prisões): {len(extras)}")

# Detalhar gaps
if gaps:
    print("\n" + "="*80)
    print(f"BAIRROS COM GAPS ({len(gaps)} total)")
    print("="*80)
    
    # Contar operações por gap
    gap_counts = defaultdict(int)
    for bairro_raw in bairros_raio_raw:
        if bairro_raw in gaps:
            gap_counts[bairro_raw] += 1
    
    gap_sorted = sorted(gap_counts.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\n{'Bairro':<40} {'Operações RAIO':>15}")
    print("-" * 55)
    
    total_ops_faltando = 0
    for bairro, count in gap_sorted:
        print(f"{bairro:<40} {count:>15}")
        total_ops_faltando += count
    
    print("-" * 55)
    print(f"{'TOTAL':<40} {total_ops_faltando:>15}")

# Estatísticas
print("\n" + "="*80)
print("IMPACTO POTENCIAL")
print("="*80)

df_raio = pd.DataFrame({'bairro': bairros_raio_raw})
df_gaps = df_raio[df_raio['bairro'].isin(gaps)]

pct_ops_faltando = (total_ops_faltando / len(records)) * 100

print(f"\nOperações RAIO em bairros faltantes: {total_ops_faltando} / {len(records)} ({pct_ops_faltando:.1f}%)")
print(f"Impacto na validação: {pct_ops_faltando:.0f}% dos dados não podem ser validados")

if pct_ops_faltando > 30:
    print("\n⚠️  CRÍTICO: Mais de 30% das operações não podem ser validadas!")
    print("   Ação necessária: Expandir modelo para incluir gaps")
elif pct_ops_faltando > 10:
    print("\n⚠️  IMPORTANTE: 10-30% das operações em bairros não mapeados")
    print("   Recomendação: Expandir modelo antes de conclusões finais")
else:
    print("\n✓ Cobertura aceitável (menos de 10% de gaps)")

# Sugestões
print("\n" + "="*80)
print("PLANO DE ACAO")
print("="*80)

print("""
[1] ETAPA 1: Mapeamento Completo
    - Adicionar os bairros faltantes ao mapa de treinamento
    - Executar ETL 180d novamente com cobertura 100%
    - Tempo estimado: ~5 minutos

[2] ETAPA 2: Retreinamento
    - Executar graph builder com dados expandidos
    - Treinar ST-GCN com novo tensor
    - Tempo estimado: ~10 minutos

[3] ETAPA 3: Revalidação
    - Comparar correlação ANTES vs DEPOIS
    - Validar se correlação com RAIO melhora
    - Estimar impacto dos bairros faltantes

[4] ETAPA 4: Análise Comparativa
    - Gerar relatório de melhorias
    - Documentar impacto da expansão
""")

print("\n" + "="*80)
print("ARQUIVO DE SAIDA")
print("="*80)

# Salvar gaps para referência
gaps_data = []
for bairro, count in gap_sorted:
    gaps_data.append({
        'bairro_normalizado': bairro,
        'operacoes_raio': count,
        'status': 'FALTANTE_NO_MODELO'
    })

df_gaps_export = pd.DataFrame(gaps_data)
df_gaps_export.to_csv('outputs/analise_gaps_bairros.csv', index=False)
print(f"\n[SALVO] Lista de gaps em: outputs/analise_gaps_bairros.csv")

# Salvar bairros para código
print(f"\n[INFO] Bairros faltantes para adicionar:")
print(f"       {json.dumps(sorted(list(gaps)), indent=2)}")

print("\n" + "="*80)
print("FIM DA ANALISE")
print("="*80)
