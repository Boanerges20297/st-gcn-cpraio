#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETL para janela 180d com COBERTURA COMPLETA (390 bairros)
Incorpora os 70 bairros faltantes do RAIO 2025

Etapas:
1. Carregar novo mapeamento (390 bairros)
2. Recalcular criticidade com cobertura 100%
3. Gerar tensores para treinamento
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import warnings
import os
import sys

# Fix encoding
if sys.platform.startswith('win'):
    os.environ['PYTHONIOENCODING'] = 'utf-8'

warnings.filterwarnings('ignore')

print("\n" + "="*80)
print("ETL JANELA 180d COM COBERTURA COMPLETA")
print("="*80)

# ============================================================================
# ETAPA 1: CARREGAR MAPEAMENTO COMPLETO
# ============================================================================
print("\n[1] Carregando novo mapeamento de bairros...")

metadata_path = Path("data/tensors/metadata_janela180d_completo.json")
if not metadata_path.exists():
    print("[ERRO] {} nao encontrado".format(metadata_path))
    exit(1)

with open(metadata_path, 'r', encoding='utf-8') as f:
    metadata_completo = json.load(f)

bairro_mapping_completo = metadata_completo['bairro_mapping']
print(f"[OK] {len(bairro_mapping_completo)} bairros no mapeamento completo")
print(f"    - Original: 318 bairros")
print(f"    - Adicionados: {len(bairro_mapping_completo) - 318} bairros")
print(f"    - Total: {len(bairro_mapping_completo)} bairros")

# ============================================================================
# ETAPA 2: CARREGAR DADOS CVLI E CALCULAR CRITICIDADE
# ============================================================================
print("\n[2] Carregando dados CVLI...")

# Tentar múltiplas fontes
cvli_sources = [
    Path("data/processed/base_consolidada.parquet"),  # Gerada pelo ETL após filtro CVLI
    Path("data/raw/ocorrencia_caucaia_processada.parquet"),  # Arquivo histórico
]

df_cvli = None
for source in cvli_sources:
    if source.exists():
        df_cvli = pd.read_parquet(source)
        print(f"[OK] Dados carregados de {source.name}")
        break

if df_cvli is None:
    print("[ERRO] Nenhuma fonte de dados CVLI encontrada")
    print(f"   Procurado em: {[str(s) for s in cvli_sources]}")
    exit(1)

# Normalizar coluna de data
if 'data_ocorrencia' in df_cvli.columns:
    df_cvli['data_ocorrencia'] = pd.to_datetime(df_cvli['data_ocorrencia'])
elif 'data_hora' in df_cvli.columns:
    df_cvli['data_ocorrencia'] = pd.to_datetime(df_cvli['data_hora'])
elif 'data' in df_cvli.columns:
    df_cvli['data_ocorrencia'] = pd.to_datetime(df_cvli['data'])
else:
    print("[ERRO] Coluna de data não encontrada")
    exit(1)

print(f"[OK] {len(df_cvli)} registros CVLI carregados")
print(f"    Data range: {df_cvli['data_ocorrencia'].min().date()} a {df_cvli['data_ocorrencia'].max().date()}")

# ============================================================================
# ETAPA 3: CRIAR ESTRUTURA DE TENSORES
# ============================================================================
print("\n[3] Preparando tensores para janela 180d...")

# Data range: 2022-01-01 a 2025-12-31
data_inicio = datetime(2022, 1, 1)
data_fim = datetime(2025, 12, 31)
num_dias = (data_fim - data_inicio).days + 1
num_bairros = len(bairro_mapping_completo)

print(f"[OK] Período: {data_inicio.date()} a {data_fim.date()}")
print(f"    Dias: {num_dias}")
print(f"    Bairros: {num_bairros}")

# Criar tensor vazio
criticidade = np.zeros((num_dias, num_bairros, 1), dtype=np.float32)

# ============================================================================
# ETAPA 4: MAPEAR CVLI AOS BAIRROS
# ============================================================================
print("\n[4] Mapeando ocorrências CVLI aos bairros...")

# Normalizar nome de bairro
def normalizar_bairro(nome):
    if pd.isna(nome):
        return None
    nome = str(nome).strip().upper()
    nome = nome.replace('Á', 'A').replace('À', 'A').replace('Ã', 'A').replace('Â', 'A')
    nome = nome.replace('É', 'E').replace('È', 'E').replace('Ê', 'E')
    nome = nome.replace('Í', 'I')
    nome = nome.replace('Ó', 'O').replace('Ò', 'O').replace('Õ', 'O').replace('Ô', 'O')
    nome = nome.replace('Ú', 'U')
    nome = nome.replace('Ç', 'C')
    return nome.strip()

# Criar coluna de bairro normalizado
# Tentar múltiplas colunas possíveis
if 'descricao_bairro' in df_cvli.columns:
    df_cvli['bairro_norm'] = df_cvli['descricao_bairro'].apply(normalizar_bairro)
elif 'local_oficial' in df_cvli.columns:
    df_cvli['bairro_norm'] = df_cvli['local_oficial'].apply(normalizar_bairro)
elif 'bairro' in df_cvli.columns:
    df_cvli['bairro_norm'] = df_cvli['bairro'].apply(normalizar_bairro)
elif 'bairro_ciops' in df_cvli.columns:
    df_cvli['bairro_norm'] = df_cvli['bairro_ciops'].apply(normalizar_bairro)
else:
    print("[ERRO] Coluna de bairro não encontrada")
    print(f"   Colunas disponíveis: {df_cvli.columns.tolist()}")
    exit(1)

# Contar ocorrências por bairro (verificação)
bairro_counts = df_cvli['bairro_norm'].value_counts()
print(f"[OK] {len(bairro_counts)} bairros únicos em CVLI")
print(f"    Top 5 bairros em CVLI:")
for i, (bairro, count) in enumerate(bairro_counts.head(5).items(), 1):
    status = "✓" if bairro in bairro_mapping_completo else "✗"
    try:
        print(f"      {status} {i}. {bairro}: {count} ocorrências")
    except:
        print(f"      {status} {i}. [erro de encoding]: {count} ocorrências")

# ============================================================================
# ETAPA 5: CALCULAR CRITICIDADE COM JANELA 180d
# ============================================================================
print("\n[5] Calculando criticidade com janela 180d...")

# Para cada dia, calcular criticidade = proporção de dias com ocorrências nos últimos 180 dias
for dia_idx in range(num_dias):
    data_atual = data_inicio + timedelta(days=dia_idx)
    data_inicio_janela = data_atual - timedelta(days=179)  # 180 dias incluindo hoje
    
    # Filtrar dados da janela
    df_janela = df_cvli[
        (df_cvli['data_ocorrencia'] >= data_inicio_janela) &
        (df_cvli['data_ocorrencia'] <= data_atual)
    ]
    
    # Contar dias com ocorrências por bairro
    dias_ocorrencia = df_janela.groupby('bairro_norm').size()
    
    # Calcular criticidade (dias_com_ocorrências / 180)
    for bairro, dias_count in dias_ocorrencia.items():
        if bairro in bairro_mapping_completo:
            bairro_idx = bairro_mapping_completo[bairro]
            criticidade[dia_idx, bairro_idx, 0] = min(1.0, dias_count / 180.0)
    
    if (dia_idx + 1) % 200 == 0:
        print(f"    {dia_idx + 1}/{num_dias} dias processados...")

print(f"[OK] Criticidade calculada para {num_dias} dias")

# ============================================================================
# ETAPA 6: NORMALIZAR CRITICIDADE
# ============================================================================
print("\n[6] Normalizando criticidade...")

# Usar percentis diários (como no modelo original)
criticidade_norm = np.copy(criticidade).astype(np.float32)

for dia_idx in range(num_dias):
    valores = criticidade[dia_idx, :, 0]
    valores_nao_zero = valores[valores > 0]
    
    if len(valores_nao_zero) > 0:
        p25 = np.percentile(valores_nao_zero, 25)
        p75 = np.percentile(valores_nao_zero, 75)
        
        # Normalizar para [0, 1]
        if p75 > p25:
            mask = valores > 0
            criticidade_norm[dia_idx, mask, 0] = (valores[mask] - p25) / (p75 - p25)
            criticidade_norm[dia_idx, mask, 0] = np.clip(criticidade_norm[dia_idx, mask, 0], 0, 1)

print(f"[OK] Criticidade normalizada")
print(f"    Shape: {criticidade_norm.shape}")
print(f"    Min: {criticidade_norm.min():.4f}")
print(f"    Max: {criticidade_norm.max():.4f}")
print(f"    Mean: {criticidade_norm.mean():.4f}")

# ============================================================================
# ETAPA 7: SALVAR TENSORES
# ============================================================================
print("\n[7] Salvando tensores...")

tensor_path = Path("data/tensors/tensores_janela180d_completo.npy")
tensor_path.parent.mkdir(parents=True, exist_ok=True)
np.save(tensor_path, criticidade_norm)
print(f"[OK] Tensor salvo em {tensor_path}")
print(f"    Shape: {criticidade_norm.shape}")
print(f"    Size: {criticidade_norm.nbytes / 1024 / 1024:.2f} MB")

# ============================================================================
# ETAPA 8: SALVAR METADADOS COMPLETOS
# ============================================================================
print("\n[8] Salvando metadados completos...")

metadata_output = {
    'data_inicio': data_inicio.isoformat(),
    'data_fim': data_fim.isoformat(),
    'num_dias': num_dias,
    'num_bairros': num_bairros,
    'num_features': 1,
    'janela_dias': 180,
    'bairro_mapping': bairro_mapping_completo,
    'normalizacao': 'percentil_diario',
    'tensor_shape': list(criticidade_norm.shape),
    'data_criacao': datetime.now().isoformat()
}

metadata_output_path = Path("data/tensors/metadata_janela180d_completo.json")
with open(metadata_output_path, 'w', encoding='utf-8') as f:
    json.dump(metadata_output, f, indent=2, ensure_ascii=False)
print(f"[OK] Metadados salvos em {metadata_output_path}")

# ============================================================================
# ETAPA 9: RELATÓRIO FINAL
# ============================================================================
print("\n" + "="*80)
print("RELATÓRIO FINAL - ETL JANELA 180d COMPLETO")
print("="*80)

print(f"""
DADOS DE ENTRADA:
  - Registros: {len(df_cvli)}
  - Data range: {df_cvli['data_ocorrencia'].min().date()} a {df_cvli['data_ocorrencia'].max().date()}

CONFIGURAÇÃO:
  - Janela: 180 dias
  - Período: {data_inicio.date()} a {data_fim.date()}
  - Número de dias: {num_dias}
  - Número de bairros: {num_bairros}
    * Original (180d): 318
    * Adicionados (RAIO gaps): {num_bairros - 318}
    * Total: {num_bairros}

TENSOR DE CRITICIDADE:
  - Shape: {criticidade_norm.shape}
  - Dtype: {criticidade_norm.dtype}
  - Size: {criticidade_norm.nbytes / 1024 / 1024:.2f} MB
  - Sparsidade: {(criticidade_norm == 0).sum() / criticidade_norm.size * 100:.1f}%

ESTATÍSTICAS:
  - Min: {criticidade_norm.min():.4f}
  - Max: {criticidade_norm.max():.4f}
  - Mean: {criticidade_norm.mean():.4f}
  - Std: {criticidade_norm.std():.4f}

ARQUIVOS DE SAÍDA:
  [OK] {tensor_path}
  [OK] {metadata_output_path}

PROXIMOS PASSOS:
  1. Atualizar dashboard com novo tensor
  2. Revalidar com dados CVLI apenas
""")

print("[OK] ETL 180d com cobertura completa!")
