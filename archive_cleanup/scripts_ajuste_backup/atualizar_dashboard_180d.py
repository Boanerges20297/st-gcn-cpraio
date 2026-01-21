#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
atualizar_dashboard_180d.py

Atualiza criticidade do dashboard usando o tensor ST-GCN de 180 dias
(Usa criticidade ATUAL em vez de predição futura para manter modelo original intacto)
"""

import torch
import numpy as np
import pandas as pd
import json
from datetime import datetime
from pathlib import Path

# ================== [1] CARREGAR DADOS 180D ==================
print("[1] Carregando dados ST-GCN 180 dias...")

# Caminhos
tensor_dir = Path('data/tensors')
output_dir = Path('outputs/reports')
output_dir.mkdir(parents=True, exist_ok=True)

# Carregar tensor de criticidade - USAR NOVO TENSOR COM CVLI CORRIGIDO
# Prioridade: novo tensor (388 bairros) > tensor antigo (319 bairros)
tensor_paths = [
    tensor_dir / 'tensores_janela180d_completo.npy',  # NOVO - com CVLI apenas
    tensor_dir / 'dataset_criticidade_janela180d.pt',   # ANTIGO - fallback
]

criticidade_tensor = None
for tensor_path in tensor_paths:
    if tensor_path.exists():
        if str(tensor_path).endswith('.npy'):
            criticidade_tensor = np.load(tensor_path)
        else:
            criticidade_tensor = torch.load(tensor_path).numpy()
        print(f"   Tensor carregado: {tensor_path.name}")
        break

if criticidade_tensor is None:
    print("   ERRO: Nenhum tensor encontrado!")
    exit(1)

print(f"   Tensor shape: {criticidade_tensor.shape}")

# Metadata - USAR NOVA METADATA COM 388 BAIRROS
metadata_paths = [
    tensor_dir / 'metadata_janela180d_completo.json',  # NOVA - com CVLI apenas
    tensor_dir / 'metadata_janela180d.json',            # ANTIGA - fallback
]

metadata = None
for metadata_path in metadata_paths:
    if metadata_path.exists():
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        print(f"   Metadata carregada: {metadata_path.name}")
        break

if metadata is None:
    print("   ERRO: Nenhuma metadata encontrada!")
    exit(1)

bairro_mapping = metadata.get('bairro_mapping', {})
print(f"   Bairros no modelo: {len(bairro_mapping)}")

# ================== [2] GERAR CRITICIDADE ATUAL ==================
print("\n[2] Calculando criticidade atual por bairro...")

# Usar últimos 15 dias para estimar criticidade ATUAL
criticidade_atual = criticidade_tensor[-15:, :, 0].mean(axis=0)  # (nodes,)
print(f"   Criticidade calculada para {len(criticidade_atual)} nós")

# Normalizar (min-max)
crit_min = criticidade_atual.min()
crit_max = criticidade_atual.max()
if crit_max > crit_min:
    criticidade_normalizada = (criticidade_atual - crit_min) / (crit_max - crit_min)
else:
    criticidade_normalizada = np.zeros_like(criticidade_atual)

# ================== [3] MAPEAR PARA BAIRROS ==================
print("\n[3] Mapeando criticidade para bairros...")

bairro_criticidade = {}
for bairro, idx in bairro_mapping.items():
    if 0 <= idx < len(criticidade_normalizada):
        crit = float(criticidade_normalizada[idx])
        
        # Classificar em BAIXO, MÉDIO, ALTO, CRÍTICO
        if crit > 0.8:
            nivel = "CRÍTICO"
        elif crit > 0.5:
            nivel = "ALTO"
        elif crit > 0.2:
            nivel = "MÉDIO"
        else:
            nivel = "BAIXO"
        
        bairro_criticidade[bairro] = {
            'criticidade': crit,
            'nivel': nivel,
            'data_calculo': datetime.now().isoformat()
        }

print(f"   Bairros mapeados: {len(bairro_criticidade)}")

# ================== [4] DISTRIBUIÇÃO ==================
print("\n[4] Distribuição de criticidade:")

distribuicao = {'CRÍTICO': 0, 'ALTO': 0, 'MÉDIO': 0, 'BAIXO': 0}
for data in bairro_criticidade.values():
    distribuicao[data['nivel']] += 1

for nivel in ['CRÍTICO', 'ALTO', 'MÉDIO', 'BAIXO']:
    count = distribuicao[nivel]
    pct = (count / len(bairro_criticidade) * 100) if bairro_criticidade else 0
    print(f"   {nivel}: {count} ({pct:.1f}%)")

# ================== [5] SALVAR PREDIÇÕES ==================
print("\n[5] Salvando em outputs/reports/...")

df_pred = pd.DataFrame([
    {
        'local_oficial': bairro,
        'risco_previsto': data['criticidade'],
        'nivel_risco': data['nivel']
    }
    for bairro, data in bairro_criticidade.items()
])

print(f"   Total de bairros com criticidade calculada: {len(df_pred)}")

# Carregar base consolidada para mapear regiões
bairro_regiao = {}
parquets_para_tentar = [
    'data/processed/base_consolidada.parquet',
    'data/processed/base_consolidada_orcrim_v3.parquet',
    'data/processed/base_consolidada_orcrim_v2.parquet'
]

for parquet_path in parquets_para_tentar:
    try:
        print(f"   Tentando carregar {parquet_path}...")
        base_consolidada = pd.read_parquet(parquet_path)
        
        # Mapear bairros para regiões
        if 'local_oficial' in base_consolidada.columns and 'regiao_sistema' in base_consolidada.columns:
            for _, row in base_consolidada.iterrows():
                bairro = row.get('local_oficial')
                regiao = row.get('regiao_sistema')
                if bairro and regiao:
                    bairro_regiao[bairro] = regiao
        
        if len(bairro_regiao) > 0:
            print(f"   ✓ Mapeamento de regiões carregado: {len(bairro_regiao)} bairros")
            break
    except Exception as e:
        print(f"   Erro ao carregar {parquet_path}: {e}")
        continue

# Se não conseguiu mapear, criar mapeamento genérico
if len(bairro_regiao) == 0:
    print("   ⚠️ Usando mapeamento genérico (fallback)")
    # Lista conhecida de bairros por região
    capital_bairros = {'BOM JARDIM', 'BORA', 'CENTRO', 'PRAIA DE IRACEMA', 'MEIRELES', 'ALDEOTA', 'CAIS DO PORTO'}
    
    for bairro in bairro_criticidade.keys():
        if bairro in capital_bairros:
            bairro_regiao[bairro] = 'CAPITAL'
        elif 'FORTALEZA' in bairro:
            bairro_regiao[bairro] = 'CAPITAL'
        else:
            bairro_regiao[bairro] = 'INTERIOR'  # Default

# Dividir predições por região com tratamento de NaN
print("\n   Dividindo predições por região...")
distribuicao_regiao = {'CAPITAL': 0, 'RMF': 0, 'INTERIOR': 0}

for regiao in ['CAPITAL', 'RMF', 'INTERIOR']:
    df_regiao = df_pred[df_pred['local_oficial'].map(lambda x: bairro_regiao.get(x, 'INTERIOR') == regiao)]
    
    if len(df_regiao) > 0:
        output_file = output_dir / f'pred_{regiao.lower()}_bairros.csv'
        df_regiao.to_csv(output_file, index=False)
        print(f"   ✓ {output_file}: {len(df_regiao)} locais")
        distribuicao_regiao[regiao] = len(df_regiao)
    else:
        # Criar arquivo vazio se não existir dados
        output_file = output_dir / f'pred_{regiao.lower()}_bairros.csv'
        print(f"   ⚠️  {regiao}: nenhum local mapeado")

# ================== [6] RESUMO ==================
print("\n[6] RESUMO")
print("=" * 80)
print(f"Criticidade atualizada com modelo ST-GCN 180 dias")
print(f"Período base: Últimos 15 dias do tensor (2022-10-01 até hoje)")
print(f"Normalizacao: Min-Max (0-1)")
print(f"Bairros com criticidade: {len(bairro_criticidade)}")
print(f"\nDistribuição:")
for nivel in ['CRÍTICO', 'ALTO', 'MÉDIO', 'BAIXO']:
    print(f"  {nivel}: {distribuicao[nivel]}")

print(f"\n✅ Predições salvas em outputs/reports/")
print(f"   Dashboard está pronto para recarregar (F5 no navegador)")
print("=" * 80)

