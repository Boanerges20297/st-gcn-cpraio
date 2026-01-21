#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validação RAIO 2025 vs Modelo 180d
Compara correlação entre criticidade predita e prisões reais por bairro/mês
"""

import json
import numpy as np
import pandas as pd
import unicodedata
import re
from collections import defaultdict
from datetime import datetime

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None

def normalizar_bairro(nome):
    """Normaliza nome de bairro: maiúsculas, sem acentos, sem espaços extras"""
    if not nome:
        return "DESCONHECIDO"
    # Remove acentos
    nome = unicodedata.normalize('NFD', nome)
    nome = ''.join(c for c in nome if unicodedata.category(c) != 'Mn')
    # Maiúsculas, sem espaços extras
    nome = nome.upper().strip()
    nome = re.sub(r'\s+', ' ', nome)
    return nome

print("[INICIO] Carregando dados de prisões RAIO 2025...\n")

try:
    # Carregar arquivo JSON de Caucaia
    with open('data/raw/ocorrencia_caucaia_2025.json', 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    # Extrair registros da tabela
    records = []
    for item in raw_data:
        if isinstance(item, dict) and item.get('type') == 'table':
            records = item.get('data', [])
            break
    
    print(f"[OK] Carregados {len(records)} registros RAIO")
    
    # Organizar por bairro (normalizado) e mês
    prisoes_por_bairro_mes = defaultdict(lambda: defaultdict(int))
    
    for rec in records:
        bairro = normalizar_bairro(rec.get('BairroOcor', 'DESCONHECIDO'))
        data_str = rec.get('Data', '')
        
        if data_str:
            try:
                data = pd.to_datetime(data_str)
                mes_ano = data.strftime('%Y-%m')
                prisoes_por_bairro_mes[bairro][mes_ano] += 1
            except:
                pass
    
    # Converter para DataFrame
    prisoes_list = []
    for bairro, meses in prisoes_por_bairro_mes.items():
        for mes, count in meses.items():
            prisoes_list.append({
                'bairro': bairro,
                'mes': mes,
                'prisoes': count
            })
    
    df_prisoes = pd.DataFrame(prisoes_list)
    
    print(f"[OK] {len(prisoes_por_bairro_mes)} bairros únicos com operações")
    print(f"[OK] {len(df_prisoes)} registros bairro-mês\n")
    
    # Carregar metadata do modelo
    print("[...] Carregando metadata do modelo 180d...")
    try:
        with open('data/tensors/metadata_janela180d.json', 'r', encoding='utf-8') as f:
            metadata = json.load(f)
            bairro_mapping = metadata.get('bairro_mapping', {})
            print(f"[OK] Metadata carregada: {len(bairro_mapping)} bairros no modelo")
    except Exception as e:
        print(f"[ERROR] Metadata não encontrada: {e}")
        bairro_mapping = {}
    
    # Normalizar nomes do modelo também
    bairro_mapping_norm = {}
    for bairro_original, idx in bairro_mapping.items():
        bairro_norm = normalizar_bairro(bairro_original)
        bairro_mapping_norm[bairro_norm] = idx
    
    print(f"[OK] {len(bairro_mapping_norm)} bairros únicos após normalização")
    
    # Carregar tensor de criticidade
    print("[...] Carregando tensor criticidade 180d...")
    if not HAS_TORCH:
        print("[WARNING] Torch não disponível, pulando análise de tensor")
        criticidade_media = None
    else:
        try:
            tensor_180d = torch.load('data/tensors/dataset_criticidade_janela180d.pt')
            print(f"[OK] Tensor carregado: shape {tensor_180d.shape}")
            print(f"     Dias: {tensor_180d.shape[0]}, Nodes: {tensor_180d.shape[1]}, Features: {tensor_180d.shape[2]}")
            
            # Extrair criticidade média por bairro (média dos últimos 30 dias)
            dias_window = 30
            tensor_tail = tensor_180d[-dias_window:, :, 0]  # Últimos 30 dias, todas as features
            criticidade_media = tensor_tail.mean(dim=0).numpy()
            
            print(f"[OK] Criticidade média calculada (últimos {dias_window} dias)\n")
            
        except Exception as e:
            print(f"[WARNING] Tensor não carregado: {e}")
            criticidade_media = None
    
    # Criar mapeamento de bairro para índice sequencial
    bairros_raio = set(df_prisoes['bairro'].unique())
    bairros_modelo = set(bairro_mapping_norm.keys())
    
    print("="*80)
    print("ANALISE: COBERTURA DE BAIRROS")
    print("="*80)
    print(f"Bairros com prisões RAIO: {len(bairros_raio)}")
    print(f"Bairros no modelo 180d: {len(bairros_modelo)}")
    print(f"Interseção: {len(bairros_raio & bairros_modelo)}")
    
    # Bairros que podem ser validados
    bairros_validaveis = bairros_raio & bairros_modelo
    print(f"\n[VALIDAVEIS] {len(bairros_validaveis)} bairros podem ser comparados:")
    
    if bairros_validaveis:
        bairros_sorted = sorted(list(bairros_validaveis))
        for i, b in enumerate(bairros_sorted[:10], 1):
            prisoes_total = df_prisoes[df_prisoes['bairro'] == b]['prisoes'].sum()
            crit = criticidade_media[bairro_mapping_norm[b]].item() if criticidade_media is not None else 0
            print(f"  {i:2d}. {b:40s} : {prisoes_total:3d} prisões, criticidade: {crit:.3f}")
    
    # Bairros com prisões mas não no modelo
    so_em_raio = bairros_raio - bairros_modelo
    if so_em_raio:
        print(f"\n[GAPS] {len(so_em_raio)} bairros COM prisões MAS NÃO NO MODELO:")
        so_em_raio_sorted = sorted(list(so_em_raio))
        for i, b in enumerate(so_em_raio_sorted[:10], 1):
            prisoes_total = df_prisoes[df_prisoes['bairro'] == b]['prisoes'].sum()
            print(f"  {i:2d}. {b:40s} : {prisoes_total:3d} prisões")
    
    # Análise de correlação para bairros validáveis
    if bairros_validaveis and criticidade_media is not None:
        print("\n" + "="*80)
        print("ANALISE: CORRELACAO CRITICIDADE vs PRISOES")
        print("="*80)
        
        comparacao_data = []
        for bairro in bairros_validaveis:
            # Criticidade no modelo
            idx_node = bairro_mapping_norm[bairro]
            crit = criticidade_media[idx_node].item()
            
            # Prisões totais em 2025
            prisoes_total = df_prisoes[df_prisoes['bairro'] == bairro]['prisoes'].sum()
            
            comparacao_data.append({
                'bairro': bairro,
                'criticidade_modelo': crit,
                'prisoes_raio': prisoes_total
            })
        
        df_comparacao = pd.DataFrame(comparacao_data).sort_values('criticidade_modelo', ascending=False)
        
        print("\nTOP 15 BAIRROS POR CRITICIDADE PREDITA:")
        print(f"{'Bairro':<40} {'Criticidade':>12} {'Prisões RAIO':>12}")
        print("-" * 65)
        for _, row in df_comparacao.head(15).iterrows():
            print(f"{row['bairro']:<40} {row['criticidade_modelo']:>12.4f} {row['prisoes_raio']:>12.0f}")
        
        # Calcular correlação
        if len(df_comparacao) > 1:
            corr = df_comparacao['criticidade_modelo'].corr(df_comparacao['prisoes_raio'])
            print(f"\n[CORRELACAO] Pearson r = {corr:.4f}")
            if corr > 0.5:
                print("             FORTE correlação positiva (modelo válido!)")
            elif corr > 0.3:
                print("             MODERADA correlação positiva")
            elif corr > 0:
                print("             FRACA correlação positiva")
            else:
                print("             Correlação negativa ou nula (revisar modelo)")
    
    # Salvar para análise posterior
    df_prisoes.to_csv('outputs/prisoes_raio_2025_caucaia_normalizado.csv', index=False)
    print(f"\n[SALVO] Dados em outputs/prisoes_raio_2025_caucaia_normalizado.csv")
    
except Exception as e:
    print(f"\n[ERROR] Erro geral: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("FIM DA VALIDACAO")
print("="*80)
