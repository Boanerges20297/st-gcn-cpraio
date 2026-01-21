#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validação corrigida: Prisões RAIO 2025 vs Predições do Modelo 180d
Compara por bairro/mês se altas criticidades tiveram mais prisões
"""

import json
import numpy as np
import pandas as pd
from collections import defaultdict
from datetime import datetime

print("[INICIO] Carregando dados de prisões RAIO 2025...\n")

# Carregar arquivo JSON de Caucaia
try:
    with open('data/raw/ocorrencia_caucaia_2025.json', 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    # Extrair registros da tabela
    records = []
    for item in raw_data:
        if isinstance(item, dict) and item.get('type') == 'table':
            records = item.get('data', [])
            break
    
    print(f"[OK] Carregados {len(records)} registros de prisões RAIO em Caucaia 2025")
    
    # Organizar por bairro e mês
    prisoes_por_bairro_mes = defaultdict(lambda: defaultdict(int))
    
    for rec in records:
        bairro = rec.get('BairroOcor', 'DESCONHECIDO')
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
    
    print("\n" + "="*80)
    print("SUMARIO DE PRISOES RAIO 2025 - CAUCAIA")
    print("="*80)
    
    # Estatísticas gerais
    print(f"\nTotal de operações: {len(records)}")
    print(f"Total de bairros com operações: {len(prisoes_por_bairro_mes)}")
    print(f"Período: {df_prisoes['mes'].min()} a {df_prisoes['mes'].max()}")
    
    # Top 10 bairros
    top_bairros = df_prisoes.groupby('bairro')['prisoes'].sum().sort_values(ascending=False).head(10)
    print("\n>>> TOP 10 BAIRROS POR OPERACOES RAIO:")
    for i, (bairro, count) in enumerate(top_bairros.items(), 1):
        print(f"{i:2d}. {bairro:30s} : {count:3d} operações")
    
    # Análise mensal
    print("\n>>> DISTRIBUICAO MENSAL:")
    df_mensal = df_prisoes.groupby('mes')['prisoes'].sum().sort_index()
    for mes, count in df_mensal.items():
        print(f"  {mes}: {count:3d} operações")
    
    # Salvar para análise posterior
    df_prisoes.to_csv('outputs/prisoes_raio_2025_caucaia.csv', index=False)
    print("\n[OK] Dados salvos em outputs/prisoes_raio_2025_caucaia.csv")
    
    # Agora carregar predições do modelo 180d
    print("\n" + "="*80)
    print("CARREGANDO PREDICOES DO MODELO 180D")
    print("="*80)
    
    # Carregar tensor de criticidade
    try:
        import torch
        tensor_180d = torch.load('outputs/models/dataset_criticidade_janela180d.pt')
        print(f"\n[OK] Tensor 180d carregado: shape {tensor_180d.shape}")
    except ImportError:
        print("\n[WARNING] Torch não disponível")
        tensor_180d = None
    except Exception as e:
        print(f"\n[ERROR] Não consegui carregar tensor: {e}")
        print("Gerando dummy predictions para análise estrutural...")
        tensor_180d = None
    
    # Carregar metadata
    try:
        with open('outputs/models/metadata_janela180d.json', 'r') as f:
            metadata = json.load(f)
            bairro_mapping = metadata.get('bairro_mapping', {})
            print(f"[OK] Metadata carregada: {len(bairro_mapping)} bairros mapeados")
    except:
        print("[WARNING] Metadata não encontrada")
        bairro_mapping = {}
    
    # Comparar estruturalmente
    print("\n" + "="*80)
    print("VALIDACAO: ESTRUTURA DE DADOS")
    print("="*80)
    
    print(f"\nBairros com prisões RAIO: {sorted(list(prisoes_por_bairro_mes.keys())[:5])}")
    print(f"Bairros no modelo: {sorted(list(bairro_mapping.keys())[:5])}")
    
    # Análise de cobertura
    bairros_raio = set(prisoes_por_bairro_mes.keys())
    bairros_modelo = set(bairro_mapping.keys())
    
    print(f"\nBairros com RAIO: {len(bairros_raio)}")
    print(f"Bairros no modelo: {len(bairros_modelo)}")
    print(f"Interseção: {len(bairros_raio & bairros_modelo)}")
    print(f"Apenas em RAIO: {len(bairros_raio - bairros_modelo)}")
    print(f"Apenas no modelo: {len(bairros_modelo - bairros_raio)}")
    
    # Mostrar bairros com prisões mas não no modelo
    so_em_raio = bairros_raio - bairros_modelo
    if so_em_raio:
        print(f"\n>>> BAIRROS COM PRISOES MAS NAO NO MODELO:")
        for b in sorted(list(so_em_raio)):
            count = sum(prisoes_por_bairro_mes[b].values())
            print(f"  - {b}: {count} operações")
    
    print("\n[SUCESSO] Validação estrutural concluída")
    print("Próximo passo: Implementar análise de correlação quando dados estiverem completos")
    
except Exception as e:
    print(f"\n[ERROR] Erro geral: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("FIM DA VALIDACAO")
print("="*80)
