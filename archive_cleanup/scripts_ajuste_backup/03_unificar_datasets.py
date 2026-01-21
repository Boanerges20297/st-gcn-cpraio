#!/usr/bin/env python3
"""
Sprint 1, Task 1.3: Unificar dados operacionais com dados de tropas

Objetivos:
- Combinar datasets operacional e tropas
- Criar features agregadas (por dia/região)
- Gerar dataset unificado para o modelo

Usage:
    python scripts_ajuste/03_unificar_datasets.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from src.data_loader import parse_operational_json, parse_tropa_dataset

def main():
    print("\n" + "="*70)
    print("Sprint 1, Task 1.3: UNIFICAR DATASETS OPERACIONAL + TROPAS")
    print("="*70)
    
    # Carregar datasets
    print("\n[1] Carregando datasets...")
    
    df_operacional = parse_operational_json("data/raw/ocorrencia_policial_operacional.json")
    print(f"    - Operacional: {len(df_operacional)} registros")
    
    # Carregar tropa (mantemos, mas NÃO o usaremos para extração de PRISÕES)
    df_tropas = parse_tropa_dataset("data/raw/ocorrencias_tropa.json")
    print(f"    - Tropas: {len(df_tropas)} registros")

    # EXTRAÇÃO ESPECIAL: utilizar apenas o arquivo operacional para PRISÕES em 2025
    print("\n[1.1] Extraindo registros de prisão (2025) apenas do arquivo operacional...")
    is_prisao = df_operacional['natureza'].astype(str).str.contains('PRIS', case=False, na=False) | df_operacional['natureza'].astype(str).str.contains('MANDADO DE PRISÃO', case=False, na=False)
    df_prisoes_2025 = df_operacional[is_prisao & (df_operacional['data'].dt.year == 2025)].copy()
    print(f"    - Prisões 2025 (operacional): {len(df_prisoes_2025)} registros")
    # Salvar arquivo separado de prisões 2025
    os.makedirs('data/processed', exist_ok=True)
    prisoes_out = 'data/processed/prisoes_2025.parquet'
    tmp = df_prisoes_2025.copy()
    tmp['data'] = tmp['data'].astype(str)
    if len(tmp) > 0:
        tmp.to_parquet(prisoes_out, index=False)
        print(f"    - Prisões 2025 salvas em: {prisoes_out}")
    else:
        print("    - Nenhuma prisão 2025 encontrada no arquivo operacional")
    
    # Preparar para unificação
    print("\n[2] Preparando datasets para unificação...")
    
    # Seleção de colunas comuns
    cols_operacional = [
        'data', 'municipio', 'lat', 'long', 'area_faccao', 
        'is_cvli', 'total_armas', 'total_drogas_g', 
        'has_large_seizure', 'has_weapons_drugs', 'dinheiro_apreendido'
    ]
    
    cols_tropas = [
        'data', 'municipio', 'lat', 'long',
        'total_drogas_g', 'total_armas'
    ]
    
    df_op_select = df_operacional[cols_operacional].copy()
    df_op_select['fonte'] = 'operacional'
    df_op_select['area_faccao'] = df_op_select['area_faccao'].fillna('SEM_FACCAO')
    df_op_select = df_op_select.fillna(0)
    
    df_tr_select = df_tropas[cols_tropas].copy()
    df_tr_select['fonte'] = 'tropas'
    df_tr_select['area_faccao'] = 'TROPA'
    df_tr_select['is_cvli'] = 0
    df_tr_select['has_large_seizure'] = df_tr_select['total_drogas_g'] >= 1000
    df_tr_select['has_weapons_drugs'] = (df_tr_select['total_armas'] > 0) & (df_tr_select['total_drogas_g'] > 0)
    df_tr_select['dinheiro_apreendido'] = 0
    df_tr_select = df_tr_select.fillna(0)
    
    # Reordenar colunas
    col_order = [
        'data', 'municipio', 'lat', 'long', 'area_faccao', 'fonte',
        'is_cvli', 'total_armas', 'total_drogas_g', 
        'has_large_seizure', 'has_weapons_drugs', 'dinheiro_apreendido'
    ]
    
    df_op_select = df_op_select[col_order]
    df_tr_select = df_tr_select[col_order]
    
    # Combinar
    df_unified = pd.concat([df_op_select, df_tr_select], ignore_index=True)
    df_unified = df_unified.sort_values('data').reset_index(drop=True)
    
    print(f"    - Unified: {len(df_unified)} registros total")
    print(f"      · Operacional: {(df_unified['fonte'] == 'operacional').sum()}")
    print(f"      · Tropas: {(df_unified['fonte'] == 'tropas').sum()}")
    
    # Estatísticas
    print("\n[3] Estatísticas do dataset unificado:")
    print(f"    - Data range: {df_unified['data'].min().date()} → {df_unified['data'].max().date()}")
    print(f"    - Dias únicos: {df_unified['data'].dt.date.nunique()}")
    print(f"    - Municípios: {df_unified['municipio'].nunique()}")
    print(f"    - Facções: {df_unified['area_faccao'].nunique()}")
    
    print(f"\n    CVLI:")
    print(f"      - Total: {df_unified['is_cvli'].sum()}")
    print(f"      - % da operacional: {100 * df_unified[df_unified['fonte']=='operacional']['is_cvli'].sum() / len(df_unified[df_unified['fonte']=='operacional']):.1f}%")
    
    print(f"\n    Apreensões grandes (>= 1kg):")
    print(f"      - Total: {df_unified['has_large_seizure'].sum()}")
    print(f"      - % do total: {100 * df_unified['has_large_seizure'].sum() / len(df_unified):.1f}%")
    
    print(f"\n    Arma + Droga:")
    print(f"      - Total: {df_unified['has_weapons_drugs'].sum()}")
    print(f"      - % do total: {100 * df_unified['has_weapons_drugs'].sum() / len(df_unified):.1f}%")
    
    print(f"\n    Drogas totais: {df_unified['total_drogas_g'].sum():.0f}g")
    print(f"    Armas totais: {df_unified['total_armas'].sum():.0f}")
    print(f"    Dinheiro total: R$ {df_unified['dinheiro_apreendido'].sum():.2f}")
    
    # Agregações por dia e município
    print("\n[4] Agregações por dia-município...")
    
    df_daily = df_unified.groupby(['data', 'municipio']).agg({
        'is_cvli': 'sum',
        'total_armas': 'sum',
        'total_drogas_g': 'sum',
        'has_large_seizure': 'sum',
        'has_weapons_drugs': 'sum',
        'dinheiro_apreendido': 'sum'
    }).reset_index()
    
    df_daily.columns = [
        'data', 'municipio',
        'n_cvli', 'n_armas', 'drogas_total_g', 'n_large_seizures', 'n_weapons_drugs', 'dinheiro_total'
    ]
    
    df_daily['crime_count'] = df_daily['n_cvli'] + df_daily['n_armas'] + df_daily['n_large_seizures']
    
    print(f"    - {len(df_daily)} registros de agregação diária")
    print(f"    - Date range: {df_daily['data'].min()} → {df_daily['data'].max()}")
    
    # Salvar datasets
    print("\n[5] Salvando datasets...")
    
    # Salvar unified completo
    output_unified = "data/processed/unified_2025.parquet"
    os.makedirs(os.path.dirname(output_unified), exist_ok=True)
    df_unified['data'] = df_unified['data'].astype(str)
    df_unified.to_parquet(output_unified, index=False)
    print(f"    - Unified: {output_unified}")
    
    # Salvar agregação diária
    output_daily = "data/processed/unified_daily_2025.parquet"
    df_daily['data'] = df_daily['data'].astype(str)
    df_daily.to_parquet(output_daily, index=False)
    print(f"    - Daily: {output_daily}")
    
    # Salvar CSV sample
    output_csv = "outputs/unified_sample.csv"
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_unified_display = df_unified.copy()
    df_unified_display['data'] = df_unified_display['data'].astype(str)
    df_unified_display.head(100).to_csv(output_csv, index=False)
    print(f"    - Sample CSV: {output_csv}")
    
    # Mostrar exemplo
    print("\n[6] Exemplo de dados agregados (top 20 por crime_count):")
    top_daily = df_daily.nlargest(20, 'crime_count')[['data', 'municipio', 'n_cvli', 'drogas_total_g', 'n_armas', 'crime_count']]
    print(top_daily.to_string())
    
    print("\n" + "="*70)
    print("[✓] Sprint 1 Completa - Normalização de dados concluída!")
    print("="*70)
    
    print("\n[PRÓXIMOS PASSOS - Sprint 2]:")
    print("    1. Integrar features territoriais (facções GeoJSON)")
    print("    2. Calcular feature importance (CVLI 3x, drogas 2x, armas 2x)")
    print("    3. Gerar tensors para modelo (features x days x nodes)")
    print("\n")

if __name__ == "__main__":
    main()
