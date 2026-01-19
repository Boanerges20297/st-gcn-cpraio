#!/usr/bin/env python3
"""
Sprint 1, Task 1.2: Teste de parsing das tropas

Objetivos:
- Validar extração de presos, drogas, armas de narrativas estruturadas
- Validar parsing de coordenadas
- Verificar taxa de sucesso de parsing
- Gerar estatísticas básicas

Usage:
    python scripts_ajuste/02_teste_parse_tropa_v2.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from src.data_loader import parse_tropa_dataset, normalize_tropa_coordinates

def main():
    print("\n" + "="*70)
    print("Sprint 1, Task 1.2: TESTE PARSE JSON TROPAS (PHPMyAdmin)")
    print("="*70)
    
    # Test 1: Funções auxiliares
    print("\n[1] Teste de funções auxiliares - Conversão de Coordenadas:")
    
    test_coords = [
        ("-3.7668038,-38.584197", "Decimal"),
        ("-5°15'53.4\"S,-37°56'37.8\"W", "DMS completo"),
    ]
    
    for coord_str, tipo in test_coords:
        lat, long = normalize_tropa_coordinates(coord_str)
        if lat is not None:
            print(f"    [OK] {tipo}: ({lat:.4f}, {long:.4f})")
        else:
            print(f"    [FAIL] {tipo}: Falha ao parsear")
    
    # Test 2: Parse dataset completo
    print(f"\n[2] Testando parse dataset completo de tropas:")
    
    df_tropa = parse_tropa_dataset("data/raw/ocorrencias_tropa.json")
    
    print(f"\n    Total de registros: {len(df_tropa)}")
    
    if len(df_tropa) > 0:
        print(f"\n[3] Análises do dataset de tropas:")
        print(f"    - Registros com sucesso de parsing: {(df_tropa['success_score'] > 0).sum()}")
        print(f"    - Score médio de sucesso: {df_tropa['success_score'].mean():.2%}")
        
        print(f"\n[4] Estatísticas de Presos:")
        print(f"    - Total: {df_tropa['total_presos'].sum():.0f}")
        print(f"    - Mean: {df_tropa['total_presos'].mean():.1f}")
        
        print(f"\n[5] Estatísticas de Drogas (gramas):")
        print(f"    - Total: {df_tropa['total_drogas_g'].sum():.0f}g")
        print(f"    - Mean: {df_tropa['total_drogas_g'].mean():.1f}g")
        print(f"    - Max: {df_tropa['total_drogas_g'].max():.0f}g")
        print(f"    - Registros com drogas: {(df_tropa['total_drogas_g'] > 0).sum()}")
        
        print(f"\n[6] Estatísticas de Armas:")
        print(f"    - Total: {df_tropa['total_armas'].sum():.0f}")
        print(f"    - Mean: {df_tropa['total_armas'].mean():.1f}")
        print(f"    - Max: {df_tropa['total_armas'].max():.0f}")
        print(f"    - Registros com armas: {(df_tropa['total_armas'] > 0).sum()}")
        
        print(f"\n[7] Registros com Coordenadas:")
        coords_valid = df_tropa[['lat', 'long']].notna().all(axis=1).sum()
        print(f"    - Com coords: {coords_valid} / {len(df_tropa)}")
        
        print(f"\n[8] Municípios mais frequentes:")
        top_municipios = df_tropa['municipio'].value_counts().head(10)
        for mun, count in top_municipios.items():
            print(f"    - {mun}: {count}")
        
        print(f"\n[9] Distribuição de Operações:")
        op_dist = df_tropa['operacao_tipo'].value_counts()
        for op_tipo, count in op_dist.items():
            print(f"    - {op_tipo}: {count}")
        
        print(f"\n[10] Score de sucesso (distribuição):")
        score_bins = pd.cut(df_tropa['success_score'], bins=[0, 0.25, 0.5, 0.75, 1.0])
        for bin_range, count in score_bins.value_counts().sort_index().items():
            print(f"    - {bin_range}: {count}")
        
        # Salvar sample
        output_csv = "outputs/tropas_normalizado_sample.csv"
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        df_tropa.head(50).to_csv(output_csv, index=False)
        print(f"\n[11] Amostra salva em: {output_csv}")
        
        # Salvar completo
        output_parquet = "data/processed/tropas_normalizado.parquet"
        os.makedirs(os.path.dirname(output_parquet), exist_ok=True)
        
        # Converter colunas datetime para string para evitar problemas com parquet
        df_tropa_save = df_tropa.copy()
        df_tropa_save['data'] = df_tropa_save['data'].astype(str)
        df_tropa_save.to_parquet(output_parquet, index=False)
        print(f"[12] Dataset completo salvo em: {output_parquet}")
        
        # Mostrar exemplo
        print(f"\n[13] Exemplo de registros parseados (top 10 por drogas):")
        top_records = df_tropa.nlargest(10, 'total_drogas_g')[['data', 'municipio', 'total_drogas_g', 'total_armas', 'success_score']]
        print(top_records.to_string())
    
    print("\n" + "="*70)
    print("[✓] Teste concluído!")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
