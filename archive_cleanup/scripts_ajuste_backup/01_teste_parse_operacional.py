#!/usr/bin/env python3
"""
Sprint 1, Task 1.1: Teste de parsing do JSON operacional

Objetivos:
- Validar extração de coordenadas (lat_long)
- Validar detecção de CVLI
- Validar cálculo de seizures grandes e arma+droga
- Verificar taxa de sucesso de normalização
- Gerar estatísticas básicas

Usage:
    python scripts_ajuste/01_teste_parse_operacional.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from src.data_loader import parse_operational_json
import json

def main():
    print("\n" + "="*70)
    print("Sprint 1, Task 1.1: TESTE PARSE JSON OPERACIONAL")
    print("="*70)
    
    json_path = "data/raw/ocorrencia_policial_operacional.json"
    
    # Verificar arquivo
    if not os.path.exists(json_path):
        print(f"\n[ERRO] Arquivo não encontrado: {json_path}")
        print("Procurando arquivos JSON disponíveis...")
        for root, dirs, files in os.walk("data/raw"):
            for f in files:
                if f.endswith('.json'):
                    print(f"  - {os.path.join(root, f)}")
        return
    
    print(f"\n[1] Testando parse do JSON operacional...")
    print(f"    Arquivo: {json_path}")
    
    try:
        df = parse_operational_json(json_path)
    except Exception as e:
        print(f"\n[ERRO] Falha ao fazer parse: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Análises
    print(f"\n[2] Análises do dataset normalizado:")
    print(f"    - Total de registros: {len(df)}")
    print(f"    - Data range: {df['data'].min()} → {df['data'].max()}")
    print(f"    - Dias: {(df['data'].max() - df['data'].min()).days}")
    
    print(f"\n[3] Verificação de CVLI:")
    cvli_count = df['is_cvli'].sum()
    print(f"    - Total CVLI: {cvli_count} ({100*cvli_count/len(df):.1f}%)")
    if cvli_count > 0:
        cvli_sample = df[df['is_cvli']]['natureza'].head(5).values
        print(f"    - Exemplos: {cvli_sample}")
    
    print(f"\n[4] Verificação de Apreensões Grandes (>= 1kg):")
    large_seizure_count = df['has_large_seizure'].sum()
    print(f"    - Total: {large_seizure_count} ({100*large_seizure_count/len(df):.1f}%)")
    if large_seizure_count > 0:
        print(f"    - Range de droga (g): {df[df['has_large_seizure']]['total_drogas_g'].min():.0f} → {df[df['has_large_seizure']]['total_drogas_g'].max():.0f}")
    
    print(f"\n[5] Verificação de Arma + Droga:")
    weapons_drugs_count = df['has_weapons_drugs'].sum()
    print(f"    - Total: {weapons_drugs_count} ({100*weapons_drugs_count/len(df):.1f}%)")
    if weapons_drugs_count > 0:
        sample = df[df['has_weapons_drugs']][['total_armas', 'total_drogas_g']].head(5)
        print(f"    - Amostra:")
        print(sample.to_string(index=False))
    
    print(f"\n[6] Verificação de Coordenadas:")
    valid_coords = df[['lat', 'long']].notna().all(axis=1).sum()
    print(f"    - Registros com coords válidas: {valid_coords} / {len(df)}")
    print(f"    - Lat range: {df['lat'].min():.4f} → {df['lat'].max():.4f}")
    print(f"    - Long range: {df['long'].min():.4f} → {df['long'].max():.4f}")
    
    print(f"\n[7] Verificação de Municípios:")
    print(f"    - Únicos: {df['municipio'].nunique()}")
    top_municipios = df['municipio'].value_counts().head(5)
    print(f"    - Top 5:")
    for mun, count in top_municipios.items():
        print(f"      · {mun}: {count}")
    
    print(f"\n[8] Verificação de Naturezas (top 10):")
    top_naturezas = df['natureza'].value_counts().head(10)
    for nat, count in top_naturezas.items():
        print(f"    · {nat}: {count}")
    
    print(f"\n[9] Verificação de Facções:")
    print(f"    - Únicas: {df['area_faccao'].nunique()}")
    top_faccoes = df['area_faccao'].value_counts().head(10)
    for facc, count in top_faccoes.items():
        print(f"      · {facc}: {count}")
    
    print(f"\n[10] Estatísticas de Apreensões:")
    print(f"    Armas (total):")
    print(f"      - Mean: {df['total_armas'].mean():.2f}")
    print(f"      - Std: {df['total_armas'].std():.2f}")
    print(f"      - Max: {df['total_armas'].max():.0f}")
    
    print(f"    Drogas (gramas):")
    print(f"      - Mean: {df['total_drogas_g'].mean():.2f}g")
    print(f"      - Std: {df['total_drogas_g'].std():.2f}g")
    print(f"      - Max: {df['total_drogas_g'].max():.0f}g")
    
    print(f"    Dinheiro Apreendido:")
    print(f"      - Mean: R$ {df['dinheiro_apreendido'].mean():.2f}")
    print(f"      - Total: R$ {df['dinheiro_apreendido'].sum():.2f}")
    
    # Salvar amostra de dados normalizados
    output_csv = "outputs/operacional_normalizado_sample.csv"
    df.head(100).to_csv(output_csv, index=False)
    print(f"\n[11] Amostra salva em: {output_csv}")
    
    # Salvar dataset completo
    output_parquet = "data/processed/operacional_normalizado.parquet"
    os.makedirs(os.path.dirname(output_parquet), exist_ok=True)
    df.to_parquet(output_parquet, index=False)
    print(f"[12] Dataset completo salvo em: {output_parquet}")
    
    print("\n" + "="*70)
    print("[✓] Teste concluído com sucesso!")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
