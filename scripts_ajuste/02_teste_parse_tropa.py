#!/usr/bin/env python3
"""
Sprint 1, Task 1.2: Teste de parsing das tropas

Objetivos:
- Validar extração de presos, drogas, armas de narrativas
- Validar parsing de coordenadas (decimal e DMS)
- Verificar taxa de sucesso de parsing
- Gerar estatísticas básicas

Usage:
    python scripts_ajuste/02_teste_parse_tropa.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from src.data_loader import parse_tropa_dataset, normalize_tropa_coordinates, parse_tropa_narrative

def main():
    print("\n" + "="*70)
    print("Sprint 1, Task 1.2: TESTE PARSE JSON TROPAS")
    print("="*70)
    
    # Test 1: Funções auxiliares
    print("\n[1] Teste de funções auxiliares:")
    
    # Test coordinate parsing
    test_coords = [
        ("-3.7668038,-38.584197", "Decimal"),
        ("-5°15'53.4\"S,-37°56'37.8\"W", "DMS"),
        ("-5°15',  -37°56'", "DMS Parcial"),
    ]
    
    for coord_str, tipo in test_coords:
        lat, long = normalize_tropa_coordinates(coord_str)
        if lat is not None:
            print(f"    ✓ {tipo}: ({lat:.4f}, {long:.4f})")
        else:
            print(f"    ✗ {tipo}: Falha ao parsear")
    
    # Test narrative parsing
    print(f"\n[2] Teste de parsing de narrativas:")
    
    test_narratives = [
        ("02 presos, 250g de cocaína e 1 revólver apreendidos", "operacao_padrão"),
        ("Grande apreensão: 5kg de maconha, 3 revólveres, 5 presos", "megaoperacao"),
        ("Patrulhamento rotineiro sem incidentes", "operacao_vazia"),
    ]
    
    for narrative, tipo in test_narratives:
        result = parse_tropa_narrative(narrative, "2025-01-15")
        print(f"\n    {tipo}:")
        print(f"      Entrada: {narrative[:50]}...")
        print(f"      → Presos: {result['total_presos']}")
        print(f"      → Drogas: {result['total_drogas_g']:.0f}g")
        print(f"      → Armas: {result['total_armas']}")
        print(f"      → Score: {result['success_score']:.2%}")
    
    # Test 3: Parse dataset completo
    print(f"\n[3] Testando parse dataset completo de tropas:")
    
    df_tropa = parse_tropa_dataset()
    
    if len(df_tropa) == 0:
        print("    [!] Nenhum arquivo de tropas encontrado. Criando dados de exemplo...")
        
        # Criar dados de exemplo para teste
        example_data = [
            {
                'Data': '2025-01-15',
                'Municipio': 'Fortaleza',
                'Natureza': 'Operação de Patrulhamento',
                'Observação': '03 presos, 500g de cocaína, 2 revólveres apreendidos'
            },
            {
                'Data': '2025-01-16',
                'Municipio': 'Sobral',
                'Natureza': 'Operação Especial',
                'Observação': 'Megaoperação com 1.5kg de maconha, 5 revólveres, 8 presos em local -5°15\'53.4"S,-37°56\'37.8"W'
            },
            {
                'Data': '2025-01-17',
                'Municipio': 'Caucaia',
                'Natureza': 'Patrulha Noturna',
                'Observação': 'Coordenadas: -3.7668038,-38.584197. Sem incidentes maiores'
            }
        ]
        
        with open("data/raw/ocorrencias_tropa_exemplo.json", 'w', encoding='utf-8') as f:
            import json
            json.dump(example_data, f, ensure_ascii=False, indent=2)
        
        print(f"    [+] Arquivo de exemplo criado: data/raw/ocorrencias_tropa_exemplo.json")
        
        df_tropa = parse_tropa_dataset("data/raw/ocorrencias_tropa_exemplo.json")
    
    print(f"\n    Total de registros: {len(df_tropa)}")
    
    if len(df_tropa) > 0:
        print(f"\n[4] Análises do dataset de tropas:")
        print(f"    - Registros com sucesso de parsing: {(df_tropa['success_score'] > 0).sum()}")
        print(f"    - Score médio de sucesso: {df_tropa['success_score'].mean():.2%}")
        
        print(f"\n[5] Estatísticas de Presos:")
        print(f"    - Total: {df_tropa['total_presos'].sum():.0f}")
        print(f"    - Mean: {df_tropa['total_presos'].mean():.1f}")
        print(f"    - Max: {df_tropa['total_presos'].max():.0f}")
        
        print(f"\n[6] Estatísticas de Drogas (gramas):")
        print(f"    - Total: {df_tropa['total_drogas_g'].sum():.0f}g")
        print(f"    - Mean: {df_tropa['total_drogas_g'].mean():.1f}g")
        print(f"    - Max: {df_tropa['total_drogas_g'].max():.0f}g")
        
        print(f"\n[7] Estatísticas de Armas:")
        print(f"    - Total: {df_tropa['total_armas'].sum():.0f}")
        print(f"    - Mean: {df_tropa['total_armas'].mean():.1f}")
        print(f"    - Max: {df_tropa['total_armas'].max():.0f}")
        
        print(f"\n[8] Registros com Coordenadas:")
        coords_valid = df_tropa[['lat', 'long']].notna().all(axis=1).sum()
        print(f"    - Com coords: {coords_valid} / {len(df_tropa)}")
        
        print(f"\n[9] Distribuição de Operações:")
        op_dist = df_tropa['operacao_tipo'].value_counts()
        for op_tipo, count in op_dist.items():
            print(f"    - {op_tipo}: {count}")
        
        # Salvar sample
        output_csv = "outputs/tropas_normalizado_sample.csv"
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        df_tropa.head(20).to_csv(output_csv, index=False)
        print(f"\n[10] Amostra salva em: {output_csv}")
        
        # Salvar completo
        output_parquet = "data/processed/tropas_normalizado.parquet"
        os.makedirs(os.path.dirname(output_parquet), exist_ok=True)
        df_tropa.to_parquet(output_parquet, index=False)
        print(f"[11] Dataset completo salvo em: {output_parquet}")
        
        # Mostrar exemplo
        print(f"\n[12] Exemplo de registros parseados:")
        print(df_tropa[['data', 'municipio', 'total_presos', 'total_drogas_g', 'total_armas', 'success_score']].head(5).to_string())
    
    print("\n" + "="*70)
    print("[✓] Teste concluído!")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
