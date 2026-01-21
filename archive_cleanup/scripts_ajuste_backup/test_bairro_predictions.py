#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Validação de Predições por Bairro para Fortaleza
Verifica se o sistema agora está usando as 140 predições de bairro
em vez das 7 predições de local_oficial
"""

import sys
import pandas as pd
import geopandas as gpd
from pathlib import Path
from src.config import ARTIFACTS, DATA_GRAPH, REPORT_DIR

def test_bairro_predictions():
    print("\n" + "="*80)
    print("VALIDAÇÃO: PREDIÇÕES DISCRIMINADAS POR BAIRRO PARA FORTALEZA")
    print("="*80)
    
    # 1. Verificar novo arquivo de predições
    pred_file = ARTIFACTS['CAPITAL']['prediction']
    print(f"\n1. Arquivo de predições configurado: {pred_file.name}")
    
    if not pred_file.exists():
        print(f"   ❌ ERRO: Arquivo não encontrado em {pred_file}")
        return False
    
    pred_df = pd.read_csv(pred_file)
    print(f"   ✓ Arquivo carregado: {len(pred_df)} linhas")
    print(f"   Colunas: {list(pred_df.columns)}")
    
    # 2. Validar que temos predições por bairro (não por local_oficial)
    print(f"\n2. Verificar granularidade das predições")
    print(f"   Locais únicos em predições: {pred_df['local_oficial'].nunique()}")
    print(f"   Primeiros 5 locais: {list(pred_df['local_oficial'].head(5))}")
    
    # 3. Comparar com GeoJSON (deve ter ~138 bairros)
    geojson_path = DATA_GRAPH / "fortaleza_bairros.geojson"
    gdf = gpd.read_file(geojson_path)
    print(f"\n3. Validação com GeoJSON")
    print(f"   Total de bairros em GeoJSON: {len(gdf)}")
    print(f"   Total de predições carregadas: {len(pred_df)}")
    
    # 4. Verificar cobertura
    bairros_geojson = set(gdf['name'].str.upper().unique())
    bairros_pred = set(pred_df['local_oficial'].str.upper().unique())
    
    print(f"\n4. Cobertura de predições")
    covered = bairros_geojson & bairros_pred
    uncovered = bairros_geojson - bairros_pred
    print(f"   Bairros cobertos: {len(covered)}/{len(bairros_geojson)}")
    if uncovered:
        print(f"   Bairros não cobertos: {uncovered}")
    else:
        print(f"   ✓ 100% de cobertura!")
    
    # 5. Estatísticas de risco
    print(f"\n5. Estatísticas de risco previsto")
    print(f"   Mínimo: {pred_df['risco_previsto'].min():.4f}")
    print(f"   Máximo: {pred_df['risco_previsto'].max():.4f}")
    print(f"   Média: {pred_df['risco_previsto'].mean():.4f}")
    print(f"   Mediana: {pred_df['risco_previsto'].median():.4f}")
    
    # 6. Top 10 bairros de maior risco
    print(f"\n6. Top 10 bairros de MAIOR risco previsto")
    top = pred_df.nlargest(10, 'risco_previsto')[['local_oficial', 'risco_previsto']]
    for idx, (_, row) in enumerate(top.iterrows(), 1):
        print(f"   {idx:2}. {row['local_oficial']:35} → {row['risco_previsto']:.4f}")
    
    # 7. Bottom 10 bairros de menor risco
    print(f"\n7. Top 10 bairros de MENOR risco previsto")
    bottom = pred_df.nsmallest(10, 'risco_previsto')[['local_oficial', 'risco_previsto']]
    for idx, (_, row) in enumerate(bottom.iterrows(), 1):
        print(f"   {idx:2}. {row['local_oficial']:35} → {row['risco_previsto']:.4f}")
    
    # 8. Validação final
    print(f"\n" + "="*80)
    is_valid = len(pred_df) >= 130  # Deve ter pelo menos 130 bairros
    if is_valid:
        print(f"✓ SUCESSO: Sistema operando em NÍVEL DE GRANULARIDADE BAIRRO")
        print(f"✓ Fortaleza tem {len(pred_df)} predições por bairro para operações táticas")
    else:
        print(f"❌ ERRO: Predições insuficientes ({len(pred_df)} < 130)")
    
    print("="*80 + "\n")
    
    return is_valid

if __name__ == "__main__":
    success = test_bairro_predictions()
    sys.exit(0 if success else 1)
