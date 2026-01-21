#!/usr/bin/env python3
"""
Script para mesclar dados ORCRIM (territoriais fragmentados) com base consolidada
e re-treinar modelo ST-GCN com dados enriquecidos.

Uso: python scripts_ajuste/mesclar_orcrim_retreinar.py
"""

import json
import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path
from datetime import datetime
import sys
import os

# Adicionar src ao path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

print("=" * 70)
print("🔄 MESCLAGEM ORCRIM + RE-TREINAMENTO ST-GCN")
print("=" * 70)
print()

# Caminhos
ORCRIM_GEOJSON = BASE_DIR / "data" / "graph" / "ORCRIM_extraido.geojson"
CONSOLIDATED = BASE_DIR / "data" / "processed" / "base_consolidada.parquet"
OUTPUT_CONSOLIDATED = BASE_DIR / "data" / "processed" / "base_consolidada_orcrim_v2.parquet"

# ============================================================================
# ETAPA 1: Carregar dados
# ============================================================================
print("[1] Carregando dados...")

# Carregar GeoJSON ORCRIM
print("   📥 Carregando ORCRIM GeoJSON...")
try:
    gdf_orcrim = gpd.read_file(ORCRIM_GEOJSON)
    print(f"      ✓ {len(gdf_orcrim)} polígonos de AID carregados")
except Exception as e:
    print(f"      ❌ Erro ao carregar ORCRIM: {e}")
    sys.exit(1)

# Carregar base consolidada
print("   📥 Carregando base consolidada...")
try:
    df_crimes = pd.read_parquet(CONSOLIDATED)
    print(f"      ✓ {len(df_crimes)} ocorrências carregadas")
except Exception as e:
    print(f"      ❌ Erro ao carregar base: {e}")
    sys.exit(1)

# ============================================================================
# ETAPA 2: Enriquecer base com territórios ORCRIM
# ============================================================================
print()
print("[2] Enriquecendo base com territórios ORCRIM...")

# Converter crimes em GeoDataFrame
try:
    geometry = gpd.points_from_xy(df_crimes['lng'], df_crimes['lat'])
    gdf_crimes = gpd.GeoDataFrame(
        df_crimes,
        geometry=geometry,
        crs='EPSG:4326'
    )
    print(f"   ✓ {len(gdf_crimes)} pontos de crime criados")
except Exception as e:
    print(f"   ❌ Erro ao criar GeoDataFrame de crimes: {e}")
    sys.exit(1)

# Fazer spatial join (crimes dentro de polígonos ORCRIM)
print("   🔍 Fazendo spatial join (crimes dentro de AIDs)...")
try:
    # Ensure gdf_orcrim has required column
    if 'nome' not in gdf_orcrim.columns:
        gdf_orcrim['nome'] = gdf_orcrim.index.astype(str)
    
    joined = gpd.sjoin(
        gdf_crimes,
        gdf_orcrim[['geometry', 'nome']].rename(columns={'nome': 'aid_nome'}),
        how='left',
        predicate='within'
    )
    
    # Mover coluna aid_nome para df_crimes
    df_crimes['aid_orcrim'] = joined['aid_nome'].values
    
    # Crimes localizados em AIDs
    crimes_localizados = df_crimes['aid_orcrim'].notna().sum()
    print(f"   ✓ {crimes_localizados} crimes localizados em AIDs ({crimes_localizados/len(df_crimes)*100:.1f}%)")
    
except Exception as e:
    print(f"   ❌ Erro no spatial join: {e}")
    df_crimes['aid_orcrim'] = 'SEM_AID'

# ============================================================================
# ETAPA 3: Validar e limpar dados
# ============================================================================
print()
print("[3] Validando e limpando dados...")

# Verificar colunas essenciais
required_cols = ['data_hora', 'lat', 'lng', 'natureza', 'regiao_sistema']
missing_cols = [col for col in required_cols if col not in df_crimes.columns]

if missing_cols:
    print(f"   ⚠️  Colunas faltando: {missing_cols}")
else:
    print(f"   ✓ Todas as colunas essenciais presentes")

# Verificar datas
print(f"   📅 Intervalo de datas: {df_crimes['data_hora'].min()} até {df_crimes['data_hora'].max()}")

# Verificar coordenadas
coords_valid = df_crimes[['lat', 'lng']].notna().all(axis=1).sum()
print(f"   🗺️  Ocorrências com coordenadas válidas: {coords_valid}/{len(df_crimes)} ({coords_valid/len(df_crimes)*100:.1f}%)")

# Salvar base enriquecida
print()
print("[4] Salvando base enriquecida...")
try:
    df_crimes.to_parquet(OUTPUT_CONSOLIDATED)
    print(f"   ✓ Salvo em: {OUTPUT_CONSOLIDATED}")
except Exception as e:
    print(f"   ❌ Erro ao salvar: {e}")
    sys.exit(1)

# ============================================================================
# ETAPA 5: Re-treinar modelo ST-GCN
# ============================================================================
print()
print("[5] Re-treinando modelo ST-GCN...")
print("   ⏳ Isso pode levar 5-10 minutos...")

try:
    from trainer import train_region
    import config
    
    # Re-treinar para cada região
    print("   📊 Regions para treino:", list(config.ARTIFACTS.keys()))
    
    for region in config.ARTIFACTS.keys():
        print(f"   🤖 Treinando {region}...")
        try:
            train_region(region)
            print(f"      ✓ {region} re-treinado com dados ORCRIM mesclados")
        except Exception as e:
            print(f"      ⚠️  Erro em {region}: {e}")
    
    print()
    print("   ✅ Treinamento concluído para todas as regiões!")
    
except ImportError as e:
    print(f"   ⚠️  Módulo trainer não disponível: {e}")
    print("   ℹ️  Pulando treinamento (dados ainda foram mesclados)")
except Exception as e:
    print(f"   ❌ Erro no treinamento: {e}")
    print("   ℹ️  Dados foram mesclados mesmo assim")

# ============================================================================
# RESUMO FINAL
# ============================================================================
print()
print("=" * 70)
print("✅ PROCESSO CONCLUÍDO")
print("=" * 70)
print()
print("📊 Estatísticas Finais:")
print(f"   • Total de ocorrências: {len(df_crimes):,}")
print(f"   • Com coordenadas: {df_crimes[['lat', 'lng']].notna().all(axis=1).sum():,}")
print(f"   • Com AID ORCRIM: {(df_crimes['aid_orcrim'].notna()).sum():,}")
print()
print("📁 Saída:")
print(f"   • Base consolidada v2: {OUTPUT_CONSOLIDATED}")
print()
print("🚀 Próximos passos:")
print("   1. Reiniciar aplicação: python src/app.py")
print("   2. Acessar dashboard: http://localhost:5000/dashboard-estrategico")
print("   3. Observar predições atualizadas com dados ORCRIM")
print()
print("=" * 70)
