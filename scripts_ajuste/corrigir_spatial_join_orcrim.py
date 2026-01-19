#!/usr/bin/env python3
"""
Script corrigido para fazer spatial join de crimes com AIDs ORCRIM.
Versão simplificada e robusta.

Uso: python scripts_ajuste/corrigir_spatial_join_orcrim.py
"""

import pandas as pd
import geopandas as gpd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

print("=" * 70)
print("CORRIGINDO SPATIAL JOIN: Crimes + ORCRIM AIDs")
print("=" * 70)
print()

# Caminhos
ORCRIM_GEOJSON = BASE_DIR / "data" / "graph" / "ORCRIM_extraido.geojson"
CONSOLIDATED = BASE_DIR / "data" / "processed" / "base_consolidada_orcrim_v2.parquet"

# [1] Carregar GeoJSON ORCRIM
print("[1] Carregando GeoJSON ORCRIM...")
try:
    gdf_orcrim = gpd.read_file(ORCRIM_GEOJSON)
    print(f"    OK: {len(gdf_orcrim)} polígonos")
    print(f"    CRS: {gdf_orcrim.crs}")
except Exception as e:
    print(f"    ERROR: {e}")
    exit(1)

# [2] Carregar crimes
print("[2] Carregando crimes...")
try:
    df_crimes = pd.read_parquet(CONSOLIDATED)
    print(f"    OK: {len(df_crimes)} ocorrências")
except Exception as e:
    print(f"    ERROR: {e}")
    exit(1)

# [3] Converter crimes em GeoDataFrame
print("[3] Criando GeoDataFrame de crimes...")
try:
    # Garantir que lat/lng são floats válidos
    df_crimes = df_crimes[df_crimes['lat'].notna() & df_crimes['lng'].notna()].copy()
    
    geometry = gpd.points_from_xy(df_crimes['lng'], df_crimes['lat'])
    gdf_crimes = gpd.GeoDataFrame(df_crimes, geometry=geometry, crs='EPSG:4326')
    
    print(f"    OK: {len(gdf_crimes)} pontos com coordenadas válidas")
except Exception as e:
    print(f"    ERROR: {e}")
    exit(1)

# [4] Garantir que ORCRIM tem nome da AID
print("[4] Preparando ORCRIM para sjoin...")
try:
    if 'nome' not in gdf_orcrim.columns:
        gdf_orcrim = gdf_orcrim.reset_index()
        gdf_orcrim['nome'] = 'AID_' + gdf_orcrim.index.astype(str)
    
    gdf_orcrim = gdf_orcrim[['geometry', 'nome']].copy()
    print(f"    OK: {gdf_orcrim.columns.tolist()}")
except Exception as e:
    print(f"    ERROR: {e}")
    exit(1)

# [5] Fazer spatial join (IMPORTANTE: relação geométrica correta)
print("[5] Fazendo spatial join (crimes dentro de AIDs)...")
try:
    # sjoin com 'within': crime point dentro de polígono AID
    sjoin_result = gpd.sjoin(
        gdf_crimes,
        gdf_orcrim,
        how='left',
        predicate='within'
    )
    
    print(f"    OK: sjoin concluído")
    print(f"    Linhas após sjoin: {len(sjoin_result)} (pode ter duplicatas)")
    
    # Pegar apenas a primeira AID (se houver múltiplas coincidências)
    aid_map = sjoin_result.groupby(sjoin_result.index)['nome'].first()
    aid_series = gdf_crimes.index.map(aid_map).fillna('SEM_AID')
    
    print(f"    Crimes com AID: {(aid_series != 'SEM_AID').sum()} / {len(gdf_crimes)}")
    
except Exception as e:
    print(f"    ERROR no sjoin: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# [6] Atualizar coluna no DataFrame original
print("[6] Atualizando base consolidada...")
try:
    df_crimes_updated = df_crimes.copy()
    df_crimes_updated['aid_orcrim'] = aid_series.values
    
    print(f"    OK: Coluna aid_orcrim atualizada ({len(df_crimes_updated)} linhas)")
    print(f"    Distribuição:")
    print(df_crimes_updated['aid_orcrim'].value_counts().head(10))
    
except Exception as e:
    print(f"    ERROR: {e}")
    exit(1)

# [7] Salvar base corrigida
print("[7] Salvando base corrigida...")
try:
    output_path = BASE_DIR / "data" / "processed" / "base_consolidada_orcrim_v3.parquet"
    df_crimes_updated.to_parquet(output_path)
    print(f"    OK: Salvo em {output_path.name}")
except Exception as e:
    print(f"    ERROR: {e}")
    exit(1)

print()
print("=" * 70)
print("CONCLUÍDO - Spatial join corrigido!")
print("=" * 70)
print(f"Arquivo: base_consolidada_orcrim_v3.parquet")
print()
print("Próximos passos:")
print("1. Atualizar config.py para usar v3")
print("2. Reiniciar app")
print("3. Verificar dashboard com territórios mapeados")
