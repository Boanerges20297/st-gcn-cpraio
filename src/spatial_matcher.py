import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import os
import config

# Caminho do Arquivo que você já tem
INPUT_PARQUET = config.DATA_PROCESSED / "crime_data_geocoded.parquet"

def load_maps():
    print("[-] Carregando camadas táticas (Mapas)...")
    try:
        # Nível 1: Micro (Capital)
        gdf_bairros = gpd.read_file(config.GEOJSON_PATH)
        
        # Nível 2: Meso (RMF)
        gdf_rmf = gpd.read_file("data/graph/ceara_rmf.geojson")
        
        # Nível 3: Macro (Interior)
        gdf_interior = gpd.read_file("data/graph/ceara_interior.geojson")
        
        return gdf_bairros, gdf_rmf, gdf_interior
    except Exception as e:
        print(f"[X] Erro ao carregar mapas: {e}")
        print("    Verifique se rodou o 'fetch_ceara_ibge.py'.")
        exit(1)

def main():
    print("==============================================")
    print("   ROTEADOR GEOESPACIAL (VIA PARQUET PRONTO)  ")
    print("==============================================")
    
    if not INPUT_PARQUET.exists():
        print(f"[X] Arquivo não encontrado: {INPUT_PARQUET}")
        print("    Por favor, mova o arquivo 'crime_data_geocoded.parquet' para a pasta data/processed/")
        return

    # 1. Carregar Ouro (Seu arquivo)
    print(f"[-] Lendo arquivo base: {INPUT_PARQUET.name}...")
    df = pd.read_parquet(INPUT_PARQUET)
    
    # Validação rápida de colunas
    cols = df.columns.str.lower()
    if 'lat' not in cols and 'latitude' not in cols:
        print("[!] AVISO: Colunas de coordenada não padronizadas. Tentando normalizar...")
        # Normalização de emergência caso o parquet venha com nomes diferentes
        rename_map = {c: 'lat' for c in df.columns if c.lower() in ['latitude', 'lat_geo']}
        rename_map.update({c: 'long' for c in df.columns if c.lower() in ['longitude', 'long_geo', 'lng']})
        df = df.rename(columns=rename_map)

    print(f"    Total de registros: {len(df)}")

    # 2. Carregar Mapas
    gdf_bairros, gdf_rmf, gdf_interior = load_maps()
    
    # 3. Converter para GeoDataFrame
    print("[-] Criando geometria espacial...")
    df['geometry'] = df.apply(lambda x: Point(float(x['long']), float(x['lat'])), axis=1)
    gdf_crimes = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")
    
    # --- FILTRO 1: CAPITAL (Bairros) ---
    print("[-] Roteando CAPITAL (Intersecção com Bairros)...")
    capital_join = gpd.sjoin(gdf_crimes, gdf_bairros, how="inner", predicate='within')
    
    path_cap = config.DATA_PROCESSED / "crimes_capital_mapped.parquet"
    capital_join.to_parquet(path_cap)
    print(f"    [V] {len(capital_join)} ocorrências alocadas em Fortaleza.")
    
    # Remove o que já foi classificado
    remaining = gdf_crimes[~gdf_crimes.index.isin(capital_join.index)]
    
    # --- FILTRO 2: RMF (Municípios) ---
    print("[-] Roteando RMF (Intersecção com Municípios Metropolitanos)...")
    if not remaining.empty:
        rmf_join = gpd.sjoin(remaining, gdf_rmf, how="inner", predicate='within')
        
        path_rmf = config.DATA_PROCESSED / "crimes_rmf_mapped.parquet"
        rmf_join.to_parquet(path_rmf)
        print(f"    [V] {len(rmf_join)} ocorrências alocadas na RMF.")
        
        remaining = remaining[~remaining.index.isin(rmf_join.index)]
    else:
        print("    [!] Sem sobras para RMF.")

    # --- FILTRO 3: INTERIOR (Municípios) ---
    print("[-] Roteando INTERIOR (Restante do Estado)...")
    if not remaining.empty:
        interior_join = gpd.sjoin(remaining, gdf_interior, how="inner", predicate='within')
        
        path_int = config.DATA_PROCESSED / "crimes_interior_mapped.parquet"
        interior_join.to_parquet(path_int)
        print(f"    [V] {len(interior_join)} ocorrências alocadas no Interior.")
    else:
        print("    [!] Sem sobras para Interior.")
        
    print("\n[V] DISTRIBUIÇÃO CONCLUÍDA.")

if __name__ == "__main__":
    main()