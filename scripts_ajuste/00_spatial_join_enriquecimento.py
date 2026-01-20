"""
SPATIAL JOIN - Preencher bairros usando lat/lng
===============================================
Carrega dados_status_ocorrencias_gerais.json e usa spatial join
para mapear coordenadas aos bairros/municípios
"""

import pandas as pd
import geopandas as gpd
import json
from pathlib import Path
from shapely.geometry import Point
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import config

def load_dados_status():
    """Carrega dados_status_ocorrencias_gerais.json"""
    json_path = config.DATA_RAW / "dados_status_ocorrencias_gerais.json"
    
    print("[-] Carregando dados_status_ocorrencias_gerais.json...")
    
    with open(json_path, 'r', encoding='utf-8-sig', errors='replace') as f:
        content = json.load(f)
    
    raw_data = []
    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict) and 'data' in item:
                raw_data = item['data']
                break
    
    if not raw_data:
        print("[X] Nenhum registro encontrado")
        return pd.DataFrame()
    
    df = pd.DataFrame(raw_data)
    print(f"[+] Carregados {len(df)} registros")
    
    # Converter coordenadas
    df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
    df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
    
    # Remover sem coordenadas
    df = df.dropna(subset=['latitude', 'longitude'])
    
    # Criar geometria
    geometry = [Point(xy) for xy in zip(df.longitude, df.latitude)]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
    
    print(f"[V] Após validação GPS: {len(gdf)} registros")
    
    return gdf

def load_all_geojsons():
    """Carrega e funde todos os geojsons"""
    print("\n[-] Carregando geojsons...")
    
    all_gdfs = []
    
    for region, path in config.GEOJSON_PATHS.items():
        if path.exists():
            print(f"  [+] Carregando {region}...")
            gdf = gpd.read_file(path)
            
            if gdf.crs != "EPSG:4326":
                gdf = gdf.to_crs("EPSG:4326")
            
            # Normalizar coluna de nome
            if 'name' in gdf.columns:
                gdf['local_name'] = gdf['name'].str.upper().str.strip()
            elif 'NM_MUNICIP' in gdf.columns:
                gdf['local_name'] = gdf['NM_MUNICIP'].str.upper().str.strip()
            elif 'NM_BAIRRO' in gdf.columns:
                gdf['local_name'] = gdf['NM_BAIRRO'].str.upper().str.strip()
            else:
                gdf['local_name'] = 'DESCONHECIDO'
            
            gdf['regiao'] = region
            all_gdfs.append(gdf)
    
    if not all_gdfs:
        print("[X] Nenhum geojson encontrado")
        return None
    
    # Concatenar
    gdf_total = pd.concat(all_gdfs, ignore_index=True)
    print(f"[V] Total de {len(gdf_total)} polígonos de {len(all_gdfs)} regiões")
    
    return gdf_total

def spatial_join_bairros(gdf_crimes, gdf_map):
    """
    Faz spatial join para mapear crimes aos bairros
    """
    print("\n[-] Fazendo spatial join...")
    
    # Spatial join: crimes dentro de polígonos
    joined = gpd.sjoin(gdf_crimes, gdf_map[['geometry', 'local_name', 'regiao']], 
                       how='left', predicate='within')
    
    print(f"[V] {len(joined)} crimes mapeados")
    
    # Contar sucessos
    mapped = joined['local_name'].notna().sum()
    unmapped = joined['local_name'].isna().sum()
    
    print(f"  [+] Mapeados: {mapped}")
    print(f"  [!] Não mapeados: {unmapped}")
    
    # Preencher com "DESCONHECIDO" onde não mapeou
    joined['local_name'] = joined['local_name'].fillna('DESCONHECIDO')
    joined['regiao'] = joined['regiao'].fillna('DESCONHECIDO')
    
    return joined

def save_enriched_data(gdf_enriched):
    """
    Salva dados enriquecidos em parquet
    """
    output_dir = config.DATA_PROCESSED
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Converter para DataFrame e remover geometria
    df_clean = pd.DataFrame(gdf_enriched)
    
    # Remover colunas duplicadas e desnecessárias
    cols_drop = ['geometry', 'index_right', 'index_left']
    cols_drop = [c for c in cols_drop if c in df_clean.columns]
    df_clean = df_clean.drop(columns=cols_drop)
    
    # Remover bairro null (duplicado de local_name)
    if 'bairro' in df_clean.columns and 'local_name' in df_clean.columns:
        df_clean = df_clean.drop(columns=['bairro'])
    
    # Renomear para compatibilidade com pipeline
    df_clean = df_clean.rename(columns={
        'local_name': 'bairro',
        'regiao': 'regiao_sistema',
        'tipo': 'tipo_crime'
    })
    
    # Garantir que tipo_crime está correto
    if 'tipo_crime' not in df_clean.columns and 'tipo' in df_clean.columns:
        df_clean = df_clean.rename(columns={'tipo': 'tipo_crime'})
    
    output_file = output_dir / "dados_status_enriquecidos_com_bairros.parquet"
    df_clean.to_parquet(output_file, index=False)
    
    print(f"\n[+] Dados enriquecidos salvos: {output_file}")
    
    # Salvar estatísticas
    print("\n[ESTATÍSTICAS]")
    print(f"  Total registros: {len(df_clean)}")
    print(f"  Bairros únicos: {df_clean['bairro'].nunique()}")
    if 'regiao_sistema' in df_clean.columns:
        print(f"  Regiões: {df_clean['regiao_sistema'].unique()}")
    
    print("\n  Top 10 Bairros por crimes:")
    top_bairros = df_clean['bairro'].value_counts().head(10)
    for bairro, count in top_bairros.items():
        pct = 100 * count / len(df_clean)
        print(f"    {bairro}: {count} ({pct:.1f}%)")
    
    if 'tipo_crime' in df_clean.columns:
        print("\n  CVLI vs CVP:")
        tipo_dist = df_clean['tipo_crime'].value_counts()
        for tipo, count in tipo_dist.items():
            pct = 100 * count / len(df_clean)
            print(f"    {tipo}: {count} ({pct:.1f}%)")
    
    return output_file

def main():
    print("=" * 70)
    print(" SPATIAL JOIN - ENRIQUECIMENTO DE DADOS COM BAIRROS")
    print("=" * 70)
    
    # 1. Carregar dados de crimes
    gdf_crimes = load_dados_status()
    if gdf_crimes.empty:
        return
    
    # 2. Carregar mapas
    gdf_map = load_all_geojsons()
    if gdf_map is None:
        return
    
    # 3. Spatial join
    gdf_joined = spatial_join_bairros(gdf_crimes, gdf_map)
    
    # 4. Salvar
    output_file = save_enriched_data(gdf_joined)
    
    print("\n" + "=" * 70)
    print(" SPATIAL JOIN CONCLUÍDO COM SUCESSO")
    print("=" * 70)
    
    return output_file

if __name__ == "__main__":
    main()
