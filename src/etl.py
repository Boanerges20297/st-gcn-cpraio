import pandas as pd
import geopandas as gpd
import json
import os
import sys
from pathlib import Path
from shapely.geometry import Point

# Configuração de caminhos
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_RAW = BASE_DIR / "data" / "raw"
DATA_GRAPH = BASE_DIR / "data" / "graph"
DATA_PROCESSED = BASE_DIR / "data" / "processed"

def load_geojsons():
    """Carrega os mapas administrativos."""
    paths = {
        'CAPITAL': DATA_GRAPH / "fortaleza_bairros.geojson",
        'RMF': DATA_GRAPH / "ceara_rmf.geojson",
        'INTERIOR': DATA_GRAPH / "ceara_interior.geojson"
    }
    gdfs = {}
    for region, p in paths.items():
        if p.exists():
            gdfs[region] = gpd.read_file(p)
            # Garantir CRS WGS84
            if gdfs[region].crs != "EPSG:4326":
                gdfs[region] = gdfs[region].to_crs("EPSG:4326")
    return gdfs

def process_crimes():
    print("[-] Carregando Ocorrências (JSON)...")
    json_path = DATA_RAW / "dados_status_ocorrencias_gerais.json"
    
    with open(json_path, 'r', encoding='utf-8') as f:
        content = json.load(f)
        
    # Extração robusta (lida com lista ou dict wrapper)
    raw_data = []
    if isinstance(content, list):
        # Tenta achar o dict que tem a chave 'data'
        for item in content:
            if isinstance(item, dict) and 'data' in item:
                raw_data = item['data']
                break
        if not raw_data and len(content) > 0 and 'latitude' in content[0]:
             raw_data = content # É uma lista direta
    elif isinstance(content, dict) and 'data' in content:
        raw_data = content['data']

    if not raw_data:
        print("[X] ERRO: JSON vazio ou formato desconhecido.")
        return pd.DataFrame()

    df = pd.DataFrame(raw_data)
    
    # Padronização
    cols_map = {
        'latitude': 'lat', 'longitude': 'lng',
        'data': 'data', 'hora': 'hora',
        'tipo_evento': 'natureza', 'id': 'id_ocorrencia'
    }
    # Mantém colunas originais se não estiverem no map
    df = df.rename(columns=cols_map)
    
    # Limpeza de Coordenadas
    df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
    df['lng'] = pd.to_numeric(df['lng'], errors='coerce')
    df = df.dropna(subset=['lat', 'lng'])
    
    # Criar Geometria
    geometry = [Point(xy) for xy in zip(df.lng, df.lat)]
    gdf_crimes = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
    
    return gdf_crimes

def run_pipeline():
    print("==============================================")
    print("   ETL GEOESPACIAL: OCORRÊNCIAS + FACÇÕES     ")
    print("==============================================")
    
    # 1. Carregar Mapas
    gdfs_mapas = load_geojsons()
    if not gdfs_mapas:
        print("[X] Nenhum mapa geojson encontrado em data/graph/")
        return

    # 2. Carregar Crimes
    gdf_crimes = process_crimes()
    if gdf_crimes.empty:
        return

    print(f"[-] Roteando {len(gdf_crimes)} crimes via GPS...")
    
    dfs_final = []
    
    # 3. Spatial Join Hierárquico (Capital -> RMF -> Interior)
    # Crimes que caem em Fortaleza ficam na Capital. O resto sobra para RMF, etc.
    
    remaining = gdf_crimes.copy()
    
    # Ordem de prioridade
    for region in ['CAPITAL', 'RMF', 'INTERIOR']:
        if region not in gdfs_mapas: continue
        
        mapa = gdfs_mapas[region]
        
        # JOIN ESPACIAL: Ponto dentro de Polígono
        # op='within' ou predicate='within' dependendo da versão do geopandas
        try:
            joined = gpd.sjoin(remaining, mapa, how="inner", predicate="within")
        except:
            joined = gpd.sjoin(remaining, mapa, how="inner", op="within")
            
        if not joined.empty:
            print(f"    [+] {region}: {len(joined)} ocorrências localizadas.")
            
            # Adiciona metadados da região
            joined['regiao_sistema'] = region
            joined['local_oficial'] = joined['name'].str.upper().str.strip() # Nome oficial do polígono (Bairro ou Cidade)
            
            # Remove duplicatas de colunas criadas pelo sjoin
            cols_to_keep = ['id_ocorrencia', 'data', 'hora', 'natureza', 'lat', 'lng', 'regiao_sistema', 'local_oficial']
            dfs_final.append(joined[cols_to_keep])
            
            # Remove os processados da fila
            remaining = remaining[~remaining.index.isin(joined.index)]
    
    if not dfs_final:
        print("[X] ERRO: Nenhum crime caiu dentro dos mapas fornecidos. Verifique Lat/Lng e CRS.")
        return

    df_consolidado = pd.concat(dfs_final)
    
    # 4. Cruzar com Inteligência de Facções (CSV)
    # Agora que temos o 'local_oficial' garantido pelo mapa, cruzamos com o CSV de facções
    csv_intel = DATA_RAW / "inteligencia_faccoes.csv"
    if csv_intel.exists():
        print("[-] Aplicando Inteligência de Facções...")
        df_intel = pd.read_csv(csv_intel)
        df_intel['local_norm'] = df_intel['local'].astype(str).str.upper().str.strip()
        
        # Merge
        df_consolidado = df_consolidado.merge(
            df_intel[['local_norm', 'faccao_predominante']],
            left_on='local_oficial',
            right_on='local_norm',
            how='left'
        )
        df_consolidado['faccao_predominante'] = df_consolidado['faccao_predominante'].fillna('DESCONHECIDO')
    else:
        print("[!] CSV de facções não encontrado. Definindo como DESCONHECIDO.")
        df_consolidado['faccao_predominante'] = 'DESCONHECIDO'

    # Tratamento de Data
    df_consolidado['data_hora'] = pd.to_datetime(df_consolidado['data'] + ' ' + df_consolidado['hora'], errors='coerce')
    
    # Salvar
    os.makedirs(DATA_PROCESSED, exist_ok=True)
    out_file = DATA_PROCESSED / "base_consolidada.parquet"
    df_consolidado.to_parquet(out_file, index=False)
    
    print(f"\n[V] SUCESSO: Base Consolidada salva em {out_file}")
    print(f"    Total de Crimes Processados: {len(df_consolidado)}")
    print("    Amostra de Facções:")
    print(df_consolidado['faccao_predominante'].value_counts().head())

if __name__ == "__main__":
    run_pipeline()