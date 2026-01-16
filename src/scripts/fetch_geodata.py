import os
import osmnx as ox
import geopandas as gpd
from shapely.geometry import Polygon

# Configurações de diretório
RAW_DATA_DIR = "data/raw"
GRAPH_DATA_DIR = "data/graph"
FILENAME = "fortaleza_bairros.geojson"

def setup_dirs():
    """Garante que as pastas existem."""
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    os.makedirs(GRAPH_DATA_DIR, exist_ok=True)

def fetch_bairros_fortaleza():
    """
    Busca os limites dos bairros de Fortaleza usando OpenStreetMap.
    O OSM usa admin_level=10 para bairros no Brasil.
    """
    print("[-] Iniciando download dos polígonos de bairros de Fortaleza via OSM...")
    print("    Isso pode levar alguns segundos...")

    try:
        # Busca por "place" (Cidade) e filtra por tags administrativas
        # Fortaleza, Ceará
        tags = {'admin_level': '10', 'boundary': 'administrative'}
        gdf = ox.features_from_place("Fortaleza, Ceara, Brazil", tags=tags)

        # Filtrar colunas relevantes e garantir que são Polígonos
        # O OSM pode trazer pontos ou linhas, queremos apenas as áreas (Polígonos)
        gdf = gdf[gdf.geometry.type.isin(['Polygon', 'MultiPolygon'])]
        
        # Selecionar apenas colunas essenciais
        cols_to_keep = ['name', 'geometry']
        # Se houver 'official_name' ou 'alt_name', pode ser útil, mas 'name' é o padrão
        if 'official_name' in gdf.columns:
            cols_to_keep.append('official_name')
            
        gdf_clean = gdf[cols_to_keep].copy()
        
        # Padronizar nomes (Maiúsculas e sem acentos se desejar depois, por enquanto manter original)
        gdf_clean['name'] = gdf_clean['name'].astype(str).str.upper()
        
        # Resetar index para garantir formato limpo
        gdf_clean = gdf_clean.reset_index(drop=True)
        
        # Verificar quantos bairros foram baixados
        qtd = len(gdf_clean)
        print(f"[V] Sucesso! {qtd} bairros encontrados.")
        
        # Validação simples: Fortaleza tem aprox 121 bairros oficiais.
        if qtd < 100:
            print("[!] ATENÇÃO: Número de bairros abaixo do esperado. Verifique a fonte.")
        
        return gdf_clean

    except Exception as e:
        print(f"[X] Erro ao buscar dados do OSM: {e}")
        return None

def save_data(gdf):
    if gdf is None:
        return
    
    # Salvar em GeoJSON (formato leve e padronizado para web/python)
    output_path = os.path.join(GRAPH_DATA_DIR, FILENAME)
    
    # Converter para CRS projetado (EPSG:31984 - SIRGAS 2000 / UTM zone 24S) para cálculos de área precisos futuramente
    # Mas para salvar o GeoJSON, o padrão web é EPSG:4326 (Lat/Long)
    gdf.to_crs(epsg=4326).to_file(output_path, driver='GeoJSON')
    
    print(f"[V] Arquivo salvo em: {output_path}")
    print("    Próximo passo: Usar este arquivo para fazer o Spatial Join com as ocorrências.")

if __name__ == "__main__":
    setup_dirs()
    gdf_bairros = fetch_bairros_fortaleza()
    save_data(gdf_bairros)