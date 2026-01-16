import requests
import geopandas as gpd
import pandas as pd
import os
from io import BytesIO

# Configurações de Caminho
PATHS = {
    'rmf': "data/graph/ceara_rmf.geojson",
    'interior': "data/graph/ceara_interior.geojson"
}

# Lista Oficial da RMF (Com acentos, pois o IBGE usa acentos)
RMF_LIST = [
    "Aquiraz", "Cascavel", "Caucaia", "Chorozinho", "Eusébio", "Fortaleza", 
    "Guaiúba", "Horizonte", "Itaitinga", "Maracanaú", "Maranguape", 
    "Pacajus", "Pacatuba", "Paracuru", "Paraipaba", "Pindoretama", 
    "São Gonçalo do Amarante", "São Luís do Curu", "Trairi"
]

def fetch_ibge_data():
    print("[-] Conectando à API de Malhas do IBGE (Ceará)...")
    
    # URL TÁTICA CORRIGIDA:
    # 1. /estados/23 -> Alvo: Ceará
    # 2. intrarregiao=municipio -> COMANDO: "Quebre em municípios" (O Pulo do Gato)
    url = "https://servicodados.ibge.gov.br/api/v3/malhas/estados/23?qualidade=minima&formato=application/vnd.geo+json&intrarregiao=municipio"
    
    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        
        print("[-] Malha baixada. Processando geometria...")
        # Carregar GeoJSON direto da memória
        gdf = gpd.read_file(BytesIO(response.content))
        
        count = len(gdf)
        print(f"    Total de polígonos recebidos: {count}")
        
        if count < 180:
            print("[X] ERRO: A API retornou poucos polígonos. Verifique se 'intrarregiao=municipio' funcionou.")
            return

        # --- CRUZAMENTO DE NOMES (De-Para) ---
        print("[-] Baixando dicionário de nomes (API Localidades)...")
        names_url = "https://servicodados.ibge.gov.br/api/v1/localidades/estados/23/municipios"
        names_resp = requests.get(names_url)
        names_data = names_resp.json()
        
        # Mapa: 6 primeiros dígitos do ID -> Nome
        # Ex: IBGE Malha usa "230440" (6 dígitos) | IBGE Localidades usa "2304400" (7 dígitos)
        id_to_name = {str(item['id'])[:6]: item['nome'] for item in names_data}
        
        # Identificar coluna de ID no GeoJSON (Geralmente é 'codarea')
        col_id = 'codarea' if 'codarea' in gdf.columns else gdf.columns[0]
        
        # Normalizar ID do GeoJSON para 6 dígitos para garantir o match
        gdf['id_match'] = gdf[col_id].astype(str).str.slice(0, 6)
        
        # Aplicar o Mapeamento
        gdf['name'] = gdf['id_match'].map(id_to_name)
        
        # Validação de Segurança
        found = gdf['name'].notna().sum()
        print(f"    Cruzamento concluído: {found} municípios identificados de {count}.")
        
        if found == 0:
            print("[X] FALHA NO MATCH DE NOMES. Abortando para não gerar lixo.")
            return

        # Limpeza e Padronização
        gdf = gdf.dropna(subset=['name'])
        gdf['name_upper'] = gdf['name'].str.upper().str.strip()
        
        # --- SEGMENTAÇÃO ESTRATÉGICA (RMF vs Interior) ---
        print("[-] Segmentando RMF vs Interior...")
        rmf_upper = [x.upper() for x in RMF_LIST]
        
        gdf_rmf = gdf[gdf['name_upper'].isin(rmf_upper)]
        gdf_interior = gdf[~gdf['name_upper'].isin(rmf_upper)]
        
        # Converter CRS para Lat/Long (Padrão Web WGS84)
        # Isso garante que o Folium/Streamlit plote no lugar certo
        if gdf_rmf.crs != "EPSG:4326": gdf_rmf = gdf_rmf.to_crs(epsg=4326)
        if gdf_interior.crs != "EPSG:4326": gdf_interior = gdf_interior.to_crs(epsg=4326)
            
        # Salvar Arquivos Finais
        os.makedirs(os.path.dirname(PATHS['rmf']), exist_ok=True)
        gdf_rmf.to_file(PATHS['rmf'], driver='GeoJSON')
        gdf_interior.to_file(PATHS['interior'], driver='GeoJSON')
        
        print(f"\n[V] MAPAS TÁTICOS PRONTOS:")
        print(f"    - RMF (Metropolitana): {len(gdf_rmf)} municípios -> {PATHS['rmf']}")
        print(f"    - Interior (Sertão/Litoral): {len(gdf_interior)} municípios -> {PATHS['interior']}")

    except Exception as e:
        print(f"[X] Erro Crítico: {e}")

if __name__ == "__main__":
    fetch_ibge_data()