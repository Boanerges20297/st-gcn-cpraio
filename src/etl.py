import pandas as pd
import geopandas as gpd
import json
import os
import sys
from pathlib import Path
from shapely.geometry import Point

# Configuração
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_RAW = BASE_DIR / "data" / "raw"
DATA_GRAPH = BASE_DIR / "data" / "graph"
DATA_PROCESSED = BASE_DIR / "data" / "processed"

def load_geojsons():
    """Carrega e funde os mapas, criando uma hierarquia de prioridade."""
    # ORDEM DE PRIORIDADE PARA DESEMPATE:
    # 1. RMF (Para garantir que Caucaia/Maracanaú não sejam engolidos por Fortaleza)
    # 2. CAPITAL
    # 3. INTERIOR
    priority_order = ['RMF', 'CAPITAL', 'INTERIOR']
    
    paths = {
        'CAPITAL': DATA_GRAPH / "fortaleza_bairros.geojson",
        'RMF': DATA_GRAPH / "ceara_rmf.geojson",
        'INTERIOR': DATA_GRAPH / "ceara_interior.geojson"
    }
    
    parts = []
    for region in priority_order:
        p = paths.get(region)
        if p and p.exists():
            gdf = gpd.read_file(p)
            if gdf.crs != "EPSG:4326":
                gdf = gdf.to_crs("EPSG:4326")
            
            # Etiqueta a região
            gdf['regiao_sistema'] = region
            
            # Normaliza o nome do local
            if 'name' in gdf.columns:
                gdf['local_oficial'] = gdf['name'].str.upper().str.strip()
            elif 'NM_MUNICIP' in gdf.columns:
                gdf['local_oficial'] = gdf['NM_MUNICIP'].str.upper().str.strip()
            else:
                gdf['local_oficial'] = 'DESCONHECIDO'
                
            parts.append(gdf)
            
    if not parts: return None
    
    # Junta tudo num único mapa do Ceará
    full_map = pd.concat(parts, ignore_index=True)
    return full_map

def process_crimes():
    print("[-] Carregando Ocorrências...")
    json_path = DATA_RAW / "dados_status_ocorrencias_gerais.json"
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            content = json.load(f)
    except Exception as e:
        print(f"[X] Erro ao ler JSON: {e}")
        return pd.DataFrame()
        
    raw_data = []
    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict) and 'data' in item:
                raw_data = item['data']
                break
        if not raw_data and len(content) > 0 and 'latitude' in content[0]:
             raw_data = content
    elif isinstance(content, dict) and 'data' in content:
        raw_data = content['data']

    if not raw_data: return pd.DataFrame()

    df = pd.DataFrame(raw_data)
    
    cols_map = {
        'latitude': 'lat', 'longitude': 'lng',
        'data': 'data', 'hora': 'hora',
        'tipo_evento': 'natureza', 'id': 'id_ocorrencia',
        'bairro': 'bairro_ciops',
        'tipo': 'tipo'  # Campo CVLI/CVP já vem do JSON
    }
    df = df.rename(columns=cols_map)
    
    # Limpeza GPS
    df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
    df['lng'] = pd.to_numeric(df['lng'], errors='coerce')
    df = df.dropna(subset=['lat', 'lng'])
    
    # Remove coordenadas zeradas ou fora do Ceará (filtro grosseiro)
    df = df[(df['lat'] < -2) & (df['lat'] > -8) & (df['lng'] < -37) & (df['lng'] > -42)]
    
    geometry = [Point(xy) for xy in zip(df.lng, df.lat)]
    return gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

def run_pipeline():
    print("==============================================")
    print("   ETL UNIFICADO (CORREÇÃO DE FRONTEIRA)      ")
    print("==============================================")
    
    # 1. Mapa Unificado
    gdf_mapa_total = load_geojsons()
    if gdf_mapa_total is None:
        print("[X] Erro: Mapas não encontrados.")
        return

    # 2. Crimes
    gdf_crimes = process_crimes()
    if gdf_crimes.empty:
        print("[X] Erro: Sem crimes.")
        return

    print(f"[-] Roteando {len(gdf_crimes)} crimes no mapa unificado...")
    
    # 3. Spatial Join Único
    # op='within' garante que o ponto está DENTRO do polígono.
    # Se um ponto cair em 2 polígonos (sobreposição), ele duplica.
    joined = gpd.sjoin(gdf_crimes, gdf_mapa_total, how="inner", predicate="within")
    
    # 4. Deduplicação Estratégica
    # Se um crime caiu em "Fortaleza" e "Caucaia" ao mesmo tempo, removemos a duplicata.
    # Como carregamos RMF antes de CAPITAL na lista `parts`, se usarmos 'drop_duplicates'
    # mantendo o primeiro, a prioridade da lista original é preservada? 
    # Não necessariamente após o sjoin e concat.
    
    # Vamos ordenar por região para garantir a prioridade: RMF > CAPITAL > INTERIOR
    # Criamos uma coluna de peso para ordenar
    region_weight = {'RMF': 1, 'CAPITAL': 2, 'INTERIOR': 3}
    joined['peso_prioridade'] = joined['regiao_sistema'].map(region_weight)
    
    # Ordena: Menor peso (RMF) aparece primeiro
    joined = joined.sort_values('peso_prioridade')
    
    # Remove duplicatas de ID de ocorrência, mantendo a primeira (RMF ganha de Capital)
    before_dedup = len(joined)
    joined = joined.drop_duplicates(subset=['id_ocorrencia'], keep='first')
    print(f"    [i] {before_dedup - len(joined)} conflitos de fronteira resolvidos (Prioridade RMF).")

    # 5. Cruzar Facções
    csv_intel = DATA_RAW / "inteligencia_faccoes.csv"
    if csv_intel.exists():
        print("[-] Aplicando Inteligência...")
        df_intel = pd.read_csv(csv_intel)
        df_intel['local_norm'] = df_intel['local'].astype(str).str.upper().str.strip()
        
        joined = joined.merge(
            df_intel[['local_norm', 'faccao_predominante']],
            left_on='local_oficial',
            right_on='local_norm',
            how='left'
        )
        joined['faccao_predominante'] = joined['faccao_predominante'].fillna('DESCONHECIDO')
    else:
        joined['faccao_predominante'] = 'DESCONHECIDO'

    # Finalização
    joined['data_hora'] = pd.to_datetime(joined['data'] + ' ' + joined['hora'], errors='coerce')
    
    # Seleção de colunas finais
    cols = ['id_ocorrencia', 'data_hora', 'natureza', 'lat', 'lng', 
            'regiao_sistema', 'local_oficial', 'bairro_ciops', 'faccao_predominante', 'tipo']
            
    # Garante que existem
    cols = [c for c in cols if c in joined.columns]
    df_final = pd.DataFrame(joined[cols])
    
    os.makedirs(DATA_PROCESSED, exist_ok=True)
    out_file = DATA_PROCESSED / "base_consolidada.parquet"
    df_final.to_parquet(out_file, index=False)
    
    print(f"\n[V] SUCESSO. Base salva em: {out_file}")
    print(f"    Total: {len(df_final)}")
    print("    Distribuição por Região:")
    print(df_final['regiao_sistema'].value_counts())

if __name__ == "__main__":
    run_pipeline()