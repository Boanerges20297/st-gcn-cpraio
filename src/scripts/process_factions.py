import geopandas as gpd
import pandas as pd
import os
import sys
from pathlib import Path
import warnings

# Ignorar avisos de geometria para limpar o output
warnings.filterwarnings('ignore')

# Adicionar raiz ao path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import config

# Mapeamento de Arquivos -> Códigos de Facção
FACTION_FILES = {
    "COMANDO VERMELHO.geojson": "CV",
    "OKAIDA.geojson": "GDE", # Guardiões do Estado
    "PRIMEIRO COMANDO DA CAPITAL.geojson": "PCC",
    "TERCEIRO COMANDO PURO.geojson": "TCP",
    "COMUNIDADES EM DISPUTA.geojson": "DISPUTA",
    "MASSA.geojson": "MASSA", # Neutro/Civil
    "TERRITÓRIOS FANTASMAS.geojson": "DESCONHECIDO"
}

INPUT_DIR = config.DATA_RAW / "inteligencia"
OUTPUT_CSV = config.DATA_RAW / "inteligencia_faccoes.csv"

def load_all_factions():
    print("[-] Carregando camadas de inteligência criminal...")
    gdf_list = []
    
    for filename, code in FACTION_FILES.items():
        path = INPUT_DIR / filename
        if path.exists():
            try:
                gdf = gpd.read_file(path)
                # Manter apenas a geometria e atribuir a facção
                # Forçar CRS para 4326 se não tiver definido, para evitar erro no to_crs depois
                if gdf.crs is None:
                    gdf.set_crs(epsg=4326, inplace=True)
                
                # Normalizar colunas
                gdf = gdf[['geometry']].copy()
                gdf['faccao'] = code
                gdf_list.append(gdf)
                print(f"    [+] Camada carregada: {code} ({len(gdf)} polígonos)")
            except Exception as e:
                print(f"    [X] Erro ao ler {filename}: {e}")
        else:
            print(f"    [!] Arquivo não encontrado: {filename}")
            
    if not gdf_list:
        return None
        
    # Juntar tudo num único mapa de "Risco"
    print("[-] Unificando mapa de calor das facções...")
    
    # CORREÇÃO AQUI: Usar pd.concat em vez de gpd.concat
    full_map = pd.concat(gdf_list, ignore_index=True)
    
    # Garantir que continua sendo um GeoDataFrame após o concat
    if not isinstance(full_map, gpd.GeoDataFrame):
        full_map = gpd.GeoDataFrame(full_map, geometry='geometry')
    
    # Validar CRS (Converter para Lat/Long padrão se precisar)
    # Se perdermos o CRS no concat, recuperamos do primeiro item da lista
    if full_map.crs is None and gdf_list:
        full_map.set_crs(gdf_list[0].crs, inplace=True)
        
    if full_map.crs != "EPSG:4326":
        full_map = full_map.to_crs("EPSG:4326")
        
    return full_map

def calculate_dominance(admin_map, faction_map, region_name):
    print(f"    > Cruzando territórios na região: {region_name}...")
    
    # Preparar mapa administrativo
    # Projetar para Metros (UTM 24S) para cálculo de área preciso
    try:
        admin_map = admin_map.to_crs("EPSG:31984") 
        faction_map_proj = faction_map.to_crs("EPSG:31984")
    except Exception as e:
        print(f"    [!] Aviso de CRS: {e}. Tentando fallback para EPSG:3857 (Web Mercator)")
        admin_map = admin_map.to_crs("EPSG:3857") 
        faction_map_proj = faction_map.to_crs("EPSG:3857")
    
    # Calcular área total de cada bairro/cidade
    admin_map['area_total'] = admin_map.geometry.area
    
    # --- INTERSECÇÃO ESPACIAL (OVERLAY) ---
    # Isso recorta os pedaços das facções que caem dentro de cada bairro
    # keep_geom_type=False ajuda a evitar erros com geometrias mistas
    intersection = gpd.overlay(admin_map, faction_map_proj, how='intersection', keep_geom_type=False)
    
    # Calcular área de cada pedaço
    intersection['area_faccao'] = intersection.geometry.area
    
    results = []
    
    # Para cada bairro/cidade no mapa administrativo
    # Usamos tqdm se disponível, senão iterrows normal
    for idx, row in admin_map.iterrows():
        local_name = row['name']
        
        # Pegar pedaços que caem neste local
        fragments = intersection[intersection['name'] == local_name]
        
        if fragments.empty:
            dom_faction = "NEUTRO"
            coverage = 0.0
        else:
            # Somar área por facção dentro deste bairro
            stats = fragments.groupby('faccao')['area_faccao'].sum()
            
            if stats.empty:
                dom_faction = "NEUTRO"
                coverage = 0.0
            else:
                # Quem tem a maior área?
                dom_faction = stats.idxmax()
                dom_area = stats.max()
                
                # Porcentagem de domínio
                coverage = dom_area / row['area_total'] if row['area_total'] > 0 else 0
                
                # Regra de Disputa: Se 'DISPUTA' tiver área relevante
                if 'DISPUTA' in stats and stats['DISPUTA'] > (dom_area * 0.5):
                    dom_faction = "DISPUTA"
        
        results.append({
            'local': local_name,
            'regiao_sistema': region_name,
            'faccao_predominante': dom_faction,
            'grau_dominio': round(coverage, 2)
        })
        
    return results

def main():
    # 1. Carregar Facções
    gdf_factions = load_all_factions()
    if gdf_factions is None:
        print("[X] Abortando: Nenhuma camada de facção carregada.")
        return

    all_intelligence = []

    # 2. Processar cada Nível Administrativo
    # Capital (Bairros)
    if config.GEOJSON_PATHS['CAPITAL'].exists():
        gdf_cap = gpd.read_file(config.GEOJSON_PATHS['CAPITAL'])
        all_intelligence.extend(calculate_dominance(gdf_cap, gdf_factions, 'CAPITAL'))
        
    # RMF (Municípios)
    if config.GEOJSON_PATHS['RMF'].exists():
        gdf_rmf = gpd.read_file(config.GEOJSON_PATHS['RMF'])
        all_intelligence.extend(calculate_dominance(gdf_rmf, gdf_factions, 'RMF'))
        
    # Interior (Municípios)
    if config.GEOJSON_PATHS['INTERIOR'].exists():
        gdf_int = gpd.read_file(config.GEOJSON_PATHS['INTERIOR'])
        all_intelligence.extend(calculate_dominance(gdf_int, gdf_factions, 'INTERIOR'))

    # 3. Salvar Resultado
    if not all_intelligence:
        print("[!] Nenhum dado de inteligência gerado. Verifique os mapas administrativos.")
        return

    df_final = pd.DataFrame(all_intelligence)
    
    # Tratamento de Strings
    df_final['local'] = df_final['local'].astype(str).str.upper().str.strip()
    
    print("\n[RESULTADO DA ANÁLISE DE INTELIGÊNCIA]")
    print(df_final['faccao_predominante'].value_counts())
    
    df_final.to_csv(OUTPUT_CSV, index=False)
    print(f"\n[V] Matriz de Dominio Territorial salva em: {OUTPUT_CSV}")
    print("    O graph_builder.py usará este arquivo na próxima execução.")

if __name__ == "__main__":
    main()