"""
Enriquece dados operacionais com lat/long dos bairros (IBGE) desde 2022
Busca em: Fortaleza, RMF (Região Metropolitana) e Interior de Ceará
"""
import json
import pandas as pd
import geopandas as gpd
from pathlib import Path
from shapely.geometry import shape, Point

# Caminhos
DATA_DIR = Path("data")
NORMALIZED_CSV = DATA_DIR / "raw" / "View_Ocorrencias_Operacionais_Modelo_NORMALIZADO.csv"
FORTALEZA_GEOJSON = DATA_DIR / "graph" / "fortaleza_bairros.geojson"
RMF_GEOJSON = DATA_DIR / "graph" / "ceara_rmf.geojson"
INTERIOR_GEOJSON = DATA_DIR / "graph" / "ceara_interior.geojson"
OUTPUT_CSV = DATA_DIR / "raw" / "View_Ocorrencias_2022_ENRIQUECIDO.csv"
CENTROS_OUTPUT = DATA_DIR / "raw" / "bairros_centros_latlong.json"

print("\n" + "="*80)
print("ENRIQUECIMENTO DE DADOS COM LAT/LONG (FORTALEZA + RMF + INTERIOR) 2022+")
print("="*80)

# 1. Carregar dados operacionais
print("\n[1] Carregando dados operacionais normalizados...")
df = pd.read_csv(NORMALIZED_CSV)
print(f"   ✓ {len(df):,} registros carregados")

# 2. Filtrar desde 2022
print("\n[2] Filtrando dados desde 2022...")
df['Data'] = pd.to_datetime(df['Data'])
df_2022 = df[df['Data'].dt.year >= 2022].copy()
print(f"   ✓ {len(df_2022):,} registros após filtro (período: {df_2022['Data'].min().date()} a {df_2022['Data'].max().date()})")

# 3. Carregar GeoJSON de múltiplas fontes (Fortaleza + RMF + Interior)
print("\n[3] Carregando dados geográficos (IBGE - Fortaleza, RMF e Interior)...")
geojson_files = {
    'fortaleza': (FORTALEZA_GEOJSON, 'Fortaleza'),
    'rmf': (RMF_GEOJSON, 'RMF'),
    'interior': (INTERIOR_GEOJSON, 'Interior')
}

all_features = {}
for region_key, (geojson_path, region_name) in geojson_files.items():
    try:
        with open(geojson_path, 'r', encoding='utf-8') as f:
            geojson_data = json.load(f)
        count = len(geojson_data.get('features', []))
        all_features[region_key] = geojson_data.get('features', [])
        print(f"   ✓ {region_name}: {count} features carregadas")
    except FileNotFoundError:
        print(f"   ⚠️ {region_name}: arquivo não encontrado ({geojson_path})")
        all_features[region_key] = []

total_features = sum(len(v) for v in all_features.values())
print(f"   ✓ Total: {total_features} features de bairros/regiões carregadas")

# 4. Extrair centróides e criar mapa de lat/long por região
print("\n[4] Extraindo centróides dos bairros por região...")
bairro_coords = {}
bairro_by_city = {}

for region_key, features in all_features.items():
    for feature in features:
        props = feature['properties']
        
        # Extrair nomes (diferentes estruturas em cada GeoJSON)
        bairro_name = props.get('name', props.get('NOME', props.get('nameFeature', ''))).upper().strip()
        cidade_name = props.get('city', props.get('CIDADE', props.get('municipality', ''))).upper().strip()
        
        if not bairro_name:
            continue
        
        try:
            geom = shape(feature['geometry'])
            centroid = geom.centroid
            
            coord_entry = {
                'lat': centroid.y,
                'long': centroid.x,
                'nome_ibge': bairro_name,
                'regiao': region_key,
                'cidade_ibge': cidade_name
            }
            
            # Armazenar por bairro (chave única)
            key = f"{bairro_name}"
            if key not in bairro_coords or region_key == 'fortaleza':  # Priorize Fortaleza
                bairro_coords[key] = coord_entry
            
            # Armazenar por cidade também
            if cidade_name:
                if cidade_name not in bairro_by_city:
                    bairro_by_city[cidade_name] = []
                bairro_by_city[cidade_name].append(coord_entry)
                
        except Exception as e:
            pass

print(f"   ✓ {len(bairro_coords)} bairros únicos com coordenadas extraídas")
print(f"   ✓ {len(bairro_by_city)} cidades mapeadas")

# 5. Enriquecer dataset com lat/long
print("\n[5] Enriquecendo dataset com lat/long...")
def get_coords(bairro_norm, cidade_ocor):
    """Busca coordenadas do bairro normalizado"""
    if pd.isna(bairro_norm) or not isinstance(bairro_norm, str):
        return None, None
    
    bairro_upper = str(bairro_norm).upper().strip()
    cidade_upper = str(cidade_ocor).upper().strip() if pd.notna(cidade_ocor) else ""
    
    # 1. Buscar por chave exata (bairro + cidade)
    if cidade_upper in bairro_by_city:
        for entry in bairro_by_city[cidade_upper]:
            if entry['nome_ibge'] == bairro_upper:
                return entry['lat'], entry['long']
    
    # 2. Buscar por bairro único
    if bairro_upper in bairro_coords:
        return bairro_coords[bairro_upper]['lat'], bairro_coords[bairro_upper]['long']
    
    return None, None

df_2022[['lat', 'long']] = df_2022.apply(
    lambda row: pd.Series(get_coords(row['BairroOcor'], row['CidadeOcor'])),
    axis=1
)

# Estatísticas de preenchimento
filled = df_2022[['lat', 'long']].notna().all(axis=1).sum()
total = len(df_2022)
pct = (filled / total * 100)
print(f"   ✓ {filled:,}/{total:,} registros com lat/long ({pct:.1f}%)")

# 6. Salvar dataset enriquecido
print("\n[6] Salvando dataset enriquecido...")
df_2022.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')
print(f"   ✅ {OUTPUT_CSV}")

# 7. Salvar mapa de centros para referência
print("\n[7] Salvando mapa de centros dos bairros...")
with open(CENTROS_OUTPUT, 'w', encoding='utf-8') as f:
    json.dump(bairro_coords, f, indent=2, ensure_ascii=False)
print(f"   ✅ {CENTROS_OUTPUT}")

# 8. Resumo de cobertura
print("\n[8] RESUMO DE COBERTURA GEOGRÁFICA")
print(f"   Período: {df_2022['Data'].min().date()} a {df_2022['Data'].max().date()}")
print(f"   Total de registros: {len(df_2022):,}")
print(f"   Registros com coords: {filled:,} ({pct:.1f}%)")
print(f"   Bairros únicos: {df_2022['BairroOcor'].nunique()}")
print(f"   Cidades únicas: {df_2022['CidadeOcor'].nunique()}")

# Análise CVLI
cvli_mask = df_2022['Natureza'].str.contains('Homicídio|Latrocínio', case=False, na=False)
cvli_count = cvli_mask.sum()
cvli_with_coords = df_2022[cvli_mask][['lat', 'long']].notna().all(axis=1).sum()
print(f"   Eventos CVLI: {cvli_count} ({cvli_count/len(df_2022)*100:.2f}%)")
print(f"   CVLI com coords: {cvli_with_coords}/{cvli_count}")

print("\n[9] TOP 10 CIDADES")
top_cities = df_2022['CidadeOcor'].value_counts().head(10)
for i, (city, count) in enumerate(top_cities.items(), 1):
    print(f"   {i:2}. {city}: {count:,}")

print("\n" + "="*80)
print("✅ ENRIQUECIMENTO CONCLUÍDO")
print("="*80)
