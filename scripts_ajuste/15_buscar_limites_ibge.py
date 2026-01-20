#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para buscar limites do Ceará diretamente do IBGE
e salvar em um GeoJSON otimizado para renderização no Leaflet
"""

import requests
import geopandas as gpd
import json
from pathlib import Path
from io import BytesIO

print("="*80)
print("BUSCANDO LIMITES DO CEARÁ DO IBGE")
print("="*80)

# URL da API do IBGE (Estado 23 = Ceará)
url = "https://servicodados.ibge.gov.br/api/v3/malhas/estados/23?qualidade=minima&formato=application/vnd.geo+json"

print(f"\n[1] Conectando ao IBGE...")
print(f"    URL: {url}")

try:
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    print(f"    ✓ Status: {response.status_code}")
    
    # Salvar raw GeoJSON
    data = response.json()
    print(f"    ✓ GeoJSON recebido: {len(data.get('features', []))} features")
    
    # Converter para GeoDataFrame para processamento
    gdf = gpd.read_file(BytesIO(response.content))
    print(f"    ✓ GeoDataFrame criado: {len(gdf)} registros")
    
    # Verificar CRS
    print(f"    CRS: {gdf.crs}")
    
    # Converter para WGS84 se necessário
    if gdf.crs != "EPSG:4326":
        print(f"    Convertendo para EPSG:4326...")
        gdf = gdf.to_crs("EPSG:4326")
    
    # Simplificar geometrias para arquivo menor
    print(f"\n[2] Simplificando geometrias...")
    gdf['geometry'] = gdf['geometry'].simplify(tolerance=0.01)  # ~1km de tolerance
    
    # Salvar em 3 formatos diferentes para testar
    
    # FORMATO 1: GeoJSON completo
    output1 = Path('data/raw/limites_ceara_ibge_completo.geojson')
    gdf.to_file(output1, driver='GeoJSON')
    size1 = output1.stat().st_size / 1024
    print(f"\n[3] Formato 1 - Completo:")
    print(f"    Arquivo: {output1.name}")
    print(f"    Tamanho: {size1:.1f} KB")
    
    # FORMATO 2: GeoJSON simplificado (apenas limites)
    output2 = Path('data/raw/limites_ceara_ibge_simples.geojson')
    gdf_simples = gdf[['geometry']].copy()
    gdf_simples['name'] = 'Ceará'
    gdf_simples.to_file(output2, driver='GeoJSON')
    size2 = output2.stat().st_size / 1024
    print(f"\n[4] Formato 2 - Simples (apenas geometria):")
    print(f"    Arquivo: {output2.name}")
    print(f"    Tamanho: {size2:.1f} KB")
    
    # FORMATO 3: LineString (apenas contornos, sem preenchimento)
    print(f"\n[5] Convertendo para LineString...")
    lines = []
    for idx, row in gdf.iterrows():
        geom = row['geometry']
        if geom.geom_type == 'Polygon':
            exterior = geom.exterior
            lines.append({
                'type': 'Feature',
                'geometry': {
                    'type': 'LineString',
                    'coordinates': list(exterior.coords)
                },
                'properties': {'name': 'Ceará', 'tipo': 'limite_externo'}
            })
        elif geom.geom_type == 'MultiPolygon':
            for poly in geom.geoms:
                exterior = poly.exterior
                lines.append({
                    'type': 'Feature',
                    'geometry': {
                        'type': 'LineString',
                        'coordinates': list(exterior.coords)
                    },
                    'properties': {'name': 'Ceará', 'tipo': 'limite_externo'}
                })
    
    output3 = Path('data/raw/limites_ceara_ibge_linhas.geojson')
    with open(output3, 'w', encoding='utf-8') as f:
        json.dump({
            'type': 'FeatureCollection',
            'features': lines
        }, f)
    size3 = output3.stat().st_size / 1024
    print(f"\n[6] Formato 3 - Apenas Linhas (sem preenchimento):")
    print(f"    Arquivo: {output3.name}")
    print(f"    Tamanho: {size3:.1f} KB")
    print(f"    Features: {len(lines)} linhas")
    
    # Estatísticas finais
    print(f"\n{'='*80}")
    print(f"✓ SUCESSO - Todos os 3 formatos criados!")
    print(f"{'='*80}")
    print(f"\nRecomendação para testar no Leaflet:")
    print(f"  1. Tente {output3.name} (LineString - mais leve)")
    print(f"  2. Se não funcionar, tente {output2.name} (simples)")
    print(f"  3. Como fallback use {output1.name} (completo)")
    
    print(f"\nPróximo passo: Atualizar dashboard_estrategico.html para usar:")
    print(f"  /data/raw/limites_ceara_ibge_linhas.geojson")
    
except requests.exceptions.RequestException as e:
    print(f"✗ Erro na requisição: {e}")
except Exception as e:
    print(f"✗ Erro geral: {e}")
    import traceback
    traceback.print_exc()

print("\n")
