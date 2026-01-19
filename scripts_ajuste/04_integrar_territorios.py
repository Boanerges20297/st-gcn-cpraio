#!/usr/bin/env python3
"""
Sprint 2 Task 2.1: Integrar GeoJSONs de facções e calcular pesos de features

- Carrega `data/processed/unified_2025.parquet`
- Carrega GeoJSONs de `data/raw/inteligencia/*.geojson`
- Atribui `area_faccao` por ponto (ponto-em-polígono)
- Calcula `feature_score` com regras: CVLI*3, drogas(kg)*2 (cap), armas+droga*2, armas*0.1
- Salva `unified_with_territories.parquet` e `territory_daily_features.parquet`
"""

import os
import glob
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point

UNIFIED_IN = 'data/processed/unified_2025.parquet'
OUT_UNIFIED = 'data/processed/unified_with_territories.parquet'
OUT_TERR_DAILY = 'data/processed/territory_daily_features.parquet'


def load_geojsons(intel_dir='data/raw/inteligencia'):
    files = glob.glob(os.path.join(intel_dir, '*.geojson'))
    gdfs = []
    for f in files:
        try:
            g = gpd.read_file(f)
            name = os.path.splitext(os.path.basename(f))[0]
            # normalize column name for facção
            if 'name' in g.columns:
                g['faccao_src'] = g['name']
            else:
                g['faccao_src'] = name
            gdfs.append(g[['geometry','faccao_src']])
        except Exception as e:
            print(f"Erro lendo {f}: {e}")
    if not gdfs:
        return None
    all_g = pd.concat(gdfs, ignore_index=True)
    g_all = gpd.GeoDataFrame(all_g, geometry='geometry', crs=gdfs[0].crs)
    return g_all


def main():
    print('\n[1] Carregando unified dataset...')
    if not os.path.exists(UNIFIED_IN):
        print(f'Arquivo não encontrado: {UNIFIED_IN}')
        return
    df = pd.read_parquet(UNIFIED_IN)
    # converter data para datetime
    df['data'] = pd.to_datetime(df['data'], errors='coerce')

    print(f"    - Registros: {len(df)} | Datas: {df['data'].min()} → {df['data'].max()}")

    print('\n[2] Carregando GeoJSONs de inteligencia...')
    g_all = load_geojsons()
    if g_all is None or len(g_all)==0:
        print('Nenhum GeoJSON de inteligencia encontrado. Abortando.')
        return
    print(f'    - GeoPolígonos carregados: {len(g_all)}')

    # Criar GeoDataFrame de pontos
    print('\n[3] Criando GeoDataFrame de pontos a partir de lat/long...')
    df_points = df.copy()
    df_points['lat'] = pd.to_numeric(df_points['lat'], errors='coerce')
    df_points['long'] = pd.to_numeric(df_points['long'], errors='coerce')
    df_points = df_points.dropna(subset=['lat','long']).copy()
    geometry = [Point(xy) for xy in zip(df_points['long'], df_points['lat'])]
    # Garantir CRS consistente (usar EPSG:4326 se ausente)
    target_crs = g_all.crs if g_all.crs is not None else 'EPSG:4326'
    gpts = gpd.GeoDataFrame(df_points, geometry=geometry, crs=target_crs)

    print(f"    - Pontos válidos: {len(gpts)}")

    print('\n[4] Realizando spatial join (ponto → polígono)')
    # Antes do sjoin, preservar índice original como coluna para re-merge claro
    gpts = gpts.reset_index().rename(columns={'index': 'orig_index'})
    joined = gpd.sjoin(gpts, g_all, how='left', predicate='within')
    # joined agora tem coluna 'faccao_src' quando bateu
    joined_cols = joined.copy()
    joined_cols['area_faccao_assigned'] = joined_cols['faccao_src'].fillna(joined_cols['area_faccao'])

    # Mapear orig_index -> area_faccao_assigned
    if 'orig_index' in joined_cols.columns:
        joined_map = joined_cols[['orig_index','area_faccao_assigned']].dropna(subset=['area_faccao_assigned']).drop_duplicates(subset=['orig_index'])
        joined_map = joined_map.set_index('orig_index')
        df_updated = df.copy()
        for idx, row in joined_map.iterrows():
            try:
                df_updated.at[int(idx),'area_faccao'] = row['area_faccao_assigned']
            except Exception:
                pass
    else:
        # Fallback: se não existir orig_index, ignorar atribuição em lote
        df_updated = df.copy()
        print('    [WARN] orig_index não encontrado após sjoin — nenhuma atribuição aplicada')

    print('\n[5] Calculando feature_score por registro...')
    # Recompute helper flags
    df_updated['is_cvli'] = df_updated['is_cvli'].astype(bool)
    df_updated['has_large_seizure'] = df_updated['has_large_seizure'].astype(bool)
    df_updated['has_weapons_drugs'] = df_updated['has_weapons_drugs'].astype(bool)
    df_updated['total_drogas_g'] = pd.to_numeric(df_updated['total_drogas_g'], errors='coerce').fillna(0)
    df_updated['total_armas'] = pd.to_numeric(df_updated['total_armas'], errors='coerce').fillna(0)

    # Formula: CVLI*3 + min(drogas_kg,10)*2 + has_weapons_drugs*2 + armas*0.1
    df_updated['drogas_kg'] = df_updated['total_drogas_g'] / 1000.0
    df_updated['feature_score'] = 0.0
    df_updated['feature_score'] += df_updated['is_cvli'].astype(int) * 3.0
    df_updated['feature_score'] += np.minimum(df_updated['drogas_kg'], 10.0) * 2.0
    df_updated['feature_score'] += df_updated['has_weapons_drugs'].astype(int) * 2.0
    df_updated['feature_score'] += df_updated['total_armas'] * 0.1

    print(f"    - feature_score stats: mean={df_updated['feature_score'].mean():.3f}, max={df_updated['feature_score'].max():.3f}")

    print('\n[6] Salvando unified com territórios e scores...')
    os.makedirs(os.path.dirname(OUT_UNIFIED), exist_ok=True)
    save_df = df_updated.copy()
    save_df['data'] = save_df['data'].astype(str)
    save_df.to_parquet(OUT_UNIFIED, index=False)
    print(f"    - Salvo: {OUT_UNIFIED}")

    print('\n[7] Agregando por territorio (area_faccao) e dia...')
    agg = df_updated.copy()
    agg['date_day'] = agg['data'].dt.floor('d')
    agg_group = agg.groupby(['date_day','area_faccao']).agg({
        'feature_score':'sum',
        'is_cvli':'sum',
        'total_armas':'sum',
        'total_drogas_g':'sum',
        'has_large_seizure':'sum',
        'has_weapons_drugs':'sum'
    }).reset_index()
    agg_group = agg_group.rename(columns={'date_day':'data'})

    # salvar
    agg_group['data'] = agg_group['data'].astype(str)
    agg_group.to_parquet(OUT_TERR_DAILY, index=False)
    print(f"    - Salvo: {OUT_TERR_DAILY}")

    print('\n[8] Finalizado. Próximo: construir grafo ponderado e gerar tensores por nó/dia.')

if __name__ == '__main__':
    main()
