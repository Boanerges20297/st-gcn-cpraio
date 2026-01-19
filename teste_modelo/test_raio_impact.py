#!/usr/bin/env python
import os
import sys
import argparse
from datetime import datetime, timedelta
import json
import pandas as pd
import geopandas as gpd
import numpy as np

# ensure src in path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT, 'src')
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import config
from graph_builder import build_graph
import trainer
import predict
from data_loader import normalize_columns


def override_artifacts_for_test():
    # Create test dirs
    base_tensors = config.BASE_DIR / 'data' / 'tensors' / 'test_raio'
    base_models = config.BASE_DIR / 'outputs' / 'models' / 'test_raio'
    base_reports = config.BASE_DIR / 'outputs' / 'reports' / 'test_raio'
    os.makedirs(base_tensors, exist_ok=True)
    os.makedirs(base_models, exist_ok=True)
    os.makedirs(base_reports, exist_ok=True)

    # Backup original
    original = config.ARTIFACTS.get('CAPITAL').copy()

    config.ARTIFACTS['CAPITAL'] = {
        'dataset': base_tensors / 'dataset_capital_test_raio.pt',
        'model': base_models / 'model_capital_test_raio.pth',
        'stats': base_models / 'stats_capital_test_raio.pt',
        'prediction': base_reports / 'pred_capital_test_raio.csv'
    }

    return original


def restore_artifacts(original):
    config.ARTIFACTS['CAPITAL'] = original


def read_json_as_df(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, dict) and 'data' in data:
        df = pd.DataFrame(data['data'])
    elif isinstance(data, list):
        df = pd.DataFrame(data)
    else:
        df = pd.DataFrame([data])
    df = normalize_columns(df)
    return df


def map_points_to_nodes(df_points, region_geojson_path):
    # df_points must have lat and long
    gdf_pts = gpd.GeoDataFrame(df_points.copy(), geometry=gpd.points_from_xy(df_points['long'], df_points['lat']), crs='EPSG:4326')
    gdf_nodes = gpd.read_file(region_geojson_path)
    if gdf_nodes.crs != gdf_pts.crs:
        gdf_nodes = gdf_nodes.to_crs(gdf_pts.crs)
    joined = gpd.sjoin(gdf_pts, gdf_nodes, how='left', predicate='within')
    node_col = 'name' if 'name' in gdf_nodes.columns else gdf_nodes.columns[0]
    joined['local_oficial'] = joined[node_col].astype(str).str.upper().str.strip()
    return pd.DataFrame(joined.drop(columns='geometry'))


def ensure_latlon(df):
    # Normalize various possible latitude/longitude column names into 'lat' and 'long'
    cols = df.columns.str.lower()
    lat_candidates = [c for c in df.columns if 'lat' in c.lower()]
    lon_candidates = [c for c in df.columns if any(x in c.lower() for x in ['lon','lng','long','longitude'])]

    if lat_candidates and lon_candidates:
        df['lat'] = pd.to_numeric(df[lat_candidates[0]], errors='coerce')
        df['long'] = pd.to_numeric(df[lon_candidates[0]], errors='coerce')
        return df

    # Try nested geometry like GeoJSON feature
    if 'geometry' in df.columns:
        try:
            # geometry may be dict with coordinates
            def extract_coords(g):
                if isinstance(g, dict):
                    coords = g.get('coordinates') or g.get('coord')
                    if isinstance(coords, (list, tuple)) and len(coords) >= 2:
                        return coords[1], coords[0]
                return (None, None)

            coords = df['geometry'].apply(lambda x: extract_coords(x) if pd.notna(x) else (None, None))
            df['lat'] = coords.apply(lambda v: v[0])
            df['long'] = coords.apply(lambda v: v[1])
            return df
        except Exception:
            pass

    # Fallback: try to parse a 'location' or 'coords' text
    for c in df.columns:
        if 'coord' in c.lower() or 'location' in c.lower():
            try:
                parsed = df[c].astype(str).str.extract(r"\[?\s*([0-9\-\.]+)\s*,\s*([0-9\-\.]+)\s*\]?")
                if parsed.shape[1] == 2:
                    df['lat'] = pd.to_numeric(parsed[0], errors='coerce')
                    df['long'] = pd.to_numeric(parsed[1], errors='coerce')
                    return df
            except Exception:
                continue

    return df


def detect_arrest_rows(df):
    # Heurística: procura palavras-chave em colunas 'natureza','descricao','tipo' etc
    kw = ['pris', 'prisa', 'prisao', 'detenc', 'detido', 'apreens', 'apreensão', 'prisão']
    cols = [c for c in df.columns if c.lower() in ['natureza','tipo','descricao','descricao_evento','acao'] or 'desc' in c.lower()]
    if not cols:
        cols = df.columns.tolist()

    mask = pd.Series(False, index=df.index)
    for c in cols:
        mask = mask | df[c].astype(str).str.lower().fillna('').str.contains('|'.join(kw), na=False)
    return mask


def summarize_effects(pred_csv, observed_df_window, arrests_mask, nodes, target_window):
    # Load predictions
    pred = pd.read_csv(pred_csv)
    pred['local_norm'] = pred['local'].astype(str).str.upper().str.strip()

    # Observed counts per node in window
    obs_counts = observed_df_window.groupby('local_oficial').size().rename('count').reset_index()
    obs_counts['local_norm'] = obs_counts['local_oficial'].astype(str).str.upper().str.strip()

    # Observed without arrests
    df_no_arrests = observed_df_window[~arrests_mask]
    obs_no_counts = df_no_arrests.groupby('local_oficial').size().rename('count_no_arrests').reset_index()
    obs_no_counts['local_norm'] = obs_no_counts['local_oficial'].astype(str).str.upper().str.strip()

    # Merge
    merged = pd.DataFrame({'local_norm': nodes})
    merged = merged.merge(pred[['local_norm','risco_previsto']], on='local_norm', how='left')
    merged = merged.merge(obs_counts[['local_norm','count']], on='local_norm', how='left')
    merged = merged.merge(obs_no_counts[['local_norm','count_no_arrests']], on='local_norm', how='left')
    merged = merged.fillna(0)

    # Convert counts to mean per day
    merged['obs_mean'] = merged['count'] / target_window
    merged['obs_no_arrests_mean'] = merged['count_no_arrests'] / target_window

    # Pred is assumed to be mean per day predicted
    merged['pred_mean'] = merged['risco_previsto']

    # Reduction metrics
    merged['pred_minus_obs'] = merged['pred_mean'] - merged['obs_mean']
    merged['obs_no_minus_obs'] = merged['obs_no_arrests_mean'] - merged['obs_mean']
    merged['arrests_effect_count'] = merged['count'] - merged['count_no_arrests']

    # Aggregate summary
    total_pred = merged['pred_mean'].sum()
    total_obs = merged['obs_mean'].sum()
    total_no_arrests = merged['obs_no_arrests_mean'].sum()

    summary = {
        'total_pred_mean_per_day': float(total_pred),
        'total_obs_mean_per_day': float(total_obs),
        'total_no_arrests_mean_per_day': float(total_no_arrests),
        'delta_pred_obs_per_day': float(total_pred - total_obs),
        'delta_no_arrests_obs_per_day': float(total_no_arrests - total_obs)
    }

    return merged, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', help='Caminho para JSON de ocorrências 2025 (com prisões)', required=False,
                        default=r'C:\Users\STI01\Downloads\ocorrencia_policial_operacional.json')
    parser.add_argument('--region', default='CAPITAL')
    args = parser.parse_args()

    json_path = args.input
    region = args.region

    if not os.path.exists(json_path):
        print('[ERR] Arquivo de input não encontrado:', json_path)
        return

    print('[*] Iniciando avaliação RAIO impact — arquivo:', json_path)

    # Backup and override artifacts
    original = override_artifacts_for_test()

    try:
        # 1. Load consolidated history and filter 2022-2024
        cons = pd.read_parquet(config.CONSOLIDATED_FILE)
        cons['data_hora'] = pd.to_datetime(cons['data_hora'])
        hist = cons[(cons['data_hora'] >= '2022-01-01') & (cons['data_hora'] < '2025-01-01')].copy()
        print(f"[*] Histórico 2022-2024 carregado: {len(hist)} registros")

        # 2. Build graph/dataset from hist
        print('[*] Construindo dataset de treino (histórico)...')
        build_graph(region, hist)

        # 3. Train model (short run for test)
        print('[*] Ajustando parâmetros de treino para execução de teste...')
        trainer.TRAIN_CONFIG['epochs'] = 20
        trainer.TRAIN_CONFIG['batch_size'] = 16
        trainer.TRAIN_CONFIG['patience'] = 6
        trainer.train_region(region)

        # 4. Predict baseline for 2025 window
        print('[*] Gerando predição baseline (modelo treinado em 2022-2024)...')
        predict.predict_region(region)

        pred_csv = str(config.ARTIFACTS[region]['prediction'])

        # 5. Load provided 2025 data and map to nodes
        df_2025 = read_json_as_df(json_path)
        # ensure date col
        if 'date' in df_2025.columns:
            df_2025['date'] = pd.to_datetime(df_2025['date'], errors='coerce')
        elif 'data_hora' in df_2025.columns:
            df_2025['date'] = pd.to_datetime(df_2025['data_hora'], errors='coerce')
        else:
            # try any datetime-like
            for c in df_2025.columns:
                if 'data' in c.lower() or 'date' in c.lower():
                    df_2025['date'] = pd.to_datetime(df_2025[c], errors='coerce')
                    break

        # Ensure lat/long columns exist (try to normalize common names)
        df_2025 = ensure_latlon(df_2025)
        # Spatial join to map nodes
        mapped = map_points_to_nodes(df_2025.dropna(subset=['lat','long']), config.GEOJSON_PATHS[region])
        mapped['date'] = pd.to_datetime(mapped['date'])

        # 6. Determine predicted window from dataset dates
        ds = __import__('torch').load(config.ARTIFACTS[region]['dataset'], weights_only=False)
        dates = ds['dates']
        last_date = pd.to_datetime(dates[-1]).date()
        target_window = config.HyperParams['target_window']
        pred_start = last_date + timedelta(days=1)
        pred_end = last_date + timedelta(days=target_window)
        print(f"[*] Janela de predição: {pred_start} -> {pred_end} ({target_window} dias)")

        # Filter observed 2025 to the predicted window
        obs_window = mapped[(mapped['date'].dt.date >= pred_start) & (mapped['date'].dt.date <= pred_end)].copy()
        print(f"[*] Ocorrências observadas na janela: {len(obs_window)}")

        # 7. Detect arrests in observed window
        arrests_mask_full = detect_arrest_rows(obs_window)
        arrests_mask = arrests_mask_full.fillna(False)

        # 8. Summarize effects
        nodes = ds['nodes']
        merged, summary = summarize_effects(pred_csv, obs_window, arrests_mask, nodes, target_window)

        out_report = config.BASE_DIR / 'outputs' / 'reports' / 'test_raio_impact_report.csv'
        merged.to_csv(out_report, index=False)

        print('\n[RESULT] Sumário (médias por dia na janela):')
        for k,v in summary.items():
            print(f" - {k}: {v:.3f}")

        print(f"\n[RESULT] Relatório detalhado salvo em: {out_report}")

    finally:
        restore_artifacts(original)


if __name__ == '__main__':
    main()
