"""Detect arrests in consolidated raw data, aggregate per day+neighborhood,
merge with `prisoes_with_features.parquet` and compute correlations.

Outputs:
 - outputs/arrests_aggregated.csv
 - outputs/arrests_vs_occurrences_by_neighborhood.csv
 - outputs/arrests_overall_correlation.csv
"""
import os
import sys
import numpy as np
import pandas as pd

# ensure project root on path and import config
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src import config


def detect_arrest_rows(df):
    kw = ['pris', 'prisa', 'prisao', 'detenc', 'detido', 'apreens', 'apreensão', 'prisão']
    cols = [c for c in df.columns if c.lower() in ['natureza','tipo','descricao','descricao_evento','acao'] or 'desc' in c.lower()]
    if not cols:
        cols = df.columns.tolist()
    mask = pd.Series(False, index=df.index)
    for c in cols:
        mask = mask | df[c].astype(str).str.lower().fillna('').str.contains('|'.join(kw), na=False)
    return mask


def main():
    out = 'outputs'
    os.makedirs(out, exist_ok=True)

    cons_path = str(config.CONSOLIDATED_FILE)
    if os.path.exists(cons_path):
        print('Loading consolidated data (may be large):', cons_path)
        cons = pd.read_parquet(cons_path)
    else:
        # Fallback: try raw exports (JSON) that contain 'Natureza' field
        alt = os.path.join('data', 'raw', 'ocorrencia_policial_operacional.json')
        if os.path.exists(alt):
            print('[WARN] Consolidated file not found; falling back to raw JSON:', alt)
            # load JSON array exported by phpmyadmin style
            cons = pd.read_json(alt)
            # the JSON contains a top-level array with header/meta; try to find the 'data' table
            if isinstance(cons, pd.DataFrame) and 'type' in cons.columns:
                # not expected, try reading differently
                try:
                    # read file as lines and parse the array inside 'data' key
                    import json as _json
                    with open(alt, 'r', encoding='utf-8') as f:
                        raw = _json.load(f)
                    # find the object with name 'table' and 'data'
                    for item in raw:
                        if isinstance(item, dict) and item.get('type') == 'table' and 'data' in item:
                            cons = pd.DataFrame(item['data'])
                            break
                except Exception:
                    pass
        else:
            print('[ERR] Consolidated file not found:', cons_path)
            return
    print('Consolidated rows:', len(cons))

    # normalize date column
    date_col = None
    for c in ['data_hora', 'data', 'date', 'datetime']:
        if c in cons.columns:
            date_col = c
            break
    if date_col is None:
        # try to find any datetime-like column
        for c in cons.columns:
            if 'data' in c.lower() or 'date' in c.lower() or 'hora' in c.lower():
                date_col = c
                break

    if date_col is None:
        print('[ERR] No date column found in consolidated data')
        return

    cons['date'] = pd.to_datetime(cons[date_col], errors='coerce').dt.date

    # ensure neighborhood mapping exists: try 'local_oficial' or 'bairro' or 'bairro_id'
    loc_col = None
    for c in ['local_oficial', 'bairro', 'bairro_id', 'local_norm', 'local']:
        if c in cons.columns:
            loc_col = c
            break
    if loc_col is None:
        print('[WARN] No neighborhood column found; attempting to use coordenadas mapping later')
        # fallback: abort per-neighborhood aggregation

    # detect arrests
    print('Detecting arrest rows (keyword heuristic)')
    arrests_mask = detect_arrest_rows(cons)
    cons['is_arrest'] = arrests_mask.astype(bool)

    # aggregate arrests per day and neighborhood (if possible)
    if loc_col:
        agg = cons[cons['is_arrest']].groupby(['date', loc_col]).size().rename('arrest_count').reset_index()
        agg.to_csv(os.path.join(out, 'arrests_aggregated.csv'), index=False)
        print('Saved arrests aggregated (date, neighborhood) to outputs/arrests_aggregated.csv')
    else:
        # aggregate only by date
        agg = cons[cons['is_arrest']].groupby(['date']).size().rename('arrest_count').reset_index()
        agg.to_csv(os.path.join(out, 'arrests_aggregated_by_date.csv'), index=False)
        print('Saved arrests aggregated by date to outputs/arrests_aggregated_by_date.csv')

    # load occurrences aggregated (prisoes_with_features.parquet)
    occ_path = os.path.join('data', 'processed', 'prisoes_with_features.parquet')
    if not os.path.exists(occ_path):
        print('[ERR] Occurrence features file not found:', occ_path)
        return
    occ = pd.read_parquet(occ_path)
    # normalize date in occ
    if 'Data' in occ.columns:
        occ['date'] = pd.to_datetime(occ['Data'], errors='coerce').dt.date
    elif 'data' in occ.columns:
        occ['date'] = pd.to_datetime(occ['data'], errors='coerce').dt.date
    else:
        print('[WARN] No date column found in prisoes_with_features.parquet')

    # Aggregate occurrences per date (all neighborhoods)
    if 'operacoes_diarias' in occ.columns:
        occ_daily = occ.groupby('date')['operacoes_diarias'].sum().reset_index().rename(columns={'operacoes_diarias': 'operacoes_total'})
    else:
        # fallback: count rows per date
        occ_daily = occ.groupby('date').size().reset_index().rename(columns={0: 'operacoes_total'})

    # merge with arrests aggregated by date
    if 'arrest_count' in agg.columns:
        arrests_daily = agg.groupby('date')['arrest_count'].sum().reset_index()
    else:
        arrests_daily = agg.copy()

    merged_daily = pd.merge(occ_daily, arrests_daily, on='date', how='inner')
    if merged_daily.empty:
        print('[WARN] No overlapping dates between occurrences and detected arrests; result empty')
    else:
        merged_daily.to_csv(os.path.join(out, 'arrests_daily_vs_occurrences.csv'), index=False)
        r = merged_daily['operacoes_total'].corr(merged_daily['arrest_count'])
        pd.DataFrame([{'operacoes_vs_arrests_corr': r}]).to_csv(os.path.join(out, 'arrests_overall_correlation.csv'), index=False)
        print(f"Saved daily merged series to outputs/arrests_daily_vs_occurrences.csv and overall corr r={r:.4f} to outputs/arrests_overall_correlation.csv")

    # Note: per-neighborhood mapping requires spatial join (coords -> polygons) which is not done here.


if __name__ == '__main__':
    main()
