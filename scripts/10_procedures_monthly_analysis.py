"""Aggregate 'procedimentos realizados' from raw JSON, compute monthly correlations
and estimate policing effectiveness heuristics.

Outputs saved to `outputs/`:
 - procedures_monthly_by_bairro.csv
 - procedures_monthly_summary.csv
 - procedures_vs_homicides_lagged.csv
 - policing_effectiveness_by_bairro.csv

Run:
    python scripts/10_procedures_monthly_analysis.py
"""
import os
import sys
import json
import numpy as np
import pandas as pd

# project root
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

OUT = 'outputs'
os.makedirs(OUT, exist_ok=True)


def load_raw_json(path):
    # The file exported by phpMyAdmin has nested structure; attempt to extract 'data' table
    with open(path, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    # raw is a list; find item with 'type'=='table' and 'data'
    for item in raw:
        if isinstance(item, dict) and item.get('type') == 'table' and 'data' in item:
            return pd.DataFrame(item['data'])
    # fallback: convert to DataFrame directly
    return pd.DataFrame(raw)


def normalize_cols(df):
    # normalize keys to common names
    mapping = {}
    for c in df.columns:
        lc = c.lower()
        if 'data' == lc or lc.startswith('data'):
            mapping[c] = 'date'
        if 'bairro' in lc:
            mapping[c] = 'bairro'
        if 'cidade' in lc:
            mapping[c] = 'cidade'
        if 'natureza' in lc:
            mapping[c] = 'natureza'
        if 'total_armas' in lc or 'armas' in lc:
            mapping[c] = 'total_armas'
        if 'total_drogas' in lc or 'drogas' in lc:
            mapping[c] = 'total_drogas'
        if 'dinheiro' in lc or 'dinheiro_apreendido' in lc:
            mapping[c] = 'dinheiro'
    df = df.rename(columns=mapping)
    return df


def detect_homicide(natureza):
    try:
        s = str(natureza).lower()
        return int('homicid' in s or 'homicídio' in s or 'homicidio' in s)
    except Exception:
        return 0


def safe_to_numeric(x):
    try:
        return float(str(x).replace(',', '.'))
    except Exception:
        return 0.0


def main():
    raw_path = os.path.join('data', 'raw', 'ocorrencia_policial_operacional.json')
    if not os.path.exists(raw_path):
        print('Raw JSON not found:', raw_path)
        return

    print('Loading raw procedures JSON...')
    df = load_raw_json(raw_path)
    print('Rows loaded:', len(df))

    df = normalize_cols(df)

    # parse date
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    else:
        # try common names
        for c in df.columns:
            if 'data' in c.lower():
                df['date'] = pd.to_datetime(df[c], errors='coerce')
                break

    df['year_month'] = df['date'].dt.to_period('M').astype(str)

    # ensure bairro
    if 'bairro' not in df.columns:
        # try BairroOcor style
        for c in df.columns:
            if 'bairro' in c.lower():
                df['bairro'] = df[c]
                break

    df['bairro'] = df['bairro'].fillna('UNKNOWN').astype(str).str.upper().str.strip()

    # Mark procedure indicator (every row is a procedure)
    df['procedimento'] = 1

    # numeric conversions
    df['total_armas'] = df.get('total_armas', df.get('total_armas_cache', 0))
    df['total_armas'] = df['total_armas'].apply(safe_to_numeric).fillna(0).astype(int)
    # try several possible drug columns
    drug_cols = [c for c in df.columns if 'drog' in c.lower()]
    if drug_cols:
        df['total_drogas'] = df[drug_cols[0]].apply(safe_to_numeric).fillna(0)
    else:
        df['total_drogas'] = 0.0
    money_cols = [c for c in df.columns if 'dinheiro' in c.lower()]
    if money_cols:
        df['dinheiro'] = df[money_cols[0]].apply(safe_to_numeric).fillna(0)
    else:
        df['dinheiro'] = 0.0

    # homicide flag
    df['homicidio'] = df.get('natureza', '').apply(detect_homicide)

    # Aggregate monthly per bairro
    agg = df.groupby(['year_month', 'bairro']).agg(
        procedimentos=('procedimento', 'sum'),
        armas_sequestradas=('total_armas', 'sum'),
        drogas_kg=('total_drogas', 'sum'),
        dinheiro_reais=('dinheiro', 'sum'),
        homicidios=('homicidio', 'sum')
    ).reset_index()

    agg.to_csv(os.path.join(OUT, 'procedures_monthly_by_bairro.csv'), index=False)
    print('Saved monthly aggregation by bairro to outputs/procedures_monthly_by_bairro.csv')

    # Global monthly summary
    monthly = agg.groupby('year_month').agg(
        procedimentos_total=('procedimentos', 'sum'),
        armas_total=('armas_sequestradas', 'sum'),
        drogas_total=('drogas_kg', 'sum'),
        dinheiro_total=('dinheiro_reais', 'sum'),
        homicidios_total=('homicidios', 'sum')
    ).reset_index()
    monthly.to_csv(os.path.join(OUT, 'procedures_monthly_summary.csv'), index=False)
    print('Saved monthly summary to outputs/procedures_monthly_summary.csv')

    # Correlations contemporaneous and lagged (procedures/armas/drogas vs homicidios)
    dfm = monthly.copy()
    dfm['year_month'] = pd.to_datetime(dfm['year_month']).dt.to_period('M')
    dfm = dfm.sort_values('year_month')
    dfm = dfm.set_index('year_month')

    results = []
    for feature in ['procedimentos_total', 'armas_total', 'drogas_total', 'dinheiro_total']:
        for lag in [0, 1, 3]:
            # correlate feature at t with homicidios at t+lag
            x = dfm[feature].shift(0)
            y = dfm['homicidios_total'].shift(-lag)
            valid = x.notna() & y.notna()
            if valid.sum() < 2:
                r = np.nan
            else:
                r = x[valid].corr(y[valid])
            results.append({'feature': feature, 'lag_months': lag, 'pearson_r': r})

    pd.DataFrame(results).to_csv(os.path.join(OUT, 'procedures_vs_homicides_lagged.csv'), index=False)
    print('Saved lagged correlations to outputs/procedures_vs_homicides_lagged.csv')

    # Policing effectiveness heuristic per bairro
    # For each bairro, find months in top quartile of procedimentos and compare avg homicidios next month vs prev month
    eff_rows = []
    for bairro, g in agg.groupby('bairro'):
        g = g.sort_values('year_month')
        # convert year_month to period for shifting
        g['ym'] = pd.to_datetime(g['year_month']).dt.to_period('M')
        if len(g) < 6:
            continue
        thresh = g['procedimentos'].quantile(0.75)
        high_months = g[g['procedimentos'] >= thresh]
        deltas = []
        for _, row in high_months.iterrows():
            ym = pd.Period(row['year_month'], freq='M')
            prev = g[g['ym'] == (ym - 1)]['homicidios']
            nxt = g[g['ym'] == (ym + 1)]['homicidios']
            if prev.empty or nxt.empty:
                continue
            prev_mean = prev.mean()
            next_mean = nxt.mean()
            # reduction positive means fewer homicides after high procedures
            reduction = prev_mean - next_mean
            deltas.append(reduction)
        if deltas:
            eff = float(np.nanmean(deltas))
            eff_rows.append({'bairro': bairro, 'n_high_months': len(deltas), 'mean_homicide_reduction_next_vs_prev': eff})

    pd.DataFrame(eff_rows).to_csv(os.path.join(OUT, 'policing_effectiveness_by_bairro.csv'), index=False)
    print('Saved policing effectiveness heuristic to outputs/policing_effectiveness_by_bairro.csv')

    # Print top-level summaries
    print('\nTop lagged correlations (abs):')
    dfcorr = pd.read_csv(os.path.join(OUT, 'procedures_vs_homicides_lagged.csv'))
    dfcorr['abs_r'] = dfcorr['pearson_r'].abs()
    print(dfcorr.sort_values('abs_r', ascending=False).head(10).to_string(index=False))


if __name__ == '__main__':
    main()
