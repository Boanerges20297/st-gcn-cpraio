"""CVLI seasonality analysis using dados_status_ocorrencias_gerais.json only.

Outputs (into `outputs/` and `outputs/docs/`):
 - cvli_timeseries_status_daily.csv
 - cvli_monthly_status_summary.csv
 - cvli_by_bairro_cidade_status_monthly.csv
 - cvli_bairro_cidade_status_stats.csv
 - cvli_dow_status_summary.csv
 - cvli_autocorr_status.csv
 - outputs/docs/cvli_seasonality_status.md
"""
import json
import os
from pathlib import Path
import pandas as pd
import numpy as np


def load_table_status(path):
    with open(path, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    # find object that contains a 'data' array
    for item in raw:
        if isinstance(item, dict) and 'data' in item and isinstance(item['data'], list):
            return pd.DataFrame(item['data'])
    # fallback: try to build DataFrame from top-level list of dicts
    return pd.DataFrame([r for r in raw if isinstance(r, dict)])


def main():
    os.makedirs('outputs', exist_ok=True)
    os.makedirs('outputs/docs', exist_ok=True)

    fp = 'data/raw/dados_status_ocorrencias_gerais.json'
    df = load_table_status(fp)
    orig = len(df)

    # normalize column names
    df.columns = [c.strip() for c in df.columns]
    # date column in this file is 'data'
    if 'data' not in df.columns:
        raise RuntimeError('Expected column `data` not found in status file')
    df['Data'] = pd.to_datetime(df['data'], errors='coerce')
    df = df[df['Data'].notna()].copy()
    rows_with_date = len(df)

    # normalize tipo
    if 'tipo' in df.columns:
        df['tipo_norm'] = df['tipo'].astype(str).str.strip().str.lower()
    else:
        df['tipo_norm'] = ''

    # select CVLI rows (case-insensitive contains 'cvli')
    df_cvli = df[df['tipo_norm'].str.contains('cvli', na=False)].copy()
    total_cvli = len(df_cvli)

    # ensure city/bairro columns exist
    df_cvli['cidade_up'] = df_cvli.get('cidade', '').astype(str).str.upper()
    df_cvli['bairro_up'] = df_cvli.get('bairro', '').astype(str).str.upper()
    df_cvli['bairro_cidade'] = df_cvli['bairro_up'].fillna('') + ' / ' + df_cvli['cidade_up'].fillna('')

    # enforce analysis window 2022-01-01 .. 2026-12-31
    start_period = pd.to_datetime('2022-01-01')
    end_period = pd.to_datetime('2026-12-31')
    df_cvli = df_cvli[(df_cvli['Data'] >= start_period) & (df_cvli['Data'] <= end_period)].copy()

    # daily series
    full_index = pd.date_range(start_period, end_period, freq='D')
    ts = df_cvli.groupby(df_cvli['Data'].dt.normalize()).size().reindex(full_index, fill_value=0).rename('cvli_count').reset_index().rename(columns={'index':'Data'})
    ts.to_csv('outputs/cvli_timeseries_status_daily.csv', index=False)

    # monthly
    df_cvli['month'] = df_cvli['Data'].dt.to_period('M')
    monthly = df_cvli.groupby('month').size().rename('cvli_monthly_total').reset_index()
    monthly['month'] = monthly['month'].astype(str)
    monthly.to_csv('outputs/cvli_monthly_status_summary.csv', index=False)

    # by bairro/cidade monthly
    by_bc = df_cvli.groupby([df_cvli['bairro_cidade'], df_cvli['month']]).size().rename('cvli_monthly').reset_index()
    by_bc.to_csv('outputs/cvli_by_bairro_cidade_status_monthly.csv', index=False)

    stats = by_bc.groupby('bairro_cidade')['cvli_monthly'].agg(['mean','std','max']).reset_index().rename(columns={'mean':'mean_monthly','std':'std_monthly','max':'peak_monthly'})
    stats['cv'] = stats['std_monthly'] / stats['mean_monthly'].replace(0, np.nan)
    stats = stats.fillna(0).sort_values('mean_monthly', ascending=False)
    stats.to_csv('outputs/cvli_bairro_cidade_status_stats.csv', index=False)

    # day-of-week
    df_cvli['dow'] = df_cvli['Data'].dt.day_name()
    dow_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    dow = df_cvli.groupby('dow').size().reindex(dow_order).fillna(0).rename('cvli_dow_total').reset_index()
    dow.to_csv('outputs/cvli_dow_status_summary.csv', index=False)

    # autocorr
    ts_idx = pd.date_range(ts['Data'].min(), ts['Data'].max(), freq='D')
    daily = ts.set_index('Data').reindex(ts_idx, fill_value=0)['cvli_count'].values
    n = len(daily)
    maxlag = min(60, n-1)
    acf = [1.0]
    daily_mean = daily.mean() if n>0 else 0.0
    daily_var = ((daily - daily_mean)**2).sum()
    for lag in range(1, maxlag+1):
        if daily_var > 0:
            c = ((daily[:-lag] - daily_mean) * (daily[lag:] - daily_mean)).sum() / daily_var
        else:
            c = 0.0
        acf.append(float(c))
    acf_df = pd.DataFrame({'lag': list(range(0, maxlag+1)), 'acf': acf})
    acf_df.to_csv('outputs/cvli_autocorr_status.csv', index=False)

    # summary and markdown
    avg_per_day = daily.mean() if n>0 else 0.0
    peak_idx = int(np.argmax(daily)) if n>0 else None
    peak_date = ts_idx[peak_idx].date().isoformat() if peak_idx is not None else ''
    peak_value = int(daily[peak_idx]) if peak_idx is not None else 0

    mdp = Path('outputs/docs/cvli_seasonality_status.md')
    with mdp.open('w', encoding='utf-8') as f:
        f.write('# Sazonalidade CVLI (arquivo dados_status_ocorrencias_gerais.json)\n\n')
        f.write('Fonte: `data/raw/dados_status_ocorrencias_gerais.json`\n\n')
        f.write('## Parâmetros\n')
        f.write('- Identificação: campo `tipo`, filtrando por valores que contêm `cvli` (case-insensitive).\n')
        f.write('- Período: 2022-01-01 .. 2026-12-31 (forçado).\n')
        f.write(f'- Linhas lidas: {orig}; linhas com data válida: {rows_with_date}; CVLI selecionados: {total_cvli}.\n\n')
        f.write('## Resultados resumidos\n')
        f.write(f'- Média diária: {avg_per_day:.3f} CVLI/dia\n')
        f.write(f'- Dia de pico: {peak_date} ({peak_value} CVLI)\n')
        f.write('\n## Top bairros/cidades (média mensal)\n')
        top5 = stats.head(10)
        for _, r in top5.head(10).iterrows():
            f.write(f"- {r['bairro_cidade']}: média mensal={r['mean_monthly']:.2f}, pico={int(r['peak_monthly'])}, CV={r['cv']:.2f}\n")
        f.write('\n## Arquivos gerados\n')
        f.write('- outputs/cvli_timeseries_status_daily.csv\n')
        f.write('- outputs/cvli_monthly_status_summary.csv\n')
        f.write('- outputs/cvli_by_bairro_cidade_status_monthly.csv\n')
        f.write('- outputs/cvli_bairro_cidade_status_stats.csv\n')
        f.write('- outputs/cvli_dow_status_summary.csv\n')
        f.write('- outputs/cvli_autocorr_status.csv\n')

    print('Status CVLI seasonality complete. Report at', mdp)


if __name__ == '__main__':
    main()
