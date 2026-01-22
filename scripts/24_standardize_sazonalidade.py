import json
from pathlib import Path

import pandas as pd


OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

MONTHLY_CSV = OUT / 'sazonalidade_bairro_cidade_monthly.csv'
WEEKDAY_CSV = OUT / 'sazonalidade_bairro_cidade_weekday.csv'
HOURLY_CSV = OUT / 'sazonalidade_bairro_cidade_hourly.csv'

MONTHLY_INDEX_CSV = OUT / 'sazonalidade_bairro_cidade_monthly_index.csv'
WEEKDAY_INDEX_CSV = OUT / 'sazonalidade_bairro_cidade_weekday_index.csv'
HOURLY_INDEX_CSV = OUT / 'sazonalidade_bairro_cidade_hourly_index.csv'
MONTHLY_INDEX_MATRIX = OUT / 'sazonalidade_bairro_cidade_monthly_index_matrix.csv'

REPORT_MD = OUT / 'sazonalidade_report_bairro_cidade.md'


def load_csv(p):
    if not p.exists():
        raise FileNotFoundError(p)
    return pd.read_csv(p)


def compute_monthly_index(monthly_df):
    monthly_df['count'] = pd.to_numeric(monthly_df['count'], errors='coerce').fillna(0)
    # average across years for same calendar month
    avg = (
        monthly_df.groupby(['cidade','bairro','month'], dropna=False)['count']
        .mean()
        .reset_index(name='mean_count')
    )

    def compute_idx(group):
        mean_overall = group['mean_count'].mean()
        if mean_overall == 0:
            group['index'] = 0.0
        else:
            group['index'] = group['mean_count'] / mean_overall
        total = group['mean_count'].sum()
        group['proportion'] = group['mean_count'] / total if total > 0 else 0.0
        return group

    idx = avg.groupby(['cidade','bairro']).apply(compute_idx).reset_index(drop=True)
    return idx


def compute_weekday_index(weekday_df):
    weekday_df['count'] = pd.to_numeric(weekday_df['count'], errors='coerce').fillna(0)
    avg = (
        weekday_df.groupby(['cidade','bairro','weekday'], dropna=False)['count']
        .mean()
        .reset_index(name='mean_count')
    )

    def compute_idx(group):
        mean_overall = group['mean_count'].mean()
        if mean_overall == 0:
            group['index'] = 0.0
        else:
            group['index'] = group['mean_count'] / mean_overall
        total = group['mean_count'].sum()
        group['proportion'] = group['mean_count'] / total if total > 0 else 0.0
        return group

    idx = avg.groupby(['cidade','bairro']).apply(compute_idx).reset_index(drop=True)
    return idx


def compute_hourly_index(hourly_df):
    hourly_df['count'] = pd.to_numeric(hourly_df['count'], errors='coerce').fillna(0)
    avg = (
        hourly_df.groupby(['cidade','bairro','hour'], dropna=False)['count']
        .mean()
        .reset_index(name='mean_count')
    )

    def compute_idx(group):
        mean_overall = group['mean_count'].mean()
        if mean_overall == 0:
            group['index'] = 0.0
        else:
            group['index'] = group['mean_count'] / mean_overall
        total = group['mean_count'].sum()
        group['proportion'] = group['mean_count'] / total if total > 0 else 0.0
        return group

    idx = avg.groupby(['cidade','bairro']).apply(compute_idx).reset_index(drop=True)
    return idx


def main():
    monthly = load_csv(MONTHLY_CSV)
    weekday = load_csv(WEEKDAY_CSV)
    hourly = load_csv(HOURLY_CSV)

    m_idx = compute_monthly_index(monthly)
    w_idx = compute_weekday_index(weekday)
    h_idx = compute_hourly_index(hourly)

    m_idx.to_csv(MONTHLY_INDEX_CSV, index=False)
    w_idx.to_csv(WEEKDAY_INDEX_CSV, index=False)
    h_idx.to_csv(HOURLY_INDEX_CSV, index=False)

    # pivot monthly index to matrix
    pivot = m_idx.pivot_table(index=['cidade','bairro'], columns='month', values='index', fill_value=0)
    pivot.reset_index(inplace=True)
    pivot.to_csv(MONTHLY_INDEX_MATRIX, index=False)

    # append short summary to report
    summary_lines = []
    summary_lines.append('\n')
    summary_lines.append('## Índices padronizados de sazonalidade (mês/dia/horário)')
    summary_lines.append('\n')
    summary_lines.append(f'- Monthly index CSV: {MONTHLY_INDEX_CSV}')
    summary_lines.append(f'- Monthly index matrix: {MONTHLY_INDEX_MATRIX}')
    summary_lines.append(f'- Weekday index CSV: {WEEKDAY_INDEX_CSV}')
    summary_lines.append(f'- Hourly index CSV: {HOURLY_INDEX_CSV}')
    summary_lines.append('\n')

    with open(REPORT_MD, 'a', encoding='utf-8') as f:
        f.write('\n'.join(summary_lines))

    print('Wrote index CSVs and updated report:')
    print(MONTHLY_INDEX_CSV)
    print(WEEKDAY_INDEX_CSV)
    print(HOURLY_INDEX_CSV)


if __name__ == '__main__':
    main()
