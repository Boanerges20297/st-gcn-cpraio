import json
from pathlib import Path
import pandas as pd
import numpy as np
from collections import Counter

OUT = Path('outputs')
DOCS = OUT / 'docs'
DOCS.mkdir(exist_ok=True)

ASSIGNED = Path('data/raw/dados_status_ocorrencias_gerais_bairros_atribuidos.json')
OUT_CSV = OUT / 'cvli_trends_bairro_cidade.csv'
OUT_MD = DOCS / 'cvli_trends_summary_bairro_cidade.md'


def load_wrapper(path):
    with open(path, 'r', encoding='utf-8') as f:
        wrapper = json.load(f)
    # find data array
    if isinstance(wrapper, dict) and 'data' in wrapper:
        return wrapper['data']
    if isinstance(wrapper, list):
        for el in wrapper:
            if isinstance(el, dict) and 'data' in el:
                return el['data']
    # fallback: if list of dicts
    if isinstance(wrapper, list) and len(wrapper) > 0 and isinstance(wrapper[0], dict):
        return wrapper
    raise ValueError('Could not find data array in wrapper')


def prepare_df(records):
    df = pd.json_normalize(records)
    # normalize column names
    # detect date/time
    date_col = None
    time_col = None
    for c in df.columns:
        lc = c.lower()
        if lc == 'data' or lc.startswith('data'):
            date_col = c
        if lc == 'hora' or lc.startswith('hora'):
            time_col = c
    if date_col is None:
        raise ValueError('No date column found')
    df['hora'] = df[time_col] if time_col in df.columns else None
    df['datetime'] = pd.to_datetime(df[date_col].astype(str) + ' ' + df['hora'].fillna('00:00:00').astype(str), errors='coerce')
    df = df[df['datetime'].notna()].copy()
    df['year'] = df['datetime'].dt.year
    df['month'] = df['datetime'].dt.month
    df['year_month'] = df['datetime'].dt.to_period('M').astype(str)
    df['weekday'] = df['datetime'].dt.day_name(locale='pt_BR')
    df['hour'] = df['datetime'].dt.hour
    # normalize bairro/cidade
    if 'bairro' not in df.columns:
        df['bairro'] = ''
    if 'cidade' not in df.columns:
        df['cidade'] = ''
    df['bairro'] = df['bairro'].fillna('').astype(str).str.upper().str.strip()
    df['cidade'] = df['cidade'].fillna('').astype(str).str.upper().str.strip()
    # filter CVLI
    tipo_col = None
    for c in df.columns:
        if c.lower() == 'tipo':
            tipo_col = c
            break
    if tipo_col:
        df = df[df[tipo_col].astype(str).str.lower() == 'cvli'].copy()
    return df


def analyze(df):
    # filter since 2022
    df = df[df['year'] >= 2022].copy()

    results = []
    # iterate over pairs
    grouped = df.groupby(['cidade','bairro'])
    for (cidade, bairro), g in grouped:
        # monthly series
        monthly = g.groupby('year_month').size().sort_index()
        if monthly.empty:
            continue
        latest_month = monthly.index.max()
        latest_count = int(monthly.loc[latest_month])
        # previous month
        try:
            prev_period = (pd.Period(latest_month, freq='M') - 1).strftime('%Y-%m')
            prev_count = int(monthly.get(prev_period, 0))
        except Exception:
            prev_count = 0

        month_pct = None
        if prev_count > 0:
            month_pct = (latest_count - prev_count) / prev_count

        # year-over-year for same month
        try:
            prev_year_period = (pd.Period(latest_month, freq='M') - 12).strftime('%Y-%m')
            yoy_count = int(monthly.get(prev_year_period, np.nan))
            yoy_pct = None
            if not np.isnan(yoy_count) and yoy_count > 0:
                yoy_pct = (latest_count - yoy_count) / yoy_count
        except Exception:
            yoy_count = np.nan
            yoy_pct = None

        # year totals
        by_year = g.groupby(g['datetime'].dt.to_period('Y')).size()
        if len(by_year) >= 2:
            last_year = str(by_year.index.max())
            prev_year = str(by_year.index.max() - 1)
            last_year_total = int(by_year.get(by_year.index.max(), 0))
            prev_year_total = int(by_year.get(by_year.index.max() - 1, 0))
            year_pct = None
            if prev_year_total > 0:
                year_pct = (last_year_total - prev_year_total) / prev_year_total
        else:
            last_year_total = int(by_year.sum())
            prev_year_total = 0
            year_pct = None

        # weekday distribution comparison: baseline 2022-2023 vs recent 2024+
        baseline = g[g['year'] <= 2023]
        recent = g[g['year'] >= 2024]
        wd_change_label = ''
        if not baseline.empty and not recent.empty:
            b_dist = baseline['weekday'].value_counts(normalize=True)
            r_dist = recent['weekday'].value_counts(normalize=True)
            # compute weekday with largest relative increase
            days = set(b_dist.index).union(set(r_dist.index))
            diffs = {d: r_dist.get(d,0) - b_dist.get(d,0) for d in days}
            best_day = max(diffs.items(), key=lambda x: x[1])
            wd_change_label = f"{best_day[0]}:+{best_day[1]:.3f}"

        # hour distribution change
        hr_change_label = ''
        if not baseline.empty and not recent.empty:
            b_h = baseline['hour'].value_counts(normalize=True)
            r_h = recent['hour'].value_counts(normalize=True)
            hrs = set(b_h.index).union(set(r_h.index))
            diffh = {h: r_h.get(h,0) - b_h.get(h,0) for h in hrs}
            best_hr = max(diffh.items(), key=lambda x: x[1])
            hr_change_label = f"{int(best_hr[0])}:+{best_hr[1]:.3f}"

        results.append({
            'cidade': cidade,
            'bairro': bairro,
            'latest_month': latest_month,
            'latest_count': latest_count,
            'prev_month_count': prev_count,
            'month_pct_change': month_pct if month_pct is not None else '',
            'yoy_count_same_month': int(yoy_count) if not np.isnan(yoy_count) else '',
            'yoy_pct_change': yoy_pct if yoy_pct is not None else '',
            'last_year_total': last_year_total,
            'prev_year_total': prev_year_total,
            'year_pct_change': year_pct if year_pct is not None else '',
            'weekday_shift': wd_change_label,
            'hour_shift': hr_change_label,
            'total_events': int(len(g))
        })

    return pd.DataFrame(results)


def write_summary(md_path, df_trends):
    lines = []
    lines.append('# CVLI Trends por Bairro / Cidade — resumo')
    lines.append('')
    if df_trends.empty:
        lines.append('Nenhum par com dados suficientes.')
    else:
        lines.append(f'- Pares analisados: {len(df_trends)}')
        # top increases month-to-month
        dfm = df_trends.copy()
        dfm['month_pct_change_num'] = pd.to_numeric(dfm['month_pct_change'], errors='coerce')
        top_inc = dfm.sort_values('month_pct_change_num', ascending=False).head(10)
        lines.append('')
        lines.append('## Top 10 aumentos mês-a-mês (último mês vs anterior)')
        lines.append('bairro / cidade | month_pct_change | latest_count | prev_month_count')
        lines.append('--- | ---: | ---: | ---:')
        for _, r in top_inc.iterrows():
            lines.append(f"{r['bairro']} / {r['cidade']} | {r['month_pct_change_num']:.2f} | {int(r['latest_count'])} | {int(r['prev_month_count'])}")

        lines.append('')
        lines.append('## Exemplos de mudanças por dia/hora (baseline 2022-2023 vs recent 2024+):')
        for _, r in df_trends.head(10).iterrows():
            lines.append(f"- {r['bairro']} / {r['cidade']}: weekday_shift={r['weekday_shift']}, hour_shift={r['hour_shift']}")

    Path(md_path).write_text('\n'.join(lines), encoding='utf-8')


def main():
    if not ASSIGNED.exists():
        print('Missing assigned JSON:', ASSIGNED)
        return
    records = load_wrapper(ASSIGNED)
    df = prepare_df(records)
    df_trends = analyze(df)
    df_trends.to_csv(OUT_CSV, index=False)
    write_summary(OUT_MD, df_trends)
    print('Wrote', OUT_CSV)
    print('Wrote', OUT_MD)


if __name__ == '__main__':
    main()
