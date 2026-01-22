import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy import stats

OUT = Path('outputs')
DOCS = OUT / 'docs'
DOCS.mkdir(exist_ok=True)

PROCEDURES = OUT / 'procedures_monthly_by_bairro.csv'
CVLI_CITY = OUT / 'cvli_by_bairro_cidade_monthly.csv'
CVLI_BAIRRO = OUT / 'cvli_by_bairro_monthly.csv'

OUT_CSV = OUT / 'effectiveness_prisoes_cvli_by_bairro_cidade.csv'
OUT_SUM = DOCS / 'efetividade_prisoes_cvli.md'


def load_procedures():
    df = pd.read_csv(PROCEDURES)
    # normalize
    if 'year_month' not in df.columns and 'year_month' in df.columns:
        pass
    # ensure year_month
    if 'year_month' not in df.columns:
        if 'date' in df.columns:
            df['year_month'] = pd.to_datetime(df['date'], errors='coerce').dt.to_period('M').astype(str)
        else:
            # try year and month
            if 'year' in df.columns and 'month' in df.columns:
                df['year_month'] = df['year'].astype(str) + '-' + df['month'].astype(str).str.zfill(2)
    df['bairro'] = df['bairro'].astype(str).str.upper().str.strip()
    df['cidade'] = df.get('cidade', pd.Series(['']*len(df))).astype(str).str.upper().str.strip()
    df['procedimentos'] = pd.to_numeric(df.get('procedimentos', 0), errors='coerce').fillna(0).astype(int)
    return df


def load_cvli():
    # prefer city-level file if exists
    if CVLI_CITY.exists():
        df = pd.read_csv(CVLI_CITY)
        # expected columns: can be 'bairro'+'cidade' separate or a combined 'bairro_cidade'
        if 'month' in df.columns and 'cvli_monthly' in df.columns:
            df['year_month'] = df['month'].astype(str)
        # handle combined field
        if 'bairro_cidade' in df.columns:
            parts = df['bairro_cidade'].astype(str).str.split('/')
            df['bairro'] = parts.map(lambda x: x[0].strip().upper() if len(x) > 0 else '')
            df['cidade'] = parts.map(lambda x: x[1].strip().upper() if len(x) > 1 else '')
        else:
            if 'bairro' in df.columns:
                df['bairro'] = df['bairro'].astype(str).str.upper().str.strip()
            if 'cidade' in df.columns:
                df['cidade'] = df['cidade'].astype(str).str.upper().str.strip()
        df['cvli_monthly'] = pd.to_numeric(df['cvli_monthly'], errors='coerce').fillna(0).astype(int)
        return df
    else:
        df = pd.read_csv(CVLI_BAIRRO)
        # columns: bairro_up, month (YYYY-MM), cvli_monthly
        if 'bairro_up' in df.columns:
            df['bairro'] = df['bairro_up'].astype(str).str.upper().str.strip()
        if 'month' in df.columns:
            df['year_month'] = df['month'].astype(str)
        df['cidade'] = df.get('cidade', pd.Series(['']*len(df))).astype(str).str.upper().str.strip()
        df['cvli_monthly'] = pd.to_numeric(df.get('cvli_monthly', 0), errors='coerce').fillna(0).astype(int)
        return df


def period_shifted_value(series, year_month_index, offset):
    # year_month_index is list of period strings like '2025-03'
    # build mapping
    ser = pd.Series(series.values, index=pd.PeriodIndex(year_month_index, freq='M'))
    def get(val):
        try:
            p = pd.Period(val, freq='M') + offset
            return ser.get(p, np.nan)
        except Exception:
            return np.nan
    return get


def analyze_effect(df_proc, df_cvli, start_period='2022-01'):
    # filter period
    df_proc = df_proc.copy()
    df_cvli = df_cvli.copy()
    df_proc['year_month'] = df_proc['year_month'].astype(str)
    df_cvli['year_month'] = df_cvli['year_month'].astype(str)

    df_proc = df_proc[df_proc['year_month'] >= start_period]
    df_cvli = df_cvli[df_cvli['year_month'] >= start_period]

    results = []

    # group by cidade,bairro
    group_keys = ['cidade','bairro']
    # ensure keys exist
    for k in group_keys:
        if k not in df_proc.columns:
            df_proc[k] = ''
        if k not in df_cvli.columns:
            df_cvli[k] = ''

    merged = None

    # We'll iterate over unique pairs from procedures
    pairs = df_proc.groupby(group_keys)
    for (cidade, bairro), g in pairs:
        proc_series = g.set_index('year_month')['procedimentos'].sort_index()
        # get cvli series for same pair
        # If procedures data lacks city info (empty), aggregate CVLI across cities matching the same bairro name
        if cidade == '' or pd.isna(cidade):
            cvli_tmp = df_cvli[df_cvli['bairro'] == bairro]
            if cvli_tmp.empty:
                continue
            cvli_g = cvli_tmp.groupby('year_month')['cvli_monthly'].sum().sort_index()
        else:
            cvli_g = df_cvli[(df_cvli['cidade'] == cidade) & (df_cvli['bairro'] == bairro)].set_index('year_month')['cvli_monthly'].sort_index()
        if cvli_g.empty:
            continue
        # compute threshold for high-procedures months (top quartile)
        thresh = proc_series.quantile(0.75)
        high_months = proc_series[proc_series >= thresh]
        if len(high_months) == 0:
            continue

        reductions = []
        prev_vals = []
        next_vals = []
        months = []
        for ym, val in high_months.items():
            try:
                p = pd.Period(ym, freq='M')
            except Exception:
                continue
            prev = cvli_g.get(str(p - 1), np.nan)
            nxt = cvli_g.get(str(p + 1), np.nan)
            if np.isnan(prev) or np.isnan(nxt):
                continue
            reductions.append(prev - nxt)
            prev_vals.append(prev)
            next_vals.append(nxt)
            months.append(str(p))

        if len(reductions) == 0:
            continue

        mean_red = float(np.nanmean(reductions))
        median_red = float(np.nanmedian(reductions))
        n = len(reductions)
        prop_positive = float(np.mean(np.array(reductions) > 0))
        # paired t-test prev vs next
        t_stat, p_val = (np.nan, np.nan)
        try:
            if n >= 2:
                t_stat, p_val = stats.ttest_rel(prev_vals, next_vals, nan_policy='omit')
        except Exception:
            t_stat, p_val = (np.nan, np.nan)

        results.append({
            'cidade': cidade,
            'bairro': bairro,
            'n_events': n,
            'mean_reduction_prev_minus_next': mean_red,
            'median_reduction': median_red,
            'prop_reduction_positive': prop_positive,
            't_stat': float(t_stat) if not pd.isna(t_stat) else None,
            'p_value': float(p_val) if not pd.isna(p_val) else None,
            'high_months_example': ','.join(months[:5])
        })

    return pd.DataFrame(results)


def summarize_and_append(md_path, df_res):
    lines = []
    lines.append('# Efetividade de Prisões sobre CVLI (desde 2022)')
    lines.append('')
    if df_res.empty:
        lines.append('Nenhum par `bairro / cidade` com dados suficientes para análise.')
    else:
        total_pairs = len(df_res)
        avg_mean_red = df_res['mean_reduction_prev_minus_next'].mean()
        prop_significant = (df_res['p_value'] < 0.05).sum() if 'p_value' in df_res.columns else 0
        lines.append(f'- Pares analisados (bairro/cidade): {total_pairs}')
        lines.append(f'- Redução média de CVLI (prev - next) nos meses de alto procedimento: {avg_mean_red:.3f} CVLI')
        lines.append(f'- Pares com p-value < 0.05: {prop_significant}')
        lines.append('')
        lines.append('## Top 20 bairros/cidades com maior redução média')
        lines.append('bairro / cidade | mean_reduction | n_events | p_value')
        lines.append('--- | ---: | ---: | ---:')
        top = df_res.sort_values('mean_reduction_prev_minus_next', ascending=False).head(20)
        for _, r in top.iterrows():
            lines.append(f"{r['bairro']} / {r['cidade']} | {r['mean_reduction_prev_minus_next']:.2f} | {int(r['n_events'])} | {r['p_value']:.4f}" )

        lines.append('')
        lines.append('## Top 20 bairros/cidades com maior aumento (efeito oposto)')
        lines.append('bairro / cidade | mean_reduction (negative means increase) | n_events | p_value')
        lines.append('--- | ---: | ---: | ---:')
        bot = df_res.sort_values('mean_reduction_prev_minus_next', ascending=True).head(20)
        for _, r in bot.iterrows():
            lines.append(f"{r['bairro']} / {r['cidade']} | {r['mean_reduction_prev_minus_next']:.2f} | {int(r['n_events'])} | {r['p_value']:.4f}" )

    # append to md file
    with open(md_path, 'a', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def main():
    if not PROCEDURES.exists():
        print('Missing procedures file:', PROCEDURES)
        return
    if not (CVLI_CITY.exists() or CVLI_BAIRRO.exists()):
        print('Missing CVLI monthly file')
        return

    df_proc = load_procedures()
    df_cvli = load_cvli()

    df_res = analyze_effect(df_proc, df_cvli, start_period='2022-01')
    df_res.to_csv(OUT_CSV, index=False)
    print('Wrote', OUT_CSV)

    summarize_and_append(OUT_SUM, df_res)
    print('Appended summary to', OUT_SUM)


if __name__ == '__main__':
    main()
