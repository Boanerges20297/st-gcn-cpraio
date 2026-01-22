import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr

OUT = Path('outputs')
OUT.mkdir(exist_ok=True)

PRISOES_MONTHLY_IDX = OUT / 'sazonalidade_bairro_cidade_monthly_index.csv'
CVLI_MONTHLY = OUT / 'cvli_by_bairro_monthly.csv'

OUT_CSV = OUT / 'sazonalidade_comparacao_cvli_prisoes.csv'
OUT_MD = OUT / 'sazonalidade_comparacao_summary.md'


def load_prisoes_monthly():
    df = pd.read_csv(PRISOES_MONTHLY_IDX)
    # pivot to (cidade,bairro) x month
    df['bairro_up'] = df['bairro'].astype(str).str.upper().str.strip()
    df['month'] = pd.to_numeric(df['month'], errors='coerce').fillna(0).astype(int)
    pivot = df.pivot_table(index=['bairro_up'], columns='month', values='index', aggfunc='mean', fill_value=0)
    return pivot


def load_cvli_monthly_index():
    # cvli_by_bairro_monthly.csv has bairro_up,month,cvli_monthly
    df = pd.read_csv(CVLI_MONTHLY)
    if 'bairro_up' not in df.columns:
        df['bairro_up'] = df['bairro'].astype(str).str.upper().str.strip()
    else:
        df['bairro_up'] = df['bairro_up'].astype(str).str.upper().str.strip()
    # month may be YYYY-MM; extract month number
    def month_num(m):
        try:
            if isinstance(m, str) and '-' in m:
                return int(m.split('-')[1])
            return int(m)
        except Exception:
            return 0
    df['month_num'] = df['month'].apply(month_num)

    # compute mean across years for each bairro-month
    avg = df.groupby(['bairro_up','month_num'])['cvli_monthly'].mean().reset_index()
    # pivot
    pivot = avg.pivot_table(index='bairro_up', columns='month_num', values='cvli_monthly', fill_value=0)

    # convert to index similar to prisoes: divide month mean by overall mean per bairro
    pivot_idx = pivot.copy()
    for idx in pivot_idx.index:
        row = pivot_idx.loc[idx]
        mean_overall = row.mean()
        if mean_overall == 0:
            pivot_idx.loc[idx] = 0.0
        else:
            pivot_idx.loc[idx] = row / mean_overall
    return pivot_idx


def compare(prisoes, cvli):
    rows = []
    shared = set(prisoes.index).intersection(set(cvli.index))
    for b in sorted(shared):
        p = prisoes.loc[b].reindex(range(1,13), fill_value=0).values.astype(float)
        c = cvli.loc[b].reindex(range(1,13), fill_value=0).values.astype(float)
        # compute correlation if variance>0
        try:
            if np.nanstd(p) == 0 or np.nanstd(c) == 0:
                r = np.nan
            else:
                r, _ = pearsonr(p, c)
        except Exception:
            r = np.nan
        diff = np.abs(p - c)
        rows.append({
            'bairro_up': b,
            'pearson_r': float(r) if not pd.isna(r) else None,
            'mean_abs_diff': float(np.nanmean(diff)),
            'max_abs_diff': float(np.nanmax(diff)),
            'months_prisoes_nonzero': int((p != 0).sum()),
            'months_cvli_nonzero': int((c != 0).sum()),
        })
    return pd.DataFrame(rows)


def main():
    if not PRISOES_MONTHLY_IDX.exists():
        print('Missing prisoes monthly index:', PRISOES_MONTHLY_IDX)
        return
    if not CVLI_MONTHLY.exists():
        print('Missing cvli monthly file:', CVLI_MONTHLY)
        return

    prisoes = load_prisoes_monthly()
    cvli = load_cvli_monthly_index()

    comp = compare(prisoes, cvli)
    comp = comp.sort_values('pearson_r', ascending=False)
    comp.to_csv(OUT_CSV, index=False)

    # write brief markdown summary
    top_pos = comp.head(20)
    top_neg = comp.dropna(subset=['pearson_r']).sort_values('pearson_r').head(20)

    lines = []
    lines.append('# Comparação sazonalidade CVLI vs PRISÕES')
    lines.append('')
    lines.append(f'- Total bairros comparados: {len(comp)}')
    lines.append('')
    lines.append('## Top 20 correlações positivas')
    lines.append('bairro | pearson_r | mean_abs_diff | max_abs_diff')
    lines.append('--- | ---: | ---: | ---:')
    for _, r in top_pos.iterrows():
        lines.append(f"{r['bairro_up']} | {r['pearson_r']:.3f} | {r['mean_abs_diff']:.3f} | {r['max_abs_diff']:.3f}")

    lines.append('')
    lines.append('## Top 20 correlações negativas')
    lines.append('bairro | pearson_r | mean_abs_diff | max_abs_diff')
    lines.append('--- | ---: | ---: | ---:')
    for _, r in top_neg.iterrows():
        lines.append(f"{r['bairro_up']} | {r['pearson_r']:.3f} | {r['mean_abs_diff']:.3f} | {r['max_abs_diff']:.3f}")

    OUT_MD.write_text('\n'.join(lines), encoding='utf-8')
    print('Wrote comparison CSV and summary:')
    print(OUT_CSV)
    print(OUT_MD)


if __name__ == '__main__':
    main()
