import json
from pathlib import Path

import pandas as pd


OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

MONTHLY_CSV = OUT / 'sazonalidade_bairro_cidade_monthly.csv'
WEEKDAY_CSV = OUT / 'sazonalidade_bairro_cidade_weekday.csv'
HOURLY_CSV = OUT / 'sazonalidade_bairro_cidade_hourly.csv'
SUMMARY_JSON = OUT / 'sazonalidade_summary_bairro_cidade.json'

REPORT_MD = OUT / 'sazonalidade_report_bairro_cidade.md'
MATRIX_CSV = OUT / 'sazonalidade_matrix_bairro_cidade.csv'


def load_safe(p):
    if not p.exists():
        return None
    return pd.read_csv(p, dtype=str)


def main():
    monthly = load_safe(MONTHLY_CSV)
    weekday = load_safe(WEEKDAY_CSV)
    hourly = load_safe(HOURLY_CSV)
    summary = json.loads(SUMMARY_JSON.read_text(encoding='utf-8')) if SUMMARY_JSON.exists() else {}

    if monthly is None:
        print(f"Missing {MONTHLY_CSV}")
        return

    # normalize types
    monthly['count'] = pd.to_numeric(monthly['count'], errors='coerce').fillna(0).astype(int)
    monthly['year'] = monthly['year'].astype(int)
    monthly['month'] = monthly['month'].astype(int)

    # create year-month column
    monthly['year_month'] = monthly['year'].astype(str) + '-' + monthly['month'].astype(str).str.zfill(2)

    # aggregate total per bairro/cidade
    total_by_bairro = monthly.groupby(['cidade','bairro'])['count'].sum().reset_index()
    top10 = total_by_bairro.sort_values('count', ascending=False).head(10)

    # pivot to matrix (columns = year_month)
    matrix = monthly.pivot_table(index=['cidade','bairro'], columns='year_month', values='count', aggfunc='sum', fill_value=0)
    matrix.reset_index(inplace=True)
    matrix.to_csv(MATRIX_CSV, index=False)

    # write a simple markdown report
    md = []
    md.append('# Sazonalidade por Cidade e Bairro\n')
    md.append('**Resumo rápido**\n')
    md.append('\n')
    md.append(f'- Total CVLI (agrupados usados): {summary.get("total_cvli_rows", monthly["count"].sum())}')
    md.append(f'- Cidades distintas: {summary.get("distinct_cidades", monthly["cidade"].nunique())}')
    md.append(f'- Bairros distintos: {summary.get("distinct_bairros", monthly["bairro"].nunique())}')
    md.append('\n')
    md.append('**Top 10 bairros (por total de crimes)**\n')
    md.append('\n')
    md.append('| Rank | Cidade | Bairro | Total CVLI |')
    md.append('|---:|---|---|---:|')
    for i, row in enumerate(top10.itertuples(index=False), 1):
        md.append(f'| {i} | {row.cidade} | {row.bairro} | {int(row.count)} |')

    md.append('\n')
    md.append('**Arquivos gerados**\n')
    md.append('\n')
    md.append(f'- Matriz (bairros x meses): {MATRIX_CSV}')
    md.append(f'- Monthly CSV: {MONTHLY_CSV}')
    md.append(f'- Weekday CSV: {WEEKDAY_CSV}')
    md.append(f'- Hourly CSV: {HOURLY_CSV}')
    md.append(f'- Summary JSON: {SUMMARY_JSON}')

    REPORT_MD.write_text('\n'.join(md), encoding='utf-8')

    print('Wrote report and matrix:')
    print(MATRIX_CSV)
    print(REPORT_MD)


if __name__ == '__main__':
    main()
