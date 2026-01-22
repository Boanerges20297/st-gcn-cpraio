import json
from pathlib import Path

import pandas as pd


OUT = Path("outputs")
DOCS = OUT / "docs"
DOCS.mkdir(exist_ok=True)

MONTHLY = OUT / 'sazonalidade_bairro_cidade_monthly.csv'
WEEKDAY = OUT / 'sazonalidade_bairro_cidade_weekday.csv'
HOURLY = OUT / 'sazonalidade_bairro_cidade_hourly.csv'
MIDX = OUT / 'sazonalidade_bairro_cidade_monthly_index.csv'
WIDX = OUT / 'sazonalidade_bairro_cidade_weekday_index.csv'
HIDX = OUT / 'sazonalidade_bairro_cidade_hourly_index.csv'

OUT_MD = DOCS / 'cvli_seasonality_patterns.md'


def read_safe(p):
    if not p.exists():
        return None
    return pd.read_csv(p)


def top_overall(monthly_df, top_n=20):
    monthly_df['count'] = pd.to_numeric(monthly_df['count'], errors='coerce').fillna(0)
    tot = monthly_df.groupby(['cidade','bairro'])['count'].sum().reset_index(name='total')
    return tot.sort_values('total', ascending=False).head(top_n)


def format_row_label(cidade, bairro):
    return f"{bairro} / {cidade}" if pd.notna(bairro) and bairro != '' else f"{cidade}"


def main():
    monthly = read_safe(MONTHLY)
    weekday = read_safe(WEEKDAY)
    hourly = read_safe(HOURLY)
    m_idx = read_safe(MIDX)
    w_idx = read_safe(WIDX)
    h_idx = read_safe(HIDX)

    if monthly is None or m_idx is None:
        print('Required CSVs missing; run previous steps first')
        return

    top = top_overall(monthly, top_n=20)

    lines = []
    lines.append('# Padrões de Sazonalidade — CVLI (detalhado)')
    lines.append('Fonte: dados processados com bairros atribuídos (spatial join quando ausente).')
    lines.append('')
    lines.append('## Sumário executivo')
    lines.append('')

    # overall month-level trend across all bairros (mean index per month)
    m_idx['month'] = m_idx['month'].astype(int)
    month_overall = m_idx.groupby('month')['index'].mean().reset_index()
    peak_month = int(month_overall.sort_values('index', ascending=False).iloc[0]['month'])
    lines.append(f'- Mês com maior índice médio (across bairros): {peak_month}')

    # weekday overall
    if w_idx is not None:
        # try to order weekdays by average index
        week_overall = w_idx.groupby('weekday')['index'].mean().reset_index()
        top_week = week_overall.sort_values('index', ascending=False).iloc[0]['weekday']
        lines.append(f'- Dia da semana com maior índice médio: {top_week}')

    lines.append('')
    lines.append('## Top 20 bairros por total de CVLI (detalhes de padrão)')
    lines.append('')

    for _, row in top.iterrows():
        cidade = row['cidade']
        bairro = row['bairro']
        total = int(row['total'])
        label = format_row_label(cidade, bairro)

        lines.append(f'### {label} — total CVLI: {total}')

        # monthly peaks (use m_idx filtered)
        sel_m = m_idx[(m_idx['cidade'] == cidade) & (m_idx['bairro'] == bairro)].copy()
        if not sel_m.empty:
            sel_m = sel_m.sort_values('index', ascending=False)
            top_months = sel_m.head(3)
            lines.append('- Meses com maior índice (index, proporção):')
            for _, mm in top_months.iterrows():
                lines.append(f"  - Mês {int(mm['month'])}: index={mm['index']:.2f}, prop={mm.get('proportion',0):.2f}")
        else:
            lines.append('- Meses: sem dados suficientes')

        # weekday
        if w_idx is not None:
            sel_w = w_idx[(w_idx['cidade'] == cidade) & (w_idx['bairro'] == bairro)].copy()
            if not sel_w.empty:
                top_w = sel_w.sort_values('index', ascending=False).head(2)
                lines.append('- Dias da semana com maior índice:')
                for _, ww in top_w.iterrows():
                    lines.append(f"  - {ww['weekday']}: index={ww['index']:.2f}, prop={ww.get('proportion',0):.2f}")
            else:
                lines.append('- Dias da semana: sem dados suficientes')

        # hourly
        sel_h = h_idx[(h_idx['cidade'] == cidade) & (h_idx['bairro'] == bairro)].copy() if h_idx is not None else pd.DataFrame()
        if not sel_h.empty:
            top_h = sel_h.sort_values('index', ascending=False).head(3)
            lines.append('- Horários com maior índice:')
            for _, hh in top_h.iterrows():
                lines.append(f"  - Hora {int(hh['hour'])}: index={hh['index']:.2f}, prop={hh.get('proportion',0):.2f}")
        else:
            lines.append('- Horários: sem dados suficientes')

        # small separator
        lines.append('')

    # write file
    OUT_MD.write_text('\n'.join(lines), encoding='utf-8')
    print('Wrote', OUT_MD)


if __name__ == '__main__':
    main()
