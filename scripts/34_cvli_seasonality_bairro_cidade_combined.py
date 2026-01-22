import pandas as pd
from pathlib import Path
from collections import Counter

OUT = Path('outputs')
DOCS = OUT / 'docs'
DOCS.mkdir(exist_ok=True)

MONTHLY = OUT / 'sazonalidade_bairro_cidade_monthly.csv'
WEEKDAY = OUT / 'sazonalidade_bairro_cidade_weekday.csv'
HOURLY = OUT / 'sazonalidade_bairro_cidade_hourly.csv'
MIDX = OUT / 'sazonalidade_bairro_cidade_monthly_index.csv'
WIDX = OUT / 'sazonalidade_bairro_cidade_weekday_index.csv'
HIDX = OUT / 'sazonalidade_bairro_cidade_hourly_index.csv'

OUT_MD = DOCS / 'cvli_seasonality_analysis_cold_bairro_cidade.md'


def read_csv_safe(p):
    if not p.exists():
        return None
    return pd.read_csv(p)


def main():
    monthly = read_csv_safe(MONTHLY)
    m_idx = read_csv_safe(MIDX)
    w_idx = read_csv_safe(WIDX)
    h_idx = read_csv_safe(HIDX)

    if monthly is None or m_idx is None:
        print('Missing required CSVs')
        return

    # ensure city and bairro columns
    monthly['cidade'] = monthly.get('cidade', pd.Series(['']*len(monthly))).fillna('').astype(str).str.strip()
    monthly['bairro'] = monthly.get('bairro', pd.Series(['']*len(monthly))).fillna('').astype(str).str.strip()

    m_idx['cidade'] = m_idx.get('cidade', pd.Series(['']*len(m_idx))).fillna('').astype(str).str.strip()
    m_idx['bairro'] = m_idx.get('bairro', pd.Series(['']*len(m_idx))).fillna('').astype(str).str.strip()

    # create combined key
    m_idx['bairro_cidade'] = m_idx['bairro'].astype(str) + ' / ' + m_idx['cidade'].astype(str)

    lines = []
    lines.append('# Análise Fria de Sazonalidade CVLI — Chave Bairro / Cidade')
    lines.append('')

    # 1) Peak month per pair
    m_idx['month'] = pd.to_numeric(m_idx['month'], errors='coerce').fillna(0).astype(int)
    m_idx['index'] = pd.to_numeric(m_idx['index'], errors='coerce').fillna(0)

    # find peak month per (cidade,bairro)
    peak_idx = m_idx.loc[m_idx.groupby(['cidade','bairro'])['index'].idxmax()][['cidade','bairro','month','index']].copy()
    peak_idx['pair'] = peak_idx['bairro'].astype(str) + ' / ' + peak_idx['cidade'].astype(str)

    month_freq = Counter(peak_idx['month'].values)
    most_common_months = month_freq.most_common(5)

    total_pairs = peak_idx['pair'].nunique()
    lines.append('## 1. Mês com maior incidência — por `bairro / cidade`')
    lines.append('')
    lines.append('**Meses que aparecem como picos mais frequentemente (top 5):**')
    lines.append('')
    for month, count in most_common_months:
        pct = count / total_pairs * 100
        lines.append(f'- Mês {int(month)}: {count} bairros/cidades ({pct:.1f}%) têm seu pico aqui')

    lines.append('')
    if most_common_months:
        lines.append(f'**Conclusão:** Mês {int(most_common_months[0][0])} é o de maior incidência padrão ({most_common_months[0][1]} / {total_pairs} = {most_common_months[0][1]/total_pairs*100:.1f}%)')
    lines.append('')

    # 2) Peak hour per pair
    lines.append('## 2. Horário com maior incidência — por `bairro / cidade`')
    lines.append('')
    if h_idx is not None:
        h_idx['cidade'] = h_idx.get('cidade', pd.Series(['']*len(h_idx))).fillna('').astype(str).str.strip()
        h_idx['bairro'] = h_idx.get('bairro', pd.Series(['']*len(h_idx))).fillna('').astype(str).str.strip()
        h_idx['hour'] = pd.to_numeric(h_idx['hour'], errors='coerce').fillna(0).astype(int)
        h_idx['index'] = pd.to_numeric(h_idx['index'], errors='coerce').fillna(0)
        peak_hours = h_idx.loc[h_idx.groupby(['cidade','bairro'])['index'].idxmax()].copy()
        peak_hours['pair'] = peak_hours['bairro'].astype(str) + ' / ' + peak_hours['cidade'].astype(str)
        hour_freq = Counter(peak_hours['hour'].values)
        most_common_hours = hour_freq.most_common(5)
        for hour, count in most_common_hours:
            pct = count / peak_hours['pair'].nunique() * 100
            lines.append(f'- Hora {int(hour)}: {count} bairros/cidades ({pct:.1f}%) têm seu pico aqui')
    else:
        lines.append('(Dados de horário não disponíveis)')

    lines.append('')
    # 3) Peak weekday per pair
    lines.append('## 3. Dia da semana com maior incidência — por `bairro / cidade`')
    lines.append('')
    if w_idx is not None:
        w_idx['cidade'] = w_idx.get('cidade', pd.Series(['']*len(w_idx))).fillna('').astype(str).str.strip()
        w_idx['bairro'] = w_idx.get('bairro', pd.Series(['']*len(w_idx))).fillna('').astype(str).str.strip()
        w_idx['index'] = pd.to_numeric(w_idx['index'], errors='coerce').fillna(0)
        peak_wd = w_idx.loc[w_idx.groupby(['cidade','bairro'])['index'].idxmax()].copy()
        peak_wd['pair'] = peak_wd['bairro'].astype(str) + ' / ' + peak_wd['cidade'].astype(str)
        wd_freq = Counter(peak_wd['weekday'].values)
        most_common_wd = wd_freq.most_common(5)
        for wd, count in most_common_wd:
            pct = count / peak_wd['pair'].nunique() * 100
            lines.append(f'- {wd}: {count} bairros/cidades ({pct:.1f}%) têm seu pico neste dia')
    else:
        lines.append('(Dados de dia da semana não disponíveis)')

    lines.append('')
    # 4) Consistency per pair
    lines.append('## 4. Bairros / Cidades com padrão sazonal consistente')
    lines.append('')
    monthly['count'] = pd.to_numeric(monthly['count'], errors='coerce').fillna(0)
    monthly['month'] = pd.to_numeric(monthly['month'], errors='coerce').fillna(0).astype(int)
    # compute consistency per (cidade,bairro)
    scores = []
    for (cidade, bairro), g in monthly.groupby(['cidade','bairro']):
        gm = g.groupby('month')['count'].mean()
        if len(gm) == 0:
            continue
        mean_all = gm.mean()
        std_all = gm.std()
        cv = std_all / mean_all if mean_all > 0 else 0
        consistency = 1.0 / (1.0 + cv)
        scores.append({'cidade': cidade, 'bairro': bairro, 'consistency': consistency, 'total': int(g['count'].sum())})

    if scores:
        dfc = pd.DataFrame(scores).sort_values('consistency', ascending=False)
        top = dfc.head(20)
        lines.append('**Top 20 bairros / cidades por consistência sazonal:**')
        lines.append('')
        for i, r in enumerate(top.itertuples(index=False), 1):
            lines.append(f'{i}. {r.bairro} / {r.cidade}: consistência={r.consistency:.2f}, total={r.total}')

    OUT_MD.write_text('\n'.join(lines), encoding='utf-8')
    print('Wrote', OUT_MD)


if __name__ == '__main__':
    main()
