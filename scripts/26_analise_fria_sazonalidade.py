import json
from pathlib import Path
from collections import Counter

import pandas as pd


OUT = Path("outputs")
DOCS = OUT / "docs"

MONTHLY = OUT / 'sazonalidade_bairro_cidade_monthly.csv'
WEEKDAY = OUT / 'sazonalidade_bairro_cidade_weekday.csv'
HOURLY = OUT / 'sazonalidade_bairro_cidade_hourly.csv'
MIDX = OUT / 'sazonalidade_bairro_cidade_monthly_index.csv'
WIDX = OUT / 'sazonalidade_bairro_cidade_weekday_index.csv'
HIDX = OUT / 'sazonalidade_bairro_cidade_hourly_index.csv'

OUT_MD = DOCS / 'cvli_seasonality_analysis_cold.md'


def read_safe(p):
    if not p.exists():
        return None
    return pd.read_csv(p)


def main():
    monthly = read_safe(MONTHLY)
    weekday = read_safe(WEEKDAY)
    hourly = read_safe(HOURLY)
    m_idx = read_safe(MIDX)
    w_idx = read_safe(WIDX)
    h_idx = read_safe(HIDX)

    if monthly is None or m_idx is None:
        print('Required CSVs missing')
        return

    lines = []
    lines.append('# Análise Fria de Sazonalidade CVLI')
    lines.append('')
    lines.append('Perguntas: (1) Existe mês com padrão constante? (2) Horário com padrão? (3) Dia com padrão? (4) Quais bairros têm sazonalidade como "estações"?')
    lines.append('')

    # 1) Mês com maior incidência (frequência de pico entre bairros)
    lines.append('## 1. Mês com maior incidência — Análise de picos')
    lines.append('')

    m_idx['month'] = m_idx['month'].astype(int)
    m_idx['index'] = pd.to_numeric(m_idx['index'], errors='coerce').fillna(0)

    # For each bairro, find month with highest index
    peak_months_per_bairro = (
        m_idx.loc[m_idx.groupby(['cidade','bairro'])['index'].idxmax()]
        [['cidade','bairro','month']].copy()
    )

    month_frequency = Counter(peak_months_per_bairro['month'].values)
    most_common_months = month_frequency.most_common(5)

    lines.append('**Meses que aparecem como picos mais frequentemente (top 5):**')
    lines.append('')

    for month, count in most_common_months:
        pct = count / len(peak_months_per_bairro) * 100
        lines.append(f'- Mês {int(month)}: {count} bairros ({pct:.1f}%) têm seu pico aqui')

    lines.append('')
    lines.append(f'**Conclusão:** Mês {int(most_common_months[0][0])} é o de maior incidência "padrão" ({most_common_months[0][1]} / {len(peak_months_per_bairro)} bairros = {most_common_months[0][1]/len(peak_months_per_bairro)*100:.1f}%)')
    lines.append('')

    # 2) Horário com padrão
    lines.append('## 2. Horário com maior incidência — Análise de picos')
    lines.append('')

    if h_idx is not None:
        h_idx['hour'] = pd.to_numeric(h_idx['hour'], errors='coerce').fillna(0).astype(int)
        h_idx['index'] = pd.to_numeric(h_idx['index'], errors='coerce').fillna(0)

        peak_hours = (
            h_idx.loc[h_idx.groupby(['cidade','bairro'])['index'].idxmax()]
            [['cidade','bairro','hour']].copy()
        )

        hour_frequency = Counter(peak_hours['hour'].values)
        most_common_hours = hour_frequency.most_common(5)

        lines.append('**Horários que aparecem como picos mais frequentemente (top 5):**')
        lines.append('')

        for hour, count in most_common_hours:
            pct = count / len(peak_hours) * 100
            lines.append(f'- Hora {int(hour)}: {count} bairros ({pct:.1f}%) têm seu pico aqui')

        lines.append('')
        lines.append(f'**Conclusão:** Hora {int(most_common_hours[0][0])} (horário de pico padrão) aparece em {most_common_hours[0][1]} / {len(peak_hours)} bairros = {most_common_hours[0][1]/len(peak_hours)*100:.1f}%')
    else:
        lines.append('(Dados de horário não disponíveis)')

    lines.append('')

    # 3) Dia da semana com padrão
    lines.append('## 3. Dia da semana com maior incidência — Análise de picos')
    lines.append('')

    if w_idx is not None:
        w_idx['index'] = pd.to_numeric(w_idx['index'], errors='coerce').fillna(0)

        peak_weekdays = (
            w_idx.loc[w_idx.groupby(['cidade','bairro'])['index'].idxmax()]
            [['cidade','bairro','weekday']].copy()
        )

        weekday_frequency = Counter(peak_weekdays['weekday'].values)
        most_common_weekdays = weekday_frequency.most_common(5)

        lines.append('**Dias que aparecem como picos mais frequentemente (top 5):**')
        lines.append('')

        for day, count in most_common_weekdays:
            pct = count / len(peak_weekdays) * 100
            lines.append(f'- {day}: {count} bairros ({pct:.1f}%) têm seu pico neste dia')

        lines.append('')
        lines.append(f'**Conclusão:** {most_common_weekdays[0][0]} é o padrão de pico mais comum ({most_common_weekdays[0][1]} / {len(peak_weekdays)} bairros = {most_common_weekdays[0][1]/len(peak_weekdays)*100:.1f}%)')
    else:
        lines.append('(Dados de dia da semana não disponíveis)')

    lines.append('')

    # 4) Bairros com sazonalidade constante ("estações")
    lines.append('## 4. Bairros com padrão sazonal consistente ("estações do ano")')
    lines.append('')

    # For each bairro, check if it has consistent behavior across months
    # We'll measure this by looking at the coefficient of variation in monthly counts
    monthly['count'] = pd.to_numeric(monthly['count'], errors='coerce').fillna(0)
    monthly['year'] = pd.to_numeric(monthly['year'], errors='coerce').fillna(0).astype(int)
    monthly['month'] = pd.to_numeric(monthly['month'], errors='coerce').fillna(0).astype(int)

    # Group by bairro and compute consistency
    consistency_scores = []
    for (cidade, bairro), group in monthly.groupby(['cidade','bairro']):
        # For each month, compute mean count across years
        month_means = group.groupby('month')['count'].agg(['mean', 'std', 'count']).reset_index()
        month_means = month_means[month_means['count'] >= 1]  # at least 1 year of data

        if len(month_means) > 0:
            # coefficient of variation: std / mean
            overall_mean = month_means['mean'].mean()
            if overall_mean > 0:
                monthly_cv = (month_means['std'].mean() / month_means['mean'].mean())
                consistency = 1.0 / (1.0 + monthly_cv)  # higher = more consistent
            else:
                consistency = 0

            consistency_scores.append({
                'cidade': cidade,
                'bairro': bairro,
                'consistency': consistency,
                'total_count': group['count'].sum()
            })

    consistency_df = pd.DataFrame(consistency_scores).sort_values('consistency', ascending=False)

    lines.append('**Bairros com maior consistência sazonal (padrão repetido ano a ano):**')
    lines.append('')

    top_consistent = consistency_df.head(15)
    for i, row in enumerate(top_consistent.itertuples(index=False), 1):
        label = f"{row.bairro} / {row.cidade}" if pd.notna(row.bairro) and row.bairro != '' else row.cidade
        lines.append(f'{i}. {label}: consistência={row.consistency:.2f}, total_cvli={int(row.total_count)}')

    lines.append('')
    lines.append('(Consistência próxima a 1.0 = padrão forte; próxima a 0 = aleatório)')
    lines.append('')

    # Write file
    OUT_MD.write_text('\n'.join(lines), encoding='utf-8')
    print('Wrote', OUT_MD)


if __name__ == '__main__':
    main()
