import json
from pathlib import Path
from collections import Counter

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

OUT_MD = DOCS / 'cvli_seasonality_analysis_cold_min10cvli.md'
OUT_CSV = DOCS / 'cvli_bairros_volume_analysis_min10.csv'


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

    # Filter: only bairros with >= 10 CVLI total
    monthly['count'] = pd.to_numeric(monthly['count'], errors='coerce').fillna(0)
    total_per_bairro = monthly.groupby(['cidade','bairro'])['count'].sum().reset_index(name='total')
    valid_bairros = total_per_bairro[total_per_bairro['total'] >= 10][['cidade','bairro']]
    
    # Filter all data
    m_idx_filtered = m_idx.merge(valid_bairros, on=['cidade','bairro'], how='inner')
    w_idx_filtered = w_idx.merge(valid_bairros, on=['cidade','bairro'], how='inner') if w_idx is not None else None
    h_idx_filtered = h_idx.merge(valid_bairros, on=['cidade','bairro'], how='inner') if h_idx is not None else None

    lines = []
    lines.append('# Análise Fria de Sazonalidade CVLI — Volume Mínimo 10')
    lines.append('')
    lines.append('**Filtro aplicado:** apenas bairros com ≥10 CVLI de histórico (padrões mais robustos)')
    lines.append('')
    lines.append(f'Bairros incluídos: {len(valid_bairros)} / {len(total_per_bairro)} ({len(valid_bairros)/len(total_per_bairro)*100:.1f}%)')
    lines.append(f'CVLI no dataset filtrado: {int(m_idx_filtered["mean_count"].sum())} de {int(monthly["count"].sum())} ({int(m_idx_filtered["mean_count"].sum())/int(monthly["count"].sum())*100:.1f}%)')
    lines.append('')

    # 1) Mês com maior incidência
    lines.append('## 1. Mês com maior incidência — Análise de picos (volume ≥10)')
    lines.append('')

    m_idx_filtered['month'] = m_idx_filtered['month'].astype(int)
    m_idx_filtered['index'] = pd.to_numeric(m_idx_filtered['index'], errors='coerce').fillna(0)

    peak_months_per_bairro = (
        m_idx_filtered.loc[m_idx_filtered.groupby(['cidade','bairro'])['index'].idxmax()]
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
    lines.append('## 2. Horário com maior incidência — Análise de picos (volume ≥10)')
    lines.append('')

    if h_idx_filtered is not None and len(h_idx_filtered) > 0:
        h_idx_filtered['hour'] = pd.to_numeric(h_idx_filtered['hour'], errors='coerce').fillna(0).astype(int)
        h_idx_filtered['index'] = pd.to_numeric(h_idx_filtered['index'], errors='coerce').fillna(0)

        peak_hours = (
            h_idx_filtered.loc[h_idx_filtered.groupby(['cidade','bairro'])['index'].idxmax()]
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
    lines.append('## 3. Dia da semana com maior incidência — Análise de picos (volume ≥10)')
    lines.append('')

    if w_idx_filtered is not None and len(w_idx_filtered) > 0:
        w_idx_filtered['index'] = pd.to_numeric(w_idx_filtered['index'], errors='coerce').fillna(0)

        peak_weekdays = (
            w_idx_filtered.loc[w_idx_filtered.groupby(['cidade','bairro'])['index'].idxmax()]
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

    # 4) Bairros com sazonalidade constante
    lines.append('## 4. Bairros com padrão sazonal consistente ("estações do ano") — volume ≥10')
    lines.append('')

    # Consistency analysis
    consistency_scores = []
    for (cidade, bairro), group in m_idx_filtered.groupby(['cidade','bairro']):
        group = group.copy()
        group['month'] = group['month'].astype(int)
        group['mean_count'] = pd.to_numeric(group['mean_count'], errors='coerce').fillna(0)
        
        if len(group) > 0:
            overall_mean = group['mean_count'].mean()
            if overall_mean > 0:
                std = group['mean_count'].std()
                cv = std / overall_mean if overall_mean > 0 else 0
                consistency = 1.0 / (1.0 + cv)
            else:
                consistency = 0

            consistency_scores.append({
                'cidade': cidade,
                'bairro': bairro,
                'consistency': consistency,
                'total_cvli': int(group['mean_count'].sum())
            })

    if consistency_scores:
        consistency_df = pd.DataFrame(consistency_scores).sort_values('consistency', ascending=False)
        consistency_df.to_csv(OUT_CSV, index=False)

        lines.append('**Bairros com maior consistência sazonal (padrão repetido ano a ano) — volume ≥10:**')
        lines.append('')

        top_consistent = consistency_df.head(20)
        for i, row in enumerate(top_consistent.itertuples(index=False), 1):
            label = f"{row.bairro} / {row.cidade}" if pd.notna(row.bairro) and row.bairro != '' else row.cidade
            lines.append(f'{i}. {label}: consistência={row.consistency:.2f}, total_cvli={int(row.total_cvli)}')

        lines.append('')
        lines.append('(Consistência próxima a 1.0 = padrão muito forte; próxima a 0 = padrão fraco)')

    lines.append('')
    lines.append('---')
    lines.append(f'Sazonalidade volume filtrado salvo em: {OUT_CSV}')

    OUT_MD.write_text('\n'.join(lines), encoding='utf-8')
    print('Wrote', OUT_MD)
    print('Wrote', OUT_CSV)


if __name__ == '__main__':
    main()
