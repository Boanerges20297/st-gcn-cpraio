"""Analyze seasonality of CVLI occurrences (total over time).

Produces:
 - outputs/cvli_timeseries_daily.csv
 - outputs/cvli_monthly_summary.csv
 - outputs/cvli_dow_summary.csv
 - outputs/cvli_autocorr.csv
 - outputs/docs/cvli_seasonality.md
"""
import os
import json
from pathlib import Path
import pandas as pd
import numpy as np


def load_table(path):
    with open(path, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    for item in raw:
        if isinstance(item, dict) and item.get('type') == 'table' and 'data' in item:
            return pd.DataFrame(item['data'])
    return pd.DataFrame(raw)


def is_cvli(nat: str):
    if not isinstance(nat, str):
        return False
    s = nat.lower()
    keys = ['homicid', 'latroc', 'tentativa de homicid', 'intervenção policial letal', 'intervencao policial letal']
    return any(k in s for k in keys)


def safe_day_name(dt):
    try:
        return dt.day_name()
    except Exception:
        return str(dt.weekday())


def main():
    os.makedirs('outputs', exist_ok=True)
    os.makedirs('outputs/docs', exist_ok=True)

    df = load_table('data/raw/ocorrencia_policial_operacional.json')
    df['Data'] = pd.to_datetime(df['Data'], errors='coerce')
    df = df[df['Data'].notna()].copy()

    # Prefer structured `tipo` column if present
    if 'tipo' in df.columns:
        df['tipo'] = df['tipo'].astype(str).str.upper()
        df_cvli = df[df['tipo'] == 'CVLI'].copy()
        df_cvli['is_cvli'] = 1
    else:
        df['is_cvli'] = df['Natureza'].apply(is_cvli)
        df_cvli = df[df['is_cvli']].copy()
        df_cvli['is_cvli'] = 1
    total_cvli = int(df_cvli['is_cvli'].sum())

    # enforce analysis window 2022-01-01 .. 2026-12-31
    start_period = pd.to_datetime('2022-01-01')
    end_period = pd.to_datetime('2026-12-31')
    df_cvli = df_cvli[(df_cvli['Data'] >= start_period) & (df_cvli['Data'] <= end_period)].copy()

    # daily time series (total CVLI per day) over requested period
    full_index = pd.date_range(start_period, end_period, freq='D')
    ts = df_cvli.groupby(df_cvli['Data'].dt.normalize())['is_cvli'].sum().reindex(full_index, fill_value=0).rename('cvli_count').reset_index().rename(columns={'index':'Data'})
    ts.to_csv('outputs/cvli_timeseries_daily.csv', index=False)

    # monthly summary (over CVLI subset)
    df_cvli['month'] = df_cvli['Data'].dt.to_period('M')
    monthly = df_cvli.groupby('month')['is_cvli'].sum().rename('cvli_monthly_total').reset_index()
    monthly['month'] = monthly['month'].astype(str)
    monthly.to_csv('outputs/cvli_monthly_summary.csv', index=False)

    # per-bairro monthly totals and stats (if bairro present)
    if 'BairroOcor' in df_cvli.columns:
        df_cvli['bairro_up'] = df_cvli['BairroOcor'].astype(str).str.upper()
        by_bairro_month = df_cvli.groupby([df_cvli['bairro_up'], df_cvli['month']])['is_cvli'].sum().rename('cvli_monthly').reset_index()
        by_bairro_month.to_csv('outputs/cvli_by_bairro_monthly.csv', index=False)

        stats = by_bairro_month.groupby('bairro_up')['cvli_monthly'].agg(['mean','std','max']).reset_index().rename(columns={'mean':'mean_monthly','std':'std_monthly','max':'peak_monthly'})
        stats['cv'] = stats['std_monthly'] / stats['mean_monthly'].replace(0, np.nan)
        stats = stats.fillna(0).sort_values('mean_monthly', ascending=False)
        stats.to_csv('outputs/cvli_bairro_stats.csv', index=False)
    else:
        by_bairro_month = None

    # day-of-week summary
    df_cvli['dow'] = df_cvli['Data'].dt.day_name()
    dow_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    dow = df_cvli.groupby('dow')['is_cvli'].sum().reindex(dow_order).fillna(0).rename('cvli_dow_total').reset_index()
    dow.to_csv('outputs/cvli_dow_summary.csv', index=False)

    # basic seasonality: month-of-year average
    df_cvli['month_of_year'] = df_cvli['Data'].dt.month
    moy = df_cvli.groupby('month_of_year')['is_cvli'].mean().rename('avg_cvli_by_month').reset_index()
    moy.to_csv('outputs/cvli_avg_by_month.csv', index=False)

    # autocorrelation up to 60 lags
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
    acf_df.to_csv('outputs/cvli_autocorr.csv', index=False)

    # summary stats
    start = start_period.date().isoformat()
    end = end_period.date().isoformat()
    days = (end_period - start_period).days + 1
    avg_per_day = daily.mean() if n>0 else 0.0
    peak_idx = int(np.argmax(daily)) if n>0 else None
    peak_date = ts_idx[peak_idx].date().isoformat() if peak_idx is not None else ''
    peak_value = int(daily[peak_idx]) if peak_idx is not None else 0

    # write markdown report (Português)
    md_path = Path('outputs/docs/cvli_seasonality.md')
    with md_path.open('w', encoding='utf-8') as f:
        f.write('# Análise de Sazonalidade de CVLI\n\n')
        f.write('Escopo: contagem total de ocorrências ao longo do tempo, filtradas por crimes do tipo CVLI (homicídios, latrocínios, tentativas).\n\n')
        f.write('Fonte dos dados: `data/raw/ocorrencia_policial_operacional.json`\n\n')
        f.write('## Parâmetros e filtro\n')
        f.write('- Palavras-chave usadas para filtrar CVLI: `homicid`, `latroc`, `tentativa de homicid`, `intervenção policial letal` (busca por substring, sem diferenciação de maiúsculas/minúsculas)\n')
        f.write('- Agregações calculadas: diário / mensal / dia da semana / autocorrelação (lags 0..60)\n')
        f.write(f'- Período analisado: {start} até {end} ({days} dias)\n')
        f.write(f'- Total de ocorrências CVLI encontradas: {total_cvli}\n\n')

        f.write('## Principais resultados\n')
        f.write(f'- Média diária de CVLI: {avg_per_day:.3f}\n')
        f.write(f'- Dia com pico de CVLI: {peak_date} com {peak_value} ocorrências\n')
        f.write('\n## Totais mensais\n')
        f.write('Consulte `outputs/cvli_monthly_summary.csv` para os totais mensais detalhados.\n\n')

        f.write('## Totais por dia da semana\n')
        f.write('Consulte `outputs/cvli_dow_summary.csv` para os totais por dia da semana.\n\n')

        f.write('## Sazonalidade por bairro\n')
        if by_bairro_month is not None:
            f.write('Foram gerados os arquivos `outputs/cvli_by_bairro_monthly.csv` (contagem por mês e bairro) e `outputs/cvli_bairro_stats.csv` (média, desvio, CV por bairro).\n')
            # include top-5 bairros
            top5 = pd.read_csv('outputs/cvli_bairro_stats.csv').head(5)
            f.write('\nTop 5 bairros por média mensal de CVLI:\n')
            for _, r in top5.iterrows():
                f.write(f"- {r['bairro_up']}: média mensal={r['mean_monthly']:.2f}, pico mensal={int(r['peak_monthly'])}, CV={r['cv']:.2f}\n")
        else:
            f.write('Campo `BairroOcor` não disponível; não foi possível gerar análise por bairro.\n')

        f.write('\n## Autocorrelação\n')
        f.write('Consulte `outputs/cvli_autocorr.csv` para a função de autocorrelação (lags até 60 dias).\n\n')

        f.write('## Interpretação prática e próximos passos\n')
        f.write('- Se houver padrões mensais ou por dia da semana relevantes, adicionar indicadores sazonais (dummies de mês, dia da semana) aos modelos.\n')
        f.write('- Se a autocorrelação em certos lags for alta, aumentar a janela temporal ou adicionar alvos defasados como features.\n')
        f.write('- Se as contagens de CVLI forem esparsas, considerar agregações por bairro/mês ou usar modelos de contagem (Poisson/Negativa Binomial) para estabilizar o sinal.\n')

    print('CVLI seasonality analysis complete. Outputs written to outputs/ and report at', md_path)


if __name__ == '__main__':
    main()
