import json
from pathlib import Path
import pandas as pd

INPUT = Path('data/raw/ocorrencia_policial_operacional.json')
OUT = Path('outputs')
OUT.mkdir(exist_ok=True)

PT_MONTHS = ['janeiro','fevereiro','março','abril','maio','junho','julho','agosto','setembro','outubro','novembro','dezembro']
PT_WEEKDAYS = ['segunda-feira','terça-feira','quarta-feira','quinta-feira','sexta-feira','sábado','domingo']


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        wrapper = json.load(f)
    # find data array
    if isinstance(wrapper, list):
        for item in wrapper:
            if isinstance(item, dict) and item.get('type') == 'table' and 'data' in item:
                return pd.json_normalize(item['data'])
    if isinstance(wrapper, dict) and 'data' in wrapper:
        return pd.json_normalize(wrapper['data'])
    raise ValueError('Could not find data array in JSON wrapper')


def main():
    if not INPUT.exists():
        print('input not found:', INPUT)
        return

    df = load_json(INPUT)
    # Ensure columns for date and time
    # Try common names
    date_col = None
    time_col = None
    for c in df.columns:
        lc = c.lower()
        if lc == 'data' or lc.startswith('data'):
            date_col = c
        if lc in ('horai','hora','horai') or lc.startswith('hora'):
            time_col = c
    if date_col is None:
        print('No date column found')
        return

    if time_col is None:
        df['hora'] = None
    else:
        df['hora'] = df[time_col]

    df['datetime'] = pd.to_datetime(df[date_col].astype(str) + ' ' + df['hora'].fillna('00:00:00').astype(str), errors='coerce')
    df = df[df['datetime'].notna()].copy()

    df['year'] = df['datetime'].dt.year
    df['month'] = df['datetime'].dt.month
    df['month_name'] = df['month'].apply(lambda m: PT_MONTHS[m-1] if pd.notna(m) and 1 <= m <= 12 else '')
    df['weekday'] = df['datetime'].dt.dayofweek.apply(lambda d: PT_WEEKDAYS[d] if pd.notna(d) else '')
    df['hour'] = df['datetime'].dt.hour

    # standardize bairro/cidade
    if 'BairroOcor' in df.columns:
        df['bairro'] = df['BairroOcor']
    elif 'bairro' in df.columns:
        df['bairro'] = df['bairro']
    else:
        df['bairro'] = ''

    if 'CidadeOcor' in df.columns:
        df['cidade'] = df['CidadeOcor']
    elif 'cidade' in df.columns:
        df['cidade'] = df['cidade']
    else:
        df['cidade'] = ''

    df['bairro'] = df['bairro'].fillna('').astype(str).str.strip()
    df['cidade'] = df['cidade'].fillna('').astype(str).str.strip()

    # Monthly counts per cidade/bairro
    monthly = (
        df.groupby(['cidade','bairro','year','month','month_name'])
        .size()
        .reset_index(name='count')
        .sort_values(['cidade','bairro','year','month'])
    )
    monthly.to_csv(OUT / 'sazonalidade_bairro_cidade_monthly.csv', index=False)

    # Weekday distribution
    weekday = (
        df.groupby(['cidade','bairro','weekday'])
        .size()
        .reset_index(name='count')
        .sort_values(['cidade','bairro','count'], ascending=[True, True, False])
    )
    weekday.to_csv(OUT / 'sazonalidade_bairro_cidade_weekday.csv', index=False)

    # Hourly distribution
    hourly = (
        df.groupby(['cidade','bairro','hour'])
        .size()
        .reset_index(name='count')
        .sort_values(['cidade','bairro','hour'])
    )
    hourly.to_csv(OUT / 'sazonalidade_bairro_cidade_hourly.csv', index=False)

    summary = {
        'total_rows': int(len(df)),
        'distinct_cidades': int(df['cidade'].nunique()),
        'distinct_bairros': int(df['bairro'].nunique()),
        'monthly_file': str(OUT / 'sazonalidade_bairro_cidade_monthly.csv'),
        'weekday_file': str(OUT / 'sazonalidade_bairro_cidade_weekday.csv'),
        'hourly_file': str(OUT / 'sazonalidade_bairro_cidade_hourly.csv'),
    }

    (OUT / 'sazonalidade_summary_prisoes.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
