import json
from pathlib import Path
import sys

import pandas as pd


INPUT = Path("data/raw/dados_status_ocorrencias_gerais_bairros_atribuidos.json")
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)


def load_data(path):
    with open(path, "r", encoding="utf-8") as f:
        wrapper = json.load(f)
    if isinstance(wrapper, dict) and "data" in wrapper:
        records = wrapper["data"]
    elif isinstance(wrapper, list):
        records = None
        for el in wrapper:
            if isinstance(el, dict) and "data" in el:
                records = el["data"]
                break
        if records is None:
            raise ValueError("Could not find 'data' array in wrapper list")
    else:
        raise ValueError("Unexpected JSON wrapper format")
    return pd.json_normalize(records)


def main():
    if not INPUT.exists():
        print(f"input not found: {INPUT}")
        sys.exit(1)

    df = load_data(INPUT)

    # Normalize columns
    # Ensure datetime
    if 'data' not in df.columns:
        print("missing 'data' column")
        sys.exit(1)

    if 'hora' not in df.columns:
        df['hora'] = None

    df['datetime'] = pd.to_datetime(df['data'].astype(str) + ' ' + df['hora'].fillna('00:00:00').astype(str), errors='coerce')

    # Filter CVLI records
    tipo_col = 'tipo' if 'tipo' in df.columns else None
    if tipo_col:
        mask = df[tipo_col].astype(str).str.lower() == 'cvli'
        df = df[mask].copy()

    df['year'] = df['datetime'].dt.year
    df['month'] = df['datetime'].dt.month
    df['month_name'] = df['datetime'].dt.month_name(locale='pt_BR')
    df['weekday'] = df['datetime'].dt.day_name(locale='pt_BR')
    df['hour'] = df['datetime'].dt.hour

    # fill missing bairro/cidade so grouping keys exist
    df['bairro'] = df.get('bairro', pd.Series(['']*len(df))).fillna('')
    df['cidade'] = df.get('cidade', pd.Series(['']*len(df))).fillna('')

    # Monthly counts per cidade/bairro
    monthly = (
        df.groupby(['cidade', 'bairro', 'year', 'month', 'month_name'])
        .size()
        .reset_index(name='count')
        .sort_values(['cidade','bairro','year','month'])
    )

    monthly_out = OUT_DIR / 'sazonalidade_bairro_cidade_monthly.csv'
    monthly.to_csv(monthly_out, index=False)

    # Weekday distribution
    weekday = (
        df.groupby(['cidade','bairro','weekday'])
        .size()
        .reset_index(name='count')
        .sort_values(['cidade','bairro','count'], ascending=[True, True, False])
    )
    weekday_out = OUT_DIR / 'sazonalidade_bairro_cidade_weekday.csv'
    weekday.to_csv(weekday_out, index=False)

    # Hourly distribution
    hourly = (
        df.groupby(['cidade','bairro','hour'])
        .size()
        .reset_index(name='count')
        .sort_values(['cidade','bairro','hour'])
    )
    hourly_out = OUT_DIR / 'sazonalidade_bairro_cidade_hourly.csv'
    hourly.to_csv(hourly_out, index=False)

    summary = {
        'total_cvli_rows': int(len(df)),
        'distinct_cidades': int(df['cidade'].nunique()),
        'distinct_bairros': int(df['bairro'].nunique()),
        'monthly_file': str(monthly_out),
        'weekday_file': str(weekday_out),
        'hourly_file': str(hourly_out),
    }

    (OUT_DIR / 'sazonalidade_summary_bairro_cidade.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
