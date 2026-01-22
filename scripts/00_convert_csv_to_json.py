import pandas as pd
import json
from pathlib import Path

CSV = Path('data/raw/View_Ocorrencias_Operacionais_Modelo.csv')
OUT = Path('data/raw/ocorrencia_policial_operacional.json')

if not CSV.exists():
    print('CSV not found:', CSV)
    raise SystemExit(1)

# Read with pandas (handle quoted headers)
df = pd.read_csv(CSV, dtype=str)
# Replace NaN with None
records = df.where(pd.notnull(df), None).to_dict(orient='records')
# Wrap into phpMyAdmin-like table structure for compatibility
wrapper = [
    {'type': 'header', 'version': '1'},
    {'type': 'table', 'name': 'ocorrencia_policial_operacional', 'data': records}
]
with open(OUT, 'w', encoding='utf-8') as f:
    json.dump(wrapper, f, ensure_ascii=False)

print(f'Wrote JSON with {len(records)} records to {OUT}')
