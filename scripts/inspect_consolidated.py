import sys
from pathlib import Path
BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
proc = BASE / 'data' / 'processed'
paths = [
    proc / 'base_consolidada_orcrim_v3.parquet',
    proc / 'base_consolidada_orcrim_v2.parquet',
    proc / 'base_consolidada.parquet'
]
parquet_path = None
for p in paths:
    if p.exists():
        parquet_path = p
        break
if parquet_path is None:
    print('Nenhum arquivo consolidado encontrado em data/processed')
    raise SystemExit(1)
print('Usando:', parquet_path)

try:
    df = pd.read_parquet(parquet_path)
    print('Columns:', list(df.columns))
    print('\nSample rows:')
    print(df.head(5).to_dict(orient='records'))
except Exception as e:
    print('Erro ao ler parquet:', e)
