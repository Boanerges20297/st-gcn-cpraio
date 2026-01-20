import pandas as pd
import json
import numpy as np
from pathlib import Path

df = pd.read_parquet(Path('data/processed/prisoes_raio_2025.parquet'))
print("Colunas do parquet:", df.columns.tolist())
print("Shape:", df.shape)

for i, row in df.iterrows():
    print(f"\nRow {i}:")
    print(f"  type: {row['type']}")
    print(f"  name: {row['name']}")
    print(f"  data type: {type(row['data'])}")
    if row['data'] is not None:
        print(f"  data len: {len(row['data']) if hasattr(row['data'], '__len__') else 'N/A'}")
        if isinstance(row['data'], (list, np.ndarray)):
            try:
                if isinstance(row['data'], np.ndarray):
                    # Tenta como JSON string
                    records = json.loads(str(row['data']))
                    print(f"  Parsed as JSON: {len(records)} records")
                    if len(records) > 0:
                        print(f"    Keys: {list(records[0].keys())}")
                else:
                    print(f"  First item: {row['data'][0]}")
            except:
                print(f"  Could not parse")
