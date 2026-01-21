"""
Inspect CidadeOcor field to check for city name variations
"""

import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data.operations_loader import OperationsLoader

# Load operations
loader = OperationsLoader('data/raw/ocorrencia_policial_operacional.json')
df = loader.load()

# Inspect CidadeOcor field
print('=' * 80)
print('INSPEÇÃO DO CAMPO CidadeOcor')
print('=' * 80)
print(f'\nUnique cities: {df["CidadeOcor"].nunique()}')
print(f'Total records: {len(df)}')

# Show top cities
print('\nTop 25 cities by frequency:')
top_cities = df['CidadeOcor'].value_counts().head(25)
for city, count in top_cities.items():
    print(f'  {city}: {count}')

# Show all unique cities
print(f'\n\nAll {df["CidadeOcor"].nunique()} unique cities:')
all_cities = sorted(df['CidadeOcor'].dropna().unique())
for i, city in enumerate(all_cities, 1):
    print(f'  {i:2d}. {city}')
