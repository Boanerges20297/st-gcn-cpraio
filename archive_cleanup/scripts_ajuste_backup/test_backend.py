import sys
sys.path.insert(0, 'src')
from app import load_risk_map, load_occurrences
import json

print('=== TESTANDO LOAD_RISK_MAP ===')
risk = load_risk_map('CAPITAL')
if risk:
    print(f'Features: {len(risk["features"])}')
    f = risk["features"][0]
    print(f'Primeiro feature properties: {f["properties"]}')
else:
    print('Retornou None')

print('\n=== TESTANDO LOAD_OCCURRENCES ===')
df = load_occurrences()
print(f'Total de registros: {len(df)}')
print(f'Colunas: {df.columns.tolist()}')

df_capital = df[df['regiao_sistema'] == 'CAPITAL']
print(f'\nEm CAPITAL: {len(df_capital)}')
print(f'Facções únicas: {sorted(df_capital["faccao"].unique())}')
print(f'Contagem de facções:')
print(df_capital['faccao'].value_counts())

# Testa filtro de facção
df_cv = df_capital[df_capital['faccao'].str.upper() == 'CV']
print(f'\nFiltro CV: {len(df_cv)} registros')
