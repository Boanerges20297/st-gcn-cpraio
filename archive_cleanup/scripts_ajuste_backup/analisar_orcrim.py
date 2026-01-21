import json
import pandas as pd
import geopandas as gpd

print("=" * 80)
print("ANÁLISE DO ARQUIVO ORCRIM_extraido.geojson")
print("=" * 80)

# 1. Carrega o arquivo
with open('data/graph/ORCRIM_extraido.geojson', 'r', encoding='utf-8') as f:
    orcrim_data = json.load(f)

features = orcrim_data.get('features', [])
print(f'\nTotal de features: {len(features)}')

# 2. Extrai nomes/propriedades
nomes = set()
for f in features:
    props = f.get('properties', {})
    if 'nome' in props:
        nomes.add(props['nome'])

print(f'Nomes únicos (ORCRIM): {len(nomes)}')
print(f'\nExemplos de nomes:')
for nome in sorted(list(nomes))[:10]:
    print(f'  - {nome}')

# 3. Compara com dados de crime
df_crime = pd.read_parquet('data/processed/base_consolidada.parquet')

# Extrai locais dos dados de crime
locais_crime = set(df_crime['local_oficial'].unique())
print(f'\n\nLocais em dados de crime: {len(locais_crime)}')

# Normaliza nomes do ORCRIM
nomes_orcrim_base = set()
for nome in nomes:
    # Remove sufixos como "- AIS 01"
    base = nome.split(' - ')[0].upper().strip()
    nomes_orcrim_base.add(base)

print(f'Locais base em ORCRIM: {len(nomes_orcrim_base)}')
print(f'\nExemplos:')
for nome in sorted(list(nomes_orcrim_base))[:10]:
    print(f'  - {nome}')

# 4. Comparação
cobertura = nomes_orcrim_base & locais_crime
faltando_em_orcrim = locais_crime - nomes_orcrim_base
extras_em_orcrim = nomes_orcrim_base - locais_crime

print(f'\n\n=== COMPARAÇÃO ===')
print(f'Locais com cobertura ORCRIM: {len(cobertura)}')
print(f'Locais SEM cobertura ORCRIM: {len(faltando_em_orcrim)}')
print(f'  (Extras em ORCRIM): {len(extras_em_orcrim)}')

if faltando_em_orcrim:
    print(f'\nLocais em crime SEM cobertura ORCRIM:')
    for local in sorted(list(faltando_em_orcrim))[:15]:
        print(f'  - {local}')

if extras_em_orcrim:
    print(f'\nLocais EXTRAS em ORCRIM (sem crime):')
    for local in sorted(list(extras_em_orcrim))[:15]:
        print(f'  - {local}')

# 5. Verifica distribuição por região
print(f'\n\n=== DISTRIBUIÇÃO POR REGIÃO ===')
for regiao in ['CAPITAL', 'RMF', 'INTERIOR']:
    locais_regiao = set(df_crime[df_crime['regiao_sistema'] == regiao]['local_oficial'].unique())
    cobertura_regiao = locais_regiao & nomes_orcrim_base
    print(f'{regiao}: {len(cobertura_regiao)}/{len(locais_regiao)} cobertos')

print("\n" + "=" * 80)
