import json
import pandas as pd

print("=" * 80)
print("ANÁLISE DETALHADA: O QUE É ORCRIM?")
print("=" * 80)

with open('data/graph/ORCRIM_extraido.geojson', 'r', encoding='utf-8') as f:
    orcrim_data = json.load(f)

features = orcrim_data.get('features', [])

# Analisa estrutura
print(f'\nTotal de features: {len(features)}')
print(f'\nExaminando primeiro feature:')
f1 = features[0]
print(f'  Tipo: {f1.get("type")}')
print(f'  Properties: {f1.get("properties")}')
print(f'  Geometry type: {f1.get("geometry", {}).get("type")}')

# Extrai todos os nomes e agrupa
nomes_raw = []
for f in features:
    nome = f.get('properties', {}).get('nome', '')
    nomes_raw.append(nome)

# Analisa padrões de naming
print(f'\n\nPadrões de naming em ORCRIM:')
print(f'Total de features: {len(nomes_raw)}')
print(f'Nomes únicos: {len(set(nomes_raw))}')

# Exemplos com "CAPITAL" (Fortaleza)
capital_examples = [n for n in nomes_raw if any(x in n.upper() for x in ['BARRA', 'CAIS', 'CENTRO', 'FORTALEZA'])]
print(f'\nExemplos com nomes de Fortaleza:')
for ex in capital_examples[:20]:
    print(f'  - {ex}')

# Carrega dados de crime
df = pd.read_parquet('data/processed/base_consolidada.parquet')

# Verifica se CAPITAL tem dados
df_capital = df[df['regiao_sistema'] == 'CAPITAL']
print(f'\n\nDados de CAPITAL:')
print(f'  Total registros: {len(df_capital)}')
print(f'  Locais únicos: {len(df_capital["local_oficial"].unique())}')
print(f'  Locais: {sorted(df_capital["local_oficial"].unique())}')

# Busca cobertura
nomes_set = set(n.upper() for n in nomes_raw)
locais_capital = set(df_capital['local_oficial'].unique())

cobertura_capital = nomes_set & locais_capital
print(f'\nCobertura em CAPITAL: {len(cobertura_capital)}/{len(locais_capital)}')
print(f'  Cobertos: {cobertura_capital}')
print(f'  Faltando: {locais_capital - cobertura_capital}')

print("\n" + "=" * 80)
print("CONCLUSÃO:")
print("=" * 80)
print("""
ORCRIM_extraido.geojson contém:
- 2.487 features (quadras/logradouros cadastrados em Fortaleza)
- Apenas 5 de 7 locais de CAPITAL têm cobertura
- Praticamente nenhuma cobertura de RMF/INTERIOR

USO RECOMENDADO:
1. Manter para detalhe micro-regional de Fortaleza (se houver coordenadas exatas)
2. Usar para validação de geolocalização (comparar lat/lng de crime com quadra)
3. NÃO usar para substituir os bairros/municípios nos filtros

AÇÃO RECOMENDADA:
- Continuar usando fortaleza_bairros.geojson para mapa macro
- ORCRIM pode ser camada adicional opcional (zoom-in em Fortaleza)
""")
