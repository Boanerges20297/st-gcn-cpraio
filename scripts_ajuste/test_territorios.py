import sys
sys.path.insert(0, 'src')
from pathlib import Path
import json

# Verifica se os arquivos territoriais existem
base = Path('data/graph')
territoriais = list(base.glob('territorio_*.geojson'))
print(f'Arquivos territoriais encontrados: {len(territoriais)}')
for t in sorted(territoriais):
    print(f'  - {t.name}')

# Testa carregamento
with open('data/graph/territorio_cv_capital.geojson', 'r', encoding='utf-8') as f:
    data = json.load(f)
    features = data["features"]
    print(f'\nTerritório CV - CAPITAL:')
    print(f'  Features: {len(features)}')
    if features:
        f1 = features[0]
        props = f1["properties"]
        print(f'  Exemplo: {props.get("name")} - Percentual: {props.get("percentual")}%')
