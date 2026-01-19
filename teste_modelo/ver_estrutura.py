import json
from pathlib import Path

raio_path = Path(__file__).parent / "ocorrencia_policial_operacional.json"

with open(raio_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

print('Estrutura do arquivo:')
for i, item in enumerate(data):
    print(f'Item {i}:')
    print(f'  type: {item.get("type")}')
    print(f'  name: {item.get("name")}')
    if 'data' in item:
        dados = item['data']
        print(f'  data type: {type(dados)}')
        if isinstance(dados, list) and len(dados) > 0:
            print(f'  data length: {len(dados)}')
            primeiro = dados[0]
            if isinstance(primeiro, dict):
                print(f'  colunas: {list(primeiro.keys())}')
                print(f'  primeiro registro:')
                for k, v in list(primeiro.items())[:10]:
                    print(f'    {k}: {v}')
