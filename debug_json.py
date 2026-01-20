import json

with open('data/raw/ocorrencia_caucaia_2025.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"Total items: {len(data)}")
print(f"Tipos únicos: {set(type(item).__name__ for item in data)}")

# Verificar estrutura dos primeiros items
for i in range(min(10, len(data))):
    item = data[i]
    if isinstance(item, dict):
        print(f"\nItem {i}: {list(item.keys())}")
    else:
        print(f"\nItem {i}: {type(item).__name__}")

# Contar items com 'Controle'
controle_count = sum(1 for item in data if isinstance(item, dict) and 'Controle' in item)
print(f"\nTotal com Controle: {controle_count}")

# Procurar a tabela
table_item = None
for i, item in enumerate(data):
    if isinstance(item, dict) and item.get('type') == 'table':
        print(f"\nTabela encontrada no índice {i}")
        print(f"Chaves: {list(item.keys())}")
        if 'data' in item:
            print(f"Número de registros: {len(item['data'])}")
            if item['data']:
                print(f"Primeiro registro: {list(item['data'][0].keys())[:5]}")
                print(f"Exemplo: {item['data'][0]}")
        break
