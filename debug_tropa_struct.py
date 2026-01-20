import json

with open('data/raw/ocorrencias_tropa.json', encoding='utf-8') as f:
    data = json.load(f)

print(f"Total items: {len(data)}")
for i, item in enumerate(data):
    if isinstance(item, dict):
        print(f"{i}: type={item.get('type')}, name={item.get('name', 'N/A')[:40]}")
        if item.get('type') == 'table':
            print(f"   -> {len(item.get('data', []))} registros")
            if len(item.get('data', [])) > 0:
                first_record = item['data'][0]
                print(f"   -> Colunas: {list(first_record.keys())[:10]}")
