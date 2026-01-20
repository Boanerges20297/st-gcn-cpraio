import json

with open('data/raw/ocorrencias_tropa.json', encoding='utf-8') as f:
    data = json.load(f)

# Encontrar table
for item in data:
    if isinstance(item, dict) and item.get('type') == 'table':
        records = item.get('data', [])
        if records:
            print(f"Total de registros: {len(records)}")
            print(f"\nColunas disponíveis:")
            cols = list(records[0].keys())
            for col in cols:
                print(f"  - {col}")
            
            print(f"\nExemplo de primeiro registro:")
            for col in cols[:15]:
                print(f"  {col}: {records[0].get(col, 'N/A')}")
