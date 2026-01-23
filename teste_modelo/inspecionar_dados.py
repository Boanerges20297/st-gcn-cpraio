import json
from pathlib import Path
import pandas as pd

print("="*60)
print("INSPEÇÃO DE ESTRUTURA DE DADOS")
print("="*60)

file = Path('data/raw/ocorrencia_policial_operacional.json')
if file.exists():
    print(f"\n✅ Arquivo encontrado: {file}")
    
    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"\nTipo raiz: {type(data)}")
    
    if isinstance(data, dict):
        print(f"Chaves: {list(data.keys())}")
        
        if 'data' in data:
            inner_data = data['data']
            print(f"Tipo de 'data': {type(inner_data)}")
            
            if isinstance(inner_data, list) and len(inner_data) > 0:
                print(f"Número de registros: {len(inner_data)}")
                print(f"\nPrimeiro registro:")
                print(json.dumps(inner_data[0], ensure_ascii=False, indent=2))
    
    elif isinstance(data, list):
        print(f"Número de registros: {len(data)}")
        if len(data) > 0:
            print(f"Tipo primeiro elemento: {type(data[0])}")
            print(f"\nPrimeiro registro:")
            if isinstance(data[0], dict):
                print(json.dumps(data[0], ensure_ascii=False, indent=2))
else:
    print(f"❌ Arquivo não encontrado: {file}")

# Verificar dados processados também
print("\n" + "="*60)
print("DADOS PROCESSADOS DISPONÍVEIS")
print("="*60)

parquet_file = Path('data/processed/prisoes_with_features.parquet')
if parquet_file.exists():
    print(f"\n✅ Arquivo parquet encontrado: {parquet_file}")
    df = pd.read_parquet(parquet_file)
    print(f"Shape: {df.shape}")
    print(f"Colunas: {df.columns.tolist()}")
    print(f"\nPrimeiras linhas:")
    print(df.head())
else:
    print(f"❌ Arquivo não encontrado: {parquet_file}")
