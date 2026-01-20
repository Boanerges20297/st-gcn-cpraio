"""Debug: Inspecionar estrutura JSON e colunas"""
import json
import pandas as pd
from pathlib import Path

data_path = Path("data/raw/dados_status_ocorrencias_gerais.json")

print("Carregando JSON...")
with open(data_path, 'r', encoding='utf-8-sig', errors='replace') as f:
    data = json.load(f)

print(f"Tipo de data: {type(data)}")
print(f"Comprimento: {len(data)}")

if isinstance(data, list):
    # Verificar estrutura
    for i, item in enumerate(data[:5]):
        print(f"\nItem {i}: {item.keys() if isinstance(item, dict) else type(item)}")
        if isinstance(item, dict) and 'data' in item:
            if isinstance(item['data'], list) and len(item['data']) > 0:
                print(f"  Data item 0 keys: {item['data'][0].keys()}")
                rec = item['data'][0]
                print(f"  Amostra: bairro={rec.get('bairro')}, tipo={rec.get('tipo')}, "
                      f"lat={rec.get('latitude')}, lng={rec.get('longitude')}, "
                      f"faccao={rec.get('area_faccao')}")
                break

# Extrair registros
raw_data = []
if isinstance(data, list):
    for item in data:
        if isinstance(item, dict) and 'data' in item:
            raw_data = item['data']
            break

print(f"\nTotal registros: {len(raw_data)}")

if raw_data:
    df = pd.DataFrame(raw_data)
    print(f"\nColunas: {list(df.columns)}")
    print(f"\nAmostra (primeiras 5 linhas):")
    print(df[['bairro', 'tipo', 'latitude', 'longitude', 'area_faccao', 'data']].head())
    print(f"\nBairros únicos: {df['bairro'].nunique()}")
    print(f"Valores NaN em bairro: {df['bairro'].isna().sum()}")
