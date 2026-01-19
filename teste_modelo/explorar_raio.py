"""
ANÁLISE EXPLORATÓRIA DO ARQUIVO RAIO
=====================================
"""

import json
import pandas as pd
from pathlib import Path

raio_path = Path(__file__).parent / "ocorrencia_policial_operacional.json"

print("="*80)
print("EXPLORAÇÃO DO ARQUIVO RAIO")
print("="*80)

with open(raio_path, 'r', encoding='utf-8') as f:
    raio_data = json.load(f)

print(f"\n📊 Tipo: {type(raio_data)}")
print(f"  Tamanho: {len(raio_data) if isinstance(raio_data, (list, dict)) else 'N/A'}")

# Se é dict, mostrar chaves
if isinstance(raio_data, dict):
    print(f"\n  Chaves principais:")
    for key in list(raio_data.keys())[:20]:
        val_type = type(raio_data[key])
        if isinstance(raio_data[key], (list, dict)):
            size = len(raio_data[key])
            print(f"    • {key:<30} ({val_type.__name__}): {size} items")
        else:
            print(f"    • {key:<30} ({val_type.__name__}): {str(raio_data[key])[:50]}")

# Se é list, explorar primeiro item
elif isinstance(raio_data, list):
    print(f"\n  Primeiro item (amostra):")
    if len(raio_data) > 0:
        primeiro = raio_data[0]
        if isinstance(primeiro, dict):
            for key in list(primeiro.keys())[:30]:
                print(f"    • {key:<30} = {str(primeiro[key])[:60]}")
        else:
            print(f"    Tipo: {type(primeiro)}")
            print(f"    Valor: {primeiro}")

# Converter para DataFrame e explorar
if isinstance(raio_data, dict):
    # Se é dict, procurar a chave que contém os dados
    for key, val in raio_data.items():
        if isinstance(val, list) and len(val) > 0 and isinstance(val[0], dict):
            print(f"\n✓ Dados encontrados em chave '{key}'")
            df = pd.DataFrame(val)
            print(f"  Shape: {df.shape}")
            print(f"  Colunas: {list(df.columns)[:20]}")
            break
elif isinstance(raio_data, list) and len(raio_data) > 0 and isinstance(raio_data[0], dict):
    df = pd.DataFrame(raio_data)
    print(f"\n✓ DataFrame criado")
    print(f"  Shape: {df.shape}")
    print(f"  Colunas:")
    for col in df.columns[:30]:
        print(f"    • {col}")
