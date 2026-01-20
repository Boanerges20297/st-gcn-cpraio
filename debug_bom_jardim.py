#!/usr/bin/env python3
import pandas as pd
import numpy as np
from pathlib import Path
import json

# Verificar se Bom Jardim está no tensor
print('='*70)
print('PROCURANDO BOM JARDIM NO TENSOR')
print('='*70)

metadata_path = Path('data/tensors/metadata_janela180d_completo.json')
if metadata_path.exists():
    with open(metadata_path, encoding='utf-8') as f:
        metadata = json.load(f)
    bairro_mapping = metadata.get('bairro_mapping', {})
    
    # bairro_mapping tem formato {nome: indice}
    encontrados = {k: v for k, v in bairro_mapping.items() if 'BOM' in k.upper() or 'JARDIM' in k.upper()}
    
    if encontrados:
        print('Encontrado no tensor:')
        for nome, idx in encontrados.items():
            print(f'  {nome}: Index {idx}')
    else:
        print('NAO ENCONTRADO no tensor')
    
    print(f'\nTotal de bairros no tensor: {len(bairro_mapping)}')
else:
    print('Metadata não encontrada')

# Verificar dados brutos
print('\n' + '='*70)
print('PROCURANDO BOM JARDIM NOS DADOS BRUTOS')
print('='*70)

try:
    base = pd.read_parquet('data/processed/base_consolidada.parquet')
    bom_jardim = base[base['local_oficial'].str.contains('BOM JARDIM', case=False, na=False)]
    print(f'Total de registros com BOM JARDIM: {len(bom_jardim)}')
    if len(bom_jardim) > 0:
        print(f'\nTipos de crime:')
        print(bom_jardim['tipo'].value_counts())
    else:
        print('NAO ENCONTRADO nos dados brutos')
except Exception as e:
    print(f'Erro: {e}')

# Verificar com diferentes variações
print('\n' + '='*70)
print('PROCURANDO VARIAÇÕES DE BOM JARDIM')
print('='*70)

try:
    base = pd.read_parquet('data/processed/base_consolidada.parquet')
    # Procurar várias variações
    variacoes = ['BOM JARDIM', 'BOM-JARDIM', 'BOA JARDIM', 'BOOMJARDIM']
    
    for var in variacoes:
        mask = base['local_oficial'].str.contains(var, case=False, na=False)
        if mask.any():
            print(f'\n{var}: {mask.sum()} registros')
            print(base[mask]['local_oficial'].unique())
except:
    pass

# Listar todos os bairros únicos da capital
print('\n' + '='*70)
print('TODOS OS BAIRROS NA CAPITAL (primeiros 30)')
print('='*70)

try:
    base = pd.read_parquet('data/processed/base_consolidada.parquet')
    capital = base[base['regiao'].str.contains('FORTALEZA', case=False, na=False) | 
                   (base['regiao'] == 'CAPITAL')]
    bairros = sorted(capital['local_oficial'].unique())
    for i, b in enumerate(bairros[:30]):
        print(f'{i+1}. {b}')
except:
    pass
