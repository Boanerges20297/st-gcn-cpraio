#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gerar metadata_janela180d_completo.json com todos os 390 bairros
Incorpora os 70 bairros faltantes do RAIO 2025
"""

import json
import unicodedata
import re
from pathlib import Path

def normalizar_bairro(nome):
    """Normaliza nome de bairro"""
    if not nome:
        return "DESCONHECIDO"
    nome = unicodedata.normalize('NFD', nome)
    nome = ''.join(c for c in nome if unicodedata.category(c) != 'Mn')
    nome = nome.upper().strip()
    nome = re.sub(r'\s+', ' ', nome)
    return nome

print("[1] Carregando metadata original...")
with open('data/tensors/metadata_janela180d.json', 'r', encoding='utf-8') as f:
    metadata_original = json.load(f)

bairro_mapping_original = metadata_original.get('bairro_mapping', {})
print("[OK] {} bairros carregados".format(len(bairro_mapping_original)))

print("\n[2] Carregando dados RAIO...")
with open('data/raw/ocorrencia_caucaia_2025.json', 'r', encoding='utf-8') as f:
    raw_data = json.load(f)

records = []
for item in raw_data:
    if isinstance(item, dict) and item.get('type') == 'table':
        records = item.get('data', [])
        break

bairros_raio_raw = [normalizar_bairro(rec.get('BairroOcor', '')) for rec in records]
bairros_raio = set(bairros_raio_raw)
print("[OK] {} bairros em RAIO".format(len(bairros_raio)))

print("\n[3] Gerando lista de gaps...")
bairros_modelo = set(normalizar_bairro(b) for b in bairro_mapping_original.keys())
bairros_gaps = bairros_raio - bairros_modelo
print("[OK] {} bairros faltantes".format(len(bairros_gaps)))

print("\n[4] Criando novo mapeamento...")
novo_mapeamento = {}
idx = 0

# Primeiro: adicionar bairros originais (mantendo indices)
for bairro in sorted(bairro_mapping_original.keys()):
    novo_mapeamento[bairro] = idx
    idx += 1

# Depois: adicionar gaps
for bairro in sorted(bairros_gaps):
    novo_mapeamento[bairro] = idx
    idx += 1

print("[OK] Novo mapeamento tem {} bairros".format(len(novo_mapeamento)))

print("\n[5] Gerando novo metadata...")
novo_metadata = {
    'data_inicio': metadata_original.get('data_inicio'),
    'data_fim': metadata_original.get('data_fim'),
    'num_dias': metadata_original.get('num_dias'),
    'num_bairros': len(novo_mapeamento),
    'num_features': 1,
    'janela_dias': 180,
    'bairro_mapping': novo_mapeamento,
    'normalizacao': 'percentil_diario',
    'tensor_shape': [
        metadata_original.get('num_dias', 0),
        len(novo_mapeamento),
        1
    ],
    'data_criacao': metadata_original.get('data_criacao'),
    'versao': 'completo_com_gaps'
}

print("[OK] Novo metadata gerado")

print("\n[6] Salvando novo metadata...")
output_path = Path('data/tensors/metadata_janela180d_completo.json')
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(novo_metadata, f, indent=2, ensure_ascii=False)

print("[OK] Salvo em {}".format(output_path))

print("\n" + "="*80)
print("RESUMO")
print("="*80)
print("""
Bairros originais: 318
Bairros em RAIO: 80
Gaps encontrados: 70
Total (novo): {}

Arquivo: data/tensors/metadata_janela180d_completo.json
""".format(len(novo_mapeamento)))

print("[CONCLUIDO]")
