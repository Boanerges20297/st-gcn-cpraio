#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
from pathlib import Path
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

data_dir = Path(__file__).parent.parent / 'data' / 'processed'

files = [
    data_dir / "base_consolidada_orcrim_v3.parquet",
    data_dir / "base_consolidada_orcrim_v2.parquet",
    data_dir / "base_consolidada.parquet"
]

for f in files:
    if f.exists():
        print(f"\n{'='*60}")
        print(f"ARQUIVO: {f.name}")
        print(f"{'='*60}")
        df = pd.read_parquet(f)
        print(f"Shape: {df.shape}")
        print(f"Colunas: {df.columns.tolist()}")
        
        # Buscar facção
        if 'faccao_predominante' in df.columns:
            vals = df['faccao_predominante'].dropna().unique()
            print(f"faccao_predominante: {vals[:5] if len(vals) > 0 else 'VAZIO'}")
        elif 'faccao' in df.columns:
            vals = df['faccao'].dropna().unique()
            print(f"faccao: {vals[:5] if len(vals) > 0 else 'VAZIO'}")
        else:
            print("NÃO TEM COLUNA DE FACÇÃO!")
            
        # Verificar se tem local_oficial
        if 'local_oficial' in df.columns:
            print(f"Local_oficial (primeiros 3): {df['local_oficial'].unique()[:3]}")
