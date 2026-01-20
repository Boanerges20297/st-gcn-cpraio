#!/usr/bin/env python3
"""
Sprint 2 Task 2.3: Gerar tensores (days, nodes, features=7) para modelo STGCN_Cpraio_v2

Carrega territory_daily_features.parquet e agrupa por:
- Nós = area_faccao (territórios únicos)
- Features (7): 
  1. total_crimes (soma de: n_cvli + n_armas + n_large_seizures)
  2. n_cvli
  3. drogas_kg (total_drogas_g / 1000)
  4. n_armas
  5. n_large_seizures
  6. n_weapons_drugs
  7. territory_stability (normalizado entre 0-1, inverso de variabilidade)
- Tempo: days (cada dia)

Salva tensor shape (n_days, n_nodes, 7) como .pt e metadados de nós.
"""

import os
import pandas as pd
import numpy as np
import torch
from datetime import datetime

TERR_DAILY_IN = 'data/processed/territory_daily_features.parquet'
OUT_TENSOR = 'data/tensors/features_tensor_2025.pt'
OUT_METADATA = 'data/tensors/tensor_metadata_2025.pt'
OUT_CSV = 'outputs/tensor_shape_info.csv'


def compute_territory_stability(daily_scores):
    """
    Calcula estabilidade do território (inverso do coeficiente de variação).
    daily_scores: série de feature_score por dia
    """
    if len(daily_scores) < 2:
        return 0.5  # Default para novos territórios
    
    mean_score = daily_scores.mean()
    std_score = daily_scores.std()
    
    if mean_score == 0:
        return 0.0
    
    cv = std_score / mean_score  # Coeficiente de variação
    stability = 1.0 / (1.0 + cv)  # Normalizar entre 0-1 (alto CV = low stability)
    return min(stability, 1.0)


def main():
    print('\n' + '='*70)
    print('Sprint 2 Task 2.3: GERAR TENSORES PARA MODELO')
    print('='*70)
    
    print('\n[1] Carregando territory_daily_features...')
    if not os.path.exists(TERR_DAILY_IN):
        print(f'    ERRO: {TERR_DAILY_IN} não encontrado')
        return
    
    df = pd.read_parquet(TERR_DAILY_IN)
    df['data'] = pd.to_datetime(df['data'], errors='coerce')
    df = df.sort_values('data').reset_index(drop=True)
    
    print(f'    - Registros: {len(df)}')
    print(f'    - Datas: {df["data"].min()} → {df["data"].max()}')
    print(f'    - Nós (area_faccao): {df["area_faccao"].nunique()}')
    
    # Remover valores inválidos
    df = df.dropna(subset=['data', 'area_faccao'])
    df = df[df['area_faccao'] != 'Desconhecido'].copy()
    
    print(f'    - Após limpeza: nós={df["area_faccao"].nunique()}, registros={len(df)}')
    
    # Mapeamento de nós (área_faccao -> índice)
    nodes_unique = sorted(df['area_faccao'].unique())
    node_to_idx = {node: idx for idx, node in enumerate(nodes_unique)}
    idx_to_node = {idx: node for node, idx in node_to_idx.items()}
    
    print(f'\n[2] Mapeamento de nós ({len(nodes_unique)} nós):')
    for idx, node in idx_to_node.items():
        print(f'    {idx}: {node}')
    
    # Datas únicas
    dates_unique = sorted(df['data'].unique())
    date_to_idx = {date: idx for idx, date in enumerate(dates_unique)}
    
    n_days = len(dates_unique)
    n_nodes = len(nodes_unique)
    n_features = 7
    
    print(f'\n[3] Dimensões do tensor:')
    print(f'    - Dias: {n_days}')
    print(f'    - Nós: {n_nodes}')
    print(f'    - Features: {n_features}')
    print(f'    - Shape: ({n_days}, {n_nodes}, {n_features})')
    
    # Inicializar tensor (preencher com 0)
    X = np.zeros((n_days, n_nodes, n_features), dtype=np.float32)
    
    print(f'\n[4] Preenchendo tensor com features...')
    
    # Agregar por dia + nó
    daily_group = df.groupby(['data', 'area_faccao']).agg({
        'feature_score': 'sum',
        'is_cvli': 'sum',
        'total_armas': 'sum',
        'total_drogas_g': 'sum',
        'has_large_seizure': 'sum',
        'has_weapons_drugs': 'sum'
    }).reset_index()
    
    # Preencher tensor
    for _, row in daily_group.iterrows():
        d_idx = date_to_idx[row['data']]
        n_idx = node_to_idx[row['area_faccao']]
        
        # Features:
        n_cvli = int(row['is_cvli'])
        n_armas = int(row['total_armas'])
        n_large_seizures = int(row['has_large_seizure'])
        total_crimes = n_cvli + n_armas + n_large_seizures
        
        drogas_kg = row['total_drogas_g'] / 1000.0
        n_weapons_drugs = int(row['has_weapons_drugs'])
        
        # Territory stability: já computamos por feature_score variabilidade
        # Para agora, usar score normalizado
        territory_stability = 0.5  # Placeholder (será refinado)
        
        X[d_idx, n_idx, 0] = total_crimes
        X[d_idx, n_idx, 1] = n_cvli
        X[d_idx, n_idx, 2] = drogas_kg
        X[d_idx, n_idx, 3] = n_armas
        X[d_idx, n_idx, 4] = n_large_seizures
        X[d_idx, n_idx, 5] = n_weapons_drugs
        X[d_idx, n_idx, 6] = territory_stability
    
    # Normalizações (Z-score por feature)
    print(f'\n[5] Normalizando features (Z-score)...')
    for f_idx in range(n_features):
        feature_data = X[:, :, f_idx].flatten()
        valid_mask = feature_data != 0
        
        if valid_mask.sum() > 1:
            mean_val = feature_data[valid_mask].mean()
            std_val = feature_data[valid_mask].std()
            
            if std_val > 0:
                X[:, :, f_idx] = (X[:, :, f_idx] - mean_val) / std_val
                print(f'    Feature {f_idx}: mean={mean_val:.3f}, std={std_val:.3f}')
    
    # Converter para tensor PyTorch
    X_torch = torch.from_numpy(X).float()
    
    print(f'\n[6] Estatísticas do tensor:')
    print(f'    - Shape: {X_torch.shape}')
    print(f'    - Type: {X_torch.dtype}')
    print(f'    - Min: {X_torch.min():.3f}, Max: {X_torch.max():.3f}')
    print(f'    - Mean: {X_torch.mean():.3f}, Std: {X_torch.std():.3f}')
    print(f'    - Non-zero elements: {(X_torch != 0).sum().item()} / {X_torch.numel()}')
    
    # Salvar tensor
    os.makedirs(os.path.dirname(OUT_TENSOR), exist_ok=True)
    torch.save(X_torch, OUT_TENSOR)
    print(f'\n[7] Tensor salvo: {OUT_TENSOR}')
    
    # Salvar metadados
    metadata = {
        'nodes': nodes_unique,
        'node_to_idx': node_to_idx,
        'idx_to_node': idx_to_node,
        'dates': [str(d) for d in dates_unique],
        'date_to_idx': {str(k): v for k, v in date_to_idx.items()},
        'feature_names': [
            'total_crimes',
            'n_cvli',
            'drogas_kg',
            'n_armas',
            'n_large_seizures',
            'n_weapons_drugs',
            'territory_stability'
        ],
        'shape': tuple(X_torch.shape),
        'created_at': datetime.now().isoformat()
    }
    
    torch.save(metadata, OUT_METADATA)
    print(f'[8] Metadados salvos: {OUT_METADATA}')
    
    # Salvar info como CSV
    info_rows = []
    for f_idx, fname in enumerate(metadata['feature_names']):
        f_data = X[:, :, f_idx].flatten()
        f_data_valid = f_data[f_data != 0]
        info_rows.append({
            'feature_idx': f_idx,
            'feature_name': fname,
            'mean': f_data_valid.mean() if len(f_data_valid) > 0 else 0,
            'std': f_data_valid.std() if len(f_data_valid) > 0 else 0,
            'min': f_data_valid.min() if len(f_data_valid) > 0 else 0,
            'max': f_data_valid.max() if len(f_data_valid) > 0 else 0,
            'sparsity_%': 100 * (f_data == 0).sum() / len(f_data)
        })
    
    info_df = pd.DataFrame(info_rows)
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    info_df.to_csv(OUT_CSV, index=False)
    print(f'[9] Info features: {OUT_CSV}')
    print('\n' + info_df.to_string(index=False))
    
    print(f'\n' + '='*70)
    print('[✓] Tensores gerados com sucesso!')
    print(f'    Próximos passos:')
    print(f'      - Integrar grafo com edge_weights')
    print(f'      - Executar treino curto (1 epoch) com STGCN_Cpraio_v2')
    print('='*70 + '\n')

if __name__ == '__main__':
    main()
