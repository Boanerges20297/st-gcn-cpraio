"""
GRAPH BUILDER - NOVO CRITÉRIO CVLI-CENTRIC
===========================================
Constrói grafos temporais com:
1. Nós = Bairros
2. Peso das arestas = Correlação CVLI entre bairros vizinhos
3. Features = [count_cvli_daily, count_cvp_daily, faccao_influence]
"""

import pandas as pd
import numpy as np
import torch
import geopandas as gpd
from pathlib import Path
from scipy.spatial.distance import pdist, squareform
from datetime import datetime, timedelta
import sys
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import config

def load_geographic_data():
    """Carrega os mapas e extrai lista de bairros"""
    print("[-] Carregando dados geográficos...")
    
    maps = {}
    bairros_total = []
    
    for region, path in config.GEOJSON_PATHS.items():
        if path.exists():
            gdf = gpd.read_file(path)
            maps[region] = gdf
            
            # Extrair nomes dos locais (bairros)
            if 'name' in gdf.columns:
                bairros = gdf['name'].unique().tolist()
            elif 'NM_MUNICIP' in gdf.columns:
                bairros = gdf['NM_MUNICIP'].unique().tolist()
            else:
                bairros = []
            
            bairros_total.extend([b.upper() for b in bairros if b])
            print(f"  [+] {region}: {len(bairros)} locais")
    
    bairros_total = list(set(bairros_total))
    print(f"  [V] Total de bairros/municípios: {len(bairros_total)}")
    
    return maps, bairros_total

def build_adjacency_matrix(maps, bairros_list):
    """
    Constrói matriz de adjacência baseada em proximidade geográfica
    """
    print("[-] Construindo matriz de adjacência...")
    
    n = len(bairros_list)
    adj_matrix = np.zeros((n, n))
    
    # Mapear bairro -> índice
    bairro_to_idx = {b: i for i, b in enumerate(bairros_list)}
    
    # Para cada região, verificar adjacências
    for region, gdf in maps.items():
        if gdf is None:
            continue
        
        # Normalizar nomes
        if 'name' in gdf.columns:
            gdf['bairro_norm'] = gdf['name'].str.upper()
        elif 'NM_MUNICIP' in gdf.columns:
            gdf['bairro_norm'] = gdf['NM_MUNICIP'].str.upper()
        else:
            continue
        
        # Spatial join com si mesmo para encontrar adjacências (touches)
        touches = gpd.sjoin(gdf, gdf, how='inner', predicate='touches')
        
        for _, row in touches.iterrows():
            b1 = row['bairro_norm_left']
            b2 = row['bairro_norm_right']
            
            if b1 in bairro_to_idx and b2 in bairro_to_idx:
                idx1 = bairro_to_idx[b1]
                idx2 = bairro_to_idx[b2]
                adj_matrix[idx1, idx2] = 1.0
                adj_matrix[idx2, idx1] = 1.0
    
    # Garantir self-loops
    np.fill_diagonal(adj_matrix, 1.0)
    
    print(f"  [V] Arestas criadas: {np.sum(adj_matrix > 0) // 2}")
    
    return adj_matrix, bairro_to_idx

def create_time_series_from_dataset(df_train, bairro_to_idx):
    """
    Transforma dataset em série temporal
    Dim: [num_days, num_nodes, num_features]
    Features: [cvli_count, cvp_count, faccao_CV, faccao_PCC, faccao_GDE, faccao_outras]
    """
    print("[-] Criando série temporal...")
    
    # Ordenar por data
    df_train = df_train.sort_values('data').copy()
    
    date_range = pd.date_range(
        start=df_train['data'].min(),
        end=df_train['data'].max(),
        freq='D'
    )
    
    num_days = len(date_range)
    num_nodes = len(bairro_to_idx)
    
    # Features: CVLI, CVP, CV, PCC, GDE, OUTRAS
    num_features = 6
    
    X = np.zeros((num_days, num_nodes, num_features))
    
    # Preencher série temporal
    for day_idx, date in enumerate(date_range):
        day_data = df_train[df_train['data'].dt.date == date.date()]
        
        for _, row in day_data.iterrows():
            bairro = row['bairro'].upper() if pd.notna(row['bairro']) else 'DESCONHECIDO'
            
            if bairro not in bairro_to_idx:
                continue
            
            node_idx = bairro_to_idx[bairro]
            
            # Feature selection baseado em tipo de crime
            if row['tipo_crime'].lower() == 'cvli':
                X[day_idx, node_idx, 0] += 1  # count_cvli
            elif row['tipo_crime'].lower() == 'cvp':
                X[day_idx, node_idx, 1] += 1  # count_cvp
            
            # Facção
            faccao = str(row['faccao']).upper() if pd.notna(row['faccao']) else 'SEM_FACCAO'
            if 'CV' in faccao:
                X[day_idx, node_idx, 2] += 1
            elif 'PCC' in faccao:
                X[day_idx, node_idx, 3] += 1
            elif 'GDE' in faccao:
                X[day_idx, node_idx, 4] += 1
            else:
                X[day_idx, node_idx, 5] += 1
    
    # Suavizar com rolling window (3 dias)
    X_smooth = np.zeros_like(X)
    for node in range(num_nodes):
        for feat in range(num_features):
            series = X[:, node, feat]
            # Rolling mean
            for t in range(len(series)):
                start = max(0, t - 1)
                end = min(len(series), t + 2)
                X_smooth[t, node, feat] = np.mean(series[start:end])
    
    X_torch = torch.tensor(X_smooth, dtype=torch.float32)
    
    print(f"  [V] Série temporal: {X_torch.shape}")
    print(f"      Dias: {num_days}, Bairros: {num_nodes}, Features: {num_features}")
    
    return X_torch, date_range

def build_graph_datasets():
    """Orquestração principal"""
    print("=" * 60)
    print(" GRAPH BUILDER - NOVO CRITÉRIO CVLI")
    print("=" * 60)
    
    # 1. Carregar dados
    maps, bairros_list = load_geographic_data()
    
    # 2. Matriz de adjacência
    adj_matrix, bairro_to_idx = build_adjacency_matrix(maps, bairros_list)
    
    # 3. Converter para edge_index (PyTorch)
    edge_index_np = np.array(np.nonzero(adj_matrix))
    edge_index = torch.tensor(edge_index_np, dtype=torch.long)
    
    print(f"[-] Edge Index formato: {edge_index.shape}")
    
    # 4. Carregar dataset de treino
    train_file = config.DATA_PROCESSED / "dataset_treino_cvli_2022_2024.parquet"
    
    if not train_file.exists():
        print(f"[X] Arquivo não encontrado: {train_file}")
        print("    Execute: python scripts_ajuste/01_etl_novo_criterio.py")
        return
    
    df_train = pd.read_parquet(train_file)
    print(f"[-] Carregados {len(df_train)} registros de treino")
    
    # 5. Criar série temporal
    X, date_range = create_time_series_from_dataset(df_train, bairro_to_idx)
    
    # 6. Salvar artefatos
    output_dir = config.TENSOR_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    dataset = {
        'X': X,
        'edge_index': edge_index,
        'bairro_to_idx': bairro_to_idx,
        'date_range': [d.isoformat() for d in date_range],
        'num_nodes': len(bairros_list),
        'num_features': X.shape[2],
        'num_days': X.shape[0]
    }
    
    dataset_path = output_dir / "dataset_cvli_novo_criterio.pt"
    torch.save(dataset, dataset_path)
    print(f"[+] Dataset salvo: {dataset_path}")
    
    # Também salvar adjacência
    adj_path = output_dir / "adjacency_matrix.npy"
    np.save(adj_path, adj_matrix)
    print(f"[+] Adjacência salva: {adj_path}")
    
    # Metadados
    metadata = {
        'num_nodes': len(bairros_list),
        'num_features': X.shape[2],
        'num_days': X.shape[0],
        'date_start': date_range[0].isoformat(),
        'date_end': date_range[-1].isoformat(),
        'features': ['cvli_count', 'cvp_count', 'faccao_CV', 'faccao_PCC', 'faccao_GDE', 'faccao_outras'],
        'bairros': bairros_list
    }
    
    metadata_path = output_dir / "metadata_cvli.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"[+] Metadados salvos: {metadata_path}")
    
    print("\n" + "=" * 60)
    print(" GRAPH BUILD CONCLUÍDO")
    print("=" * 60)
    
    return dataset, adj_matrix

if __name__ == "__main__":
    build_graph_datasets()
