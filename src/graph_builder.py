import torch
import pandas as pd
import geopandas as gpd
import numpy as np
import os
import sys
from tqdm import tqdm
import random

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import config

def build_graph(region_name, df_full):
    print(f"\n>>> CONSTRUINDO GRAFO: {region_name}")
    
    # 1. Carregar Mapa (Nós)
    geo_path = config.GEOJSON_PATHS[region_name]
    if not geo_path.exists(): return
    gdf = gpd.read_file(geo_path)
    nodes = gdf['name'].str.upper().str.strip().tolist()
    node_map = {name: i for i, name in enumerate(nodes)}
    
    # 2. Filtrar Crimes da Região
    df_region = df_full[df_full['regiao_sistema'] == region_name].copy()
    print(f"    - Crimes nesta região: {len(df_region)}")
    
    if df_region.empty:
        print("    [!] Sem dados. Pulando.")
        return

    # 3. Topologia Híbrida (Física + Facção)
    adj_list = []
    
    # A. Física (Vizinhos)
    for idx, row in tqdm(gdf.iterrows(), total=len(gdf), desc="Topologia Física"):
        neighbors = gdf[gdf.geometry.touches(row.geometry)].index.tolist()
        for n in neighbors:
            adj_list.append([idx, n])
            
    # B. Lógica (Facções)
    # Mapear Facção de cada Nó (usando a moda dos crimes ou o CSV de inteligência se carregado separadamente)
    # Como já cruzamos no ETL, podemos inferir a facção do nó pelos crimes que ocorrem nele
    # (Ou idealmente carregar o CSV de inteligência direto aqui também, mas vamos usar o consolidado para simplificar)
    
    node_factions = {}
    for node_name in nodes:
        # Pega a facção mais frequente neste local
        factions = df_region[df_region['local_oficial'] == node_name]['faccao_predominante']
        if not factions.empty:
            node_factions[node_name] = factions.mode()[0]
        else:
            node_factions[node_name] = 'DESCONHECIDO'
            
    # Criar conexões entre facções iguais
    faction_groups = {}
    for name, fac in node_factions.items():
        if fac not in ['DESCONHECIDO', 'NEUTRO', 'nan']:
            if fac not in faction_groups: faction_groups[fac] = []
            if name in node_map: faction_groups[fac].append(node_map[name])
            
    count_logical = 0
    for fac, indices in faction_groups.items():
        if len(indices) > 1:
            # Conecta aleatoriamente para criar "Small World"
            for i in indices:
                targets = random.sample(indices, min(len(indices), 3))
                for t in targets:
                    if i != t:
                        adj_list.append([i, t])
                        count_logical += 1
                        
    print(f"    - Conexões Lógicas (Facção): {count_logical}")
    
    # Converter para Tensor
    edge_index = torch.tensor(adj_list, dtype=torch.long).t().contiguous()
    
    # 4. Série Temporal (Features)
    df_region['data_hora'] = pd.to_datetime(df_region['data_hora'])
    min_date = df_region['data_hora'].min()
    max_date = df_region['data_hora'].max()
    all_dates = pd.date_range(min_date, max_date, freq='D')
    
    num_days = len(all_dates)
    num_nodes = len(nodes)
    X = torch.zeros((num_days, num_nodes, 1), dtype=torch.float)
    
    # Preencher
    daily = df_region.groupby([df_region['data_hora'].dt.date, 'local_oficial']).size().reset_index(name='count')
    date_map = {d.date(): i for i, d in enumerate(all_dates)}
    
    for _, row in daily.iterrows():
        d_idx = date_map.get(row['data_hora'])
        n_idx = node_map.get(row['local_oficial'])
        if d_idx is not None and n_idx is not None:
            X[d_idx, n_idx, 0] = row['count']
            
    # 5. Salvar
    dataset = {
        'X': X,
        'edge_index': edge_index,
        'nodes': nodes,
        'dates': all_dates,
        'features': ['CVLI']
    }
    torch.save(dataset, config.ARTIFACTS[region_name]['dataset'])
    print(f"    [V] Dataset salvo.")

if __name__ == "__main__":
    if not config.CONSOLIDATED_FILE.exists():
        print("[!] Rode o etl.py primeiro.")
        exit()
        
    df = pd.read_parquet(config.CONSOLIDATED_FILE)
    for reg in ['CAPITAL', 'RMF', 'INTERIOR']:
        try:
            build_graph(reg, df)
        except Exception as e:
            print(f"Erro em {reg}: {e}")