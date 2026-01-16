import torch
import pandas as pd
import geopandas as gpd
import numpy as np
<<<<<<< HEAD
import config
from tqdm import tqdm

def build_adjacency(gdf):
    """
    Constrói a lista de conexões (Edge Index) baseada em vizinhança geográfica.
    """
    print(f"    - Calculando vizinhança para {len(gdf)} nós...")
    adj_list = []
    
    # Cria índice espacial para performance
    # Para cada polígono, verifica quais outros tocam nele
    for idx, row in tqdm(gdf.iterrows(), total=len(gdf), desc="Topologia"):
        # Quem são os vizinhos? (touche)
        neighbors = gdf[gdf.geometry.touches(row.geometry)].index.tolist()
        
        # Adiciona arestas bidirecionais
        for neighbor in neighbors:
            adj_list.append([idx, neighbor])
            
    # Se algum nó ficar isolado (ilhas), conecta consigo mesmo para não quebrar a matriz
    connected_nodes = set(np.array(adj_list).flatten()) if adj_list else set()
    for i in range(len(gdf)):
        if i not in connected_nodes:
            adj_list.append([i, i])
            
    # Formato PyTorch Geometric: (2, Num_Edges)
    edge_index = torch.tensor(adj_list, dtype=torch.long).t().contiguous()
    return edge_index

def process_region(region_name):
    print(f"\n[BUILDER] Processando Região: {region_name}")
    
    # 1. Caminhos
    geo_path = config.GEOJSON_PATHS[region_name]
    crime_path = config.CRIME_DATA_PATHS[region_name]
    output_path = config.ARTIFACTS[region_name]['dataset']
    
    if not geo_path.exists() or not crime_path.exists():
        print(f"    [!] Arquivos não encontrados para {region_name}. Pulando.")
        return

    # 2. Carregar Mapa
    gdf = gpd.read_file(geo_path)
    nodes = gdf['name'].tolist() # Lista ordenada de nós
    
    # Mapeamento Nome -> Índice (0, 1, 2...)
    node_to_idx = {name: i for i, name in enumerate(nodes)}
    
    # 3. Construir Grafo (Arestas)
    edge_index = build_adjacency(gdf)
    print(f"    - Grafo construído: {edge_index.shape[1]} conexões.")

    # 4. Processar Crimes (Features)
    print("    - Processando Séries Temporais...")
    df_crimes = pd.read_parquet(crime_path)
    
    # Agrupar por Data e Nó (Município/Bairro)
    # Primeiro, garantir que temos uma grade temporal completa
    if df_crimes.empty:
        print("    [!] Sem crimes para processar. Gerando dataset vazio.")
        return

    df_crimes['date'] = pd.to_datetime(df_crimes['date'])
    min_date = df_crimes['date'].min()
    max_date = df_crimes['date'].max()
    
    all_dates = pd.date_range(start=min_date, end=max_date, freq='D')
    
    # Matriz: (Dias, Nós, Features) -> Features = 1 (Contagem de Crimes)
    # Inicializar tensor com zeros
    num_days = len(all_dates)
    num_nodes = len(nodes)
    num_features = 1 # Apenas contagem total por enquanto
    
    X = torch.zeros((num_days, num_nodes, num_features), dtype=torch.float)
    
    # Preencher Tensor
    # Isso pode ser lento em loop, vamos otimizar com pandas
    # Contagem diária por local
    
    # Assumindo que a coluna de local no parquet bate com 'name' ou 'name_upper' do GeoJSON
    # Vamos normalizar para garantir
    
    # Ajuste: O spatial_matcher usou 'name' do GeoJSON para fazer o join, 
    # mas o parquet resultante tem as colunas do GeoJSON original.
    # Geralmente é 'name' ou 'index_right' que guarda o nome do local.
    
    # Vamos usar o 'index_right' que é o índice do GeoDataFrame original se foi sjoin
    # Mas o sjoin salva colunas. Vamos ver se tem 'name'.
    location_col = 'name' if 'name' in df_crimes.columns else 'municipio'
    
    # Pivotar: Index=Date, Columns=Local, Values=Count
    daily_counts = df_crimes.groupby(['date', location_col]).size().reset_index(name='count')
    
    # Mapear para a matriz X
    for _, row in tqdm(daily_counts.iterrows(), total=len(daily_counts), desc="Tensorizing"):
        d_idx = (row['date'] - min_date).days
        loc_name = row[location_col]
        
        # Tenta achar o índice do nó.
        # Atenção: O GeoJSON pode ter 'Fortaleza' e o CSV 'FORTALEZA'.
        # O spatial_matcher já deve ter cuidado disso, mas vamos garantir.
        
        n_idx = -1
        # Tenta match direto
        if loc_name in node_to_idx:
            n_idx = node_to_idx[loc_name]
        else:
            # Tenta match upper
            for name, idx in node_to_idx.items():
                if str(loc_name).upper() == str(name).upper():
                    n_idx = idx
                    break
        
        if n_idx != -1 and d_idx < num_days:
            X[d_idx, n_idx, 0] = row['count']

    # 5. Salvar
    dataset = {
        'X': X,                     # O Tensor de Dados
        'edge_index': edge_index,   # A Topologia
        'nodes': nodes,             # Os Nomes dos Locais
        'dates': all_dates,         # O Calendário
        'features': ['CVLI']        # Nome das Features
    }
    
    torch.save(dataset, output_path)
    print(f"    [V] Dataset salvo: {output_path}")

def main():
    # Processa as 3 regiões sequencialmente
    for region in ['CAPITAL', 'RMF', 'INTERIOR']:
        try:
            process_region(region)
        except Exception as e:
            print(f"    [X] Falha crítica em {region}: {e}")

if __name__ == "__main__":
    main()
=======
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
>>>>>>> 73db3feb (Initial commit: add project files, exclude venv)
