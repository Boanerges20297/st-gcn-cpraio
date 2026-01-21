"""
Cria datasets regionais (CAPITAL, RMF, INTERIOR) a partir de
`data/tensors/dataset_cvli_novo_criterio.pt` usando `config.GEOJSON_PATHS`.
Salva em `data/tensors/dataset_<region>.pt` com chaves compatíveis.
"""
import json
from pathlib import Path
import torch
import geopandas as gpd

repo = Path(__file__).parent.parent
from src import config

SRC = repo / 'data' / 'tensors' / 'dataset_cvli_novo_criterio.pt'
OUT = repo / 'data' / 'tensors'

if not SRC.exists():
    print('Source dataset not found:', SRC)
    raise SystemExit(1)

print('Loading source dataset...')
d = torch.load(SRC)
X = d['X']  # (days, nodes, features)
b2i = d.get('bairro_to_idx') or {}
# normalize bairro names
bairro_names = {k.upper(): v for k, v in b2i.items()}

for region, geo_path in config.GEOJSON_PATHS.items():
    if not geo_path.exists():
        print('Geojson not found for', region)
        continue
    print('Processing region', region)
    gdf = gpd.read_file(geo_path)
    # try columns name or NM_MUNICIP
    if 'name' in gdf.columns:
        names = [str(x).upper() for x in gdf['name'].unique()]
    elif 'NM_MUNICIP' in gdf.columns:
        names = [str(x).upper() for x in gdf['NM_MUNICIP'].unique()]
    else:
        names = []

    idxs = [v for k, v in bairro_names.items() if k in names]
    if not idxs:
        print('  No matching bairros found for', region)
        continue

    X_region = X[:, idxs, :]
    # Build sub adjacency by slicing adjacency_matrix if present
    adj_path = OUT / 'adjacency_matrix.npy'
    try:
        import numpy as np
        adj = np.load(adj_path)
        adj_region = adj[np.ix_(idxs, idxs)]
        edge_index = torch.tensor(np.array(np.nonzero(adj_region)), dtype=torch.long)
    except Exception:
        edge_index = d.get('edge_index')

    bairro_to_idx_region = {nome: i for i, nome in enumerate([k for k, v in sorted(bairro_names.items(), key=lambda x: x[1]) if v in idxs])}

    dataset = {
        'X': X_region,
        'edge_index': edge_index,
        'bairro_to_idx': bairro_to_idx_region,
        'date_range': d.get('date_range'),
        'num_nodes': X_region.shape[1],
        'num_features': X_region.shape[2],
        'num_days': X_region.shape[0]
    }

    out_path = OUT / f'dataset_{region.lower()}.pt'
    torch.save(dataset, out_path)
    # metadata
    meta = {
        'num_nodes': dataset['num_nodes'],
        'num_features': dataset['num_features'],
        'num_days': dataset['num_days'],
        'bairros': list(bairro_to_idx_region.keys())
    }
    with open(OUT / f'metadata_{region.lower()}.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print('  Saved', out_path)

print('Done')
