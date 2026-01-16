# Estrutura do projeto

Árvore de pastas e arquivos do projeto `projeto-stgcn-cpraio`:

```
projeto-stgcn-cpraio/
├── main.py
├── requirements.txt
├── data/
│   ├── cache/
│   │   ├── 9d310ae198a9334827cef101d206d60cf21d617b.json
│   │   └── d803b764fb0928a783495a33ae8e614bb7270b12.json
│   ├── graph/
│   │   └── fortaleza_bairros.geojson
│   ├── processed/
│   └── raw/
│       ├── dados_status_ocorrencias_gerais.json
│       ├── data_with_coordinates.js
│       └── ocorrencias_tropa.json
├── tensors/
│   └── stgcn_dataset.pt
├── notebooks/
│   ├── 01_analise_exploratoria.ipynb
│   └── 02_teste_grafo.ipynb
├── outputs/
│   └── mapa_tatico.html
├── models/
│   ├── best_stgcn_strategic.pth
│   ├── best_stgcn.pth
│   ├── scaler_stats_strategic.pt
│   └── scaler_stats.pt
├── reports/
│   └── previsao_diaria.csv
└── src/
    ├── __init__.py
    ├── config.py
    ├── data_loader.py
    ├── graph_builder.py
    ├── model.py
    ├── predict.py
    ├── spatial_matcher.py
    ├── trainer.py
    ├── visualizar.py
    ├── __pycache__/
    └── scripts/
        ├── clean_topology.py
        └── fetch_geodata.py

```

Arquivo gerado automaticamente.
