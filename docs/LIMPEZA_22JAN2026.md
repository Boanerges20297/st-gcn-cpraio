# Limpeza do Repositório — 22 de janeiro de 2026

## Sumário da Limpeza

### Arquivos Movidos para `archive_cleanup/`

#### `archive_cleanup/data_raw_backup/`
- `dados_status_ocorrencias_gerais_bairros_atribuidos.json` (arquivo vazio)
- `ocorrencias_tropa.json` (antigo, não utilizado)
- `ocorrencia_caucaia_2025.json` (dados regionais antigos)
- `data_with_coordinates.js` (formato obsoleto)
- `limites_ceara.geojson` (geojson grande, não utilizado)
- `limites_ceara_ibge_linhas.geojson` (geojson grande, não utilizado)

#### `archive_cleanup/scripts_ajuste_backup/`
- `scripts/35_debug_matching.py` (script de depuração, não utilizado)

#### `archive_cleanup/outputs_backup/`
- `01_etl_completo.log` (log antigo)
- `04_treino_completo.log` (log antigo)
- `api_diagnostico.log` (log antigo)
- `bairro_counts.json` (amostra temporária)
- `bairro_samples.json` (amostra temporária)
- `tipo_samples.json` (amostra temporária)
- `tipo_counts.csv` (amostra temporária)

### Mudanças em `.gitignore`

Adicionadas as seguintes entradas:
```
# Archive and cleanup (unused/old files)
archive_cleanup/
*.log
logs/
```

### O que foi mantido

- **`scripts/`**: Pasta mantida intacta com todos os 41 scripts (00-36 + inspect_cities + _check_data_shapes)
- **`outputs/`**: Relatórios atuais e dados processados (sazonalidade, trends, efetividade)
- **`data/`**: Dados principais (processed, cache, tensors, graph)
- **`src/`**: Código-fonte do projeto

### Estrutura Final

```
st-gcn_cpraio/
├── scripts/                    # ✅ MANTIDO (41 scripts operacionais)
├── data/
│   ├── raw/                   # ✅ Limpo (dados principais + geojson útil)
│   ├── processed/             # ✅ Mantido
│   ├── cache/
│   ├── tensors/
│   └── graph/
├── outputs/                   # ✅ Limpo (apenas relatórios atuais)
│   ├── docs/                  # ✅ Análises MD e CSV
│   └── models/
├── archive_cleanup/           # 🔄 IGNORADO PELO GIT
│   ├── data_raw_backup/
│   ├── scripts_ajuste_backup/
│   ├── outputs_backup/
│   └── docs_old/
├── src/                       # ✅ Mantido
├── notebooks/                 # ✅ Mantido
├── .gitignore                 # ✅ ATUALIZADO
└── ...
```

### Próximos Passos

1. Fazer commit com as mudanças de limpeza:
   ```bash
   git add -A
   git commit -m "Refactor: move unused files to archive_cleanup, update .gitignore"
   ```

2. Repositório agora está mais organizado e git ignorará `archive_cleanup/`

