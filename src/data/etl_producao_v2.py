"""
ETL DE PRODUÇÃO V2 - Pipeline Completo
Integra dados CVLI corrigidos com contexto operacional
Estrutura: Facções por data, tensores normalizados, datasets enriquecidos

Fluxo:
1. Carregar CVLI de outputs/cvli_with_bairro.csv (tipo=cvli desde 2022)
2. Cruzar com dados operacionais normalizados
3. Enriquecer com coordenadas IBGE
4. Estruturar tensores T×N ou T×N×F
5. Salvar em data/processed/ (produção)
6. Integrar com estrutura de facções por data
"""

import json
import pandas as pd
import numpy as np
import geopandas as gpd
from pathlib import Path
from datetime import datetime
from difflib import SequenceMatcher
import logging

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"
RAW_DIR = DATA_DIR / "raw"
GRAPH_DIR = DATA_DIR / "graph"
OUTPUTS_DIR = Path("outputs")

# Criar diretórios se necessário
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

print("\n" + "="*80)
print("ETL DE PRODUÇÃO V2 - Pipeline Completo CVLI+Contexto")
print("="*80)

# ============================================================================
# ETAPA 1: Carregar e validar dados brutos
# ============================================================================
print("\n[ETAPA 1] Carregando dados brutos...")

# 1.1 Carregar CVLI de qualidade
logger.info("Carregando CVLI (outputs/cvli_with_bairro.csv tipo=cvli)...")
df_cvli = pd.read_csv(OUTPUTS_DIR / "cvli_with_bairro.csv", low_memory=False)
df_cvli['data'] = pd.to_datetime(df_cvli['data'])
df_cvli = df_cvli[(df_cvli['data'].dt.year >= 2022) & (df_cvli['tipo'].str.lower() == 'cvli')].copy()
df_cvli = df_cvli.dropna(subset=['latitude', 'longitude'])
logger.info(f"✓ {len(df_cvli):,} eventos CVLI carregados")

# 1.2 Carregar dados operacionais
logger.info("Carregando dados operacionais...")
df_op = pd.read_csv(RAW_DIR / "View_Ocorrencias_Operacionais_Modelo.csv", low_memory=False)
df_op['Data'] = pd.to_datetime(df_op['Data'])
df_op = df_op[(df_op['Data'].dt.year >= 2022)].copy()
logger.info(f"✓ {len(df_op):,} registros operacionais carregados")

# 1.3 Carregar GeoJSON de bairros (IBGE multi-região)
logger.info("Carregando dados geográficos IBGE...")
geojson_files = {
    'fortaleza': GRAPH_DIR / "fortaleza_bairros.geojson",
    'rmf': GRAPH_DIR / "ceara_rmf.geojson",
    'interior': GRAPH_DIR / "ceara_interior.geojson"
}

bairro_coords = {}
for region, path in geojson_files.items():
    try:
        with open(path) as f:
            geojson = json.load(f)
        for feature in geojson.get('features', []):
            props = feature['properties']
            bairro = props.get('name', props.get('NOME', '')).upper().strip()
            if bairro:
                try:
                    from shapely.geometry import shape
                    geom = shape(feature['geometry'])
                    centroid = geom.centroid
                    bairro_coords[bairro] = {'lat': centroid.y, 'long': centroid.x}
                except:
                    pass
    except Exception as e:
        logger.warning(f"Erro lendo {region}: {e}")

logger.info(f"✓ {len(bairro_coords)} bairros com coordenadas IBGE")

# ============================================================================
# ETAPA 2: Normalizar e enriquecer
# ============================================================================
print("\n[ETAPA 2] Normalizando e enriquecendo dados...")

# 2.1 Usar bairro_assigned do CVLI como padrão
df_cvli_norm = df_cvli.dropna(subset=['bairro_assigned']).copy()
bairros_normalized = set(df_cvli_norm['bairro_assigned'].unique())
logger.info(f"✓ {len(bairros_normalized)} bairros normalizados (padrão CVLI)")

# 2.2 Normalizar dados operacionais
def fuzzy_match(bairro_op, threshold=0.5):
    if pd.isna(bairro_op):
        return None
    bairro_op_upper = str(bairro_op).upper().strip()
    best_match = None
    best_ratio = threshold
    for bairro_norm in bairros_normalized:
        ratio = SequenceMatcher(None, bairro_op_upper, bairro_norm.upper()).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_match = bairro_norm
    return best_match

df_op['bairro_norm'] = df_op['BairroOcor'].apply(fuzzy_match)
df_op_norm = df_op.dropna(subset=['bairro_norm'])
logger.info(f"✓ {len(df_op_norm):,}/{len(df_op):,} registros operacionais normalizados ({len(df_op_norm)/len(df_op)*100:.1f}%)")

# 2.3 Enriquecer CVLI com coordenadas IBGE (já tem latitude/longitude)
# Verificar e preencher se houver gaps
df_cvli_norm['lat_enriquecido'] = df_cvli_norm['latitude']
df_cvli_norm['long_enriquecido'] = df_cvli_norm['longitude']
logger.info(f"✓ CVLI com coordenadas enriquecidas")

# 2.4 Enriquecer operacional com coordenadas IBGE
def get_bairro_coords(bairro_norm):
    if pd.isna(bairro_norm):
        return None, None
    coords = bairro_coords.get(str(bairro_norm).upper().strip())
    if coords:
        return coords['lat'], coords['long']
    return None, None

df_op_norm[['lat_ibge', 'long_ibge']] = df_op_norm['bairro_norm'].apply(
    lambda x: pd.Series(get_bairro_coords(x))
)
filled_coords = df_op_norm[['lat_ibge', 'long_ibge']].notna().all(axis=1).sum()
logger.info(f"✓ {filled_coords:,} registros operacionais com coordenadas IBGE")

# ============================================================================
# ETAPA 3: Criar período unificado e matrizes
# ============================================================================
print("\n[ETAPA 3] Criando tensores spatio-temporais...")

# 3.1 Período comum
date_min = max(df_cvli_norm['data'].min(), df_op_norm['Data'].min())
date_max = min(df_cvli_norm['data'].max(), df_op_norm['Data'].max())
dates = pd.date_range(date_min, date_max, freq='D')
T = len(dates)
N = len(bairros_normalized)
bairro_to_idx = {b: i for i, b in enumerate(sorted(bairros_normalized))}

logger.info(f"✓ Período: {date_min.date()} a {date_max.date()} ({T} dias)")
logger.info(f"✓ Bairros: {N}")
logger.info(f"✓ Dimensões tensor univariado: {T}×{N}={T*N:,} células")

# 3.2 Tensor univariado (CVLI only)
matrix_cvli = np.zeros((T, N))
for idx, row in df_cvli_norm.iterrows():
    t_idx = (row['data'].date() - dates[0].date()).days
    if 0 <= t_idx < T and row['bairro_assigned'] in bairro_to_idx:
        n_idx = bairro_to_idx[row['bairro_assigned']]
        matrix_cvli[t_idx, n_idx] += 1

logger.info(f"✓ Tensor CVLI: {int(matrix_cvli.sum()):,} eventos em {np.count_nonzero(matrix_cvli):,} células")

# 3.3 Tensores para contexto (Prisões, Apreensões)
matrix_prisoes = np.zeros((T, N))
matrix_apreensoes = np.zeros((T, N))

# Prisões
df_prisoes = df_op_norm[df_op_norm['Natureza'].str.contains('Prisão|Preso|Mandado', case=False, na=False)]
for idx, row in df_prisoes.iterrows():
    t_idx = (row['Data'].date() - dates[0].date()).days
    if 0 <= t_idx < T and row['bairro_norm'] in bairro_to_idx:
        n_idx = bairro_to_idx[row['bairro_norm']]
        matrix_prisoes[t_idx, n_idx] += 1

# Apreensões
df_apreensoes = df_op_norm[
    ((df_op_norm['total_armas_cache'] > 0) | 
     (df_op_norm['total_drogas_cache'] > 0) | 
     (df_op_norm['Dinheiro_Apreendido'] > 0))
]
for idx, row in df_apreensoes.iterrows():
    t_idx = (row['Data'].date() - dates[0].date()).days
    if 0 <= t_idx < T and row['bairro_norm'] in bairro_to_idx:
        n_idx = bairro_to_idx[row['bairro_norm']]
        matrix_apreensoes[t_idx, n_idx] += 1

logger.info(f"✓ Tensor Prisões: {int(matrix_prisoes.sum()):,} eventos")
logger.info(f"✓ Tensor Apreensões: {int(matrix_apreensoes.sum()):,} eventos")

# 3.4 Tensor multivariado (T×N×3)
tensor_multi = np.stack([matrix_cvli, matrix_prisoes, matrix_apreensoes], axis=2)
logger.info(f"✓ Tensor multivariado: {tensor_multi.shape}")

# ============================================================================
# ETAPA 4: Salvar em produção
# ============================================================================
print("\n[ETAPA 4] Salvando artefatos em produção...")

# Backup de dados antigos (se existirem)
for f in PROCESSED_DIR.glob("*.npy"):
    backup_path = PROCESSED_DIR / f"{f.stem}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npy"
    f.rename(backup_path)
    logger.info(f"✓ Backup: {f.name} → {backup_path.name}")

# Salvar tensores
np.save(PROCESSED_DIR / "tensor_cvli_univariado.npy", matrix_cvli)
np.save(PROCESSED_DIR / "tensor_prisoes.npy", matrix_prisoes)
np.save(PROCESSED_DIR / "tensor_apreensoes.npy", matrix_apreensoes)
np.save(PROCESSED_DIR / "tensor_multivariado.npy", tensor_multi)
logger.info(f"✓ Tensores salvos em {PROCESSED_DIR}")

# Salvar metadados
metadata = {
    'criacao': datetime.now().isoformat(),
    'periodo': f"{date_min.date()} a {date_max.date()}",
    'dias': T,
    'bairros': N,
    'eventos_cvli': int(matrix_cvli.sum()),
    'eventos_prisoes': int(matrix_prisoes.sum()),
    'eventos_apreensoes': int(matrix_apreensoes.sum()),
    'tensor_univariado_shape': [T, N],
    'tensor_multivariado_shape': [T, N, 3],
    'bairros_normalizados': sorted(bairros_normalized),
    'data_sources': {
        'cvli': 'outputs/cvli_with_bairro.csv (tipo=cvli)',
        'operacional': 'data/raw/View_Ocorrencias_Operacionais_Modelo.csv',
        'geojson': 'fortaleza_bairros.geojson + ceara_rmf.geojson + ceara_interior.geojson'
    }
}

with open(PROCESSED_DIR / "metadata_producao_v2.json", 'w') as f:
    json.dump(metadata, f, indent=2)
logger.info(f"✓ Metadados salvos")

# ============================================================================
# ETAPA 5: Salvar CSVs enriquecidos
# ============================================================================
print("\n[ETAPA 5] Salvando CSVs enriquecidos...")

# CVLI enriquecido
df_cvli_prod = df_cvli_norm[[
    'id', 'data', 'bairro_assigned', 'cidade_norm',
    'latitude', 'longitude', 'tipo', 'tipo_evento'
]].copy()
df_cvli_prod.to_csv(PROCESSED_DIR / "cvli_producao.csv", index=False)
logger.info(f"✓ cvli_producao.csv ({len(df_cvli_prod):,} registros)")

# Operacional normalizado
df_op_prod = df_op_norm[[
    'Controle', 'Data', 'Natureza', 'BairroOcor', 'bairro_norm',
    'CidadeOcor', 'total_armas_cache', 'total_drogas_cache',
    'Dinheiro_Apreendido', 'lat_ibge', 'long_ibge'
]].copy()
df_op_prod.to_csv(PROCESSED_DIR / "operacional_producao.csv", index=False)
logger.info(f"✓ operacional_producao.csv ({len(df_op_prod):,} registros)")

# ============================================================================
# ETAPA 6: Integração com facções por data
# ============================================================================
print("\n[ETAPA 6] Preparando integração com facções por data...")

facoes_dir = GRAPH_DIR / f"faccoes_{datetime.now().strftime('%d_%m_%Y')}"
if not facoes_dir.exists():
    facoes_dir.mkdir(parents=True)
    logger.info(f"✓ Criada pasta de facções por data: {facoes_dir.name}")
else:
    logger.info(f"✓ Pasta de facções por data já existe: {facoes_dir.name}")

# Verificar facções mais recentes
latest_faccoes_dir = max(
    (d for d in GRAPH_DIR.glob("faccoes_*") if d.is_dir()),
    key=lambda p: p.name,
    default=None
)
if latest_faccoes_dir:
    logger.info(f"✓ Facções mais recentes: {latest_faccoes_dir.name}")

# ============================================================================
# ETAPA 7: Relatório final
# ============================================================================
print("\n[ETAPA 7] Gerando relatório final...")

report = f"""# ETL DE PRODUÇÃO V2 - Relatório de Execução

**Data:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Resumo

- **Período:** {date_min.date()} a {date_max.date()} ({T} dias)
- **Bairros normalizados:** {N}
- **Eventos CVLI:** {int(matrix_cvli.sum()):,}
- **Eventos Prisões:** {int(matrix_prisoes.sum()):,}
- **Eventos Apreensões:** {int(matrix_apreensoes.sum()):,}

## Tensores Gerados

- Tensor univariado (CVLI): {T}×{N} = {T*N:,} células
- Tensor multivariado: {T}×{N}×3 = {T*N*3:,} células
- Sparsidade CVLI: {(1-np.count_nonzero(matrix_cvli)/(T*N))*100:.2f}%

## Arquivos Salvos

### Tensores
- `tensor_cvli_univariado.npy` ({T}×{N})
- `tensor_prisoes.npy` ({T}×{N})
- `tensor_apreensoes.npy` ({T}×{N})
- `tensor_multivariado.npy` ({T}×{N}×3)

### CSVs
- `cvli_producao.csv` ({len(df_cvli_prod):,} registros)
- `operacional_producao.csv` ({len(df_op_prod):,} registros)

### Metadados
- `metadata_producao_v2.json`

## Próximos Passos

1. Usar tensores em `src/trainer.py` para retreinamento
2. Adaptar `src/model.py` para nova estrutura
3. Atualizar `src/data_loader.py` com novos paths
4. Validar com `src/predict.py`

---
**Status:** ✅ CONCLUÍDO
"""

report_path = PROCESSED_DIR / "ETL_PRODUCAO_V2_RELATORIO.md"
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(report)
logger.info(f"✓ Relatório salvo: {report_path}")

print("\n" + "="*80)
print(f"✅ ETL DE PRODUÇÃO V2 CONCLUÍDO COM SUCESSO")
print(f"   Tensores: {PROCESSED_DIR}")
print(f"   CSVs: {PROCESSED_DIR}")
print(f"   Relatório: {report_path}")
print("="*80 + "\n")
