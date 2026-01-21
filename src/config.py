import os
from pathlib import Path
from dotenv import load_dotenv

# 1. Definir Raiz
BASE_DIR = Path(__file__).resolve().parent.parent

# 2. Carregar .env
env_path = BASE_DIR / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)

# --- CAMINHOS DE DADOS ---
DATA_RAW = BASE_DIR / "data" / "raw"
DATA_GRAPH = BASE_DIR / "data" / "graph"
DATA_PROCESSED = BASE_DIR / "data" / "processed"

# NOVO ALVO CENTRAL
CONSOLIDATED_FILE_V1 = DATA_PROCESSED / "base_consolidada.parquet"
CONSOLIDATED_FILE_V2 = DATA_PROCESSED / "base_consolidada_orcrim_v2.parquet"
CONSOLIDATED_FILE_V3 = DATA_PROCESSED / "base_consolidada_orcrim_v3.parquet"

# Usar v3 (sjoin corrigido) se existir, senão v2, senão v1
if CONSOLIDATED_FILE_V3.exists():
    CONSOLIDATED_FILE = CONSOLIDATED_FILE_V3
elif CONSOLIDATED_FILE_V2.exists():
    CONSOLIDATED_FILE = CONSOLIDATED_FILE_V2
else:
    CONSOLIDATED_FILE = CONSOLIDATED_FILE_V1

# Mapas (Geometria) - Mantidos para definir quem é vizinho físico
GEOJSON_PATHS = {
    'CAPITAL': DATA_GRAPH / "fortaleza_bairros.geojson",
    'RMF': DATA_GRAPH / "ceara_rmf.geojson",
    'INTERIOR': DATA_GRAPH / "ceara_interior.geojson"
}

# Caminho para dados meteorológicos (INMET) — CSV com colunas de estação/coord/variáveis por data
INMET_CSV = BASE_DIR / "data" / "raw" / "inmet_stations.csv"

# Lista de features exógenas que podem ser agregadas (ordenadas)
EXOGENEOUS_FEATURES = ['precip_mm', 'temp_mean', 'holiday']

# --- ARTEFATOS (GRAFOS E MODELOS) ---
TENSOR_DIR = BASE_DIR / "data" / "tensors"
MODEL_DIR = BASE_DIR / "outputs" / "models"
OUTPUT_DIR = BASE_DIR / "outputs"
REPORT_DIR = OUTPUT_DIR / "reports"

# Dicionário de Artefatos
ARTIFACTS = {
    'CAPITAL': {
        'dataset': TENSOR_DIR / "dataset_capital.pt",
        'model': MODEL_DIR / "model_capital.pth",
        'stats': MODEL_DIR / "stats_capital.pt",
        'prediction': REPORT_DIR / "pred_capital_bairros.csv"
    },
    'RMF': {
        'dataset': TENSOR_DIR / "dataset_rmf.pt",
        'model': MODEL_DIR / "model_rmf.pth",
        'stats': MODEL_DIR / "stats_rmf.pt",
        'prediction': REPORT_DIR / "pred_rmf.csv"
    },
    'INTERIOR': {
        'dataset': TENSOR_DIR / "dataset_interior.pt",
        'model': MODEL_DIR / "model_interior.pth",
        'stats': MODEL_DIR / "stats_interior.pt",
        'prediction': REPORT_DIR / "pred_interior.csv"
    }
}

# --- HIPERPARÂMETROS ---
HyperParams = {
    # Parâmetros gerais (podem ser sobrescritos por alvo específico)
    'window_size': 14,    # dias de histórico (default genérico)
    'target_window': 15,  # dias de previsão (default genérico)
    # Parâmetros por tipo de evento: CVLI (homicídios) e CVP (patrimoniais)
    'window_size_cvli': 180,   # histórico de 180 dias para CVLI
    'target_window_cvli': 30,  # prever 30 dias à frente para CVLI
    'window_size_cvp': 30,     # histórico de 30 dias para CVP
    'target_window_cvp': 7,    # prever 7 dias à frente para CVP
    # Limiar de criticidade em percentil (ex.: 0.90 significa top 10% como crítico)
    'criticality_percentile': 0.90,
    # Lista de features exógenas que NÃO entram no cálculo do limiar (podem influenciar vieses)
    'threshold_excluded_features': EXOGENEOUS_FEATURES,
    # Multiplicador aplicado ao limiar do percentil para tornar o corte mais conservador
    # Ex.: 1.15 eleva o limiar em 15% (mais estrito)
    'criticality_multiplier': 1.15,
    'hidden_dim': 32,     # Neurônios
    'batch_size': 32,
    'epochs': 200,
    'learning_rate': 0.001,
    'weight_decay': 1e-4,
    'dropout': 0.4,
    'cvli_weight': 5.0
}

# --- ROTAÇÃO API ---
GEMINI_KEYS_POOL = [os.getenv(f"GEMINI_KEY_{i}") for i in range(1, 4)]
GEMINI_KEYS_POOL = [k for k in GEMINI_KEYS_POOL if k]

def check_structure():
    for d in [REPORT_DIR, MODEL_DIR, TENSOR_DIR, OUTPUT_DIR]:
        os.makedirs(d, exist_ok=True)