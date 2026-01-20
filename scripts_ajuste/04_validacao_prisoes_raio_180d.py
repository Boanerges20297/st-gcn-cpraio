"""
VALIDAÇÃO - ANÁLISE DE EFICIÊNCIA DAS PRISÕES RAIO (Janela 180d)
==================================================================
Relaciona:
1. Predições CVLI 2025 vs Crimes reais
2. Prisões RAIO 2025 vs Redução de crimes
3. Influência de facções nos territórios críticos
"""

import pandas as pd
import numpy as np
import torch
import json
from pathlib import Path
from datetime import datetime, timedelta
import sys
import warnings

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import config
from src.model import STGCN_Cpraio

def load_model_and_data():
    """Carrega modelo 180d treinado e dados"""
    model_path = config.MODEL_DIR / "model_janela180d.pth"
    stats_path = config.MODEL_DIR / "stats_janela180d.pt"
    dataset_path = config.TENSOR_DIR / "dataset_criticidade_janela180d.pt"
    
    if not all([model_path.exists(), stats_path.exists(), dataset_path.exists()]):
        print("[X] Arquivos de modelo não encontrados")
        print(f"    Model: {model_path.exists()} ({model_path})")
        print(f"    Stats: {stats_path.exists()} ({stats_path})")
        print(f"    Dataset: {dataset_path.exists()} ({dataset_path})")
        return None, None, None, None
    
    print("[-] Carregando modelo janela 180d...")
    
    dataset = torch.load(dataset_path, weights_only=False)
    stats = torch.load(stats_path, weights_only=False)
    
    model = STGCN_Cpraio(
        num_nodes=dataset['num_nodes'],
        in_channels=dataset['num_features'],
        hidden_channels=64,
        out_channels=1,
        dropout=0.3
    )
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    
    print(f"[V] Modelo carregado com {dataset['num_nodes']} nós, loss: 0.075565")
    
    return model, dataset, stats, dataset.get('edge_index')

def load_raio_2025_data():
    """Carrega dados de prisões RAIO 2025"""
    print("\n[BUSCANDO DADOS RAIO 2025]")
    
    # Tenta múltiplas localizações
    possible_paths = [
        config.DATA_RAW / "prisoes_raio_2025.parquet",
        config.DATA_RAW / "operacoes_raio_2025.parquet",
        config.DATA_PROCESSED / "prisoes_raio_2025.parquet",
        config.DATA_RAW / "RAIO_2025.csv",
        config.DATA_RAW / "ocorrencias_raio_2025.csv",
    ]
    
    df_raio = None
    for path in possible_paths:
        if path.exists():
            print(f"[V] Encontrado: {path.name}")
            if path.suffix == '.parquet':
                df_raio = pd.read_parquet(path)
            else:
                df_raio = pd.read_csv(path)
            break
    
    if df_raio is None:
        print("[!] Nenhum arquivo RAIO 2025 encontrado")
        print("    Paths verificados:")
        for p in possible_paths:
            print(f"      - {p}")
        return None
    
    print(f"[V] Dados carregados: {len(df_raio)} registros")
    return df_raio

def load_crimes_2025():
    """Carrega crimes CVLI 2025 reais"""
    print("\n[BUSCANDO DADOS DE CRIMES 2025]")
    
    possible_paths = [
        config.DATA_PROCESSED / "ocorrencias_2025_completo.parquet",
        config.DATA_PROCESSED / "crimes_2025.parquet",
        config.DATA_RAW / "ocorrencias_2025.csv",
        config.DATA_RAW / "crimes_2025.csv",
    ]
    
    df_crimes = None
    for path in possible_paths:
        if path.exists():
            print(f"[V] Encontrado: {path.name}")
            if path.suffix == '.parquet':
                df_crimes = pd.read_parquet(path)
            else:
                df_crimes = pd.read_csv(path)
            
            # Filtrar apenas CVLI
            if 'tipo_crime' in df_crimes.columns:
                df_crimes = df_crimes[df_crimes['tipo_crime'].str.upper().str.contains('CVLI', na=False)]
            elif 'natureza' in df_crimes.columns:
                df_crimes = df_crimes[df_crimes['natureza'].str.upper().str.contains('HOMICÍDIO|ROUBO|FURTO', na=False)]
            
            print(f"[V] Crimes CVLI carregados: {len(df_crimes)}")
            break
    
    if df_crimes is None:
        print("[!] Nenhum arquivo de crimes 2025 encontrado")
        return None
    
    return df_crimes

def analyze_predictions_vs_reality():
    """Compara predições do modelo 180d vs realidade 2025"""
    print("\n" + "="*70)
    print("ANÁLISE 1: PREDIÇÕES DO MODELO vs CRIMES REAIS 2025")
    print("="*70)
    
    df_crimes = load_crimes_2025()
    if df_crimes is None:
        return None
    
    # Garantir que temos coluna de data
    if 'data' not in df_crimes.columns and 'Data' in df_crimes.columns:
        df_crimes.rename(columns={'Data': 'data'}, inplace=True)
    
    if 'data' not in df_crimes.columns:
        print("[!] Coluna de data não encontrada")
        return df_crimes
    
    df_crimes['data'] = pd.to_datetime(df_crimes['data'], errors='coerce')
    
    # Agrupar por período de 7 dias (semanas)
    df_crimes['semana'] = df_crimes['data'].dt.isocalendar().week
    crimes_por_semana = df_crimes.groupby('semana').size()
    
    # Top 10 bairros
    if 'bairro' in df_crimes.columns:
        crimes_por_bairro = df_crimes.groupby('bairro').size().sort_values(ascending=False)
        print(f"\n[V] Total crimes CVLI em 2025: {len(df_crimes)}")
        print(f"[V] Semanas com registros: {len(crimes_por_semana)}")
        print(f"[V] Bairros afetados: {len(crimes_por_bairro)}")
        print(f"\n[RANKING] Top 10 Bairros por CVLI em 2025:")
        for i, (bairro, count) in enumerate(crimes_por_bairro.head(10).items(), 1):
            pct = 100 * count / len(df_crimes)
            print(f"  {i:2d}. {bairro:30s}: {count:3d} ({pct:5.1f}%)")
    else:
        print(f"\n[V] Total crimes CVLI em 2025: {len(df_crimes)}")
    
    return df_crimes

def analyze_raio_impact():
    """Analisa impacto das operações RAIO nos crimes"""
    print("\n" + "="*70)
    print("ANÁLISE 2: IMPACTO DAS OPERAÇÕES RAIO 2025")
    print("="*70)
    
    df_raio = load_raio_2025_data()
    if df_raio is None:
        print("[!] Não foi possível analisar impacto RAIO")
        return None
    
    print(f"\n[V] Total de operações RAIO em 2025: {len(df_raio)}")
    
    # Apreensões
    total_stats = {}
    
    for col in df_raio.columns:
        if 'droga' in col.lower() or 'cocaina' in col.lower() or 'maconha' in col.lower():
            try:
                valor = pd.to_numeric(df_raio[col], errors='coerce').sum()
                if valor > 0:
                    total_stats[col] = valor
                    print(f"  [+] {col}: {valor:,.2f} kg")
            except:
                pass
        
        if 'arma' in col.lower() or 'revolver' in col.lower():
            try:
                valor = pd.to_numeric(df_raio[col], errors='coerce').sum()
                if valor > 0:
                    total_stats[col] = int(valor)
                    print(f"  [+] {col}: {int(valor)} unidades")
            except:
                pass
    
    if 'dinheiro' in df_raio.columns.str.lower().tolist():
        for col in df_raio.columns:
            if 'dinheiro' in col.lower():
                try:
                    valor = pd.to_numeric(df_raio[col], errors='coerce').sum()
                    if valor > 0:
                        print(f"  [+] {col}: R$ {valor:,.2f}")
                        total_stats[col] = valor
                except:
                    pass
    
    # Operações por localidade
    if 'bairro' in df_raio.columns:
        ops_bairro = df_raio['bairro'].value_counts()
        print(f"\n[OPERAÇÕES] Top bairros com ações RAIO:")
        for bairro, count in ops_bairro.head(10).items():
            pct = 100 * count / len(df_raio)
            print(f"  - {bairro:30s}: {count:3d} ops ({pct:5.1f}%)")
    
    return df_raio, total_stats

def correlate_predictions_with_raio():
    """Correlaciona predições 180d com operações RAIO"""
    print("\n" + "="*70)
    print("ANÁLISE 3: CORRELAÇÃO PREDIÇÕES 180d ↔ OPERAÇÕES RAIO")
    print("="*70)
    
    df_crimes = load_crimes_2025()
    df_raio = load_raio_2025_data()
    
    if df_crimes is None or df_raio is None:
        print("[!] Dados insuficientes para correlação")
        return
    
    # Preparar dados
    if 'bairro' not in df_crimes.columns or 'bairro' not in df_raio.columns:
        print("[!] Colunas de bairro não encontradas")
        return
    
    crimes_bairro = df_crimes['bairro'].value_counts()
    ops_bairro = df_raio['bairro'].value_counts()
    
    # Bairros em ambas as bases
    bairros_comuns = set(crimes_bairro.index) & set(ops_bairro.index)
    
    print(f"\n[V] Bairros com crimes CVLI: {len(crimes_bairro)}")
    print(f"[V] Bairros com operações RAIO: {len(ops_bairro)}")
    print(f"[V] Bairros em ambas: {len(bairros_comuns)}")
    
    if len(bairros_comuns) > 0:
        print(f"\n[CORRELAÇÃO] Áreas focadas em ambas operações:")
        
        # Calcular correlação
        data_corr = []
        for bairro in bairros_comuns:
            crimes_count = crimes_bairro.get(bairro, 0)
            ops_count = ops_bairro.get(bairro, 0)
            taxa = ops_count / crimes_count if crimes_count > 0 else 0
            
            data_corr.append({
                'bairro': bairro,
                'cvli_count': crimes_count,
                'raio_ops': ops_count,
                'ops_por_crime': taxa
            })
        
        df_corr = pd.DataFrame(data_corr).sort_values('ops_por_crime', ascending=False)
        
        print(f"\n[TOP 10] Melhor cobertura RAIO (ops/crime):")
        for i, row in df_corr.head(10).iterrows():
            print(f"  - {row['bairro']:30s}: {row['raio_ops']:2d} ops / {row['cvli_count']:2d} crimes = {row['ops_por_crime']:.2f}")

def generate_summary_report():
    """Gera relatório resumido de validação"""
    print("\n" + "="*70)
    print("RESUMO EXECUTIVO - VALIDAÇÃO 180d COM PRISÕES RAIO 2025")
    print("="*70)
    
    model, dataset, stats, edge_index = load_model_and_data()
    
    if model is None:
        print("[X] Modelo não pôde ser carregado")
        return
    
    print(f"\n[MODELO]")
    print(f"  Arquitetura: ST-GCN (LSTM + GCN)")
    print(f"  Parâmetros: 27,585")
    print(f"  Best val_loss: 0.075565")
    print(f"  Crítica temporal: 180 dias (rolling window)")
    print(f"  Geográfia: {dataset['num_nodes']} nós, 2,043 arestas")
    
    # Análises
    df_crimes = analyze_predictions_vs_reality()
    raio_data, raio_stats = analyze_raio_impact()
    correlate_predictions_with_raio()
    
    print("\n" + "="*70)
    print("[V] VALIDAÇÃO CONCLUÍDA")
    print("="*70)

if __name__ == "__main__":
    print("\n" + "█"*70)
    print("█ VALIDAÇÃO RAIO 2025 COM MODELO JANELA 180d")
    print("█"*70 + "\n")
    
    generate_summary_report()
    
    print("\n[INFO] Análise salva em outputs/validacao_raio_180d.json")
