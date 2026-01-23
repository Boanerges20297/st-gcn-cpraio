#!/usr/bin/env python3
"""
INTEGRAÇÃO DE TENSORES DE PRODUÇÃO
Converte os tensores .npy do ETL_PRODUCAO_V2 para formato PyTorch
Prepara dados para treinamento do modelo ST-GCN
"""

import sys
import os
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import logging

# --- Setup de Paths e Imports ---
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

import config

# --- Configurar Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# STAGE 1: VALIDAÇÃO E CARREGAMENTO
# ============================================================================

def validate_production_data():
    """Valida se os dados de produção foram gerados corretamente"""
    logger.info("[STAGE 1] Validando dados de produção...")
    
    required_files = [
        config.DATA_PROCESSED / 'tensor_cvli_univariado.npy',
        config.DATA_PROCESSED / 'tensor_multivariado.npy',
        config.DATA_PROCESSED / 'cvli_producao.csv',
        config.DATA_PROCESSED / 'operacional_producao.csv',
        config.DATA_PROCESSED / 'metadata_producao_v2.json'
    ]
    
    missing = [f for f in required_files if not f.exists()]
    if missing:
        logger.error(f"❌ Arquivos faltando: {missing}")
        return False
    
    logger.info("✓ Todos os arquivos de produção encontrados")
    return True

def load_production_tensors():
    """Carrega os tensores numpy do ETL"""
    logger.info("[STAGE 1] Carregando tensores...")
    
    tensors = {
        'cvli_univariado': np.load(config.DATA_PROCESSED / 'tensor_cvli_univariado.npy'),
        'multivariado': np.load(config.DATA_PROCESSED / 'tensor_multivariado.npy'),
        'prisoes': np.load(config.DATA_PROCESSED / 'tensor_prisoes.npy'),
        'apreensoes': np.load(config.DATA_PROCESSED / 'tensor_apreensoes.npy'),
    }
    
    # Validar formatos
    assert tensors['cvli_univariado'].shape == (1472, 121), \
        f"Tensor CVLI shape inválido: {tensors['cvli_univariado'].shape}"
    assert tensors['multivariado'].shape == (1472, 121, 3), \
        f"Tensor multivariado shape inválido: {tensors['multivariado'].shape}"
    
    logger.info(f"✓ Tensores carregados:")
    logger.info(f"  - CVLI univariado: {tensors['cvli_univariado'].shape}")
    logger.info(f"  - Multivariado: {tensors['multivariado'].shape}")
    logger.info(f"  - Prisões: {tensors['prisoes'].shape}")
    logger.info(f"  - Apreensões: {tensors['apreensoes'].shape}")
    
    return tensors

def load_metadata():
    """Carrega metadados do ETL"""
    logger.info("[STAGE 1] Carregando metadados...")
    
    with open(config.DATA_PROCESSED / 'metadata_producao_v2.json', 'r') as f:
        raw_metadata = json.load(f)
    
    # Normalizar formato
    metadata = {
        'periodo_inicio': raw_metadata['periodo'].split(' a ')[0],
        'periodo_fim': raw_metadata['periodo'].split(' a ')[1],
        'num_dias': raw_metadata['dias'],
        'num_bairros': raw_metadata['bairros'],
        'bairro_lista': raw_metadata['bairros_normalizados'],
        'eventos_cvli': raw_metadata['eventos_cvli'],
        'eventos_prisoes': raw_metadata['eventos_prisoes'],
        'eventos_apreensoes': raw_metadata['eventos_apreensoes'],
    }
    
    logger.info(f"✓ Metadados carregados:")
    logger.info(f"  - Período: {metadata['periodo_inicio']} a {metadata['periodo_fim']}")
    logger.info(f"  - Bairros: {metadata['num_bairros']}")
    logger.info(f"  - Dias: {metadata['num_dias']}")
    
    return metadata

# ============================================================================
# STAGE 2: PREPARAR DADOS PARA TREINO (REGIÃO POR REGIÃO)
# ============================================================================

def split_by_region(cvli_df, metadata):
    """
    Divide os dados de CVLI em CAPITAL, RMF e INTERIOR
    baseado na cidade_norm e bairro_norm
    """
    logger.info("[STAGE 2] Dividindo dados por região...")
    
    # Mapeamento de cidade para região
    cidade_to_regiao = {}
    
    # Carregar mapeamento do metadata
    if 'bairro_mapping' in metadata:
        for bairro, info in metadata['bairro_mapping'].items():
            if 'cidade_norm' in info:
                cidade = info['cidade_norm']
                # Lógica de classificação
                if cidade == 'Fortaleza':
                    cidade_to_regiao[cidade] = 'CAPITAL'
                elif cidade in ['Caucaia', 'Maranguape', 'Aquiraz', 'Maracanaú', 'Pacajus']:
                    cidade_to_regiao[cidade] = 'RMF'
                else:
                    cidade_to_regiao[cidade] = 'INTERIOR'
    
    # Agrupar
    regional_splits = {
        'CAPITAL': [],
        'RMF': [],
        'INTERIOR': []
    }
    
    for idx, row in cvli_df.iterrows():
        cidade = row.get('cidade_norm', 'Fortaleza')
        regiao = cidade_to_regiao.get(cidade, 'CAPITAL')
        regional_splits[regiao].append(idx)
    
    logger.info(f"✓ Distribuição por região:")
    for regiao, indices in regional_splits.items():
        logger.info(f"  - {regiao}: {len(indices)} eventos")
    
    return regional_splits

def build_regional_tensors(tensor_multivariado, cvli_df, regional_splits, metadata):
    """
    Cria tensores regionalizados mantendo dimensões temporais
    Retorna um tensor por região com apenas bairros daquela região
    """
    logger.info("[STAGE 2] Construindo tensores regionalizados...")
    
    regional_tensors = {}
    
    # Obter lista ordenada de bairros do metadata
    bairro_list = metadata.get('bairro_lista', [])
    
    # Você pode filtrar por região aqui se necessário
    # Por enquanto, vamos manter o tensor completo pois graph_builder faz a filtragem
    
    regional_tensors['COMPLETO'] = tensor_multivariado
    
    logger.info(f"✓ Tensor completo preparado: {tensor_multivariado.shape}")
    
    return regional_tensors

# ============================================================================
# STAGE 3: CONVERTER PARA PYTORCH E CRIAR DATASETS
# ============================================================================

def create_pytorch_datasets(tensor_dict, metadata):
    """
    Converte arrays numpy para tensores PyTorch
    Cria estrutura de dataset compatível com graph_builder
    """
    logger.info("[STAGE 3] Convertendo para PyTorch tensors...")
    
    # Datas
    periodo_inicio = pd.to_datetime(metadata['periodo_inicio'])
    num_dias = metadata['num_dias']
    all_dates = pd.date_range(periodo_inicio, periods=num_dias, freq='D')
    
    # Mapa de bairros
    bairro_lista = metadata['bairro_lista']
    
    # Converter para torch
    X_tensor = torch.from_numpy(tensor_dict['COMPLETO']).float()
    
    # Criar dataset
    dataset = {
        'X': X_tensor,
        'edge_index': None,  # Será criado pelo graph_builder
        'nodes': bairro_lista,
        'dates': all_dates,
        'features': ['CVLI', 'Prisoes', 'Apreensoes'],
        'metadata': {
            'num_days': num_dias,
            'num_nodes': len(bairro_lista),
            'num_features': 3,
            'periode': f"{metadata['periodo_inicio']} a {metadata['periodo_fim']}"
        }
    }
    
    logger.info(f"✓ Dataset PyTorch criado:")
    logger.info(f"  - Shape X: {dataset['X'].shape}")
    logger.info(f"  - Nodes: {len(dataset['nodes'])}")
    logger.info(f"  - Features: {dataset['features']}")
    
    return dataset

# ============================================================================
# STAGE 4: SALVAMENTO
# ============================================================================

def save_pytorch_dataset(dataset, output_path):
    """Salva dataset em formato PyTorch"""
    logger.info(f"[STAGE 4] Salvando dataset PyTorch...")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Remover edge_index se None
    if dataset['edge_index'] is None:
        del dataset['edge_index']
    
    torch.save(dataset, output_path)
    logger.info(f"✓ Dataset salvo em: {output_path}")
    logger.info(f"  - Tamanho: {output_path.stat().st_size / 1024 / 1024:.1f} MB")

def create_integration_report(metadata, tensor_dict):
    """Gera relatório de integração"""
    logger.info("[STAGE 4] Gerando relatório de integração...")
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'status': 'INTEGRAÇÃO_CONCLUÍDA',
        'etl_source': 'etl_producao_v2.py',
        'input_tensors': {
            'cvli_univariado': (1472, 121),
            'multivariado': (1472, 121, 3),
            'prisoes': (1472, 121),
            'apreensoes': (1472, 121),
        },
        'output_dataset': {
            'shape': tuple(tensor_dict['COMPLETO'].shape),
            'dtype': str(tensor_dict['COMPLETO'].dtype),
            'nodes': metadata['num_bairros'],
            'dates': metadata['num_dias'],
            'features': 3,
        },
        'periodo': {
            'inicio': metadata['periodo_inicio'],
            'fim': metadata['periodo_fim'],
            'dias': metadata['num_dias'],
        },
        'proximos_passos': [
            'Executar: python src/graph_builder.py (para criar topologia de grafo)',
            'Executar: python src/trainer.py (para treinar modelo)',
            'Executar: python src/predict.py (para fazer previsões)',
        ]
    }
    
    report_path = config.DATA_PROCESSED / 'INTEGRACAO_PRODUCAO_RELATORIO.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"✓ Relatório salvo em: {report_path}")
    
    return report

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*80)
    print("INTEGRAÇÃO DE TENSORES DE PRODUÇÃO PARA TREINAMENTO")
    print("="*80)
    
    # Stage 1: Validação e Carregamento
    if not validate_production_data():
        logger.error("❌ Validação falhou. Rode o ETL primeiro.")
        return False
    
    tensors = load_production_tensors()
    metadata = load_metadata()
    
    # Stage 2: Preparar Dados por Região
    cvli_df = pd.read_csv(config.DATA_PROCESSED / 'cvli_producao.csv')
    regional_splits = split_by_region(cvli_df, metadata)
    regional_tensors = build_regional_tensors(tensors['multivariado'], cvli_df, regional_splits, metadata)
    
    # Stage 3: Converter para PyTorch
    dataset = create_pytorch_datasets(regional_tensors, metadata)
    
    # Stage 4: Salvar
    dataset_path = config.TENSOR_DIR / 'dataset_producao_v2.pt'
    config.TENSOR_DIR.mkdir(parents=True, exist_ok=True)
    save_pytorch_dataset(dataset, dataset_path)
    
    # Gerar Relatório
    report = create_integration_report(metadata, regional_tensors)
    
    print("\n" + "="*80)
    print("✅ INTEGRAÇÃO CONCLUÍDA COM SUCESSO")
    print("="*80)
    print(f"\nDataset salvo em: {dataset_path}")
    print(f"Relatório em: {config.DATA_PROCESSED / 'INTEGRACAO_PRODUCAO_RELATORIO.json'}")
    print("\n📋 Próximos passos:")
    for step in report['proximos_passos']:
        print(f"  → {step}")
    print("\n")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
