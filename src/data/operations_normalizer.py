"""
OPERATIONS NORMALIZER
─────────────────────

Responsável por normalizar dados de operações policiais.

Entrada: DataFrame de operações (raw)
Processamento:
  1. Parse de tipos (Data → datetime, drogas/armas → float)
  2. Mapeamento de bairros para IDs (0-387)
  3. Normalização Min-Max dos valores numéricos
  4. Agregação temporal (diária)

Saída: data/processed/prisoes_normalized.parquet

Schema de Saída:
  - bairro_id: int (0-387)
  - data: datetime
  - operacoes_diarias: int (count)
  - drogas_gramas_total_norm: float [0, 1]
  - armas_total_norm: float [0, 1]
  - dinheiro_total_reais_norm: float [0, 1]
  - natureza_list: str (lista de tipos de crime)
  - faccoes_list: str (lista de facções)

Autor: ST-GCN Pipeline
Data: 21 de Janeiro de 2026
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import logging
from typing import Dict, Tuple, Optional
import json

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class OperationsNormalizer:
    """Normaliza dados de operações policiais."""
    
    def __init__(self, 
                 bairro_mapping: Optional[Dict[str, int]] = None,
                 percentile_drogas: float = 99.0,
                 percentile_armas: float = 99.0,
                 percentile_dinheiro: float = 99.0):
        """
        Inicializa normalizador.
        
        Args:
            bairro_mapping: dict {nome_bairro → bairro_id}. Se None, constrói automaticamente.
            percentile_drogas: percentil para usar como max na normalização (para lidar outliers)
            percentile_armas: idem armas
            percentile_dinheiro: idem dinheiro
        """
        self.bairro_mapping = bairro_mapping
        self.percentile_drogas = percentile_drogas
        self.percentile_armas = percentile_armas
        self.percentile_dinheiro = percentile_dinheiro
        
        # Normalización será calculada dinamicamente
        self.drogas_max = None
        self.armas_max = None
        self.dinheiro_max = None
        
        self.normalization_params = {}
    
    def _build_bairro_mapping(self, df: pd.DataFrame) -> Dict[str, int]:
        """
        Constrói mapeamento bairro → ID automático.
        
        Args:
            df: DataFrame com coluna 'BairroOcor'
            
        Returns:
            dict {bairro_nome → bairro_id}
        """
        unique_bairros = df['BairroOcor'].unique()
        unique_bairros = sorted([b.strip().upper() for b in unique_bairros if pd.notna(b)])
        
        logger.info(f"Mapeando {len(unique_bairros)} bairros únicos para IDs 0-{len(unique_bairros)-1}")
        
        mapping = {bairro: idx for idx, bairro in enumerate(unique_bairros)}
        
        logger.info(f"Mapeamento criado:")
        for bairro, idx in list(mapping.items())[:5]:
            logger.info(f"  {bairro} → {idx}")
        logger.info(f"  ... e {len(mapping) - 5} mais")
        
        return mapping
    
    def normalize(self, df: pd.DataFrame, validate: bool = True) -> pd.DataFrame:
        """
        Normaliza DataFrame de operações.
        
        Args:
            df: DataFrame raw de operações
            validate: se True, valida resultado
            
        Returns:
            DataFrame normalizado
        """
        logger.info("Iniciando normalização de operações...")
        
        df = df.copy()
        
        # Etapa 1: Construir ou usar mapeamento de bairros
        if self.bairro_mapping is None:
            self.bairro_mapping = self._build_bairro_mapping(df)
        
        # Etapa 2: Preparar dados numéricos
        logger.info("Convertendo tipos...")
        
        # Data
        df['Data'] = pd.to_datetime(df['Data'], errors='coerce')
        invalid_dates = df['Data'].isna().sum()
        if invalid_dates > 0:
            logger.warning(f"⚠ {invalid_dates} datas inválidas, removendo...")
            df = df.dropna(subset=['Data'])
        
        # Drogas (gramas)
        df['total_drogas_cache'] = pd.to_numeric(df['total_drogas_cache'], errors='coerce').fillna(0)
        
        # Armas
        df['total_armas_cache'] = pd.to_numeric(df['total_armas_cache'], errors='coerce').fillna(0)
        
        # Dinheiro
        if 'Dinheiro_Apreendido' in df.columns:
            df['Dinheiro_Apreendido'] = pd.to_numeric(df['Dinheiro_Apreendido'], errors='coerce').fillna(0)
        else:
            df['Dinheiro_Apreendido'] = 0.0
        
        # Etapa 3: Calcular max para normalização (usando percentil para robusto)
        logger.info("Calculando parâmetros de normalização...")
        
        self.drogas_max = df['total_drogas_cache'].quantile(self.percentile_drogas / 100.0)
        self.armas_max = df['total_armas_cache'].quantile(self.percentile_armas / 100.0)
        self.dinheiro_max = df['Dinheiro_Apreendido'].quantile(self.percentile_dinheiro / 100.0)
        
        logger.info(f"  Drogas max ({self.percentile_drogas}th percentil): {self.drogas_max:.2f} g")
        logger.info(f"  Armas max ({self.percentile_armas}th percentil): {self.armas_max:.2f}")
        logger.info(f"  Dinheiro max ({self.percentile_dinheiro}th percentil): {self.dinheiro_max:.2f} R$")
        
        # Etapa 4: Normalizar (Min-Max com clipping)
        logger.info("Aplicando normalização Min-Max...")
        
        df['drogas_norm'] = (df['total_drogas_cache'] / self.drogas_max).clip(0, 1)
        df['armas_norm'] = (df['total_armas_cache'] / self.armas_max).clip(0, 1)
        df['dinheiro_norm'] = (df['Dinheiro_Apreendido'] / self.dinheiro_max).clip(0, 1)
        
        # Etapa 5: Mapear bairro para ID
        logger.info("Mapeando bairros para IDs...")
        
        df['BairroOcor'] = df['BairroOcor'].str.strip().str.upper()
        df['bairro_id'] = df['BairroOcor'].map(self.bairro_mapping)
        
        unmapped = df['bairro_id'].isna().sum()
        if unmapped > 0:
            logger.warning(f"⚠ {unmapped} bairros não mapeáveis, removendo...")
            df = df.dropna(subset=['bairro_id'])
        
        df['bairro_id'] = df['bairro_id'].astype(int)
        
        # Etapa 6: Agregação Temporal (diária)
        logger.info("Agregando por data e bairro...")
        
        agg_dict = {
            'drogas_norm': 'sum',  # Soma de todas as operações naquele dia
            'armas_norm': 'sum',
            'dinheiro_norm': 'sum',
            'Controle': 'count',  # Número de operações
        }
        
        # Se houver natureza e facção, agregá-las também
        if 'Natureza' in df.columns:
            df['Natureza'] = df['Natureza'].fillna('DESCONHECIDA')
            agg_dict['Natureza'] = lambda x: '|'.join(x.unique())
        
        if 'area_faccao' in df.columns:
            df['area_faccao'] = df['area_faccao'].fillna('SEM_FACCAO')
            agg_dict['area_faccao'] = lambda x: '|'.join(x.unique())
        
        # Agrupar por (Data, bairro_id)
        df_agg = df.groupby([pd.Grouper(key='Data', freq='D'), 'bairro_id']).agg(agg_dict).reset_index()
        
        # Renomear coluna de contagem
        df_agg = df_agg.rename(columns={'Controle': 'operacoes_diarias'})
        
        logger.info(f"✓ Agregação completa: {len(df_agg)} registros (data, bairro)")
        
        # Etapa 7: Garantir intervalo de datas contínuo
        logger.info("Preenchendo datas faltantes com zeros...")
        
        date_range = pd.date_range(df_agg['Data'].min(), df_agg['Data'].max(), freq='D')
        bairro_ids = df_agg['bairro_id'].unique()
        
        index_completo = pd.MultiIndex.from_product([date_range, bairro_ids], names=['Data', 'bairro_id'])
        df_agg = df_agg.set_index(['Data', 'bairro_id']).reindex(index_completo).reset_index()
        
        # Preencher zeros onde não havia operações
        for col in ['operacoes_diarias', 'drogas_norm', 'armas_norm', 'dinheiro_norm']:
            df_agg[col] = df_agg[col].fillna(0)
        
        # Natureza e Facção preenchidas com string vazia
        if 'Natureza' in df_agg.columns:
            df_agg['Natureza'] = df_agg['Natureza'].fillna('')
        if 'area_faccao' in df_agg.columns:
            df_agg['area_faccao'] = df_agg['area_faccao'].fillna('')
        
        logger.info(f"✓ Intervalo preenchido: {len(df_agg)} registros (com zeros)")
        
        # Etapa 8: Renomear e reorganizar colunas
        df_result = df_agg[['Data', 'bairro_id', 'operacoes_diarias', 
                             'drogas_norm', 'armas_norm', 'dinheiro_norm']].copy()
        
        df_result = df_result.rename(columns={
            'Data': 'data',
            'drogas_norm': 'drogas_gramas_total_norm',
            'armas_norm': 'armas_total_norm',
            'dinheiro_norm': 'dinheiro_total_reais_norm',
        })
        
        # Adicionar colunas de natureza e facção se disponíveis
        if 'Natureza' in df_agg.columns:
            df_result['natureza_list'] = df_agg['Natureza']
        if 'area_faccao' in df_agg.columns:
            df_result['faccoes_list'] = df_agg['area_faccao']
        
        # Etapa 9: Validação (opcional)
        if validate:
            self._validate_result(df_result)
        
        # Armazenar parâmetros para posterioridade (para normalizar dados futuros)
        self.normalization_params = {
            'drogas_max': float(self.drogas_max),
            'armas_max': float(self.armas_max),
            'dinheiro_max': float(self.dinheiro_max),
            'bairro_mapping': self.bairro_mapping,
            'date_range': (df_result['data'].min().isoformat(), df_result['data'].max().isoformat()),
        }
        
        logger.info("✅ Normalização concluída!")
        
        return df_result
    
    def _validate_result(self, df: pd.DataFrame):
        """
        Valida DataFrame normalizado.
        
        Args:
            df: DataFrame normalizado
        """
        logger.info("Validando resultado...")
        
        # Check 1: Sem NaN
        nan_count = df.isna().sum().sum()
        logger.info(f"✓ Total de NaN: {nan_count}")
        
        # Check 2: Valores em [0, 1]
        norm_cols = ['drogas_gramas_total_norm', 'armas_total_norm', 'dinheiro_total_reais_norm']
        for col in norm_cols:
            if col in df.columns:
                min_val = df[col].min()
                max_val = df[col].max()
                if min_val < 0 or max_val > 1:
                    logger.warning(f"⚠ {col}: [{min_val}, {max_val}] - FORA do intervalo [0, 1]!")
                else:
                    logger.info(f"✓ {col}: [{min_val}, {max_val}]")
        
        # Check 3: Operações >= 0
        logger.info(f"✓ operacoes_diarias: min={df['operacoes_diarias'].min()}, max={df['operacoes_diarias'].max()}")
        
        # Check 4: bairro_id válidos
        logger.info(f"✓ bairro_id: {len(df['bairro_id'].unique())} bairros únicos")
        
        logger.info("✅ Validação concluída!")


def normalize_operations(df_raw: pd.DataFrame, 
                         output_path: str,
                         bairro_mapping: Optional[Dict[str, int]] = None) -> Tuple[pd.DataFrame, dict]:
    """
    Função convenience para normalizar e salvar operações.
    
    Args:
        df_raw: DataFrame raw de operações
        output_path: caminho para salvar parquet normalizado
        bairro_mapping: mapeamento opcional de bairros
        
    Returns:
        (DataFrame normalizado, parâmetros de normalização)
    """
    normalizer = OperationsNormalizer(bairro_mapping=bairro_mapping)
    df_norm = normalizer.normalize(df_raw, validate=True)
    
    # Salvar resultado
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Salvando em {output_path}...")
    df_norm.to_parquet(output_path, compression='gzip', index=False)
    
    # Salvar parâmetros de normalização
    params_path = output_path.parent / 'normalization_params.json'
    with open(params_path, 'w') as f:
        json.dump(normalizer.normalization_params, f, indent=2, default=str)
    logger.info(f"Parâmetros salvos em {params_path}")
    
    return df_norm, normalizer.normalization_params


if __name__ == '__main__':
    # Teste
    from .operations_loader import load_operations_json
    
    json_file = 'data/raw/ocorrencia_policial_operacional.json'
    output_file = 'data/processed/prisoes_normalized.parquet'
    
    try:
        # Carregar dados raw
        df_raw, _ = load_operations_json(json_file)
        
        # Normalizar
        df_norm, params = normalize_operations(df_raw, output_file)
        
        print("\n✅ Normalização concluída com sucesso!")
        print(df_norm.head())
        print(f"\nParâmetros: {json.dumps(params, indent=2, default=str)}")
        
    except Exception as e:
        print(f"❌ Erro: {e}")
        import traceback
        traceback.print_exc()
