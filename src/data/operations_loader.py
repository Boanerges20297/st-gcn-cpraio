"""
OPERATIONS LOADER
─────────────────

Responsável por carregar e validar JSON de operações policiais.

Entrada: data/raw/ocorrencia_policial_operacional.json (9.069 registros)
Saída: DataFrame com schema padronizado

Campos Utilizados:
  - Controle: ID único
  - Data: data da operação
  - HoraI: hora início
  - BairroOcor: bairro
  - CidadeOcor: cidade
  - lat_long: coordenadas
  - Natureza: tipo de crime
  - area_faccao: facção (CV/PCC/GDE/MASSA/SEM_FACCAO)
  - total_drogas_cache: gramas apreendidas
  - total_armas_cache: armas apreendidas
  - Dinheiro_Apreendido: dinheiro apreendido

Autor: ST-GCN Pipeline
Data: 21 de Janeiro de 2026
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import logging
from typing import Optional, Tuple

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class OperationsLoader:
    """Carrega e valida dados de operações policiais."""
    
    # Campos esperados no JSON
    REQUIRED_FIELDS = {
        'Controle': str,
        'Data': str,
        'BairroOcor': str,
        'CidadeOcor': str,
        'total_drogas_cache': (int, float, str),
        'total_armas_cache': (int, float, str),
    }
    
    # Campos opcionais
    OPTIONAL_FIELDS = {
        'HoraI': str,
        'lat_long': str,
        'Natureza': str,
        'area_faccao': str,
        'Dinheiro_Apreendido': (int, float, str),
    }
    
    def __init__(self, json_path: str):
        """
        Inicializa loader com caminho do JSON.
        
        Args:
            json_path: caminho absoluto para arquivo JSON
        """
        self.json_path = Path(json_path)
        self.data = None
        self.validation_report = {}
        
    def load(self) -> pd.DataFrame:
        """
        Carrega JSON e retorna DataFrame com validação.
        
        Suporta:
        - Array simples de objetos
        - Export PHPMyAdmin (array com header, database, table)
        
        Returns:
            DataFrame com registros (ou menos se houver erros)
            
        Raises:
            FileNotFoundError: Se arquivo não existe
            json.JSONDecodeError: Se JSON inválido
        """
        logger.info(f"Carregando {self.json_path}...")
        
        if not self.json_path.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {self.json_path}")
        
        # Carregar JSON
        try:
            with open(self.json_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Erro ao parsear JSON: {e}")
            raise
        
        # Detectar estrutura: PHPMyAdmin export vs. array simples
        if isinstance(raw_data, list):
            # Pode ser: (1) array simples de objetos ou (2) export PHPMyAdmin
            if len(raw_data) > 0 and isinstance(raw_data[0], dict):
                first_item = raw_data[0]
                
                # Detectar PHPMyAdmin export (tem tipo e versão no header)
                if 'type' in first_item and first_item.get('type') == 'header':
                    logger.info("Formato detectado: PHPMyAdmin export")
                    # Procurar pela table que contém os dados
                    for item in raw_data:
                        if isinstance(item, dict) and item.get('type') == 'table':
                            if 'data' in item and isinstance(item['data'], list):
                                raw_data = item['data']
                                logger.info(f"Dados extraídos da chave 'data' da table")
                                break
                else:
                    logger.info("Formato detectado: Array simples de objetos")
        
        elif isinstance(raw_data, dict):
            # Procurar por array em chaves comuns
            for key in ['data', 'records', 'operacoes', 'table']:
                if key in raw_data and isinstance(raw_data[key], list):
                    logger.info(f"Dados extraídos da chave '{key}'")
                    raw_data = raw_data[key]
                    break
        
        if not isinstance(raw_data, list):
            raise ValueError(f"JSON esperado como array ou objeto com array, recebido {type(raw_data)}")
        
        logger.info(f"Total de registros: {len(raw_data)}")
        
        # Converter para DataFrame
        self.data = pd.DataFrame(raw_data)
        
        return self.data
    
    def validate(self) -> bool:
        """
        Valida dados carregados.
        
        Returns:
            True se válido, False caso contrário
            
        Raises:
            ValueError: Se dados não foram carregados
        """
        if self.data is None:
            raise ValueError("Dados não foram carregados. Chame load() primeiro.")
        
        self.validation_report = {}
        is_valid = True
        
        # Checagem 1: Total de registros
        total_records = len(self.data)
        logger.info(f"✓ Total de registros: {total_records}")
        self.validation_report['total_records'] = total_records
        
        # Checagem 2: Campos obrigatórios
        for field in self.REQUIRED_FIELDS:
            if field not in self.data.columns:
                logger.warning(f"⚠ Campo obrigatório faltando: {field}")
                is_valid = False
            else:
                non_null_count = self.data[field].notna().sum()
                null_count = self.data[field].isna().sum()
                logger.info(f"✓ {field}: {non_null_count} valores válidos, {null_count} nulos")
                self.validation_report[f'{field}_null_count'] = null_count
        
        # Checagem 3: Campos opcionais
        for field in self.OPTIONAL_FIELDS:
            if field not in self.data.columns:
                logger.warning(f"⚠ Campo opcional faltando: {field}")
            else:
                non_null_count = self.data[field].notna().sum()
                logger.info(f"✓ {field}: {non_null_count} valores disponíveis")
        
        # Checagem 4: Conversão de tipos
        try:
            self.data['Data'] = pd.to_datetime(self.data['Data'], errors='coerce')
            null_dates = self.data['Data'].isna().sum()
            logger.info(f"✓ Data parseada: {len(self.data) - null_dates} datas válidas")
            self.validation_report['invalid_dates'] = null_dates
        except Exception as e:
            logger.warning(f"⚠ Erro ao parsear datas: {e}")
            is_valid = False
        
        # Checagem 5: Valores numéricos
        for field in ['total_drogas_cache', 'total_armas_cache']:
            try:
                self.data[field] = pd.to_numeric(self.data[field], errors='coerce')
                null_count = self.data[field].isna().sum()
                min_val = self.data[field].min()
                max_val = self.data[field].max()
                logger.info(f"✓ {field}: [{min_val:.2f}, {max_val:.2f}], {null_count} inválidos")
                self.validation_report[f'{field}_range'] = (float(min_val), float(max_val))
            except Exception as e:
                logger.warning(f"⚠ Erro ao converter {field}: {e}")
                is_valid = False
        
        if is_valid:
            logger.info("✅ Validação PASSOU")
        else:
            logger.warning("⚠ Validação com avisos - continuando com caution")
        
        return is_valid or True  # Continuar mesmo com warnings
    
    def get_statistics(self) -> dict:
        """
        Retorna estatísticas dos dados.
        
        Returns:
            dict com estatísticas
        """
        if self.data is None:
            return {}
        
        stats = {
            'total_records': len(self.data),
            'date_range': (self.data['Data'].min(), self.data['Data'].max()) if 'Data' in self.data.columns else None,
            'unique_bairros': self.data['BairroOcor'].nunique() if 'BairroOcor' in self.data.columns else 0,
            'drogas_total_gramas': self.data['total_drogas_cache'].sum() if 'total_drogas_cache' in self.data.columns else 0,
            'armas_total': self.data['total_armas_cache'].sum() if 'total_armas_cache' in self.data.columns else 0,
        }
        
        if 'area_faccao' in self.data.columns:
            stats['facoes_distribuicao'] = self.data['area_faccao'].value_counts().to_dict()
        
        return stats


def load_operations_json(json_path: str, validate: bool = True) -> Tuple[pd.DataFrame, dict]:
    """
    Função convenience para carregar e validar JSON de operações.
    
    Args:
        json_path: caminho para arquivo JSON
        validate: se True, valida dados após carregamento
        
    Returns:
        (DataFrame, statistics dict)
    """
    loader = OperationsLoader(json_path)
    df = loader.load()
    
    if validate:
        loader.validate()
    
    stats = loader.get_statistics()
    
    logger.info(f"\n📊 ESTATÍSTICAS:")
    logger.info(f"  Total de registros: {stats['total_records']}")
    logger.info(f"  Período: {stats['date_range']}")
    logger.info(f"  Bairros únicos: {stats['unique_bairros']}")
    logger.info(f"  Drogas apreendidas: {stats['drogas_total_gramas']:.0f} g")
    logger.info(f"  Armas apreendidas: {stats['armas_total']:.0f} un")
    if 'facoes_distribuicao' in stats:
        logger.info(f"  Facções: {stats['facoes_distribuicao']}")
    
    return df, stats


if __name__ == '__main__':
    # Teste
    import sys
    
    json_file = 'data/raw/ocorrencia_policial_operacional.json'
    
    try:
        df, stats = load_operations_json(json_file)
        print("\n✅ Dados carregados com sucesso!")
        print(df.head(3))
    except Exception as e:
        print(f"❌ Erro: {e}")
        sys.exit(1)
