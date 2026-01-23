#!/usr/bin/env python3
"""
ADAPTER: Integrar novos dados do modelo ST-GCN com dinâmica de facções
ao dashboard e APIs existentes
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime, timedelta

# Caminhos
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

class DataAdapter:
    """
    Adapter que sincroniza dados do modelo ST-GCN com a aplicação Flask
    
    Fornece:
    - Predições 210 dias (tensor novo)
    - Análise de dinâmica de facções
    - Mapeamento para APIs existentes
    """
    
    def __init__(self):
        self.pred_file = OUTPUT_DIR / "predicoes_cvli.csv"
        self.tensor_file = DATA_PROCESSED / "tensor_cvli_prisoes_faccoes.npy"
        self.faccoes_analysis = DATA_PROCESSED / "analise_movimentacao_faccoes.json"
        self.metadata = DATA_PROCESSED / "metadata_producao_v2.json"
        
        self.df_pred = None
        self.tensor = None
        self.faccoes = None
        self.meta = None
        self._load_all()
    
    def _load_all(self):
        """Carregar todos os dados necessários"""
        # Predições
        if self.pred_file.exists():
            self.df_pred = pd.read_csv(self.pred_file)
            print(f"✅ Predições carregadas: {len(self.df_pred)} bairros")
        else:
            print(f"❌ Predições não encontradas: {self.pred_file}")
        
        # Tensor
        if self.tensor_file.exists():
            self.tensor = np.load(self.tensor_file)
            print(f"✅ Tensor carregado: {self.tensor.shape}")
        else:
            print(f"❌ Tensor não encontrado: {self.tensor_file}")
        
        # Facções
        if self.faccoes_analysis.exists():
            with open(self.faccoes_analysis, 'r', encoding='utf-8') as f:
                self.faccoes = json.load(f)
            print(f"✅ Análise de facções carregada")
        else:
            print(f"⚠️  Análise de facções não encontrada: {self.faccoes_analysis}")
        
        # Metadados
        if self.metadata.exists():
            with open(self.metadata, 'r', encoding='utf-8') as f:
                self.meta = json.load(f)
            print(f"✅ Metadados carregados")
        else:
            print(f"⚠️  Metadados não encontrados: {self.metadata}")
    
    def get_top_bairros(self, top_n=15):
        """Retornar top N bairros por CVLI predito"""
        if self.df_pred is None:
            return []
        
        top_df = self.df_pred.nlargest(top_n, 'cvli_predito')
        max_cvli = self.df_pred['cvli_predito'].max()
        
        result = []
        for _, row in top_df.iterrows():
            # Calcular score de risco (0-100)
            score_risco = (row['cvli_predito'] / max_cvli) * 100 if max_cvli > 0 else 0
            
            # Mapear região baseado em lista de bairros conhecidos
            regioes_mapping = {
                'CAPITAL': [
                    'Aldeota', 'Meireles', 'Praia de Iracema', 'Benfica', 'Messejana',
                    'Papicu', 'Cocó', 'Mucuripe', 'Cais do Porto', 'Centro'
                ],
                'PERIFERIA': [
                    'Barra Do Ceará', 'Jangurussu', 'Granja Lisboa', 'Barroso',
                    'Canindezinho', 'Bom Jardim', 'Antônio Bezerra', 'Bonsucesso'
                ],
                'INTERIOR': [
                    'Maracanaú', 'Itaitinga', 'Caucaia', 'Pacatuba'
                ]
            }
            
            regiao = 'CAPITAL'  # padrão
            for reg, bairros in regioes_mapping.items():
                if row['bairro'] in bairros:
                    regiao = reg
                    break
            
            result.append({
                'bairro': row['bairro'],
                'cvli_predito': float(row['cvli_predito']),
                'prob_mudanca': float(row.get('prob_mudanca', 0)),
                'volatilidade': float(row.get('volatilidade', 0)),
                'score_risco': float(score_risco),
                'regiao': regiao,
                'cidade': 'Fortaleza' if regiao == 'CAPITAL' else 'RMF'
            })
        
        return result
    
    def get_bairro_info(self, bairro_name):
        """Informações detalhadas de um bairro"""
        if self.df_pred is None:
            return {}
        
        bairro_data = self.df_pred[self.df_pred['bairro'].str.lower() == bairro_name.lower()]
        if bairro_data.empty:
            return {}
        
        row = bairro_data.iloc[0]
        
        # Calcular índice de risco (0-100)
        cvli_score = (row['cvli_predito'] / self.df_pred['cvli_predito'].max()) * 100
        
        return {
            'bairro': row['bairro'],
            'cvli_predito': float(row['cvli_predito']),
            'prob_mudanca': float(row.get('prob_mudanca', 0)),
            'volatilidade': float(row.get('volatilidade', 0)),
            'score_risco': float(cvli_score),
            'horizonte': '210 dias (23/01/2026 a 21/08/2026)'
        }
    
    def get_timeline(self):
        """Timeline histórica de CVLI e mudanças territoriais"""
        if self.tensor is None or self.meta is None:
            return []
        
        # Tensor shape: (1472, 121, 7)
        # Feature 0: CVLI, Feature 3: Mudança territorial
        
        timeline = []
        inicio = datetime.strptime(self.meta.get('periodo_inicio', '2022-01-01'), '%Y-%m-%d')
        
        # Agregar por dia (média em todos os bairros)
        cvli_por_dia = np.mean(self.tensor[:, :, 0], axis=1)  # Média CVLI/dia
        mudanca_por_dia = np.mean(self.tensor[:, :, 3], axis=1)  # Média mudanças/dia
        
        for day, (cvli, mudanca) in enumerate(zip(cvli_por_dia, mudanca_por_dia)):
            data = (inicio + timedelta(days=day)).strftime('%Y-%m-%d')
            timeline.append({
                'data': data,
                'cvli_medio': float(cvli),
                'mudancas_territoriais': float(mudanca),
                'volatilidade': float(np.mean(self.tensor[day, :, 6]))  # Feature 6: volatilidade
            })
        
        return timeline
    
    def get_risco_por_regiao(self):
        """Agregar risco por região geográfica"""
        if self.df_pred is None:
            return {}
        
        # Mapeamento bairro → região (simplificado)
        regioes_mapping = {
            'CAPITAL': [
                'Aldeota', 'Meireles', 'Praia de Iracema', 'Benfica', 'Messejana',
                'Papicu', 'Cocó', 'Mucuripe', 'Cais do Porto', 'Centro'
            ],
            'PERIFERIA': [
                'Barra Do Ceará', 'Jangurussu', 'Granja Lisboa', 'Barroso',
                'Canindezinho', 'Bom Jardim', 'Antônio Bezerra', 'Bonsucesso'
            ],
            'INTERIOR': [
                'Maracanaú', 'Itaitinga', 'Caucaia', 'Pacatuba'
            ]
        }
        
        resumo_regioes = {}
        for regiao, bairros in regioes_mapping.items():
            df_regiao = self.df_pred[self.df_pred['bairro'].isin(bairros)]
            if not df_regiao.empty:
                resumo_regioes[regiao] = {
                    'cvli_medio': float(df_regiao['cvli_predito'].mean()),
                    'cvli_max': float(df_regiao['cvli_predito'].max()),
                    'bairros_criticos': int((df_regiao['cvli_predito'] > df_regiao['cvli_predito'].quantile(0.9)).sum()),
                    'volatilidade_media': float(df_regiao.get('volatilidade', 0).mean())
                }
        
        return resumo_regioes
    
    def export_para_dashboard(self):
        """Exportar dados no formato esperado pelo dashboard"""
        return {
            'top_15_bairros': self.get_top_bairros(15),
            'metricas_globais': {
                'total_bairros': len(self.df_pred) if self.df_pred is not None else 0,
                'bairros_criticos': int((self.df_pred['cvli_predito'] > self.df_pred['cvli_predito'].quantile(0.9)).sum()) if self.df_pred is not None else 0,
                'cvli_medio': float(self.df_pred['cvli_predito'].mean()) if self.df_pred is not None else 0,
                'periodo': '210 dias (23/01/2026 a 21/08/2026)'
            },
            'por_regiao': self.get_risco_por_regiao(),
            'timeline_ultimos_30_dias': self.get_timeline()[-30:] if self.get_timeline() else []
        }


# Instância global
adapter = None

def init_adapter():
    """Inicializar adapter globalmente"""
    global adapter
    if adapter is None:
        adapter = DataAdapter()
    return adapter

def get_adapter():
    """Obter instância do adapter"""
    if adapter is None:
        init_adapter()
    return adapter
