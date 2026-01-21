#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Dashboard Estratégico Descritivo
Análise inteligente com Gemini para recomendações de atuação operacional
"""

import json
import pandas as pd
import geopandas as gpd
from pathlib import Path
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))

import config
from gemini_client import GeminiClient

def get_strategic_insights():
    """Coleta dados agregados para análise estratégica"""
    
    # 1. Carregar crimes históricos
    if not config.CONSOLIDATED_FILE.exists():
        return None
    
    df_crimes = pd.read_parquet(config.CONSOLIDATED_FILE)
    
    # 2. Carregar predições por bairro
    pred_file = config.ARTIFACTS['CAPITAL']['prediction']
    if not pred_file.exists():
        return None
    
    df_pred = pd.read_csv(pred_file)
    
    # Análise por tipo de crime
    crime_types = {
        'CVP': len(df_crimes[df_crimes['tipo'] == 'CVP']),
        'CVLI': len(df_crimes[df_crimes['tipo'] == 'CVLI'])
    }
    
    # Análise por facção
    facction_crimes = df_crimes[df_crimes['regiao_sistema'] == 'CAPITAL'].groupby('faccao_predominante').size().to_dict()
    
    # Top bairros críticos
    top_bairros = df_pred.nlargest(10, 'risco_previsto').to_dict('records')
    
    # Estatísticas gerais
    stats = {
        'total_crimes': len(df_crimes),
        'crimes_capital': len(df_crimes[df_crimes['regiao_sistema'] == 'CAPITAL']),
        'crime_types': crime_types,
        'facctions': facction_crimes,
        'top_bairros': top_bairros,
        'data_analise': datetime.now().strftime('%d/%m/%Y %H:%M')
    }
    
    return stats

def generate_ai_analysis(stats):
    """Gera análise com Gemini para recomendações de atuação"""
    
    if not stats:
        return {"erro": "Dados não disponíveis"}
    
    client = GeminiClient()
    
    # Prompt estratégico para o Gemini
    prompt = f"""
    Você é um analista estratégico de segurança pública. 
    
    DADOS DA SITUAÇÃO EM FORTALEZA:
    - Total de crimes analisados: {stats['total_crimes']}
    - Crimes em Fortaleza (CAPITAL): {stats['crimes_capital']}
    - Roubos/Patrimoniais (CVP): {stats['crime_types']['CVP']}
    - Homicídios (CVLI): {stats['crime_types']['CVLI']}
    
    DISTRIBUIÇÃO POR FACÇÃO (Crimes em CAPITAL):
    {json.dumps(stats['facctions'], ensure_ascii=False, indent=2)}
    
    BAIRROS CRÍTICOS (Predição de Risco - 15 dias):
    """
    
    for i, bairro in enumerate(stats['top_bairros'][:5], 1):
        prompt += f"\n{i}. {bairro['local_oficial']}: Risco {bairro['risco_previsto']:.2%}"
    
    prompt += """
    
    TAREFA:
    Com base nestes dados, gere um parecer estratégico CONCISO (máx 500 palavras) que contenha:
    
    1. **DIAGNÓSTICO RÁPIDO**: Qual é a situação em Fortaleza? (2-3 linhas)
    
    2. **HOTSPOTS CRÍTICOS**: Quais são os bairros que demandam ação imediata? (2-3 bairros principais)
    
    3. **TIPOLOGIA DO PROBLEMA**: 
       - Predominância de roubos (CVP) ou homicídios (CVLI)?
       - Qual facção está ativa onde?
    
    4. **RECOMENDAÇÕES DE ATUAÇÃO** (Operacional):
       - Onde colocar reforço policial?
       - Qual estratégia por tipo de crime?
       - Qual a prioridade?
    
    5. **MÉTRICA DE SUCESSO**: Como medir se a atuação está funcionando?
    
    Escreva de forma que um gestor de segurança pública entenda CLARAMENTE onde aplicar recursos.
    """
    
    try:
        response = client.generate_content(prompt)
        return {
            "sucesso": True,
            "analise": response,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "sucesso": False,
            "erro": str(e),
            "timestamp": datetime.now().isoformat()
        }

if __name__ == "__main__":
    print("\n" + "="*80)
    print("TESTE: Dashboard Estratégico com Análise Gemini")
    print("="*80)
    
    print("\n1. Coletando dados agregados...")
    stats = get_strategic_insights()
    
    if stats:
        print(f"   ✓ Dados coletados:")
        print(f"     - Total crimes: {stats['total_crimes']}")
        print(f"     - Crimes em Fortaleza: {stats['crimes_capital']}")
        print(f"     - Top bairro: {stats['top_bairros'][0]['local_oficial']} (risco {stats['top_bairros'][0]['risco_previsto']:.2%})")
        
        print("\n2. Gerando análise com Gemini...")
        analysis = generate_ai_analysis(stats)
        
        if analysis['sucesso']:
            print("   ✓ Análise gerada com sucesso!")
            print("\n" + "-"*80)
            print(analysis['analise'])
            print("-"*80)
        else:
            print(f"   ❌ Erro: {analysis['erro']}")
    else:
        print("   ❌ Erro ao coletar dados")
    
    print("\n" + "="*80)
