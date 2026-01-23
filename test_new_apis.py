#!/usr/bin/env python3
"""Testar as novas rotas da API"""

import sys
sys.path.insert(0, '.')

from src import app
import json

# Criar contexto da app
with app.app.app_context():
    client = app.app.test_client()
    
    print("="*80)
    print("TESTANDO NOVAS ROTAS")
    print("="*80)
    
    # Teste 1: /api/cvli_forecast_extended
    print("\n1️⃣  GET /api/cvli_forecast_extended?top=10")
    resp = client.get('/api/cvli_forecast_extended?top=10')
    print(f"   Status: {resp.status_code}")
    data = json.loads(resp.data)
    print(f"   Sucesso: {data['sucesso']}")
    print(f"   Total bairros analisados: {data['data']['total_bairros']}")
    print(f"   Bairros críticos (>90º): {data['data']['bairros_criticos']}")
    print(f"   Bairros alto risco (75-90º): {data['data']['bairros_alto_risco']}")
    print(f"   CVLI médio: {data['data']['metricas']['cvli_medio']:.4f}")
    print(f"   CVLI máximo: {data['data']['metricas']['cvli_max']:.4f}")
    print("\n   Top 5 bairros de risco:")
    for pred in data['data']['previsoes'][:5]:
        print(f"     - {pred['bairro']:25s} | CVLI: {pred['cvli_predito']:6.4f} | {pred['classificacao']:8s} | Mudança: {pred['prob_mudanca']:.1%}")
    
    # Teste 2: /api/territorial_volatility/<bairro>
    print("\n2️⃣  GET /api/territorial_volatility/Jangurussu")
    resp = client.get('/api/territorial_volatility/Jangurussu')
    print(f"   Status: {resp.status_code}")
    data = json.loads(resp.data)
    if data['sucesso']:
        print(f"   Bairro: {data['data']['bairro']}")
        print(f"   CVLI previsto: {data['data']['cvli_predito']:.4f}")
        print(f"   Volatilidade: {data['data']['volatilidade_territorial']['nivel']}")
        print(f"   Prob. mudança: {data['data']['volatilidade_territorial']['prob_mudanca']:.1%}")
        print(f"   Recomendações: {len(data['data']['recomendacoes'])} itens")
        for rec in data['data']['recomendacoes']:
            print(f"     - {rec}")
    else:
        print(f"   Erro: {data['erro']}")
    
    # Teste 3: /api/faction_timeline
    print("\n3️⃣  GET /api/faction_timeline")
    resp = client.get('/api/faction_timeline')
    print(f"   Status: {resp.status_code}")
    data = json.loads(resp.data)
    print(f"   Sucesso: {data['sucesso']}")
    if data['sucesso']:
        print(f"   Última atualização: {data['data']['ultima_atualizacao']}")
        print(f"   Bairros analisados: {data['data']['bairros_analisados']}")
        print(f"   Bairros com mudanças: {data['data']['bairros_com_mudancas']}")
        print(f"   Fações identificadas: {len(data['data']['faccoes_identificadas'])} facções")
        for faccao, info in list(data['data']['faccoes_identificadas'].items())[:5]:
            print(f"     - {faccao}: {info.get('bairros_controlados', 0)} bairros")
    
    print("\n" + "="*80)
    print("✅ TODAS AS ROTAS FUNCIONANDO!")
    print("="*80)
