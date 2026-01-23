#!/usr/bin/env python3
"""Testar rotas de sincronização com dashboard"""

import sys
sys.path.insert(0, '.')

from src import app
import json

# Criar contexto da app
with app.app.app_context():
    client = app.app.test_client()
    
    print("="*80)
    print("TESTANDO SINCRONIZAÇÃO COM DASHBOARD")
    print("="*80)
    
    # Teste 1: Dashboard Sync
    print("\n1️⃣  GET /api/dashboard_sync")
    resp = client.get('/api/dashboard_sync')
    print(f"   Status: {resp.status_code}")
    data = json.loads(resp.data)
    print(f"   Sucesso: {data['sucesso']}")
    
    if data['sucesso']:
        print(f"   Fonte: {data['fonte']}")
        print(f"   Top bairros: {len(data['data']['top_15_bairros'])}")
        print(f"   Bairros críticos: {data['data']['metricas_globais']['bairros_criticos']}")
        print(f"   CVLI médio: {data['data']['metricas_globais']['cvli_medio']:.4f}")
        print(f"   Regiões analisadas: {len(data['data']['por_regiao'])}")
        print(f"   Timeline dias: {len(data['data']['timeline_ultimos_30_dias'])}")
        print("\n   Top 3 bairros:")
        for pred in data['data']['top_15_bairros'][:3]:
            print(f"     - {pred['bairro']:25s} | CVLI: {pred['cvli_predito']:6.4f}")
    else:
        print(f"   Erro: {data['erro']}")
    
    # Teste 2: Detalhes de um bairro
    print("\n2️⃣  GET /api/bairro_detalhes/Jangurussu")
    resp = client.get('/api/bairro_detalhes/Jangurussu')
    print(f"   Status: {resp.status_code}")
    data = json.loads(resp.data)
    
    if data['sucesso']:
        info = data['data']
        print(f"   Bairro: {info['bairro']}")
        print(f"   Score de risco: {info['score_risco']:.1f}/100")
        print(f"   CVLI previsto: {info['cvli_predito']:.4f}")
        print(f"   Recomendação: {info['recomendacao']}")
        print(f"   Risco territorial: {info['risco_territorial']}")
        print(f"   Volatilidade: {info['volatilidade_status']}")
    else:
        print(f"   Erro: {data['erro']}")
    
    print("\n" + "="*80)
    print("✅ SINCRONIZAÇÃO COMPLETA!")
    print("="*80)
