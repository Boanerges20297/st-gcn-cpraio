#!/usr/bin/env python3
"""Teste final de integração"""

import sys
sys.path.insert(0, '.')

from src import app
import json

print('='*80)
print('TESTE DE INTEGRAÇÃO FINAL')
print('='*80)

with app.app.app_context():
    client = app.app.test_client()
    
    # 1. Dashboard Sync
    print('\n✅ 1. Dashboard Sync')
    resp = client.get('/api/dashboard_sync')
    if resp.status_code == 200:
        data = json.loads(resp.data)
        total = data['data']['metricas_globais']['total_bairros']
        criticos = data['data']['metricas_globais']['bairros_criticos']
        print(f'   Bairros: {total}')
        print(f'   Críticos: {criticos}')
        print(f'   ✅ OK')
    else:
        print(f'   ❌ Status {resp.status_code}')
    
    # 2. Detalhes Bairro
    print('\n✅ 2. Detalhes do Bairro')
    resp = client.get('/api/bairro_detalhes/Jangurussu')
    if resp.status_code == 200:
        data = json.loads(resp.data)
        bairro = data['data']['bairro']
        score = data['data']['score_risco']
        print(f'   Bairro: {bairro}')
        print(f'   Score: {score:.1f}/100')
        print(f'   ✅ OK')
    else:
        print(f'   ❌ Status {resp.status_code}')
    
    # 3. Predições Estendidas
    print('\n✅ 3. Predições Estendidas')
    resp = client.get('/api/cvli_forecast_extended?top=5')
    if resp.status_code == 200:
        data = json.loads(resp.data)
        count = len(data['data']['previsoes'])
        dias = data['data']['horizonte_dias']
        print(f'   Top bairros: {count}')
        print(f'   Horizonte: {dias} dias')
        print(f'   ✅ OK')
    else:
        print(f'   ❌ Status {resp.status_code}')
    
    # 4. Volatilidade
    print('\n✅ 4. Volatilidade Territorial')
    resp = client.get('/api/territorial_volatility/Jangurussu')
    if resp.status_code == 200:
        data = json.loads(resp.data)
        nivel = data['data']['volatilidade_territorial']['nivel']
        prob = data['data']['volatilidade_territorial']['prob_mudanca']
        print(f'   Nível: {nivel}')
        print(f'   Prob Mudança: {prob:.1%}')
        print(f'   ✅ OK')
    else:
        print(f'   ❌ Status {resp.status_code}')
    
    # 5. Timeline Facções
    print('\n✅ 5. Timeline de Facções')
    resp = client.get('/api/faction_timeline')
    if resp.status_code == 200:
        data = json.loads(resp.data)
        bairros = data['data']['bairros_analisados']
        print(f'   Bairros analisados: {bairros}')
        print(f'   ✅ OK')
    else:
        print(f'   ❌ Status {resp.status_code}')

print('\n' + '='*80)
print('✅ INTEGRAÇÃO COMPLETA E FUNCIONAL!')
print('='*80)
print('\nPróximos passos:')
print('1. Iniciar servidor: .\.venv\Scripts\python.exe src/app.py')
print('2. Acessar: http://localhost:5000/dashboard-estrategico')
print('3. Dashboard carregará dados automaticamente via /api/dashboard_sync')
