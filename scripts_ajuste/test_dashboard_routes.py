#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Teste das rotas do Dashboard Estratégico
Verifica se /api/strategic_insights funciona
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

# Test imports
try:
    from app import app, get_strategic_insights, get_ai_analysis
    print("✓ Flask app importado com sucesso")
except Exception as e:
    print(f"❌ Erro ao importar app: {e}")
    sys.exit(1)

# Test rotas
print("\n" + "="*80)
print("TESTE: Rotas do Dashboard Estratégico")
print("="*80)

with app.test_client() as client:
    
    # Teste 1: Dashboard UI
    print("\n1. Testando GET /dashboard-estrategico")
    try:
        response = client.get('/dashboard-estrategico')
        if response.status_code == 200:
            print(f"   ✓ Status 200")
            print(f"   ✓ Content-Type: {response.content_type}")
            if 'Dashboard Estratégico' in response.get_data(as_text=True):
                print(f"   ✓ HTML contém título esperado")
        else:
            print(f"   ❌ Status {response.status_code}")
    except Exception as e:
        print(f"   ❌ Erro: {e}")
    
    # Teste 2: API de insights
    print("\n2. Testando GET /api/strategic_insights")
    try:
        response = client.get('/api/strategic_insights')
        if response.status_code == 200:
            data = response.get_json()
            print(f"   ✓ Status 200")
            print(f"   ✓ Response JSON recebido")
            
            if data.get('sucesso'):
                print(f"   ✓ Sucesso: {data['sucesso']}")
                
                stats = data.get('data', {})
                print(f"   ✓ Total de crimes: {stats.get('total_crimes', 'N/A')}")
                print(f"   ✓ Crimes em Capital: {stats.get('crimes_capital', 'N/A')}")
                print(f"   ✓ Facções encontradas: {len(stats.get('facctions', {}))}")
                print(f"   ✓ Top bairros: {len(stats.get('top_bairros', []))}")
            else:
                print(f"   ⚠ Sucesso = False: {data.get('erro', 'sem erro')}")
        else:
            print(f"   ❌ Status {response.status_code}")
    except Exception as e:
        print(f"   ❌ Erro: {e}")
    
    # Teste 3: API de análise (sem processar Gemini)
    print("\n3. Testando POST /api/ai_analysis")
    try:
        response = client.post('/api/ai_analysis', json={})
        if response.status_code == 200:
            data = response.get_json()
            print(f"   ✓ Status 200")
            print(f"   ✓ Response JSON recebido")
            
            if data.get('sucesso'):
                print(f"   ✓ Sucesso: {data['sucesso']}")
                print(f"   ✓ Análise gerada ({len(data.get('analise', ''))} caracteres)")
            else:
                print(f"   ⚠ Sucesso = False: {data.get('erro', 'sem erro')}")
        else:
            print(f"   ❌ Status {response.status_code}")
    except Exception as e:
        print(f"   ❌ Erro: {e}")

print("\n" + "="*80)
print("✓ Testes concluídos")
print("="*80 + "\n")
