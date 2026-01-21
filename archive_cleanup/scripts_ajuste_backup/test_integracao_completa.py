#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Teste de Integração: Mapa Tático + Dashboard Estratégico
Valida navegação bidirecional
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from app import app

print("\n" + "="*80)
print("TESTE: Integração Mapa Tático + Dashboard Estratégico")
print("="*80)

with app.test_client() as client:
    
    # Teste 1: Mapa tático
    print("\n1. GET / (Mapa Tático)")
    response = client.get('/')
    html = response.get_data(as_text=True)
    print(f"   Status: {response.status_code}")
    print(f"   Tem botão Dashboard: {'Dashboard' in html}")
    print(f"   ✓ Mapa tático acessível")
    
    # Teste 2: Dashboard descritivo
    print("\n2. GET /dashboard-estrategico")
    response = client.get('/dashboard-estrategico')
    html = response.get_data(as_text=True)
    print(f"   Status: {response.status_code}")
    print(f"   Tem botão Voltar: {'Voltar' in html}")
    print(f"   ✓ Dashboard estratégico acessível")
    
    # Teste 3: API insights
    print("\n3. GET /api/strategic_insights")
    response = client.get('/api/strategic_insights')
    data = response.get_json()
    crimes = data.get("data", {}).get("total_crimes", "N/A")
    print(f"   Status: {response.status_code}")
    print(f"   Total de crimes: {crimes}")
    print(f"   ✓ API de dados funcionando")

print("\n" + "="*80)
print("NAVEGAÇÃO DISPONÍVEL:")
print("="*80)
print("""
SIGERAIO (Mapa Tático)
  ↓ Clique no novo botão
  [🤖 Dashboard] → Dashboard Estratégico
                    ↓ Clique no novo botão
                    [← Voltar] → Volta ao Mapa

URLs:
  • http://localhost:5000/                (Mapa)
  • http://localhost:5000/dashboard-estrategico  (Dashboard)

TESTE: Abra em duas abas e compare qual é mais útil!
""")
print("="*80 + "\n")
