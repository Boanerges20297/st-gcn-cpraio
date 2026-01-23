#!/usr/bin/env python3
"""Teste simples de validação"""
import requests
import json

print("🧪 VALIDAÇÃO DO DASHBOARD")
print("=" * 80)

# Test 1: API Response
print("\n✅ Test 1: /api/dashboard_sync")
r = requests.get('http://localhost:5000/api/dashboard_sync')
data = r.json()

print(f"  Status: {r.status_code}")
print(f"  Sucesso: {data.get('sucesso')}")
print(f"  Top 15 bairros: {len(data['data']['top_15_bairros'])}")
print(f"  Métricas globais: {list(data['data']['metricas_globais'].keys())}")

# Test 2: Estrutura dos bairros
print("\n✅ Test 2: Estrutura de dados dos bairros")
first_bairro = data['data']['top_15_bairros'][0]
print(f"  Bairro: {first_bairro.get('bairro')}")
print(f"  CVLI: {first_bairro.get('cvli_predito')}")
print(f"  Score Risco: {first_bairro.get('score_risco')}")
print(f"  Região: {first_bairro.get('regiao')}")
print(f"  Chaves do bairro: {list(first_bairro.keys())}")

# Test 3: HTML Load
print("\n✅ Test 3: Dashboard HTML Load")
r = requests.get('http://localhost:5000/dashboard-estrategico')
print(f"  Status: {r.status_code}")
print(f"  Tamanho: {len(r.text)} bytes")

print("\n" + "=" * 80)
print("✅ VALIDAÇÃO CONCLUÍDA")
