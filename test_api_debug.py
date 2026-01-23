#!/usr/bin/env python3
"""Debug API response"""
import requests
import json

print("=" * 80)
print("TESTANDO /api/dashboard_sync")
print("=" * 80)

r = requests.get('http://localhost:5000/api/dashboard_sync')
data = r.json()

print(f"\nStatus HTTP: {r.status_code}")
print(f"Sucesso: {data.get('sucesso')}")

if data.get('sucesso'):
    metrics = data['data']['metricas_globais']
    print(f"\n📊 MÉTRICAS GLOBAIS:")
    print(f"  Total bairros: {metrics['total_bairros']}")
    print(f"  Bairros críticos: {metrics['bairros_criticos']}")
    print(f"  CVLI médio: {metrics['cvli_medio']:.6f}")
    print(f"  Período: {metrics['periodo']}")
    
    print(f"\n🏙️ TOP 15 BAIRROS:")
    for i, b in enumerate(data['data']['top_15_bairros'][:3], 1):
        print(f"\n  {i}. {b['bairro']}")
        print(f"     CVLI: {b['cvli_predito']:.4f}")
        print(f"     Região: {b.get('regiao', 'N/A')}")
        print(f"     Score Risco: {b.get('score_risco', 'N/A')}")
else:
    print(f"\nErro: {data.get('erro')}")
    print(f"Ação: {data.get('acao')}")
