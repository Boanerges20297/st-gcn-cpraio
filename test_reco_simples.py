#!/usr/bin/env python3
"""Teste recomendações simples"""
import requests
import json

print("🧪 TESTE: /api/recomendacoes_simples")
print("=" * 80)

r = requests.get('http://localhost:5000/api/recomendacoes_simples')
data = r.json()

print(f"\nStatus: {r.status_code}")
print(f"Sucesso: {data.get('sucesso')}")
print(f"Recomendações: {len(data['data']['recomendacoes'])}")

if data['data']['recomendacoes']:
    print("\n✅ Primeiras 3 recomendações:")
    for rec in data['data']['recomendacoes'][:3]:
        print(f"\n  {rec['icon']} {rec['tipo']}")
        print(f"     Prioridade: {rec['prioridade']}")
        print(f"     Ação: {rec['acao']}")
        print(f"     Score: {rec['score_risco']:.1f}/100")
else:
    print("\n❌ Nenhuma recomendação gerada")

print("\n" + "=" * 80)
