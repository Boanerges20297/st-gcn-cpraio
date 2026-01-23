#!/usr/bin/env python3
"""Teste de recomendações"""
import requests
from datetime import datetime, timedelta

print("🧪 TESTE: /api/recomendacoes_operacionais")
print("=" * 80)

# Datas padrão
hoje = datetime.now()
trinta = hoje - timedelta(days=30)

data_inicio = trinta.strftime('%Y-%m-%d')
data_fim = hoje.strftime('%Y-%m-%d')

params = {
    'data_inicio': data_inicio,
    'data_fim': data_fim,
    'regiao': 'CAPITAL'
}

print(f"\nParâmetros:")
print(f"  data_inicio: {data_inicio}")
print(f"  data_fim: {data_fim}")
print(f"  regiao: CAPITAL")

r = requests.get('http://localhost:5000/api/recomendacoes_operacionais', params=params)

print(f"\nResposta:")
print(f"  Status: {r.status_code}")
data = r.json()
print(f"  Sucesso: {data.get('sucesso')}")
if data.get('sucesso'):
    print(f"  Recomendações: {len(data['data']['recomendacoes'])}")
    if data['data']['recomendacoes']:
        print(f"  Primeira: {data['data']['recomendacoes'][0].get('tipo', 'N/A')}")
else:
    print(f"  Erro: {data.get('erro')}")

print("\n" + "=" * 80)
