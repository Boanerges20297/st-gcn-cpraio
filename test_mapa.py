#!/usr/bin/env python3
"""Teste mapa com novo modelo"""
import requests
import json

print("🧪 TESTE: /api/dashboard_data")
print("=" * 80)

r = requests.get('http://localhost:5000/api/dashboard_data?region=CAPITAL')

print(f"\nStatus: {r.status_code}")
data = r.json()

if data.get('polygons'):
    print(f"✅ Polígonos carregados: {len(data['polygons']['features'])}")
    
    # Verificar se tem risco_previsto
    first_feature = data['polygons']['features'][0] if data['polygons']['features'] else None
    if first_feature:
        props = first_feature['properties']
        print(f"\nPrimeiro bairro:")
        print(f"  Nome: {props.get('name') or props.get('bairro')}")
        print(f"  Nível: {props.get('nivel_alerta')}")
        print(f"  Risco: {props.get('risco')}")
        print(f"  CVLI Predito: {props.get('cvli_predito', 'N/A')}")
else:
    print(f"❌ Erro: {data.get('erro')}")

print(f"\nTop alvos: {len(data.get('targets', []))}")
for target in data.get('targets', [])[:3]:
    print(f"  - {target['local']}: {target['nivel']} (score: {target['score']})")

print("\n" + "=" * 80)
