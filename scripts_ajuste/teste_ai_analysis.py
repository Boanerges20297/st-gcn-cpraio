#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Teste do endpoint /api/ai_analysis com nova análise factual
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import urllib.request
import json
from datetime import datetime, timedelta

BASE_URL = "http://127.0.0.1:5000"

print("="*70)
print("TESTE: Endpoint /api/ai_analysis")
print("="*70)

# Parâmetros de data
hoje = datetime.now().date()
data_fim = hoje
data_inicio = hoje - timedelta(days=365)

params = {
    "data_inicio": data_inicio.isoformat(),
    "data_fim": data_fim.isoformat()
}

print(f"\nParametros:")
print(f"   Data inicio: {params['data_inicio']}")
print(f"   Data fim: {params['data_fim']}")
print(f"\nAguardando resposta (pode levar alguns segundos)...")

try:
    data_bytes = json.dumps(params).encode('utf-8')
    request = urllib.request.Request(
        f"{BASE_URL}/api/ai_analysis",
        data=data_bytes,
        headers={"Content-Type": "application/json"}
    )
    
    with urllib.request.urlopen(request, timeout=60) as response:
        data = json.loads(response.read().decode('utf-8'))
        
        print(f"\nStatus: {response.status}")
        
        if data.get("sucesso"):
            print("\n[OK] ANALISE RECEBIDA COM SUCESSO:\n")
            print("="*70)
            analise = data.get("analise", "")
            if len(analise) > 1500:
                print(analise[:1500])
                print(f"\n... [truncado, total: {len(analise)} caracteres]")
            else:
                print(analise)
            print("="*70)
            print(f"\nTimestamp: {data.get('timestamp')}")
        else:
            print(f"\n[ERRO] Erro na analise: {data.get('erro')}")

except urllib.error.URLError as e:
    print(f"\n[ERRO] Conexao: {e}")
    print("Verifique se o servidor esta rodando em http://127.0.0.1:5000")
except Exception as e:
    print(f"\n[ERRO] {e}")

print("\n" + "="*70)
