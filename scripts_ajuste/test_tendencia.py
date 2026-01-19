#!/usr/bin/env python
# -*- coding: utf-8 -*-
import requests
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

r = requests.get('http://localhost:5000/api/recomendacoes_operacionais')
data = r.json()

if data.get('sucesso'):
    recs = data.get('recomendacoes', [])
    print('=== TESTE DE TENDÊNCIA ===\n')
    print(f"{'Bairro':<20} | {'Ação':<12} | {'Tendência':<10} | CVP | CVLI")
    print("-" * 70)
    for rec in recs[:5]:
        bairro = rec['bairro'][:18]
        acao = rec['acao'][:10]
        tend = f"{rec.get('tendencia_percentual', 0):.1f}%"
        cvp = rec.get('cvp', 0)
        cvli = rec.get('cvli', 0)
        print(f"{bairro:<20} | {acao:<12} | {tend:<10} | {cvp:3} | {cvli:3}")
else:
    print(f"Erro: {data.get('erro')}")
