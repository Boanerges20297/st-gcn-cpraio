#!/usr/bin/env python
# -*- coding: utf-8 -*-
import urllib.request
import json
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

try:
    response = urllib.request.urlopen('http://localhost:5000/api/recomendacoes_operacionais')
    data = json.loads(response.read().decode('utf-8'))
    
    if data.get('sucesso'):
        recs = data.get('data', {}).get('recomendacoes', [])
        print(f"\n✓ Sucesso! Total: {len(recs)} recomendações\n")
        
        if recs:
            rec = recs[0]
            print(f"Bairro: {rec['bairro']}")
            print(f"CVP: {rec.get('cvp', 0)} | CVLI: {rec.get('cvli', 0)}")
            print(f"Tendência: {rec.get('tendencia_percentual', 0):.1f}%")
        else:
            print("Nenhuma recomendação retornada!")
    else:
        print(f"Erro: {data.get('erro')}")
        
except Exception as e:
    print(f"Erro: {str(e)}")
