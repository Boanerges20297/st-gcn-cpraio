import urllib.request
import json

try:
    response = urllib.request.urlopen('http://localhost:5000/api/recomendacoes_operacionais', timeout=10)
    data = json.loads(response.read().decode('utf-8'))
    
    if data.get('recomendacoes'):
        rec = data['recomendacoes'][0]
        print("Campos retornados:")
        for k in rec.keys():
            print(f"  - {k}: {rec[k]}")
    else:
        print("Nenhuma recomendação retornada")
        
except Exception as e:
    print(f"Erro: {e}")
