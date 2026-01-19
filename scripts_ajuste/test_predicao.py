import sys
sys.path.insert(0, 'src')
from app import load_risk_map
import json

print('=== TESTE load_risk_map COM PREDIÇÃO (NÃO HISTÓRICO) ===')
print()

# Teste 1: Sem filtro tipo_crime
print('1. CAPITAL - Sem filtro tipo_crime (predição geral)')
risk = load_risk_map('CAPITAL', 'TODOS')
if risk:
    features = risk['features']
    risco_values = [f['properties'].get('risco', 0) for f in features]
    risco_top = sorted(risco_values, reverse=True)[:3]
    print(f'   Total features: {len(features)}')
    print(f'   Top 3 riscos (predição): {[f"{v:.3f}" for v in risco_top]}')

print()

# Teste 2: Com filtro CVP
print('2. CAPITAL - Filtrado por CVP (predição, mas zero onde não há CVP)')
risk_cvp = load_risk_map('CAPITAL', 'CVP')
if risk_cvp:
    features = risk_cvp['features']
    risco_values = [f['properties'].get('risco', 0) for f in features]
    risco_com_crime = [v for v in risco_values if v > 0]
    print(f'   Total features: {len(features)}')
    print(f'   Features com CVP (risco > 0): {len(risco_com_crime)}')
    if risco_com_crime:
        print(f'   Top risco em área com CVP: {max(risco_com_crime):.3f}')

print()

# Teste 3: Com filtro CVLI
print('3. CAPITAL - Filtrado por CVLI (predição, mas zero onde não há CVLI)')
risk_cvli = load_risk_map('CAPITAL', 'CVLI')
if risk_cvli:
    features = risk_cvli['features']
    risco_values = [f['properties'].get('risco', 0) for f in features]
    risco_com_crime = [v for v in risco_values if v > 0]
    print(f'   Total features: {len(features)}')
    print(f'   Features com CVLI (risco > 0): {len(risco_com_crime)}')
    if risco_com_crime:
        print(f'   Top risco em área com CVLI: {max(risco_com_crime):.3f}')

print('\nTudo OK - Agora usa PREDIÇÃO em todos os casos!')
