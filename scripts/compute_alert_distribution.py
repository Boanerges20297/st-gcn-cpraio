from pathlib import Path
import csv
BASE = Path(__file__).resolve().parent.parent
pred_file = BASE / 'outputs' / 'reports' / 'pred_capital_bairros.csv'

def get_alert_level(score, region='CAPITAL'):
    factor = 30 if region == 'INTERIOR' else 6 if region == 'RMF' else 3
    intensity = score * factor
    if intensity > 0.8:
        return 'CRÍTICO'
    if intensity > 0.5:
        return 'ALTO'
    if intensity > 0.2:
        return 'MÉDIO'
    return 'BAIXO'

rows = []
if not pred_file.exists():
    print('Pred file not found:', pred_file)
    raise SystemExit(1)

with open(pred_file, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for r in reader:
        name = (r.get('local') or r.get('local_oficial') or r.get('bairro') or '').strip()
        try:
            score = float(r.get('risco_previsto') or r.get('risk') or 0)
        except:
            score = 0.0
        rows.append((name.upper(), score))

n = len(rows)
levels = {'CRÍTICO':0,'ALTO':0,'MÉDIO':0,'BAIXO':0}
for name,score in rows:
    lvl = get_alert_level(score)
    levels[lvl]+=1

# thresholds in terms of score for CAPITAL (factor=3)
factor=3
thr_crit = 0.8/factor
thr_alto = 0.5/factor
thr_medio = 0.2/factor

print('Pred file:', pred_file)
print('Total bairros:', n)
print('\nThresholds (CAPITAL, factor=3):')
print(f'  CRÍTICO if score > {thr_crit:.6f}')
print(f'  ALTO    if score > {thr_alto:.6f}')
print(f'  MÉDIO   if score > {thr_medio:.6f}')
print('\nCounts by alert level:')
for k in ['CRÍTICO','ALTO','MÉDIO','BAIXO']:
    v = levels[k]
    print(f'  {k}: {v} ({v/n*100:.1f}%)')

# show top 15 by score
rows_sorted = sorted(rows, key=lambda x: x[1], reverse=True)
print('\nTop 15 bairros by predicted score:')
for i,(name,score) in enumerate(rows_sorted[:15],1):
    print(f'  {i:2d}. {name:30s} {score:.6f} {get_alert_level(score)}')

# find CENTRO
centro = [r for r in rows if 'CENTRO' in r[0]]
if centro:
    for name,score in centro:
        print('\nExample: CENTRO ->', name, score, get_alert_level(score))
else:
    print('\nExample: CENTRO not found in prediction file')
