import csv, json, urllib.request, statistics
from pathlib import Path
BASE = Path(__file__).resolve().parent.parent
pred_file = BASE / 'outputs' / 'reports' / 'pred_capital_bairros.csv'
print('Pred file:', pred_file)
vals = []
names = []
if pred_file.exists():
    with open(pred_file, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                v = float(r.get('risco_previsto') or r.get('risk') or 0)
            except:
                v = 0.0
            vals.append(v)
            names.append(r.get('local') or r.get('local_oficial') or '')
    if vals:
        print('count:', len(vals))
        print('min:', min(vals))
        print('max:', max(vals))
        print('mean:', statistics.mean(vals))
        print('median:', statistics.median(vals))
        print('1st pct (approx):', sorted(vals)[max(0,int(len(vals)*0.01))])
        print('90pct (approx):', sorted(vals)[min(len(vals)-1,int(len(vals)*0.9))])
        print('\nSample rows:')
        for i in range(10):
            print(names[i], vals[i])
else:
    print('Pred file not found')

# Call API and inspect polygons props
BASE_URL = 'http://127.0.0.1:5000'
url = BASE_URL + '/api/dashboard_data?region=CAPITAL&faccao=TODAS&tipo_crime=TODOS'
print('\nCalling', url)
try:
    with urllib.request.urlopen(url, timeout=10) as resp:
        data = json.load(resp)
    poly = data.get('polygons')
    if not poly:
        print('No polygons in response')
    else:
        feats = poly.get('features', [])
        print('Polygons count:', len(feats))
        # Print first 12 properties
        for i,f in enumerate(feats[:12]):
            props = f.get('properties', {})
            print('\n[feature]', i)
            for k in ['name','bairro','risco_previsto','risco','nivel_alerta','source','risk_by','historical_cvli_count','historical_cvp_count','count_tipo_crime']:
                if k in props:
                    print(' ',k,':',props.get(k))
except Exception as e:
    print('Error calling API:', e)
