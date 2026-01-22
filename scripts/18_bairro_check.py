import json
from collections import Counter
from pathlib import Path

p = Path('data/raw/dados_status_ocorrencias_gerais.json')
assert p.exists(), p

with p.open('r', encoding='utf-8') as f:
    top = json.load(f)

# file is an array of wrapper objects; find the one with key 'data'
items = None
for obj in top:
    if isinstance(obj, dict) and 'data' in obj:
        items = obj['data']
        break
if items is None:
    raise SystemExit('data array not found in top-level JSON')

total = len(items)

# fields that may contain neighborhood info
candidate_fields = ['bairro', 'BairroOcor', 'BairroAbord', 'bairro_ocor', 'bairro_orig']

non_null = 0
counter = Counter()
samples = []
for it in items:
    b = None
    for k in candidate_fields:
        if k in it and it.get(k) not in (None, '', 'null'):
            b = it.get(k)
            break
    if b:
        non_null += 1
        counter[b.strip()] += 1
        if len(samples) < 10:
            samples.append({k: it.get(k) for k in ['id','data','cidade','latitude','longitude','bairro']})

print('total_rows:', total)
print('non_null_bairro:', non_null)
print('distinct_bairros_sample_count:', len(counter))
print('top10_bairros:')
for b,c in counter.most_common(10):
    print(f'{b}\t{c}')

print('\nexamples (up to 10):')
for s in samples:
    print(s)

# write outputs
out = Path('outputs')
out.mkdir(exist_ok=True)
with (out/'bairro_counts.json').open('w', encoding='utf-8') as f:
    json.dump({'total': total, 'non_null': non_null, 'distinct': len(counter), 'top': counter.most_common(50)}, f, ensure_ascii=False, indent=2)

with (out/'bairro_samples.json').open('w', encoding='utf-8') as f:
    json.dump(samples, f, ensure_ascii=False, indent=2)

print('\nWrote outputs/bairro_counts.json and outputs/bairro_samples.json')
