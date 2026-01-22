import json
from collections import Counter
from pathlib import Path

paths = [
    'data/raw/ocorrencia_policial_operacional.json',
    'data/raw/dados_status_ocorrencias_gerais.json'
]
out = Path('outputs/tipo_counts.csv')
counts = Counter()
samples = {}
for p in paths:
    try:
        with open(p, 'r', encoding='utf-8') as f:
            text = f.read()
    except Exception as e:
        print('ERROR reading', p, e)
        continue
    # try json.load if full array
    try:
        data = json.loads(text)
        # traverse to 'data' inside structure if present
        if isinstance(data, dict):
            data = [data]
    except Exception:
        # fallback: parse line by line, find objects
        data = []
        for line in text.splitlines():
            line = line.strip().rstrip(',')
            if line.startswith('{') and line.endswith('}'):
                try:
                    obj = json.loads(line)
                    data.append(obj)
                except Exception:
                    continue
        def walk_and_count(obj):
            # recursively walk dict/list and count 'tipo' occurrences
            if isinstance(obj, dict):
                # if this dict has a tipo-like key, count it
                for k in ('tipo','Tipo','TYPE','type'):
                    if k in obj and isinstance(obj[k], (str, int)):
                        tval = str(obj[k]).strip().lower()
                        counts[tval] += 1
                        if tval not in samples:
                            samples[tval] = obj
                # also descend into nested lists/dicts
                for v in obj.values():
                    if isinstance(v, (list, dict)):
                        walk_and_count(v)
            elif isinstance(obj, list):
                for item in obj:
                    walk_and_count(item)

        walk_and_count(data)

with out.open('w', encoding='utf-8') as f:
    f.write('tipo,count\n')
    for k,v in counts.most_common():
        f.write(f'{k},{v}\n')

with open('outputs/tipo_samples.json','w',encoding='utf-8') as f:
    json.dump({k: (v if isinstance(v, dict) else {}) for k,v in samples.items()}, f, ensure_ascii=False, indent=2)

print('Wrote', out)
print('Top tipos:', counts.most_common(10))
