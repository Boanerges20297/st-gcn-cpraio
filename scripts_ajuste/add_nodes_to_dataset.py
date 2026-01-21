import torch
from pathlib import Path
p = Path('data/tensors/dataset_capital.pt')
print('Loading', p)
d = torch.load(p)
if 'nodes' in d:
    print('nodes already present')
else:
    b2i = d.get('bairro_to_idx') or {}
    n = d.get('num_nodes') or (max(b2i.values()) + 1 if b2i else 0)
    nodes = [None] * n
    for k, v in b2i.items():
        nodes[v] = k
    nodes = [x if x is not None else 'UNKNOWN' for x in nodes]
    d['nodes'] = nodes
    torch.save(d, p)
    print('nodes added, count=', len(nodes))
