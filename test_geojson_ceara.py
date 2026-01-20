#!/usr/bin/env python3
import json
from pathlib import Path

f = Path('data/raw/limites_ceara.geojson')
print(f"Arquivo existe: {f.exists()}")

if f.exists():
    data = json.load(open(f))
    features = data.get('features', [])
    print(f"Features: {len(features)}")
    if features:
        geom_type = features[0]['geometry']['type']
        print(f"Geometry type: {geom_type}")
        
        # Mostrar bbox
        if 'properties' in features[0]:
            print(f"Properties: {list(features[0]['properties'].keys())}")
