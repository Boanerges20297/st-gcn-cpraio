#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inspeccionar dataset_criticidade_janela180d.pt
"""

import torch
from pathlib import Path

print("[1] Carregando dataset...")
dataset_path = Path("data/tensors/dataset_criticidade_janela180d.pt")
data = torch.load(dataset_path, weights_only=False)

print("[2] Inspecionando estrutura...")
print("Tipo: {}".format(type(data)))

if isinstance(data, dict):
    print("Keys: {}".format(data.keys()))
    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            print("  {}: Tensor shape {}".format(key, value.shape))
        else:
            print("  {}: {} (type: {})".format(key, type(value), str(type(value))[:50]))
elif isinstance(data, torch.Tensor):
    print("Shape: {}".format(data.shape))
    print("Dtype: {}".format(data.dtype))
else:
    print("Type: {}".format(type(data)))
    print("Content: {}".format(str(data)[:200]))
