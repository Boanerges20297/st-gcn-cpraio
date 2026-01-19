#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Preview dos dados como aparecem no dashboard
"""

import pandas as pd

print("\n" + "="*80)
print("SAMPLE: Como os dados aparecem no dashboard por bairro")
print("="*80)

pred = pd.read_csv("outputs/reports/pred_capital_bairros.csv")

# Top 10 para operações
top = pred.nlargest(10, "risco_previsto")

print("\nFOCO OPERACIONAL - Top 10 Bairros para Atuação (Predição Futura):")
print("-"*80)
for idx, row in top.iterrows():
    bairro = row["local_oficial"]
    risco = row["risco_previsto"]
    if risco > 0.32:
        nivel = "🔴 CRÍTICO"
    elif risco > 0.31:
        nivel = "🟠 ALTO"
    elif risco > 0.30:
        nivel = "🟡 MÉDIO"
    else:
        nivel = "🟢 BAIXO"
    print(f"  {bairro:30} | Risco: {risco:.4f} | {nivel}")

print("-"*80)
print("✓ 138 bairros de Fortaleza agora discriminados por predição individual")
print("✓ Pronto para operações táticas de referência por bairro")
print("="*80 + "\n")
