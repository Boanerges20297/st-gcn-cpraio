"""Organize report artifacts: create outputs/docs/testes and copy relevant CSVs and scripts there.

Produces a snapshot of test artifacts for reproducibility.
"""
import os
import shutil
from pathlib import Path

os.makedirs('outputs/docs/testes', exist_ok=True)

# files to include
candidates = [
    'outputs/docs/cvli_seasonality.md',
    'outputs/cvli_timeseries_daily.csv',
    'outputs/cvli_monthly_summary.csv',
    'outputs/cvli_dow_summary.csv',
    'outputs/cvli_autocorr.csv',
    'outputs/cvli_avg_by_month.csv',
    'outputs/cvli_by_bairro_monthly.csv',
    'outputs/cvli_bairro_stats.csv',
]

for p in candidates:
    if os.path.exists(p):
        shutil.copy(p, 'outputs/docs/testes/')

# copy the scripts used for analysis
scripts = ['scripts/14_cvli_seasonality.py', 'scripts/13_exogenous_faction_experiment.py']
for s in scripts:
    if os.path.exists(s):
        shutil.copy(s, 'outputs/docs/testes/')

print('Organized reports and test artifacts into outputs/docs/testes')