"""Analysis: correlations between occurrences and arrests + suggestions.

Generates:
 - prints of dataset columns and basic stats
 - Pearson correlation between occurrence-related features and arrest counts
 - per-neighborhood correlation summary saved to `outputs/correlation_by_neighborhood.csv`
 - feature correlation heatmap data saved to `outputs/feature_correlation.csv`

Run in project root venv:
    python scripts/08_analysis_correlations.py
"""
import os
import sys
import json
import numpy as np
import pandas as pd

# ensure project root on path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def main():
    inp = os.path.join('data', 'processed', 'prisoes_with_features.parquet')
    out_dir = 'outputs'
    os.makedirs(out_dir, exist_ok=True)

    print('Loading', inp)
    df = pd.read_parquet(inp)
    print('Rows:', len(df))
    print('Columns:', df.columns.tolist())

    # Identify possible occurrence and arrest columns
    candidates = [c for c in df.columns if any(k in c.lower() for k in ['priso', 'pris', 'prisao', 'arrest'])]
    occ_candidates = [c for c in df.columns if any(k in c.lower() for k in ['operac', 'operacoes', 'operacao', 'seiz', 'drogas', 'armas', 'dinheiro'])]

    print('\nDetected arrest-like columns:', candidates)
    print('Detected occurrence-like columns (candidates):', occ_candidates)

    # If explicit arrest column not found, try to infer 'prisoes' in outputs folder
    arrest_col = None
    if candidates:
        arrest_col = candidates[0]
    else:
        # try common name
        for name in ['prisoes', 'prisoes_diarias', 'arrests', 'num_prisoes']:
            if name in df.columns:
                arrest_col = name
                break

    if arrest_col is None:
        print('\n[WARN] No clear arrest column found. Correlation with arrests will be skipped.')

    # Select numeric columns for correlation
    num_df = df.select_dtypes(include=[np.number]).copy()
    print('\nNumeric columns for correlation:', num_df.columns.tolist())

    # Global correlation matrix
    corr = num_df.corr(method='pearson')
    corr.to_csv(os.path.join(out_dir, 'feature_correlation.csv'))
    print('\nSaved full feature correlation matrix to outputs/feature_correlation.csv')

    # If arrest column found, compute correlations of features with arrests
    if arrest_col and arrest_col in num_df.columns:
        target_corr = corr[arrest_col].sort_values(ascending=False)
        target_corr.to_csv(os.path.join(out_dir, 'correlation_with_arrests.csv'))
        print(f"Saved correlations with '{arrest_col}' to outputs/correlation_with_arrests.csv")

        # Per-neighborhood correlations: assume df has 'bairro_id' or 'bairro'
        loc_col = None
        for name in ['bairro_id', 'bairro', 'neighborhood', 'local']:
            if name in df.columns:
                loc_col = name
                break

        if loc_col is not None:
            groups = []
            for loc, g in df.groupby(loc_col):
                if g[arrest_col].dtype.kind not in 'biufc':
                    continue
                # compute Pearson correlation between arrest_col and each numeric feature
                row = {'neighborhood': loc, 'n': len(g)}
                for feat in num_df.columns:
                    try:
                        if g[feat].nunique() < 2 or g[arrest_col].nunique() < 2:
                            row[f'corr_{feat}'] = np.nan
                        else:
                            row[f'corr_{feat}'] = g[feat].corr(g[arrest_col])
                    except Exception:
                        row[f'corr_{feat}'] = np.nan
                groups.append(row)
            pd.DataFrame(groups).to_csv(os.path.join(out_dir, 'correlation_by_neighborhood.csv'), index=False)
            print('Saved per-neighborhood correlations to outputs/correlation_by_neighborhood.csv')
        else:
            print('[WARN] No neighborhood identifier column found; skipping per-neighborhood correlations.')

    # Compute change-based correlations (delta day-to-day) to capture co-movement
    print('\nComputing day-to-day deltas and cross-correlations for top occurrence features...')
    deltas = num_df.diff().dropna()
    # pick top 5 occurrence-like numeric cols by variance
    occ_numeric = [c for c in occ_candidates if c in num_df.columns]
    if not occ_numeric:
        occ_numeric = list(num_df.columns[:5])
    top_occ = sorted(occ_numeric, key=lambda c: deltas[c].var() if c in deltas else 0, reverse=True)[:5]
    print('Top occurrence-like columns used:', top_occ)

    if arrest_col and arrest_col in deltas.columns:
        out_rows = []
        for feat in top_occ:
            series_x = deltas[feat]
            series_y = deltas[arrest_col]
            # Pearson on deltas
            valid = series_x.notna() & series_y.notna()
            if valid.sum() < 2:
                r = np.nan
            else:
                r = series_x[valid].corr(series_y[valid])
            out_rows.append({'feature': feat, 'delta_corr_with_arrests': r})
        pd.DataFrame(out_rows).to_csv(os.path.join(out_dir, 'delta_correlations.csv'), index=False)
        print('Saved delta correlations to outputs/delta_correlations.csv')
    else:
        print('Skipping delta correlations: no arrest column found in numeric data.')

    # Simple suggestions for new variables (heuristic)
    suggestions = [
        {'variable': 'police_patrol_hours', 'reason': 'Control for policing intensity that affects both occurrences and arrests.'},
        {'variable': 'response_time_minutes', 'reason': 'Longer response may reduce arrests even if occurrences high.'},
        {'variable': 'population_density', 'reason': 'Normalize occurrence rates by population at risk.'},
        {'variable': 'socioeconomic_index', 'reason': 'Capture underlying drivers correlated with both crime and arrests.'},
        {'variable': 'reporting_rate', 'reason': 'Distinguish between true occurrence change and reporting artifacts.'},
        {'variable': 'camera_coverage', 'reason': 'Affects detection and arrest likelihood.'},
        {'variable': 'time_of_day_bucket', 'reason': 'Temporal granularity (night vs day) may modulate arrest probability.'}
    ]

    with open(os.path.join(out_dir, 'feature_suggestions.json'), 'w', encoding='utf-8') as f:
        json.dump(suggestions, f, ensure_ascii=False, indent=2)

    print('\nSaved feature suggestion list to outputs/feature_suggestions.json')


if __name__ == '__main__':
    main()
