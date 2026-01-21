# 🔧 Quick Reference: Using Deduplicated Data

## Loading the Processed Data

```python
import pandas as pd

# Load deduplicated and normalized operations
df = pd.read_parquet('data/processed/prisoes_normalized_deduplicated.parquet')

# Load normalization parameters
import json
with open('data/processed/normalization_params_deduplicated.json') as f:
    params = json.load(f)

print(f"Shape: {df.shape}")
print(f"Date range: {df['Data'].min()} to {df['Data'].max()}")
print(f"Neighborhoods: {df['bairro_id'].nunique()}")
print(f"Normalized values range:")
print(f"  Drogas: [{df['drogas_gramas_total_norm'].min():.4f}, {df['drogas_gramas_total_norm'].max():.4f}]")
print(f"  Armas: [{df['armas_total_norm'].min():.4f}, {df['armas_total_norm'].max():.4f}]")
```

## Accessing Mapping Information

```python
# Load mapping report
with open('outputs/neighborhood_mapping_report.json') as f:
    report = json.load(f)

# Get official neighborhoods
official_neighborhoods = report['official_neighborhoods']
print(f"Official neighborhoods: {len(official_neighborhoods)}")
print(official_neighborhoods[:10])

# Get specific mapping with similarity score
mapping_cache = report['mapping_cache']
example_raw = 'Genibau'
if example_raw in mapping_cache:
    match = mapping_cache[example_raw]
    print(f"'{example_raw}' → '{match['matched']}' (similarity: {match['similarity']:.1%})")

# Get unmapped values
unmapped = report['unmapped_values']
print(f"Unmapped values: {len(unmapped)}")
```

## Creating Features for ST-GCN

```python
# Reshape for temporal modeling
df_pivot = df.pivot_table(
    index='Data',
    columns='bairro_id',
    values='drogas_gramas_total_norm',
    fill_value=0
)

# Create lagged features
df_pivot['drogas_lag7'] = df_pivot.rolling(7, min_periods=1).mean()
df_pivot['drogas_lag30'] = df_pivot.rolling(30, min_periods=1).mean()

# Use as node features for ST-GCN
X = df_pivot.values  # Shape: (time_steps, num_nodes)
```

## Using Neighborhood IDs

```python
# Get bairro_id to official name mapping
bairro_to_id = params['neighborhood_mapping']
id_to_bairro = {v: k for k, v in bairro_to_id.items()}

# Example: Get name for bairro_id 0
neighborhood_name = id_to_bairro[0]
print(f"Bairro ID 0: {neighborhood_name}")

# Get seizure statistics for a specific neighborhood
bairro_id = 0
bairro_data = df[df['bairro_id'] == bairro_id]
print(f"\n{neighborhood_name}:")
print(f"  Total operations: {bairro_data['operacoes_diarias'].sum()}")
print(f"  Total seizures: {bairro_data['drogas_gramas_total'].sum():.0f}g")
print(f"  Max daily seizures: {bairro_data['drogas_gramas_total'].max():.0f}g")
```

## Data Quality Checks

```python
# Verify no NaN values in normalized fields
assert df['drogas_gramas_total_norm'].notna().all()
assert df['armas_total_norm'].notna().all()
assert df['dinheiro_total_reais_norm'].notna().all()

# Verify bounds
assert (df['drogas_gramas_total_norm'] >= 0).all()
assert (df['drogas_gramas_total_norm'] <= 1).all()

# Verify temporal coverage
expected_records = 375 * 138  # days × neighborhoods
assert len(df) == expected_records
print(f"✓ All quality checks passed")
```

## Understanding the Data Structure

```
Each record represents:
- One day (Data)
- One neighborhood (bairro_id)
- All operations in that neighborhood on that day (aggregated)

Columns:
- Data: datetime, date of operations
- bairro_id: int 0-137, neighborhood identifier
- operacoes_diarias: int, count of operations
- drogas_gramas_total: float, raw grams seized
- drogas_gramas_total_norm: float [0,1], normalized
- armas_total: float, raw count
- armas_total_norm: float [0,1], normalized
- dinheiro_total_reais: float, raw amount
- dinheiro_total_reais_norm: float [0,1], normalized

Example record:
  2025-01-02, bairro_id=45, operacoes_diarias=3, 
  drogas_gramas_total=234.5, drogas_gramas_total_norm=0.14,
  armas_total=2, armas_total_norm=0.67,
  dinheiro_total_reais=1500, dinheiro_total_reais_norm=0.82
```

## Performance Metrics

```python
# Understand operation distribution
print("Operations distribution:")
print(df['operacoes_diarias'].describe())

# Find most active neighborhoods
top_neighborhoods = df.groupby('bairro_id')['operacoes_diarias'].sum().nlargest(10)
for bairro_id, count in top_neighborhoods.items():
    name = id_to_bairro[bairro_id]
    print(f"  {name}: {count} operations")

# Find seizure hotspots
top_seizures = df.groupby('bairro_id')['drogas_gramas_total'].sum().nlargest(5)
for bairro_id, total_g in top_seizures.items():
    name = id_to_bairro[bairro_id]
    print(f"  {name}: {total_g:.0f}g")
```

## Important Notes

⚠️ **Unmapped Records (7%)**
- 633 operations couldn't be matched to official neighborhoods
- Mostly invalid entries: "-", ".", special characters
- Some are outside Fortaleza proper (rural areas, other municipalities)
- These records are excluded from the normalized dataset

✅ **Normalization Method**
- Uses 99th percentile as max (robust to outliers)
- Post-aggregation normalization (after summing daily totals)
- All values properly bounded in [0, 1]

✅ **Reproducibility**
- All parameters stored in `normalization_params_deduplicated.json`
- Full mapping audit trail in `neighborhood_mapping_report.json`
- Can regenerate exactly the same output

📊 **Temporal Coverage**
- Starts: 2025-01-02
- Ends: 2026-01-11
- Days: 375 (covers full 12-month period)
- All missing dates zero-filled

## Next Phase: Feature Engineering

Ready to proceed with:
1. Temporal lag features (7-day, 30-day MA)
2. Operation intensity indicators
3. Faction distribution encoding
4. One-hot encoding for categorical variables
5. Feeding into ST-GCN as node features
