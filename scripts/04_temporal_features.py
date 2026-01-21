"""
Phase 2: Generate temporal features from normalized seizure data

This script creates:
1. Lagged features (t-1, t-7, t-30 days)
2. Moving averages (7-day, 30-day)
3. Intensity scores
4. Volatility measures
5. Cyclical temporal features (day of week, month)
6. Faction distribution features

Output: data/processed/prisoes_with_features.parquet
"""

import sys
import os
import json
import logging
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from features.temporal_features import (
    TemporalFeatureEngineer,
    FactionDistributionFeatures,
    load_normalized_data,
    save_featured_data
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / 'data' / 'processed'
NORMALIZED_DATA = DATA_PATH / 'prisoes_normalized_deduplicated.parquet'
OUTPUT_DATA = DATA_PATH / 'prisoes_with_features.parquet'
FEATURE_METADATA = DATA_PATH / 'feature_metadata.json'


def main():
    """Main feature engineering pipeline."""
    
    logger.info("=" * 80)
    logger.info("PHASE 2: TEMPORAL FEATURE ENGINEERING")
    logger.info("=" * 80)
    
    # ========================================================================
    # Step 1: Load normalized data
    # ========================================================================
    logger.info("\n[1/4] Loading normalized data...")
    if not NORMALIZED_DATA.exists():
        logger.error(f"File not found: {NORMALIZED_DATA}")
        logger.error("Run Phase 1 first: scripts/02_normalize_with_deduplication.py")
        sys.exit(1)
    
    df = load_normalized_data(str(NORMALIZED_DATA))
    initial_shape = df.shape
    logger.info(f"  ✓ Loaded {initial_shape[0]} records, {initial_shape[1]} columns")
    
    # Verify required columns
    required_cols = ['Data', 'bairro_id', 'drogas_gramas_total_norm', 
                     'armas_total_norm', 'dinheiro_total_reais_norm']
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        logger.error(f"Missing columns: {missing_cols}")
        sys.exit(1)
    
    # ========================================================================
    # Step 2: Create temporal features
    # ========================================================================
    logger.info("\n[2/4] Creating temporal features...")
    
    temporal_engineer = TemporalFeatureEngineer(
        df,
        neighborhood_col='bairro_id',
        date_col='Data',
        seizure_cols=[
            'drogas_gramas_total_norm',
            'armas_total_norm',
            'dinheiro_total_reais_norm'
        ]
    )
    
    df_temporal = temporal_engineer.create_all_temporal_features(
        lag_periods=[1, 7, 30],
        ma_windows=[7, 30],
        volatility_window=7
    )
    
    temporal_stats = temporal_engineer.get_feature_stats(df_temporal)
    logger.info(f"  ✓ Created {temporal_stats['total_features']} temporal features:")
    for group, count in temporal_stats['feature_groups'].items():
        logger.info(f"    - {group}: {count}")
    
    # ========================================================================
    # Step 3: Data quality checks
    # ========================================================================
    logger.info("\n[3/4] Data quality validation...")
    
    df_with_features = df_temporal
    
    # Check for NaN values
    nan_counts = df_with_features.isnull().sum()
    nan_cols = nan_counts[nan_counts > 0]
    if len(nan_cols) > 0:
        logger.warning(f"  ⚠ Found NaN values:")
        for col, count in nan_cols.items():
            logger.warning(f"    - {col}: {count} ({100*count/len(df_with_features):.2f}%)")
    else:
        logger.info("  ✓ No NaN values found")
    
    # Check feature ranges
    numeric_cols = df_with_features.select_dtypes(include=[np.number]).columns
    logger.info(f"  ✓ {len(numeric_cols)} numeric features")
    
    # Verify normalized columns stay in [0, 1]
    norm_cols = [c for c in df_with_features.columns if '_norm' in c]
    for col in norm_cols:
        min_val = df_with_features[col].min()
        max_val = df_with_features[col].max()
        if min_val < -0.01 or max_val > 1.01:  # Allow small floating point tolerance
            logger.warning(f"  ⚠ {col} out of bounds: [{min_val:.4f}, {max_val:.4f}]")
        else:
            logger.info(f"  ✓ {col} in valid range: [{min_val:.4f}, {max_val:.4f}]")
    
    # Check duplicates
    duplicate_count = df_with_features.duplicated(
        subset=['Data', 'bairro_id']
    ).sum()
    if duplicate_count > 0:
        logger.warning(f"  ⚠ Found {duplicate_count} duplicate (Data, bairro_id) pairs")
    else:
        logger.info("  ✓ No duplicate (Data, bairro_id) pairs")
    
    # ========================================================================
    # Step 4: Save results
    # ========================================================================
    logger.info("\n[4/4] Saving results...")
    
    save_featured_data(df_with_features, str(OUTPUT_DATA))
    logger.info(f"  ✓ Saved to {OUTPUT_DATA}")
    
    # Save metadata
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'input_file': str(NORMALIZED_DATA),
        'output_file': str(OUTPUT_DATA),
        'input_shape': initial_shape,
        'output_shape': df_with_features.shape,
        'columns': list(df_with_features.columns),
        'temporal_features': temporal_stats['total_features'],
        'feature_groups': temporal_stats['feature_groups'],
        'unique_neighborhoods': int(df_with_features['bairro_id'].nunique()),
        'date_range': {
            'start': str(df_with_features['Data'].min()),
            'end': str(df_with_features['Data'].max()),
            'days': int((df_with_features['Data'].max() - df_with_features['Data'].min()).days) + 1
        }
    }
    
    with open(FEATURE_METADATA, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    logger.info(f"  ✓ Saved metadata to {FEATURE_METADATA}")
    
    # ========================================================================
    # Summary
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("FEATURE ENGINEERING SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Input records:              {initial_shape[0]:,}")
    logger.info(f"Output records:             {df_with_features.shape[0]:,}")
    logger.info(f"Input columns:              {initial_shape[1]}")
    logger.info(f"Output columns:             {df_with_features.shape[1]}")
    logger.info(f"New features created:       {df_with_features.shape[1] - initial_shape[1]}")
    logger.info(f"  - Temporal features:      {temporal_stats['total_features']}")
    logger.info(f"Neighborhoods:              {df_with_features['bairro_id'].nunique()}")
    logger.info(f"Temporal coverage:          {metadata['date_range']['days']} days")
    logger.info("=" * 80)
    logger.info("✅ Phase 2 Complete!")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
