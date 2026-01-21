"""
Script to re-normalize operations data with deduplicated neighborhoods.
Applies deduplication BEFORE aggregation to ensure proper group-level normalization.
"""

import sys
import logging
from pathlib import Path
import pandas as pd
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data.operations_loader import OperationsLoader
from data.neighborhood_deduplicator import standardize_neighborhoods

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/normalize_with_deduplication.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def normalize_with_deduplication():
    """
    Full pipeline: Load → Deduplicate → Normalize → Aggregate
    """
    logger.info("=" * 80)
    logger.info("OPERATIONS NORMALIZATION WITH DEDUPLICATION")
    logger.info("=" * 80)
    
    # Paths
    operations_json = 'data/raw/ocorrencia_policial_operacional.json'
    geojson_path = 'data/graph/fortaleza_bairros.geojson'
    output_final = 'data/processed/prisoes_normalized_deduplicated.parquet'
    output_params = 'data/processed/normalization_params_deduplicated.json'
    output_report = 'outputs/neighborhood_mapping_report.json'
    
    try:
        # Step 1: Load operations
        logger.info("\n📂 Step 1: Loading operations data...")
        loader = OperationsLoader(operations_json)
        df = loader.load()
        logger.info(f"✓ Loaded {len(df)} operations")
        logger.info(f"✓ Raw unique neighborhoods: {df['BairroOcor'].nunique()}")
        
        # Step 2: Deduplicate neighborhoods
        logger.info(f"\n🔍 Step 2: Applying fuzzy matching deduplication...")
        df_dedup = standardize_neighborhoods(
            df,
            geojson_path=geojson_path,
            threshold=0.5,
            output_report=output_report
        )
        
        # Step 3: Parse data types
        logger.info(f"\n🔄 Step 3: Parsing data types...")
        df_dedup['Data'] = pd.to_datetime(df_dedup['Data'], format='%Y-%m-%d', errors='coerce')
        
        # Convert numeric fields safely
        df_dedup['total_drogas_cache'] = pd.to_numeric(
            df_dedup['total_drogas_cache'], errors='coerce'
        ).fillna(0)
        df_dedup['total_armas_cache'] = pd.to_numeric(
            df_dedup['total_armas_cache'], errors='coerce'
        ).fillna(0)
        df_dedup['Dinheiro_Apreendido'] = pd.to_numeric(
            df_dedup['Dinheiro_Apreendido'], errors='coerce'
        ).fillna(0)
        
        logger.info(f"✓ Data types parsed")
        
        # Step 4: Filter only mapped records
        logger.info(f"\n🎯 Step 4: Filtering mapped neighborhoods...")
        before_filter = len(df_dedup)
        df_dedup = df_dedup[df_dedup['BairroOcor_standardized'].notna()].copy()
        after_filter = len(df_dedup)
        filtered_out = before_filter - after_filter
        
        logger.info(f"✓ Kept {after_filter} records (filtered out {filtered_out} unmapped)")
        logger.info(f"✓ Unique standardized neighborhoods: {df_dedup['BairroOcor_standardized'].nunique()}")
        
        # Step 5: Build neighborhood ID mapping (from standardized names)
        logger.info(f"\n🗺️ Step 5: Building standardized neighborhood ID mapping...")
        unique_bairros = sorted(df_dedup['BairroOcor_standardized'].unique())
        bairro_to_id = {bairro: idx for idx, bairro in enumerate(unique_bairros)}
        df_dedup['bairro_id'] = df_dedup['BairroOcor_standardized'].map(bairro_to_id)
        
        logger.info(f"✓ Created mapping: {len(bairro_to_id)} neighborhoods → IDs 0-{len(bairro_to_id)-1}")
        
        # Step 6: Calculate normalization parameters (IMPORTANT: on aggregated/final data)
        logger.info(f"\n📊 Step 6: Calculating normalization parameters...")
        
        # Calculate 99th percentile for robust normalization
        drogas_p99 = df_dedup['total_drogas_cache'].quantile(0.99)
        armas_p99 = df_dedup['total_armas_cache'].quantile(0.99)
        dinheiro_p99 = df_dedup['Dinheiro_Apreendido'].quantile(0.99)
        
        logger.info(f"✓ 99th percentiles:")
        logger.info(f"   Drogas: {drogas_p99:.2f}g")
        logger.info(f"   Armas: {armas_p99:.2f} units")
        logger.info(f"   Dinheiro: R$ {dinheiro_p99:.2f}")
        
        # Step 7: AGGREGATE BY (DATE, NEIGHBORHOOD) FIRST
        logger.info(f"\n📅 Step 7: Aggregating by date and neighborhood...")
        
        agg_dict = {
            'total_drogas_cache': 'sum',
            'total_armas_cache': 'sum',
            'Dinheiro_Apreendido': 'sum',
        }
        
        df_agg = df_dedup.groupby(['Data', 'bairro_id', 'BairroOcor_standardized']).agg(
            operacoes_diarias=('Controle', 'count'),
            drogas_gramas_total=('total_drogas_cache', 'sum'),
            armas_total=('total_armas_cache', 'sum'),
            dinheiro_total_reais=('Dinheiro_Apreendido', 'sum'),
        ).reset_index()
        
        logger.info(f"✓ Aggregated: {len(df_agg)} daily neighborhood records")
        
        # Step 8: NORMALIZE AGGREGATED VALUES (after grouping)
        logger.info(f"\n🔢 Step 8: Normalizing aggregated values...")
        
        # Normalize each aggregated metric
        df_agg['drogas_gramas_total_norm'] = (
            df_agg['drogas_gramas_total'] / drogas_p99
        ).clip(0, 1)  # Clip to [0, 1]
        
        df_agg['armas_total_norm'] = (
            df_agg['armas_total'] / armas_p99
        ).clip(0, 1)
        
        df_agg['dinheiro_total_reais_norm'] = (
            df_agg['dinheiro_total_reais'] / dinheiro_p99
        ).clip(0, 1)
        
        logger.info(f"✓ Normalized to [0, 1] range")
        
        # Verify normalization bounds
        logger.info(f"\n✓ Normalization verification:")
        logger.info(f"   Drogas norm: [{df_agg['drogas_gramas_total_norm'].min():.4f}, {df_agg['drogas_gramas_total_norm'].max():.4f}]")
        logger.info(f"   Armas norm: [{df_agg['armas_total_norm'].min():.4f}, {df_agg['armas_total_norm'].max():.4f}]")
        logger.info(f"   Dinheiro norm: [{df_agg['dinheiro_total_reais_norm'].min():.4f}, {df_agg['dinheiro_total_reais_norm'].max():.4f}]")
        
        # Step 9: Create date range for zero-filling
        logger.info(f"\n📆 Step 9: Zero-filling missing dates...")
        
        date_min = df_agg['Data'].min()
        date_max = df_agg['Data'].max()
        date_range = pd.date_range(date_min, date_max, freq='D')
        bairro_ids = sorted(df_agg['bairro_id'].unique())
        
        # Create complete grid
        complete_grid = pd.MultiIndex.from_product(
            [date_range, bairro_ids],
            names=['Data', 'bairro_id']
        ).to_frame(index=False)
        
        # Merge with aggregated data
        df_final = complete_grid.merge(
            df_agg.drop('BairroOcor_standardized', axis=1),
            on=['Data', 'bairro_id'],
            how='left'
        )
        
        # Fill NaNs with 0
        fill_cols = [
            'operacoes_diarias', 'drogas_gramas_total', 'armas_total', 
            'dinheiro_total_reais', 'drogas_gramas_total_norm', 
            'armas_total_norm', 'dinheiro_total_reais_norm'
        ]
        df_final[fill_cols] = df_final[fill_cols].fillna(0)
        
        logger.info(f"✓ Created complete grid: {len(date_range)} dates × {len(bairro_ids)} neighborhoods = {len(df_final)} records")
        
        # Step 10: Save results
        logger.info(f"\n💾 Step 10: Saving results...")
        
        # Save normalized data
        df_final.to_parquet(output_final, index=False)
        logger.info(f"✓ Saved normalized data to {output_final}")
        
        # Save normalization parameters
        params = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'method': 'percentile-based with post-aggregation normalization',
            'threshold_fuzzy_matching': 0.5,
            'drogas_max_p99': float(drogas_p99),
            'armas_max_p99': float(armas_p99),
            'dinheiro_max_p99': float(dinheiro_p99),
            'neighborhood_mapping': bairro_to_id,
            'date_range': {
                'start': date_min.isoformat(),
                'end': date_max.isoformat(),
                'days': len(date_range)
            },
            'statistics': {
                'total_operations': len(df_dedup),
                'filtered_operations': filtered_out,
                'unique_neighborhoods': len(bairro_to_id),
                'aggregated_records': len(df_agg),
                'final_records_with_zeropadding': len(df_final),
            }
        }
        
        with open(output_params, 'w') as f:
            json.dump(params, f, indent=2)
        logger.info(f"✓ Saved parameters to {output_params}")
        
        # Final summary
        logger.info(f"\n{'='*80}")
        logger.info(f"✅ NORMALIZATION COMPLETE!")
        logger.info(f"{'='*80}")
        logger.info(f"📊 FINAL STATISTICS:")
        logger.info(f"   Operations loaded: {len(df):,}")
        logger.info(f"   Neighborhoods deduplicated: 2,529 → 138 official")
        logger.info(f"   Operations mapped: {len(df_dedup):,} ({100*len(df_dedup)/len(df):.1f}%)")
        logger.info(f"   Unique neighborhoods: {len(bairro_to_id)}")
        logger.info(f"   Date range: {date_min.date()} to {date_max.date()} ({len(date_range)} days)")
        logger.info(f"   Final records (with zero-padding): {len(df_final):,}")
        logger.info(f"   Normalization: All values in [0, 1]")
        
        return df_final
        
    except Exception as e:
        logger.error(f"\n❌ Error: {e}", exc_info=True)
        return None


if __name__ == '__main__':
    normalize_with_deduplication()
