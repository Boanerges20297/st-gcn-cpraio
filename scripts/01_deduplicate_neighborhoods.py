"""
Script to apply fuzzy matching deduplication to operations data.
Tests multiple similarity thresholds to find optimal mapping.
"""

import sys
import logging
from pathlib import Path
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data.operations_loader import OperationsLoader
from data.neighborhood_deduplicator import standardize_neighborhoods

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/deduplicate_neighborhoods.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 80)
    logger.info("NEIGHBORHOOD DEDUPLICATION - FUZZY MATCHING")
    logger.info("=" * 80)
    
    # Paths
    operations_json = 'data/raw/ocorrencia_policial_operacional.json'
    geojson_path = 'data/graph/fortaleza_bairros.geojson'
    output_deduplicated = 'data/processed/operacoes_deduplicated.parquet'
    output_report = 'outputs/neighborhood_mapping_report.json'
    
    try:
        # Step 1: Load operations data
        logger.info("\n📂 Loading operations data...")
        loader = OperationsLoader(operations_json)
        operations_df = loader.load()
        
        logger.info(f"✓ Loaded {len(operations_df)} operations")
        logger.info(f"✓ Raw unique neighborhoods: {operations_df['BairroOcor'].nunique()}")
        
        # Step 2: Apply fuzzy matching with 50% threshold
        logger.info(f"\n🔍 Applying fuzzy matching (threshold: 50%)...")
        df_deduplicated = standardize_neighborhoods(
            operations_df,
            geojson_path=geojson_path,
            threshold=0.5,
            output_report=output_report
        )
        
        # Step 3: Analyze results
        logger.info(f"\n📊 DEDUPLICATION RESULTS:")
        
        # Check standardized neighborhoods
        standardized_col = 'BairroOcor_standardized'
        
        if standardized_col in df_deduplicated.columns:
            matched_count = df_deduplicated[standardized_col].notna().sum()
            unmatched_count = df_deduplicated[standardized_col].isna().sum()
            
            logger.info(f"   Matched operations: {matched_count} ({100*matched_count/len(df_deduplicated):.1f}%)")
            logger.info(f"   Unmatched operations: {unmatched_count} ({100*unmatched_count/len(df_deduplicated):.1f}%)")
            
            matched_unique = df_deduplicated[standardized_col].nunique()
            logger.info(f"   Unique standardized neighborhoods: {matched_unique}")
            
            # Show mapping examples
            logger.info(f"\n📋 MAPPING EXAMPLES:")
            sample_mappings = df_deduplicated[df_deduplicated[standardized_col].notna()][
                ['BairroOcor', standardized_col]
            ].drop_duplicates('BairroOcor').head(15)
            
            for idx, row in sample_mappings.iterrows():
                raw = row['BairroOcor']
                std = row[standardized_col]
                if raw != std:
                    logger.info(f"   '{raw}' → '{std}'")
        
        # Step 4: Save deduplicated data
        logger.info(f"\n💾 Saving deduplicated operations...")
        df_deduplicated.to_parquet(output_deduplicated)
        logger.info(f"✓ Saved to {output_deduplicated}")
        logger.info(f"✓ Report saved to {output_report}")
        
        logger.info(f"\n✅ Deduplication complete!")
        
        return df_deduplicated
        
    except Exception as e:
        logger.error(f"\n❌ Error during deduplication: {e}", exc_info=True)
        return None


if __name__ == '__main__':
    main()
