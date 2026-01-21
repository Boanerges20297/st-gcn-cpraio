"""
Script to test and apply fuzzy matching to CidadeOcor field.
"""

import sys
import logging
from pathlib import Path
import pandas as pd
import json

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data.operations_loader import OperationsLoader
from data.city_deduplicator import CityDeduplicator
from data.ceara_municipalities import CEARA_MUNICIPALITIES

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/deduplicate_cities.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 80)
    logger.info("CITY NAME DEDUPLICATION - FUZZY MATCHING")
    logger.info("=" * 80)
    
    try:
        # Step 1: Load operations data
        logger.info("\n📂 Step 1: Loading operations data...")
        loader = OperationsLoader('data/raw/ocorrencia_policial_operacional.json')
        df = loader.load()
        
        logger.info(f"✓ Loaded {len(df)} operations")
        logger.info(f"✓ Raw unique cities: {df['CidadeOcor'].nunique()}")
        
        # Step 2: Apply fuzzy matching with 50% threshold
        logger.info(f"\n🔍 Step 2: Applying fuzzy matching (threshold: 50%)...")
        dedup = CityDeduplicator(similarity_threshold=0.5)
        df_dedup = dedup.deduplicate_dataframe(df, 'CidadeOcor')
        
        # Step 3: Analyze results
        logger.info(f"\n📊 DEDUPLICATION RESULTS:")
        
        standardized_col = 'CidadeOcor_standardized'
        
        if standardized_col in df_dedup.columns:
            matched_count = df_dedup[standardized_col].notna().sum()
            unmatched_count = df_dedup[standardized_col].isna().sum()
            
            logger.info(f"   Total operations: {len(df_dedup)}")
            logger.info(f"   Matched operations: {matched_count} ({100*matched_count/len(df_dedup):.1f}%)")
            logger.info(f"   Unmatched operations: {unmatched_count} ({100*unmatched_count/len(df_dedup):.1f}%)")
            
            matched_unique = df_dedup[standardized_col].nunique()
            logger.info(f"   Unique standardized cities: {matched_unique}")
            
            # Show mapping examples
            logger.info(f"\n📋 MAPPING EXAMPLES:")
            sample_mappings = df_dedup[df_dedup[standardized_col].notna()][
                ['CidadeOcor', standardized_col]
            ].drop_duplicates('CidadeOcor').head(20)
            
            for idx, row in sample_mappings.iterrows():
                raw = row['CidadeOcor']
                std = row[standardized_col]
                if raw != std:
                    logger.info(f"   '{raw}' → '{std}'")
                else:
                    logger.info(f"   '{raw}' (exact match)")
            
            # Get unmapped cities
            logger.info(f"\n⚠️ UNMAPPED CITIES ({len(dedup.get_unmapped_values())}):")
            unmapped = sorted(dedup.get_unmapped_values())
            for city in unmapped[:20]:
                logger.info(f"   - {city}")
            if len(unmapped) > 20:
                logger.info(f"   ... and {len(unmapped) - 20} more")
            
            # Get statistics
            logger.info(f"\n📊 STATISTICS:")
            stats = dedup.get_mapping_stats()
            logger.info(f"   Official municipalities: {stats['official_cities']}")
            logger.info(f"   Unique raw city names: {len(dedup.mapping_cache)}")
            logger.info(f"   Successfully mapped: {stats['mapped_raw_names']}")
            logger.info(f"   Unmapped: {stats['unmapped_raw_names']}")
            logger.info(f"   Threshold: {stats['similarity_threshold']*100:.0f}%")
            
        logger.info(f"\n✅ City deduplication analysis complete!")
        
        return df_dedup, dedup
        
    except Exception as e:
        logger.error(f"\n❌ Error: {e}", exc_info=True)
        return None, None


if __name__ == '__main__':
    df_result, dedup_result = main()
