"""
Fuzzy matching for neighborhood name standardization.
Compares raw BairroOcor values against official Fortaleza neighborhoods
with configurable similarity threshold.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Tuple, Set, Optional
from difflib import SequenceMatcher
import pandas as pd

logger = logging.getLogger(__name__)


class NeighborhoodDeduplicator:
    """
    Standardizes neighborhood names using fuzzy string matching.
    Maps raw BairroOcor values to official Fortaleza neighborhoods.
    """
    
    def __init__(self, similarity_threshold: float = 0.5):
        """
        Initialize the deduplicator.
        
        Args:
            similarity_threshold: Minimum character similarity to match (0-1)
        """
        self.threshold = similarity_threshold
        self.official_neighborhoods = set()
        self.mapping_cache = {}
        self.unmapped_values = set()
        
    def load_official_neighborhoods(self, geojson_path: str) -> None:
        """
        Load official neighborhood names from GeoJSON file.
        
        Args:
            geojson_path: Path to fortaleza_bairros.geojson
        """
        try:
            with open(geojson_path, 'r', encoding='utf-8') as f:
                geojson = json.load(f)
            
            # Extract neighborhood names from GeoJSON features
            for feature in geojson.get('features', []):
                props = feature.get('properties', {})
                name = props.get('name', '').strip()
                if name:
                    self.official_neighborhoods.add(name.upper())
            
            logger.info(f"✓ Loaded {len(self.official_neighborhoods)} official neighborhoods")
            
        except Exception as e:
            logger.error(f"✗ Failed to load GeoJSON: {e}")
            raise
    
    def _calculate_similarity(self, str1: str, str2: str) -> float:
        """
        Calculate character-level similarity between two strings (0-1).
        Uses SequenceMatcher ratio.
        
        Args:
            str1: First string
            str2: Second string
        
        Returns:
            Similarity score (0-1)
        """
        s1 = str1.upper().strip()
        s2 = str2.upper().strip()
        
        # Exact match
        if s1 == s2:
            return 1.0
        
        # Use SequenceMatcher for character-level similarity
        return SequenceMatcher(None, s1, s2).ratio()
    
    def find_best_match(self, raw_name: str) -> Optional[Tuple[str, float]]:
        """
        Find the best matching official neighborhood for a raw name.
        
        Args:
            raw_name: Raw neighborhood name to match
        
        Returns:
            Tuple of (matched_official_name, similarity_score) or None if no match
        """
        if not raw_name or pd.isna(raw_name):
            return None
        
        raw_name = str(raw_name).strip()
        
        # Check cache first
        if raw_name in self.mapping_cache:
            return self.mapping_cache[raw_name]
        
        best_match = None
        best_score = 0.0
        
        # Compare against all official neighborhoods
        for official in self.official_neighborhoods:
            similarity = self._calculate_similarity(raw_name, official)
            
            # Track best match if above threshold
            if similarity >= self.threshold and similarity > best_score:
                best_match = official
                best_score = similarity
        
        # Cache result
        result = (best_match, best_score) if best_match else None
        self.mapping_cache[raw_name] = result
        
        if not best_match:
            self.unmapped_values.add(raw_name)
        
        return result
    
    def deduplicate_dataframe(self, df: pd.DataFrame, 
                             bairro_column: str = 'BairroOcor') -> pd.DataFrame:
        """
        Apply fuzzy matching to deduplicate neighborhoods in a dataframe.
        
        Args:
            df: DataFrame with raw neighborhood names
            bairro_column: Name of column containing raw neighborhood names
        
        Returns:
            DataFrame with standardized neighborhood names
        """
        df = df.copy()
        
        if bairro_column not in df.columns:
            logger.warning(f"Column '{bairro_column}' not found in dataframe")
            return df
        
        # Apply fuzzy matching
        def match_neighborhood(raw_name):
            result = self.find_best_match(raw_name)
            return result[0] if result else None
        
        df[f'{bairro_column}_standardized'] = df[bairro_column].apply(match_neighborhood)
        
        # Count matches
        matched = df[f'{bairro_column}_standardized'].notna().sum()
        total = len(df)
        
        logger.info(f"✓ Matched {matched}/{total} records ({100*matched/total:.1f}%)")
        
        return df
    
    def get_mapping_stats(self) -> Dict:
        """
        Get statistics about the mapping process.
        
        Returns:
            Dictionary with mapping statistics
        """
        stats = {
            'official_neighborhoods': len(self.official_neighborhoods),
            'mapped_raw_names': len([v for v in self.mapping_cache.values() if v is not None]),
            'unmapped_raw_names': len(self.unmapped_values),
            'similarity_threshold': self.threshold,
        }
        return stats
    
    def get_unmapped_values(self) -> Set[str]:
        """Get all raw neighborhood names that couldn't be matched."""
        return self.unmapped_values.copy()
    
    def export_mapping_report(self, output_path: str) -> None:
        """
        Export detailed mapping report for audit trail.
        
        Args:
            output_path: Path to save report JSON
        """
        report = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'threshold': self.threshold,
            'official_neighborhoods': sorted(list(self.official_neighborhoods)),
            'mapping_results': {
                'total_unique_raw_values': len(self.mapping_cache),
                'successfully_matched': len([v for v in self.mapping_cache.values() if v is not None]),
                'unmapped': len(self.unmapped_values),
            },
            'unmapped_values': sorted(list(self.unmapped_values)),
            'mapping_cache': {k: {'matched': v[0], 'similarity': v[1]} if v else {'matched': None, 'similarity': 0.0} 
                            for k, v in self.mapping_cache.items()},
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✓ Mapping report saved to {output_path}")


def standardize_neighborhoods(
    operations_df: pd.DataFrame,
    geojson_path: str = 'data/graph/fortaleza_bairros.geojson',
    threshold: float = 0.5,
    output_report: Optional[str] = None
) -> pd.DataFrame:
    """
    Convenience function to standardize neighborhoods in operations data.
    
    Args:
        operations_df: DataFrame with operations data
        geojson_path: Path to official neighborhoods GeoJSON
        threshold: Similarity threshold (0-1)
        output_report: Optional path to save mapping report
    
    Returns:
        DataFrame with standardized neighborhoods
    """
    dedup = NeighborhoodDeduplicator(similarity_threshold=threshold)
    dedup.load_official_neighborhoods(geojson_path)
    
    df_deduplicated = dedup.deduplicate_dataframe(operations_df, 'BairroOcor')
    
    if output_report:
        dedup.export_mapping_report(output_report)
    
    # Log statistics
    stats = dedup.get_mapping_stats()
    logger.info(f"\n📊 DEDUPLICATION STATISTICS:")
    logger.info(f"   Official Neighborhoods: {stats['official_neighborhoods']}")
    logger.info(f"   Matched Raw Names: {stats['mapped_raw_names']}")
    logger.info(f"   Unmapped Raw Names: {stats['unmapped_raw_names']}")
    logger.info(f"   Threshold: {stats['similarity_threshold']*100:.0f}%")
    
    # Warn about unmapped values
    unmapped = dedup.get_unmapped_values()
    if unmapped:
        logger.warning(f"\n⚠ {len(unmapped)} raw neighborhood names could not be matched:")
        for name in sorted(list(unmapped))[:20]:  # Show first 20
            logger.warning(f"   - {name}")
        if len(unmapped) > 20:
            logger.warning(f"   ... and {len(unmapped) - 20} more")
    
    return df_deduplicated
