"""
City name deduplication using fuzzy matching.
Standardizes CidadeOcor field against official Ceará municipalities.
"""

import logging
from typing import Dict, Tuple, Set, Optional
from difflib import SequenceMatcher
import pandas as pd
from .ceara_municipalities import CEARA_MUNICIPALITIES

logger = logging.getLogger(__name__)


class CityDeduplicator:
    """
    Standardizes city names using fuzzy string matching.
    Maps raw CidadeOcor values to official Ceará municipalities.
    """
    
    def __init__(self, similarity_threshold: float = 0.5):
        """
        Initialize the deduplicator.
        
        Args:
            similarity_threshold: Minimum character similarity to match (0-1)
        """
        self.threshold = similarity_threshold
        self.official_cities = set(CEARA_MUNICIPALITIES)
        self.mapping_cache = {}
        self.unmapped_values = set()
        
        logger.info(f"✓ Loaded {len(self.official_cities)} official Ceará municipalities")
    
    def _calculate_similarity(self, str1: str, str2: str) -> float:
        """
        Calculate character-level similarity between two strings (0-1).
        
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
        Find the best matching official city for a raw name.
        
        Args:
            raw_name: Raw city name to match
        
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
        
        # Compare against all official cities
        for official in self.official_cities:
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
                             cidade_column: str = 'CidadeOcor') -> pd.DataFrame:
        """
        Apply fuzzy matching to standardize cities in a dataframe.
        
        Args:
            df: DataFrame with raw city names
            cidade_column: Name of column containing raw city names
        
        Returns:
            DataFrame with standardized city names
        """
        df = df.copy()
        
        if cidade_column not in df.columns:
            logger.warning(f"Column '{cidade_column}' not found in dataframe")
            return df
        
        # Apply fuzzy matching
        def match_city(raw_name):
            result = self.find_best_match(raw_name)
            return result[0] if result else None
        
        df[f'{cidade_column}_standardized'] = df[cidade_column].apply(match_city)
        
        # Count matches
        matched = df[f'{cidade_column}_standardized'].notna().sum()
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
            'official_cities': len(self.official_cities),
            'mapped_raw_names': len([v for v in self.mapping_cache.values() if v is not None]),
            'unmapped_raw_names': len(self.unmapped_values),
            'similarity_threshold': self.threshold,
        }
        return stats
    
    def get_unmapped_values(self) -> Set[str]:
        """Get all raw city names that couldn't be matched."""
        return self.unmapped_values.copy()
