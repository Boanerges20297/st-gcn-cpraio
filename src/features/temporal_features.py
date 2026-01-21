"""
Feature Engineering Module for ST-GCN

Generates temporal and spatial features from normalized seizure data.
Creates lagged features, moving averages, intensity scores, and faction distributions.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class TemporalFeatureEngineer:
    """Generate temporal features for crime prediction."""
    
    def __init__(self, df: pd.DataFrame, neighborhood_col: str = 'bairro_id',
                 date_col: str = 'Data', seizure_cols: List[str] = None):
        """
        Initialize feature engineer.
        
        Args:
            df: DataFrame with normalized data (from prisoes_normalized_deduplicated.parquet)
            neighborhood_col: Column name for neighborhood identifiers
            date_col: Column name for dates
            seizure_cols: Columns to create lagged features from
                         Default: ['drogas_gramas_total_norm', 'armas_total_norm', 'dinheiro_total_norm']
        """
        self.df = df.sort_values([date_col, neighborhood_col]).reset_index(drop=True)
        self.neighborhood_col = neighborhood_col
        self.date_col = date_col
        self.seizure_cols = seizure_cols or [
            'drogas_gramas_total_norm',
            'armas_total_norm',
            'dinheiro_total_norm'
        ]
        
        # Validate that required columns exist
        for col in [neighborhood_col, date_col] + self.seizure_cols:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in DataFrame")
        
        logger.info(f"Initialized TemporalFeatureEngineer with {len(df)} records, "
                   f"{df[neighborhood_col].nunique()} neighborhoods, "
                   f"{df[date_col].nunique()} time steps")
    
    def create_lag_features(self, lags: List[int] = [1, 7, 30]) -> pd.DataFrame:
        """
        Create lagged features for each seizure column.
        
        Args:
            lags: List of lag periods (days)
        
        Returns:
            DataFrame with original columns + lag features
        """
        df = self.df.copy()
        
        for col in self.seizure_cols:
            for lag in lags:
                # Group by neighborhood to ensure lags are within same neighborhood
                df[f'{col}_lag{lag}'] = df.groupby(self.neighborhood_col)[col].shift(lag)
        
        # Fill initial NaN values with 0 (no previous data)
        lag_cols = [c for c in df.columns if '_lag' in c]
        df[lag_cols] = df[lag_cols].fillna(0)
        
        logger.info(f"Created {len(lag_cols)} lag features (lags: {lags})")
        return df
    
    def create_moving_averages(self, windows: List[int] = [7, 30]) -> pd.DataFrame:
        """
        Create moving average features.
        
        Args:
            windows: Window sizes (days) for moving averages
        
        Returns:
            DataFrame with moving average features
        """
        df = self.df.copy()
        
        for col in self.seizure_cols:
            for window in windows:
                ma_col = f'{col}_ma{window}'
                # Calculate moving average grouped by neighborhood
                df[ma_col] = df.groupby(self.neighborhood_col)[col].transform(
                    lambda x: x.rolling(window=window, min_periods=1).mean()
                )
        
        ma_cols = [c for c in df.columns if '_ma' in c]
        logger.info(f"Created {len(ma_cols)} moving average features (windows: {windows})")
        return df
    
    def create_intensity_score(self) -> pd.DataFrame:
        """
        Create composite intensity score combining all seizure types.
        
        Intensity = (drogas_norm + armas_norm + dinheiro_norm) / 3
        
        Returns:
            DataFrame with intensity_score column
        """
        df = self.df.copy()
        
        # Get the dinheiro column (might be named differently)
        dinheiro_col = [c for c in df.columns if 'dinheiro' in c.lower() and '_norm' in c][0]
        
        # Weighted combination of normalized seizures
        df['intensity_score'] = (
            0.4 * df['drogas_gramas_total_norm'] +  # Higher weight for drugs
            0.3 * df['armas_total_norm'] +           # Medium weight for weapons
            0.3 * df[dinheiro_col]                   # Medium weight for money
        )
        
        # Clip to [0, 1]
        df['intensity_score'] = df['intensity_score'].clip(0, 1)
        
        logger.info("Created intensity_score feature")
        return df
    
    def create_operation_volatility(self, window: int = 7) -> pd.DataFrame:
        """
        Create volatility feature (rolling std of seizures).
        
        Captures how unpredictable seizures are in each neighborhood.
        
        Args:
            window: Window size (days) for rolling std
        
        Returns:
            DataFrame with volatility features
        """
        df = self.df.copy()
        
        for col in self.seizure_cols:
            vol_col = f'{col}_volatility'
            df[vol_col] = df.groupby(self.neighborhood_col)[col].transform(
                lambda x: x.rolling(window=window, min_periods=1).std()
            )
            # Fill NaN with 0
            df[vol_col] = df[vol_col].fillna(0)
        
        vol_cols = [c for c in df.columns if '_volatility' in c]
        logger.info(f"Created {len(vol_cols)} volatility features (window: {window})")
        return df
    
    def create_day_of_week_features(self) -> pd.DataFrame:
        """
        Create day-of-week cyclical features (sin/cos encoding).
        
        Returns:
            DataFrame with day_sin and day_cos columns
        """
        df = self.df.copy()
        df['date_parsed'] = pd.to_datetime(df[self.date_col])
        
        # Day of week (0=Monday, 6=Sunday)
        day_of_week = df['date_parsed'].dt.dayofweek
        
        # Cyclical encoding (sine and cosine)
        df['day_of_week_sin'] = np.sin(2 * np.pi * day_of_week / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * day_of_week / 7)
        
        df = df.drop('date_parsed', axis=1)
        
        logger.info("Created day-of-week cyclical features")
        return df
    
    def create_month_of_year_features(self) -> pd.DataFrame:
        """
        Create month-of-year cyclical features (sin/cos encoding).
        
        Returns:
            DataFrame with month_sin and month_cos columns
        """
        df = self.df.copy()
        df['date_parsed'] = pd.to_datetime(df[self.date_col])
        
        # Month (1-12)
        month = df['date_parsed'].dt.month
        
        # Cyclical encoding
        df['month_sin'] = np.sin(2 * np.pi * month / 12)
        df['month_cos'] = np.cos(2 * np.pi * month / 12)
        
        df = df.drop('date_parsed', axis=1)
        
        logger.info("Created month-of-year cyclical features")
        return df
    
    def create_all_temporal_features(self, 
                                    lag_periods: List[int] = [1, 7, 30],
                                    ma_windows: List[int] = [7, 30],
                                    volatility_window: int = 7) -> pd.DataFrame:
        """
        Create all temporal features at once.
        
        Args:
            lag_periods: Lag periods for lagged features
            ma_windows: Windows for moving averages
            volatility_window: Window for volatility calculation
        
        Returns:
            DataFrame with all features added
        """
        logger.info("Starting feature engineering pipeline...")
        
        # Sequential application
        df = self.create_lag_features(lag_periods)
        df = pd.DataFrame(df)
        
        # Re-initialize with new df to ensure proper grouping
        self.df = df
        
        df = self.create_moving_averages(ma_windows)
        self.df = df
        
        df = self.create_intensity_score()
        self.df = df
        
        df = self.create_operation_volatility(volatility_window)
        self.df = df
        
        df = self.create_day_of_week_features()
        self.df = df
        
        df = self.create_month_of_year_features()
        self.df = df
        
        # Count new features
        original_cols = [self.neighborhood_col, self.date_col] + self.seizure_cols
        new_cols = [c for c in df.columns if c not in original_cols]
        
        logger.info(f"Feature engineering complete! Created {len(new_cols)} new features")
        logger.info(f"Final DataFrame shape: {df.shape}")
        
        return df
    
    def get_feature_stats(self, df: pd.DataFrame) -> Dict:
        """
        Print statistics of created features.
        
        Args:
            df: DataFrame with features
        
        Returns:
            Dictionary with feature statistics
        """
        original_cols = [self.neighborhood_col, self.date_col] + self.seizure_cols
        feature_cols = [c for c in df.columns if c not in original_cols]
        
        stats = {
            'total_features': len(feature_cols),
            'feature_groups': {
                'lag_features': len([c for c in feature_cols if '_lag' in c]),
                'moving_averages': len([c for c in feature_cols if '_ma' in c]),
                'volatility': len([c for c in feature_cols if '_volatility' in c]),
                'temporal_cyclical': len([c for c in feature_cols if '_sin' in c or '_cos' in c]),
                'intensity': len([c for c in feature_cols if 'intensity' in c]),
            },
            'feature_names': feature_cols,
            'null_counts': df[feature_cols].isnull().sum().to_dict(),
            'value_ranges': {
                col: (df[col].min(), df[col].max())
                for col in feature_cols
                if df[col].dtype in ['float64', 'int64']
            }
        }
        
        return stats


class FactionDistributionFeatures:
    """Handle faction-related features."""
    
    def __init__(self, df: pd.DataFrame, neighborhood_col: str = 'bairro_id',
                 date_col: str = 'Data', faction_col: str = 'area_faccao'):
        """
        Initialize faction feature generator.
        
        Args:
            df: DataFrame with seizure data
            neighborhood_col: Column for neighborhoods
            date_col: Column for dates
            faction_col: Column containing faction information
        """
        self.df = df
        self.neighborhood_col = neighborhood_col
        self.date_col = date_col
        self.faction_col = faction_col
        
        if faction_col not in df.columns:
            raise ValueError(f"Column '{faction_col}' not found in DataFrame")
        
        self.unique_factions = df[faction_col].dropna().unique()
        logger.info(f"Found {len(self.unique_factions)} unique factions")
    
    def create_faction_one_hot(self) -> pd.DataFrame:
        """
        Create one-hot encoding for factions.
        
        Returns:
            DataFrame with faction one-hot columns
        """
        df = self.df.copy()
        
        # One-hot encode factions
        faction_dummies = pd.get_dummies(
            df[self.faction_col],
            prefix='faction',
            prefix_sep='_',
            dummy_na=False
        )
        
        # Rename columns to be more readable
        faction_dummies.columns = [f'is_{col}' for col in faction_dummies.columns]
        
        df = pd.concat([df, faction_dummies], axis=1)
        
        logger.info(f"Created {len(faction_dummies.columns)} one-hot encoded faction columns")
        return df
    
    def create_faction_concentration(self) -> pd.DataFrame:
        """
        Create faction concentration features (Herfindahl index).
        
        Measures how concentrated seizures are within a faction in each neighborhood-date.
        
        Returns:
            DataFrame with faction_concentration column
        """
        df = self.df.copy()
        
        # Group by neighborhood-date and calculate faction concentration
        def calc_concentration(group):
            # Count operations by faction
            faction_counts = group[self.faction_col].value_counts(normalize=True)
            # Herfindahl index: sum of squared market shares
            concentration = (faction_counts ** 2).sum()
            return concentration
        
        df['faction_concentration'] = df.groupby(
            [self.neighborhood_col, self.date_col]
        ).apply(calc_concentration).reset_index(drop=True)
        
        logger.info("Created faction_concentration feature")
        return df
    
    def create_faction_dominance(self) -> pd.DataFrame:
        """
        Create faction dominance feature (share of dominant faction).
        
        Returns:
            DataFrame with faction_dominance column
        """
        df = self.df.copy()
        
        def calc_dominance(group):
            if len(group) == 0:
                return 0
            # Get the faction with highest count
            return group[self.faction_col].value_counts().iloc[0] / len(group)
        
        df['faction_dominance'] = df.groupby(
            [self.neighborhood_col, self.date_col]
        ).apply(calc_dominance).reset_index(drop=True)
        
        logger.info("Created faction_dominance feature")
        return df
    
    def create_all_faction_features(self) -> pd.DataFrame:
        """Create all faction-related features."""
        logger.info("Starting faction feature engineering...")
        
        df = self.create_faction_one_hot()
        self.df = df
        
        df = self.create_faction_concentration()
        self.df = df
        
        df = self.create_faction_dominance()
        self.df = df
        
        logger.info("Faction feature engineering complete!")
        return df


def aggregate_to_neighborhood_date(df: pd.DataFrame,
                                   neighborhood_col: str = 'bairro_id',
                                   date_col: str = 'Data',
                                   feature_cols: List[str] = None) -> pd.DataFrame:
    """
    Aggregate features to (neighborhood, date) level.
    
    Some features (like counts, sums) need aggregation, while
    already-aggregated features (like normalized seizures) should be averaged.
    
    Args:
        df: DataFrame with individual records
        neighborhood_col: Neighborhood column name
        date_col: Date column name
        feature_cols: Columns to aggregate
    
    Returns:
        DataFrame at (neighborhood, date) level
    """
    if feature_cols is None:
        feature_cols = [c for c in df.columns 
                       if c not in [neighborhood_col, date_col]]
    
    # Aggregate: numeric columns are averaged
    agg_dict = {col: 'mean' for col in feature_cols if col in df.columns}
    
    grouped = df.groupby([date_col, neighborhood_col]).agg(agg_dict).reset_index()
    
    logger.info(f"Aggregated to {len(grouped)} (neighborhood, date) pairs")
    return grouped


# ============================================================================
# Utility functions
# ============================================================================

def load_normalized_data(path: str) -> pd.DataFrame:
    """Load normalized deduplicated parquet file."""
    df = pd.read_parquet(path)
    logger.info(f"Loaded {len(df)} records from {path}")
    return df


def save_featured_data(df: pd.DataFrame, path: str) -> None:
    """Save featured DataFrame to parquet."""
    df.to_parquet(path, index=False)
    logger.info(f"Saved {len(df)} records to {path}")
