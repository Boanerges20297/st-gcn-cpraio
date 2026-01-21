"""
Node Feature Matrix Construction for ST-GCN

Converts time-series data into tensor format:
  X shape: (T, N, F) where
    T = time steps (375 days)
    N = nodes/neighborhoods (138)
    F = features per node (variable)
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class NodeFeatureMatrix:
    """Build node feature matrices for spatio-temporal graph neural networks."""
    
    def __init__(self, df: pd.DataFrame,
                 date_col: str = 'Data',
                 node_col: str = 'bairro_id',
                 sequence_col: str = 'operacoes_diarias'):
        """
        Initialize feature matrix builder.
        
        Args:
            df: DataFrame with (date, node_id, features)
            date_col: Column name for time steps
            node_col: Column name for node identifiers
            sequence_col: Column to use for ordering
        """
        self.df = df.sort_values([date_col, node_col]).reset_index(drop=True)
        self.date_col = date_col
        self.node_col = node_col
        self.sequence_col = sequence_col
        
        # Extract unique times and nodes
        self.times = sorted(df[date_col].unique())
        self.nodes = sorted(df[node_col].unique())
        self.T = len(self.times)
        self.N = len(self.nodes)
        
        # Create mapping dictionaries
        self.time_to_idx = {t: i for i, t in enumerate(self.times)}
        self.node_to_idx = {n: i for i, n in enumerate(self.nodes)}
        self.idx_to_node = {i: n for n, i in self.node_to_idx.items()}
        
        logger.info(f"Initialized NodeFeatureMatrix: T={self.T}, N={self.N}")
    
    def build_feature_tensor(self,
                            feature_cols: Optional[List[str]] = None) -> np.ndarray:
        """
        Build feature tensor X of shape (T, N, F).
        
        Args:
            feature_cols: List of feature columns to include.
                         If None, uses all numeric columns except date/node_id.
        
        Returns:
            Feature tensor of shape (T, N, F)
        """
        if feature_cols is None:
            # Use all numeric columns except identifiers
            exclude_cols = {self.date_col, self.node_col}
            feature_cols = [c for c in self.df.columns 
                          if c not in exclude_cols 
                          and self.df[c].dtype in ['float64', 'int64']]
        
        F = len(feature_cols)
        logger.info(f"Building feature tensor with {F} features: {feature_cols}")
        
        # Initialize tensor with zeros (for missing data)
        X = np.zeros((self.T, self.N, F), dtype=np.float32)
        
        # Fill tensor
        for _, row in self.df.iterrows():
            t_idx = self.time_to_idx[row[self.date_col]]
            n_idx = self.node_to_idx[row[self.node_col]]
            X[t_idx, n_idx, :] = row[feature_cols].values.astype(np.float32)
        
        logger.info(f"✓ Feature tensor shape: {X.shape}")
        logger.info(f"  - Time steps: {self.T}")
        logger.info(f"  - Nodes: {self.N}")
        logger.info(f"  - Features: {F}")
        logger.info(f"  - Total elements: {X.size:,}")
        logger.info(f"  - Memory: {X.nbytes / 1e6:.1f} MB")
        
        return X, feature_cols
    
    def build_temporal_slices(self, X: np.ndarray,
                             window_size: int = 7) -> List[np.ndarray]:
        """
        Build temporal windows for sequential models.
        
        Args:
            X: Feature tensor (T, N, F)
            window_size: Size of temporal window
        
        Returns:
            List of tensors of shape (window_size, N, F)
        """
        T, N, F = X.shape
        slices = []
        
        for t in range(T - window_size + 1):
            slices.append(X[t:t+window_size])
        
        logger.info(f"✓ Created {len(slices)} temporal windows (size={window_size})")
        logger.info(f"  - Window tensor shape: {slices[0].shape}")
        
        return slices
    
    def get_tensor_statistics(self, X: np.ndarray) -> Dict:
        """Get statistics of feature tensor."""
        stats = {
            'shape': X.shape,
            'dtype': str(X.dtype),
            'memory_mb': X.nbytes / 1e6,
            'null_count': int(np.isnan(X).sum()),
            'zero_count': int((X == 0).sum()),
            'min': float(np.nanmin(X)),
            'max': float(np.nanmax(X)),
            'mean': float(np.nanmean(X)),
            'std': float(np.nanstd(X)),
            'percentiles': {
                '25': float(np.nanpercentile(X, 25)),
                '50': float(np.nanpercentile(X, 50)),
                '75': float(np.nanpercentile(X, 75)),
                '95': float(np.nanpercentile(X, 95)),
                '99': float(np.nanpercentile(X, 99))
            }
        }
        return stats
    
    def validate_tensor(self, X: np.ndarray, feature_cols: List[str]) -> Dict:
        """
        Validate tensor for ST-GCN compatibility.
        
        Args:
            X: Feature tensor (T, N, F)
            feature_cols: Feature column names
        
        Returns:
            Validation report
        """
        T, N, F = X.shape
        issues = []
        
        # Check for NaN
        if np.isnan(X).any():
            count = int(np.isnan(X).sum())
            issues.append(f"Contains {count} NaN values")
        
        # Check for infinite
        if np.isinf(X).any():
            count = int(np.isinf(X).sum())
            issues.append(f"Contains {count} infinite values")
        
        # Check dimensions
        if T != self.T or N != self.N or F != len(feature_cols):
            issues.append(f"Dimension mismatch: expected ({self.T}, {self.N}, {len(feature_cols)}), got {X.shape}")
        
        # Check value ranges (should be mostly [0,1] for normalized features)
        if X.max() > 1.5:
            issues.append(f"Some values exceed expected range [0,1.5]: max={X.max():.4f}")
        
        report = {
            'valid': len(issues) == 0,
            'shape': X.shape,
            'features': len(feature_cols),
            'feature_names': feature_cols,
            'issues': issues,
            'statistics': self.get_tensor_statistics(X),
            'time_coverage': f"{self.times[0]} to {self.times[-1]}",
            'nodes': self.N
        }
        
        return report
    
    def save_tensor(self, X: np.ndarray, path: str) -> None:
        """Save tensor to numpy file."""
        np.save(path, X)
        logger.info(f"✓ Saved tensor to {path}")
    
    def load_tensor(self, path: str) -> np.ndarray:
        """Load tensor from numpy file."""
        X = np.load(path)
        logger.info(f"✓ Loaded tensor from {path}, shape: {X.shape}")
        return X
    
    def get_node_mapping(self) -> Dict[int, int]:
        """Get mapping from neighborhood ID to node index."""
        return self.node_to_idx.copy()
    
    def get_time_mapping(self) -> Dict:
        """Get mapping from date to time index."""
        return self.time_to_idx.copy()
    
    def get_reverse_mappings(self) -> Tuple[Dict, Dict]:
        """Get reverse mappings (index -> ID)."""
        idx_to_node = {i: n for n, i in self.node_to_idx.items()}
        idx_to_time = {i: t for t, i in self.time_to_idx.items()}
        return idx_to_node, idx_to_time


class TensorMetadata:
    """Save and load tensor metadata for reproducibility."""
    
    def __init__(self, X: np.ndarray, feature_cols: List[str],
                 node_mapping: Dict, time_mapping: Dict,
                 validation_report: Dict):
        """Initialize metadata."""
        self.X = X
        self.feature_cols = feature_cols
        self.node_mapping = node_mapping
        self.time_mapping = time_mapping
        self.validation_report = validation_report
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        # Convert node/time mappings to strings for JSON serialization
        node_map_str = {str(k): int(v) for k, v in self.node_mapping.items()}
        time_map_str = {str(k): int(v) for k, v in self.time_mapping.items()}
        
        return {
            'shape': list(self.X.shape),
            'feature_columns': self.feature_cols,
            'node_mapping': node_map_str,
            'num_nodes': len(self.node_mapping),
            'num_features': len(self.feature_cols),
            'num_timesteps': self.X.shape[0],
            'validation': self.validation_report
        }
    
    def save(self, path: str) -> None:
        """Save metadata to JSON."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        logger.info(f"✓ Saved metadata to {path}")
    
    @staticmethod
    def load(path: str) -> Dict:
        """Load metadata from JSON."""
        with open(path, 'r') as f:
            data = json.load(f)
        logger.info(f"✓ Loaded metadata from {path}")
        return data


# ============================================================================
# Utility functions
# ============================================================================

def select_features_for_model(df: pd.DataFrame,
                             include_types: List[str] = None) -> List[str]:
    """
    Select features by type for the model.
    
    Args:
        df: DataFrame with features
        include_types: Types to include:
                      'raw' (original seizures)
                      'normalized' (_norm columns)
                      'lag' (lagged features)
                      'moving_avg' (moving averages)
                      'volatility'
                      'intensity'
                      'temporal' (cyclical features)
    
    Returns:
        List of feature column names
    """
    if include_types is None:
        include_types = ['normalized', 'lag', 'moving_avg', 'intensity', 'temporal']
    
    type_patterns = {
        'raw': ['drogas_gramas', 'armas_total', 'dinheiro_total_reais'] if not any(x in c for c in df.columns for x in ['_norm', '_lag', '_ma']) else [],
        'normalized': '_norm',
        'lag': '_lag',
        'moving_avg': '_ma',
        'volatility': '_volatility',
        'intensity': 'intensity_score',
        'temporal': ['_sin', '_cos']
    }
    
    selected = []
    for col in df.columns:
        if col in ['Data', 'bairro_id', 'operacoes_diarias']:
            continue
        
        for feat_type, pattern in type_patterns.items():
            if feat_type not in include_types:
                continue
            
            if isinstance(pattern, list):
                if any(p in col for p in pattern):
                    selected.append(col)
                    break
            else:
                if pattern in col:
                    selected.append(col)
                    break
    
    return list(set(selected))  # Remove duplicates
