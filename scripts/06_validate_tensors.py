"""
Phase 3B: Tensor Validation for ST-GCN

Validates all input tensors:
1. Node feature tensor (T, N, F)
2. Edge index compatibility
3. Value ranges and data types
4. No NaN/Inf values
5. Tensor shapes match graph structure
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

from features.node_matrix import NodeFeatureMatrix, select_features_for_model

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / 'data' / 'processed'
FEATURED_DATA = DATA_PATH / 'prisoes_with_features.parquet'
EDGE_INDEX = DATA_PATH / 'edge_index.npy'
ADJACENCY = DATA_PATH / 'adjacency_matrix.npy'
GRAPH_METADATA = DATA_PATH / 'graph_structure.json'
VALIDATION_REPORT = DATA_PATH / 'tensor_validation_report.json'
NODE_TENSOR_NPY = DATA_PATH / 'node_feature_tensor.npy'


def main():
    """Main tensor validation pipeline."""
    
    logger.info("=" * 80)
    logger.info("PHASE 3B: TENSOR VALIDATION FOR ST-GCN")
    logger.info("=" * 80)
    
    # ========================================================================
    # Step 1: Load all components
    # ========================================================================
    logger.info("\n[1/5] Loading components...")
    
    if not all([FEATURED_DATA.exists(), EDGE_INDEX.exists(), GRAPH_METADATA.exists()]):
        logger.error("Missing required files from Phase 3A")
        sys.exit(1)
    
    # Load featured data
    df = pd.read_parquet(str(FEATURED_DATA))
    logger.info(f"  ✓ Loaded featured data: {df.shape}")
    
    # Load graph structure
    with open(GRAPH_METADATA, 'r') as f:
        graph_meta = json.load(f)
    num_nodes = graph_meta['num_nodes']
    num_edges = graph_meta['num_edges']
    logger.info(f"  ✓ Loaded graph: {num_nodes} nodes, {num_edges} edges")
    
    # Load edge index and adjacency
    edge_index = np.load(str(EDGE_INDEX))
    adjacency = np.load(str(ADJACENCY))
    logger.info(f"  ✓ Loaded edge_index: {edge_index.shape}")
    logger.info(f"  ✓ Loaded adjacency: {adjacency.shape}")
    
    # ========================================================================
    # Step 2: Build node feature tensor
    # ========================================================================
    logger.info("\n[2/5] Building node feature tensor...")
    
    node_matrix = NodeFeatureMatrix(df)
    
    # Select features for ST-GCN
    feature_cols = select_features_for_model(
        df,
        include_types=['normalized', 'lag', 'moving_avg', 'intensity', 'temporal']
    )
    logger.info(f"  ✓ Selected {len(feature_cols)} features for model")
    
    # Build tensor
    X, feature_names = node_matrix.build_feature_tensor(feature_cols)
    T, N, F = X.shape
    logger.info(f"  ✓ Built feature tensor: shape ({T}, {N}, {F})")
    
    # Save tensor
    np.save(str(NODE_TENSOR_NPY), X)
    logger.info(f"  ✓ Saved node tensor: {NODE_TENSOR_NPY.name}")
    
    # ========================================================================
    # Step 3: Validate dimensions
    # ========================================================================
    logger.info("\n[3/5] Validating dimensions...")
    
    issues = []
    
    # Check tensor shape
    if N != num_nodes:
        issues.append(f"Node mismatch: tensor has {N} nodes, graph has {num_nodes}")
    else:
        logger.info(f"  ✓ Nodes match: {N} == {num_nodes}")
    
    # Check edge index
    if edge_index.shape[0] != 2:
        issues.append(f"Edge index shape invalid: expected (2, E), got {edge_index.shape}")
    else:
        logger.info(f"  ✓ Edge index shape valid: {edge_index.shape}")
    
    # Check adjacency
    if adjacency.shape != (num_nodes, num_nodes):
        issues.append(f"Adjacency shape invalid: expected ({num_nodes}, {num_nodes}), got {adjacency.shape}")
    else:
        logger.info(f"  ✓ Adjacency shape valid: {adjacency.shape}")
    
    # ========================================================================
    # Step 4: Validate values and data types
    # ========================================================================
    logger.info("\n[4/5] Validating values and data types...")
    
    # Check for NaN/Inf
    nan_count = int(np.isnan(X).sum())
    inf_count = int(np.isinf(X).sum())
    
    if nan_count > 0:
        issues.append(f"Found {nan_count} NaN values in node tensor")
    else:
        logger.info(f"  ✓ No NaN values in node tensor")
    
    if inf_count > 0:
        issues.append(f"Found {inf_count} infinite values in node tensor")
    else:
        logger.info(f"  ✓ No infinite values in node tensor")
    
    # Check value ranges (most normalized features should be [0,1])
    min_val = X.min()
    max_val = X.max()
    logger.info(f"  ✓ Node tensor value range: [{min_val:.4f}, {max_val:.4f}]")
    
    if max_val > 10:
        logger.warning(f"    ⚠ Some values exceed expected range [0,1]: max={max_val:.4f}")
    
    # Check edge weights
    if edge_index.shape[1] > 0:
        min_weight = adjacency[adjacency > 0].min()
        max_weight = adjacency.max()
        logger.info(f"  ✓ Edge weights: [{min_weight:.4f}, {max_weight:.4f}]")
    
    # Check data types
    logger.info(f"  ✓ Node tensor dtype: {X.dtype}")
    logger.info(f"  ✓ Edge index dtype: {edge_index.dtype}")
    logger.info(f"  ✓ Adjacency dtype: {adjacency.dtype}")
    
    # ========================================================================
    # Step 5: Tensor compatibility check
    # ========================================================================
    logger.info("\n[5/5] Checking ST-GCN compatibility...")
    
    # Check tensor can be sliced into windows
    min_window_size = 7
    if T < min_window_size:
        issues.append(f"Insufficient time steps: {T} < {min_window_size}")
    else:
        logger.info(f"  ✓ Sufficient time steps for {T - min_window_size} windows of size {min_window_size}")
    
    # Check all nodes have data
    zero_nodes = (X.sum(axis=(0, 2)) == 0).sum()
    if zero_nodes > 0:
        logger.warning(f"    ⚠ {zero_nodes} nodes have all-zero features")
    else:
        logger.info(f"  ✓ All nodes have non-zero features")
    
    # ========================================================================
    # Generate validation report
    # ========================================================================
    logger.info("\n[Final] Generating validation report...")
    
    # Get tensor statistics
    tensor_stats = node_matrix.get_tensor_statistics(X)
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'validation_status': 'PASSED' if len(issues) == 0 else 'FAILED',
        'issues': issues,
        'graph': {
            'num_nodes': num_nodes,
            'num_edges': num_edges,
            'graph_density': float(graph_meta['statistics']['density']),
            'avg_degree': float(graph_meta['statistics']['avg_degree'])
        },
        'node_tensor': {
            'shape': [int(T), int(N), int(F)],
            'dtype': str(X.dtype),
            'size_mb': float(X.nbytes / 1e6),
            'features': feature_names,
            'statistics': {
                'min': float(min_val),
                'max': float(max_val),
                'mean': float(X.mean()),
                'std': float(X.std()),
                'nan_count': int(nan_count),
                'inf_count': int(inf_count)
            }
        },
        'edge_index': {
            'shape': list(edge_index.shape),
            'dtype': str(edge_index.dtype),
            'min_node_id': int(edge_index.min()),
            'max_node_id': int(edge_index.max())
        },
        'adjacency': {
            'shape': list(adjacency.shape),
            'dtype': str(adjacency.dtype),
            'nonzero_count': int(np.count_nonzero(adjacency)),
            'density': float(np.count_nonzero(adjacency) / (num_nodes ** 2))
        },
        'compatibility': {
            'compatible_with_stgcn': len(issues) == 0,
            'ready_for_training': len(issues) == 0 and zero_nodes == 0,
            'min_time_steps': min_window_size,
            'available_windows': int(T - min_window_size)
        }
    }
    
    # Save report
    with open(VALIDATION_REPORT, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    logger.info(f"  ✓ Saved validation report: {VALIDATION_REPORT.name}")
    
    # ========================================================================
    # Summary
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("TENSOR VALIDATION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Status:                   {report['validation_status']}")
    logger.info(f"Issues:                   {len(issues)}")
    logger.info(f"\nTensor Shapes:")
    logger.info(f"  • Node features:        ({T}, {N}, {F}) = {X.size:,} values")
    logger.info(f"  • Edge index:           {edge_index.shape}")
    logger.info(f"  • Adjacency matrix:     {adjacency.shape}")
    logger.info(f"\nST-GCN Compatibility:")
    logger.info(f"  • Compatible:           {report['compatibility']['compatible_with_stgcn']}")
    logger.info(f"  • Ready for training:   {report['compatibility']['ready_for_training']}")
    logger.info(f"  • Temporal windows:     {report['compatibility']['available_windows']}")
    logger.info(f"\nFeatures ({len(feature_names)}):")
    for i, feat in enumerate(feature_names[:5]):
        logger.info(f"  {i+1}. {feat}")
    if len(feature_names) > 5:
        logger.info(f"  ... and {len(feature_names) - 5} more")
    
    if issues:
        logger.info(f"\n⚠ Validation Issues:")
        for issue in issues:
            logger.warning(f"  - {issue}")
    else:
        logger.info(f"\n✅ ALL CHECKS PASSED!")
    
    logger.info("=" * 80)
    logger.info("✅ Phase 3B Complete!")
    logger.info("=" * 80)
    
    return len(issues) == 0


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
