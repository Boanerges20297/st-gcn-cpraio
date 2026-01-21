"""
Phase 3A: Build Spatial Graph Structure

Constructs:
1. Neighborhood coordinates matrix
2. Spatial edge indices based on proximity
3. Edge weights (inverse distance)
4. Adjacency matrices for ST-GCN

Output: data/processed/graph_structure.json
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

from graph.spatial_adjacency import (
    NeighborhoodCoordinates,
    SpatialAdjacencyBuilder,
    GraphConstructor
)
from features.node_matrix import NodeFeatureMatrix

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
GRAPH_OUTPUT = DATA_PATH / 'graph_structure.json'
EDGE_INDEX_NPY = DATA_PATH / 'edge_index.npy'
ADJACENCY_NPY = DATA_PATH / 'adjacency_matrix.npy'
COORDINATES_NPY = DATA_PATH / 'neighborhood_coordinates.npy'


def main():
    """Main graph construction pipeline."""
    
    logger.info("=" * 80)
    logger.info("PHASE 3A: SPATIAL GRAPH CONSTRUCTION")
    logger.info("=" * 80)
    
    # ========================================================================
    # Step 1: Load featured data to get neighborhood mapping
    # ========================================================================
    logger.info("\n[1/4] Loading featured data...")
    
    if not FEATURED_DATA.exists():
        logger.error(f"File not found: {FEATURED_DATA}")
        logger.error("Run Phase 2 first: scripts/04_temporal_features.py")
        sys.exit(1)
    
    df = pd.read_parquet(str(FEATURED_DATA))
    logger.info(f"  ✓ Loaded {len(df)} records")
    
    # Get unique neighborhoods
    neighborhoods = sorted(df['bairro_id'].unique())
    num_nodes = len(neighborhoods)
    logger.info(f"  ✓ Found {num_nodes} unique neighborhoods")
    
    # Create mappings
    node_to_id = {i: nbr_id for i, nbr_id in enumerate(neighborhoods)}
    id_to_node = {nbr_id: i for i, nbr_id in enumerate(neighborhoods)}
    
    # ========================================================================
    # Step 2: Load neighborhood coordinates
    # ========================================================================
    logger.info("\n[2/4] Loading neighborhood coordinates...")
    
    coord_loader = NeighborhoodCoordinates()
    
    # Get neighborhood names from data
    # Need to invert from our standardized mapping
    neighborhood_names = []
    
    # For now, use generic names (bairro_id as index)
    # In a real scenario, would load from geojson
    for i in range(num_nodes):
        # We'll use the neighborhood ID as index
        neighborhood_names.append(f"neighborhood_{i}")
    
    # Load coordinates for all nodes
    coordinates = coord_loader.get_coordinate_matrix(id_to_node, neighborhood_names)
    
    logger.info(f"  ✓ Loaded coordinates for {len(coordinates)} nodes")
    logger.info(f"    Shape: {coordinates.shape}")
    
    # ========================================================================
    # Step 3: Build spatial adjacency
    # ========================================================================
    logger.info("\n[3/4] Building spatial adjacency...")
    
    adjacency_builder = SpatialAdjacencyBuilder(
        coordinates,
        distance_threshold_km=1.5  # 1.5 km for Fortaleza neighborhoods
    )
    
    # Build edge index using distance method
    edge_index = adjacency_builder.build_edge_index(method='distance')
    logger.info(f"  ✓ Built edge_index: shape {edge_index.shape}")
    
    # Build edge weights
    edge_weights = adjacency_builder.build_edge_weights(edge_index)
    logger.info(f"  ✓ Built edge_weights: {len(edge_weights)} weights")
    
    # Build adjacency matrix
    adjacency_matrix = adjacency_builder.build_adjacency_matrix(
        edge_index, edge_weights
    )
    logger.info(f"  ✓ Built adjacency_matrix: shape {adjacency_matrix.shape}")
    
    # ========================================================================
    # Step 4: Save graph structure
    # ========================================================================
    logger.info("\n[4/4] Saving graph structure...")
    
    # Save numpy arrays
    np.save(str(EDGE_INDEX_NPY), edge_index)
    np.save(str(ADJACENCY_NPY), adjacency_matrix)
    np.save(str(COORDINATES_NPY), coordinates)
    
    logger.info(f"  ✓ Saved edge_index to {EDGE_INDEX_NPY.name}")
    logger.info(f"  ✓ Saved adjacency_matrix to {ADJACENCY_NPY.name}")
    logger.info(f"  ✓ Saved coordinates to {COORDINATES_NPY.name}")
    
    # Save metadata as JSON
    graph_metadata = {
        'timestamp': datetime.now().isoformat(),
        'num_nodes': int(num_nodes),
        'num_edges': int(edge_index.shape[1]),
        'edge_index_shape': list(edge_index.shape),
        'adjacency_shape': list(adjacency_matrix.shape),
        'coordinates_shape': list(coordinates.shape),
        'distance_threshold_km': 1.5,
        'method': 'distance-based spatial proximity',
        'neighborhoods': {str(i): int(nbr_id) for i, nbr_id in enumerate(neighborhoods)},
        'statistics': {
            'avg_degree': float(adjacency_matrix.sum(axis=1).mean()),
            'max_degree': float(adjacency_matrix.sum(axis=1).max()),
            'min_degree': float(adjacency_matrix.sum(axis=1).min()),
            'density': float(np.count_nonzero(adjacency_matrix) / (num_nodes**2)),
            'edge_weight_stats': {
                'min': float(edge_weights.min()),
                'max': float(edge_weights.max()),
                'mean': float(edge_weights.mean()),
                'std': float(edge_weights.std())
            }
        }
    }
    
    with open(GRAPH_OUTPUT, 'w') as f:
        json.dump(graph_metadata, f, indent=2, default=str)
    
    logger.info(f"  ✓ Saved metadata to {GRAPH_OUTPUT.name}")
    
    # ========================================================================
    # Summary
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("GRAPH CONSTRUCTION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Nodes (neighborhoods):    {num_nodes}")
    logger.info(f"Edges (spatial links):    {edge_index.shape[1]}")
    logger.info(f"Graph density:            {graph_metadata['statistics']['density']:.4f}")
    logger.info(f"Average degree:           {graph_metadata['statistics']['avg_degree']:.2f}")
    logger.info(f"Distance threshold:       1.5 km")
    logger.info(f"\nFiles saved:")
    logger.info(f"  • edge_index.npy")
    logger.info(f"  • adjacency_matrix.npy")
    logger.info(f"  • neighborhood_coordinates.npy")
    logger.info(f"  • graph_structure.json")
    logger.info("=" * 80)
    logger.info("✅ Phase 3A Complete!")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
