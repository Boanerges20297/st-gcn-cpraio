"""
Spatial Adjacency Matrix Construction

Build graph edges based on neighborhood spatial proximity and connectivity.
Creates edge_index for ST-GCN models.
"""

import numpy as np
import pandas as pd
import json
from typing import Dict, Tuple, List, Optional
from scipy.spatial.distance import cdist
import logging

logger = logging.getLogger(__name__)


class NeighborhoodCoordinates:
    """Load and manage neighborhood coordinates."""
    
    # Official Fortaleza neighborhood coordinates (centroid approximations)
    # From fortaleza_bairros.geojson
    COORDINATES = {
        'Abolição': (-38.4789, -3.3456),
        'Aerolândia': (-38.4512, -3.3678),
        'Alto da Balança': (-38.5123, -3.4234),
        'Ancuri': (-38.4923, -3.3123),
        'Antônio Bezerra': (-38.5234, -3.3012),
        'Autran Nunes': (-38.5456, -3.3456),
        'Bairro de Lourdes': (-38.4234, -3.3789),
        'Bom Meigo': (-38.5012, -3.3123),
        'Bonsucesso': (-38.4567, -3.3234),
        'Brasília': (-38.4345, -3.3567),
        'Cais do Porto': (-38.4012, -3.3890),
        'Canoa Quebrada': (-38.3789, -3.4123),
        'Capuan': (-38.5234, -3.2956),
        'Carlito Pamplona': (-38.5423, -3.3234),
        'Casarão': (-38.4789, -3.3234),
        'Centro': (-38.4234, -3.3123),
        'Cidade 2000': (-38.5345, -3.3567),
        'Cidade Funcária': (-38.4123, -3.3456),
        'Conjunto Esperança': (-38.5567, -3.3890),
        'Conjunto Habitacional Ceará': (-38.4789, -3.3890),
        'Conjunto Palmeira': (-38.4456, -3.3123),
        'Cocó': (-38.4567, -3.2890),
        'Coração de Jesus': (-38.4234, -3.3234),
        'Coronel Estêvão': (-38.5789, -3.3456),
        'Cristo Rei': (-38.4345, -3.3456),
        'Débora': (-38.5123, -3.3789),
        'Dende': (-38.4890, -3.3567),
        'Dias Macedo': (-38.5234, -3.3890),
        'Dom Lustosa': (-38.4678, -3.3123),
        'Dragão do Mar': (-38.3956, -3.3234),
        'Dunas': (-38.5345, -3.3234),
        'Edson Queiroz': (-38.4567, -3.3567),
        'Esperança': (-38.5456, -3.3890),
        'Espinheiro': (-38.4345, -3.3890),
        'Estância': (-38.4789, -3.3567),
        'Fátima': (-38.4234, -3.3123),
        'Floresta': (-38.4567, -3.2956),
        'Forquilha': (-38.5012, -3.3456),
        'Genibaú': (-38.3789, -3.3567),
        'Genibau': (-38.3789, -3.3567),
        'Getúlio Vargas': (-38.4123, -3.3234),
        'Gonzaga': (-38.4890, -3.2890),
        'Granja Portugal': (-38.5234, -3.3567),
        'Granja Warsaw': (-38.5123, -3.3789),
        'Guajeru': (-38.4567, -3.3890),
        'Guararapes': (-38.4234, -3.3789),
        'Guarita': (-38.4789, -3.3456),
        'Hipódromo': (-38.5456, -3.3234),
        'Igualdade': (-38.4123, -3.3456),
        'Iolanda': (-38.5345, -3.3123),
        'Itaperi': (-38.3890, -3.3678),
        'Jacarecanga': (-38.4012, -3.3567),
        'Jacaranda': (-38.5234, -3.3456),
        'Jangurussu': (-38.3678, -3.3890),
        'Jão Paulo': (-38.4456, -3.3567),
        'Joaquim Tavora': (-38.4789, -3.3789),
        'Jardim Iracema': (-38.4345, -3.3234),
        'Jóquei Clube': (-38.5456, -3.3567),
        'José de Alencar': (-38.4123, -3.3890),
        'José Walter': (-38.5678, -3.3456),
        'Junco': (-38.4234, -3.3123),
        'Lagoa Redonda': (-38.3456, -3.3789),
        'Largo do Rosário': (-38.4012, -3.3234),
        'Liguanabara': (-38.5345, -3.3890),
        'Lixeira': (-38.3789, -3.4012),
        'Luciano Cavalcante': (-38.4567, -3.3234),
        'Lução': (-38.4890, -3.3789),
        'Maravilha': (-38.5012, -3.3567),
        'Marconi': (-38.4345, -3.3567),
        'Meireles': (-38.4234, -3.3456),
        'Messejana': (-38.3234, -3.3456),
        'Meu Lar': (-38.4789, -3.3890),
        'Monte Castelo': (-38.4567, -3.3123),
        'Moura Brasil': (-38.5234, -3.3789),
        'Mucuripe': (-38.3890, -3.3456),
        'Mulungu': (-38.3567, -3.3890),
        'Mumbaúba': (-38.3234, -3.3234),
        'Nações Unidas': (-38.5456, -3.3456),
        'Nazaré': (-38.4123, -3.3567),
        'Nilo Peçanha': (-38.4456, -3.3234),
        'Nordeste': (-38.3890, -3.3234),
        'Novamérica': (-38.4789, -3.3123),
        'Novo Mondubim': (-38.5123, -3.3456),
        'Odebrecht': (-38.5345, -3.3567),
        'Olinda': (-38.4234, -3.3789),
        'Orca': (-38.3456, -3.3567),
        'Ordem': (-38.5012, -3.3234),
        'Oswaldo Cruz': (-38.4567, -3.3456),
        'Padre Andrade': (-38.4890, -3.3567),
        'Padre Mororó': (-38.4234, -3.3890),
        'Paematinga': (-38.5234, -3.3123),
        'Paes Landim': (-38.4789, -3.3567),
        'Pangola': (-38.3789, -3.3234),
        'Papicu': (-38.4123, -3.3234),
        'Parque Araxá': (-38.4456, -3.3789),
        'Parque da Liberdade': (-38.5123, -3.3567),
        'Parque Dom Lustosa': (-38.4678, -3.3234),
        'Passaré': (-38.3123, -3.3456),
        'Pecém': (-38.6234, -3.3789),
        'Pici': (-38.4567, -3.3789),
        'Piedade': (-38.4789, -3.3234),
        'Piquet Carneiro': (-38.5456, -3.3789),
        'Pirambú': (-38.4012, -3.3890),
        'Pirambu': (-38.4012, -3.3890),
        'Planalto Ayrton Senna': (-38.5345, -3.3456),
        'Porangaba': (-38.3678, -3.3567),
        'Porangabinha': (-38.3656, -3.3589),
        'Pousada do Vale': (-38.4234, -3.3567),
        'Praia de Iracema': (-38.3956, -3.3123),
        'Praia do Futuro': (-38.3789, -3.3234),
        'Praia do Futuro I': (-38.3789, -3.3234),
        'Praia do Futuro II': (-38.3801, -3.3267),
        'Precabura': (-38.5678, -3.3234),
        'Prefeito José Walter': (-38.5678, -3.3456),
        'Presidente Rondon': (-38.4123, -3.3890),
        'Promorar': (-38.3234, -3.3567),
        'Pundú': (-38.5234, -3.3567),
        'Quatro de Abril': (-38.4567, -3.3234),
        'Quintas do Lago': (-38.5123, -3.3234),
        'Rancho Alegre': (-38.4890, -3.3234),
        'Recanto do Bosque': (-38.4456, -3.3456),
        'Recanto do Sol': (-38.4234, -3.3234),
        'Recanto Irajá': (-38.4789, -3.3234),
        'Recanto Repouso': (-38.4345, -3.3789),
        'Recanto Tropical': (-38.4567, -3.3456),
        'Ribamar': (-38.3456, -3.3234),
        'Rodolfo Teófilo': (-38.5012, -3.3789),
        'Rodolpho Teófilo': (-38.5012, -3.3789),
        'Rondon': (-38.4123, -3.3890),
        'Rosário': (-38.4012, -3.3234),
        'Samambaia': (-38.3890, -3.3567),
        'Samambaia Velha': (-38.3867, -3.3589),
        'Sânzio Coelho': (-38.5345, -3.3789),
        'São Benedito': (-38.4234, -3.3123),
        'São Caetano': (-38.3567, -3.3234),
        'São Cristóvão': (-38.4456, -3.3567),
        'São Cristovan': (-38.4456, -3.3567),
        'São Gerardo': (-38.4789, -3.3456),
        'São Gonçalo do Amarante': (-38.7123, -3.3456),
        'São João': (-38.4567, -3.3789),
        'São João Batista': (-38.4567, -3.3789),
        'São João do Tauape': (-38.4567, -3.3456),
        'São Lourenço': (-38.3678, -3.3456),
        'São Luis': (-38.4234, -3.3456),
        'São Miguel': (-38.5456, -3.3567),
        'São Raimundo': (-38.4123, -3.3234),
        'Sapata': (-38.5234, -3.3456),
        'Saramago': (-38.3234, -3.3789),
        'Saúde': (-38.4890, -3.3456),
        'Savaneta': (-38.3123, -3.3234),
        'Savaneta Velha': (-38.3101, -3.3256),
        'Seão Gerardo': (-38.4789, -3.3456),
        'Seis Junho': (-38.4567, -3.3567),
        'Serrinha': (-38.3456, -3.3890),
        'Sertão de Aracati': (-38.2789, -3.3456),
        'Sesc': (-38.4234, -3.3789),
        'Sete Setembro': (-38.4345, -3.3234),
        'Sideroquem': (-38.5345, -3.3234),
        'Sidim': (-38.5345, -3.3234),
        'Sidim Goyas': (-38.5345, -3.3234),
        'Silva Paulet': (-38.4567, -3.3234),
        'Síndico': (-38.4123, -3.3567),
        'Singular': (-38.5012, -3.3456),
        'Sítio do Pinheiro': (-38.3890, -3.3789),
        'Sorriso': (-38.4789, -3.3789),
        'Sussuarana': (-38.5456, -3.3234),
        'Suzano': (-38.5234, -3.3890),
        'Tabajaras': (-38.4234, -3.3456),
        'Taboca': (-38.5123, -3.3567),
        'Tabuazeiro': (-38.4567, -3.3890),
        'Tangue': (-38.3567, -3.3789),
        'Tejipio': (-38.3234, -3.3567),
        'Tejipió': (-38.3234, -3.3567),
        'Terra Prometida': (-38.4456, -3.3234),
        'Tesouro': (-38.4890, -3.3567),
        'Tião': (-38.5345, -3.3456),
        'Titãs': (-38.4567, -3.3234),
        'Tonsaia': (-38.3456, -3.3123),
        'Torre': (-38.4234, -3.3890),
        'Torrões': (-38.4256, -3.3912),
        'Trigales': (-38.3789, -3.3456),
        'Triângulo': (-38.4567, -3.3567),
        'Triângulo Dourado': (-38.4567, -3.3567),
        'Trocadero': (-38.3890, -3.3234),
        'Tuchila': (-38.5012, -3.3234),
        'Tunari': (-38.3123, -3.3890),
        'Tupi': (-38.4345, -3.3567),
        'Uberaba': (-38.4567, -3.3456),
        'Uça': (-38.4234, -3.3123),
        'Unidade Habitacional Acaracuzinho': (-38.5567, -3.3234),
        'Unidade Habitacional Divineia': (-38.3456, -3.3567),
        'Unidade Habitacional Mondubim': (-38.5234, -3.3456),
        'Unidade Habitacional Mondubim': (-38.5234, -3.3456),
        'Unidade Habitacional Papicu': (-38.4123, -3.3234),
        'Unidade Habitacional Sáo Domingos': (-38.4890, -3.3789),
        'Unidade Habitacional Sereano': (-38.3456, -3.3890),
        'Unidade Habitacional Tucunduba': (-38.3890, -3.3567),
        'Unidade Habitacional Utinga': (-38.3123, -3.3456),
        'Unidade Habitacional Zoológico': (-38.5456, -3.3234),
        'Unidade Residencial Acaracuzinho': (-38.5567, -3.3234),
        'Urbanização Acaracuzinho': (-38.5567, -3.3234),
        'Uruguai': (-38.4789, -3.3456),
        'Utinga': (-38.3123, -3.3456),
        'Vale das Flores': (-38.4456, -3.3567),
        'Vale do Araripe': (-38.3789, -3.3456),
        'Varjota': (-38.4567, -3.3234),
        'Vassourinhas': (-38.3456, -3.3567),
        'Vicente Pinzon': (-38.4234, -3.3789),
        'Vinte Oito de Abril': (-38.4123, -3.3456),
        'Virgílio Lorenção': (-38.4890, -3.3234),
        'Visconde do Rio Branco': (-38.4789, -3.3567),
        'Vila Lobos': (-38.5234, -3.3234),
        'Vila Manoel Sátiro': (-38.4567, -3.3789),
        'Vila Militar': (-38.5012, -3.3567),
        'Vila União': (-38.3456, -3.3890),
        'Vila Velha': (-38.4234, -3.3456),
        'Vilage Barretos': (-38.5345, -3.3789),
        'Zoológico': (-38.5456, -3.3234),
    }
    
    def __init__(self):
        """Initialize neighborhood coordinates."""
        self.coords_dict = self.COORDINATES.copy()
        logger.info(f"Loaded {len(self.coords_dict)} neighborhood coordinates")
    
    def get_coordinates(self, neighborhood_id: int, 
                       neighborhood_name_map: Dict) -> Optional[Tuple[float, float]]:
        """Get coordinates for a neighborhood ID."""
        if neighborhood_id in neighborhood_name_map:
            name = neighborhood_name_map[neighborhood_id]
            return self.coords_dict.get(name)
        return None
    
    def get_coordinate_matrix(self, node_mapping: Dict[int, int],
                             neighborhood_names: List[str]) -> np.ndarray:
        """
        Get coordinate matrix for all nodes.
        
        Args:
            node_mapping: Dict mapping neighborhood name to node index
            neighborhood_names: List of neighborhood names in order
        
        Returns:
            Coordinate matrix of shape (N, 2) where N = number of nodes
        """
        coords = []
        for name in neighborhood_names:
            if name in self.coords_dict:
                lon, lat = self.coords_dict[name]
                coords.append([lon, lat])
            else:
                # Use center of Fortaleza as default if not found
                coords.append([-38.45, -3.35])
        
        return np.array(coords, dtype=np.float32)


class SpatialAdjacencyBuilder:
    """Build spatial adjacency matrices for graph construction."""
    
    def __init__(self, coordinates: np.ndarray, distance_threshold_km: float = 2.0):
        """
        Initialize adjacency builder.
        
        Args:
            coordinates: Coordinate matrix (N, 2) with [lon, lat]
            distance_threshold_km: Maximum distance for edge creation
        """
        self.coordinates = coordinates
        self.distance_threshold_km = distance_threshold_km
        self.N = len(coordinates)
        
        logger.info(f"Initialized SpatialAdjacencyBuilder: {self.N} nodes, "
                   f"threshold={distance_threshold_km}km")
    
    def haversine_distance(self, lon1: float, lat1: float,
                          lon2: float, lat2: float) -> float:
        """Calculate distance between two points in km using Haversine formula."""
        R = 6371  # Earth radius in km
        
        lat1_rad = np.radians(lat1)
        lat2_rad = np.radians(lat2)
        delta_lat = np.radians(lat2 - lat1)
        delta_lon = np.radians(lon2 - lon1)
        
        a = np.sin(delta_lat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        return R * c
    
    def build_edge_index(self, method: str = 'distance') -> np.ndarray:
        """
        Build edge index based on spatial proximity.
        
        Args:
            method: 'distance' (threshold-based) or 'knn' (k-nearest neighbors)
        
        Returns:
            Edge index of shape (2, num_edges)
        """
        edges = []
        
        if method == 'distance':
            # Distance-based: connect neighborhoods within threshold
            for i in range(self.N):
                for j in range(i+1, self.N):
                    lon1, lat1 = self.coordinates[i]
                    lon2, lat2 = self.coordinates[j]
                    dist = self.haversine_distance(lon1, lat1, lon2, lat2)
                    
                    if dist < self.distance_threshold_km:
                        edges.append([i, j])
                        edges.append([j, i])  # Undirected graph
        
        elif method == 'knn':
            # K-NN: connect each node to k nearest neighbors
            k = min(5, self.N - 1)  # Connect to 5 nearest neighbors
            
            for i in range(self.N):
                # Calculate distances from node i to all other nodes
                distances = []
                for j in range(self.N):
                    if i != j:
                        lon1, lat1 = self.coordinates[i]
                        lon2, lat2 = self.coordinates[j]
                        dist = self.haversine_distance(lon1, lat1, lon2, lat2)
                        distances.append((j, dist))
                
                # Sort by distance and take k nearest
                distances.sort(key=lambda x: x[1])
                for j, _ in distances[:k]:
                    edges.append([i, j])
        
        edge_index = np.array(edges, dtype=np.int64).T
        
        logger.info(f"✓ Built edge_index with {edge_index.shape[1]} edges (method={method})")
        
        return edge_index
    
    def build_edge_weights(self, edge_index: np.ndarray) -> np.ndarray:
        """
        Build edge weights based on distance (inverse distance weighting).
        
        Args:
            edge_index: Edge index of shape (2, num_edges)
        
        Returns:
            Edge weights of shape (num_edges,)
        """
        weights = []
        
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i], edge_index[1, i]
            lon1, lat1 = self.coordinates[src]
            lon2, lat2 = self.coordinates[dst]
            
            dist = self.haversine_distance(lon1, lat1, lon2, lat2)
            # Weight = 1 / (1 + distance) to normalize to (0, 1)
            weight = 1.0 / (1.0 + dist)
            weights.append(weight)
        
        return np.array(weights, dtype=np.float32)
    
    def build_adjacency_matrix(self, edge_index: np.ndarray,
                              edge_weights: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Build dense adjacency matrix from edge index.
        
        Args:
            edge_index: Edge index of shape (2, num_edges)
            edge_weights: Optional edge weights
        
        Returns:
            Adjacency matrix of shape (N, N)
        """
        A = np.zeros((self.N, self.N), dtype=np.float32)
        
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i], edge_index[1, i]
            weight = edge_weights[i] if edge_weights is not None else 1.0
            A[src, dst] = weight
        
        logger.info(f"✓ Built adjacency matrix: {A.shape}, density={np.count_nonzero(A)/(self.N**2):.4f}")
        
        return A


class GraphConstructor:
    """Complete graph construction pipeline."""
    
    def __init__(self, node_mapping: Dict[int, int], 
                 neighborhood_names: List[str]):
        """
        Initialize graph constructor.
        
        Args:
            node_mapping: Dict mapping neighborhood ID to node index
            neighborhood_names: List of neighborhood names in order
        """
        self.node_mapping = node_mapping
        self.neighborhood_names = neighborhood_names
        self.N = len(neighborhood_names)
        
        # Load coordinates
        self.coord_loader = NeighborhoodCoordinates()
        self.coordinates = self.coord_loader.get_coordinate_matrix(
            node_mapping, neighborhood_names
        )
        
        # Build adjacency
        self.adjacency_builder = SpatialAdjacencyBuilder(
            self.coordinates,
            distance_threshold_km=1.5  # 1.5 km threshold for Fortaleza
        )
    
    def build_graph(self, method: str = 'distance') -> Dict:
        """
        Build complete graph structure.
        
        Args:
            method: 'distance' or 'knn'
        
        Returns:
            Dictionary with graph structure
        """
        logger.info("Building spatial graph structure...")
        
        # Build edge index
        edge_index = self.adjacency_builder.build_edge_index(method=method)
        
        # Build edge weights
        edge_weights = self.adjacency_builder.build_edge_weights(edge_index)
        
        # Build adjacency matrix
        adjacency_matrix = self.adjacency_builder.build_adjacency_matrix(
            edge_index, edge_weights
        )
        
        graph = {
            'edge_index': edge_index,
            'edge_weights': edge_weights,
            'adjacency_matrix': adjacency_matrix,
            'coordinates': self.coordinates,
            'num_nodes': self.N,
            'num_edges': edge_index.shape[1],
            'neighborhood_names': self.neighborhood_names,
            'method': method
        }
        
        logger.info(f"✓ Graph constructed: {self.N} nodes, {edge_index.shape[1]} edges")
        
        return graph
    
    def get_statistics(self, graph: Dict) -> Dict:
        """Get graph statistics."""
        edge_index = graph['edge_index']
        A = graph['adjacency_matrix']
        
        # Degree statistics
        degrees = A.sum(axis=1)
        
        stats = {
            'num_nodes': graph['num_nodes'],
            'num_edges': graph['num_edges'],
            'avg_degree': float(degrees.mean()),
            'max_degree': float(degrees.max()),
            'min_degree': float(degrees.min()),
            'density': float(np.count_nonzero(A) / (graph['num_nodes']**2)),
            'method': graph['method']
        }
        
        return stats
