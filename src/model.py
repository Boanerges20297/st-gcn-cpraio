import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class STGCN_Cpraio(nn.Module):
    """
    ST-GCN para CPRAIO com processamento correto de batch.
    
    Arquitetura:
    1. LSTM: Aprende sequência temporal de cada nó
    2. GCN: Aprende influência espacial entre nós (contagio)
    3. FC: Predição final
    """
    def __init__(self, num_nodes, in_channels, hidden_channels, out_channels, dropout=0.3):
        super(STGCN_Cpraio, self).__init__()
        
        self.num_nodes = num_nodes
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.dropout_rate = dropout
        
        # Temporal: LSTM por nó
        self.lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=hidden_channels,
            batch_first=True
        )
        
        # Spatial: GCN
        self.gcn1 = GCNConv(hidden_channels, hidden_channels)
        self.gcn2 = GCNConv(hidden_channels, hidden_channels)
        
        # Prediction head
        self.fc1 = nn.Linear(hidden_channels, hidden_channels // 2)
        self.fc2 = nn.Linear(hidden_channels // 2, out_channels)
        
    def forward(self, x, edge_index):
        """
        x shape: (batch_size, nodes, features) - sequência já agregada
        edge_index shape: (2, num_edges)
        
        Retorna: (batch_size * nodes, out_channels)
        """
        batch_size, num_nodes, num_features = x.size()
        
        # 1. LSTM: processar cada nó como série temporal
        # Reshape: (batch*nodes, seq_len, features)
        x_lstm = x.view(batch_size * num_nodes, 1, num_features)
        
        # LSTM output
        _, (h_n, _) = self.lstm(x_lstm)
        h_n = h_n.squeeze(0)  # (batch*nodes, hidden)
        
        # 2. GCN: processar conexões espaciais
        # Aplicar GCN 1
        x_gcn = self.gcn1(h_n, edge_index)
        x_gcn = F.relu(x_gcn)
        x_gcn = F.dropout(x_gcn, p=self.dropout_rate, training=self.training)
        
        # Aplicar GCN 2
        x_gcn = self.gcn2(x_gcn, edge_index)
        x_gcn = F.relu(x_gcn)
        x_gcn = F.dropout(x_gcn, p=self.dropout_rate, training=self.training)
        
        # 3. FC: predição
        out = self.fc1(x_gcn)
        out = F.relu(out)
        out = self.fc2(out)
        
        return out
        # Retorna ao formato original: (Batch, Nodes, Output_Features)
        out = out.view(batch_size, num_nodes, -1)
        
        return out