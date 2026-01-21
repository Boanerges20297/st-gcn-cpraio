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
        Supports inputs with or without explicit temporal dimension.
        Expected shapes:
        - (batch_size, seq_len, nodes, features)  # temporal sequence
        - (batch_size, nodes, features)            # aggregated (seq_len==1)

        edge_index shape: (2, num_edges)

        Returns: Tensor shaped (batch_size, nodes, out_channels)
        """
        # Normalize input dims
        if x.dim() == 4:
            # (batch, seq_len, nodes, features)
            batch_size, seq_len, num_nodes, num_features = x.size()
            # bring nodes to inner batch for LSTM: (batch*nodes, seq_len, features)
            x_perm = x.permute(0, 2, 1, 3).contiguous()
            x_lstm = x_perm.view(batch_size * num_nodes, seq_len, num_features)
        elif x.dim() == 3:
            # (batch, nodes, features) -> treat as seq_len=1
            batch_size, num_nodes, num_features = x.size()
            x_lstm = x.view(batch_size * num_nodes, 1, num_features)
        else:
            raise ValueError(f"Unexpected input tensor shape: {x.shape}")

        # LSTM output: take last hidden state
        _, (h_n, _) = self.lstm(x_lstm)
        h_n = h_n.squeeze(0)  # (batch*nodes, hidden)

        # Apply GCN per batch sample because edge_index indexes nodes 0..num_nodes-1
        # and cannot be reused directly on concatenated batch unless offsets applied.
        outputs = []
        for b in range(batch_size):
            start = b * num_nodes
            end = (b + 1) * num_nodes
            h_slice = h_n[start:end]  # (num_nodes, hidden)

            x_gcn = self.gcn1(h_slice, edge_index)
            x_gcn = F.relu(x_gcn)
            x_gcn = F.dropout(x_gcn, p=self.dropout_rate, training=self.training)

            x_gcn = self.gcn2(x_gcn, edge_index)
            x_gcn = F.relu(x_gcn)
            x_gcn = F.dropout(x_gcn, p=self.dropout_rate, training=self.training)

            outputs.append(x_gcn)

        x_gcn_all = torch.cat(outputs, dim=0)  # (batch*nodes, hidden)

        # FC head
        out = self.fc1(x_gcn_all)
        out = F.relu(out)
        out = self.fc2(out)  # (batch*nodes, out_channels)

        # reshape to (batch, nodes, out_channels)
        out = out.view(batch_size, num_nodes, -1)
        return out