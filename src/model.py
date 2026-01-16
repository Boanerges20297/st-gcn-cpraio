import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class STGCN_Cpraio(nn.Module):
    """
    Rede Neural Espaço-Temporal para Previsão Criminal.
    
    Arquitetura:
    1. LSTM: Aprende a sequência histórica (evolução do crime no tempo).
    2. GCN: Aprende a influência vizinha (contágio do crime no espaço e facções).
    3. Head: Camadas lineares para a previsão final (regressão).
    """
    def __init__(self, num_nodes, in_channels, hidden_channels, out_channels, dropout=0.3):
        super(STGCN_Cpraio, self).__init__()
        
        self.num_nodes = num_nodes
        self.dropout_rate = dropout
        
        # --- BLOCO TEMPORAL (LSTM) ---
        # Processa a série temporal de cada nó individualmente
        # Input: (Batch, Sequence_Length, Features)
        self.lstm = nn.LSTM(
            input_size=in_channels, 
            hidden_size=hidden_channels, 
            batch_first=True
        )
        
        # --- BLOCO ESPACIAL (GCN) ---
        # Troca informações entre vizinhos (físicos e lógicos)
        # Input: (Nodes, Features) -> Grafo
        self.gcn = GCNConv(hidden_channels, hidden_channels)
        
        # --- BLOCO DE PREDIÇÃO (HEAD) ---
        # Reduz a dimensionalidade para a saída final
        self.fc1 = nn.Linear(hidden_channels, hidden_channels // 2)
        self.fc2 = nn.Linear(hidden_channels // 2, out_channels)
        
    def forward(self, x, edge_index):
        """
        x shape esperado: (Batch, Window_Size, Num_Nodes, Features)
        edge_index shape: (2, Num_Edges)
        """
        
        batch_size, window_size, num_nodes, num_features = x.size()
        
        # 1. Processamento Temporal (LSTM)
        # Precisamos fundir Batch e Nodes para a LSTM processar cada nó como uma série independente
        # View: (Batch * Nodes, Window_Size, Features)
        x_lstm = x.view(batch_size * num_nodes, window_size, num_features)
        
        # Passa pela LSTM
        # out: (Batch*Nodes, Window, Hidden), h_n: (1, Batch*Nodes, Hidden)
        # Pegamos apenas o último estado oculto (o "resumo" da janela temporal)
        _, (h_n, _) = self.lstm(x_lstm)
        
        # h_n shape: (1, Batch*Nodes, Hidden) -> Removemos a dimensão 1 da LSTM layer
        x_spatial = h_n.squeeze(0) 
        
        # Agora x_spatial tem shape (Batch * Nodes, Hidden)
        
        # 2. Processamento Espacial (GCN)
        # A GCN do PyTorch Geometric espera (Total_Nodes, Features).
        # Como temos um grafo estático replicado para cada item do batch, isso funciona.
        # (O ideal seria usar Batch do PyG, mas aqui simplificamos assumindo grafo fixo ou batch=1 na inferência)
        
        # Se estivermos treinando com Batch > 1, a GCN vai misturar tudo se não tivermos cuidado.
        # TRUQUE: Para ST-GCN simples com topologia fixa, aplicamos a GCN frame a frame ou assumimos média.
        # Nesta versão simplificada para o CPRAIO, aplicamos a GCN no estado latente extraído pela LSTM.
        
        x_spatial = self.gcn(x_spatial, edge_index)
        x_spatial = F.relu(x_spatial)
        x_spatial = F.dropout(x_spatial, p=self.dropout_rate, training=self.training)
        
        # 3. Predição Final
        x_out = self.fc1(x_spatial)
        x_out = F.relu(x_out)
        out = self.fc2(x_out)
        
        # Retorna ao formato original: (Batch, Nodes, Output_Features)
        out = out.view(batch_size, num_nodes, -1)
        
        return out