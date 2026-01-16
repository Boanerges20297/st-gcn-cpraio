import torch
import torch.nn as nn
import torch.nn.functional as F
<<<<<<< HEAD
from torch_geometric.nn import DenseGCNConv
from torch_geometric.utils import to_dense_adj

class STGCN_Cpraio(nn.Module):
    """
    Arquitetura Espaço-Temporal para Predição de Risco Criminal.
    
    Fluxo de Dados:
    1. Entrada: Tensor (Batch, Time_Steps, Nodes, Features)
    2. Spatial Block: Mistura informações entre bairros vizinhos a cada timestep.
    3. Temporal Block: Analisa a evolução histórica de cada bairro (agora enriquecido pelos vizinhos).
    4. Head de Predição: Projeta o risco futuro (com foco em CVLI).
    """
    
    def __init__(self, num_nodes, in_channels, hidden_channels, out_channels, dropout=0.3):
        super().__init__()
        self.num_nodes = num_nodes
        self.dropout = dropout
        
        # --- 1. Processamento Espacial (GCN) ---
        # Aumenta a dimensão das features misturando com vizinhos
        self.gcn1 = DenseGCNConv(in_channels, hidden_channels)
        self.gcn2 = DenseGCNConv(hidden_channels, hidden_channels)
        
        # --- 2. Processamento Temporal (LSTM) ---
        # O LSTM processa a sequência temporal para CADA nó independentemente
        # input_size = hidden_channels (que veio da GCN)
        self.lstm = nn.LSTM(
            input_size=hidden_channels, 
            hidden_size=hidden_channels * 2, # Aumenta capacidade para memória temporal
            num_layers=1, 
            batch_first=True
        )
        
        # --- 3. Decodificador (Output Head) ---
        # Reduz de volta para a dimensão de predição (Ex: 1 canal de Risco CVLI ou N canais de crimes)
        self.head = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels)
        )

    def forward(self, x, edge_index):
        """
        x: (Batch, Time, Nodes, Features) -> O Histórico recente
        edge_index: (2, Num_Edges) -> A topologia da cidade
        """
        B, T, N, F_in = x.size()
        
        # Conversão para Matriz de Adjacência Densa (B, N, N) necessária para DenseGCNConv
        # Como a cidade é a mesma para todo o batch, repetimos a adjacência
        adj = to_dense_adj(edge_index, max_num_nodes=N)[0] # (N, N)
        adj = adj.unsqueeze(0).repeat(B * T, 1, 1) # (Batch*Time, N, N)
        
        # --- Passo A: Spatial Mixing ---
        # Achatar Batch e Time para aplicar GCN em todos os frames de uma vez
        # (Batch * Time, Nodes, Features)
        x_flat = x.view(B * T, N, F_in)
        
        # Aplica GCN
        h = F.relu(self.gcn1(x_flat, adj))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.relu(self.gcn2(h, adj)) # (B*T, N, Hidden)
        
        # --- Passo B: Temporal Processing ---
        # Agora queremos que o LSTM veja: "Para o Bairro X, aqui está a sequência histórica"
        # Reshape para: (Batch * Nodes, Time, Hidden)
        # Primeiro voltamos para (B, T, N, H)
        h = h.view(B, T, N, -1)
        # Transpomos para (B, N, T, H) e achatamos B e N
        h = h.permute(0, 2, 1, 3).contiguous().view(B * N, T, -1)
        
        # Passar pelo LSTM
        # out: (B*N, T, Hidden*2), hn: (1, B*N, Hidden*2)
        lstm_out, _ = self.lstm(h)
        
        # Pegamos apenas o ÚLTIMO estado (o momento T, resumo de toda a história)
        last_state = lstm_out[:, -1, :] # (B*N, Hidden*2)
        
        # --- Passo C: Predição ---
        out = self.head(last_state) # (B*N, Out_Channels)
        
        # Retornar formato estruturado (Batch, Nodes, Out_Channels)
        return out.view(B, N, -1)
=======
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
>>>>>>> 73db3feb (Initial commit: add project files, exclude venv)
