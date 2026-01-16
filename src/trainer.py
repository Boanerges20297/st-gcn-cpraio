import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
<<<<<<< HEAD
import config
from model import STGCN_Cpraio
from tqdm import tqdm

# --- Configurações Estratégicas (Quinzenal) ---
HyperParams = {
    'window_size': 14,      # Olhar as últimas 2 semanas
    'target_window': 15,    # PREVER A MÉDIA DA PRÓXIMA QUINZENA
    'batch_size': 32,
    'hidden_dim': 32,       
    'epochs': 200,          # Treino longo para capturar tendências
    'learning_rate': 0.001, 
    'weight_decay': 1e-4,   
    'dropout': 0.4,
    'cvli_weight': 5.0      # Foco total no CVLI
}

class CrimeStrategyDataset(Dataset):
    """
    Dataset Estratégico:
    Input: 14 dias de histórico.
    Target: Intensidade Média dos próximos 15 dias (Suavização de Tendência).
    """
    def __init__(self, X, window_size=14, target_window=15):
        self.X = torch.FloatTensor(X)
        self.window_size = window_size
        self.target_window = target_window
        
    def __len__(self):
        # Garante dados suficientes para Input + Output
        return len(self.X) - self.window_size - self.target_window + 1

    def __getitem__(self, idx):
        # Input: Dias [0 ... 13]
        x_seq = self.X[idx : idx + self.window_size]
        
        # Target: Média dos dias [14 ... 28] (A Quinzena Futura)
        target_seq = self.X[idx + self.window_size : idx + self.window_size + self.target_window]
        y_target = target_seq.mean(dim=0) 
        
        return x_seq, y_target

class WeightedCrimeLoss(nn.Module):
    """Função de Perda que pune mais severamente erros em CVLI."""
    def __init__(self, cvli_weight=5.0):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')
        self.cvli_weight = cvli_weight

    def forward(self, pred, target):
        loss = self.mse(pred, target)
        # O Dataset tem apenas 1 feature por enquanto (CVLI), 
        # mas deixamos preparado para multiclasse.
        # Se for só 1 feature, o peso aplica em tudo.
        return (loss * self.cvli_weight).mean()

def train_region(region_name):
    print(f"\n" + "="*60)
    print(f" [ESTRATÉGIA] Iniciando Treino Tático: {region_name}")
    print("="*60)
    
    # 1. Carregar Artefatos da Região Específica
    paths = config.ARTIFACTS[region_name]
    
    if not paths['dataset'].exists():
        print(f"[!] Dataset não encontrado para {region_name}. Pulando.")
        return

    data = torch.load(paths['dataset'])
    X_full = data['X'] 
    edge_index = data['edge_index']
    
    # Validação de Segurança
    if len(X_full) < (HyperParams['window_size'] + HyperParams['target_window'] + 5):
        print(f"[!] Dados insuficientes em {region_name} para janela quinzenal.")
        return

    # 2. Normalização (Z-Score)
    mean = X_full.mean(dim=(0, 1), keepdim=True)
    std = X_full.std(dim=(0, 1), keepdim=True) + 1e-5
    X_norm = (X_full - mean) / std
    
    # Salvar estatísticas (Vital para o predict.py)
    torch.save({'mean': mean, 'std': std}, paths['stats'])
    
    # 3. Split Temporal (80% Treino / 20% Validação)
=======
import sys
from tqdm import tqdm

# Adiciona o diretório pai para importar config e model
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import config
from model import STGCN_Cpraio

# --- Configurações de Treino ---
TRAIN_CONFIG = {
    'batch_size': 32,
    'epochs': 200,          # Treino profundo
    'learning_rate': 0.001,
    'weight_decay': 1e-4,   # Regularização para evitar overfit
    'patience': 25          # Early Stopping (para se parar de aprender)
}

class CrimeSeriesDataset(Dataset):
    """
    Transforma a série temporal contínua em janelas de treino.
    Input: X dias (ex: 14)
    Target: Média dos próximos Y dias (ex: 15)
    """
    def __init__(self, X, window_size=14, target_window=15):
        self.X = X
        self.window_size = window_size
        self.target_window = target_window

    def __len__(self):
        # Garante que temos espaço para a janela de entrada + janela de saída
        return len(self.X) - self.window_size - self.target_window + 1

    def __getitem__(self, idx):
        # Janela Deslizante
        # X: (Dias, Nós, Features)
        
        # Input: Do dia 'idx' até 'idx + 14'
        x_seq = self.X[idx : idx + self.window_size]
        
        # Target: Do dia 'idx + 14' até 'idx + 29'
        # Pegamos a média desse período futuro para prever a "Intensidade Quinzenal"
        target_seq = self.X[idx + self.window_size : idx + self.window_size + self.target_window]
        y_target = target_seq.mean(dim=0) # Reduz dimensão do tempo, mantém (Nós, Features)
        
        return x_seq, y_target

def train_region(region_name):
    print(f"\n" + "="*50)
    print(f" [TREINADOR] Iniciando Protocolo: {region_name}")
    print(f"="*50)
    
    # 1. Carregar Artefatos (Dataset .pt criado pelo graph_builder)
    paths = config.ARTIFACTS[region_name]
    dataset_path = paths['dataset']
    
    if not dataset_path.exists():
        print(f"[!] Dataset não encontrado: {dataset_path}")
        print("    Execute 'python src/graph_builder.py' primeiro.")
        return

    # weights_only=False pois estamos carregando um dicionário python completo
    data = torch.load(dataset_path, weights_only=False)
    
    X_full = data['X']           # Tensor (Dias, Nós, 1)
    edge_index = data['edge_index'] # Topologia
    
    # Validação de Tamanho
    min_days = config.HyperParams['window_size'] + config.HyperParams['target_window'] + 5
    if len(X_full) < min_days:
        print(f"[!] Histórico muito curto ({len(X_full)} dias). Mínimo necessário: {min_days}.")
        return

    # 2. Normalização (Z-Score)
    # Vital para redes neurais não explodirem com números altos
    mean = X_full.mean(dim=(0, 1), keepdim=True)
    std = X_full.std(dim=(0, 1), keepdim=True) + 1e-5 # +epsilon para não dividir por zero
    
    X_norm = (X_full - mean) / std
    
    # Salvar Estatísticas (Para o preditor usar depois)
    torch.save({'mean': mean, 'std': std}, paths['stats'])
    print(f"[-] Stats salvas: Mean={mean.item():.4f}, Std={std.item():.4f}")

    # 3. Divisão Treino / Validação (80/20 cronológico)
>>>>>>> 73db3feb (Initial commit: add project files, exclude venv)
    split_idx = int(len(X_norm) * 0.8)
    train_data = X_norm[:split_idx]
    val_data = X_norm[split_idx:]
    
<<<<<<< HEAD
    # 4. DataLoaders
    train_ds = CrimeStrategyDataset(train_data, HyperParams['window_size'], HyperParams['target_window'])
    val_ds = CrimeStrategyDataset(val_data, HyperParams['window_size'], HyperParams['target_window'])
    
    train_loader = DataLoader(train_ds, batch_size=HyperParams['batch_size'], shuffle=True)
    # Val loader pode falhar se dataset for muito pequeno (interior), tratamento:
    if len(val_ds) > 0:
        val_loader = DataLoader(val_ds, batch_size=HyperParams['batch_size'], shuffle=False)
    else:
        val_loader = None
        print("[!] Aviso: Sem dados suficientes para validação nesta região.")

    # 5. Modelo
    model = STGCN_Cpraio(
        num_nodes=X_full.shape[1],
        in_channels=X_full.shape[2],
        hidden_channels=HyperParams['hidden_dim'],
        out_channels=X_full.shape[2],
        dropout=HyperParams['dropout']
    )
    
    optimizer = optim.Adam(model.parameters(), lr=HyperParams['learning_rate'], weight_decay=HyperParams['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15)
    criterion = WeightedCrimeLoss(cvli_weight=HyperParams['cvli_weight'])
    
    # 6. Loop de Treino
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(HyperParams['epochs']):
        model.train()
        total_loss = 0
        
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            out = model(x_batch, edge_index)
            loss = criterion(out, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_train_loss = total_loss / len(train_loader)
        
        # Validação
        if val_loader:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for x_batch, y_batch in val_loader:
                    out = model(x_batch, edge_index)
                    loss = criterion(out, y_batch)
                    val_loss += loss.item()
            avg_val_loss = val_loss / len(val_loader)
            scheduler.step(avg_val_loss)
            
            # Checkpoint
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), paths['model'])
                patience_counter = 0
            else:
                patience_counter += 1
        else:
            # Se não tem validação, salva sempre (overfit risk, mas necessário se dados forem poucos)
            torch.save(model.state_dict(), paths['model'])
            avg_val_loss = 0.0

        if (epoch+1) % 20 == 0:
            print(f"    Epoca {epoch+1:03d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
            
        if patience_counter >= 30:
            print("    [!] Early Stopping (Estagnação detectada).")
            break

    print(f"[V] Modelo {region_name} salvo em: {paths['model']}")

def main():
    # Loop Multi-Escala
    regions = ['CAPITAL', 'RMF', 'INTERIOR']
    for region in regions:
        try:
            train_region(region)
        except Exception as e:
            print(f"[X] Falha no treino de {region}: {e}")
=======
    # Dataset e DataLoader
    ws = config.HyperParams['window_size']
    tw = config.HyperParams['target_window']
    
    train_ds = CrimeSeriesDataset(train_data, ws, tw)
    val_ds = CrimeSeriesDataset(val_data, ws, tw)
    
    # Se a validação ficar vazia (poucos dados), usa o treino para validar (fallback)
    if len(val_ds) == 0:
        val_ds = train_ds
        print("[!] Aviso: Dados insuficientes para validação separada. Usando treino.")

    train_loader = DataLoader(train_ds, batch_size=TRAIN_CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=TRAIN_CONFIG['batch_size'], shuffle=False)

    # 4. Inicializar Modelo ST-GCN
    num_nodes = X_full.shape[1]
    num_features = X_full.shape[2]
    hidden_dim = config.HyperParams['hidden_dim']
    
    model = STGCN_Cpraio(
        num_nodes=num_nodes,
        in_channels=num_features,
        hidden_channels=hidden_dim,
        out_channels=num_features,
        dropout=config.HyperParams['dropout']
    )
    
    optimizer = optim.Adam(model.parameters(), lr=TRAIN_CONFIG['learning_rate'], weight_decay=TRAIN_CONFIG['weight_decay'])
    criterion = nn.MSELoss()
    
    # Scheduler: Reduz learning rate se o erro parar de cair
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    # 5. Loop de Treinamento
    best_loss = float('inf')
    patience_counter = 0
    
    # Barra de progresso total
    pbar = tqdm(range(TRAIN_CONFIG['epochs']), desc=f"Treinando {region_name}")
    
    for epoch in pbar:
        # --- TREINO ---
        model.train()
        train_loss = 0
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            
            # Forward
            out = model(x_batch, edge_index)
            
            # Loss
            loss = criterion(out, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # --- VALIDAÇÃO ---
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                out = model(x_batch, edge_index)
                loss = criterion(out, y_batch)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        # Atualiza Scheduler
        scheduler.step(avg_val_loss)
        
        # Atualiza Barra de Progresso
        pbar.set_postfix({'T_Loss': f'{avg_train_loss:.4f}', 'V_Loss': f'{avg_val_loss:.4f}'})
        
        # Checkpoint e Early Stopping
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), paths['model'])
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= TRAIN_CONFIG['patience']:
            print(f"\n    [!] Early Stopping na época {epoch+1}")
            break
            
    print(f"[V] Modelo salvo em: {paths['model']}")

def main():
    # Processa sequencialmente as 3 regiões estratégicas
    for region in ['CAPITAL', 'RMF', 'INTERIOR']:
        try:
            train_region(region)
        except Exception as e:
            print(f"[X] Falha crítica ao treinar {region}: {e}")
>>>>>>> 73db3feb (Initial commit: add project files, exclude venv)
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()