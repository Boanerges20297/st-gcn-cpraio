import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# --- CORREÇÃO DE IMPORTS (Blindagem) ---
# Força o Python a priorizar a pasta local (src) e a raiz do projeto
current_dir = os.path.dirname(os.path.abspath(__file__)) # Pasta src/
project_root = os.path.dirname(current_dir)              # Pasta raiz/

# Insere no INÍCIO da lista de busca (prioridade máxima)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Agora os imports funcionam garantidos
import config
try:
    from model import STGCN_Cpraio
except ImportError:
    # Fallback caso o Python entenda 'src' como pacote
    from src.model import STGCN_Cpraio

# --- Configurações de Treino ---
TRAIN_CONFIG = {
    'batch_size': 32,
    'epochs': 200,          # Treino profundo
    'learning_rate': 0.001,
    'weight_decay': 1e-4,   # Regularização
    'patience': 25          # Early Stopping
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
        # X: (Dias, Nós, Features)
        
        # Input: Do dia 'idx' até 'idx + 14'
        x_seq = self.X[idx : idx + self.window_size]
        
        # Target: Do dia 'idx + 14' até 'idx + 29'
        # Pegamos a média para suavizar a previsão quinzenal
        target_seq = self.X[idx + self.window_size : idx + self.window_size + self.target_window]
        y_target = target_seq.mean(dim=0) 
        
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

    # weights_only=False pois estamos carregando dicionário completo
    data = torch.load(dataset_path, weights_only=False)
    
    X_full = data['X']           
    edge_index = data['edge_index'] 
    
    # Validação de Tamanho
    min_days = config.HyperParams['window_size'] + config.HyperParams['target_window'] + 5
    if len(X_full) < min_days:
        print(f"[!] Histórico muito curto ({len(X_full)} dias). Mínimo necessário: {min_days}.")
        return

    # 2. Normalização (Z-Score)
    mean = X_full.mean(dim=(0, 1), keepdim=True)
    std = X_full.std(dim=(0, 1), keepdim=True) + 1e-5 
    
    X_norm = (X_full - mean) / std
    
    # Salvar Estatísticas
    torch.save({'mean': mean, 'std': std}, paths['stats'])
    print(f"[-] Stats salvas: Mean={mean.item():.4f}, Std={std.item():.4f}")

    # 3. Divisão Treino / Validação (80/20)
    split_idx = int(len(X_norm) * 0.8)
    train_data = X_norm[:split_idx]
    val_data = X_norm[split_idx:]
    
    # Dataset e DataLoader
    ws = config.HyperParams['window_size']
    tw = config.HyperParams['target_window']
    
    train_ds = CrimeSeriesDataset(train_data, ws, tw)
    val_ds = CrimeSeriesDataset(val_data, ws, tw)
    
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
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    # 5. Loop de Treinamento
    best_loss = float('inf')
    patience_counter = 0
    
    pbar = tqdm(range(TRAIN_CONFIG['epochs']), desc=f"Treinando {region_name}")
    
    for epoch in pbar:
        model.train()
        train_loss = 0
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            out = model(x_batch, edge_index)
            loss = criterion(out, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                out = model(x_batch, edge_index)
                loss = criterion(out, y_batch)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)
        
        pbar.set_postfix({'T_Loss': f'{avg_train_loss:.4f}', 'V_Loss': f'{avg_val_loss:.4f}'})
        
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
    for region in ['CAPITAL', 'RMF', 'INTERIOR']:
        try:
            train_region(region)
        except Exception as e:
            print(f"[X] Falha crítica ao treinar {region}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()