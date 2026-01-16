import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
from model import STGCN_Cpraio

# --- Configurações Táticas V2 (Anti-Overfitting) ---
CONFIG = {
    'window_size': 7,
    'batch_size': 32,
    
    # Reduzimos a "memória" para evitar decoreba
    'hidden_dim': 32,       # Antes era 64
    
    'epochs': 150,          # Mais épocas, pois o aprendizado será mais lento (harder)
    'learning_rate': 0.001,
    
    # Aumentamos a punição por pesos complexos
    'weight_decay': 1e-4,   # L2 Regularization (NOVO)
    'dropout': 0.5,         # Dropout Agressivo (Antes era 0.3)
    
    'cvli_weight': 10.0,
    'data_path': 'data/tensors/stgcn_dataset.pt',
    'model_save_path': 'outputs/models/best_stgcn.pth'
}

class CrimeTimeDataset(Dataset):
    def __init__(self, X, window_size=7):
        self.X = torch.FloatTensor(X)
        self.window_size = window_size
    def __len__(self): return len(self.X) - self.window_size
    def __getitem__(self, idx):
        return self.X[idx : idx + self.window_size], self.X[idx + self.window_size]

class WeightedCrimeLoss(nn.Module):
    def __init__(self, feature_names, cvli_weight=10.0):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')
        self.cvli_weight = cvli_weight
        try:
            self.cvli_idx = feature_names.index('CVLI')
            print(f"[-] Foco Tático: CVLI detectado no índice {self.cvli_idx}. Peso aplicado: {cvli_weight}x")
        except ValueError:
            self.cvli_idx = -1

    def forward(self, pred, target):
        loss = self.mse(pred, target)
        if self.cvli_idx >= 0:
            weights = torch.ones_like(loss)
            weights[:, :, self.cvli_idx] = self.cvli_weight
            loss = loss * weights
        return loss.mean()

def train():
    os.makedirs(os.path.dirname(CONFIG['model_save_path']), exist_ok=True)
    
    print("[-] Carregando Tensores...")
    data = torch.load(CONFIG['data_path'])
    X_full = data['X'] 
    edge_index = data['edge_index']
    
    # Normalização Robusta
    mean = X_full.mean(dim=(0, 1), keepdim=True)
    std = X_full.std(dim=(0, 1), keepdim=True) + 1e-5
    X_norm = (X_full - mean) / std
    torch.save({'mean': mean, 'std': std}, 'outputs/models/scaler_stats.pt')
    
    # Split Temporal Estrito
    split_idx = int(len(X_norm) * 0.8)
    train_data = X_norm[:split_idx]
    val_data = X_norm[split_idx:]
    
    train_loader = DataLoader(CrimeTimeDataset(train_data, CONFIG['window_size']), batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(CrimeTimeDataset(val_data, CONFIG['window_size']), batch_size=CONFIG['batch_size'], shuffle=False)
    
    # Inicializar Modelo com Dropout Agressivo
    model = STGCN_Cpraio(
        num_nodes=X_full.shape[1],
        in_channels=X_full.shape[2],
        hidden_channels=CONFIG['hidden_dim'], # 32
        out_channels=X_full.shape[2],
        dropout=CONFIG['dropout']             # 0.5
    )
    
    # Otimizador com Weight Decay
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
    
    # Scheduler para reduzir o LR se o loss estagnar (AJUSTE FINO)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    
    criterion = WeightedCrimeLoss(data['features'], cvli_weight=CONFIG['cvli_weight'])
    
    print(f"[-] Iniciando Treinamento V2 (Regularizado)...")
    best_val_loss = float('inf')
    patience_counter = 0 # Early Stopping manual
    
    for epoch in range(CONFIG['epochs']):
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
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                out = model(x_batch, edge_index)
                loss = criterion(out, y_batch)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        # Atualiza o Learning Rate se necessário
        scheduler.step(avg_val_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoca {epoch+1:03d} | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f}")
            
        # Checkpoint com Early Stopping simplificado
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), CONFIG['model_save_path'])
            patience_counter = 0
        else:
            patience_counter += 1
            
        # Se a loss de validação não melhorar por 25 épocas, para.
        if patience_counter >= 25:
            print(f"[!] Early Stopping ativado na época {epoch+1}. Modelo parou de melhorar.")
            break
            
    print(f"[V] Treino Finalizado. Melhor Val Loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    train()