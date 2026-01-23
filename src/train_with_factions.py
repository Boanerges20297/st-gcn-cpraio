#!/usr/bin/env python3
"""
TREINADOR ST-GCN COM DINÂMICA DE FACÇÕES
Treina modelo considerando movimentação territorial
"""

import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import json
from pathlib import Path
from datetime import datetime
import logging

# --- Setup de Paths ---
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

import config
from model_faction_adapter import STGCN_DynamicFactions, DynamicFactionLoss

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# DATASET COM DINÂMICA DE FACÇÕES
# ============================================================================

class TimeSeriesDatasetWithFactions:
    """
    Carrega tensor 7D e cria windows para treino
    
    Entrada: (1472, 121, 7)
      - Dims 0-2: Crime data
      - Dims 3-6: Faction dynamics
    
    Output: Windows de 14 dias para predição de 15 dias
    """
    
    def __init__(self, tensor_path, window_size=14, target_window=15):
        logger.info(f"Carregando tensor: {tensor_path}")
        
        X = np.load(tensor_path)
        self.X = torch.from_numpy(X).float()  # (T, N, F)
        
        self.window_size = window_size
        self.target_window = target_window
        self.num_samples = len(X) - window_size - target_window + 1
        
        logger.info(f"✓ Tensor carregado: {self.X.shape}")
        logger.info(f"✓ Amostras disponíveis: {self.num_samples}")
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        """Retorna (X_window, y_target, faction_features)"""
        # Input window: 14 dias
        X_window = self.X[idx : idx + self.window_size]  # (14, N, 7)
        
        # Target window: próximos 15 dias (média)
        y_window = self.X[idx + self.window_size : idx + self.window_size + self.target_window, :, :3]  # (15, N, 3)
        y_target = y_window.mean(dim=0, keepdim=True)  # (1, N, 3) → tomar CVLI apenas
        y_target = y_target[:, :, 0:1]  # (1, N, 1) - apenas CVLI
        
        # Features de facções (últimas 4 dimensões)
        faction_features = X_window[:, :, 3:7]  # (14, N, 4)
        
        return X_window, y_target.squeeze(0), faction_features

# ============================================================================
# TREINAMENTO
# ============================================================================

class Trainer:
    def __init__(self, model, loss_fn, optimizer, device, config_dict):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.config = config_dict
        self.best_val_loss = float('inf')
        self.patience_counter = 0
    
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        
        for batch_idx, (X_batch, y_batch, faction_batch) in enumerate(train_loader):
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            faction_batch = faction_batch.to(self.device)
            
            # Forward pass
            pred, aux_pred = self.model(X_batch, return_aux=True)
            
            # Criar target para mudança (simplificado: qualquer mudança territorial é 1)
            aux_target = (faction_batch[:, -1, :, 0:1] > 0.5).float()  # Mudança recente?
            
            # Loss
            loss = self.loss_fn(pred, y_batch, faction_batch[:, -1], aux_pred, aux_target)
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for X_batch, y_batch, faction_batch in val_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                faction_batch = faction_batch.to(self.device)
                
                pred, aux_pred = self.model(X_batch, return_aux=True)
                aux_target = (faction_batch[:, -1, :, 0:1] > 0.5).float()
                
                loss = self.loss_fn(pred, y_batch, faction_batch[:, -1], aux_pred, aux_target)
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def fit(self, train_loader, val_loader, epochs=200, patience=25):
        logger.info(f"\n{'='*60}")
        logger.info(f"INICIANDO TREINAMENTO")
        logger.info(f"{'='*60}")
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            
            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self._save_checkpoint()
            else:
                self.patience_counter += 1
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1:3d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Patience: {self.patience_counter}/{patience}")
            
            if self.patience_counter >= patience:
                logger.info(f"✅ Early stopping em epoch {epoch+1}")
                break
        
        return self.best_val_loss
    
    def _save_checkpoint(self):
        """Salva melhor modelo"""
        checkpoint = {
            'model_state': self.model.state_dict(),
            'best_val_loss': self.best_val_loss,
            'timestamp': datetime.now().isoformat()
        }
        
        model_path = config.OUTPUT_DIR / 'model_stgcn_faccoes.pth'
        torch.save(checkpoint, model_path)
        logger.info(f"   💾 Modelo salvo: {model_path}")

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*80)
    print("TREINADOR ST-GCN COM DINÂMICA DE FACÇÕES")
    print("="*80)
    print()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    
    # Criar dataset
    logger.info("\n[STAGE 1] Carregando dados...")
    tensor_path = config.DATA_PROCESSED / 'tensor_cvli_prisoes_faccoes.npy'
    
    if not tensor_path.exists():
        logger.error(f"❌ Tensor não encontrado: {tensor_path}")
        logger.error("   Execute: python src/data/analyze_faction_movements.py")
        return False
    
    dataset = TimeSeriesDatasetWithFactions(tensor_path)
    
    # Split
    total_samples = len(dataset)
    train_size = int(0.7 * total_samples)
    val_size = int(0.15 * total_samples)
    test_size = total_samples - train_size - val_size
    
    indices = torch.randperm(total_samples)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    logger.info(f"✓ Split: Train={train_size}, Val={val_size}, Test={test_size}")
    
    # DataLoaders
    batch_size = 16
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, train_indices),
        batch_size=batch_size,
        shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, val_indices),
        batch_size=batch_size
    )
    
    # Modelo
    logger.info("\n[STAGE 2] Inicializando modelo...")
    model = STGCN_DynamicFactions(
        input_features=7,
        hidden_dim=32,
        output_dim=1,
        dropout=0.4,
        num_nodes=121
    )
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"✓ Modelo criado: {total_params:,} parâmetros")
    
    # Loss & Optimizer
    loss_fn = DynamicFactionLoss(cvli_weight=5.0, faction_weight=0.5)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10)
    
    # Trainer
    trainer = Trainer(model, loss_fn, optimizer, device, {})
    
    # Treinar
    logger.info("\n[STAGE 3] Iniciando treinamento...")
    best_val_loss = trainer.fit(train_loader, val_loader, epochs=200, patience=25)
    
    # Avaliação
    logger.info("\n[STAGE 4] Avaliação em test set...")
    model.eval()
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, test_indices),
        batch_size=batch_size
    )
    
    test_predictions = []
    test_targets = []
    
    with torch.no_grad():
        for X_batch, y_batch, _ in test_loader:
            X_batch = X_batch.to(device)
            pred, _ = model(X_batch, return_aux=False)
            test_predictions.append(pred.cpu().numpy())
            test_targets.append(y_batch.cpu().numpy())
    
    test_predictions = np.concatenate(test_predictions, axis=0)
    test_targets = np.concatenate(test_targets, axis=0)
    
    # Métricas
    test_mse = np.mean((test_predictions - test_targets) ** 2)
    test_rmse = np.sqrt(test_mse)
    test_mae = np.mean(np.abs(test_predictions - test_targets))
    
    logger.info(f"✓ Test MSE:  {test_mse:.4f}")
    logger.info(f"✓ Test RMSE: {test_rmse:.4f}")
    logger.info(f"✓ Test MAE:  {test_mae:.4f}")
    
    # Salvar relatório
    report = {
        'timestamp': datetime.now().isoformat(),
        'status': 'TREINAMENTO_CONCLUIDO',
        'model': 'STGCN_DynamicFactions',
        'dataset': {
            'tensor': str(tensor_path),
            'shape': [1472, 121, 7],
            'split': {'train': train_size, 'val': val_size, 'test': test_size}
        },
        'hyperparameters': {
            'window_size': 14,
            'target_window': 15,
            'batch_size': 16,
            'hidden_dim': 32,
            'dropout': 0.4,
            'learning_rate': 0.001,
            'epochs': 200
        },
        'performance': {
            'best_val_loss': float(best_val_loss),
            'test_mse': float(test_mse),
            'test_rmse': float(test_rmse),
            'test_mae': float(test_mae)
        }
    }
    
    report_path = config.OUTPUT_DIR / 'TREINAMENTO_FACCOES_RELATORIO.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print("\n" + "="*80)
    print("✅ TREINAMENTO CONCLUÍDO COM SUCESSO")
    print("="*80)
    print(f"\n📊 Resultados:")
    print(f"   - Best Val Loss: {best_val_loss:.4f}")
    print(f"   - Test RMSE: {test_rmse:.4f}")
    print(f"   - Test MAE: {test_mae:.4f}")
    print(f"\n💾 Modelo salvo em: outputs/model_stgcn_faccoes.pth")
    print(f"📋 Relatório em: {report_path}")
    print()
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
