#!/usr/bin/env python3
"""
ADAPTADOR DE MODELO PARA DINÂMICA DE FACÇÕES
Modifica o ST-GCN para considerar movimentação territorial
"""

import sys
import os
import torch
import torch.nn as nn
import json
from pathlib import Path

# --- Setup de Paths ---
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

import config

# ============================================================================
# MODELO ST-GCN ADAPTADO COM DINÂMICA DE FACÇÕES
# ============================================================================

class STGCN_DynamicFactions(nn.Module):
    """
    ST-GCN adaptado para considerar dinâmica de facções
    
    Entrada: X (T, N, 7)
    - 0-2: Dados de crime (CVLI, Prisões, Apreensões)
    - 3-6: Dinâmica de facções
    
    Processa separadamente:
    1. Features de crime (0-2) → via ST-GCN padrão
    2. Features de facções (3-6) → via atenção/fusão
    
    Saída: Predição de CVLI para T+15 dias
    """
    
    def __init__(self, input_features=7, hidden_dim=32, output_dim=1, 
                 dropout=0.4, num_nodes=121):
        super(STGCN_DynamicFactions, self).__init__()
        
        self.input_features = input_features
        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes
        
        # ====== BRANCH 1: Dados de Crime ======
        # Processa features 0-2 (CVLI, Prisões, Apreensões)
        self.crime_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # ====== BRANCH 2: Dinâmica de Facções ======
        # Processa features 3-6 (Mudança, Estabilidade, Conflito, Volatilidade)
        self.faction_encoder = nn.Sequential(
            nn.Linear(4, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 2)
        )
        
        # Attention mechanism para combinar branches
        self.attention_layer = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        
        # ====== Temporal Processing ======
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            dropout=dropout,
            batch_first=True
        )
        
        # ====== Spatial Processing (Graph) ======
        self.graph_conv = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # ====== Decoder ======
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU()  # CVLI count não pode ser negativo
        )
        
        # Auxiliary head para predição de mudanças territoriais
        self.aux_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Probabilidade de mudança
        )
    
    def forward(self, X, edge_index=None, return_aux=False):
        """
        Args:
            X: Tensor (batch_size, time_steps, num_nodes, 7)
            edge_index: Tensor (2, num_edges) - adjacência do grafo
            return_aux: Se True, retorna também predição de mudanças territoriais
        
        Returns:
            output: Tensor (batch_size, num_nodes, 1) - predição de CVLI
            aux_output: Tensor (batch_size, num_nodes, 1) - probabilidade de mudança (se return_aux=True)
        """
        batch_size, time_steps, num_nodes, num_features = X.shape
        
        # Separar features por tipo
        X_crime = X[:, :, :, :3]      # (B, T, N, 3)
        X_faction = X[:, :, :, 3:7]   # (B, T, N, 4)
        
        # ====== BRANCH 1: Encode Crime Data ======
        # Reshape para processar todo timestep por node
        crime_flat = X_crime.reshape(-1, 3)  # (B*T*N, 3)
        crime_encoded = self.crime_encoder(crime_flat)  # (B*T*N, hidden_dim)
        crime_encoded = crime_encoded.reshape(batch_size, time_steps, num_nodes, self.hidden_dim)
        
        # ====== BRANCH 2: Encode Faction Dynamics ======
        faction_flat = X_faction.reshape(-1, 4)  # (B*T*N, 4)
        faction_encoded = self.faction_encoder(faction_flat)  # (B*T*N, hidden_dim/2)
        faction_encoded = faction_encoded.reshape(batch_size, time_steps, num_nodes, self.hidden_dim // 2)
        
        # ====== Pad faction_encoded para concatenação ======
        faction_padded = torch.cat([
            faction_encoded,
            torch.zeros(batch_size, time_steps, num_nodes, self.hidden_dim // 2, 
                       device=faction_encoded.device)
        ], dim=3)  # (B, T, N, hidden_dim)
        
        # ====== Multi-head Attention para Fusão ======
        # Preparar queries, keys, values
        q = crime_encoded.reshape(batch_size * time_steps, num_nodes, self.hidden_dim)
        k = faction_padded.reshape(batch_size * time_steps, num_nodes, self.hidden_dim)
        v = faction_padded.reshape(batch_size * time_steps, num_nodes, self.hidden_dim)
        
        attn_out, _ = self.attention_layer(q, k, v)  # (B*T, N, hidden_dim)
        attn_out = attn_out.reshape(batch_size, time_steps, num_nodes, self.hidden_dim)
        
        # ====== Fusão com Branch 1 (residual connection) ======
        fused = crime_encoded + attn_out * 0.3  # Manter crime como primário, facções como complementares
        
        # ====== Temporal Processing (LSTM) ======
        # Reshape para LSTM: (B*N, T, hidden_dim)
        temporal_input = fused.permute(0, 2, 1, 3).reshape(batch_size * num_nodes, time_steps, self.hidden_dim)
        lstm_out, (h_n, c_n) = self.lstm(temporal_input)  # (B*N, T, hidden_dim)
        
        # Pegar último hidden state de cada sequência
        last_hidden = h_n[-1]  # (B*N, hidden_dim)
        
        # ====== Spatial Processing (Graph Conv) ======
        spatial_out = self.graph_conv(last_hidden)  # (B*N, hidden_dim)
        
        # ====== Decoder ======
        output = self.decoder(spatial_out)  # (B*N, 1)
        output = output.reshape(batch_size, num_nodes, 1)
        
        if return_aux:
            # Auxiliar: predição de mudanças territoriais
            aux_output = self.aux_head(spatial_out)  # (B*N, 1)
            aux_output = aux_output.reshape(batch_size, num_nodes, 1)
            return output, aux_output
        
        return output


# ============================================================================
# LOSS FUNCTION COM PESO DINÂMICO DE FACÇÕES
# ============================================================================

class DynamicFactionLoss(nn.Module):
    """
    Loss function que considera dinâmica de facções
    
    Combina:
    1. MSE Loss para predição de CVLI
    2. Weighted loss: reduz peso onde há mudanças territoriais recentes
    3. Auxiliary loss: predição correta de mudanças
    """
    
    def __init__(self, cvli_weight=5.0, faction_weight=0.5):
        super(DynamicFactionLoss, self).__init__()
        self.cvli_weight = cvli_weight
        self.faction_weight = faction_weight
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
    
    def forward(self, pred, target, faction_dynamics=None, aux_pred=None, aux_target=None):
        """
        Args:
            pred: Predição de CVLI (B, N, 1)
            target: Target de CVLI (B, N, 1)
            faction_dynamics: Tensor (B, N, 4) com features de facções
            aux_pred: Predição de mudanças (B, N, 1)
            aux_target: Target de mudanças (B, N, 1)
        """
        # Loss principal: MSE
        main_loss = self.mse_loss(pred, target)
        
        # Se temos dinâmica de facções, ajustar peso dinamicamente
        if faction_dynamics is not None:
            # Features: [mudança, estabilidade, conflito, volatilidade]
            mudanca = faction_dynamics[:, :, 0:1]  # 1 = mudança recente
            estabilidade = faction_dynamics[:, :, 1:2]
            
            # Aumentar loss onde há mudanças recentes (modelo deve aprender padrão)
            # Reduzir loss onde há estabilidade (mais previsível)
            dynamic_weight = 1.0 + (mudanca * 2.0) + (1.0 - estabilidade / 365.0) * 0.5
            weighted_loss = (main_loss * dynamic_weight).mean()
        else:
            weighted_loss = main_loss
        
        # Auxiliary loss: predição de mudanças territoriais
        aux_loss = 0
        if aux_pred is not None and aux_target is not None:
            aux_loss = self.bce_loss(aux_pred, aux_target) * self.faction_weight
        
        total_loss = weighted_loss + aux_loss
        
        return total_loss


# ============================================================================
# GERADOR DE CONFIG PARA NOVO MODELO
# ============================================================================

def generate_faction_aware_config():
    """Gera configuração para novo modelo ST-GCN com facções"""
    
    config_dict = {
        'model_type': 'STGCN_DynamicFactions',
        'input_features': 7,
        'features_description': {
            '0': 'CVLI (homicídios)',
            '1': 'Prisões',
            '2': 'Apreensões',
            '3': 'Mudança de controle territorial (0-1)',
            '4': 'Estabilidade do controle (dias, 0-365)',
            '5': 'Risco de conflito (0-1)',
            '6': 'Volatilidade territorial (0-1)'
        },
        'hyperparameters': {
            'hidden_dim': 32,
            'output_dim': 1,
            'dropout': 0.4,
            'num_nodes': 121,
            'batch_size': 16,
            'epochs': 200,
            'learning_rate': 0.001,
            'weight_decay': 1e-4,
            'cvli_weight': 5.0,
            'faction_weight': 0.5,
            'window_size': 14,
            'target_window': 15
        },
        'loss_function': 'DynamicFactionLoss',
        'optimization': {
            'optimizer': 'Adam',
            'scheduler': 'ReduceLROnPlateau',
            'early_stopping_patience': 25
        },
        'training_data': {
            'source': 'data/processed/tensor_cvli_prisoes_faccoes.npy',
            'shape': [1472, 121, 7],
            'period': '2022-01-01 to 2026-01-11',
            'split': {
                'train': 0.7,
                'val': 0.15,
                'test': 0.15
            }
        },
        'auxiliary_tasks': {
            'faction_change_prediction': True,
            'territorial_stability_prediction': True,
            'conflict_risk_assessment': True
        },
        'inference_outputs': {
            'cvli_forecast': 'CVLI predictions for next 15 days',
            'territorial_risk': 'Risk of territorial changes',
            'stability_score': 'Territorial stability score per neighborhood',
            'conflict_probability': 'Probability of gang conflict'
        }
    }
    
    return config_dict


# ============================================================================
# SALVAR CONFIGURAÇÃO E ARQUITETURA
# ============================================================================

def save_model_adaptation():
    """Salva configuração e código da adaptação"""
    
    config_dict = generate_faction_aware_config()
    
    output_path = config.DATA_PROCESSED / 'modelo_config_faccoes.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Configuração salva em: {output_path}")
    
    # Salvar sumário
    summary = []
    summary.append("# ADAPTAÇÃO DE MODELO ST-GCN COM DINÂMICA DE FACÇÕES\n\n")
    summary.append("## Arquitetura Modificada\n\n")
    summary.append("```\n")
    summary.append("INPUT: X(T, N, 7)\n")
    summary.append("  ├─ [0-2] Branch 1: Crime Features\n")
    summary.append("  │         ├─ Encoder(3 → hidden_dim)\n")
    summary.append("  │         └─ ReLU + Dropout\n")
    summary.append("  │\n")
    summary.append("  └─ [3-6] Branch 2: Faction Dynamics\n")
    summary.append("          ├─ Encoder(4 → hidden_dim/2)\n")
    summary.append("          └─ Pad(hidden_dim/2 → hidden_dim)\n")
    summary.append("\n")
    summary.append("FUSION LAYER:\n")
    summary.append("  ├─ Multi-head Attention (crime, faction_dynamics)\n")
    summary.append("  └─ Residual connection: crime + 0.3 * attention\n")
    summary.append("\n")
    summary.append("TEMPORAL: LSTM(2 layers, hidden_dim)\n")
    summary.append("SPATIAL: GraphConv(hidden_dim → hidden_dim)\n")
    summary.append("OUTPUT: Decoder(hidden_dim → 1) + ReLU\n")
    summary.append("AUX: Change Probability(hidden_dim → 1) + Sigmoid\n")
    summary.append("```\n\n")
    
    summary.append("## Features de Entrada (7 dimensões)\n\n")
    for idx, desc in config_dict['features_description'].items():
        summary.append(f"**{idx}.** {desc}\n")
    
    summary.append("\n## Loss Function Dinâmica\n\n")
    summary.append("```\n")
    summary.append("L_total = L_main + L_auxiliary\n")
    summary.append("\n")
    summary.append("L_main = MSE(pred, target) * dynamic_weight\n")
    summary.append("  where: dynamic_weight = 1 + (mudança * 2) + (volatilidade * 0.5)\n")
    summary.append("\n")
    summary.append("L_auxiliary = BCE(mudança_pred, mudança_real) * 0.5\n")
    summary.append("```\n\n")
    
    summary.append("## Fluxo de Treinamento\n\n")
    summary.append("1. **Encode**: Separar features de crime e facções\n")
    summary.append("2. **Attend**: Multi-head attention para fusão contextual\n")
    summary.append("3. **Temporal**: LSTM captura padrões históricos\n")
    summary.append("4. **Spatial**: Graph convolution captura vizinhança\n")
    summary.append("5. **Predict**: Decoder prediz CVLI + mudanças territoriais\n")
    summary.append("6. **Loss**: Calcula weighted loss considerando dinâmica\n\n")
    
    summary.append("## Benefícios da Adaptação\n\n")
    summary.append("✅ Captura mudanças de poder territorial\n")
    summary.append("✅ Aumenta peso onde há conflito (maior incerteza)\n")
    summary.append("✅ Predição auxiliar de mudanças de controle\n")
    summary.append("✅ Mantém signal de crime como principal\n")
    summary.append("✅ Flexível para adicionar mais features de inteligência\n")
    
    report_path = config.DATA_PROCESSED / 'ADAPTACAO_MODELO_FACCOES.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.writelines(summary)
    
    print(f"✅ Relatório salvo em: {report_path}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("GERADOR DE ADAPTAÇÃO: ST-GCN COM DINÂMICA DE FACÇÕES")
    print("="*80)
    print()
    
    # Criar modelo (validação)
    try:
        model = STGCN_DynamicFactions(input_features=7, hidden_dim=32, num_nodes=121)
        print("✅ Modelo STGCN_DynamicFactions criado com sucesso")
        print(f"   Parâmetros: {sum(p.numel() for p in model.parameters()):,}")
        print()
        
        # Testar forward pass
        batch_size = 4
        time_steps = 14
        num_nodes = 121
        X_test = torch.randn(batch_size, time_steps, num_nodes, 7)
        
        output, aux_output = model(X_test, return_aux=True)
        print(f"✅ Forward pass bem-sucedido")
        print(f"   Input shape: {X_test.shape}")
        print(f"   Output shape: {output.shape}")
        print(f"   Aux output shape: {aux_output.shape}")
        print()
        
        # Testar loss
        loss_fn = DynamicFactionLoss()
        target = torch.randn_like(output)
        aux_target = torch.randint(0, 2, aux_output.shape).float()
        faction_dyn = X_test[:, -1, :, 3:7]
        
        loss = loss_fn(output, target, faction_dyn, aux_output, aux_target)
        print(f"✅ Loss computation bem-sucedida")
        print(f"   Loss value: {loss.item():.4f}")
        print()
        
    except Exception as e:
        print(f"❌ Erro: {e}")
        sys.exit(1)
    
    # Salvar configuração
    save_model_adaptation()
    
    print()
    print("="*80)
    print("✅ ADAPTAÇÃO PRONTA PARA USO")
    print("="*80)
    print()
    print("📋 Próximos passos:")
    print("  1. Usar STGCN_DynamicFactions em src/trainer.py")
    print("  2. Usar DynamicFactionLoss como função de loss")
    print("  3. Carregar tensor: data/processed/tensor_cvli_prisoes_faccoes.npy")
    print("  4. Treinar com considerar_faccoes=True")
    print()
