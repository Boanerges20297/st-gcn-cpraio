import sys
import os
import torch
import pandas as pd
import numpy as np
from datetime import timedelta

# --- BLINDAGEM DE IMPORTS ---
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.dirname(current_dir))

import config
try:
    from model import STGCN_Cpraio
except ImportError:
    from src.model import STGCN_Cpraio

def load_artifacts(region_name):
    """Carrega o Modelo Treinado e os Dados da Região."""
    paths = config.ARTIFACTS[region_name]
    
    # 1. Checagens
    if not paths['model'].exists():
        print(f"    [!] Modelo não encontrado: {paths['model']}")
        return None
    if not paths['dataset'].exists():
        print(f"    [!] Dataset não encontrado.")
        return None
    if not paths['stats'].exists():
        print(f"    [!] Estatísticas (stats.pt) não encontradas. Rode o trainer.py.")
        return None

    # 2. Carregar
    # weights_only=False para carregar dicionários complexos
    dataset = torch.load(paths['dataset'], weights_only=False)
    stats = torch.load(paths['stats'], weights_only=False)
    
    return dataset, stats, paths['model']

def predict_region(region_name):
    print(f"\n>>> PREVENDO FUTURO: {region_name}")
    
    loaded = load_artifacts(region_name)
    if not loaded: return

    data, stats, model_path = loaded
    X_full = data['X']          # Histórico completo
    edge_index = data['edge_index']
    nodes = data['nodes']       # Lista de Bairros/Cidades
    features = data.get('features', ['CVLI'])
    
    # 3. Preparar Input (Última Janela)
    # Determinar janela com base no alvo previsto (CVLI ou CVP)
    features = data.get('features', ['CVLI'])
    if 'CVLI' in features:
        window_size = int(config.HyperParams.get('window_size_cvli', config.HyperParams.get('window_size', 180)))
    elif 'CVP' in features:
        window_size = int(config.HyperParams.get('window_size_cvp', config.HyperParams.get('window_size', 30)))
    else:
        window_size = int(config.HyperParams.get('window_size', 14))

    if len(X_full) < window_size:
        print(f"    [!] Histórico insuficiente. Necessário {window_size} registros, disponível {len(X_full)}")
        return

    # Pega os últimos N dias conforme janela configurada
    last_window = X_full[-window_size:]
    
    # Normalizar (Usando a média/desvio salvos no treino)
    mean = stats['mean']
    std = stats['std']
    last_window_norm = (last_window - mean) / std
    
    # Formato de Batch: (1, 14, Nodes, 1)
    input_tensor = last_window_norm.unsqueeze(0)

    # 4. Inicializar e Carregar Modelo
    num_nodes = X_full.shape[1]
    num_features = X_full.shape[2]
    
    model = STGCN_Cpraio(
        num_nodes=num_nodes,
        in_channels=num_features,
        hidden_channels=config.HyperParams['hidden_dim'],
        out_channels=num_features,
        dropout=0.0
    )
    
    try:
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.eval()
    except Exception as e:
        print(f"    [X] Erro ao carregar pesos: {e}")
        return

    # 5. Inferência
    with torch.no_grad():
        # Saída: (1, Nodes, 1) -> Média Quinzenal Normalizada
        out_norm = model(input_tensor, edge_index)
        try:
            print(f"    [DEBUG] input_tensor.shape={tuple(input_tensor.shape)}, edge_index.shape={tuple(edge_index.shape)}")
            print(f"    [DEBUG] out_norm.shape={tuple(out_norm.shape)}")
            print(f"    [DEBUG] mean.shape={getattr(mean,'shape',None)}, std.shape={getattr(std,'shape',None)}")
        except Exception:
            pass
        
    # 6. Desnormalizar
    out_real = out_norm.squeeze() * std + mean
    
    # Zerar negativos (não existe crime negativo) e converter para numpy
    # Se houver múltiplas features, escolher a previsão da feature CVLI (ou primeira se não existir)
    try:
        feature_idx = features.index('CVLI')
    except Exception:
        feature_idx = 0

    # out_real can have shapes: (Nodes,), (Nodes,Features), or (Batch,Nodes,Features)
    if out_real.dim() == 1:
        pred_tensor = out_real
    elif out_real.dim() == 2:
        # (Nodes, Features)
        pred_tensor = out_real[:, feature_idx]
    elif out_real.dim() == 3:
        # (Batch, Nodes, Features)
        pred_tensor = out_real[0, :, feature_idx]
    else:
        # Fallback: try to flatten appropriately
        pred_tensor = out_real.reshape(-1)

    try:
        print(f"    [DEBUG] pred_tensor.shape={getattr(pred_tensor,'shape',None)}")
    except Exception:
        pass

    pred_values = torch.relu(pred_tensor).numpy().flatten()

    # Safety: garantir que o vetor de previsões tenha o mesmo comprimento de `nodes`
    if pred_values.size != len(nodes):
        try:
            if pred_values.size % len(nodes) == 0:
                pred_values = pred_values.reshape(len(nodes), -1)[:, 0]
            else:
                pred_values = pred_values[:len(nodes)]
        except Exception:
            pred_values = np.resize(pred_values, (len(nodes),))

    # 7. Gerar Relatório
    # Debug shapes if mismatch
    try:
        import math
        print(f"    [DEBUG] nodes={len(nodes)}, pred_values={getattr(pred_values,'size', None)}")
    except Exception:
        pass

    df_result = pd.DataFrame({
        'local': nodes,
        'risco_previsto': pred_values,
        'regiao': region_name
    })
    
    # Ordenar por risco
    df_result = df_result.sort_values(by='risco_previsto', ascending=False)
    
    # Salvar
    out_csv = config.ARTIFACTS[region_name]['prediction']
    df_result.to_csv(out_csv, index=False)
    
    print(f"    [V] Previsão salva: {out_csv}")
    print("    ⚠️  TOP 3 ALVOS:")
    for i, row in df_result.head(3).iterrows():
        print(f"       {i+1}. {row['local']}: {row['risco_previsto']:.2f}")

def main():
    print("==============================================")
    print("      SISTEMA DE PREDIÇÃO TÁTICA (IA)         ")
    print("==============================================")
    
    for region in ['CAPITAL', 'RMF', 'INTERIOR']:
        try:
            predict_region(region)
        except Exception as e:
            print(f"    [X] Falha em {region}: {e}")

if __name__ == "__main__":
    main()