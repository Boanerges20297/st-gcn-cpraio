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
    
    # 3. Preparar Input (Última Janela)
    window_size = config.HyperParams['window_size']
    if len(X_full) < window_size:
        print("    [!] Histórico insuficiente.")
        return

    # Pega os últimos 14 dias
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
        
    # 6. Desnormalizar
    out_real = out_norm.squeeze() * std + mean
    
    # Zerar negativos (não existe crime negativo) e converter para numpy
    pred_values = torch.relu(out_real).numpy().flatten() # (Nodes,)

    # 7. Gerar Relatório
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