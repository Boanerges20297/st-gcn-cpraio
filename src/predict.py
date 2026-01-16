import torch
import pandas as pd
import numpy as np
import os
import config
from model import STGCN_Cpraio
from datetime import datetime, timedelta

def load_region_artifacts(region_name):
    """
    Carrega o 'Kit de Inteligência' específico da região:
    - Grafo (Dataset)
    - Estatísticas (Média/Desvio para desnormalizar)
    - Modelo Treinado (Cérebro)
    """
    print(f"[-] Carregando artefatos: {region_name}...")
    paths = config.ARTIFACTS[region_name]
    
    # 1. Dataset (Topologia e Calendário)
    if not paths['dataset'].exists():
        print(f"    [!] Dataset não encontrado. Pule esta região.")
        return None
    
    # weights_only=False para permitir carregar a estrutura completa do dicionário
    dataset = torch.load(paths['dataset'], weights_only=False)
    
    # 2. Estatísticas (Para trazer a previsão de volta à escala real)
    if not paths['stats'].exists():
        print(f"    [!] Estatísticas não encontradas. O modelo foi treinado?")
        return None
    stats = torch.load(paths['stats'], weights_only=False)
    
    # 3. Modelo
    if not paths['model'].exists():
        print(f"    [!] Modelo treinado não encontrado.")
        return None
        
    return dataset, stats, paths['model']

def predict_region(region_name):
    print(f"\n>>> INICIANDO PREDIÇÃO TÁTICA: {region_name}")
    
    artifacts = load_region_artifacts(region_name)
    if not artifacts:
        return

    data, stats, model_path = artifacts
    
    X_full = data['X']          # (Days, Nodes, Features)
    edge_index = data['edge_index']
    nodes = data['nodes']
    dates = data['dates']
    mean, std = stats['mean'], stats['std']
    
    # Hiperparâmetros
    window_size = config.HyperParams['window_size']
    target_window = config.HyperParams['target_window']
    hidden_dim = config.HyperParams['hidden_dim']
    
    # Validação de Histórico
    if len(X_full) < window_size:
        print("    [!] Histórico insuficiente para gerar previsão.")
        return

    # --- 1. Preparar a Janela de Entrada (Últimos 14 dias) ---
    last_window = X_full[-window_size:] # Shape: (14, Nodes, 1)
    
    # Normalizar (usando a mesma régua do treino)
    last_window_norm = (last_window - mean) / std
    
    # Adicionar dimensão de Batch: (1, 14, Nodes, 1)
    input_tensor = last_window_norm.unsqueeze(0)
    
    # --- 2. Inicializar e Carregar Modelo ---
    num_nodes = X_full.shape[1]
    num_features = X_full.shape[2]
    
    model = STGCN_Cpraio(
        num_nodes=num_nodes,
        in_channels=num_features,
        hidden_channels=hidden_dim,
        out_channels=num_features,
        dropout=0.0 # Sem dropout na inferência
    )
    
    try:
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.eval() # Modo de avaliação (congela camadas)
    except Exception as e:
        print(f"    [X] Erro ao carregar pesos do modelo: {e}")
        return

    # --- 3. Inferência ---
    print("    [-] Executando rede neural (ST-GCN)...")
    with torch.no_grad():
        prediction_norm = model(input_tensor, edge_index)
    
    # --- 4. Desnormalização e Interpretação ---
    # O modelo prevê a MÉDIA quinzenal normalizada. Precisamos trazer para a escala real.
    # Prediction shape: (1, Nodes, 1)
    
    # Remove batch e feature extra -> (Nodes,)
    pred_values = prediction_norm.squeeze().numpy()
    
    # Desnormalizar: Valor * Std + Mean
    mean_val = mean.item() if isinstance(mean, torch.Tensor) else mean
    std_val = std.item() if isinstance(std, torch.Tensor) else std
    
    pred_real = pred_values * std_val + mean_val
    
    # Limpeza: Não existe crime negativo, zerar valores < 0
    pred_final = np.maximum(pred_real, 0)
    
    # --- 5. Relógio Tático ---
    try:
        last_date = pd.to_datetime(dates[-1])
        start_pred = last_date + timedelta(days=1)
        end_pred = last_date + timedelta(days=target_window)
        periodo_str = f"{start_pred.strftime('%d/%m')} a {end_pred.strftime('%d/%m')}"
    except:
        periodo_str = "Próxima Quinzena"

    # --- 6. Gerar Relatório CSV ---
    df_result = pd.DataFrame({
        'bairro': nodes,
        'CVLI': pred_final
    })
    
    # Ordenar por Risco
    df_result = df_result.sort_values(by='CVLI', ascending=False)
    
    # Salvar
    output_path = config.ARTIFACTS[region_name]['prediction']
    df_result.to_csv(output_path, index=False)
    
    print(f"    [V] Previsão salva: {output_path}")
    print(f"    [i] Horizonte Tático: {periodo_str}")
    
    # Preview
    top_3 = df_result.head(3)
    print(f"    ⚠️  Top 3 Riscos Detectados:")
    for _, row in top_3.iterrows():
        print(f"       - {row['bairro']}: {row['CVLI']:.2f}")

def main():
    print("==============================================")
    print("   SISTEMA DE PREDIÇÃO CRIMINAL MULTI-ESCALA  ")
    print("==============================================")
    
    regions = ['CAPITAL', 'RMF', 'INTERIOR']
    
    for region in regions:
        try:
            predict_region(region)
        except Exception as e:
            print(f"[X] Falha crítica ao processar {region}: {e}")
            import traceback
            traceback.print_exc()
            
    print("\n[V] Ciclo de Inteligência Concluído.")

if __name__ == "__main__":
    main()