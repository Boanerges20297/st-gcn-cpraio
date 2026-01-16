import torch
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from src import config  # Importa as configurações centralizadas
from src.model import STGCN_Cpraio

def load_artifacts():
    print("[-] Carregando artefatos do sistema...")
    
    # 1. Carregar Dataset (para pegar a topologia, nomes e DATAS)
    if not config.DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset não encontrado em {config.DATASET_PATH}. Rode main.py --build primeiro.")

    dataset = torch.load(config.DATASET_PATH)
    
    edge_index = dataset['edge_index']
    nodes = dataset['nodes']
    features = dataset['features']
    dates = dataset['dates'] # CRÍTICO: Recuperar o calendário histórico
    X_full = dataset['X']    # (Days, Nodes, Features)
    
    # 2. Carregar Estatísticas de Normalização
    if not config.STATS_PATH.exists():
        raise FileNotFoundError(f"Estatísticas não encontradas em {config.STATS_PATH}. Rode main.py --train primeiro.")
        
    stats = torch.load(config.STATS_PATH)
    mean, std = stats['mean'], stats['std']
    
    # 3. Preparar Modelo
    num_nodes = X_full.shape[1]
    num_features = X_full.shape[2]
    
    model = STGCN_Cpraio(
        num_nodes=num_nodes, 
        in_channels=num_features, 
        hidden_channels=config.HyperParams['hidden_dim'], 
        out_channels=num_features
    )
    
    # Carregar Pesos Treinados
    if not config.MODEL_PATH.exists():
        raise FileNotFoundError(f"Modelo não encontrado em {config.MODEL_PATH}. Rode main.py --train primeiro.")

    model.load_state_dict(torch.load(config.MODEL_PATH))
    model.eval() # Modo de avaliação (desliga dropout)
    
    return model, X_full, edge_index, mean, std, nodes, features, dates

def predict_future(model, X_full, edge_index, mean, std, dates):
    """
    Pega a janela mais recente de dados e prevê o próximo ciclo quinzenal.
    Calcula as datas exatas de início e fim da previsão.
    """
    window_size = config.HyperParams['window_size']
    target_window = config.HyperParams['target_window']

    # Pegar os últimos dias do histórico (Input da IA)
    last_window = X_full[-window_size:]
    
    # --- CÁLCULO DO RELÓGIO TÁTICO ---
    try:
        # A última data registrada no dataset (ex: '2025-01-14')
        last_data_str = str(dates[-1])
        # Tenta converter. Aceita YYYY-MM-DD ou DD/MM/YYYY
        if "-" in last_data_str:
            last_date_obj = datetime.strptime(last_data_str.split(" ")[0], "%Y-%m-%d")
        else:
            last_date_obj = datetime.strptime(last_data_str.split(" ")[0], "%d/%m/%Y")
            
    except Exception as e:
        print(f"[!] AVISO: Erro ao ler datas do dataset ({e}). Usando data atual do sistema.")
        last_date_obj = datetime.now()

    # O horizonte começa no dia seguinte ao último dado
    pred_start = last_date_obj + timedelta(days=1)
    pred_end = last_date_obj + timedelta(days=target_window)
    
    print("\n" + "!"*60)
    print(f"[RELÓGIO TÁTICO] Base de Dados atualizada até: {last_date_obj.strftime('%d/%m/%Y')}")
    print(f"[HORIZONTE] Gerando previsão para a QUINZENA: {pred_start.strftime('%d/%m/%Y')} a {pred_end.strftime('%d/%m/%Y')}")
    print("!"*60 + "\n")
    
    # --- INFERÊNCIA ---
    # Normalizar usando as estatísticas do treino
    last_window_norm = (last_window - mean) / std
    
    # Adicionar dimensão de Batch (PyTorch espera Batch, Time, Nodes, Features)
    input_tensor = torch.FloatTensor(last_window_norm).unsqueeze(0) 
    
    print("[-] Executando inferência na ST-GCN...")
    with torch.no_grad():
        # Output: (1, Nodes, Features)
        prediction_norm = model(input_tensor, edge_index)
        
    # --- DESNORMALIZAÇÃO ---
    # Volta para a escala real de crimes
    mean_sq = mean.squeeze(0)
    std_sq = std.squeeze(0)
    
    prediction_real = prediction_norm * std_sq + mean_sq
    
    # Remover negativos (ReLU) e remover dimensão de batch
    prediction_final = torch.relu(prediction_real.squeeze(0)) 
    
    return prediction_final.numpy(), pred_start.strftime('%d/%m/%Y'), pred_end.strftime('%d/%m/%Y')

def generate_tactical_report(prediction, nodes, features, start_date, end_date):
    print("[-] Gerando Relatório Tático CSV...")
    
    # Converter para DataFrame
    df_pred = pd.DataFrame(prediction, columns=features)
    df_pred['bairro'] = nodes
    
    # Identificar coluna alvo (CVLI)
    target_col = 'CVLI' if 'CVLI' in features else features[0]
    
    # Ordenar por Risco
    df_rank = df_pred.sort_values(by=target_col, ascending=False)
    
    # Salvar
    if not config.PREDICTION_CSV.parent.exists():
        config.PREDICTION_CSV.parent.mkdir(parents=True, exist_ok=True)
        
    df_rank.to_csv(config.PREDICTION_CSV, index=False)
    
    print(f"\n[ ALERTA TÁTICO - TOP 5 BAIRROS PARA {target_col} ]")
    print(f"Período: {start_date} até {end_date}")
    print("-" * 40)
    # Formatação bonita para o terminal
    print(df_rank[['bairro', target_col]].head(5).to_string(index=False))
    print("-" * 40)
    print(f"[V] Relatório completo salvo em: {config.PREDICTION_CSV}")

def main():
    try:
        # Carrega tudo, incluindo o calendário (dates)
        model, X_full, edge_index, mean, std, nodes, features, dates = load_artifacts()
        
        # Faz a predição temporalmente consciente
        prediction, start_date, end_date = predict_future(model, X_full, edge_index, mean, std, dates)
        
        # Gera o relatório
        generate_tactical_report(prediction, nodes, features, start_date, end_date)
        
    except Exception as e:
        print(f"[X] Erro na inferência: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()