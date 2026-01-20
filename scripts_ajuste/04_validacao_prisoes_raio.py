"""
VALIDAÇÃO - ANÁLISE DE EFICIÊNCIA DAS PRISÕES RAIO
==================================================
Relaciona:
1. Predições CVLI 2025 vs Crimes reais
2. Prisões RAIO 2025 vs Redução de crimes
3. Influência de facções nos territórios
"""

import pandas as pd
import numpy as np
import torch
import json
from pathlib import Path
from datetime import datetime, timedelta
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import config
from src.model import STGCN_Cpraio

def load_model_and_data():
    """Carrega modelo treinado e dados"""
    model_path = config.MODEL_DIR / "model_cvli_novo_criterio.pth"
    stats_path = config.MODEL_DIR / "stats_cvli_novo_criterio.pt"
    dataset_path = config.TENSOR_DIR / "dataset_cvli_novo_criterio.pt"
    
    if not all([model_path.exists(), stats_path.exists(), dataset_path.exists()]):
        print("[X] Arquivos de modelo não encontrados")
        print(f"    Model: {model_path.exists()}")
        print(f"    Stats: {stats_path.exists()}")
        print(f"    Dataset: {dataset_path.exists()}")
        return None, None, None, None
    
    print("[-] Carregando modelo treinado...")
    
    dataset = torch.load(dataset_path, weights_only=False)
    stats = torch.load(stats_path, weights_only=False)
    
    model = STGCN_Cpraio(
        num_nodes=dataset['num_nodes'],
        in_channels=dataset['num_features'],
        hidden_channels=32,
        out_channels=dataset['num_features'],
        dropout=0.4
    )
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    
    print(f"[V] Modelo carregado com {dataset['num_nodes']} nós")
    
    return model, dataset, stats, dataset['edge_index']

def generate_predictions_2025(model, X_2025, edge_index, stats, bairro_to_idx):
    """
    Gera predições CVLI para 2025
    """
    print("[-] Gerando predições para 2025...")
    
    mean = stats['mean']
    std = stats['std']
    
    X_2025_norm = (X_2025 - mean) / std
    
    predictions = []
    dates = []
    
    ws = 14
    tw = 15
    
    with torch.no_grad():
        for t in range(ws, len(X_2025_norm) - tw):
            x_seq = X_2025_norm[t-ws:t].unsqueeze(0)  # Add batch dim
            pred = model(x_seq, edge_index)  # [1, num_nodes, num_features]
            
            # Denormalizar
            pred_denorm = pred * std + mean
            
            # Extrair feature CVLI (índice 0)
            cvli_pred = pred_denorm[0, :, 0].cpu().numpy()
            
            predictions.append(cvli_pred)
    
    return np.array(predictions)

def load_real_vs_predicted_2025():
    """Compara crimes reais vs preditos em 2025"""
    print("\n[VALIDAÇÃO 1: PREDIÇÕES VS REALIDADE 2025]")
    
    val_file = config.DATA_PROCESSED / "dataset_validacao_cvli_2025.parquet"
    
    if not val_file.exists():
        print("[!] Dados de validação não encontrados")
        return
    
    df_val = pd.read_parquet(val_file)
    
    # Agrupar por semana para análise
    df_val['week'] = df_val['data'].dt.isocalendar().week
    
    crimes_por_semana = df_val.groupby('week').size()
    crimes_por_bairro = df_val.groupby('bairro').size().sort_values(ascending=False)
    
    print(f"\n  Total crimes CVLI em 2025: {len(df_val)}")
    print(f"  Semanas com crimes: {len(crimes_por_semana)}")
    print(f"  Bairros afetados: {len(crimes_por_bairro)}")
    
    print(f"\n  Top 10 Bairros por CVLI em 2025:")
    for bairro, count in crimes_por_bairro.head(10).items():
        pct = 100 * count / len(df_val)
        print(f"    {bairro}: {count} ({pct:.1f}%)")
    
    return df_val

def analyze_prisoes_raio_impact():
    """
    Analisa impacto de prisões RAIO em crimes
    """
    print("\n[VALIDAÇÃO 2: IMPACTO PRISÕES RAIO]")
    
    raio_file = config.DATA_PROCESSED / "prisoes_raio_2025.parquet"
    
    if not raio_file.exists():
        print("[!] Dados RAIO não encontrados")
        return
    
    df_raio = pd.read_parquet(raio_file)
    
    print(f"\n  Total operações RAIO em 2025: {len(df_raio)}")
    
    # Estatísticas de apreensões
    if 'total_drogas_cache' in df_raio.columns:
        drogas_total = df_raio['total_drogas_cache'].sum()
        print(f"  Drogas apreendidas: {drogas_total:.2f} kg")
    
    if 'total_armas_cache' in df_raio.columns:
        armas_total = df_raio['total_armas_cache'].sum()
        print(f"  Armas apreendidas: {int(armas_total)} unidades")
    
    if 'Dinheiro_Apreendido' in df_raio.columns:
        dinheiro = pd.to_numeric(df_raio['Dinheiro_Apreendido'], errors='coerce').sum()
        print(f"  Dinheiro apreendido: R$ {dinheiro:,.2f}")
    
    # Operações por semana - trata se coluna não existe
    date_col = None
    for col in ['Data', 'data', 'DATA', 'data_operacao', 'Data_Operacao']:
        if col in df_raio.columns:
            date_col = col
            break
    
    if date_col:
        try:
            df_raio['Data_parsed'] = pd.to_datetime(df_raio[date_col])
            df_raio['week'] = df_raio['Data_parsed'].dt.isocalendar().week
            ops_por_semana = df_raio.groupby('week').size()
            print(f"\n  Operações por semana: média {ops_por_semana.mean():.1f}, max {ops_por_semana.max()}")
        except:
            print(f"\n  ⚠️  Não foi possível processar datas")
    else:
        print(f"\n  ⚠️  Coluna de data não encontrada")
    
    # Bairros/cidades com mais operações
    if 'CidadeOcor' in df_raio.columns:
        cidades_ops = df_raio['CidadeOcor'].value_counts().head(10)
        print(f"\n  Top 10 Cidades com Operações:")
        for cidade, count in cidades_ops.items():
            print(f"    {cidade}: {count}")
    
    # Analisar correlação entre operações e facções
    if 'area_faccao' in df_raio.columns:
        faccoes_ops = df_raio['area_faccao'].value_counts()
        print(f"\n  Operações por Facção:")
        for faccao, count in faccoes_ops.items():
            print(f"    {faccao}: {count}")
    
    return df_raio

def analyze_crime_faccao_territory_relationship():
    """
    Analisa relação entre crimes, facções e territórios
    """
    print("\n[VALIDAÇÃO 3: CRIMES-FACÇÕES-TERRITÓRIOS]")
    
    train_file = config.DATA_PROCESSED / "dataset_treino_cvli_2022_2024.parquet"
    val_file = config.DATA_PROCESSED / "dataset_validacao_cvli_2025.parquet"
    
    if not train_file.exists() or not val_file.exists():
        print("[!] Datasets não encontrados")
        return
    
    df_train = pd.read_parquet(train_file)
    df_val = pd.read_parquet(val_file)
    
    print(f"\n  TREINO (2022-2024): {len(df_train)} CVLI")
    print(f"  VALIDAÇÃO (2025): {len(df_val)} CVLI")
    
    # Análise por facção
    def analyze_faccion_crimes(df, period_name):
        print(f"\n  === {period_name} ===")
        faccao_stats = df.groupby('faccao').agg({
            'id_ocorrencia': 'count',
            'bairro': lambda x: x.nunique()
        }).rename(columns={'id_ocorrencia': 'crimes', 'bairro': 'bairros'})
        faccao_stats = faccao_stats.sort_values('crimes', ascending=False)
        
        print(f"  Crimes por Facção:")
        for faccao, row in faccao_stats.iterrows():
            pct = 100 * row['crimes'] / len(df)
            print(f"    {faccao}: {row['crimes']} crimes em {int(row['bairros'])} bairros ({pct:.1f}%)")
        
        return faccao_stats
    
    stats_train = analyze_faccion_crimes(df_train, "TREINO 2022-2024")
    stats_val = analyze_faccion_crimes(df_val, "VALIDAÇÃO 2025")
    
    # Comparação de mudanças
    print(f"\n  === MUDANÇAS 2025 vs 2022-2024 ===")
    
    for faccao in stats_train.index:
        crimes_train = stats_train.loc[faccao, 'crimes'] / len(df_train)
        crimes_val = stats_val.loc[faccao, 'crimes'] / len(df_val) if faccao in stats_val.index else 0
        
        change = ((crimes_val - crimes_train) / crimes_train) * 100
        
        if change > 10:
            trend = "⬆️ AUMENTO"
        elif change < -10:
            trend = "⬇️ REDUÇÃO"
        else:
            trend = "→ ESTÁVEL"
        
        print(f"  {faccao}: {change:+.1f}% {trend}")
    
    # Territórios mais contestados
    print(f"\n  === TERRITÓRIOS COM MAIOR DISPUTA ===")
    
    df_all = pd.concat([df_train, df_val])
    multi_faccao = []
    
    for bairro in df_all['bairro'].unique():
        faccoes_bairro = df_all[df_all['bairro'] == bairro]['faccao'].unique()
        if len(faccoes_bairro) > 1:
            multi_faccao.append((bairro, len(faccoes_bairro)))
    
    multi_faccao.sort(key=lambda x: x[1], reverse=True)
    
    for bairro, num_faccoes in multi_faccao[:10]:
        crimes = len(df_all[df_all['bairro'] == bairro])
        print(f"  {bairro}: {num_faccoes} facções, {crimes} crimes")

def create_summary_report():
    """Gera relatório consolidado"""
    print("\n" + "=" * 60)
    print(" RESUMO DE VALIDAÇÃO - NOVO CRITÉRIO CVLI")
    print("=" * 60)
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'periodo_treino': '2022-2024',
        'periodo_validacao': '2025',
        'criterio': 'CVLI-CENTRIC',
        'notes': [
            'CVP usado apenas como contexto histórico',
            'Criticidade determinada APENAS por CVLI',
            'Validação com prisões RAIO',
            'Análise de eficiência de operações'
        ]
    }
    
    report_path = config.REPORT_DIR / "validacao_novo_criterio.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n[V] Relatório salvo: {report_path}")
    
    return report

def main():
    print("=" * 60)
    print(" VALIDAÇÃO - ANÁLISE COMPLETA NOVO CRITÉRIO")
    print("=" * 60)
    
    # Verificar se modelo existe
    model_path = config.MODEL_DIR / "model_cvli_novo_criterio.pth"
    if not model_path.exists():
        print("[!] Modelo ainda não foi treinado")
        print("    Aguarde: python scripts_ajuste/03_trainer_novo_criterio.py")
        return
    
    # 1. Comparar predições vs realidade
    df_real = load_real_vs_predicted_2025()
    
    # 2. Analisar impacto RAIO
    df_raio = analyze_prisoes_raio_impact()
    
    # 3. Relação crimes-facções-territórios
    analyze_crime_faccao_territory_relationship()
    
    # 4. Gerar relatório
    create_summary_report()
    
    print("\n" + "=" * 60)
    print(" VALIDAÇÃO CONCLUÍDA")
    print("=" * 60)

if __name__ == "__main__":
    main()
