"""
VALIDAÇÃO DE EFETIVIDADE - RAIO 2025
====================================
Compara predições do modelo 180d com prisões RAIO 2025 por bairro/mês
para atestar se as equipes atuaram nos locais críticos.
"""

import pandas as pd
import numpy as np
import torch
import json
from pathlib import Path
from datetime import datetime
import sys
import warnings

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import config

def load_raio_2025_data():
    """Carrega prisões RAIO 2025"""
    print("\n[CARREGANDO DADOS RAIO 2025]")
    
    # Tenta múltiplas localizações
    possible_paths = [
        config.DATA_PROCESSED / "prisoes_raio_2025.parquet",
        config.DATA_RAW / "prisoes_raio_2025.parquet",
    ]
    
    df_raio = None
    for path in possible_paths:
        if path.exists():
            print(f"[OK] Encontrado: {path.name}")
            try:
                raw = pd.read_parquet(path)
                
                # Formato especial: 'data' coluna contém array de dicts
                if 'data' in raw.columns:
                    records = []
                    for data_array in raw['data']:
                        if data_array is not None:
                            # É um numpy array de dicts
                            if hasattr(data_array, '__iter__'):
                                for item in data_array:
                                    if isinstance(item, dict):
                                        records.append(item)
                    
                    if records:
                        df_raio = pd.DataFrame(records)
                        break
            except Exception as e:
                print(f"    Erro ao processar {path.name}: {e}")
                continue
    
    if df_raio is None or len(df_raio) == 0:
        print("[ERROR] Nenhum arquivo RAIO 2025 encontrado")
        return None
    
    print(f"[OK] {len(df_raio)} registros RAIO carregados")
    
    # Padroniza colunas - procura por Data/BairroOcor
    if 'Data' in df_raio.columns:
        df_raio['data_operacao'] = pd.to_datetime(df_raio['Data'], errors='coerce')
    else:
        print("[WARNING] Coluna 'Data' não encontrada")
        return None
    
    if 'BairroOcor' in df_raio.columns:
        df_raio['bairro'] = df_raio['BairroOcor'].fillna('DESCONHECIDO').str.upper()
    else:
        print("[WARNING] Coluna de bairro não encontrada")
        return None
    
    # Filtra apenas 2025
    df_raio = df_raio[(df_raio['data_operacao'].dt.year == 2025)]
    print(f"[OK] Filtrado para 2025: {len(df_raio)} operações")
    
    return df_raio[['data_operacao', 'bairro']].copy()

def load_crimes_2025():
    """Carrega crimes CVLI 2025 reais"""
    print("\n[CARREGANDO DADOS DE CRIMES CVLI 2025]")
    
    possible_paths = [
        config.DATA_PROCESSED / "ocorrencias_2025_completo.parquet",
        config.DATA_PROCESSED / "crimes_2025.parquet",
        config.DATA_RAW / "ocorrencias_2025.parquet",
    ]
    
    df_crimes = None
    for path in possible_paths:
        if path.exists():
            print(f"[OK] Encontrado: {path.name}")
            df_crimes = pd.read_parquet(path)
            break
    
    if df_crimes is None:
        print("[WARNING] Dados CVLI 2025 não encontrados, continuando sem validação de redução")
        return None
    
    print(f"[OK] {len(df_crimes)} crimes carregados")
    
    # Filtra apenas CVLI
    if 'tipo_crime' in df_crimes.columns:
        df_crimes = df_crimes[df_crimes['tipo_crime'] == 'CVLI']
    
    return df_crimes

def load_criticidade_timeline():
    """Carrega timeline de criticidade do modelo 180d"""
    print("\n[CARREGANDO CRITICIDADE 180D]")
    
    dataset_path = config.TENSOR_DIR / "dataset_criticidade_janela180d.pt"
    metadata_path = config.TENSOR_DIR / "metadata_janela180d.json"
    
    if not dataset_path.exists():
        print(f"[ERROR] {dataset_path} não encontrado")
        return None
    
    if not metadata_path.exists():
        print(f"[ERROR] {metadata_path} não encontrado")
        return None
    
    # Carrega tensor
    criticidade_tensor = torch.load(dataset_path, weights_only=False)  # (dias, nós, features)
    print(f"[OK] Criticidade carregada: {criticidade_tensor.shape}")
    
    # Carrega metadata
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    print(f"[OK] Metadata carregado")
    
    return {
        'criticidade': criticidade_tensor,  # (1461, 319, 1)
        'bairro_mapping': metadata.get('bairro_mapping', {}),
        'dates': metadata.get('dates', [])
    }

def generate_monthly_predictions(criticidade_data):
    """Gera predições médias de criticidade por bairro/mês"""
    print("\n[GERANDO PREDIÇÕES MENSAIS]")
    
    criticidade = criticidade_data['criticidade']  # (dias, nós, features)
    criticidade = criticidade.squeeze(-1)  # Remove dimensão de features -> (dias, nós)
    
    bairro_mapping = criticidade_data['bairro_mapping']
    
    # Inverte mapping: node_idx -> bairro_nome
    idx_to_bairro = {v: k for k, v in bairro_mapping.items()}
    
    # Timeline: 2022-01-01 até 2025-12-31 (1461 dias)
    start_date = datetime(2022, 1, 1)
    dates = pd.date_range(start=start_date, periods=criticidade.shape[0], freq='D')
    
    # Filtra apenas 2025
    mask_2025 = (dates.year == 2025)
    idx_2025 = np.where(mask_2025)[0]
    
    print(f"[OK] Período 2025: dias {idx_2025[0]} a {idx_2025[-1]}")
    
    monthly_predictions = []
    
    for month in range(1, 13):
        dates_2025 = dates[mask_2025]
        mask_month = (dates_2025.month == month)
        month_idx = idx_2025[mask_month]
        
        if len(month_idx) == 0:
            continue
        
        # Média de criticidade para cada bairro neste mês
        month_criticidade = criticidade[month_idx].mean(dim=0)  # (nós,)
        
        for bairro_idx in range(len(idx_to_bairro)):
            bairro_nome = idx_to_bairro.get(bairro_idx, f"BAIRRO_{bairro_idx}")
            
            monthly_predictions.append({
                'ano_mes': f"2025-{month:02d}",
                'bairro': bairro_nome,
                'criticidade_predita': float(month_criticidade[bairro_idx])
            })
    
    df_pred = pd.DataFrame(monthly_predictions)
    print(f"[OK] {len(df_pred)} predições geradas (bairros × meses)")
    
    return df_pred

def analyze_effectiveness(df_predictions, df_raio):
    """Analisa efetividade: comparação predições vs prisões"""
    print("\n[ANALISANDO EFETIVIDADE]")
    
    # Adiciona ano_mes aos dados RAIO
    df_raio['ano_mes'] = df_raio['data_operacao'].dt.strftime('%Y-%m')
    
    # Conta prisões por bairro/mês
    df_prisoes_count = df_raio.groupby(['ano_mes', 'bairro']).size().reset_index(name='num_prisoes')
    
    # Merge com predições
    df_comparison = pd.merge(
        df_predictions,
        df_prisoes_count,
        on=['ano_mes', 'bairro'],
        how='left'
    )
    df_comparison['num_prisoes'] = df_comparison['num_prisoes'].fillna(0)
    
    print(f"[OK] {len(df_comparison)} combinações bairro/mês analisadas")
    
    # Correlação geral
    correlation = df_comparison['criticidade_predita'].corr(df_comparison['num_prisoes'])
    print(f"\n[CORRELAÇÃO GERAL]")
    print(f"  Criticidade Predita vs Prisões: {correlation:.4f}")
    
    if correlation > 0.5:
        print("  [OK] FORTE correlação - modelo prediz corretamente os locais de ação")
    elif correlation > 0.3:
        print("  [OK] CORRELAÇÃO MODERADA - modelo tem alguma validade")
    else:
        print("  [WARNING] Correlação fraca - modelo pode não estar alinhado com atuações")
    
    # Top 10 bairros por criticidade predita vs prisões reais
    print(f"\n[TOP 10 BAIRROS - CRITICIDADE PREDITA vs PRISÕES 2025]")
    
    df_bairro_stats = df_comparison.groupby('bairro').agg({
        'criticidade_predita': 'mean',
        'num_prisoes': 'sum'
    }).reset_index()
    df_bairro_stats = df_bairro_stats.sort_values('criticidade_predita', ascending=False).head(10)
    
    print(f"\n{'Ranking':<8} {'Bairro':<25} {'Criticidade':<12} {'Prisões':<10} {'Alinhamento':<15}")
    print("-" * 70)
    
    for idx, row in df_bairro_stats.iterrows():
        ranking = idx + 1
        bairro = row['bairro'][:24]
        criticidade = row['criticidade_predita']
        prisoes = int(row['num_prisoes'])
        
        # Alinhamento: esperado ~criticidade * 100 prisões (normalizado)
        max_prisoes = df_bairro_stats['num_prisoes'].max()
        criticidade_norm = criticidade * max_prisoes
        
        diff_pct = abs(prisoes - criticidade_norm) / max(criticidade_norm, 1) * 100 if criticidade_norm > 0 else 0
        
        if diff_pct < 20:
            alinhamento = "[OK] ALINHADO"
        elif diff_pct < 50:
            alinhamento = "[PARCIAL]"
        else:
            alinhamento = "[DESVIO]"
        
        print(f"{ranking:<8} {bairro:<25} {criticidade:<12.4f} {prisoes:<10} {alinhamento:<15}")
    
    # Análise por mês
    print(f"\n[ANÁLISE POR MÊS DE 2025]")
    
    df_monthly = df_comparison.groupby('ano_mes').agg({
        'criticidade_predita': 'mean',
        'num_prisoes': 'sum'
    }).reset_index()
    df_monthly = df_monthly.sort_values('ano_mes')
    
    print(f"\n{'Mês':<12} {'Crit. Média':<15} {'Prisões':<10}")
    print("-" * 37)
    
    for _, row in df_monthly.iterrows():
        print(f"{row['ano_mes']:<12} {row['criticidade_predita']:<15.4f} {int(row['num_prisoes']):<10}")
    
    # Aéreascom zero prisões apesar de alta criticidade
    print(f"\n[POSSÍVEIS LACUNAS - Alta Criticidade SEM PRISÕES]")
    
    high_crit_no_action = df_comparison[
        (df_comparison['criticidade_predita'] > df_comparison['criticidade_predita'].quantile(0.75)) &
        (df_comparison['num_prisoes'] == 0)
    ]
    
    if len(high_crit_no_action) > 0:
        print(f"  Encontrados {len(high_crit_no_action)} casos")
        top_gaps = high_crit_no_action.nlargest(10, 'criticidade_predita')[['ano_mes', 'bairro', 'criticidade_predita']]
        for _, row in top_gaps.iterrows():
            print(f"    {row['ano_mes']} - {row['bairro']:<30} Criticidade: {row['criticidade_predita']:.4f}")
    else:
        print("  Nenhum - bom alinhamento entre modelo e ações")
    
    # Áreas com muitas prisões apesar de baixa criticidade
    print(f"\n[POSSÍVEIS SOBRE-ALOCAÇÕES - Baixa Criticidade COM MUITAS PRISÕES]")
    
    low_crit_high_action = df_comparison[
        (df_comparison['criticidade_predita'] < df_comparison['criticidade_predita'].quantile(0.25)) &
        (df_comparison['num_prisoes'] > df_comparison['num_prisoes'].quantile(0.75))
    ]
    
    if len(low_crit_high_action) > 0:
        print(f"  Encontrados {len(low_crit_high_action)} casos")
        top_over = low_crit_high_action.nlargest(10, 'num_prisoes')[['ano_mes', 'bairro', 'criticidade_predita', 'num_prisoes']]
        for _, row in top_over.iterrows():
            print(f"    {row['ano_mes']} - {row['bairro']:<30} Crit: {row['criticidade_predita']:.4f} | Prisões: {int(row['num_prisoes'])}")
    else:
        print("  Nenhum - recursos não super-alocados")
    
    return {
        'correlation': correlation,
        'comparison': df_comparison,
        'bairro_stats': df_bairro_stats,
        'monthly': df_monthly
    }

def main():
    print("=" * 80)
    print("VALIDAÇÃO DE EFETIVIDADE - PRISÕES RAIO 2025 vs PREDIÇÕES 180d")
    print("=" * 80)
    
    # Carrega dados
    df_raio = load_raio_2025_data()
    if df_raio is None:
        print("\n[ERROR] Falha ao carregar dados RAIO")
        return
    
    criticidade_data = load_criticidade_timeline()
    if criticidade_data is None:
        print("\n[ERROR] Falha ao carregar criticidade")
        return
    
    # Gera predições
    df_predictions = generate_monthly_predictions(criticidade_data)
    
    # Analisa efetividade
    results = analyze_effectiveness(df_predictions, df_raio)
    
    # Salva relatório
    report_path = config.OUTPUT_DIR / "relatorio_efetividade_raio_2025.json"
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'correlacao_geral': float(results['correlation']),
        'total_comparacoes': len(results['comparison']),
        'total_prisoes_2025': int(results['comparison']['num_prisoes'].sum()),
        'recomendacoes': [
            "Correlação > 0.5: Modelo validado para alocação de recursos",
            "Correlação 0.3-0.5: Modelo tem validade mas ajustes necessários",
            "Correlação < 0.3: Modelo não alinhado com operações reais"
        ]
    }
    
    import json
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n[OK] Relatório salvo: {report_path}")
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
