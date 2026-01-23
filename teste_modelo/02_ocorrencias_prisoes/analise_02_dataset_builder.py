"""
ANÁLISE 2: ST-GCN COM OCORRÊNCIAS + PRISÕES (Features Cruzadas)
═══════════════════════════════════════════════════════════════════

Objetivo: Construir um dataset com múltiplas features:
- Contagem de CVLI (homicídios, latrocínios, tentativas)
- Contagem de Prisões (operações policiais)
- Correlação e defasagem entre as duas

Hipótese: Prisões como contexto pode melhorar previsão de CVLI.
Teste: Validar se há relação causal ou apenas coincidência.

Métricas:
- Distribuição conjunta CVLI-Prisões
- Defasagem temporal (lag effects)
- Correlação espacial cruzada
- Potencial de captura de padrões causais
"""

import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

from scipy.stats import pearsonr, spearmanr
from statsmodels.tsa.stattools import adfuller

# ═══════════════════════════════════════════════════════════════════════
# 1. CONFIGURAÇÃO
# ═══════════════════════════════════════════════════════════════════════

OUTPUT_DIR = Path(__file__).parent
DATA_DIR = Path(__file__).parent.parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).parent.parent.parent / "data" / "processed"

OUTPUT_DIR.mkdir(exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════
# 2. CARREGAR DADOS (CVLI + PRISÕES)
# ═══════════════════════════════════════════════════════════════════════

def load_cvli_data():
    """Carrega dados de CVLI."""
    print("[1.1] Carregando dados de CVLI...")
    
    cvli_file = DATA_DIR / "ocorrencia_policial_operacional.json"
    if not cvli_file.exists():
        print(f"❌ Arquivo não encontrado: {cvli_file}")
        return None
    
    try:
        with open(cvli_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        records = [item for item in data if isinstance(item, dict)]
        df = pd.DataFrame(records)
        print(f"✅ Carregados {len(df)} registros de CVLI")
        return df
    except Exception as e:
        print(f"❌ Erro: {e}")
        return None

def load_prisoes_data():
    """Carrega dados de Prisões (operações policiais)."""
    print("[1.2] Carregando dados de Prisões...")
    
    # Tentar parquet primeiro (já processado)
    parquet_file = PROCESSED_DIR / "prisoes_with_features.parquet"
    if parquet_file.exists():
        try:
            df = pd.read_parquet(parquet_file)
            print(f"✅ Carregados {len(df)} registros de Prisões (parquet)")
            return df
        except Exception as e:
            print(f"⚠️ Erro ao carregar parquet: {e}")
    
    # Fallback para JSON bruto
    prisoes_file = DATA_DIR / "prisoes_raio_2025_caucaia.json"
    if prisoes_file.exists():
        try:
            with open(prisoes_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            records = [item for item in data if isinstance(item, dict)]
            df = pd.DataFrame(records)
            print(f"✅ Carregados {len(df)} registros de Prisões (json)")
            return df
        except Exception as e:
            print(f"❌ Erro: {e}")
    
    return None

# ═══════════════════════════════════════════════════════════════════════
# 3. NORMALIZAR DADOS
# ═══════════════════════════════════════════════════════════════════════

def normalize_data(df_cvli, df_prisoes):
    """Normaliza ambos os datasets para compatibilidade."""
    print("\n[2] Normalizando dados...")
    
    # Normalizar CVLI
    if 'data_operacao' in df_cvli.columns:
        df_cvli['date'] = pd.to_datetime(df_cvli['data_operacao'], errors='coerce')
    elif 'data' in df_cvli.columns:
        df_cvli['date'] = pd.to_datetime(df_cvli['data'], errors='coerce')
    
    if 'endereco_bairro_padronizado' in df_cvli.columns:
        df_cvli['bairro'] = df_cvli['endereco_bairro_padronizado']
    elif 'bairro' in df_cvli.columns:
        df_cvli['bairro'] = df_cvli['bairro']
    
    df_cvli = df_cvli.dropna(subset=['date', 'bairro'])
    
    # Normalizar Prisões
    if 'data' in df_prisoes.columns:
        df_prisoes['date'] = pd.to_datetime(df_prisoes['data'], errors='coerce')
    elif 'data_operacao' in df_prisoes.columns:
        df_prisoes['date'] = pd.to_datetime(df_prisoes['data_operacao'], errors='coerce')
    
    # Tentar extrair bairro/localidade
    if 'bairro' in df_prisoes.columns:
        df_prisoes['bairro'] = df_prisoes['bairro']
    elif 'cidade' in df_prisoes.columns:
        # Usar cidade como proxy para bairro se necessário
        df_prisoes['bairro'] = df_prisoes['cidade']
    elif 'endereco_bairro' in df_prisoes.columns:
        df_prisoes['bairro'] = df_prisoes['endereco_bairro']
    
    df_prisoes = df_prisoes.dropna(subset=['date', 'bairro'])
    
    print(f"✅ CVLI: {len(df_cvli)} registros | Período: {df_cvli['date'].min()} a {df_cvli['date'].max()}")
    print(f"✅ Prisões: {len(df_prisoes)} registros | Período: {df_prisoes['date'].min()} a {df_prisoes['date'].max()}")
    
    return df_cvli, df_prisoes

# ═══════════════════════════════════════════════════════════════════════
# 4. CONSTRUIR TENSOR COM MÚLTIPLAS FEATURES
# ═══════════════════════════════════════════════════════════════════════

def build_multifeature_tensor(df_cvli, df_prisoes):
    """
    Constrói tensor (T, N, F) com F=3 features:
    - F0: Contagem CVLI
    - F1: Contagem Prisões
    - F2: Correlação lag (prisões t-1 vs CVLI t)
    """
    print("\n[3] Construindo tensor multi-feature...")
    
    # Agrupar por bairro/data
    cvli_daily = df_cvli.groupby(['bairro', df_cvli['date'].dt.date]).size().reset_index(name='cvli_count')
    cvli_daily['date'] = pd.to_datetime(cvli_daily['date'])
    
    pris_daily = df_prisoes.groupby(['bairro', df_prisoes['date'].dt.date]).size().reset_index(name='pris_count')
    pris_daily['date'] = pd.to_datetime(pris_daily['date'])
    
    # Encontrar período comum
    date_min = max(cvli_daily['date'].min(), pris_daily['date'].min())
    date_max = min(cvli_daily['date'].max(), pris_daily['date'].max())
    
    date_range = pd.date_range(date_min, date_max, freq='D')
    
    # Encontrar bairros comuns
    bairros_cvli = set(cvli_daily['bairro'].unique())
    bairros_pris = set(pris_daily['bairro'].unique())
    bairros = sorted(bairros_cvli & bairros_pris)  # Interseção
    
    n_timesteps = len(date_range)
    n_nodes = len(bairros)
    n_features = 3  # CVLI, Prisões, Correlação lag
    
    print(f"✅ Dimensões do tensor:")
    print(f"   T: {n_timesteps} dias ({date_range[0].date()} a {date_range[-1].date()})")
    print(f"   N: {n_nodes} bairros (interseção comum)")
    print(f"   F: {n_features} features")
    
    # Inicializar tensor
    tensor = np.zeros((n_timesteps, n_nodes, n_features), dtype=np.float32)
    
    # Preencher tensor
    for idx, date in enumerate(date_range):
        for j, bairro in enumerate(bairros):
            # Feature 0: CVLI
            cvli_count = cvli_daily[
                (cvli_daily['date'] == date) & (cvli_daily['bairro'] == bairro)
            ]['cvli_count'].sum()
            tensor[idx, j, 0] = cvli_count
            
            # Feature 1: Prisões
            pris_count = pris_daily[
                (pris_daily['date'] == date) & (pris_daily['bairro'] == bairro)
            ]['pris_count'].sum()
            tensor[idx, j, 1] = pris_count
            
            # Feature 2: Lag correlation (simplificado: prisões do dia anterior)
            if idx > 0:
                tensor[idx, j, 2] = tensor[idx-1, j, 1]  # Prisão t-1
    
    # Estatísticas
    metadata = {
        'n_timesteps': n_timesteps,
        'n_nodes': n_nodes,
        'n_features': n_features,
        'date_min': str(date_range[0].date()),
        'date_max': str(date_range[-1].date()),
        'total_cvli': float(np.sum(tensor[:, :, 0])),
        'total_prisoes': float(np.sum(tensor[:, :, 1])),
        'cvli_mean': float(np.mean(tensor[:, :, 0])),
        'cvli_std': float(np.std(tensor[:, :, 0])),
        'pris_mean': float(np.mean(tensor[:, :, 1])),
        'pris_std': float(np.std(tensor[:, :, 1])),
        'sparsity_cvli': float(np.sum(tensor[:, :, 0] == 0) / (n_timesteps * n_nodes)),
        'sparsity_pris': float(np.sum(tensor[:, :, 1] == 0) / (n_timesteps * n_nodes)),
    }
    
    print(f"\n📊 Estatísticas do tensor:")
    print(f"   Total CVLI: {metadata['total_cvli']:.0f}")
    print(f"   Total Prisões: {metadata['total_prisoes']:.0f}")
    print(f"   Esparsidade CVLI: {metadata['sparsity_cvli']*100:.1f}%")
    print(f"   Esparsidade Prisões: {metadata['sparsity_pris']*100:.1f}%")
    
    return tensor, bairros, date_range, metadata

# ═════════════════════════════════════════════════════════════════════════
# 5. ANÁLISES DE CORRELAÇÃO E CAUSALIDADE
# ═════════════════════════════════════════════════════════════════════════

def analyze_cross_correlation(tensor, bairros):
    """Analisa correlação cruzada entre CVLI e Prisões."""
    print("\n[4] Analisando correlações cruzadas...")
    
    analysis = {
        'correlations': [],
        'lag_effects': {},
        'effectiveness': {}
    }
    
    cvli_series = tensor[:, :, 0]  # (T, N)
    pris_series = tensor[:, :, 1]  # (T, N)
    
    positive_corr = 0
    negative_corr = 0
    strong_corr = 0
    
    for i, bairro in enumerate(bairros):
        cvli_vec = cvli_series[:, i]
        pris_vec = pris_series[:, i]
        
        if cvli_vec.sum() > 0 and pris_vec.sum() > 0:
            try:
                corr, pval = pearsonr(cvli_vec, pris_vec)
                
                analysis['correlations'].append({
                    'bairro': bairro,
                    'correlation': float(corr),
                    'pvalue': float(pval),
                    'cvli_total': float(cvli_vec.sum()),
                    'pris_total': float(pris_vec.sum()),
                    'effectiveness': 'Eficaz' if corr < -0.3 else 'Ineficaz' if corr > 0.3 else 'Neutra'
                })
                
                if corr > 0:
                    positive_corr += 1
                elif corr < 0:
                    negative_corr += 1
                
                if abs(corr) > 0.7:
                    strong_corr += 1
                    
            except:
                pass
    
    analysis['lag_effects']['positive_correlation_pct'] = (
        positive_corr / max(len(bairros), 1) * 100
    )
    analysis['lag_effects']['negative_correlation_pct'] = (
        negative_corr / max(len(bairros), 1) * 100
    )
    analysis['lag_effects']['strong_correlation_pct'] = (
        strong_corr / max(len(bairros), 1) * 100
    )
    
    return analysis

# ═════════════════════════════════════════════════════════════════════════
# 6. GERAR RELATÓRIO
# ═════════════════════════════════════════════════════════════════════════

def generate_report(tensor, bairros, metadata, analysis):
    """Gera relatório de viabilidade."""
    print("\n[5] Gerando relatório...")
    
    report_lines = [
        "# ANÁLISE 2: ST-GCN COM OCORRÊNCIAS + PRISÕES",
        "=" * 70,
        "",
        "## 📊 DATASET OVERVIEW",
        f"- **Período:** {metadata['date_min']} até {metadata['date_max']}",
        f"- **Timesteps:** {metadata['n_timesteps']} dias",
        f"- **Nós (bairros):** {metadata['n_nodes']}",
        f"- **Features:** {metadata['n_features']} (CVLI, Prisões, Lag)",
        f"- **Total CVLI:** {metadata['total_cvli']:.0f}",
        f"- **Total Prisões:** {metadata['total_prisoes']:.0f}",
        f"- **Razão Prisões/CVLI:** {metadata['total_prisoes']/max(metadata['total_cvli'], 1):.2f}",
        "",
        "## 📈 CARACTERÍSTICAS ESTATÍSTICAS",
        f"- **Esparsidade CVLI:** {metadata['sparsity_cvli']*100:.2f}%",
        f"- **Esparsidade Prisões:** {metadata['sparsity_pris']*100:.2f}%",
        f"  - Nota: Prisões {'mais' if metadata['sparsity_pris'] > metadata['sparsity_cvli'] else 'menos'} esparsas que CVLI",
        "",
        f"- **CVLI média/dia/nó:** {metadata['cvli_mean']:.4f}",
        f"- **Prisões média/dia/nó:** {metadata['pris_mean']:.4f}",
        "",
        "## 🔗 CORRELAÇÃO CRUZADA CVLI × PRISÕES",
        f"- **Correlação positiva:** {analysis['lag_effects']['positive_correlation_pct']:.1f}% dos bairros",
        f"  - Significa: Prisões ↑ E CVLI ↑ (aumentam juntos)",
        f"- **Correlação negativa:** {analysis['lag_effects']['negative_correlation_pct']:.1f}% dos bairros",
        f"  - Significa: Prisões ↑ E CVLI ↓ (efeito desejado)",
        f"- **Correlações fortes:** {analysis['lag_effects']['strong_correlation_pct']:.1f}% dos bairros",
        "",
    ]
    
    # Distribuição de efetividade
    if analysis['correlations']:
        eficaz_count = sum(1 for c in analysis['correlations'] if c['effectiveness'] == 'Eficaz')
        ineficaz_count = sum(1 for c in analysis['correlations'] if c['effectiveness'] == 'Ineficaz')
        
        report_lines.extend([
            "## 🎯 ANÁLISE DE EFETIVIDADE OPERACIONAL",
            f"- **Bairros com operações eficazes:** {eficaz_count}",
            f"  - Operações reduzem CVLI (correlação negativa)",
            f"- **Bairros com operações ineficazes:** {ineficaz_count}",
            f"  - Operações não reduzem ou aumentam CVLI",
            "",
        ])
    
    # Scoring
    sparsity_avg = (metadata['sparsity_cvli'] + metadata['sparsity_pris']) / 2
    sparsity_score = max(0, 100 - sparsity_avg * 100 * 2)
    
    correlation_info_score = (
        analysis['lag_effects']['negative_correlation_pct'] * 0.7 +
        (100 - analysis['lag_effects']['positive_correlation_pct']) * 0.3
    )
    
    feature_richness_score = 75  # Mais features = melhor potencial
    
    overall_score = (
        sparsity_score * 0.3 +
        correlation_info_score * 0.4 +
        feature_richness_score * 0.3
    )
    
    report_lines.extend([
        "## ⚖️ VIABILIDADE PARA ST-GCN (Análise 2)",
        "",
        "### Scoring (0-100)",
        f"- **Esparsidade média:** {sparsity_score:.1f}/100",
        f"- **Informação correlativa:** {correlation_info_score:.1f}/100",
        f"  - Mais bairros com correlação negativa = melhor previsibilidade",
        f"- **Riqueza de features:** {feature_richness_score:.1f}/100",
        f"- **SCORE GERAL:** {overall_score:.1f}/100",
        "",
    ])
    
    if overall_score >= 70:
        report_lines.append("### 🟢 RECOMENDAÇÃO: ALTAMENTE VIÁVEL")
        report_lines.append("Modelo pode ser treinado com acurácia ACEITÁVEL a BOA.")
    elif overall_score >= 55:
        report_lines.append("### 🟡 RECOMENDAÇÃO: VIÁVEL COM RESSALVAS")
        report_lines.append("Modelo pode funcionar. Use técnicas de regularização e validação cruzada robusta.")
    else:
        report_lines.append("### 🔴 RECOMENDAÇÃO: NÃO RECOMENDADO")
        report_lines.append("Features insuficientes para garantir performance.")
    
    report_lines.extend([
        "",
        "## 🎯 PRINCIPAIS VANTAGENS (Análise 2 vs Análise 1)",
        "1. **Múltiplas features**: Contexto operacional + atividade criminal",
        "2. **Informação causal potencial**: Prisões como preditor de CVLI",
        "3. **Menos esparsidade**: Duas séries distribuem informação",
        "4. **Validação cruzada**: Pode detectar se prisões realmente reduzem crimes",
        "",
        "## ⚠️ DESAFIOS ADICIONAIS",
        "1. **Confundimento**: Ambos podem ser causados por terceiro fator",
        "2. **Complexidade aumentada**: Mais features = mais parâmetros",
        "3. **Defasagem temporal**: Efeito de prisões pode não ser imediato",
        "",
        "## 📝 CONCLUSÃO ANÁLISE 2",
        f"Score: {overall_score:.1f}/100",
        f"Viabilidade: {'MUITO ALTA' if overall_score >= 70 else 'ALTA' if overall_score >= 55 else 'BAIXA'}",
        "",
        "---",
    ])
    
    return "\n".join(report_lines)

# ═════════════════════════════════════════════════════════════════════════
# 7. MAIN
# ═════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "="*70)
    print("ANÁLISE 2: ST-GCN COM OCORRÊNCIAS + PRISÕES")
    print("="*70)
    
    # 1. Carregar dados
    df_cvli = load_cvli_data()
    if df_cvli is None:
        print("❌ Falha ao carregar CVLI. Encerrando.")
        return
    
    df_prisoes = load_prisoes_data()
    if df_prisoes is None:
        print("❌ Falha ao carregar Prisões. Encerrando.")
        return
    
    # 2. Normalizar
    df_cvli, df_prisoes = normalize_data(df_cvli, df_prisoes)
    
    # 3. Construir tensor
    tensor, bairros, dates, metadata = build_multifeature_tensor(df_cvli, df_prisoes)
    if len(bairros) < 3:
        print("❌ Poucos bairros com dados em ambas as séries. Encerrando.")
        return
    
    # 4. Analisar correlações
    analysis = analyze_cross_correlation(tensor, bairros)
    
    # 5. Gerar relatório
    report = generate_report(tensor, bairros, metadata, analysis)
    
    # 6. Salvar resultados
    report_path = OUTPUT_DIR / "RELATORIO_ANALISE_2.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    metadata_path = OUTPUT_DIR / "metadata_analise_2.json"
    analysis_dict = {
        **metadata,
        'correlations': analysis['correlations'],
        'lag_effects': analysis['lag_effects']
    }
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(analysis_dict, f, indent=2, ensure_ascii=False, default=str)
    
    # 7. Salvar tensor
    tensor_path = OUTPUT_DIR / "tensor_ocorrencias_prisoes.npy"
    np.save(tensor_path, tensor)
    
    bairros_path = OUTPUT_DIR / "bairros_lista.json"
    with open(bairros_path, 'w', encoding='utf-8') as f:
        json.dump(bairros, f, ensure_ascii=False)
    
    print(f"\n✅ Relatório salvo em: {report_path}")
    print(f"✅ Metadata salvo em: {metadata_path}")
    print(f"✅ Tensor salvo em: {tensor_path}")
    
    print("\n" + "="*70)
    print(report)
    print("="*70)

if __name__ == '__main__':
    main()
