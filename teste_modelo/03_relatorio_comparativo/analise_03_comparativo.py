"""
ANÁLISE COMPARATIVA E RECOMENDAÇÃO FINAL
═════════════════════════════════════════════════════════════════════════

Compara os resultados das Análises 1 e 2 e fornece recomendação
para qual abordagem implementar ST-GCN com melhor viabilidade.

Critérios de decisão:
1. Qualidade dos dados (esparsidade, sinal)
2. Padrões espacio-temporais (autocorrelação, estacionariedade)
3. Complexidade vs. ganho informativo
4. Validação cruzada esperada
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from tabulate import tabulate

# ═══════════════════════════════════════════════════════════════════════
# 1. CARREGAR RESULTADOS
# ═══════════════════════════════════════════════════════════════════════

def load_analysis_results():
    """Carrega resultados das duas análises."""
    print("[1] Carregando resultados das análises...")
    
    base_path = Path(__file__).parent.parent
    
    analysis_1_dir = base_path / "01_apenas_ocorrencias"
    analysis_2_dir = base_path / "02_ocorrencias_prisoes"
    
    results = {}
    
    # Análise 1
    if (analysis_1_dir / "metadata_analise_1.json").exists():
        with open(analysis_1_dir / "metadata_analise_1.json", 'r') as f:
            results['analise_1'] = json.load(f)
        print("✅ Análise 1 carregada")
    else:
        print("❌ Análise 1 não encontrada. Execute: python analise_01_dataset_builder.py")
    
    # Análise 2
    if (analysis_2_dir / "metadata_analise_2.json").exists():
        with open(analysis_2_dir / "metadata_analise_2.json", 'r') as f:
            results['analise_2'] = json.load(f)
        print("✅ Análise 2 carregada")
    else:
        print("❌ Análise 2 não encontrada. Execute: python analise_02_dataset_builder.py")
    
    return results

# ═══════════════════════════════════════════════════════════════════════
# 2. EXTRAIR MÉTRICAS-CHAVE
# ═══════════════════════════════════════════════════════════════════════

def extract_metrics(results):
    """Extrai métricas-chave para comparação."""
    print("\n[2] Extraindo métricas-chave...")
    
    comparison = {
        'Métrica': [],
        'Análise 1 (Apenas CVLI)': [],
        'Análise 2 (CVLI + Prisões)': [],
        'Melhor': []
    }
    
    # 1. Dimensões do tensor
    if 'analise_1' in results:
        a1 = results['analise_1']
        t1 = a1.get('n_timesteps', 'N/A')
        n1 = a1.get('n_nodes', 'N/A')
        f1 = a1.get('n_features', 'N/A')
    else:
        t1 = n1 = f1 = 'N/A'
    
    if 'analise_2' in results:
        a2 = results['analise_2']
        t2 = a2.get('n_timesteps', 'N/A')
        n2 = a2.get('n_nodes', 'N/A')
        f2 = a2.get('n_features', 'N/A')
    else:
        t2 = n2 = f2 = 'N/A'
    
    comparison['Métrica'].append('Timesteps')
    comparison['Análise 1 (Apenas CVLI)'].append(str(t1))
    comparison['Análise 2 (CVLI + Prisões)'].append(str(t2))
    comparison['Melhor'].append('≈' if t1 == t2 else 'A1' if t1 > t2 else 'A2')
    
    comparison['Métrica'].append('Nós (bairros)')
    comparison['Análise 1 (Apenas CVLI)'].append(str(n1))
    comparison['Análise 2 (CVLI + Prisões)'].append(str(n2))
    comparison['Melhor'].append('≈' if n1 == n2 else 'A1' if n1 > n2 else 'A2')
    
    comparison['Métrica'].append('Features')
    comparison['Análise 1 (Apenas CVLI)'].append(str(f1))
    comparison['Análise 2 (CVLI + Prisões)'].append(str(f2))
    comparison['Melhor'].append('A2 (mais features)')
    
    # 2. Esparsidade
    if 'analise_1' in results:
        sparse_a1 = results['analise_1'].get('sparsity', 'N/A')
        if isinstance(sparse_a1, (int, float)):
            sparse_a1_pct = f"{sparse_a1*100:.1f}%"
        else:
            sparse_a1_pct = sparse_a1
    else:
        sparse_a1_pct = 'N/A'
    
    if 'analise_2' in results:
        sparse_a2_cvli = results['analise_2'].get('sparsity_cvli', 0)
        if isinstance(sparse_a2_cvli, (int, float)):
            sparse_a2_pct = f"{sparse_a2_cvli*100:.1f}%"
        else:
            sparse_a2_pct = 'N/A'
    else:
        sparse_a2_pct = 'N/A'
    
    comparison['Métrica'].append('Esparsidade')
    comparison['Análise 1 (Apenas CVLI)'].append(sparse_a1_pct)
    comparison['Análise 2 (CVLI + Prisões)'].append(sparse_a2_pct)
    comparison['Melhor'].append('A2 (mais dados)' if float(sparse_a2_cvli or 0) < float(sparse_a1 or 1) else 'A1')
    
    # 3. Total de eventos
    if 'analise_1' in results:
        total_a1 = results['analise_1'].get('total_cvli', 'N/A')
    else:
        total_a1 = 'N/A'
    
    if 'analise_2' in results:
        total_cvli_a2 = results['analise_2'].get('total_cvli', 0)
        total_pris_a2 = results['analise_2'].get('total_prisoes', 0)
        total_a2_str = f"CVLI: {int(total_cvli_a2)}, Pris: {int(total_pris_a2)}"
    else:
        total_a2_str = 'N/A'
    
    comparison['Métrica'].append('Total de eventos')
    comparison['Análise 1 (Apenas CVLI)'].append(str(int(total_a1) if total_a1 != 'N/A' else 'N/A'))
    comparison['Análise 2 (CVLI + Prisões)'].append(total_a2_str)
    comparison['Melhor'].append('A2 (mais contexto)')
    
    # 4. Estacionariedade
    if 'analise_1' in results and 'stationary_nodes_pct' in results['analise_1'].get('temporal_autocorr', {}):
        stat_a1 = results['analise_1']['temporal_autocorr']['stationary_nodes_pct']
        stat_a1_str = f"{stat_a1:.1f}%"
    else:
        stat_a1_str = 'N/A'
    
    if 'analise_2' in results:
        stat_a2_str = 'Análise completa'
    else:
        stat_a2_str = 'N/A'
    
    comparison['Métrica'].append('Estacionariedade')
    comparison['Análise 1 (Apenas CVLI)'].append(stat_a1_str)
    comparison['Análise 2 (CVLI + Prisões)'].append(stat_a2_str)
    comparison['Melhor'].append('✓ Ambas')
    
    # 5. Correlação informativa
    if 'analise_2' in results and 'lag_effects' in results['analise_2']:
        neg_corr = results['analise_2']['lag_effects'].get('negative_correlation_pct', 0)
        corr_a2_str = f"{neg_corr:.1f}% com correlação negativa (eficaz)"
    else:
        corr_a2_str = 'N/A'
    
    comparison['Métrica'].append('Correlação CVLI-Prisões')
    comparison['Análise 1 (Apenas CVLI)'].append('N/A (sem prisões)')
    comparison['Análise 2 (CVLI + Prisões)'].append(corr_a2_str)
    comparison['Melhor'].append('A2')
    
    return pd.DataFrame(comparison)

# ═══════════════════════════════════════════════════════════════════════
# 3. SCORING E RECOMENDAÇÃO
# ═══════════════════════════════════════════════════════════════════════

def calculate_scores(results):
    """Calcula score de viabilidade para cada análise."""
    print("\n[3] Calculando scores de viabilidade...")
    
    scores = {'analise_1': {}, 'analise_2': {}}
    
    # Análise 1
    if 'analise_1' in results:
        a1 = results['analise_1']
        
        # Fator esparsidade (menos é melhor)
        sparse_a1 = a1.get('sparsity', 0.5)
        sparsity_score_a1 = max(0, 100 - sparse_a1 * 100 * 2)
        
        # Fator estacionariedade
        stat_a1 = a1.get('temporal_autocorr', {}).get('stationary_nodes_pct', 40)
        stat_score_a1 = stat_a1
        
        # Fator signal-to-noise
        mean_a1 = a1.get('mean', 0)
        std_a1 = a1.get('std', 1)
        snr_a1 = (mean_a1 / max(std_a1, 0.01)) * 10
        signal_score_a1 = min(100, snr_a1)
        
        scores['analise_1'] = {
            'sparsity': sparsity_score_a1,
            'stationarity': stat_score_a1,
            'signal': signal_score_a1,
            'features': 33,  # 1 feature
            'overall': (sparsity_score_a1 * 0.35 + stat_score_a1 * 0.35 + signal_score_a1 * 0.2 + 33 * 0.1)
        }
    
    # Análise 2
    if 'analise_2' in results:
        a2 = results['analise_2']
        
        # Esparsidade média
        sparse_cvli = a2.get('sparsity_cvli', 0.5)
        sparse_pris = a2.get('sparsity_pris', 0.5)
        sparse_avg = (sparse_cvli + sparse_pris) / 2
        sparsity_score_a2 = max(0, 100 - sparse_avg * 100 * 2)
        
        # Correlação informativa
        neg_corr = a2.get('lag_effects', {}).get('negative_correlation_pct', 30)
        corr_score_a2 = min(100, neg_corr * 2)  # Quanto mais negativa, melhor
        
        # Features
        feature_score_a2 = 75  # 3 features
        
        scores['analise_2'] = {
            'sparsity': sparsity_score_a2,
            'correlation': corr_score_a2,
            'features': feature_score_a2,
            'overall': (sparsity_score_a2 * 0.35 + corr_score_a2 * 0.35 + feature_score_a2 * 0.3)
        }
    
    return scores

# ═══════════════════════════════════════════════════════════════════════
# 4. GERAR RELATÓRIO COMPARATIVO
# ═══════════════════════════════════════════════════════════════════════

def generate_comparative_report(results, comparison_df, scores):
    """Gera relatório comparativo final."""
    print("\n[4] Gerando relatório comparativo...")
    
    lines = [
        "# ANÁLISE COMPARATIVA: QUAL ABORDAGEM É MAIS VIÁVEL?",
        "=" * 80,
        "",
        "## 📊 COMPARAÇÃO DE DADOS",
        "",
        "### Tabela Comparativa",
        "",
        tabulate(comparison_df, headers='keys', tablefmt='github', showindex=False),
        "",
        "",
        "## 🎯 SCORING DE VIABILIDADE (0-100)",
        "",
        "### Análise 1: Apenas Ocorrências (CVLI)",
        "",
    ]
    
    if 'analise_1' in scores:
        s1 = scores['analise_1']
        lines.extend([
            f"- **Esparsidade:** {s1.get('sparsity', 0):.1f}/100",
            f"  → Capacidade de ter dados significativos",
            f"- **Estacionariedade:** {s1.get('stationarity', 0):.1f}/100",
            f"  → Previsibilidade temporal",
            f"- **Sinal-to-Noise:** {s1.get('signal', 0):.1f}/100",
            f"  → Clareza do padrão vs ruído",
            f"- **Riqueza de Features:** {s1.get('features', 0):.1f}/100",
            f"  → Informação disponível (1 feature apenas)",
            "",
            f"### **SCORE GERAL: {s1.get('overall', 0):.1f}/100**",
            "",
        ])
    
    lines.append("### Análise 2: Ocorrências + Prisões (Features Cruzadas)")
    lines.append("")
    
    if 'analise_2' in scores:
        s2 = scores['analise_2']
        lines.extend([
            f"- **Esparsidade:** {s2.get('sparsity', 0):.1f}/100",
            f"  → Distribuição de dados entre 2 séries",
            f"- **Correlação Informativa:** {s2.get('correlation', 0):.1f}/100",
            f"  → Relação entre prisões e CVLI (causalidade potencial)",
            f"- **Riqueza de Features:** {s2.get('features', 0):.1f}/100",
            f"  → Informação disponível (3 features + contexto)",
            "",
            f"### **SCORE GERAL: {s2.get('overall', 0):.1f}/100**",
            "",
        ])
    
    # Decisão final
    if 'analise_1' in scores and 'analise_2' in scores:
        score_1 = scores['analise_1']['overall']
        score_2 = scores['analise_2']['overall']
        
        lines.append("## 🏆 RECOMENDAÇÃO FINAL")
        lines.append("")
        
        if score_2 > score_1 + 10:
            lines.extend([
                f"### **✅ RECOMENDAÇÃO: ANÁLISE 2 (OCORRÊNCIAS + PRISÕES)**",
                "",
                f"**Score Análise 2: {score_2:.1f}/100** > **Score Análise 1: {score_1:.1f}/100**",
                "",
                "#### Por que Análise 2 é melhor?",
                "",
                "1. **Mais contexto informativo**",
                "   - 3 features vs 1 feature",
                "   - Captura relação causal (prisões → CVLI)",
                "",
                "2. **Correlação CVLI-Prisões revela padrões**",
                "   - Bairros onde operações são eficazes",
                "   - Bairros onde operações não funcionam",
                "   - Permite aprendizado de táticas",
                "",
                "3. **Melhor para previsão de CVLI**",
                "   - Contexto operacional ajuda o modelo",
                "   - Reduz ambiguidade de padrões",
                "",
                "#### Como implementar?",
                "",
                "```bash",
                "cd teste_modelo/02_ocorrencias_prisoes/",
                "python analise_02_dataset_builder.py",
                "# Usar tensor_ocorrencias_prisoes.npy para treinar ST-GCN",
                "```",
                "",
            ])
        
        elif score_1 > score_2 + 10:
            lines.extend([
                f"### **✅ RECOMENDAÇÃO: ANÁLISE 1 (APENAS OCORRÊNCIAS)**",
                "",
                f"**Score Análise 1: {score_1:.1f}/100** > **Score Análise 2: {score_2:.1f}/100**",
                "",
                "#### Por que Análise 1 é melhor?",
                "",
                "1. **Menos ruído de confundimento**",
                "   - Apenas o fenômeno de interesse",
                "   - Sem variáveis confundidoras",
                "",
                "2. **Baseline mais limpo**",
                "   - Facilita interpretação",
                "   - Reduz overfitting",
                "",
            ])
        
        else:
            lines.extend([
                f"### **⚠️ RECOMENDAÇÃO: IMPLEMENTAR AMBAS (com preferência em Análise 2)**",
                "",
                f"**Score Análise 1: {score_1:.1f}/100** vs **Score Análise 2: {score_2:.1f}/100**",
                "",
                "#### Estratégia híbrida:",
                "",
                "1. **Phase 1**: Treinar modelo com Análise 2",
                "   - Aproveita melhor a informação disponível",
                "   - Valida efetividade de operações",
                "",
                "2. **Phase 2**: Comparar com modelo Análise 1",
                "   - Identifica quanto prisões contribuem",
                "   - Evita overfitting por confundimento",
                "",
            ])
    
    lines.extend([
        "",
        "## 📋 PRÓXIMOS PASSOS",
        "",
        "### 1. Validação Cruzada",
        "```python",
        "# Dividir 70% treino / 30% teste",
        "# Usar k-fold CV (k=5)",
        "# Métrica: MAE, RMSE, R²",
        "```",
        "",
        "### 2. Teste de Modelos Baseline",
        "```python",
        "# ARIMA para comparação",
        "# Prophet para sazonalidade",
        "# Regressão Linear (baseline simples)",
        "```",
        "",
        "### 3. Ajuste de Hiperparâmetros ST-GCN",
        "```python",
        "# Hidden dimensions: 32, 64, 128",
        "# Número de layers: 2, 3, 4",
        "# Learning rate: 0.001, 0.01",
        "# Dropout: 0.2, 0.5",
        "```",
        "",
        "---",
    ])
    
    return "\n".join(lines)

# ═════════════════════════════════════════════════════════════════════════
# 5. MAIN
# ═════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "="*80)
    print("ANÁLISE COMPARATIVA: QUAL ABORDAGEM É MAIS VIÁVEL PARA ST-GCN?")
    print("="*80)
    
    # 1. Carregar resultados
    results = load_analysis_results()
    
    if not results:
        print("❌ Nenhuma análise encontrada. Execute ambos os scripts primeiro:")
        print("   python teste_modelo/01_apenas_ocorrencias/analise_01_dataset_builder.py")
        print("   python teste_modelo/02_ocorrencias_prisoes/analise_02_dataset_builder.py")
        return
    
    # 2. Extrair comparação
    comparison_df = extract_metrics(results)
    print("\n✅ Métricas extraídas")
    
    # 3. Calcular scores
    scores = calculate_scores(results)
    print("✅ Scores calculados")
    
    # 4. Gerar relatório
    report = generate_comparative_report(results, comparison_df, scores)
    
    # 5. Salvar
    output_dir = Path(__file__).parent
    report_path = output_dir / "RELATORIO_COMPARATIVO_FINAL.md"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n✅ Relatório salvo: {report_path}")
    
    print("\n" + "="*80)
    print(report)
    print("="*80)

if __name__ == '__main__':
    main()
