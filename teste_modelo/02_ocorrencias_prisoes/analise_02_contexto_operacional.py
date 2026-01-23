"""
ANÁLISE 2: Viabilidade ST-GCN com CVLI + Prisões (contexto operacional)
Dataset: 2022+, com coordenadas geográficas enriquecidas
Features: CVLI + Prisões + Apreensões (drogas/armas/dinheiro)
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy import stats

OUTPUT_DIR = Path("teste_modelo/02_ocorrencias_prisoes")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

DATA_FILE = Path("data/raw/View_Ocorrencias_2022_ENRIQUECIDO.csv")

print("\n" + "="*80)
print("ANÁLISE 2: VIABILIDADE ST-GCN CVLI + PRISÕES (CONTEXTO OPERACIONAL)")
print("="*80)

# 1. Carregar dados
print("\n[1] Carregando dados enriquecidos...")
df = pd.read_csv(DATA_FILE)
df['Data'] = pd.to_datetime(df['Data'])
print(f"   ✓ {len(df):,} registros")
print(f"   ✓ Período: {df['Data'].min().date()} a {df['Data'].max().date()}")

# 2. Remover registros sem coordenadas
print("\n[2] Filtrando registros com coordenadas...")
df = df.dropna(subset=['lat', 'long'])
print(f"   ✓ {len(df):,} registros com coordenadas")

# 3. Construir matriz espacial com 3 features (T × N × 3)
print("\n[3] Construindo matriz temporal com 3 features...")
dates = pd.date_range(df['Data'].min(), df['Data'].max(), freq='D')
T = len(dates)

# Agrupar bairros
bairros = df['BairroOcor'].unique()
N = len(bairros)
bairro_to_idx = {b: i for i, b in enumerate(bairros)}

print(f"   ✓ Período: {T} dias")
print(f"   ✓ Bairros: {N}")
print(f"   ✓ Features: 3 (CVLI, Prisões, Apreensões)")
print(f"   ✓ Dimensões: {T} × {N} × 3 = {T*N*3:,} células")

# Construir matrizes por feature
matrix_cvli = np.zeros((T, N))
matrix_prisoes = np.zeros((T, N))
matrix_apreensoes = np.zeros((T, N))

daily_totals_cvli = np.zeros(T)
daily_totals_prisoes = np.zeros(T)
daily_totals_apreensoes = np.zeros(T)

# Feature 1: CVLI
cvli_mask = df['Natureza'].str.contains('Homicídio|Latrocínio', case=False, na=False)
df_cvli = df[cvli_mask].copy()

# Feature 2: Prisões (operações com prisões)
prisoes_mask = df['Natureza'].str.contains('Prisão|Preso', case=False, na=False)
df_prisoes = df[prisoes_mask].copy()

# Feature 3: Apreensões significativas (drogas + armas + dinheiro)
apreensoes_mask = (
    (df['total_armas_cache'] > 0) | 
    (df['total_drogas_cache'] > 0) | 
    (df['Dinheiro_Apreendido'] > 0)
)
df_apreensoes = df[apreensoes_mask].copy()

print(f"\n   Feature 1 (CVLI): {len(df_cvli):,} registros ({len(df_cvli)/len(df)*100:.2f}%)")
print(f"   Feature 2 (Prisões): {len(df_prisoes):,} registros ({len(df_prisoes)/len(df)*100:.2f}%)")
print(f"   Feature 3 (Apreensões): {len(df_apreensoes):,} registros ({len(df_apreensoes)/len(df)*100:.2f}%)")

# Preencher matrizes
for idx, row in df.iterrows():
    t_idx = (row['Data'].date() - dates[0].date()).days
    if 0 <= t_idx < T:
        n_idx = bairro_to_idx[row['BairroOcor']]
        
        if cvli_mask.iloc[idx]:
            matrix_cvli[t_idx, n_idx] += 1
            daily_totals_cvli[t_idx] += 1
        
        if prisoes_mask.iloc[idx]:
            matrix_prisoes[t_idx, n_idx] += 1
            daily_totals_prisoes[t_idx] += 1
        
        if apreensoes_mask.iloc[idx]:
            matrix_apreensoes[t_idx, n_idx] += 1
            daily_totals_apreensoes[t_idx] += 1

print(f"\n   ✓ Matrizes construídas com sucesso")

# 4. Calcular métricas por feature
print("\n[4] Calculando métricas de viabilidade...")

features_data = {
    'CVLI': (matrix_cvli, daily_totals_cvli, 'homicídio/latrocínio'),
    'Prisões': (matrix_prisoes, daily_totals_prisoes, 'prisões'),
    'Apreensões': (matrix_apreensoes, daily_totals_apreensoes, 'drogas/armas/dinheiro')
}

scores = {}
print(f"\n   📊 ANÁLISE POR FEATURE\n")

for feature_name, (matrix, daily_totals, desc) in features_data.items():
    nonzero = np.count_nonzero(matrix)
    sparsity = 1 - (nonzero / matrix.size)
    signal_mean = matrix.sum() / (T * N)
    signal_nz = matrix[matrix > 0].mean() if nonzero > 0 else 0
    
    valid_days = daily_totals[daily_totals > 0]
    cv = valid_days.std() / valid_days.mean() if len(valid_days) > 0 and valid_days.mean() > 0 else 0
    
    bairros_with_events = np.sum(matrix.sum(axis=0) > 0)
    
    # Scores
    score_sparsity = max(0, 100 - sparsity * 150)
    score_signal = min(100, signal_nz * 1000)
    score_variability = max(0, 100 - abs(cv - 0.5) * 50)
    score_coverage = (bairros_with_events / N) * 100
    
    feature_score = (score_sparsity * 0.25 + score_signal * 0.35 + 
                    score_variability * 0.20 + score_coverage * 0.20)
    
    scores[feature_name] = feature_score
    
    print(f"   {feature_name} ({desc})")
    print(f"      Eventos: {int(matrix.sum()):,} | Células: {nonzero:,} | Esparsidade: {sparsity*100:.1f}%")
    print(f"      Sinal (nz): {signal_nz:.4f} | CV: {cv:.3f} | Score: {feature_score:.1f}/100")
    print()

# 5. Correlações entre features
print(f"   🔗 CORRELAÇÕES ENTRE FEATURES\n")

# Flatten matrices para correlação
flat_cvli = matrix_cvli.flatten()
flat_prisoes = matrix_prisoes.flatten()
flat_apreensoes = matrix_apreensoes.flatten()

corr_cvli_prisoes, p_cvli_prisoes = stats.pearsonr(flat_cvli, flat_prisoes)
corr_cvli_apreensoes, p_cvli_apreensoes = stats.pearsonr(flat_cvli, flat_apreensoes)
corr_prisoes_apreensoes, p_prisoes_apreensoes = stats.pearsonr(flat_prisoes, flat_apreensoes)

print(f"   CVLI ↔ Prisões: r={corr_cvli_prisoes:.3f} (p={p_cvli_prisoes:.4f})")
print(f"   CVLI ↔ Apreensões: r={corr_cvli_apreensoes:.3f} (p={p_cvli_apreensoes:.4f})")
print(f"   Prisões ↔ Apreensões: r={corr_prisoes_apreensoes:.3f} (p={p_prisoes_apreensoes:.4f})")

# 6. Score geral (média ponderada das features + correlações)
print(f"\n   🎯 CÁLCULO DE VIABILIDADE GERAL\n")

avg_feature_score = np.mean(list(scores.values()))

# Bônus por correlação (features correlacionadas são mais úteis)
corr_bonus = 0
if abs(corr_cvli_prisoes) > 0.1:
    corr_bonus += 10 * abs(corr_cvli_prisoes)
if abs(corr_cvli_apreensoes) > 0.1:
    corr_bonus += 10 * abs(corr_cvli_apreensoes)

overall_score = min(100, avg_feature_score + corr_bonus)

print(f"   Score médio features: {avg_feature_score:.1f}/100")
print(f"   Bônus correlação: +{corr_bonus:.1f}")
print(f"   GERAL: {overall_score:.1f}/100")

# 7. Comparação com Análise 1
print(f"\n   📊 COMPARAÇÃO COM ANÁLISE 1")

# Carregar score de Análise 1
try:
    with open("teste_modelo/01_apenas_ocorrencias/metadata_analise_1.json", 'r') as f:
        metadata_1 = json.load(f)
    score_1 = metadata_1['score']['geral']
    improvement = overall_score - score_1
    improvement_pct = (improvement / score_1) * 100 if score_1 > 0 else 0
    
    print(f"   Análise 1 (CVLI-only): {score_1:.1f}/100")
    print(f"   Análise 2 (CVLI+Prisões): {overall_score:.1f}/100")
    print(f"   Melhoria: +{improvement:.1f} ({improvement_pct:+.1f}%)")
except:
    print(f"   ⚠️ Não foi possível carregar score de Análise 1")

# 8. Salvar tensor (empilhado: T×N×3)
print(f"\n[5] Salvando tensor e metadados...")
tensor_combined = np.stack([matrix_cvli, matrix_prisoes, matrix_apreensoes], axis=2)
tensor_path = OUTPUT_DIR / "tensor_cvli_prisoes.npy"
np.save(tensor_path, tensor_combined)
print(f"   ✅ Tensor: {tensor_path} (shape: {tensor_combined.shape})")

# 9. Gerar relatório
print(f"\n[6] Gerando relatório...")

report = f"""# ANÁLISE 2: VIABILIDADE ST-GCN CVLI + PRISÕES (CONTEXTO OPERACIONAL)

## 📊 Resumo Executivo

**Dataset:** View_Ocorrencias_2022_ENRIQUECIDO.csv (dados com lat/long IBGE)
**Período:** {df['Data'].min().date()} a {df['Data'].max().date()} ({T} dias)
**Cobertura geográfica:** {N} bairros (Fortaleza + RMF + Interior)
**Dimensões do tensor:** T={T} × N={N} × F=3 → {T*N*3:,} células

### Features Utilizadas

1. **CVLI** ({len(df_cvli):,} registros - {len(df_cvli)/len(df)*100:.2f}%)
   - Homicídios e latrocínios
   - Score: {scores.get('CVLI', 0):.1f}/100

2. **Prisões** ({len(df_prisoes):,} registros - {len(df_prisoes)/len(df)*100:.2f}%)
   - Operações com prisões
   - Score: {scores.get('Prisões', 0):.1f}/100

3. **Apreensões** ({len(df_apreensoes):,} registros - {len(df_apreensoes)/len(df)*100:.2f}%)
   - Drogas, armas e/ou dinheiro apreendido
   - Score: {scores.get('Apreensões', 0):.1f}/100

## 📈 Correlações entre Features

| Relação | Correlação (r) | p-value | Interpretação |
|---------|--------|---------|--------------|
| CVLI ↔ Prisões | {corr_cvli_prisoes:.3f} | {p_cvli_prisoes:.4f} | {"Forte" if abs(corr_cvli_prisoes) > 0.5 else "Moderada" if abs(corr_cvli_prisoes) > 0.3 else "Fraca"} |
| CVLI ↔ Apreensões | {corr_cvli_apreensoes:.3f} | {p_cvli_apreensoes:.4f} | {"Forte" if abs(corr_cvli_apreensoes) > 0.5 else "Moderada" if abs(corr_cvli_apreensoes) > 0.3 else "Fraca"} |
| Prisões ↔ Apreensões | {corr_prisoes_apreensoes:.3f} | {p_prisoes_apreensoes:.4f} | {"Forte" if abs(corr_prisoes_apreensoes) > 0.5 else "Moderada" if abs(corr_prisoes_apreensoes) > 0.3 else "Fraca"} |

## 🎯 Viabilidade ST-GCN

### Scoring (0-100)
- **Score médio (features):** {avg_feature_score:.1f}/100
- **Bônus correlação:** +{corr_bonus:.1f}
- **SCORE GERAL:** **{overall_score:.1f}/100**

### Recomendação
"""

if overall_score >= 75:
    report += f"**🟢 ALTAMENTE VIÁVEL** - Score {overall_score:.1f}/100\n"
    report += "ST-GCN com contexto operacional é recomendado para implementação.\n"
elif overall_score >= 60:
    report += f"**🟡 VIÁVEL** - Score {overall_score:.1f}/100\n"
    report += "ST-GCN pode ser implementado com razoável performance.\n"
else:
    report += f"**🔴 NÃO RECOMENDADO** - Score {overall_score:.1f}/100\n"
    report += "Dataset insuficiente mesmo com contexto operacional.\n"

try:
    with open("teste_modelo/01_apenas_ocorrencias/metadata_analise_1.json", 'r') as f:
        metadata_1 = json.load(f)
    score_1 = metadata_1['score']['geral']
    improvement = overall_score - score_1
    improvement_pct = (improvement / score_1) * 100 if score_1 > 0 else 0
    
    report += f"""
## 📊 Comparação: CVLI-only vs CVLI+Prisões

| Aspecto | CVLI-only | CVLI+Prisões | Diferença |
|---------|-----------|--------------|-----------|
| **Score Geral** | {score_1:.1f}/100 | {overall_score:.1f}/100 | **{improvement:+.1f}** ({improvement_pct:+.1f}%) |
| **Recomendação** | {"✅ Viável" if score_1 >= 60 else "⚠️ Limitado"} | {"✅ Viável" if overall_score >= 60 else "⚠️ Limitado"} | {"Melhoria" if overall_score > score_1 else "Piora"} |

### Conclusão
"""
    if overall_score > score_1 + 5:
        report += f"**Análise 2 é significativamente superior.** Recomenda-se usar contexto operacional (Prisões) para melhorar previsibilidade."
    elif overall_score > score_1:
        report += f"**Análise 2 mostra melhoria marginal.** Ambas abordagens são viáveis; escolha depende de complexidade aceita."
    else:
        report += f"**Análise 1 é superior.** CVLI-only é mais adequado para este dataset."
except:
    report += "\n⚠️ Não foi possível comparar com Análise 1"

report += f"""

## 📋 Próximos Passos

1. ✅ Ambas análises concluídas - compareversão final gerada
2. Selecionar abordagem mais viável para implementação ST-GCN
3. Se score > 70: Proceder com implementação de modelo
4. Validação cruzada com holdout (últimos 30 dias)
5. Tuning de hyperparâmetros do ST-GCN

---
**Data de geração:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Arquivo tensor:** tensor_cvli_prisoes.npy ({T} × {N} × 3)
"""

report_path = OUTPUT_DIR / "RELATORIO_ANALISE_2.md"
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(report)
print(f"   ✅ Relatório: {report_path}")

# 10. Salvar metadados
metadata = {
    'periodo': f"{df['Data'].min().date()} a {df['Data'].max().date()}",
    'total_dias': int(T),
    'total_bairros': int(N),
    'features': list(scores.keys()),
    'tensor_shape': [int(T), int(N), 3],
    'tensor_cells': int(T*N*3),
    'feature_scores': scores,
    'correlacoes': {
        'cvli_prisoes': float(corr_cvli_prisoes),
        'cvli_apreensoes': float(corr_cvli_apreensoes),
        'prisoes_apreensoes': float(corr_prisoes_apreensoes)
    },
    'score': {
        'media_features': float(avg_feature_score),
        'bonus_correlacao': float(corr_bonus),
        'geral': float(overall_score)
    }
}

metadata_path = OUTPUT_DIR / "metadata_analise_2.json"
with open(metadata_path, 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=2)
print(f"   ✅ Metadados: {metadata_path}")

print("\n" + "="*80)
print(f"✅ ANÁLISE 2 CONCLUÍDA - Score: {overall_score:.1f}/100")
print("="*80)
