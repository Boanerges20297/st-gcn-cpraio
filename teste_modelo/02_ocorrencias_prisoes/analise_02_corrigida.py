"""
ANÁLISE 2 (CORRIGIDA): Viabilidade ST-GCN com CVLI + Prisões + Apreensões
Dataset: outputs/cvli_with_bairro.csv (tipo=cvli) + data/raw operacional
Features: CVLI + Prisões (contexto) + Apreensões
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy import stats

OUTPUT_DIR = Path("teste_modelo/02_ocorrencias_prisoes")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

CVLI_FILE = Path("outputs/cvli_with_bairro.csv")
OPERACIONAL_FILE = Path("data/raw/View_Ocorrencias_Operacionais_Modelo.csv")

print("\n" + "="*80)
print("ANÁLISE 2 (CORRIGIDA): VIABILIDADE ST-GCN CVLI + CONTEXTO OPERACIONAL")
print("="*80)

# 1. Carregar dados CVLI
print("\n[1] Carregando dados CVLI...")
df_cvli = pd.read_csv(CVLI_FILE, low_memory=False)
df_cvli['data'] = pd.to_datetime(df_cvli['data'])
df_cvli = df_cvli[(df_cvli['data'].dt.year >= 2022) & (df_cvli['tipo'].str.lower() == 'cvli')].copy()
df_cvli = df_cvli.dropna(subset=['latitude', 'longitude', 'bairro_assigned'])
print(f"   ✓ {len(df_cvli):,} eventos CVLI com bairro normalizado")

# 2. Carregar dados operacionais (prisões + apreensões)
print("\n[2] Carregando dados operacionais...")
df_op = pd.read_csv(OPERACIONAL_FILE, low_memory=False)
df_op['Data'] = pd.to_datetime(df_op['Data'])
df_op = df_op[(df_op['Data'].dt.year >= 2022)].copy()
print(f"   ✓ {len(df_op):,} registros operacionais")

# Identificar prisões e apreensões
prisoes = (df_op['Natureza'].str.contains('Prisão|Preso|Mandado', case=False, na=False)).sum()
apreensoes = ((df_op['total_armas_cache'] > 0) | 
              (df_op['total_drogas_cache'] > 0) | 
              (df_op['Dinheiro_Apreendido'] > 0)).sum()
print(f"   ✓ {prisoes:,} registros de prisão")
print(f"   ✓ {apreensoes:,} registros com apreensão")

# 3. Normalizar bairros operacionais
print("\n[3] Normalizando bairros operacionais...")
# Usar fuzzy matching simples
from difflib import SequenceMatcher

bairros_cvli = set(df_cvli['bairro_assigned'].dropna().unique())

def find_closest_bairro(bairro_op):
    if pd.isna(bairro_op):
        return None
    bairro_op = str(bairro_op).upper().strip()
    
    best_match = None
    best_ratio = 0.5  # Threshold mínimo
    
    for bairro_cvli in bairros_cvli:
        ratio = SequenceMatcher(None, bairro_op, bairro_cvli.upper()).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_match = bairro_cvli
    
    return best_match

df_op['bairro_norm'] = df_op['BairroOcor'].apply(find_closest_bairro)
df_op_norm = df_op.dropna(subset=['bairro_norm'])
print(f"   ✓ {len(df_op_norm):,}/{len(df_op):,} registros normalizados ({len(df_op_norm)/len(df_op)*100:.1f}%)")

# 4. Combinar datasets e construir matriz T × N × 3
print("\n[4] Construindo matriz temporal com 3 features...")

# Determinar período comum
date_min = max(df_cvli['data'].min(), df_op_norm['Data'].min())
date_max = min(df_cvli['data'].max(), df_op_norm['Data'].max())

dates = pd.date_range(date_min, date_max, freq='D')
T = len(dates)

# Bairros
bairros = sorted(list(bairros_cvli))
N = len(bairros)
bairro_to_idx = {b: i for i, b in enumerate(bairros)}

print(f"   ✓ Período: {T} dias ({date_min.date()} a {date_max.date()})")
print(f"   ✓ Bairros: {N}")
print(f"   ✓ Features: 3 (CVLI, Prisões, Apreensões)")
print(f"   ✓ Dimensões: {T} × {N} × 3 = {T*N*3:,} células")

# Matrizes por feature
matrix_cvli = np.zeros((T, N))
matrix_prisoes = np.zeros((T, N))
matrix_apreensoes = np.zeros((T, N))

daily_totals_cvli = np.zeros(T)
daily_totals_prisoes = np.zeros(T)
daily_totals_apreensoes = np.zeros(T)

# Preencher CVLI
for idx, row in df_cvli.iterrows():
    t_idx = (row['data'].date() - dates[0].date()).days
    if 0 <= t_idx < T:
        n_idx = bairro_to_idx[row['bairro_assigned']]
        matrix_cvli[t_idx, n_idx] += 1
        daily_totals_cvli[t_idx] += 1

# Preencher Prisões
df_prisoes = df_op_norm[df_op_norm['Natureza'].str.contains('Prisão|Preso|Mandado', case=False, na=False)].copy()
for idx, row in df_prisoes.iterrows():
    t_idx = (row['Data'].date() - dates[0].date()).days
    if 0 <= t_idx < T:
        n_idx = bairro_to_idx[row['bairro_norm']]
        matrix_prisoes[t_idx, n_idx] += 1
        daily_totals_prisoes[t_idx] += 1

# Preencher Apreensões
df_apreensoes = df_op_norm[
    ((df_op_norm['total_armas_cache'] > 0) | 
     (df_op_norm['total_drogas_cache'] > 0) | 
     (df_op_norm['Dinheiro_Apreendido'] > 0))
].copy()
for idx, row in df_apreensoes.iterrows():
    t_idx = (row['Data'].date() - dates[0].date()).days
    if 0 <= t_idx < T:
        n_idx = bairro_to_idx[row['bairro_norm']]
        matrix_apreensoes[t_idx, n_idx] += 1
        daily_totals_apreensoes[t_idx] += 1

print(f"\n   ✓ Matrizes construídas")
print(f"      CVLI: {int(matrix_cvli.sum()):,} eventos")
print(f"      Prisões: {int(matrix_prisoes.sum()):,} eventos")
print(f"      Apreensões: {int(matrix_apreensoes.sum()):,} eventos")

# 5. Calcular métricas por feature
print("\n[5] Calculando métricas de viabilidade...")

features_data = {
    'CVLI': (matrix_cvli, daily_totals_cvli),
    'Prisões': (matrix_prisoes, daily_totals_prisoes),
    'Apreensões': (matrix_apreensoes, daily_totals_apreensoes)
}

scores = {}
print(f"\n   📊 ANÁLISE POR FEATURE\n")

for feature_name, (matrix, daily_totals) in features_data.items():
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
    
    print(f"   {feature_name}")
    print(f"      Score: {feature_score:.1f}/100 | Eventos: {int(matrix.sum()):,} | Células: {nonzero:,}")
    print()

# 6. Correlações entre features
print(f"   🔗 CORRELAÇÕES ENTRE FEATURES\n")

flat_cvli = matrix_cvli.flatten()
flat_prisoes = matrix_prisoes.flatten()
flat_apreensoes = matrix_apreensoes.flatten()

corr_cvli_prisoes, p_cvli_prisoes = stats.pearsonr(flat_cvli, flat_prisoes)
corr_cvli_apreensoes, p_cvli_apreensoes = stats.pearsonr(flat_cvli, flat_apreensoes)
corr_prisoes_apreensoes, p_prisoes_apreensoes = stats.pearsonr(flat_prisoes, flat_apreensoes)

print(f"   CVLI ↔ Prisões: r={corr_cvli_prisoes:.3f} (p={p_cvli_prisoes:.4f})")
print(f"   CVLI ↔ Apreensões: r={corr_cvli_apreensoes:.3f} (p={p_cvli_apreensoes:.4f})")
print(f"   Prisões ↔ Apreensões: r={corr_prisoes_apreensoes:.3f} (p={p_prisoes_apreensoes:.4f})")

# 7. Score geral
print(f"\n   🎯 CÁLCULO DE VIABILIDADE GERAL\n")

avg_feature_score = np.mean(list(scores.values()))

# Bônus por correlação
corr_bonus = 0
if abs(corr_cvli_prisoes) > 0.1:
    corr_bonus += 10 * abs(corr_cvli_prisoes)
if abs(corr_cvli_apreensoes) > 0.1:
    corr_bonus += 10 * abs(corr_cvli_apreensoes)

overall_score = min(100, avg_feature_score + corr_bonus)

print(f"   Score médio features: {avg_feature_score:.1f}/100")
print(f"   Bônus correlação: +{corr_bonus:.1f}")
print(f"   GERAL: {overall_score:.1f}/100")

# 8. Salvar tensor
print(f"\n[6] Salvando tensor e metadados...")
tensor_combined = np.stack([matrix_cvli, matrix_prisoes, matrix_apreensoes], axis=2)
tensor_path = OUTPUT_DIR / "tensor_cvli_prisoes_CORRIGIDO.npy"
np.save(tensor_path, tensor_combined)
print(f"   ✅ Tensor: {tensor_path}")

# 9. Gerar relatório
print(f"\n[7] Gerando relatório...")

report = f"""# ANÁLISE 2 (CORRIGIDA): VIABILIDADE ST-GCN CVLI + CONTEXTO OPERACIONAL

## 📊 Resumo Executivo

**Dataset CVLI:** outputs/cvli_with_bairro.csv (tipo='cvli', {len(df_cvli):,} eventos)
**Dataset Operacional:** data/raw/View_Ocorrencias_Operacionais_Modelo.csv (prisões + apreensões)
**Período:** {date_min.date()} a {date_max.date()} ({T} dias)
**Cobertura geográfica:** {N} bairros normalizados
**Dimensões do tensor:** T={T} × N={N} × F=3 → {T*N*3:,} células

### Features Utilizadas

1. **CVLI** ({int(matrix_cvli.sum()):,} eventos - Score: {scores.get('CVLI', 0):.1f}/100)
   - Homicídios e latrocínios (tipo=cvli)
   
2. **Prisões** ({int(matrix_prisoes.sum()):,} eventos - Score: {scores.get('Prisões', 0):.1f}/100)
   - Operações com prisões/mandados
   
3. **Apreensões** ({int(matrix_apreensoes.sum()):,} eventos - Score: {scores.get('Apreensões', 0):.1f}/100)
   - Drogas, armas e/ou dinheiro apreendido

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
    report += "ST-GCN com contexto operacional é recomendado.\n"
elif overall_score >= 60:
    report += f"**🟡 VIÁVEL** - Score {overall_score:.1f}/100\n"
    report += "ST-GCN pode ser implementado com performance aceitável.\n"
else:
    report += f"**🔴 NÃO RECOMENDADO** - Score {overall_score:.1f}/100\n"
    report += "Dataset insuficiente.\n"

# Carregar Análise 1 para comparação
try:
    with open("teste_modelo/01_apenas_ocorrencias/metadata_analise_1_CORRIGIDA.json", 'r') as f:
        metadata_1 = json.load(f)
    score_1 = metadata_1['score']['geral']
    improvement = overall_score - score_1
    improvement_pct = (improvement / score_1) * 100 if score_1 > 0 else 0
    
    report += f"""
## 📊 Comparação: CVLI-only vs CVLI+Contexto

| Aspecto | CVLI-only | CVLI+Contexto | Diferença |
|---------|-----------|---------------|-----------|
| **Score Geral** | {score_1:.1f}/100 | {overall_score:.1f}/100 | **{improvement:+.1f}** ({improvement_pct:+.1f}%) |

### Conclusão
"""
    if overall_score > score_1 + 5:
        report += f"**Análise 2 é significativamente superior.** Recomenda-se usar contexto operacional."
    elif overall_score > score_1:
        report += f"**Análise 2 mostra melhoria.** Ambas abordagens são viáveis."
    else:
        report += f"**Análise 1 é superior.** CVLI-only é mais adequado."
except:
    pass

report += f"""

---
**Data de geração:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Arquivo tensor:** tensor_cvli_prisoes_CORRIGIDO.npy ({T} × {N} × 3)
**Fonte:** cvli_with_bairro.csv + operacional_modelo.csv
"""

report_path = OUTPUT_DIR / "RELATORIO_ANALISE_2_CORRIGIDA.md"
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(report)
print(f"   ✅ Relatório: {report_path}")

# 10. Salvar metadados
metadata = {
    'fonte_cvli': 'outputs/cvli_with_bairro.csv (tipo=cvli)',
    'fonte_operacional': 'data/raw/View_Ocorrencias_Operacionais_Modelo.csv',
    'periodo': f"{date_min.date()} a {date_max.date()}",
    'total_dias': int(T),
    'total_bairros': int(N),
    'eventos_cvli': int(matrix_cvli.sum()),
    'eventos_prisoes': int(matrix_prisoes.sum()),
    'eventos_apreensoes': int(matrix_apreensoes.sum()),
    'tensor_shape': [int(T), int(N), 3],
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

metadata_path = OUTPUT_DIR / "metadata_analise_2_CORRIGIDA.json"
with open(metadata_path, 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=2)
print(f"   ✅ Metadados: {metadata_path}")

print("\n" + "="*80)
print(f"✅ ANÁLISE 2 (CORRIGIDA) CONCLUÍDA - Score: {overall_score:.1f}/100")
print("="*80)
