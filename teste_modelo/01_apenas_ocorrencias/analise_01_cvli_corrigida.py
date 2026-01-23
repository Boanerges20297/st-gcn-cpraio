"""
ANÁLISE 1 (CORRIGIDA): Viabilidade ST-GCN com CVLI-only (dados do IBGE com lat/long)
Dataset: outputs/cvli_with_bairro.csv filtrado para tipo='cvli' desde 2022
Features: Apenas CVLI (homicídios/latrocínios)
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy import stats

OUTPUT_DIR = Path("teste_modelo/01_apenas_ocorrencias")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

DATA_FILE = Path("outputs/cvli_with_bairro.csv")

print("\n" + "="*80)
print("ANÁLISE 1 (CORRIGIDA): VIABILIDADE ST-GCN CVLI-ONLY (TIPO=CVLI)")
print("="*80)

# 1. Carregar dados CVLI
print("\n[1] Carregando dados CVLI...")
df = pd.read_csv(DATA_FILE, low_memory=False)
df['data'] = pd.to_datetime(df['data'])

# Filtrar apenas CVLI desde 2022
df = df[(df['data'].dt.year >= 2022) & (df['tipo'].str.lower() == 'cvli')].copy()
print(f"   ✓ {len(df):,} eventos CVLI (2022-2026)")
print(f"   ✓ Período: {df['data'].min().date()} a {df['data'].max().date()}")

# 2. Remover registros sem coordenadas
print("\n[2] Filtrando registros com coordenadas...")
df = df.dropna(subset=['latitude', 'longitude'])
print(f"   ✓ {len(df):,} eventos CVLI com coordenadas ({(len(df)/df.shape[0]*100):.1f}%)")

# 3. Usar bairro_assigned (normalizado pelo projeto)
print("\n[3] Preparando dados geográficos...")
df_geo = df.dropna(subset=['bairro_assigned']).copy()
print(f"   ✓ {len(df_geo):,} eventos com bairro normalizado ({(len(df_geo)/len(df)*100):.1f}%)")
print(f"   ✓ Bairros únicos: {df_geo['bairro_assigned'].nunique()}")

# 4. Construir matriz espacial (T × N)
print("\n[4] Construindo matriz temporal...")
dates = pd.date_range(df_geo['data'].min(), df_geo['data'].max(), freq='D')
T = len(dates)

# Agrupar bairros
bairros = df_geo['bairro_assigned'].unique()
N = len(bairros)
bairro_to_idx = {b: i for i, b in enumerate(bairros)}

print(f"   ✓ Período: {T} dias")
print(f"   ✓ Bairros: {N}")
print(f"   ✓ Dimensões: {T} × {N} = {T*N:,} células")

# Construir matriz de contagem
matrix = np.zeros((T, N))
daily_totals = np.zeros(T)

for idx, row in df_geo.iterrows():
    t_idx = (row['data'].date() - dates[0].date()).days
    if 0 <= t_idx < T:
        n_idx = bairro_to_idx[row['bairro_assigned']]
        matrix[t_idx, n_idx] += 1
        daily_totals[t_idx] += 1

print(f"   ✓ Matriz construída")

# 5. Calcular métricas
print("\n[5] Calculando métricas de viabilidade...")

# Esparsidade
nonzero = np.count_nonzero(matrix)
sparsity = 1 - (nonzero / matrix.size)
print(f"\n   📊 ESPARSIDADE")
print(f"      Células não-vazias: {nonzero:,}/{matrix.size:,} ({(nonzero/matrix.size)*100:.2f}%)")
print(f"      Esparsidade: {sparsity*100:.2f}%")

# Sinal (intensidade média)
signal_mean = matrix.sum() / (T * N)
signal_nz = matrix[matrix > 0].mean() if nonzero > 0 else 0
print(f"\n   🔊 SINAL (eventos/dia/bairro)")
print(f"      Média geral: {signal_mean:.6f}")
print(f"      Média (apenas células com evento): {signal_nz:.4f}")
if nonzero > 0:
    p50 = np.percentile(matrix[matrix > 0], 50)
    p75 = np.percentile(matrix[matrix > 0], 75)
    p90 = np.percentile(matrix[matrix > 0], 90)
    print(f"      Distribuição (células com evento):")
    print(f"         P50: {p50:.2f}, P75: {p75:.2f}, P90: {p90:.2f}")

# Variabilidade temporal (coeficiente de variação)
valid_days = daily_totals[daily_totals > 0]
if len(valid_days) > 0:
    cv = valid_days.std() / valid_days.mean() if valid_days.mean() > 0 else 0
    print(f"\n   📈 VARIABILIDADE TEMPORAL")
    print(f"      Dias com eventos: {len(valid_days)}/{T} ({(len(valid_days)/T)*100:.1f}%)")
    print(f"      Média (dias com evento): {valid_days.mean():.2f}")
    print(f"      Std: {valid_days.std():.2f}")
    print(f"      CV: {cv:.3f}")
    
    # Autocorrelação temporal
    if len(valid_days) > 2:
        acf_lag1 = np.corrcoef(daily_totals[:-1], daily_totals[1:])[0, 1]
        acf_lag1_str = f"{acf_lag1:.3f}"
        print(f"      Autocorrelação Lag-1: {acf_lag1_str}")
    else:
        acf_lag1_str = "N/A"
else:
    cv = 0
    acf_lag1_str = "N/A"
    print(f"\n   📈 VARIABILIDADE TEMPORAL: SEM EVENTOS")

# 6. Calcular viabilidade
print(f"\n   🎯 CÁLCULO DE VIABILIDADE")

# Score de esparsidade: alta esparsidade = ruim
score_sparsity = max(0, 100 - sparsity * 150)

# Score de sinal: quanto maior o sinal não-zero, melhor
score_signal = min(100, signal_nz * 1000)

# Score de variabilidade: CV muito alto é ruim (padrão inconsistente)
score_variability = max(0, 100 - abs(cv - 0.5) * 50)

# Score de cobertura: % de bairros com pelo menos 1 evento
bairros_with_events = np.sum(matrix.sum(axis=0) > 0)
score_coverage = (bairros_with_events / N) * 100

print(f"      Esparsidade: {score_sparsity:.1f}/100")
print(f"      Sinal (intensidade): {score_signal:.1f}/100")
print(f"      Variabilidade temporal: {score_variability:.1f}/100")
print(f"      Cobertura espacial: {score_coverage:.1f}/100")

overall_score = (score_sparsity * 0.25 + score_signal * 0.35 + 
                score_variability * 0.20 + score_coverage * 0.20)
print(f"      GERAL: {overall_score:.1f}/100")

# 7. Salvar tensor
print(f"\n[6] Salvando tensor e metadados...")
tensor_path = OUTPUT_DIR / "tensor_cvli_only_CORRIGIDO.npy"
np.save(tensor_path, matrix)
print(f"   ✅ Tensor: {tensor_path}")

# 8. Gerar relatório
print(f"\n[7] Gerando relatório...")

report = f"""# ANÁLISE 1 (CORRIGIDA): VIABILIDADE ST-GCN CVLI-ONLY

## 📊 Resumo Executivo

**Dataset:** outputs/cvli_with_bairro.csv filtrado (tipo='cvli')
**Período:** {df_geo['data'].min().date()} a {df_geo['data'].max().date()} ({T} dias)
**Eventos CVLI:** {len(df_geo):,} 
**Cobertura geográfica:** {N} bairros normalizados (Fortaleza + RMF + Interior)
**Dimensões do tensor:** T={T} × N={N} → {T*N:,} células

## 📈 Métricas de Qualidade

### Esparsidade
- **Células não-vazias:** {nonzero:,}/{matrix.size:,} ({(nonzero/matrix.size)*100:.2f}%)
- **Esparsidade:** {sparsity*100:.2f}%
- **Avaliação:** {"✅ ÓTIMA" if sparsity < 0.5 else "🟡 BOM" if sparsity < 0.8 else "⚠️ CRÍTICO"}

### Sinal Temporal
- **Intensidade média (todas as células):** {signal_mean:.6f} eventos/dia/bairro
- **Intensidade média (apenas células com evento):** {signal_nz:.4f}
- **Avaliação:** {"✅ FORTE" if signal_nz > 0.1 else "🟡 MÉDIO" if signal_nz > 0.01 else "⚠️ FRACO"}

### Variabilidade Temporal
- **Dias com eventos:** {len(valid_days)}/{T} ({(len(valid_days)/T)*100:.1f}%)
- **Coeficiente de Variação:** {cv:.3f}
- **Autocorrelação (Lag-1):** {acf_lag1_str}
- **Avaliação:** {"✅ PREVISÍVEL" if cv < 1.0 else "🟡 MODERADO" if cv < 3.0 else "⚠️ CAÓTICO"}

### Cobertura Espacial
- **Bairros com eventos:** {bairros_with_events}/{N} ({score_coverage:.1f}%)
- **Avaliação:** {"✅ EXCELENTE" if score_coverage > 80 else "🟡 BOM" if score_coverage > 50 else "⚠️ FRACO"}

## 🎯 Viabilidade ST-GCN

### Scoring (0-100)
| Aspecto | Score | Peso | Contribuição |
|---------|-------|------|--------------|
| Esparsidade | {score_sparsity:.1f} | 25% | {score_sparsity*0.25:.1f} |
| Sinal (intensidade) | {score_signal:.1f} | 35% | {score_signal*0.35:.1f} |
| Variabilidade | {score_variability:.1f} | 20% | {score_variability*0.20:.1f} |
| Cobertura | {score_coverage:.1f} | 20% | {score_coverage*0.20:.1f} |
| **GERAL** | **{overall_score:.1f}** | 100% | **{overall_score:.1f}** |

### Recomendação
"""

if overall_score >= 75:
    report += f"**🟢 ALTAMENTE VIÁVEL** - Score {overall_score:.1f}/100\n"
    report += "ST-GCN é recomendado para implementação.\n"
elif overall_score >= 60:
    report += f"**🟡 VIÁVEL** - Score {overall_score:.1f}/100\n"
    report += "ST-GCN pode funcionar com performance aceitável.\n"
else:
    report += f"**🔴 NÃO RECOMENDADO** - Score {overall_score:.1f}/100\n"
    report += "Dataset insuficiente. Considere enriquecer com features de contexto.\n"

report += f"""
## 📋 Próximos Passos

1. Comparar com Análise 2 (CVLI + Contexto Operacional)
2. Se score >= 60: Proceder com implementação
3. Validação cruzada com holdout (últimos 30 dias)
4. Tuning de hyperparâmetros do ST-GCN

---
**Data de geração:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Arquivo tensor:** tensor_cvli_only_CORRIGIDO.npy ({T} × {N})
**Fonte:** outputs/cvli_with_bairro.csv (tipo='cvli')
"""

report_path = OUTPUT_DIR / "RELATORIO_ANALISE_1_CORRIGIDA.md"
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(report)
print(f"   ✅ Relatório: {report_path}")

# 9. Salvar metadados
metadata = {
    'fonte': 'outputs/cvli_with_bairro.csv (tipo=cvli)',
    'periodo': f"{df_geo['data'].min().date()} a {df_geo['data'].max().date()}",
    'total_dias': int(T),
    'total_bairros': int(N),
    'eventos_cvli': int(len(df_geo)),
    'tensor_shape': [int(T), int(N)],
    'tensor_cells': int(T*N),
    'cells_nonzero': int(nonzero),
    'sparsity': float(sparsity),
    'signal_mean': float(signal_mean),
    'signal_nz_mean': float(signal_nz),
    'cv': float(cv),
    'score': {
        'sparsidade': float(score_sparsity),
        'sinal': float(score_signal),
        'variabilidade': float(score_variability),
        'cobertura': float(score_coverage),
        'geral': float(overall_score)
    }
}

metadata_path = OUTPUT_DIR / "metadata_analise_1_CORRIGIDA.json"
with open(metadata_path, 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=2)
print(f"   ✅ Metadados: {metadata_path}")

print("\n" + "="*80)
print(f"✅ ANÁLISE 1 (CORRIGIDA) CONCLUÍDA - Score: {overall_score:.1f}/100")
print(f"   Fonte: {len(df_geo):,} eventos CVLI (tipo=cvli)")
print("="*80)
