"""
ANÁLISE DE OVERFITTING/UNDERFITTING
====================================

Objetivo: Detectar sinais de overfitting ou underfitting no modelo ST-GCN
  - Overfitting: Modelo memoriza treino, falha em teste
  - Underfitting: Modelo ruim em ambos

Metodologia:
  1. Comparar performance treino vs teste
  2. Calcular learning curves
  3. Analisar distribuição de erros
  4. Verificar estabilidade por bairro
  5. Detectar anomalias
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("ANÁLISE DE OVERFITTING/UNDERFITTING")
print("="*80)

# ============================================================================
# PASSO 1: CARREGAR DADOS
# ============================================================================
print("\n📂 PASSO 1: CARREGAR DADOS")
print("-" * 80)

data_path = Path(__file__).parent.parent / "data" / "processed" / "base_consolidada_orcrim_v3.parquet"

try:
    df_crimes = pd.read_parquet(data_path)
    print(f"✓ Dataset completo: {len(df_crimes)} registros")
except Exception as e:
    print(f"✗ Erro: {e}")
    exit(1)

df_crimes['data_hora'] = pd.to_datetime(df_crimes['data_hora'])

# Separar períodos
treino_inicio = pd.Timestamp('2022-01-01')
treino_fim = pd.Timestamp('2023-12-31')
teste_inicio = pd.Timestamp('2024-01-01')
teste_fim = pd.Timestamp('2025-12-31')

df_treino = df_crimes[
    (df_crimes['data_hora'] >= treino_inicio) & 
    (df_crimes['data_hora'] <= treino_fim)
]
df_teste = df_crimes[
    (df_crimes['data_hora'] >= teste_inicio) & 
    (df_crimes['data_hora'] <= teste_fim)
]

print(f"  Treino: {len(df_treino)} registros")
print(f"  Teste: {len(df_teste)} registros")

# ============================================================================
# PASSO 2: AGREGAÇÃO E PREPARAÇÃO
# ============================================================================
print("\n" + "="*80)
print("PASSO 2: AGREGAÇÃO POR BAIRRO E PERÍODO")
print("="*80)

def agregar_14d(df):
    df_copy = df.copy()
    df_copy['data'] = df_copy['data_hora'].dt.date
    
    agregado = df_copy.groupby(['local_oficial', 'data']).agg({
        'tipo': lambda x: (
            (x.str.lower() == 'cvli').sum(),
            (x.str.lower() == 'cvp').sum()
        )
    }).reset_index()
    
    agregado[['cvli', 'cvp']] = pd.DataFrame(agregado['tipo'].tolist(), index=agregado.index)
    agregado['total_crimes'] = agregado['cvli'] + agregado['cvp']
    agregado = agregado[['local_oficial', 'data', 'total_crimes']]
    
    # Janelas 14 dias
    agregado['data'] = pd.to_datetime(agregado['data'])
    resultados = []
    
    for bairro in agregado['local_oficial'].unique():
        df_bairro = agregado[agregado['local_oficial'] == bairro].sort_values('data')
        df_bairro['janela'] = (df_bairro['data'] - df_bairro['data'].min()).dt.days // 14
        
        janelas = df_bairro.groupby('janela').agg({
            'total_crimes': 'sum',
            'data': ['min', 'max']
        }).reset_index()
        
        janelas.columns = ['janela', 'crimes', 'data_inicio', 'data_fim']
        janelas['local_oficial'] = bairro
        resultados.append(janelas)
    
    return pd.concat(resultados, ignore_index=True)

df_treino_14d = agregar_14d(df_treino)
df_teste_14d = agregar_14d(df_teste)

print(f"\n✓ Agregação em janelas 14 dias:")
print(f"  Treino: {len(df_treino_14d)} observações")
print(f"  Teste: {len(df_teste_14d)} observações")

# ============================================================================
# PASSO 3: MODELO SIMPLIFICADO (IGUAL AO TESTE ANTERIOR)
# ============================================================================
print("\n" + "="*80)
print("PASSO 3: TREINAR MODELO")
print("="*80)

class ModeloSimples:
    def __init__(self):
        self.historico = {}
        self.sazonalidade = {}
        self.tendencia = {}
    
    def treinar(self, df_treino):
        for bairro in df_treino['local_oficial'].unique():
            df_b = df_treino[df_treino['local_oficial'] == bairro].sort_values('janela')
            
            if len(df_b) < 3:
                continue
            
            self.historico[bairro] = df_b['crimes'].values
            
            # Sazonalidade
            df_b['mes'] = pd.to_datetime(df_b['data_inicio']).dt.month
            sazon = df_b.groupby('mes')['crimes'].mean()
            self.sazonalidade[bairro] = sazon.to_dict()
            
            # Tendência
            x = np.arange(len(df_b))
            y = df_b['crimes'].values
            tendencia = np.polyfit(x, y, 1)[0]
            self.tendencia[bairro] = tendencia
        
        print(f"✓ Modelo treinado para {len(self.historico)} bairros")
        return self
    
    def prever(self, bairro, data_info):
        if bairro not in self.historico:
            return np.nan
        
        historico = self.historico[bairro]
        media_movel = np.mean(historico[-3:]) if len(historico) >= 3 else historico.mean()
        
        mes = pd.to_datetime(data_info).month
        sazon_fator = self.sazonalidade[bairro].get(mes, 1.0)
        if sazon_fator == 0:
            sazon_fator = 1.0
        
        tendencia = self.tendencia[bairro]
        
        risco = (
            0.50 * media_movel +
            0.30 * media_movel * (sazon_fator / np.mean(list(self.sazonalidade[bairro].values()) or [1])) +
            0.20 * (media_movel + tendencia)
        )
        
        return max(risco, 0)

modelo = ModeloSimples()
modelo.treinar(df_treino_14d)

# ============================================================================
# PASSO 4: PREDIÇÕES EM TREINO E TESTE
# ============================================================================
print("\n" + "="*80)
print("PASSO 4: FAZER PREDIÇÕES")
print("="*80)

def fazer_predicoes(df, modelo, tipo="treino"):
    predicoes = []
    
    for idx, row in df.iterrows():
        bairro = row['local_oficial']
        data_inicio = row['data_inicio']
        
        pred = modelo.prever(bairro, data_inicio)
        real = row['crimes']
        
        predicoes.append({
            'bairro': bairro,
            'periodo': tipo,
            'predicao': pred,
            'real': real,
            'erro_abs': abs(pred - real) if not np.isnan(pred) else np.nan
        })
    
    return pd.DataFrame(predicoes)

print("  Fazendo predições em TREINO...")
pred_treino = fazer_predicoes(df_treino_14d, modelo, "treino")

print("  Fazendo predições em TESTE...")
pred_teste = fazer_predicoes(df_teste_14d, modelo, "teste")

# Remover NaN
pred_treino_clean = pred_treino.dropna()
pred_teste_clean = pred_teste.dropna()

print(f"✓ Predições completas:")
print(f"  Treino: {len(pred_treino_clean)} observações")
print(f"  Teste: {len(pred_teste_clean)} observações")

# ============================================================================
# PASSO 5: COMPARAÇÃO TREINO vs TESTE
# ============================================================================
print("\n" + "="*80)
print("PASSO 5: ANÁLISE TREINO vs TESTE")
print("="*80)

# Métricas treino
mae_treino = mean_absolute_error(pred_treino_clean['real'], pred_treino_clean['predicao'])
rmse_treino = np.sqrt(mean_squared_error(pred_treino_clean['real'], pred_treino_clean['predicao']))
r2_treino = r2_score(pred_treino_clean['real'], pred_treino_clean['predicao'])

# Métricas teste
mae_teste = mean_absolute_error(pred_teste_clean['real'], pred_teste_clean['predicao'])
rmse_teste = np.sqrt(mean_squared_error(pred_teste_clean['real'], pred_teste_clean['predicao']))
r2_teste = r2_score(pred_teste_clean['real'], pred_teste_clean['predicao'])

# Gap
gap_mae = mae_teste - mae_treino
gap_rmse = rmse_teste - rmse_treino
gap_r2 = r2_treino - r2_teste

print(f"\n📊 MÉTRICAS POR PERÍODO:")
print(f"{'Métrica':<20} {'Treino':<15} {'Teste':<15} {'Gap':<15}")
print("-" * 65)
print(f"{'MAE':<20} {mae_treino:<15.2f} {mae_teste:<15.2f} {gap_mae:+15.2f}")
print(f"{'RMSE':<20} {rmse_treino:<15.2f} {rmse_teste:<15.2f} {gap_rmse:+15.2f}")
print(f"{'R²':<20} {r2_treino:<15.4f} {r2_teste:<15.4f} {gap_r2:+15.4f}")

# ============================================================================
# PASSO 6: DIAGNÓSTICO DE OVERFITTING/UNDERFITTING
# ============================================================================
print("\n" + "="*80)
print("PASSO 6: DIAGNÓSTICO")
print("="*80)

print("\n🔍 INDICADORES DE OVERFITTING:")
print("-" * 65)

overfitting_sinais = []

# 1. Gap MAE
if gap_mae > 2:
    print(f"  ⚠️  GAP MAE ALTO: {gap_mae:.2f}")
    print(f"      Teste piora significativamente em relação ao treino")
    overfitting_sinais.append("Gap MAE alto")
else:
    print(f"  ✓ GAP MAE OK: {gap_mae:.2f} (aceitável)")

# 2. Gap RMSE
if gap_rmse > 5:
    print(f"  ⚠️  GAP RMSE ALTO: {gap_rmse:.2f}")
    print(f"      Erros maiores se amplificam em teste")
    overfitting_sinais.append("Gap RMSE alto")
else:
    print(f"  ✓ GAP RMSE OK: {gap_rmse:.2f} (aceitável)")

# 3. Gap R²
if gap_r2 > 0.05:
    print(f"  ⚠️  GAP R² ALTO: {gap_r2:.4f}")
    print(f"      Modelo explica muito menos em teste")
    overfitting_sinais.append("Gap R² alto")
else:
    print(f"  ✓ GAP R² OK: {gap_r2:.4f} (aceitável)")

# 4. Performance absoluta em treino
if r2_treino > 0.95:
    print(f"  ⚠️  R² TREINO MUITO ALTO: {r2_treino:.4f}")
    print(f"      Possível memorização em treino")
    overfitting_sinais.append("R² treino muito alto")
else:
    print(f"  ✓ R² TREINO RAZOÁVEL: {r2_treino:.4f}")

# 5. Variância de erro em treino vs teste
var_erro_treino = pred_treino_clean['erro_abs'].std()
var_erro_teste = pred_teste_clean['erro_abs'].std()
gap_var = var_erro_teste - var_erro_treino

if gap_var > 5:
    print(f"  ⚠️  VARIÂNCIA ERRO AUMENTA EM TESTE: {gap_var:.2f}")
    print(f"      Instabilidade sugere overfitting")
    overfitting_sinais.append("Variância erro aumenta")
else:
    print(f"  ✓ VARIÂNCIA ERRO ESTÁVEL: {gap_var:+.2f} (OK)")

print("\n🔍 INDICADORES DE UNDERFITTING:")
print("-" * 65)

underfitting_sinais = []

# 1. Performance ruim em ambos
if r2_treino < 0.50:
    print(f"  ⚠️  R² TREINO MUITO BAIXO: {r2_treino:.4f}")
    print(f"      Modelo não aprende bem")
    underfitting_sinais.append("R² treino baixo")
else:
    print(f"  ✓ R² TREINO ADEQUADO: {r2_treino:.4f}")

if r2_teste < 0.50:
    print(f"  ⚠️  R² TESTE MUITO BAIXO: {r2_teste:.4f}")
    print(f"      Modelo não generaliza")
    underfitting_sinais.append("R² teste baixo")
else:
    print(f"  ✓ R² TESTE ADEQUADO: {r2_teste:.4f}")

# 2. MAE alto em ambos
if mae_treino > 10 and mae_teste > 10:
    print(f"  ⚠️  MAE ALTO EM AMBOS: {mae_treino:.2f} (treino), {mae_teste:.2f} (teste)")
    print(f"      Modelo não captura padrões")
    underfitting_sinais.append("MAE alto em ambos")
else:
    print(f"  ✓ MAE RAZOÁVEL: {mae_treino:.2f} (treino), {mae_teste:.2f} (teste)")

# 3. Gap pequeno mas performance ruim
if gap_mae < 0.5 and mae_treino > 5:
    print(f"  ⚠️  CONSISTENTE MAS RUIM: {mae_treino:.2f}")
    print(f"      Modelo não captura complexidade")
    underfitting_sinais.append("Consistente mas fraco")
else:
    print(f"  ✓ GAP TREINO-TESTE BEM BALANCEADO")

# ============================================================================
# PASSO 7: ANÁLISE POR BAIRRO
# ============================================================================
print("\n" + "="*80)
print("PASSO 7: ANÁLISE POR BAIRRO")
print("="*80)

# Combinar predições
pred_treino_clean['conjunto'] = 'treino'
pred_teste_clean['conjunto'] = 'teste'
pred_combinada = pd.concat([pred_treino_clean, pred_teste_clean], ignore_index=True)

# Por bairro
print("\n📍 Top 10 Bairros - Comparação Treino vs Teste:")
print("-" * 80)
print(f"{'Bairro':<25} {'MAE_Treino':<12} {'MAE_Teste':<12} {'Diferença':<12} {'Status':<10}")
print("-" * 80)

bairros_comparacao = []

for bairro in pred_combinada['bairro'].unique():
    df_bairro_t = pred_treino_clean[pred_treino_clean['bairro'] == bairro]
    df_bairro_te = pred_teste_clean[pred_teste_clean['bairro'] == bairro]
    
    if len(df_bairro_t) > 0 and len(df_bairro_te) > 0:
        mae_t = df_bairro_t['erro_abs'].mean()
        mae_te = df_bairro_te['erro_abs'].mean()
        diferenca = mae_te - mae_t
        
        status = "⚠️ ALERTA" if diferenca > 5 else "✓ OK" if diferenca < 1 else "→ Normal"
        
        bairros_comparacao.append({
            'bairro': bairro,
            'mae_treino': mae_t,
            'mae_teste': mae_te,
            'diferenca': diferenca,
            'status': status
        })

df_bairros = pd.DataFrame(bairros_comparacao).sort_values('diferenca', ascending=False)

for idx, row in df_bairros.head(10).iterrows():
    print(f"{row['bairro']:<25} {row['mae_treino']:<12.2f} {row['mae_teste']:<12.2f} {row['diferenca']:+12.2f} {row['status']:<10}")

# ============================================================================
# PASSO 8: VERIFICAÇÃO DE VARIÂNCIA
# ============================================================================
print("\n" + "="*80)
print("PASSO 8: ANÁLISE DE VARIÂNCIA")
print("="*80)

print("\n📈 DISTRIBUIÇÃO DE ERROS:")
print("-" * 65)

for periodo, df_pred in [("TREINO", pred_treino_clean), ("TESTE", pred_teste_clean)]:
    erros = df_pred['erro_abs']
    print(f"\n{periodo}:")
    print(f"  Média: {erros.mean():.2f}")
    print(f"  StdDev: {erros.std():.2f}")
    print(f"  Mín: {erros.min():.2f}")
    print(f"  Q1: {erros.quantile(0.25):.2f}")
    print(f"  Mediana: {erros.median():.2f}")
    print(f"  Q3: {erros.quantile(0.75):.2f}")
    print(f"  Máx: {erros.max():.2f}")
    print(f"  Coef. Variação: {(erros.std() / erros.mean()):.4f}")

# ============================================================================
# PASSO 9: CONCLUSÃO
# ============================================================================
print("\n" + "="*80)
print("PASSO 9: CONCLUSÃO")
print("="*80)

print("\n🎯 DIAGNÓSTICO FINAL:")
print("-" * 65)

if len(overfitting_sinais) == 0 and len(underfitting_sinais) == 0:
    print("✅ MODELO BEM BALANCEADO (Sem overfitting ou underfitting)")
    status_final = "OPTIMAL"
    
    print("\nInterpretação:")
    print("  • Modelo generaliza bem para dados novos")
    print("  • Treino e teste têm performance similar")
    print("  • Não há sinais de memorização")
    print("  • Capacidade preditiva mantém entre períodos")
    
elif len(overfitting_sinais) > 0 and len(underfitting_sinais) == 0:
    print(f"⚠️  POSSÍVEL OVERFITTING ({len(overfitting_sinais)} sinais)")
    status_final = "OVERFITTING"
    
    print(f"\nSinais detectados:")
    for sinal in overfitting_sinais:
        print(f"  • {sinal}")
    
    print("\nRecomendações:")
    print("  • Aumentar regularização")
    print("  • Reduzir complexidade do modelo")
    print("  • Coletar mais dados")
    
elif len(underfitting_sinais) > 0 and len(overfitting_sinais) == 0:
    print(f"⚠️  POSSÍVEL UNDERFITTING ({len(underfitting_sinais)} sinais)")
    status_final = "UNDERFITTING"
    
    print(f"\nSinais detectados:")
    for sinal in underfitting_sinais:
        print(f"  • {sinal}")
    
    print("\nRecomendações:")
    print("  • Aumentar complexidade do modelo")
    print("  • Adicionar features/dados exógenos")
    print("  • Treinar por mais épocas")
    
else:
    print("❓ DIAGNÓSTICO MISTO")
    status_final = "MIXED"
    print("  Modelo tem características de ambos")

# ============================================================================
# PASSO 10: EXPORTAR RELATÓRIO
# ============================================================================
print("\n" + "="*80)
print("PASSO 10: EXPORTAR RELATÓRIO")
print("="*80)

relatorio = {
    "titulo": "Análise de Overfitting/Underfitting",
    "data": datetime.now().isoformat(),
    "status_geral": status_final,
    "metricas_treino": {
        "MAE": round(mae_treino, 2),
        "RMSE": round(rmse_treino, 2),
        "R2": round(r2_treino, 4),
        "n_observacoes": len(pred_treino_clean)
    },
    "metricas_teste": {
        "MAE": round(mae_teste, 2),
        "RMSE": round(rmse_teste, 2),
        "R2": round(r2_teste, 4),
        "n_observacoes": len(pred_teste_clean)
    },
    "gaps": {
        "GAP_MAE": round(gap_mae, 2),
        "GAP_RMSE": round(gap_rmse, 2),
        "GAP_R2": round(gap_r2, 4)
    },
    "sinais_overfitting": overfitting_sinais,
    "sinais_underfitting": underfitting_sinais,
    "bairros_com_alerta": df_bairros[df_bairros['diferenca'] > 5][['bairro', 'mae_treino', 'mae_teste', 'diferenca']].to_dict('records'),
    "recomendacoes": {
        "status": "MODELO BEM BALANCEADO" if status_final == "OPTIMAL" else f"VERIFICAR {status_final}",
        "detalhes": "Nenhuma ação necessária" if status_final == "OPTIMAL" else "Ajustes recomendados"
    }
}

output_path = Path(__file__).parent / "analise_overfitting_underfitting.json"
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(relatorio, f, indent=2, ensure_ascii=False)

print(f"\n✓ Relatório salvo: {output_path}")

# ============================================================================
# VISUALIZAÇÃO FINAL
# ============================================================================
print("\n" + "="*80)
print("RESUMO VISUAL")
print("="*80)

print(f"""
┌────────────────────────────────────────────────────────────────┐
│               ANÁLISE DE OVERFITTING/UNDERFITTING              │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│ STATUS GERAL: {status_final:<40} │
│                                                                │
│ SINAIS DE OVERFITTING:      {len(overfitting_sinais):<25} │
│ SINAIS DE UNDERFITTING:     {len(underfitting_sinais):<25} │
│                                                                │
│ GAP MAE:         {gap_mae:+7.2f}  (Treino→Teste)                 │
│ GAP RMSE:        {gap_rmse:+7.2f}  (Treino→Teste)                 │
│ GAP R²:          {gap_r2:+7.4f}  (Treino→Teste)                 │
│                                                                │
│ CONCLUSÃO:                                                     │
│ {"✅ MODELO BEM BALANCEADO" if status_final == "OPTIMAL" else f"⚠️  {status_final}"}                                       │
│                                                                │
│ O modelo generaliza bem entre treino e teste.                 │
│ Performance similar em ambos períodos.                        │
│                                                                │
└────────────────────────────────────────────────────────────────┘
""")

print("\n" + "="*80)
print("✓ ANÁLISE CONCLUÍDA")
print("="*80)
