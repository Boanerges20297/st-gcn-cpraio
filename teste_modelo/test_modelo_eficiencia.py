"""
TESTE DE EFICIÊNCIA DO MODELO ST-GCN
=====================================

Objetivo: Avaliar a capacidade preditiva do modelo
  - Treino: dados 2022-2023
  - Teste: dados 2024-2025 (gabarito real)
  - Métricas: MSE, MAE, RMSE, R², Acurácia de Ação

Este script NÃO requer o modelo já treinado.
Executa treinamento iterativo e valida em tempo real.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import json
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("TESTE DE EFICIÊNCIA: ST-GCN (2022-2023 Treino | 2024-2025 Teste)")
print("="*80)

# ============================================================================
# PASSO 1: CARREGAR E SEPARAR DADOS
# ============================================================================
print("\n📂 PASSO 1: CARREGAR E SEPARAR DADOS")
print("-" * 80)

data_path = Path(__file__).parent.parent / "data" / "processed" / "base_consolidada_orcrim_v3.parquet"

try:
    df_crimes = pd.read_parquet(data_path)
    print(f"✓ Dataset completo: {len(df_crimes)} registros")
    print(f"  Período: {df_crimes['data_hora'].min()} até {df_crimes['data_hora'].max()}")
except Exception as e:
    print(f"✗ Erro ao carregar: {e}")
    exit(1)

# Converter para datetime se necessário
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

print(f"\n📊 Divisão de dados:")
print(f"  TREINO (2022-2023): {len(df_treino)} registros ({len(df_treino)/len(df_crimes)*100:.1f}%)")
print(f"  TESTE  (2024-2025): {len(df_teste)} registros ({len(df_teste)/len(df_crimes)*100:.1f}%)")

# ============================================================================
# PASSO 2: AGREGAÇÃO POR BAIRRO E DIA
# ============================================================================
print("\n" + "="*80)
print("PASSO 2: AGREGAR CRIMES POR BAIRRO E DIA")
print("="*80)

def agregar_por_dia(df):
    """Agregar crimes por (bairro, data) """
    df_copy = df.copy()
    df_copy['data'] = df_copy['data_hora'].dt.date
    
    agregado = df_copy.groupby(['local_oficial', 'data']).agg({
        'tipo': lambda x: (
            (x.str.lower() == 'cvli').sum(),  # CVLI count
            (x.str.lower() == 'cvp').sum()    # CVP count
        )
    }).reset_index()
    
    # Desagrupar tupla
    agregado[['cvli', 'cvp']] = pd.DataFrame(agregado['tipo'].tolist(), index=agregado.index)
    agregado['total_crimes'] = agregado['cvli'] + agregado['cvp']
    agregado = agregado[['local_oficial', 'data', 'cvli', 'cvp', 'total_crimes']]
    
    return agregado

df_treino_agg = agregar_por_dia(df_treino)
df_teste_agg = agregar_por_dia(df_teste)

print(f"\n✓ Agregação completa:")
print(f"  Treino: {len(df_treino_agg)} registros (bairro-dia)")
print(f"  Teste:  {len(df_teste_agg)} registros (bairro-dia)")

# Top bairros (foco)
top_bairros = df_crimes['local_oficial'].value_counts().head(10).index.tolist()
print(f"\n✓ Top 10 bairros (foco na análise):")
for idx, bairro in enumerate(top_bairros, 1):
    crimes = len(df_crimes[df_crimes['local_oficial'] == bairro])
    print(f"  {idx:2d}. {bairro}: {crimes} crimes")

# ============================================================================
# PASSO 3: AGREGAÇÃO TEMPORAL (14 DIAS)
# ============================================================================
print("\n" + "="*80)
print("PASSO 3: AGREGAR PARA JANELAS DE 14 DIAS")
print("="*80)

def criar_janelas_14d(df_agg):
    """Agregar por janelas de 14 dias"""
    df_copy = df_agg.copy()
    df_copy['data'] = pd.to_datetime(df_copy['data'])
    
    # Agrupar por bairro e criar janelas de 14 dias
    resultados = []
    
    for bairro in df_copy['local_oficial'].unique():
        df_bairro = df_copy[df_copy['local_oficial'] == bairro].sort_values('data')
        
        # Criar bins de 14 dias
        df_bairro['janela'] = (df_bairro['data'] - df_bairro['data'].min()).dt.days // 14
        
        janelas_agg = df_bairro.groupby('janela').agg({
            'cvli': 'sum',
            'cvp': 'sum',
            'total_crimes': 'sum',
            'data': ['min', 'max', 'count']
        }).reset_index()
        
        janelas_agg.columns = ['janela', 'cvli', 'cvp', 'total_crimes', 'data_inicio', 'data_fim', 'dias_com_crime']
        janelas_agg['local_oficial'] = bairro
        janelas_agg['periodo'] = janelas_agg['data_inicio'].astype(str) + ' to ' + janelas_agg['data_fim'].astype(str)
        
        resultados.append(janelas_agg)
    
    return pd.concat(resultados, ignore_index=True)

df_treino_14d = criar_janelas_14d(df_treino_agg)
df_teste_14d = criar_janelas_14d(df_teste_agg)

print(f"\n✓ Janelas de 14 dias criadas:")
print(f"  Treino: {len(df_treino_14d)} janelas")
print(f"  Teste:  {len(df_teste_14d)} janelas")

# ============================================================================
# PASSO 4: SIMULAÇÃO DE MODELO ST-GCN (Simplificado)
# ============================================================================
print("\n" + "="*80)
print("PASSO 4: MODELO PREDITIVO (Média Móvel + Sazonalidade)")
print("="*80)

print("""
Nota: Este modelo é uma APROXIMAÇÃO do ST-GCN para fins de teste.
      ST-GCN real requer PyTorch e GPU. Esta versão usa:
      
      risco_previsto = α×média_móvel + β×sazonalidade + γ×tendência
      
      Onde:
        α = 0.50 (peso da série temporal)
        β = 0.30 (peso da sazonalidade mensal)
        γ = 0.20 (peso da tendência)
""")

class ModeloPreditivoSimplificado:
    def __init__(self):
        self.historico = {}
        self.sazonalidade = {}
        self.tendencia = {}
        self.scaler = MinMaxScaler()
    
    def treinar(self, df_treino):
        """Treinar com dados de 2022-2023"""
        print("\n  Treinando modelo com dados 2022-2023...")
        
        for bairro in df_treino['local_oficial'].unique():
            df_bairro = df_treino[df_treino['local_oficial'] == bairro].sort_values('janela')
            
            if len(df_bairro) < 3:  # Precisa de pelo menos 3 pontos
                continue
            
            # Armazenar histórico
            self.historico[bairro] = df_bairro['total_crimes'].values
            
            # Calcular sazonalidade (por mês do ano)
            df_bairro['mes'] = pd.to_datetime(df_bairro['data_inicio']).dt.month
            sazonalidade = df_bairro.groupby('mes')['total_crimes'].mean()
            self.sazonalidade[bairro] = sazonalidade.to_dict()
            
            # Calcular tendência (regressão linear simples)
            x = np.arange(len(df_bairro))
            y = df_bairro['total_crimes'].values
            tendencia = np.polyfit(x, y, 1)[0]  # Coeficiente linear
            self.tendencia[bairro] = tendencia
        
        print(f"    ✓ Modelo treinado para {len(self.historico)} bairros")
        return self
    
    def prever(self, bairro, janela_info):
        """Fazer predição para um bairro em uma janela futura"""
        
        if bairro not in self.historico:
            return np.nan
        
        # Componentes
        historico = self.historico[bairro]
        media_movel = np.mean(historico[-3:]) if len(historico) >= 3 else historico.mean()
        
        # Sazonalidade
        mes = pd.to_datetime(janela_info).month
        sazonalidade_fator = self.sazonalidade[bairro].get(mes, 1.0)
        if sazonalidade_fator == 0:
            sazonalidade_fator = 1.0
        
        # Tendência
        tendencia = self.tendencia[bairro]
        
        # Combinação ponderada
        risco_score = (
            0.50 * media_movel +
            0.30 * media_movel * (sazonalidade_fator / np.mean(list(self.sazonalidade[bairro].values()) or [1])) +
            0.20 * (media_movel + tendencia)
        )
        
        # Garantir positivo
        risco_score = max(risco_score, 0)
        
        return risco_score

# Treinar modelo
modelo = ModeloPreditivoSimplificado()
modelo.treinar(df_treino_14d)

# ============================================================================
# PASSO 5: FAZER PREDIÇÕES NO PERÍODO DE TESTE
# ============================================================================
print("\n" + "="*80)
print("PASSO 5: PREDIÇÕES NO PERÍODO DE TESTE (2024-2025)")
print("="*80)

predicoes = []

for idx, row in df_teste_14d.iterrows():
    bairro = row['local_oficial']
    data_inicio = row['data_inicio']
    
    pred = modelo.prever(bairro, data_inicio)
    real = row['total_crimes']
    
    predicoes.append({
        'bairro': bairro,
        'periodo': row['periodo'],
        'predicao': pred,
        'real': real,
        'erro_absoluto': abs(pred - real) if not np.isnan(pred) else np.nan,
        'erro_percentual': (abs(pred - real) / real * 100) if real > 0 and not np.isnan(pred) else np.nan
    })

df_predicoes = pd.DataFrame(predicoes)

print(f"\n✓ Predições geradas: {len(df_predicoes)} observações")

# ============================================================================
# PASSO 6: CALCULAR MÉTRICAS DE EFICIÊNCIA
# ============================================================================
print("\n" + "="*80)
print("PASSO 6: MÉTRICAS DE EFICIÊNCIA")
print("="*80)

# Remover NaN
df_pred_clean = df_predicoes.dropna()

# Métricas gerais
mae = mean_absolute_error(df_pred_clean['real'], df_pred_clean['predicao'])
mse = mean_squared_error(df_pred_clean['real'], df_pred_clean['predicao'])
rmse = np.sqrt(mse)
r2 = r2_score(df_pred_clean['real'], df_pred_clean['predicao'])

print(f"\n📊 MÉTRICAS GLOBAIS:")
print(f"  MAE (Erro Absoluto Médio):  {mae:.2f} crimes/14d")
print(f"  MSE (Erro Quadrático Médio): {mse:.2f}")
print(f"  RMSE (Raiz do MSE):          {rmse:.2f} crimes/14d")
print(f"  R² (Coeficiente):            {r2:.4f}")
print(f"\n  Interpretação:")
print(f"    • Modelo acerta em ±{mae:.0f} crimes por janela de 14 dias")
print(f"    • R² = {r2:.2%} (explica {r2:.2%} da variação)")

# Acurácia por faixa de risco
def classificar_acao(risco):
    if risco >= 0.32:
        return "INTENSIFICAR"
    elif risco >= 0.31:
        return "AUMENTAR"
    elif risco >= 0.30:
        return "MONITORAR"
    else:
        return "MANTER"

# Normalizar para escala 0-1
scaler_global = MinMaxScaler()
predicoes_norm = scaler_global.fit_transform(df_pred_clean[['predicao', 'real']])
df_pred_clean['predicao_norm'] = predicoes_norm[:, 0]
df_pred_clean['real_norm'] = predicoes_norm[:, 1]

df_pred_clean['acao_pred'] = df_pred_clean['predicao_norm'].apply(classificar_acao)
df_pred_clean['acao_real'] = df_pred_clean['real_norm'].apply(classificar_acao)
df_pred_clean['acertou_acao'] = (df_pred_clean['acao_pred'] == df_pred_clean['acao_real']).astype(int)

acuracia_acao = df_pred_clean['acertou_acao'].mean() * 100

print(f"\n🎯 ACURÁCIA DE AÇÃO OPERACIONAL:")
print(f"  Acerto em recomendação: {acuracia_acao:.1f}%")
print(f"\n  Detalhamento:")
for acao in ['INTENSIFICAR', 'AUMENTAR', 'MONITORAR', 'MANTER']:
    df_acao = df_pred_clean[df_pred_clean['acao_real'] == acao]
    if len(df_acao) > 0:
        acerto = (df_acao['acertou_acao'].sum() / len(df_acao)) * 100
        print(f"    {acao:15s}: {acerto:5.1f}% ({df_acao['acertou_acao'].sum()}/{len(df_acao)})")

# Erros por bairro
print(f"\n📍 PERFORMANCE POR BAIRRO (Top 10):")
print("-" * 60)

metricas_bairro = df_pred_clean.groupby('bairro').agg({
    'erro_absoluto': ['mean', 'std', 'count'],
    'real': 'mean'
}).round(2)

metricas_bairro.columns = ['MAE', 'StdDev', 'Amostras', 'CrimesReais']
metricas_bairro = metricas_bairro.sort_values('Amostras', ascending=False).head(10)

print(f"{'Bairro':<25} {'MAE':<8} {'StdDev':<8} {'N':<6} {'CrimesMed':<10}")
print("-" * 60)
for bairro, row in metricas_bairro.iterrows():
    print(f"{bairro:<25} {row['MAE']:<8.2f} {row['StdDev']:<8.2f} {int(row['Amostras']):<6} {row['CrimesReais']:<10.1f}")

# ============================================================================
# PASSO 7: ANÁLISE DE ERRO
# ============================================================================
print("\n" + "="*80)
print("PASSO 7: ANÁLISE DE ERRO")
print("="*80)

# Distribuição de erros
erro_abs = df_pred_clean['erro_absoluto']

print(f"\n📈 Distribuição de Erros Absolutos:")
print(f"  Mínimo:       {erro_abs.min():.2f} crimes")
print(f"  Q1 (25%):     {erro_abs.quantile(0.25):.2f} crimes")
print(f"  Mediana:      {erro_abs.median():.2f} crimes")
print(f"  Q3 (75%):     {erro_abs.quantile(0.75):.2f} crimes")
print(f"  Máximo:       {erro_abs.max():.2f} crimes")
print(f"  Média:        {erro_abs.mean():.2f} crimes")

# Classificação de erro
df_pred_clean['severidade_erro'] = pd.cut(
    df_pred_clean['erro_absoluto'],
    bins=[0, 2, 5, 10, np.inf],
    labels=['Excelente (0-2)', 'Bom (2-5)', 'Aceitável (5-10)', 'Ruim (>10)']
)

print(f"\n🎯 Classificação de Erro:")
for severidade in ['Excelente (0-2)', 'Bom (2-5)', 'Aceitável (5-10)', 'Ruim (>10)']:
    count = (df_pred_clean['severidade_erro'] == severidade).sum()
    pct = (count / len(df_pred_clean)) * 100
    print(f"  {severidade:<20}: {count:4d} ({pct:5.1f}%)")

# ============================================================================
# PASSO 8: EXPORTAR RELATÓRIO
# ============================================================================
print("\n" + "="*80)
print("PASSO 8: EXPORTAR RELATÓRIO")
print("="*80)

relatorio = {
    "titulo": "Teste de Eficiência ST-GCN",
    "data_teste": datetime.now().isoformat(),
    "dataset": {
        "total_records": len(df_crimes),
        "treino": {
            "periodo": f"{treino_inicio.date()} to {treino_fim.date()}",
            "registros": len(df_treino),
            "observacoes_14d": len(df_treino_14d)
        },
        "teste": {
            "periodo": f"{teste_inicio.date()} to {teste_fim.date()}",
            "registros": len(df_teste),
            "observacoes_14d": len(df_teste_14d)
        }
    },
    "metricas_globais": {
        "MAE_crimes_14d": round(mae, 2),
        "RMSE_crimes_14d": round(rmse, 2),
        "R_squared": round(r2, 4),
        "acuracia_acao_operacional_pct": round(acuracia_acao, 2)
    },
    "modelo_info": {
        "tipo": "Aproximacao de ST-GCN (Media Movel + Sazonalidade + Tendencia)",
        "pesos": {
            "serie_temporal": 0.50,
            "sazonalidade": 0.30,
            "tendencia": 0.20
        },
        "bairros_treinados": len(modelo.historico)
    },
    "performance_por_acao": {
        acao: {
            "acertos_pct": round(
                (df_pred_clean[df_pred_clean['acao_real'] == acao]['acertou_acao'].mean() * 100)
                if len(df_pred_clean[df_pred_clean['acao_real'] == acao]) > 0 else 0,
                2
            ),
            "amostras": int((df_pred_clean['acao_real'] == acao).sum())
        }
        for acao in ['INTENSIFICAR', 'AUMENTAR', 'MONITORAR', 'MANTER']
    },
    "top_bairros_por_mae": [
        {
            "bairro": bairro,
            "mae": round(row['MAE'], 2),
            "amostras": int(row['Amostras']),
            "crimes_reais_media": round(row['CrimesReais'], 2)
        }
        for bairro, row in metricas_bairro.head(5).iterrows()
    ]
}

output_path = Path(__file__).parent / "teste_eficiencia_modelo.json"
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(relatorio, f, indent=2, ensure_ascii=False)

print(f"\n✓ Relatório salvo em: {output_path}")

# ============================================================================
# PASSO 9: VISUALIZAÇÃO ASCII
# ============================================================================
print("\n" + "="*80)
print("PASSO 9: RESUMO VISUAL")
print("="*80)

print(f"""
┌─────────────────────────────────────────────────────────────────────────┐
│                   RESUMO DE EFICIÊNCIA DO MODELO                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│ DADOS:                                                                  │
│   Treino: {len(df_treino):6d} registros (2022-2023)                      │
│   Teste:  {len(df_teste):6d} registros (2024-2025) ← GABARITO            │
│                                                                         │
│ MODELO: Aproximação ST-GCN                                              │
│   Bairros treinados: {len(modelo.historico):3d}                                              │
│   Predições realizadas: {len(df_predicoes):3d}                                        │
│                                                                         │
│ RESULTADOS:                                                             │
│   MAE:  {mae:6.2f} crimes/14d (erro médio)                           │
│   RMSE: {rmse:6.2f} crimes/14d (raiz do erro quadrático)             │
│   R²:   {r2:6.4f} (explica {r2*100:5.1f}% da variação)                    │
│                                                                         │
│ AÇÃO OPERACIONAL:                                                       │
│   Acurácia: {acuracia_acao:5.1f}% (recomendação correta)                      │
│                                                                         │
│ INTERPRETAÇÃO:                                                          │
│   ✓ Modelo acerta ±{mae:.0f} crimes por janela de 14 dias            │
│   ✓ Recomendação operacional: {acuracia_acao:.0f}% de acurácia                 │
│   ✓ Modelo é {"ÚTIL" if acuracia_acao >= 70 else "LIMITADO"} para operações        │
│                                                                         │
│ PRÓXIMAS MELHORIAS:                                                     │
│   • Usar ST-GCN real com PyTorch                                        │
│   • Adicionar dados exógenos (temperatura, feriados, operações)         │
│   • Incorporar dados espaciais (grafo de vizinhança)                    │
│   • Validação cruzada temporal                                          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
""")

print("\n" + "="*80)
print("✓ TESTE CONCLUÍDO")
print("="*80)
print(f"\nArquivo de relatório: {output_path}")
