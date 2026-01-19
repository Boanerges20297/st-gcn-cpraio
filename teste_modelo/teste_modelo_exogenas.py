"""
TESTE DE MODELO COM DADOS EXÓGENOS
===================================

Objetivo: Comparar performance do modelo com/sem dados exógenos (prisões RAIO)
  - Modelo 1: Apenas histórico + sazonalidade (baseline)
  - Modelo 2: Com dados de prisões RAIO como feature exógena
  - Comparar: MAE, RMSE, R², Acurácia operacional
  
Hipótese: Incluir dados de prisões deve melhorar previsões em ~2-5%
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("TESTE DE MODELO COM DADOS EXÓGENOS (PRISÕES RAIO)")
print("="*80)

# ============================================================================
# PASSO 1: CARREGAR DADOS
# ============================================================================
print("\n📂 PASSO 1: PREPARAR DADOS")
print("-" * 80)

# Carregar crimes
crime_path = Path(__file__).parent.parent / "data" / "processed" / "base_consolidada_orcrim_v3.parquet"
df_crimes = pd.read_parquet(crime_path)
df_crimes['data_hora'] = pd.to_datetime(df_crimes['data_hora'])

# Carregar RAIO
raio_path = Path(__file__).parent.parent / "data" / "raw" / "data_with_coordinates.js"
with open(raio_path, 'r', encoding='utf-8') as f:
    content = f.read()
content = content.replace('module.exports = ', '').strip()
if content.endswith(';'):
    content = content[:-1]
raio_raw = json.loads(content)
df_raio = pd.DataFrame(raio_raw)
df_raio['Data'] = pd.to_datetime(df_raio['Data'])

print(f"✓ Crimes: {len(df_crimes)} registros")
print(f"✓ RAIO: {len(df_raio)} operações")

# ============================================================================
# PASSO 2: NORMALIZAR BAIRROS E AGREGAR PRISÕES
# ============================================================================
print("\n" + "="*80)
print("PASSO 2: AGREGAR DADOS")
print("="*80)

# Normalizar bairros RAIO
bairros_raio = df_raio['BairroOcor'].unique()
bairros_crime = df_crimes['local_oficial'].unique()

mapa_bairros = {}
for b_raio in bairros_raio:
    if pd.isna(b_raio):
        continue
    match = None
    for b_crime in bairros_crime:
        if str(b_raio).lower() == str(b_crime).lower():
            match = b_crime
            break
    mapa_bairros[b_raio] = match

df_raio['bairro_normalizado'] = df_raio['BairroOcor'].map(mapa_bairros)
df_raio_matched = df_raio[df_raio['bairro_normalizado'].notna()].copy()

print(f"✓ Operações RAIO em bairros conhecidos: {len(df_raio_matched)}")

# ============================================================================
# PASSO 3: CRIAR CONJUNTOS TREINO E TESTE
# ============================================================================
print("\n" + "="*80)
print("PASSO 3: SEPARAR TREINO E TESTE")
print("="*80)

# Período de treino: 2022-2023
# Período de teste: 2024-2025

treino_inicio = pd.Timestamp('2022-01-01')
treino_fim = pd.Timestamp('2023-12-31')
teste_inicio = pd.Timestamp('2024-01-01')
teste_fim = pd.Timestamp('2025-12-31')

df_crimes_treino = df_crimes[
    (df_crimes['data_hora'] >= treino_inicio) & 
    (df_crimes['data_hora'] <= treino_fim)
]
df_crimes_teste = df_crimes[
    (df_crimes['data_hora'] >= teste_inicio) & 
    (df_crimes['data_hora'] <= teste_fim)
]

# Agregar crimes em janelas 14 dias
def agregar_crimes_14d(df):
    df_copy = df.copy()
    df_copy['data'] = df_copy['data_hora'].dt.date
    
    agregado = df_copy.groupby(['local_oficial', 'data']).agg({
        'tipo': lambda x: (x.str.lower() == 'cvli').sum() + (x.str.lower() == 'cvp').sum()
    }).reset_index()
    
    agregado.columns = ['bairro', 'data', 'total_crimes']
    agregado['data'] = pd.to_datetime(agregado['data'])
    
    resultados = []
    for bairro in agregado['bairro'].unique():
        df_b = agregado[agregado['bairro'] == bairro].sort_values('data')
        data_min = df_b['data'].min()
        data_max = df_b['data'].max()
        
        current = data_min
        while current <= data_max:
            end = current + timedelta(days=13)
            df_window = df_b[(df_b['data'] >= current) & (df_b['data'] <= end)]
            
            if len(df_window) > 0:
                resultados.append({
                    'bairro': bairro,
                    'data_inicio': current,
                    'data_fim': end,
                    'crimes': df_window['total_crimes'].sum()
                })
            
            current += timedelta(days=14)
    
    return pd.DataFrame(resultados)

print("  Agregando crimes treino...")
df_crimes_treino_14d = agregar_crimes_14d(df_crimes_treino)

print("  Agregando crimes teste...")
df_crimes_teste_14d = agregar_crimes_14d(df_crimes_teste)

print(f"✓ Treino: {len(df_crimes_treino_14d)} observações")
print(f"✓ Teste: {len(df_crimes_teste_14d)} observações")

# ============================================================================
# PASSO 4: AGREGAR PRISÕES
# ============================================================================
print("\n" + "="*80)
print("PASSO 4: AGREGAR PRISÕES RAIO")
print("="*80)

def agregar_prisoes_14d(df):
    df_copy = df.copy()
    df_copy['data'] = df_copy['Data'].dt.date
    
    agregado = df_copy.groupby(['bairro_normalizado', 'data']).size().reset_index(name='prisoes')
    agregado.columns = ['bairro', 'data', 'prisoes']
    agregado['data'] = pd.to_datetime(agregado['data'])
    
    resultados = []
    for bairro in agregado['bairro'].unique():
        df_b = agregado[agregado['bairro'] == bairro].sort_values('data')
        data_min = df_b['data'].min()
        data_max = df_b['data'].max()
        
        current = data_min
        while current <= data_max:
            end = current + timedelta(days=13)
            df_window = df_b[(df_b['data'] >= current) & (df_b['data'] <= end)]
            
            if len(df_window) > 0:
                resultados.append({
                    'bairro': bairro,
                    'data_inicio': current,
                    'data_fim': end,
                    'prisoes': df_window['prisoes'].sum()
                })
            
            current += timedelta(days=14)
    
    return pd.DataFrame(resultados)

print("  Agregando prisões...")
df_prisoes_14d = agregar_prisoes_14d(df_raio_matched)
print(f"✓ Prisões agregadas: {len(df_prisoes_14d)} observações")

# ============================================================================
# PASSO 5: MESCLAR DADOS
# ============================================================================
print("\n" + "="*80)
print("PASSO 5: MESCLAR CRIMES + PRISÕES")
print("="*80)

# Mesclar treino
df_treino_merged = pd.merge(
    df_crimes_treino_14d,
    df_prisoes_14d,
    on=['bairro', 'data_inicio', 'data_fim'],
    how='left'
)
df_treino_merged['prisoes'] = df_treino_merged['prisoes'].fillna(0)

# Mesclar teste
df_teste_merged = pd.merge(
    df_crimes_teste_14d,
    df_prisoes_14d,
    on=['bairro', 'data_inicio', 'data_fim'],
    how='left'
)
df_teste_merged['prisoes'] = df_teste_merged['prisoes'].fillna(0)

print(f"✓ Treino mesclado: {len(df_treino_merged)} observações")
print(f"✓ Teste mesclado: {len(df_teste_merged)} observações")

# ============================================================================
# PASSO 6: DEFINIR MODELOS
# ============================================================================
print("\n" + "="*80)
print("PASSO 6: DEFINIR MODELOS")
print("="*80)

class ModeloBaseline:
    """Modelo sem dados exógenos (apenas histórico + sazonalidade)"""
    
    def __init__(self):
        self.historico = {}
        self.sazonalidade = {}
        self.tendencia = {}
    
    def treinar(self, df):
        for bairro in df['bairro'].unique():
            df_b = df[df['bairro'] == bairro].sort_values('data_inicio')
            
            if len(df_b) < 3:
                continue
            
            self.historico[bairro] = df_b['crimes'].values
            
            # Sazonalidade
            df_b['mes'] = df_b['data_inicio'].dt.month
            sazon = df_b.groupby('mes')['crimes'].mean()
            self.sazonalidade[bairro] = sazon.to_dict()
            
            # Tendência
            x = np.arange(len(df_b))
            y = df_b['crimes'].values
            tendencia = np.polyfit(x, y, 1)[0] if len(df_b) > 1 else 0
            self.tendencia[bairro] = tendencia
        
        print(f"✓ Modelo Baseline treinado para {len(self.historico)} bairros")
    
    def prever(self, bairro, data_inicio):
        if bairro not in self.historico:
            return np.nan
        
        historico = self.historico[bairro]
        media_movel = np.mean(historico[-3:]) if len(historico) >= 3 else historico.mean()
        
        mes = data_inicio.month
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


class ModeloComExogenas:
    """Modelo com dados exógenos (prisões RAIO)"""
    
    def __init__(self):
        self.historico = {}
        self.sazonalidade = {}
        self.tendencia = {}
        self.peso_prisoes = {}
    
    def treinar(self, df):
        for bairro in df['bairro'].unique():
            df_b = df[df['bairro'] == bairro].sort_values('data_inicio')
            
            if len(df_b) < 3:
                continue
            
            self.historico[bairro] = df_b['crimes'].values
            
            # Sazonalidade
            df_b['mes'] = df_b['data_inicio'].dt.month
            sazon = df_b.groupby('mes')['crimes'].mean()
            self.sazonalidade[bairro] = sazon.to_dict()
            
            # Tendência
            x = np.arange(len(df_b))
            y = df_b['crimes'].values
            tendencia = np.polyfit(x, y, 1)[0] if len(df_b) > 1 else 0
            self.tendencia[bairro] = tendencia
            
            # Correlação prisões-crimes (peso exógeno)
            if len(df_b) > 1 and df_b['prisoes'].sum() > 0:
                corr = np.corrcoef(df_b['crimes'], df_b['prisoes'])[0, 1]
                # Se correlação é negativa (prisões diminuem crimes), usar valor absoluto
                self.peso_prisoes[bairro] = abs(corr) if not np.isnan(corr) else 0.1
            else:
                self.peso_prisoes[bairro] = 0.1  # Peso padrão
        
        print(f"✓ Modelo com Exógenas treinado para {len(self.historico)} bairros")
    
    def prever(self, bairro, data_inicio, n_prisoes):
        if bairro not in self.historico:
            return np.nan
        
        historico = self.historico[bairro]
        media_movel = np.mean(historico[-3:]) if len(historico) >= 3 else historico.mean()
        
        mes = data_inicio.month
        sazon_fator = self.sazonalidade[bairro].get(mes, 1.0)
        if sazon_fator == 0:
            sazon_fator = 1.0
        
        tendencia = self.tendencia[bairro]
        peso_prisoes = self.peso_prisoes.get(bairro, 0.1)
        
        # Modelo base
        risco_base = (
            0.50 * media_movel +
            0.30 * media_movel * (sazon_fator / np.mean(list(self.sazonalidade[bairro].values()) or [1])) +
            0.20 * (media_movel + tendencia)
        )
        
        # Ajuste com prisões (reduz risco se há muitas prisões)
        # Assumindo que prisões reduzem crimes em ~5% por prisão significativa
        ajuste_prisoes = 1.0 - (n_prisoes * peso_prisoes * 0.05)
        ajuste_prisoes = max(0.7, min(1.3, ajuste_prisoes))  # Limitar entre 0.7 e 1.3
        
        risco = risco_base * ajuste_prisoes
        
        return max(risco, 0)

# ============================================================================
# PASSO 7: TREINAR MODELOS
# ============================================================================
print("\n" + "="*80)
print("PASSO 7: TREINAR MODELOS")
print("="*80)

modelo_baseline = ModeloBaseline()
modelo_baseline.treinar(df_treino_merged)

modelo_exogenas = ModeloComExogenas()
modelo_exogenas.treinar(df_treino_merged)

# ============================================================================
# PASSO 8: FAZER PREDIÇÕES
# ============================================================================
print("\n" + "="*80)
print("PASSO 8: FAZER PREDIÇÕES")
print("="*80)

print("  Predições com Baseline...")
predicoes_baseline = []
for idx, row in df_teste_merged.iterrows():
    pred = modelo_baseline.prever(row['bairro'], row['data_inicio'])
    if not np.isnan(pred):
        predicoes_baseline.append({
            'bairro': row['bairro'],
            'real': row['crimes'],
            'pred': pred,
            'erro': abs(pred - row['crimes'])
        })

df_pred_baseline = pd.DataFrame(predicoes_baseline).dropna()

print("  Predições com Exógenas...")
predicoes_exogenas = []
for idx, row in df_teste_merged.iterrows():
    pred = modelo_exogenas.prever(row['bairro'], row['data_inicio'], row['prisoes'])
    if not np.isnan(pred):
        predicoes_exogenas.append({
            'bairro': row['bairro'],
            'real': row['crimes'],
            'pred': pred,
            'prisoes': row['prisoes'],
            'erro': abs(pred - row['crimes'])
        })

df_pred_exogenas = pd.DataFrame(predicoes_exogenas).dropna()

print(f"✓ Baseline: {len(df_pred_baseline)} predições")
print(f"✓ Exógenas: {len(df_pred_exogenas)} predições")

# ============================================================================
# PASSO 9: COMPARAR MÉTRICAS
# ============================================================================
print("\n" + "="*80)
print("PASSO 9: COMPARAR PERFORMANCE")
print("="*80)

# Baseline
mae_baseline = mean_absolute_error(df_pred_baseline['real'], df_pred_baseline['pred'])
rmse_baseline = np.sqrt(mean_squared_error(df_pred_baseline['real'], df_pred_baseline['pred']))
r2_baseline = r2_score(df_pred_baseline['real'], df_pred_baseline['pred'])

# Exógenas
mae_exogenas = mean_absolute_error(df_pred_exogenas['real'], df_pred_exogenas['pred'])
rmse_exogenas = np.sqrt(mean_squared_error(df_pred_exogenas['real'], df_pred_exogenas['pred']))
r2_exogenas = r2_score(df_pred_exogenas['real'], df_pred_exogenas['pred'])

# Melhorias
melhoria_mae = ((mae_baseline - mae_exogenas) / mae_baseline) * 100
melhoria_rmse = ((rmse_baseline - rmse_exogenas) / rmse_baseline) * 100
melhoria_r2 = ((r2_exogenas - r2_baseline) / abs(r2_baseline)) * 100

print(f"\n📊 MÉTRICAS DE PERFORMANCE:")
print(f"{'Métrica':<15} {'Baseline':<15} {'Com Exógenas':<15} {'Melhoria':<15}")
print("-" * 60)
print(f"{'MAE':<15} {mae_baseline:<15.2f} {mae_exogenas:<15.2f} {melhoria_mae:+15.1f}%")
print(f"{'RMSE':<15} {rmse_baseline:<15.2f} {rmse_exogenas:<15.2f} {melhoria_rmse:+15.1f}%")
print(f"{'R²':<15} {r2_baseline:<15.4f} {r2_exogenas:<15.4f} {melhoria_r2:+15.1f}%")

# ============================================================================
# PASSO 10: ANÁLISE POR BAIRRO
# ============================================================================
print("\n" + "="*80)
print("PASSO 10: ANÁLISE POR BAIRRO")
print("="*80)

bairros_raio_ativos = df_pred_exogenas[df_pred_exogenas['prisoes'] > 0]['bairro'].unique()

print(f"\n  Bairros com prisões RAIO: {len(bairros_raio_ativos)}")
print(f"  {'Bairro':<25} {'MAE Baseline':<15} {'MAE Exog.':<15} {'Melhoria':<10}")
print(f"  {'-'*65}")

for bairro in bairros_raio_ativos[:10]:
    df_b_baseline = df_pred_baseline[df_pred_baseline['bairro'] == bairro]
    df_b_exog = df_pred_exogenas[df_pred_exogenas['bairro'] == bairro]
    
    if len(df_b_baseline) > 0 and len(df_b_exog) > 0:
        mae_b_baseline = df_b_baseline['erro'].mean()
        mae_b_exog = df_b_exog['erro'].mean()
        melhoria_b = ((mae_b_baseline - mae_b_exog) / mae_b_baseline) * 100 if mae_b_baseline > 0 else 0
        
        print(f"  {bairro:<25} {mae_b_baseline:<15.2f} {mae_b_exog:<15.2f} {melhoria_b:+10.1f}%")

# ============================================================================
# PASSO 11: EXPORTAR RESULTADOS
# ============================================================================
print("\n" + "="*80)
print("PASSO 11: EXPORTAR RESULTADOS")
print("="*80)

resultados = {
    "titulo": "Teste de Modelo com Dados Exógenos (RAIO)",
    "data": datetime.now().isoformat(),
    "comparacao": {
        "baseline": {
            "MAE": round(mae_baseline, 2),
            "RMSE": round(rmse_baseline, 2),
            "R2": round(r2_baseline, 4),
            "n_predicoes": len(df_pred_baseline)
        },
        "com_exogenas": {
            "MAE": round(mae_exogenas, 2),
            "RMSE": round(rmse_exogenas, 2),
            "R2": round(r2_exogenas, 4),
            "n_predicoes": len(df_pred_exogenas)
        },
        "melhorias": {
            "MAE_percentual": round(melhoria_mae, 1),
            "RMSE_percentual": round(melhoria_rmse, 1),
            "R2_percentual": round(melhoria_r2, 1)
        }
    },
    "bairros_raio_cobertos": len(bairros_raio_ativos),
    "recomendacao": {
        "conclusao": "✅ INCLUIR DADOS EXÓGENOS" if melhoria_r2 > 0.5 else "⚠️ AVALIAR MELHOR",
        "confianca": "Alta" if melhoria_mae > 5 else "Média" if melhoria_mae > 2 else "Baixa",
        "proximo_passo": "Incorporar prisões no modelo ST-GCN real"
    }
}

output_path = Path(__file__).parent / "teste_modelo_exogenas.json"
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(resultados, f, indent=2, ensure_ascii=False)

print(f"\n✓ Resultados salvos: {output_path}")

# ============================================================================
# VISUALIZAÇÃO FINAL
# ============================================================================
print("\n" + "="*80)
print("RESUMO VISUAL")
print("="*80)

print(f"""
┌────────────────────────────────────────────────────────────────┐
│       TESTE DE MODELO COM DADOS EXÓGENOS (RAIO)               │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│ COMPARAÇÃO DE PERFORMANCE:                                     │
│                                                                │
│   MAE:   {mae_baseline:.2f} → {mae_exogenas:.2f}  ({melhoria_mae:+.1f}%)              │
│   RMSE:  {rmse_baseline:.2f} → {rmse_exogenas:.2f}  ({melhoria_rmse:+.1f}%)             │
│   R²:    {r2_baseline:.4f} → {r2_exogenas:.4f}  ({melhoria_r2:+.1f}%)             │
│                                                                │
│ BAIRROS COM COBERTURA RAIO: {len(bairros_raio_ativos)}                           │
│                                                                │
│ CONCLUSÃO:                                                     │
│ {"✅ Dados exógenos melhoram previsão" if melhoria_r2 > 0.5 else "⚠️ Melhoria marginal ou negativa"}                       │
│                                                                │
│ PRÓXIMO PASSO:                                                 │
│ {"Integrar prisões no ST-GCN real" if melhoria_r2 > 0.5 else "Analisar em modelo mais complexo"}                 │
│                                                                │
└────────────────────────────────────────────────────────────────┘
""")

print("="*80)
print("✓ TESTE CONCLUÍDO")
print("="*80)
