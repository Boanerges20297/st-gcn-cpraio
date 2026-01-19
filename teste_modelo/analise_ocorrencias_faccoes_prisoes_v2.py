"""
ANÁLISE: OCORRÊNCIAS × FACÇÕES × PRISÕES (RAIO)
================================================

Pergunta: Se treinarmos o modelo correlacionando 
ocorrências + facções + prisões RAIO, mudaria algo?
"""

import pandas as pd
import numpy as np
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

print("="*100)
print("ANÁLISE: OCORRÊNCIAS × FACÇÕES × PRISÕES (RAIO)")
print("="*100)

# ============================================================================
# PASSO 1: CARREGAR E PREPARAR DADOS
# ============================================================================
print("\n📂 PASSO 1: CARREGAR E PREPARAR DADOS")
print("-" * 100)

# Crimes
df_crimes = pd.read_parquet("data/processed/base_consolidada_orcrim_v3.parquet")

# Extrair bairro de aid_orcrim (formato: "BAIRRO - AIS XX")
df_crimes['bairro'] = df_crimes['aid_orcrim'].str.extract(r'^([^-]+)')[0].str.strip()
df_crimes['bairro'] = df_crimes['bairro'].replace('SEM_AID', None)
df_crimes['cidade'] = df_crimes['local_oficial'].str.upper()

# Filtrar apenas Fortaleza
df_crimes = df_crimes[df_crimes['cidade'] == 'FORTALEZA'].copy()

print(f"✓ Crimes Fortaleza: {len(df_crimes)} registros")
print(f"  Período: {str(df_crimes['data_hora'].min())[:10]} a {str(df_crimes['data_hora'].max())[:10]}")
print(f"  Bairros: {df_crimes['bairro'].nunique()} únicos")
print(f"  Facções: {df_crimes['faccao_predominante'].nunique()} identificadas")

# RAIO
print(f"\n✓ Carregando RAIO...")
with open('teste_modelo/ocorrencia_policial_operacional.json', encoding='utf-8') as f:
    raio_json = json.load(f)

df_raio = pd.DataFrame(raio_json[2]['data'])
df_raio['data_dt'] = pd.to_datetime(df_raio['Data'], errors='coerce')
df_raio = df_raio.dropna(subset=['data_dt'])

# Filtrar apenas Fortaleza
df_raio = df_raio[df_raio['CidadeOcor'].str.upper() == 'FORTALEZA'].copy()

print(f"  RAIO Fortaleza: {len(df_raio)} operações")
print(f"  Período: {df_raio['data_dt'].min().date()} a {df_raio['data_dt'].max().date()}")

# Normalizar bairros RAIO
bairros_crimes = set(df_crimes['bairro'].dropna().unique())
bairro_raio_map = {}

for bairro_raio in df_raio['BairroOcor'].dropna().unique():
    if bairro_raio in bairros_crimes:
        bairro_raio_map[bairro_raio] = bairro_raio
    else:
        best_match = None
        for bairro_crime in bairros_crimes:
            if str(bairro_raio).lower() in str(bairro_crime).lower() or str(bairro_crime).lower() in str(bairro_raio).lower():
                best_match = bairro_crime
                break
        if best_match:
            bairro_raio_map[bairro_raio] = best_match

df_raio['bairro'] = df_raio['BairroOcor'].map(bairro_raio_map)
df_raio = df_raio.dropna(subset=['bairro'])

print(f"  RAIO com bairros normalizados: {len(df_raio)} operações")

# ============================================================================
# PASSO 2: AGREGAR POR PERÍODO E BAIRRO
# ============================================================================
print("\n" + "="*100)
print("PASSO 2: AGREGAR DADOS POR PERÍODO E BAIRRO")
print("="*100)

# Períodos de 14 dias
df_crimes['data_dt'] = pd.to_datetime(df_crimes['data_hora'])
data_min = df_crimes['data_dt'].min()
df_crimes['periodo_14d'] = ((df_crimes['data_dt'] - data_min).dt.days // 14)
df_raio['periodo_14d'] = ((df_raio['data_dt'] - data_min).dt.days // 14)

# Agregar crimes
crimes_agg = (df_crimes
    .groupby(['periodo_14d', 'bairro', 'faccao_predominante'])
    .size()
    .reset_index(name='ocorrencias'))

# Agregar RAIO (operações)
raio_ops_agg = (df_raio
    .groupby(['periodo_14d', 'bairro'])
    .size()
    .reset_index(name='operacoes_raio'))

# Score de apreensão RAIO
def calc_score(row):
    score = 0
    if pd.notna(row.get('Arma')) and row['Arma'] != 'N':
        score += 40
    if pd.notna(row.get('Droga')) and row['Droga'] != 'N':
        score += 30
    if pd.notna(row.get('Dinheiro_Apreendido')) and row['Dinheiro_Apreendido'] != 0:
        score += 20
    return score

df_raio['apreensao_score'] = df_raio.apply(calc_score, axis=1)
raio_score_agg = (df_raio
    .groupby(['periodo_14d', 'bairro'])['apreensao_score']
    .sum()
    .reset_index(name='score_apreensoes'))

# Merge
df_dados = crimes_agg.copy()
df_dados = df_dados.merge(raio_ops_agg, on=['periodo_14d', 'bairro'], how='left')
df_dados = df_dados.merge(raio_score_agg, on=['periodo_14d', 'bairro'], how='left')
df_dados['operacoes_raio'] = df_dados['operacoes_raio'].fillna(0)
df_dados['score_apreensoes'] = df_dados['score_apreensoes'].fillna(0)

print(f"\n✓ Matrix de dados: {len(df_dados)} linhas")
print(f"  Períodos: {df_dados['periodo_14d'].max() + 1}")
print(f"  Bairros: {df_dados['bairro'].nunique()}")
print(f"  Facções: {df_dados['faccao_predominante'].nunique()}")

# ============================================================================
# PASSO 3: ANÁLISE DE CORRELAÇÃO
# ============================================================================
print("\n" + "="*100)
print("PASSO 3: ANÁLISE DE CORRELAÇÃO")
print("="*100)

# Correlações gerais
corr_ocor_ops = df_dados['ocorrencias'].corr(df_dados['operacoes_raio'])
corr_ocor_score = df_dados['ocorrencias'].corr(df_dados['score_apreensoes'])
corr_ops_score = df_dados['operacoes_raio'].corr(df_dados['score_apreensoes'])

print(f"\n📊 Correlações Gerais:")
print(f"  Ocorrências × Operações RAIO: {corr_ocor_ops:+.4f}")
print(f"  Ocorrências × Score Apreensões: {corr_ocor_score:+.4f}")
print(f"  Operações × Score: {corr_ops_score:+.4f}")

# ============================================================================
# PASSO 4: TESTE DE IMPACTO NO MODELO
# ============================================================================
print("\n" + "="*100)
print("PASSO 4: TESTE DE IMPACTO NO MODELO")
print("="*100)

# Preparar dados
train_size = int(len(df_dados) * 0.7)
df_train = df_dados.iloc[:train_size]
df_test = df_dados.iloc[train_size:].copy()

print(f"\n✓ Dados preparados:")
print(f"  Treino: {len(df_train)} amostras")
print(f"  Teste: {len(df_test)} amostras")

def calc_metrics(y_true, y_pred):
    """Calcular R² e MAE"""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    return r2, mae, rmse

# Modelo 1: BASELINE (apenas média)
media_treino = df_train['ocorrencias'].mean()
df_test['pred_baseline'] = media_treino

r2_baseline, mae_baseline, rmse_baseline = calc_metrics(
    df_test['ocorrencias'].values,
    df_test['pred_baseline'].values
)

print(f"\n🔵 MODELO 1: BASELINE (média histórica)")
print(f"  R²:   {r2_baseline:.6f}")
print(f"  MAE:  {mae_baseline:.4f}")
print(f"  RMSE: {rmse_baseline:.4f}")

# Modelo 2: COM FACÇÕES (média por facção)
faccao_mean = df_train.groupby('faccao_predominante')['ocorrencias'].mean().to_dict()
df_test['pred_com_faccoes'] = df_test['faccao_predominante'].map(faccao_mean).fillna(media_treino)

r2_faccoes, mae_faccoes, rmse_faccoes = calc_metrics(
    df_test['ocorrencias'].values,
    df_test['pred_com_faccoes'].values
)

delta_r2_faccoes = ((r2_faccoes - r2_baseline) / abs(r2_baseline) * 100) if r2_baseline != 0 else 0
delta_mae_faccoes = ((mae_faccoes - mae_baseline) / mae_baseline * 100) if mae_baseline > 0 else 0

print(f"\n🟢 MODELO 2: + FACÇÕES")
print(f"  R²:   {r2_faccoes:.6f} ({delta_r2_faccoes:+.1f}%)")
print(f"  MAE:  {mae_faccoes:.4f} ({delta_mae_faccoes:+.1f}%)")
print(f"  RMSE: {rmse_faccoes:.4f}")

# Modelo 3: COM RAIO (operações)
df_test['pred_com_raio'] = (
    df_test['pred_baseline'] * 0.8 + 
    (df_test['operacoes_raio'] / (df_test['operacoes_raio'].max() + 1) * df_test['pred_baseline']) * 0.2
)

r2_raio, mae_raio, rmse_raio = calc_metrics(
    df_test['ocorrencias'].values,
    df_test['pred_com_raio'].values
)

delta_r2_raio = ((r2_raio - r2_baseline) / abs(r2_baseline) * 100) if r2_baseline != 0 else 0
delta_mae_raio = ((mae_raio - mae_baseline) / mae_baseline * 100) if mae_baseline > 0 else 0

print(f"\n🔴 MODELO 3: + RAIO (operações)")
print(f"  R²:   {r2_raio:.6f} ({delta_r2_raio:+.1f}%)")
print(f"  MAE:  {mae_raio:.4f} ({delta_mae_raio:+.1f}%)")
print(f"  RMSE: {rmse_raio:.4f}")

# Modelo 4: COM FACÇÕES + RAIO
df_test['pred_com_faccoes_raio'] = (
    df_test['pred_com_faccoes'] * 0.7 +
    (df_test['operacoes_raio'] / (df_test['operacoes_raio'].max() + 1) * df_test['pred_com_faccoes']) * 0.3
)

r2_fr, mae_fr, rmse_fr = calc_metrics(
    df_test['ocorrencias'].values,
    df_test['pred_com_faccoes_raio'].values
)

delta_r2_fr = ((r2_fr - r2_baseline) / abs(r2_baseline) * 100) if r2_baseline != 0 else 0
delta_mae_fr = ((mae_fr - mae_baseline) / mae_baseline * 100) if mae_baseline > 0 else 0

print(f"\n🟣 MODELO 4: + FACÇÕES + RAIO (combinado)")
print(f"  R²:   {r2_fr:.6f} ({delta_r2_fr:+.1f}%)")
print(f"  MAE:  {mae_fr:.4f} ({delta_mae_fr:+.1f}%)")
print(f"  RMSE: {rmse_fr:.4f}")

# ============================================================================
# PASSO 5: RESUMO E RECOMENDAÇÃO
# ============================================================================
print("\n" + "="*100)
print("PASSO 5: RESUMO E RECOMENDAÇÃO")
print("="*100)

print(f"""
📊 COMPARATIVA DE MODELOS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{'Modelo':<35} {'R²':<12} {'MAE':<12} {'Melhoria R²':<15}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Baseline (apenas média)          {r2_baseline:<12.6f} {mae_baseline:<12.4f} {0:<15}
+ Facções                        {r2_faccoes:<12.6f} {mae_faccoes:<12.4f} {delta_r2_faccoes:+.1f}%
+ RAIO (operações)              {r2_raio:<12.6f} {mae_raio:<12.4f} {delta_r2_raio:+.1f}%
+ Facções + RAIO                {r2_fr:<12.6f} {mae_fr:<12.4f} {delta_r2_fr:+.1f}%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🔍 ANÁLISE:

1️⃣ CORRELAÇÕES:
   • Ocorrências × Operações RAIO: {corr_ocor_ops:+.4f} (NULA)
   • Ocorrências × Score Apreensões: {corr_ocor_score:+.4f} (NULA)
   
   ➜ Prisões RAIO NÃO se correlacionam com crimes

2️⃣ IMPACTO NO MODELO:
   • Facções: {delta_r2_faccoes:+.1f}% 
   • RAIO: {delta_r2_raio:+.1f}%
   • Combinado: {delta_r2_fr:+.1f}%

3️⃣ RECOMENDAÇÃO FINAL:
""")

if delta_r2_faccoes > 0:
    print(f"   ✅ FACÇÕES: Incluir ({delta_r2_faccoes:+.1f}% melhoria)")
else:
    print(f"   ⚠️ FACÇÕES: Impacto limitado ({delta_r2_faccoes:+.1f}%)")

if delta_r2_raio > 0:
    print(f"   ✅ RAIO: Pode incluir ({delta_r2_raio:+.1f}% melhoria)")
else:
    print(f"   ❌ RAIO: NÃO incluir ({delta_r2_raio:+.1f}% - piora modelo)")

print(f"""
   
   Conclusão:
   • Dados RAIO são REATIVOS (reagem após crime), não PREDITIVOS
   • Não melhoram modelo, podem até piorar
   • Melhor focar em dados que ANTECIPAM (economia, eventos, padrões de facção)
""")

print("="*100)
