"""
ANÁLISE DE APREENSÕES SIGNIFICATIVAS RAIO
==========================================

Objetivo: Identificar grandes apreensões (armas, drogas, dinheiro) e correlacionar
com crimes para validar se têm melhor influência exógena que simples contagem
de operações.

Hipótese: Apreensões grandes (>5kg droga, >1 arma, >5k dinheiro) têm correlação
melhor com redução de crimes.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("ANÁLISE DE APREENSÕES SIGNIFICATIVAS RAIO")
print("="*80)

# ============================================================================
# PASSO 1: CARREGAR DADOS RAIO
# ============================================================================
print("\n📂 PASSO 1: CARREGAR DADOS RAIO")
print("-" * 80)

raio_path = Path(__file__).parent / "ocorrencia_policial_operacional.json"

if not raio_path.exists():
    print(f"✗ Arquivo não encontrado: {raio_path}")
    exit(1)

try:
    with open(raio_path, 'r', encoding='utf-8') as f:
        raio_json = json.load(f)
    
    # Extrair dados da tabela (estrutura PHPMyAdmin)
    # Item 2 é a tabela 'ocorrencia policial operacional'
    raio_raw = raio_json[2]['data']
    
    print(f"✓ Dados RAIO carregados: {len(raio_raw)} operações")
    print(f"  Tamanho do arquivo: {raio_path.stat().st_size / 1024 / 1024:.1f} MB")
except Exception as e:
    print(f"✗ Erro ao carregar: {e}")
    exit(1)

# Converter para DataFrame
df_raio = pd.DataFrame(raio_raw)
print(f"✓ DataFrame: {df_raio.shape[0]} linhas, {df_raio.shape[1]} colunas")

# Converter data
df_raio['Data'] = pd.to_datetime(df_raio['Data'], errors='coerce')
print(f"  Período: {df_raio['Data'].min().date()} a {df_raio['Data'].max().date()}")

# Info sobre apreensões
print(f"\n  Colunas de apreensão:")
apreensao_cols = ['Arma', 'Droga', 'Veiculo_Apr', 'Dinheiro_Apreendido', 'MaterialApreendido']
for col in apreensao_cols:
    if col in df_raio.columns:
        non_null = df_raio[col].notna().sum()
        print(f"    • {col:<25}: {non_null:>6} registros não-nulos")

# ============================================================================
# PASSO 2: CRIAR SCORE DE APREENSÕES SIGNIFICATIVAS
# ============================================================================
print("\n" + "="*80)
print("PASSO 2: CLASSIFICAR OPERAÇÕES POR RELEVÂNCIA")
print("="*80)

def calcular_score_apreensao(row):
    """Calcular score de relevância da operação"""
    score = 0
    
    # Armas (peso alto)
    if pd.notna(row['Arma']) and row['Arma'] not in ['-1', -1]:
        try:
            n_armas = int(row['Arma']) if row['Arma'] != '-1' else 0
            score += n_armas * 50  # 50 pontos por arma
        except:
            pass
    
    # Drogas (peso alto)
    if pd.notna(row['Droga']) and row['Droga'] not in ['-1', -1, '0', 0]:
        try:
            n_drogas = int(row['Droga']) if row['Droga'] not in ['-1', '0'] else 0
            score += n_drogas * 40  # 40 pontos por droga
        except:
            pass
    
    # Veículos (peso médio)
    if pd.notna(row['Veiculo_Apr']) and row['Veiculo_Apr'] not in ['-1', -1]:
        try:
            n_veics = int(row['Veiculo_Apr']) if row['Veiculo_Apr'] != '-1' else 0
            score += n_veics * 20  # 20 pontos por veículo
        except:
            pass
    
    # Dinheiro (peso médio-alto)
    if pd.notna(row['Dinheiro_Apreendido']):
        try:
            dinheiro = float(row['Dinheiro_Apreendido'])
            # Converter para score (cada R$1000 = 1 ponto)
            score += min(dinheiro / 1000, 100)  # Máximo 100 pontos
        except:
            pass
    
    # Material apreendido (peso baixo)
    if pd.notna(row['MaterialApreendido']):
        score += 5
    
    return score

df_raio['score_apreensao'] = df_raio.apply(calcular_score_apreensao, axis=1)

print(f"✓ Score de apreensão calculado")

# Estatísticas
print(f"\n📈 DISTRIBUIÇÃO DE SCORES:")
print(f"  Mínimo: {df_raio['score_apreensao'].min():.0f}")
print(f"  Q1: {df_raio['score_apreensao'].quantile(0.25):.0f}")
print(f"  Mediana: {df_raio['score_apreensao'].median():.0f}")
print(f"  Q3: {df_raio['score_apreensao'].quantile(0.75):.0f}")
print(f"  Máximo: {df_raio['score_apreensao'].max():.0f}")
print(f"  Média: {df_raio['score_apreensao'].mean():.0f}")

# Classificar
df_raio['relevancia'] = pd.cut(
    df_raio['score_apreensao'],
    bins=[-1, 0, 50, 100, 200, 10000],
    labels=['Nenhuma', 'Baixa', 'Média', 'Alta', 'Muito Alta']
)

print(f"\n🎯 OPERAÇÕES POR RELEVÂNCIA:")
relevancia_counts = df_raio['relevancia'].value_counts().sort_index()
for rel, count in relevancia_counts.items():
    print(f"  {rel:<12}: {count:>5} operações ({count/len(df_raio)*100:>5.1f}%)")

# ============================================================================
# PASSO 3: TOP OPERAÇÕES SIGNIFICATIVAS
# ============================================================================
print("\n" + "="*80)
print("PASSO 3: TOP 20 OPERAÇÕES MAIS SIGNIFICATIVAS")
print("="*80)

top_ops = df_raio.nlargest(20, 'score_apreensao')[
    ['Data', 'BairroOcor', 'Natureza', 'Arma', 'Droga', 'Veiculo_Apr', 
     'Dinheiro_Apreendido', 'score_apreensao', 'relevancia']
]

print(f"\n{'#':<3} {'Data':<12} {'Bairro':<20} {'Score':<8} {'Natureza':<30}")
print("-" * 85)

for idx, (i, row) in enumerate(top_ops.iterrows(), 1):
    data_fmt = pd.Timestamp(row['Data']).strftime('%d/%m/%Y')
    bairro = str(row['BairroOcor'])[:19]
    natureza = str(row['Natureza'])[:29]
    
    print(f"{idx:<3} {data_fmt:<12} {bairro:<20} {row['score_apreensao']:<8.0f} {natureza:<30}")

# ============================================================================
# PASSO 4: ANÁLISE DE TIPOS DE APREENSÃO
# ============================================================================
print("\n" + "="*80)
print("PASSO 4: DETALHAMENTO DE APREENSÕES")
print("="*80)

# Armas
armas_count = 0
try:
    for val in df_raio['Arma']:
        if pd.notna(val) and val not in ['-1', -1]:
            try:
                armas_count += int(val)
            except:
                pass
except:
    pass

# Drogas
drogas_op = (df_raio['Droga'].notna() & (df_raio['Droga'] != '-1') & (df_raio['Droga'] != '0')).sum()

# Veículos
veics_count = 0
try:
    for val in df_raio['Veiculo_Apr']:
        if pd.notna(val) and val not in ['-1', -1]:
            try:
                veics_count += int(val)
            except:
                pass
except:
    pass

# Dinheiro
dinheiro_total = 0
try:
    for val in df_raio['Dinheiro_Apreendido']:
        if pd.notna(val):
            try:
                dinheiro_total += float(val)
            except:
                pass
except:
    pass

print(f"\n🔫 RESUMO DE APREENSÕES:")
print(f"  Armas apreendidas: {armas_count:>8}")
print(f"  Operações com drogas: {drogas_op:>8}")
print(f"  Veículos apreendidos: {veics_count:>8}")
print(f"  Dinheiro apreendido: R$ {dinheiro_total:>15,.2f}")

# ============================================================================
# PASSO 5: CARREGAR CRIMES E NORMALIZAR
# ============================================================================
print("\n" + "="*80)
print("PASSO 5: CORRELACIONAR APREENSÕES COM CRIMES")
print("="*80)

crime_path = Path(__file__).parent.parent / "data" / "processed" / "base_consolidada_orcrim_v3.parquet"
df_crimes = pd.read_parquet(crime_path)
df_crimes['data_hora'] = pd.to_datetime(df_crimes['data_hora'])

# Normalizar bairros
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

print(f"✓ RAIO normalizado: {len(df_raio_matched)} operações em bairros conhecidos")

# ============================================================================
# PASSO 6: AGREGAR POR PERÍODO
# ============================================================================
print("\n" + "="*80)
print("PASSO 6: AGREGAR CRIMES E APREENSÕES POR PERÍODO")
print("="*80)

# Período teste (onde temos RAIO)
teste_inicio = pd.Timestamp('2024-01-01')
teste_fim = pd.Timestamp('2025-12-31')

df_crimes_teste = df_crimes[
    (df_crimes['data_hora'] >= teste_inicio) & 
    (df_crimes['data_hora'] <= teste_fim)
]

# Agregar crimes por bairro/mês
def agregar_por_mes(df_crime, df_apreensoes):
    df_crime['ano_mes'] = df_crime['data_hora'].dt.to_period('M')
    
    crimes_mes = df_crime.groupby(['local_oficial', 'ano_mes']).size().reset_index(name='crimes')
    
    # Agregar apreensões
    df_apreensoes['ano_mes'] = df_apreensoes['Data'].dt.to_period('M')
    
    apreensoes_mes = df_apreensoes.groupby(['bairro_normalizado', 'ano_mes']).agg({
        'score_apreensao': ['sum', 'count', 'mean']
    }).reset_index()
    
    apreensoes_mes.columns = ['bairro', 'ano_mes', 'score_total', 'n_operacoes', 'score_medio']
    
    # Mesclar
    df_merged = pd.merge(
        crimes_mes,
        apreensoes_mes,
        left_on=['local_oficial', 'ano_mes'],
        right_on=['bairro', 'ano_mes'],
        how='left'
    )
    
    df_merged['score_total'] = df_merged['score_total'].fillna(0)
    df_merged['n_operacoes'] = df_merged['n_operacoes'].fillna(0)
    df_merged['score_medio'] = df_merged['score_medio'].fillna(0)
    
    return df_merged

df_merged = agregar_por_mes(df_crimes_teste, df_raio_matched)

print(f"✓ Dados agregados por mês: {len(df_merged)} observações")

# ============================================================================
# PASSO 7: CALCULAR CORRELAÇÕES
# ============================================================================
print("\n" + "="*80)
print("PASSO 7: ANÁLISE DE CORRELAÇÃO")
print("="*80)

# Correlações
corr_crimes_ops = df_merged[['crimes', 'n_operacoes']].corr().iloc[0, 1]
corr_crimes_score = df_merged[['crimes', 'score_total']].corr().iloc[0, 1]
corr_crimes_score_medio = df_merged[['crimes', 'score_medio']].corr().iloc[0, 1]

print(f"\n📊 CORRELAÇÃO COM CRIMES (mensal):")
print(f"  Crimes × N. Operações: {corr_crimes_ops:+.4f}")
print(f"  Crimes × Score Total: {corr_crimes_score:+.4f}")
print(f"  Crimes × Score Médio: {corr_crimes_score_medio:+.4f}")

if abs(corr_crimes_score) > abs(corr_crimes_ops):
    print(f"\n✅ Score de apreensão é MELHOR preditor que contagem de operações!")
    print(f"   Ganho: {(abs(corr_crimes_score) - abs(corr_crimes_ops))*100:.1f}%")
else:
    print(f"\n❌ Score de apreensão não melhora correlação")

# Bairros com maior cobertura
print(f"\n  Top 10 Bairros (por meses com dados RAIO):")
print(f"  {'Bairro':<25} {'Meses':<8} {'Score Médio':<15} {'Corr.':<8}")
print(f"  {'-'*56}")

bairro_stats = df_merged[df_merged['score_total'] > 0].groupby('local_oficial').agg({
    'score_total': ['count', 'mean'],
    'crimes': 'mean'
}).reset_index()

bairro_stats.columns = ['bairro', 'n_meses', 'score_medio', 'crimes_medio']

for idx, row in bairro_stats.nlargest(10, 'n_meses').iterrows():
    # Calcular correlação por bairro
    df_b = df_merged[df_merged['local_oficial'] == row['bairro']]
    if len(df_b) > 1:
        corr_b = df_b[['crimes', 'score_total']].corr().iloc[0, 1]
    else:
        corr_b = 0.0
    
    print(f"  {row['bairro']:<25} {row['n_meses']:<8.0f} {row['score_medio']:<15.0f} {corr_b:+8.3f}")

# ============================================================================
# PASSO 8: TESTAR MODELO COM SCORE DE APREENSÃO
# ============================================================================
print("\n" + "="*80)
print("PASSO 8: TESTAR MODELO COM SCORE DE APREENSÃO")
print("="*80)

class ModeloComScore:
    """Modelo usando score de apreensão como exógena"""
    
    def prever(self, df_obs):
        predicoes = []
        
        for idx, row in df_obs.iterrows():
            crimes_base = row['crimes']
            score = row['score_total']
            n_ops = row['n_operacoes']
            
            # Modelo: redução proporcional ao score
            # Cada 100 pontos = 5% redução
            reducao = min((score / 100) * 0.05, 0.30)  # Máximo 30%
            
            crimes_pred = crimes_base * (1 - reducao)
            
            predicoes.append({
                'real': crimes_base,
                'pred': max(crimes_pred, 0)
            })
        
        return pd.DataFrame(predicoes)

class ModeloComOps:
    """Modelo usando contagem de operações como exógena"""
    
    def prever(self, df_obs):
        predicoes = []
        
        for idx, row in df_obs.iterrows():
            crimes_base = row['crimes']
            n_ops = row['n_operacoes']
            
            # Modelo: redução proporcional ao número de ops
            reducao = min(n_ops * 0.02, 0.20)  # Máximo 20%
            
            crimes_pred = crimes_base * (1 - reducao)
            
            predicoes.append({
                'real': crimes_base,
                'pred': max(crimes_pred, 0)
            })
        
        return pd.DataFrame(predicoes)

class ModeloBaseline:
    """Modelo sem exógenas"""
    
    def prever(self, df_obs):
        return pd.DataFrame({
            'real': df_obs['crimes'],
            'pred': df_obs['crimes'].values  # Sem mudança
        })

# Fazer predições
modelo_baseline = ModeloBaseline()
pred_baseline = modelo_baseline.prever(df_merged)
mae_baseline = mean_absolute_error(pred_baseline['real'], pred_baseline['pred'])
r2_baseline = r2_score(pred_baseline['real'], pred_baseline['pred'])

modelo_ops = ModeloComOps()
pred_ops = modelo_ops.prever(df_merged)
mae_ops = mean_absolute_error(pred_ops['real'], pred_ops['pred'])
r2_ops = r2_score(pred_ops['real'], pred_ops['pred'])

modelo_score = ModeloComScore()
pred_score = modelo_score.prever(df_merged)
mae_score = mean_absolute_error(pred_score['real'], pred_score['pred'])
r2_score_val = r2_score(pred_score['real'], pred_score['pred'])

print(f"\n📊 COMPARAÇÃO DE MODELOS:")
print(f"{'Modelo':<25} {'MAE':<12} {'R²':<12} {'vs Baseline':<15}")
print("-" * 64)
print(f"{'Baseline':<25} {mae_baseline:<12.2f} {r2_baseline:<12.4f} {'—':<15}")
print(f"{'Com N. Operações':<25} {mae_ops:<12.2f} {r2_ops:<12.4f} {((r2_ops-r2_baseline)/max(abs(r2_baseline),0.01)*100):+14.1f}%")
print(f"{'Com Score Apreensão':<25} {mae_score:<12.2f} {r2_score_val:<12.4f} {((r2_score_val-r2_baseline)/max(abs(r2_baseline),0.01)*100):+14.1f}%")

# ============================================================================
# PASSO 9: EXPORTAR ANÁLISE
# ============================================================================
print("\n" + "="*80)
print("PASSO 9: EXPORTAR RESULTADOS")
print("="*80)

analise = {
    "titulo": "Análise de Apreensões Significativas - RAIO",
    "data": datetime.now().isoformat(),
    "resumo_apreensoes": {
        "armas_total": armas_count,
        "operacoes_drogas": drogas_op,
        "veiculos": veics_count,
        "dinheiro_apreendido": dinheiro_total
    },
    "distribuicao_relevancia": relevancia_counts.to_dict(),
    "correlacoes": {
        "crimes_vs_n_operacoes": round(corr_crimes_ops, 4),
        "crimes_vs_score_total": round(corr_crimes_score, 4),
        "crimes_vs_score_medio": round(corr_crimes_score_medio, 4),
        "melhor_preditor": "Score Apreensão" if abs(corr_crimes_score) > abs(corr_crimes_ops) else "N. Operações"
    },
    "performance_modelo": {
        "baseline": {
            "MAE": round(mae_baseline, 2),
            "R2": round(r2_baseline, 4)
        },
        "com_operacoes": {
            "MAE": round(mae_ops, 2),
            "R2": round(r2_ops, 4),
            "melhoria_r2": round((r2_ops - r2_baseline) / max(abs(r2_baseline), 0.01) * 100, 1)
        },
        "com_score_apreensao": {
            "MAE": round(mae_score, 2),
            "R2": round(r2_score_val, 4),
            "melhoria_r2": round((r2_score_val - r2_baseline) / max(abs(r2_baseline), 0.01) * 100, 1)
        }
    },
    "recomendacao": "Usar Score de Apreensão" if abs(corr_crimes_score) > abs(corr_crimes_ops) else "Usar N. Operações" if abs(corr_crimes_ops) > 0.01 else "Continuar com baseline"
}

output_path = Path(__file__).parent / "analise_apreensoes_significativas.json"
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(analise, f, indent=2, ensure_ascii=False)

print(f"\n✓ Análise salva: {output_path}")

# ============================================================================
# VISUALIZAÇÃO
# ============================================================================
print("\n" + "="*80)
print("RESUMO VISUAL")
print("="*80)

print(f"""
┌────────────────────────────────────────────────────────────────┐
│     ANÁLISE DE APREENSÕES SIGNIFICATIVAS - RAIO               │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│ APREENSÕES TOTAIS:                                             │
│   Armas: {armas_count:>6} | Drogas (ops): {drogas_op:>4} | Veículos: {veics_count:>6} │
│   Dinheiro: R$ {dinheiro_total:>12,.0f}                    │
│                                                                │
│ DISTRIBUIÇÃO DE RELEVÂNCIA:                                    │
│   Muito Alta: {relevancia_counts.get('Muito Alta', 0):>5} | Alta: {relevancia_counts.get('Alta', 0):>5} | Média: {relevancia_counts.get('Média', 0):>5}    │
│                                                                │
│ CORRELAÇÃO COM CRIMES:                                         │
│   N. Operações: {corr_crimes_ops:+.4f}                        │
│   Score Apreensão: {corr_crimes_score:+.4f}                  │
│                                                                │
│ RECOMENDAÇÃO:                                                  │
│ {"✅ Score Apreensão melhora predição" if abs(corr_crimes_score) > abs(corr_crimes_ops) else "⚠️ Nenhuma correlação significativa"}       │
│                                                                │
└────────────────────────────────────────────────────────────────┘
""")

print("\n" + "="*80)
print("✓ ANÁLISE CONCLUÍDA")
print("="*80)
