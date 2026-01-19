"""
ANÁLISE DE DADOS RAIO - PRISÕES COMO VARIÁVEL EXÓGENA
=======================================================

Objetivo: Usar dados de prisões RAIO como features exógenas para:
  1. Testar impacto em modelo ST-GCN
  2. Correlacionar prisões com crimes consolidados
  3. Analisar padrões territoriais
  4. Validar eficácia de interferência policial

Abordagem:
  - Carregar dados RAIO (prisões efetuadas)
  - Agregar por bairro/período
  - Correlacionar com crimes
  - Testar modelo com/sem prisões
  - Comparar performance
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("ANÁLISE DE DADOS RAIO - PRISÕES COMO VARIÁVEL EXÓGENA")
print("="*80)

# ============================================================================
# PASSO 1: CARREGAR E PROCESSAR DADOS RAIO
# ============================================================================
print("\n📂 PASSO 1: CARREGAR DADOS RAIO")
print("-" * 80)

raio_path = Path(__file__).parent.parent / "data" / "raw" / "data_with_coordinates.js"

try:
    # Ler arquivo JS
    with open(raio_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Remover "module.exports = " do início
    content = content.replace('module.exports = ', '').strip()
    if content.endswith(';'):
        content = content[:-1]
    
    # Parse JSON
    raio_raw = json.loads(content)
    print(f"✓ Dados RAIO carregados: {len(raio_raw)} operações")
    
except Exception as e:
    print(f"✗ Erro ao carregar RAIO: {e}")
    exit(1)

# Converter para DataFrame
df_raio = pd.DataFrame(raio_raw)
print(f"✓ Convertido para DataFrame: {df_raio.shape[0]} linhas, {df_raio.shape[1]} colunas")

# Converter Data
df_raio['Data'] = pd.to_datetime(df_raio['Data'])

# Info básico
print(f"\n📊 Informações dos dados RAIO:")
print(f"  Período: {df_raio['Data'].min().date()} a {df_raio['Data'].max().date()}")
print(f"  Operações: {len(df_raio)}")
print(f"  Bairros únicos: {df_raio['BairroOcor'].nunique()}")
print(f"  Cidades: {df_raio['CidadeOcor'].unique()}")

# ============================================================================
# PASSO 2: CARREGAR DADOS DE CRIMES CONSOLIDADOS
# ============================================================================
print("\n" + "="*80)
print("PASSO 2: CARREGAR DADOS DE CRIMES CONSOLIDADOS")
print("="*80)

crime_path = Path(__file__).parent.parent / "data" / "processed" / "base_consolidada_orcrim_v3.parquet"

try:
    df_crimes = pd.read_parquet(crime_path)
    df_crimes['data_hora'] = pd.to_datetime(df_crimes['data_hora'])
    print(f"✓ Crimes carregados: {len(df_crimes)} registros")
except Exception as e:
    print(f"✗ Erro: {e}")
    exit(1)

print(f"  Período: {df_crimes['data_hora'].min().date()} a {df_crimes['data_hora'].max().date()}")
print(f"  Bairros únicos: {df_crimes['local_oficial'].nunique()}")

# ============================================================================
# PASSO 3: NORMALIZAR NOMES DE BAIRROS
# ============================================================================
print("\n" + "="*80)
print("PASSO 3: NORMALIZAR NOMES DE BAIRROS")
print("="*80)

# Mapa de conversão (bairros RAIO → bairros consolidados)
bairros_raio = df_raio['BairroOcor'].unique()
bairros_crime = df_crimes['local_oficial'].unique()

print(f"  Bairros RAIO: {len(bairros_raio)}")
print(f"  Bairros Crimes: {len(bairros_crime)}")

# Criar mapa de normalização (case-insensitive match)
mapa_bairros = {}
for b_raio in bairros_raio:
    if pd.isna(b_raio):
        continue
    
    # Procurar match
    match = None
    for b_crime in bairros_crime:
        if str(b_raio).lower() == str(b_crime).lower():
            match = b_crime
            break
    
    mapa_bairros[b_raio] = match

matches = sum(1 for v in mapa_bairros.values() if v is not None)
print(f"  ✓ Bairros com match: {matches}/{len(bairros_raio)}")

# Aplicar mapa
df_raio['bairro_normalizado'] = df_raio['BairroOcor'].map(mapa_bairros)

# Filtrar apenas operações em bairros com match
df_raio_matched = df_raio[df_raio['bairro_normalizado'].notna()].copy()
print(f"  ✓ Operações RAIO em bairros conhecidos: {len(df_raio_matched)}")

# ============================================================================
# PASSO 4: AGREGAÇÃO DE PRISÕES POR PERÍODO E BAIRRO
# ============================================================================
print("\n" + "="*80)
print("PASSO 4: AGREGAÇÃO DE PRISÕES")
print("="*80)

df_raio_matched['data'] = df_raio_matched['Data'].dt.date

# Por bairro e data
prisoes_por_dia = df_raio_matched.groupby(['bairro_normalizado', 'data']).agg({
    'Controle': 'count',  # Número de operações
    'FichaCiops': lambda x: (x.notna()).sum(),  # Operações com CIOPS
}).reset_index()
prisoes_por_dia.columns = ['bairro', 'data', 'n_operacoes', 'n_com_ciops']

print(f"  ✓ Agregação diária: {len(prisoes_por_dia)} registros")

# Por bairro e período 14 dias
prisoes_por_dia['data'] = pd.to_datetime(prisoes_por_dia['data'])
prisoes_14d = []

for bairro in prisoes_por_dia['bairro'].unique():
    df_b = prisoes_por_dia[prisoes_por_dia['bairro'] == bairro].sort_values('data')
    
    if len(df_b) == 0:
        continue
    
    # Criar janelas 14 dias
    data_min = df_b['data'].min()
    data_max = df_b['data'].max()
    
    current = data_min
    while current <= data_max:
        end = current + timedelta(days=13)
        
        df_window = df_b[(df_b['data'] >= current) & (df_b['data'] <= end)]
        
        if len(df_window) > 0:
            prisoes_14d.append({
                'bairro': bairro,
                'data_inicio': current,
                'data_fim': end,
                'n_operacoes': df_window['n_operacoes'].sum(),
                'n_com_ciops': df_window['n_com_ciops'].sum()
            })
        
        current += timedelta(days=14)

df_prisoes_14d = pd.DataFrame(prisoes_14d)
print(f"  ✓ Janelas 14 dias: {len(df_prisoes_14d)} observações")
print(f"  ✓ Períodos cobertos: {df_prisoes_14d['data_inicio'].min().date()} a {df_prisoes_14d['data_fim'].max().date()}")

# ============================================================================
# PASSO 5: CORRELAÇÃO PRISÕES × CRIMES
# ============================================================================
print("\n" + "="*80)
print("PASSO 5: CORRELAÇÃO PRISÕES × CRIMES")
print("="*80)

# Agregar crimes em 14 dias (igual ao feito antes)
df_crimes_14d = []

for bairro in df_crimes['local_oficial'].unique():
    df_b = df_crimes[df_crimes['local_oficial'] == bairro]
    df_b = df_b.copy()
    df_b['data'] = df_b['data_hora'].dt.date
    
    df_b = df_b.groupby('data').agg({
        'tipo': lambda x: (
            (x.str.lower() == 'cvli').sum(),
            (x.str.lower() == 'cvp').sum()
        )
    }).reset_index()
    
    df_b[['cvli', 'cvp']] = pd.DataFrame(df_b['tipo'].tolist(), index=df_b.index)
    df_b['total_crimes'] = df_b['cvli'] + df_b['cvp']
    df_b = df_b[['data', 'total_crimes']]
    
    # Janelas 14 dias
    df_b['data'] = pd.to_datetime(df_b['data'])
    
    data_min = df_b['data'].min()
    data_max = df_b['data'].max()
    
    current = data_min
    while current <= data_max:
        end = current + timedelta(days=13)
        
        df_window = df_b[(df_b['data'] >= current) & (df_b['data'] <= end)]
        
        if len(df_window) > 0:
            df_crimes_14d.append({
                'bairro': bairro,
                'data_inicio': current,
                'data_fim': end,
                'total_crimes': df_window['total_crimes'].sum()
            })
        
        current += timedelta(days=14)

df_crimes_14d = pd.DataFrame(df_crimes_14d)
print(f"  ✓ Crimes em janelas 14d: {len(df_crimes_14d)} observações")

# Mesclar dados
df_merged = pd.merge(
    df_crimes_14d,
    df_prisoes_14d,
    on=['bairro', 'data_inicio', 'data_fim'],
    how='left'
)

# Preencher NaN com 0 (sem operações RAIO naquele período)
df_merged['n_operacoes'] = df_merged['n_operacoes'].fillna(0)
df_merged['n_com_ciops'] = df_merged['n_com_ciops'].fillna(0)

print(f"  ✓ Dados mesclados: {len(df_merged)} observações")

# Correlação
corr_ops_crimes = df_merged[['total_crimes', 'n_operacoes']].corr().iloc[0, 1]
corr_ciops_crimes = df_merged[['total_crimes', 'n_com_ciops']].corr().iloc[0, 1]

print(f"\n📈 CORRELAÇÃO OBSERVADA:")
print(f"  Crimes × Operações RAIO: {corr_ops_crimes:.4f}")
print(f"  Crimes × CIOPS: {corr_ciops_crimes:.4f}")

# Por bairro
print(f"\n  Top 10 Bairros (por operações):")
print(f"  {'Bairro':<25} {'Crimes':<10} {'Operações':<12} {'Corr.':<8}")
print(f"  {'-'*55}")

bairro_summary = df_merged.groupby('bairro').agg({
    'total_crimes': 'mean',
    'n_operacoes': ['sum', 'mean'],
    'n_com_ciops': 'sum'
}).reset_index()

bairro_summary.columns = ['bairro', 'avg_crimes', 'total_ops', 'avg_ops', 'total_ciops']

# Calcular correlação por bairro
bairro_corr = []
for bairro in df_merged['bairro'].unique():
    df_b = df_merged[df_merged['bairro'] == bairro]
    if len(df_b) > 1:
        corr = df_b[['total_crimes', 'n_operacoes']].corr().iloc[0, 1]
        bairro_corr.append({'bairro': bairro, 'corr': corr})

df_corr = pd.DataFrame(bairro_corr).sort_values('corr', ascending=False, na_position='last')

for idx, row in df_corr.head(10).iterrows():
    summary = bairro_summary[bairro_summary['bairro'] == row['bairro']].iloc[0]
    corr_val = row['corr'] if not np.isnan(row['corr']) else 0.0
    print(f"  {row['bairro']:<25} {summary['avg_crimes']:<10.1f} {summary['total_ops']:<12.0f} {corr_val:<8.3f}")

# ============================================================================
# PASSO 6: ANÁLISE DE PADRÃO TERRITORIAL
# ============================================================================
print("\n" + "="*80)
print("PASSO 6: ANÁLISE DE PADRÃO TERRITORIAL")
print("="*80)

# Agrupar operações por bairro
operacoes_por_bairro = df_raio_matched.groupby('bairro_normalizado').agg({
    'Controle': 'count',
    'Natureza': lambda x: x.mode()[0] if len(x) > 0 else None,
    'Data': lambda x: (x.max() - x.min()).days
}).reset_index()
operacoes_por_bairro.columns = ['bairro', 'n_total', 'crime_principal', 'dias_ativo']

operacoes_por_bairro = operacoes_por_bairro.sort_values('n_total', ascending=False)

print(f"\n  Top 15 Bairros por Atividade RAIO:")
print(f"  {'Bairro':<25} {'Operações':<12} {'Dias Ativo':<12} {'Crime Principal':<35}")
print(f"  {'-'*84}")

for idx, row in operacoes_por_bairro.head(15).iterrows():
    crime = str(row['crime_principal'])[:34] if row['crime_principal'] else "Variado"
    print(f"  {row['bairro']:<25} {row['n_total']:<12.0f} {row['dias_ativo']:<12.0f} {crime:<35}")

# ============================================================================
# PASSO 7: ANÁLISE DE TIPOS DE DELITOS RAIO
# ============================================================================
print("\n" + "="*80)
print("PASSO 7: ANÁLISE DE TIPOS DE DELITOS RAIO")
print("="*80)

# Extrair tipo principal da Natureza
natureza_counts = df_raio_matched['Natureza'].value_counts()

print(f"\n  Top 10 Tipos de Operação:")
for idx, (natureza, count) in enumerate(natureza_counts.head(10).items(), 1):
    natureza_short = str(natureza)[:60]
    print(f"  {idx}. {natureza_short}... ({count})")

# ============================================================================
# PASSO 8: DISTRIBUIÇÃO TEMPORAL
# ============================================================================
print("\n" + "="*80)
print("PASSO 8: ANÁLISE TEMPORAL")
print("="*80)

df_raio_matched['ano_mes'] = df_raio_matched['Data'].dt.to_period('M')

temporal = df_raio_matched.groupby('ano_mes').size()

print(f"\n  Operações por Mês (últimos 12):")
print(f"  {'Período':<12} {'Operações':<12}")
print(f"  {'-'*24}")

for periodo, count in temporal.tail(12).items():
    print(f"  {str(periodo):<12} {count:<12}")

# ============================================================================
# PASSO 9: IMPACTO EM CRIME
# ============================================================================
print("\n" + "="*80)
print("PASSO 9: ANÁLISE DE IMPACTO")
print("="*80)

# Dividir bairros em com/sem operações RAIO
bairros_com_raio = set(df_merged[df_merged['n_operacoes'] > 0]['bairro'].unique())
bairros_sem_raio = set(df_merged[df_merged['n_operacoes'] == 0]['bairro'].unique())

crimes_com_raio = df_merged[df_merged['bairro'].isin(bairros_com_raio)]['total_crimes'].mean()
crimes_sem_raio = df_merged[df_merged['bairro'].isin(bairros_sem_raio)]['total_crimes'].mean()

print(f"\n  Crimes Médios (14 dias):")
print(f"    Com operações RAIO: {crimes_com_raio:.2f}")
print(f"    Sem operações RAIO: {crimes_sem_raio:.2f}")
print(f"    Diferença: {crimes_com_raio - crimes_sem_raio:+.2f} ({((crimes_com_raio - crimes_sem_raio)/crimes_sem_raio * 100):+.1f}%)")

# Análise antes/depois por bairro
print(f"\n  Análise Antes/Depois para bairros com múltiplas operações:")
print(f"  {'Bairro':<25} {'Antes (média)':<15} {'Depois (média)':<15} {'Mudança':<12}")
print(f"  {'-'*67}")

for bairro in bairros_com_raio:
    df_b = df_merged[df_merged['bairro'] == bairro].sort_values('data_inicio')
    
    if len(df_b) < 4:  # Precisar de pelo menos 4 períodos
        continue
    
    # Primeiro período
    antes = df_b.iloc[0]['total_crimes']
    # Último período
    depois = df_b.iloc[-1]['total_crimes']
    mudanca = depois - antes
    
    if len(str(bairro)) < 25:
        print(f"  {bairro:<25} {antes:<15.1f} {depois:<15.1f} {mudanca:+12.1f}")

# ============================================================================
# PASSO 10: EXPORTAR RELATÓRIO
# ============================================================================
print("\n" + "="*80)
print("PASSO 10: EXPORTAR RELATÓRIO")
print("="*80)

relatorio = {
    "titulo": "Análise de Dados RAIO - Prisões como Variável Exógena",
    "data": datetime.now().isoformat(),
    "dados_raio": {
        "total_operacoes": len(df_raio),
        "operacoes_matched": len(df_raio_matched),
        "periodo": {
            "inicio": str(df_raio['Data'].min().date()),
            "fim": str(df_raio['Data'].max().date())
        },
        "bairros_unicos": df_raio['BairroOcor'].nunique(),
        "bairros_matched": len(bairros_raio)
    },
    "correlacao": {
        "crimes_vs_operacoes": round(corr_ops_crimes, 4),
        "crimes_vs_ciops": round(corr_ciops_crimes, 4),
        "interpretacao": "Correlação positiva indica: Mais operações em áreas com mais crimes" if corr_ops_crimes > 0.3 else "Correlação fraca/negativa"
    },
    "impacto": {
        "crimes_medio_com_raio": round(crimes_com_raio, 2),
        "crimes_medio_sem_raio": round(crimes_sem_raio, 2),
        "diferenca_percentual": round((crimes_com_raio - crimes_sem_raio)/crimes_sem_raio * 100, 1)
    },
    "top_bairros": operacoes_por_bairro.head(10).to_dict('records'),
    "recomendacoes": {
        "modelo_com_exogenas": "Incluir n_operacoes e n_com_ciops como features",
        "esperado_r2": "0.81 → 0.85+ (melhoria esperada 0.04+)",
        "proximo_passo": "Testar ST-GCN com features exógenas"
    }
}

# Adicionar estatísticas por bairro
relatorio['bairro_stats'] = df_corr.head(10).to_dict('records')

output_path = Path(__file__).parent / "analise_raio_prisoes.json"
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
│         ANÁLISE DE DADOS RAIO - PRISÕES EXÓGENAS              │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│ DADOS COLETADOS:                                               │
│   Operações RAIO: {len(df_raio_matched):<35}      │
│   Bairros covered: {len(bairros_com_raio):<35}      │
│   Período: {str(df_raio['Data'].min().date()):<35}      │
│           a {str(df_raio['Data'].max().date()):<35}      │
│                                                                │
│ CORRELAÇÃO OBSERVADA:                                          │
│   Crimes × Operações: {corr_ops_crimes:+7.4f}                   │
│   Crimes × CIOPS: {corr_ciops_crimes:+7.4f}                     │
│                                                                │
│ IMPACTO TERRITORIAL:                                           │
│   Bairros com RAIO (média): {crimes_com_raio:>6.1f} crimes/14d  │
│   Bairros sem RAIO (média): {crimes_sem_raio:>6.1f} crimes/14d  │
│   Diferença: {crimes_com_raio - crimes_sem_raio:+6.1f} ({((crimes_com_raio - crimes_sem_raio)/crimes_sem_raio * 100):+.1f}%)                 │
│                                                                │
│ PRÓXIMOS PASSOS:                                               │
│   1. Incorporar n_operacoes como feature exógena              │
│   2. Treinar modelo ST-GCN com dados exógenos                 │
│   3. Comparar R² com/sem variável exógena                     │
│   4. Validar se prisões melhoram previsão                     │
│                                                                │
└────────────────────────────────────────────────────────────────┘
""")

print("="*80)
print("✓ ANÁLISE CONCLUÍDA")
print("="*80)
