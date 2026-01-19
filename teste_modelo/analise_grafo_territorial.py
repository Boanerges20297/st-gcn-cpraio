"""
ANÁLISE DO GRAFO TERRITORIAL - PRISÕES E VIZINHANÇA
====================================================

Objetivo: Validar como o grafo ST-GCN se comporta com dados exógenos
  1. Modelar vizinhança entre bairros
  2. Testar propagação de prisões no grafo
  3. Validar se prisões em bairros vizinhos afetam risco central
  4. Comparar modelo com/sem estrutura de grafo

Hipótese: Prisões em vizinhança reduz crimes no bairro alvo
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from scipy.spatial.distance import cdist
from sklearn.metrics import mean_absolute_error, r2_score
import json
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("ANÁLISE DO GRAFO TERRITORIAL - VIZINHANÇA E PROPAGAÇÃO")
print("="*80)

# ============================================================================
# PASSO 1: CARREGAR DADOS
# ============================================================================
print("\n📂 PASSO 1: CARREGAR DADOS")
print("-" * 80)

crime_path = Path(__file__).parent.parent / "data" / "processed" / "base_consolidada_orcrim_v3.parquet"
df_crimes = pd.read_parquet(crime_path)
df_crimes['data_hora'] = pd.to_datetime(df_crimes['data_hora'])

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
# PASSO 2: EXTRAIR COORDENADAS (Se disponível)
# ============================================================================
print("\n" + "="*80)
print("PASSO 2: EXTRAIR BAIRROS E COORDENADAS")
print("="*80)

# Bairros únicos consolidados
bairros_crime = df_crimes['local_oficial'].unique()
print(f"✓ Bairros consolidados: {len(bairros_crime)}")

# Tentar carregar coordenadas do arquivo de grafo (se existir)
grafo_path = Path(__file__).parent.parent / "data" / "graph" / "bairros_coordenadas.json"

if grafo_path.exists():
    with open(grafo_path, 'r', encoding='utf-8') as f:
        coordenadas = json.load(f)
    print(f"✓ Coordenadas carregadas: {len(coordenadas)} bairros")
else:
    print(f"⚠️  Arquivo de coordenadas não encontrado")
    print(f"   Usando bairros consolidados como referência")
    coordenadas = {b: {"lat": 0, "lon": 0} for b in bairros_crime}

# ============================================================================
# PASSO 3: CONSTRUIR MATRIZ DE VIZINHANÇA
# ============================================================================
print("\n" + "="*80)
print("PASSO 3: CONSTRUIR MATRIZ DE VIZINHANÇA")
print("="*80)

# Bairros com coordenadas
bairros_com_coords = list(coordenadas.keys())

# Calcular distâncias
print("  Calculando distâncias...")

# Usando bairros consolidados como base
bairros_usados = []
matriz_distancia = []

for b1 in bairros_crime:
    if b1 in bairros_com_coords:
        bairros_usados.append(b1)

if len(bairros_usados) > 0:
    # Extrair coordenadas
    coords = np.array([
        [coordenadas[b].get('lat', 0), coordenadas[b].get('lon', 0)] 
        for b in bairros_usados
    ])
    
    # Calcular distâncias (distância euclidiana simples)
    from scipy.spatial.distance import pdist, squareform
    dist_matrix = squareform(pdist(coords, metric='euclidean'))
    
    print(f"✓ Matriz de distância: {dist_matrix.shape}")
    print(f"  Distância média: {np.mean(dist_matrix[dist_matrix > 0]):.2f}")
    print(f"  Distância máxima: {np.max(dist_matrix):.2f}")
    
    # Vizinhos: até 2 desvios padrão de distância
    dist_threshold = np.mean(dist_matrix[dist_matrix > 0]) + 1.5 * np.std(dist_matrix[dist_matrix > 0])
    
    # Construir grafo de adjacência
    grafo_adjacencia = {}
    for i, b1 in enumerate(bairros_usados):
        vizinhos = []
        for j, b2 in enumerate(bairros_usados):
            if i != j and dist_matrix[i, j] < dist_threshold:
                vizinhos.append((b2, dist_matrix[i, j]))
        
        if vizinhos:
            grafo_adjacencia[b1] = sorted(vizinhos, key=lambda x: x[1])
    
    print(f"\n✓ Grafo construído: {len(grafo_adjacencia)} bairros com vizinhos")
    print(f"  Threshold de vizinhança: {dist_threshold:.2f}")
    
    # Estatísticas
    graus = [len(v) for v in grafo_adjacencia.values()]
    print(f"  Grau médio: {np.mean(graus):.1f}")
    print(f"  Grau máximo: {np.max(graus) if graus else 0}")
    
else:
    print("⚠️  Sem coordenadas válidas - usando bairros por nome")
    grafo_adjacencia = {}

# ============================================================================
# PASSO 4: NORMALIZAR E AGREGAR DADOS
# ============================================================================
print("\n" + "="*80)
print("PASSO 4: AGREGAR CRIMES E PRISÕES")
print("="*80)

# Normalizar RAIO
bairros_raio = df_raio['BairroOcor'].unique()
bairros_crime_list = df_crimes['local_oficial'].unique()

mapa_bairros = {}
for b_raio in bairros_raio:
    if pd.isna(b_raio):
        continue
    match = None
    for b_crime in bairros_crime_list:
        if str(b_raio).lower() == str(b_crime).lower():
            match = b_crime
            break
    mapa_bairros[b_raio] = match

df_raio['bairro_normalizado'] = df_raio['BairroOcor'].map(mapa_bairros)
df_raio_matched = df_raio[df_raio['bairro_normalizado'].notna()].copy()

print(f"✓ RAIO: {len(df_raio_matched)} operações em bairros conhecidos")

# Agregar por período 14 dias
def agregar_dados(df_crime, df_prisoes):
    treino_inicio = pd.Timestamp('2022-01-01')
    treino_fim = pd.Timestamp('2023-12-31')
    teste_inicio = pd.Timestamp('2024-01-01')
    teste_fim = pd.Timestamp('2025-12-31')
    
    df_crime_teste = df_crime[
        (df_crime['data_hora'] >= teste_inicio) & 
        (df_crime['data_hora'] <= teste_fim)
    ]
    
    df_crime_teste['data'] = df_crime_teste['data_hora'].dt.date
    
    crimes_14d = []
    for bairro in df_crime_teste['local_oficial'].unique():
        df_b = df_crime_teste[df_crime_teste['local_oficial'] == bairro]
        df_agg = df_b.groupby('data').size().reset_index(name='crimes')
        df_agg['data'] = pd.to_datetime(df_agg['data'])
        
        data_min = df_agg['data'].min()
        data_max = df_agg['data'].max()
        
        current = data_min
        while current <= data_max:
            end = current + timedelta(days=13)
            df_w = df_agg[(df_agg['data'] >= current) & (df_agg['data'] <= end)]
            
            if len(df_w) > 0:
                crimes_14d.append({
                    'bairro': bairro,
                    'data_inicio': current,
                    'crimes': df_w['crimes'].sum()
                })
            
            current += timedelta(days=14)
    
    df_crimes_14d = pd.DataFrame(crimes_14d)
    
    # Agregar prisões
    df_prisoes['data'] = df_prisoes['Data'].dt.date
    prisoes_14d = []
    
    for bairro in df_prisoes['bairro_normalizado'].unique():
        df_b = df_prisoes[df_prisoes['bairro_normalizado'] == bairro]
        df_agg = df_b.groupby('data').size().reset_index(name='prisoes')
        df_agg['data'] = pd.to_datetime(df_agg['data'])
        
        data_min = df_agg['data'].min()
        data_max = df_agg['data'].max()
        
        current = data_min
        while current <= data_max:
            end = current + timedelta(days=13)
            df_w = df_agg[(df_agg['data'] >= current) & (df_agg['data'] <= end)]
            
            if len(df_w) > 0:
                prisoes_14d.append({
                    'bairro': bairro,
                    'data_inicio': current,
                    'prisoes': df_w['prisoes'].sum()
                })
            
            current += timedelta(days=14)
    
    df_prisoes_14d = pd.DataFrame(prisoes_14d)
    
    # Mesclar
    df_merged = pd.merge(
        df_crimes_14d,
        df_prisoes_14d,
        on=['bairro', 'data_inicio'],
        how='left'
    )
    df_merged['prisoes'] = df_merged['prisoes'].fillna(0)
    
    return df_merged

df_dados = agregar_dados(df_crimes, df_raio_matched)
print(f"✓ Dados agregados: {len(df_dados)} observações")

# ============================================================================
# PASSO 5: TESTAR MODELO SEM GRAFO (Baseline)
# ============================================================================
print("\n" + "="*80)
print("PASSO 5: MODELO SEM PROPAGAÇÃO (Baseline)")
print("="*80)

class ModeloSemGrafo:
    def prever(self, df_obs):
        """Predição sem considerar vizinhança"""
        predicoes = []
        for idx, row in df_obs.iterrows():
            # Modelo simples: crimes_predict = crimes_histórico * fator_prisões
            crimes_pred = row['crimes'] * 0.95 if row['prisoes'] > 0 else row['crimes']
            predicoes.append({
                'real': row['crimes'],
                'pred': max(crimes_pred, 0),
                'bairro': row['bairro']
            })
        return pd.DataFrame(predicoes)

modelo_sem_grafo = ModeloSemGrafo()
pred_sem_grafo = modelo_sem_grafo.prever(df_dados)
mae_sem_grafo = mean_absolute_error(pred_sem_grafo['real'], pred_sem_grafo['pred'])
r2_sem_grafo = r2_score(pred_sem_grafo['real'], pred_sem_grafo['pred'])

print(f"✓ Baseline (sem grafo): MAE={mae_sem_grafo:.2f}, R²={r2_sem_grafo:.4f}")

# ============================================================================
# PASSO 6: TESTAR MODELO COM PROPAGAÇÃO NO GRAFO
# ============================================================================
print("\n" + "="*80)
print("PASSO 6: MODELO COM PROPAGAÇÃO NO GRAFO")
print("="*80)

class ModeloComGrafo:
    def __init__(self, grafo_adjacencia):
        self.grafo = grafo_adjacencia
    
    def prever(self, df_obs):
        """Predição com propagação em vizinhança"""
        predicoes = []
        
        for idx, row in df_obs.iterrows():
            bairro = row['bairro']
            crimes_base = row['crimes']
            prisoes_local = row['prisoes']
            
            # Efeito local
            efeito_local = 0.95 if prisoes_local > 0 else 1.0
            
            # Efeito da vizinhança (se existe no grafo)
            efeito_vizinhanca = 1.0
            if bairro in self.grafo:
                vizinhos = self.grafo[bairro]
                
                # Buscar prisões em vizinhos
                prisoes_vizinhos = 0
                for viz, dist in vizinhos[:3]:  # Top 3 vizinhos mais próximos
                    df_viz = df_obs[
                        (df_obs['bairro'] == viz) & 
                        (df_obs['data_inicio'] == row['data_inicio'])
                    ]
                    if len(df_viz) > 0:
                        prisoes_vizinhos += df_viz.iloc[0]['prisoes']
                
                # Reduzir risco se há prisões na vizinhança
                if prisoes_vizinhos > 0:
                    efeito_vizinhanca = 0.97  # Redução menor (-3%)
            
            crimes_pred = crimes_base * efeito_local * efeito_vizinhanca
            
            predicoes.append({
                'real': crimes_base,
                'pred': max(crimes_pred, 0),
                'bairro': bairro
            })
        
        return pd.DataFrame(predicoes)

modelo_com_grafo = ModeloComGrafo(grafo_adjacencia)
pred_com_grafo = modelo_com_grafo.prever(df_dados)
mae_com_grafo = mean_absolute_error(pred_com_grafo['real'], pred_com_grafo['pred'])
r2_com_grafo = r2_score(pred_com_grafo['real'], pred_com_grafo['pred'])

print(f"✓ Com grafo (propagação): MAE={mae_com_grafo:.2f}, R²={r2_com_grafo:.4f}")

# ============================================================================
# PASSO 7: COMPARAR MODELOS
# ============================================================================
print("\n" + "="*80)
print("PASSO 7: COMPARAÇÃO DE MODELOS")
print("="*80)

melhoria_mae = ((mae_sem_grafo - mae_com_grafo) / mae_sem_grafo) * 100
melhoria_r2 = ((r2_com_grafo - r2_sem_grafo) / abs(r2_sem_grafo)) * 100

print(f"\n📊 MÉTRICAS:")
print(f"{'Métrica':<15} {'Sem Grafo':<15} {'Com Grafo':<15} {'Melhoria':<15}")
print("-" * 60)
print(f"{'MAE':<15} {mae_sem_grafo:<15.2f} {mae_com_grafo:<15.2f} {melhoria_mae:+15.1f}%")
print(f"{'R²':<15} {r2_sem_grafo:<15.4f} {r2_com_grafo:<15.4f} {melhoria_r2:+15.1f}%")

# ============================================================================
# PASSO 8: ANÁLISE POR BAIRRO
# ============================================================================
print("\n" + "="*80)
print("PASSO 8: ANÁLISE POR BAIRRO")
print("="*80)

print(f"\n  Bairros com Vizinhança Ativa:")
print(f"  {'Bairro':<25} {'Vizinhos':<10} {'Cobertura':<10}")
print(f"  {'-'*45}")

for bairro in list(grafo_adjacencia.keys())[:10]:
    num_vizinhos = len(grafo_adjacencia[bairro])
    
    # Verificar cobertura (% com operações RAIO)
    df_b = df_dados[df_dados['bairro'] == bairro]
    cobertura = (df_b['prisoes'] > 0).sum() / len(df_b) * 100 if len(df_b) > 0 else 0
    
    print(f"  {bairro:<25} {num_vizinhos:<10} {cobertura:<10.1f}%")

# ============================================================================
# PASSO 9: EXPORTAR RESULTADOS
# ============================================================================
print("\n" + "="*80)
print("PASSO 9: EXPORTAR RESULTADOS")
print("="*80)

resultados = {
    "titulo": "Análise do Grafo Territorial com Dados Exógenos",
    "data": datetime.now().isoformat(),
    "grafo": {
        "bairros_com_vizinhanca": len(grafo_adjacencia),
        "conexoes_totais": sum(len(v) for v in grafo_adjacencia.values()),
        "grau_medio": np.mean([len(v) for v in grafo_adjacencia.values()]) if grafo_adjacencia else 0
    },
    "performance": {
        "sem_grafo": {
            "MAE": round(mae_sem_grafo, 2),
            "R2": round(r2_sem_grafo, 4)
        },
        "com_grafo": {
            "MAE": round(mae_com_grafo, 2),
            "R2": round(r2_com_grafo, 4)
        },
        "melhorias": {
            "MAE_percentual": round(melhoria_mae, 1),
            "R2_percentual": round(melhoria_r2, 1)
        }
    },
    "recomendacao": {
        "usar_grafo": "Sim" if melhoria_r2 > 0.1 else "Talvez",
        "confianca": "Alta" if melhoria_mae > 2 else "Média",
        "proximo_passo": "Implementar ST-GCN com estrutura de grafo"
    }
}

output_path = Path(__file__).parent / "analise_grafo_territorial.json"
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
│       ANÁLISE DO GRAFO TERRITORIAL COM DADOS EXÓGENOS          │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│ ESTRUTURA DO GRAFO:                                            │
│   Bairros conectados: {len(grafo_adjacencia):<29}      │
│   Conexões totais: {sum(len(v) for v in grafo_adjacencia.values()):<27}       │
│   Grau médio: {np.mean([len(v) for v in grafo_adjacencia.values()]) if grafo_adjacencia else 0:<27.1f}     │
│                                                                │
│ COMPARAÇÃO DE PERFORMANCE:                                     │
│   Sem Grafo:   MAE={mae_sem_grafo:.2f}, R²={r2_sem_grafo:.4f}          │
│   Com Grafo:   MAE={mae_com_grafo:.2f}, R²={r2_com_grafo:.4f}          │
│   Melhoria:    {melhoria_mae:+.1f}% (MAE), {melhoria_r2:+.1f}% (R²)        │
│                                                                │
│ CONCLUSÃO:                                                     │
│ {"✅ Grafo melhora previsão" if melhoria_r2 > 0.1 else "⚠️ Grafo não melhora em R²"}                    │
│                                                                │
│ PRÓXIMO PASSO:                                                 │
│ {"Implementar ST-GCN com vizinhança" if melhoria_r2 > 0.1 else "Analisar com dados reais do ST-GCN"}     │
│                                                                │
└────────────────────────────────────────────────────────────────┘
""")

print("="*80)
print("✓ ANÁLISE CONCLUÍDA")
print("="*80)
