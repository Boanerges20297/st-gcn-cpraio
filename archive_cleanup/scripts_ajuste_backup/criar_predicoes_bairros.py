import pandas as pd
import geopandas as gpd
from pathlib import Path

print("=" * 80)
print("CRIANDO PREDIÇÕES DISCRIMINADAS POR BAIRRO PARA FORTALEZA")
print("=" * 80)

# 1. Carrega dados de crime
df = pd.read_parquet('data/processed/base_consolidada.parquet')
df_capital = df[df['regiao_sistema'] == 'CAPITAL'].copy()

print(f'\nDados de CAPITAL: {len(df_capital)} registros')
print(f'Locais (local_oficial): {sorted(df_capital["local_oficial"].unique())}')

# 2. Carrega GeoJSON de bairros
gdf_bairros = gpd.read_file('data/graph/fortaleza_bairros.geojson')
print(f'\nBairros em GeoJSON: {len(gdf_bairros)} features')

# 3. Carrega predições atuais (por local_oficial)
pred_atual = pd.read_csv('outputs/reports/pred_capital.csv')
print(f'\nPredições atuais: {len(pred_atual)} locais')
print(f'Locais em predição: {sorted(pred_atual["local"].unique())}')

# 4. Estratégia: mapear cada bairro para seu local_oficial e usar sua predição
# Se um bairro não tem correspondência direta, usar vizinhança

print('\n' + '=' * 80)
print('ESTRATÉGIA DE MAPEAMENTO')
print('=' * 80)

# Normaliza nomes
gdf_bairros['name_upper'] = gdf_bairros['name'].astype(str).str.upper().str.strip()
pred_atual['local_upper'] = pred_atual['local'].astype(str).str.upper().str.strip()

# Merge: cada bairro herda a predição do seu local_oficial
gdf_pred = gdf_bairros.merge(
    pred_atual[['local_upper', 'risco_previsto']],
    left_on='name_upper',
    right_on='local_upper',
    how='left'
)

# Bairros com predição direta
com_pred = gdf_pred[gdf_pred['risco_previsto'].notna()]
print(f'Bairros com predição direta: {len(com_pred)}/{len(gdf_bairros)}')

# Bairros SEM predição direta (precisam de estratégia)
sem_pred = gdf_pred[gdf_pred['risco_previsto'].isna()]
print(f'Bairros SEM predição direta: {len(sem_pred)}')

if len(sem_pred) > 0:
    print(f'\nBairros faltando predição:')
    for idx, row in sem_pred.iterrows():
        print(f'  - {row["name"]}')

# 5. Estratégia para bairros sem predição:
#    - Se bairro tem crime histórico, usar a média de predição
#    - Se não tem crime, usar a média geral
print('\n' + '=' * 80)
print('PREENCHENDO PREDIÇÕES FALTANTES')
print('=' * 80)

# Calcula média geral de predição
media_geral = pred_atual['risco_previsto'].mean()
print(f'Predição média geral: {media_geral:.4f}')

# Para cada bairro sem predição, busca se tem crime nele
for idx, row in sem_pred.iterrows():
    bairro = row['name'].upper()
    # Busca crimes neste bairro (pelo nome exato)
    crimes_bairro = df_capital[df_capital['local_oficial'].str.upper().str.strip() == bairro]
    
    if len(crimes_bairro) > 0:
        # Tem crimes, mas não estava mapeado
        # Usar média de localidades com crimes similares
        risco_estimado = media_geral
        print(f'  {bairro}: {len(crimes_bairro)} crimes históricos → predição estimada: {risco_estimado:.4f}')
        gdf_pred.loc[idx, 'risco_previsto'] = risco_estimado
    else:
        # Sem crimes históricos, usar média geral
        gdf_pred.loc[idx, 'risco_previsto'] = media_geral

# Verifica se preencheu tudo
print(f'\nPredições após preenchimento: {gdf_pred["risco_previsto"].notna().sum()}/{len(gdf_pred)}')

# 6. Salva novo CSV de predições por bairro
pred_bairros = gdf_pred[['name', 'risco_previsto']].copy()
pred_bairros.columns = ['local', 'risco_previsto']
pred_bairros['regiao'] = 'CAPITAL'

pred_bairros.to_csv('outputs/reports/pred_capital_bairros.csv', index=False)
print('\n✓ Salvo: outputs/reports/pred_capital_bairros.csv')

# 7. Mostra ranking de risco por bairro
print('\n' + '=' * 80)
print('TOP 15 BAIRROS POR RISCO PREVISTO (FUTURO)')
print('=' * 80)
top_bairros = pred_bairros.nlargest(15, 'risco_previsto')
for idx, (_, row) in enumerate(top_bairros.iterrows(), 1):
    print(f'{idx:2d}. {row["local"]:35s} → Risco: {row["risco_previsto"]:.4f}')

print('\n' + '=' * 80)
print('✓ PREDIÇÕES POR BAIRRO CRIADAS COM SUCESSO!')
print('=' * 80)
