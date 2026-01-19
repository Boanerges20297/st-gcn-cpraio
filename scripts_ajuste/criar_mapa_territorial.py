import pandas as pd
import geopandas as gpd
import json

print("=" * 80)
print("CRIANDO MAPA TERRITORIAL DE FACÇÕES")
print("=" * 80)

# 1. Carrega dados
df = pd.read_parquet('data/processed/base_consolidada.parquet')

# 2. Para cada região, cria mapa de facção dominante por bairro/município
regiao_configs = {
    'CAPITAL': {
        'geojson': 'data/graph/fortaleza_bairros.geojson',
        'local_col': 'local_oficial'
    },
    'RMF': {
        'geojson': 'data/graph/ceara_rmf.geojson',
        'local_col': 'local_oficial'
    },
    'INTERIOR': {
        'geojson': 'data/graph/ceara_interior.geojson',
        'local_col': 'local_oficial'
    }
}

# 3. Cria tabela de dominância
dominancia_data = []

for regiao, config in regiao_configs.items():
    print(f'\n### Processando {regiao}')
    
    # Filtra dados da região
    df_regiao = df[df['regiao_sistema'] == regiao]
    
    # Agrupa por local e facção
    grupo = df_regiao.groupby(['local_oficial', 'faccao_predominante']).size().reset_index(name='count')
    grupo['total_local'] = grupo.groupby('local_oficial')['count'].transform('sum')
    grupo['percentual'] = (grupo['count'] / grupo['total_local'] * 100).round(2)
    
    # Pega facção dominante por local
    dominante = grupo.sort_values('count', ascending=False).drop_duplicates('local_oficial')
    
    # Calcula nível de criticidade (baseado em percentual de dominância)
    def get_criticidade(pct):
        if pct >= 80: return 'CRÍTICO'      # Muito consolidado
        if pct >= 60: return 'ALTO'         # Bem estabelecido
        if pct >= 40: return 'MÉDIO'        # Disputa
        return 'BAIXO'                      # Não consolidado
    
    dominante['criticidade_territorio'] = dominante['percentual'].apply(get_criticidade)
    
    # Adiciona coluna de região
    dominante['regiao'] = regiao
    dominancia_data.append(dominante)
    
    print(f'  Locais únicos: {dominante["local_oficial"].nunique()}')
    print(f'  Facções: {sorted(dominante["faccao_predominante"].unique())}')
    print(f'  Criticidade:')
    print(dominante['criticidade_territorio'].value_counts())

# Consolida tudo
df_dominancia = pd.concat(dominancia_data, ignore_index=True)

# 4. Salva como CSV
df_dominancia.to_csv('data/graph/facoes_territorio.csv', index=False, encoding='utf-8')
print('\n✓ Salvo: data/graph/facoes_territorio.csv')

# 5. Cria GeoJSONs separados por facção para cada região
facoes_unicas = sorted(df_dominancia['faccao_predominante'].unique())

for facacao in facoes_unicas:
    print(f'\n### Criando GeoJSON para {facacao}')
    
    for regiao, config in regiao_configs.items():
        # Filtra dados da facção nessa região
        df_faccao = df_dominancia[
            (df_dominancia['faccao_predominante'] == facacao) & 
            (df_dominancia['regiao'] == regiao)
        ]
        
        if df_faccao.empty:
            continue
        
        # Carrega GeoJSON da região
        gdf = gpd.read_file(config['geojson'])
        gdf['name'] = gdf['name'].astype(str).str.upper()
        
        # Faz merge
        df_faccao['local_oficial'] = df_faccao['local_oficial'].astype(str).str.upper()
        
        gdf_faccao = gdf.merge(
            df_faccao[['local_oficial', 'faccao_predominante', 'percentual', 'criticidade_territorio']],
            left_on='name',
            right_on='local_oficial',
            how='left'
        )
        
        # Filtra apenas onde a facção atua
        gdf_faccao = gdf_faccao[gdf_faccao['faccao_predominante'].notna()]
        
        if not gdf_faccao.empty:
            # Salva GeoJSON
            arquivo = f'data/graph/territorio_{facacao.lower()}_{regiao.lower()}.geojson'
            gdf_faccao.to_file(arquivo, driver='GeoJSON')
            print(f'  ✓ {arquivo} ({len(gdf_faccao)} áreas)')

print("\n" + "=" * 80)
print("MAPA TERRITORIAL CRIADO COM SUCESSO!")
print("=" * 80)
