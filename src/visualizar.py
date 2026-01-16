import pandas as pd
import geopandas as gpd
import folium
from folium.plugins import HeatMap
import os
import config

def load_street_level_data(days_back=30):
    """
    Carrega os dados pontuais (Lat/Long) para desenhar o Heatmap de Ruas.
    Usa o arquivo processado pelo spatial_matcher.py
    """
    if not config.GEOCODED_DATA_PATH.exists():
        print(f"[!] AVISO: Arquivo de dados geocodificados não encontrado em {config.GEOCODED_DATA_PATH}")
        print("    O mapa terá a previsão por bairros, mas ficará sem o detalhe de ruas.")
        return []
    
    print(f"[-] Carregando histórico de ruas ({days_back} dias)...")
    try:
        df = pd.read_parquet(config.GEOCODED_DATA_PATH)
        
        # Filtrar por tempo (apenas dados recentes para o tático)
        df['date'] = pd.to_datetime(df['date'])
        cutoff = df['date'].max() - pd.Timedelta(days=days_back)
        df_recent = df[df['date'] >= cutoff].copy()
        
        # Ponderação Tática: CVLI vale 3x mais "calor" no mapa que um furto
        # Isso garante que uma rua com 1 homicídio brilhe mais que uma com 2 furtos
        def get_weight(natureza):
            nat = str(natureza).upper()
            if 'CVLI' in nat or 'HOMICIDIO' in nat: return 3.0
            if 'ARMA' in nat or 'TRAFICO' in nat: return 2.0
            return 1.0 # Crimes patrimoniais comuns
            
        df_recent['weight'] = df_recent['natureza'].apply(get_weight)
        
        # Formato Folium: [Lat, Long, Weight]
        heat_data = df_recent[['lat', 'long', 'weight']].values.tolist()
        
        print(f"[V] Pontos de calor carregados: {len(heat_data)}")
        return heat_data
        
    except Exception as e:
        print(f"[X] Erro ao processar dados de rua: {e}")
        return []

def generate_tactical_dashboard():
    print("[-] Iniciando renderização do Painel Tático...")
    
    # 1. Carregar Previsão da IA (Nível Bairro)
    if not config.PREDICTION_CSV.exists():
        print(f"[X] CRÍTICO: Previsão não encontrada em {config.PREDICTION_CSV}")
        print("    Execute src/predict.py primeiro.")
        return

    df_pred = pd.read_csv(config.PREDICTION_CSV)
    gdf = gpd.read_file(config.GEOJSON_PATH)
    
    # Normalização de nomes para o Join
    gdf['name_upper'] = gdf['name'].str.upper().str.strip()
    df_pred['bairro'] = df_pred['bairro'].str.upper().str.strip()
    
    # Unir geometria + inteligência
    gdf_final = gdf.merge(df_pred, left_on='name_upper', right_on='bairro', how='left')
    
    # Identificar coluna de risco
    target_col = 'CVLI'
    if target_col not in df_pred.columns:
        target_col = df_pred.columns[1] # Pega a primeira coluna numérica se não achar CVLI
        
    gdf_final[target_col] = gdf_final[target_col].fillna(0)

    # 2. Mapa Base (Dark Mode para contraste)
    m = folium.Map(
        location=[-3.76, -38.52],
        zoom_start=12,
        tiles='CartoDB dark_matter'
    )

    # 3. CAMADA MACRO: Previsão ST-GCN (Polígonos)
    # Define a cor baseada no risco previsto para a quinzena
    folium.Choropleth(
        geo_data=gdf_final,
        name="IA: Risco Territorial (Quinzenal)",
        data=gdf_final,
        columns=['name', target_col],
        key_on='feature.properties.name',
        fill_color='Reds',      # Escala Vermelha
        fill_opacity=0.4,       # Transparência para ver as ruas
        line_opacity=0.3,
        legend_name=f'Índice de Risco Projetado ({target_col})'
    ).add_to(m)

    # 4. CAMADA MICRO: Mancha Criminal Real (Ruas)
    # Projeta o histórico recente nas ruas
    street_data = load_street_level_data(days_back=30)
    
    if street_data:
        HeatMap(
            street_data,
            name="Estatística: Mancha de Calor (Ruas)",
            min_opacity=0.3,
            radius=13, # Raio ajustado para ruas
            blur=10,
            gradient={0.4: 'cyan', 0.65: 'yellow', 1.0: 'red'} # Ciano (Baixo) -> Amarelo -> Vermelho (Crítico)
        ).add_to(m)

    # 5. Interatividade (Tooltip)
    style_invisible = lambda x: {'fillColor': '#ffffff', 'color':'#000000', 'fillOpacity': 0.0, 'weight': 0.1}
    
    folium.GeoJson(
        gdf_final,
        style_function=style_invisible,
        tooltip=folium.GeoJsonTooltip(
            fields=['name', target_col],
            aliases=['Bairro:', 'Risco Projetado:'],
            style="font-family: sans-serif; font-size: 14px; padding: 10px; background-color: #1e1e1e; color: #ffffff; border: 1px solid white;"
        )
    ).add_to(m)

    # Controle de Camadas (Para ligar/desligar Ruas ou Bairros)
    folium.LayerControl(position='topright', collapsed=False).add_to(m)
    
    # Salvar
    os.makedirs(os.path.dirname(config.MAP_HTML), exist_ok=True)
    m.save(config.MAP_HTML)
    
    print(f"\n[V] RELATÓRIO OPERACIONAL GERADO COM SUCESSO")
    print(f"    Arquivo: {config.MAP_HTML}")
    print(f"    Estratégia: Camada Vermelha (IA) mostra ONDE focar.")
    print(f"    Tática: Camada Colorida (Ruas) mostra COMO atuar.")

if __name__ == "__main__":
    generate_tactical_dashboard()