from flask import Flask, render_template, jsonify, request, send_file
import pandas as pd
import geopandas as gpd
import json
import sys
import os
import re
from pathlib import Path

# Import local config
from . import config

app = Flask(__name__, static_folder='../data', static_url_path='/data')

# --- FUNÇÕES AUXILIARES ---
def get_alert_level(score, region):
    """Traduz o número estatístico para linguagem de comando."""
    # Ajuste de sensibilidade baseado na escala de cada região
    factor = 30 if region == 'INTERIOR' else 6 if region == 'RMF' else 3
    intensity = score * factor
    
    if intensity > 0.8: return "CRÍTICO", "#ff0000"    # Vermelho Sangue
    if intensity > 0.5: return "ALTO", "#ff4500"       # Laranja Escuro
    if intensity > 0.2: return "MÉDIO", "#ffcc00"      # Amarelo
    return "BAIXO", "#00ff0000"                        # Transparente (não poluir)

def load_risk_map(region_name, tipo_crime='TODOS'):
    """Carrega mapa de risco baseado em CVLI (criticidade = SEMPRE CVLI, independente do filtro)."""
    # 1. Carregar GeoJSON da região
    geo_path = config.GEOJSON_PATHS.get(region_name)
    if not geo_path or not geo_path.exists(): 
        return None
    gdf = gpd.read_file(geo_path)
    
    # 2. Carregar Predição do modelo treinado para essa região
    pred_path = config.ARTIFACTS[region_name]['prediction']
    if not pred_path.exists(): 
        return json.loads(gdf.to_json())
        
    df_pred = pd.read_csv(pred_path)
    
    # ★ IMPORTANTE: Criticidade é SEMPRE calculada baseada em CVLI
    # Mesmo que o filtro seja TODOS, a criticidade vem de CVLI
    # O filtro tipo_crime apenas filtra os PONTOS de crimes, não a criticidade
    
    # Se tipo_crime está filtrado para algo que NÃO é CVLI,
    # marca risco=0 mas MANTÉM a criticidade do GeoJSON
    if tipo_crime != 'TODOS' and tipo_crime != 'CVLI':
        # Apenas CVP ou outro - mantém criticidade de CVLI, risco fica 0
        df_crimes = load_occurrences()
        df_crimes = df_crimes[(df_crimes['regiao_sistema'] == region_name) & 
                              (df_crimes['tipo'] == tipo_crime)]
        areas_com_tipo = set(df_crimes['local_oficial'].unique())
        
        # Marca: para fins de visualização de pontos apenas
        df_pred['tem_tipo'] = df_pred['local_oficial'].isin(areas_com_tipo)
    
    # 4. Merge com GeoJSON
    gdf['name_upper'] = gdf['name'].astype(str).str.upper().str.strip()
    df_pred['local_upper'] = df_pred['local_oficial'].astype(str).str.upper().str.strip()
    
    gdf_risk = gdf.merge(
        df_pred[['local_upper', 'risco_previsto']], 
        left_on='name_upper', 
        right_on='local_upper', 
        how='left'
    )
    gdf_risk['risco'] = gdf_risk['risco_previsto'].fillna(0)
    
    # 5. Calcular nível de alerta baseado em PREDIÇÃO (que é baseada em CVLI)
    gdf_risk['nivel_alerta'], gdf_risk['cor_alerta'] = zip(*gdf_risk['risco'].apply(lambda x: get_alert_level(x, region_name)))
    
    return json.loads(gdf_risk.to_json())

def classify_crime_type(natureza):
    """Classifica crime como CVP ou CVLI.
    
    CVP: Crimes Violentos Patrimoniais - roubos, assaltos, roubos a pessoa, veículos, etc.
    CVLI: Crimes Violentos Letais e Intencionais - homicídios, feminicídios, latrocínios, etc.
    """
    natureza_upper = str(natureza).upper()
    
    # CVLI - Crimes Violentos Letais e Intencionais
    cvli_keywords = [
        'HOMICIDIO', 'FEMINICIDIO', 'LATROCINIO', 'MORTE',
        'LESAO CORPORAL SEGUIDA DE MORTE'
    ]
    
    if any(k in natureza_upper for k in cvli_keywords):
        return 'CVLI'
    
    # Tudo o resto é CVP (roubos de toda espécie são patrimoniais)
    return 'CVP'

def load_occurrences():
    """Carrega ocorrências do arquivo consolidado com tipo CVLI/CVP."""
    if config.CONSOLIDATED_FILE.exists():
        df = pd.read_parquet(config.CONSOLIDATED_FILE)
        # Normaliza tipo para maiúsculo (cvli -> CVLI, cvp -> CVP)
        if 'tipo' in df.columns:
            df['tipo'] = df['tipo'].str.upper()
        else:
            # Se não tem tipo, classifica pela natureza
            df['tipo'] = df['natureza'].apply(classify_crime_type)
        
        # Garante que tem coluna 'faccao' (usa faccao_predominante se existir)
        if 'faccao_predominante' in df.columns and 'faccao' not in df.columns:
            df['faccao'] = df['faccao_predominante']
        elif 'faccao' not in df.columns:
            df['faccao'] = 'DESCONHECIDA'
        
        # Normaliza faccao para maiúsculo
        df['faccao'] = df['faccao'].fillna('DESCONHECIDA').astype(str).str.upper()
        
        # Rename lat/lng para latitude/longitude
        if 'lat' in df.columns:
            df['latitude'] = df['lat']
        if 'lng' in df.columns:
            df['longitude'] = df['lng']
        return df
    return pd.DataFrame()

# --- ROTAS ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/dashboard_data')
def dashboard_data():
    """Rota otimizada com filtros em cascata (AND logic)."""
    region = request.args.get('region', 'CAPITAL')
    faccao = request.args.get('faccao', 'TODAS')
    tipo_crime = request.args.get('tipo_crime', 'TODOS')
    
    # LÓGICA DE CASCATA:
    # Cada filtro refina progressivamente os dados
    
    if faccao != 'TODAS':
        # MODO 1: Filtro territorial ativado - mostra mapa da facção
        territory_path = config.DATA_GRAPH / f"territorio_{faccao.lower()}_{region.lower()}.geojson"
        if territory_path.exists():
            with open(territory_path, 'r', encoding='utf-8') as f:
                risk_geojson = json.load(f)
            
            # Se também tem filtro de tipo_crime, ajusta cores baseado no tipo
            if tipo_crime != 'TODOS':
                # Carrega dados de crimes por tipo nesse território
                df = load_occurrences()
                df = df[(df['regiao_sistema'] == region) & 
                        (df['faccao'] == faccao) & 
                        (df['tipo'] == tipo_crime)]
                
                # Calcula criticidade por local para esse tipo de crime
                crime_counts = df.groupby('local_oficial').size()
                max_crimes = crime_counts.max() if len(crime_counts) > 0 else 1
                
                # Aplica cores baseado na densidade de crimes desse tipo
                for feature in risk_geojson.get('features', []):
                    props = feature['properties']
                    local = props.get('name', '').upper()
                    count = crime_counts.get(local, 0)
                    
                    # Normaliza para score 0-1
                    score = count / max_crimes if max_crimes > 0 else 0
                    nivel, cor = get_alert_level(score, region)
                    
                    props['nivel_alerta'] = nivel
                    props['cor_alerta'] = cor
                    props['count_tipo_crime'] = count
            else:
                # Sem filtro de tipo, usa criticidade_territorio
                for feature in risk_geojson.get('features', []):
                    props = feature['properties']
                    crit = props.get('criticidade_territorio', 'BAIXO')
                    nivel, cor = get_alert_level(0.5 if crit == 'CRÍTICO' else 0.3 if crit == 'ALTO' else 0.1, region)
                    props['nivel_alerta'] = crit
                    props['cor_alerta'] = cor
        else:
            risk_geojson = None
        
        points = []
        top_targets = []
    
    else:
        # MODO 2: Sem filtro territorial - mostra mapa de risco geral
        risk_geojson = load_risk_map(region, tipo_crime)
        
        # Pontos com filtro de tipo_crime cascata
        df = load_occurrences()
        points = []
        if not df.empty:
            # Cascata de filtros
            df = df[df['regiao_sistema'] == region]
            
            # FILTRO: Tipo de Crime (se especificado)
            if tipo_crime != 'TODOS':
                df = df[df['tipo'] == tipo_crime]
            
            # FILTRO: Facção (apenas se não estiver em modo territorial)
            # Nesse modo, facção já é 'TODAS', então não filtra
            
            # Limite para performance
            if not df.empty:
                df = df.sort_values('data_hora', ascending=False).head(3000)
                points = df[['latitude', 'longitude', 'natureza', 'local_oficial', 'faccao', 'data_hora', 'tipo']].to_dict(orient='records')
        
        # Top Alvos (apenas risco geral, sem filtro territorial)
        top_targets = []
        if risk_geojson:
            features = risk_geojson['features']
            features.sort(key=lambda x: x['properties'].get('risco', 0), reverse=True)
            for f in features[:5]:
                props = f['properties']
                if props.get('risco', 0) > 0:
                    top_targets.append({
                        'local': props.get('name') or props.get('bairro'),
                        'nivel': props.get('nivel_alerta'),
                        'score': round(props.get('risco', 0), 3)
                    })

    return jsonify({
        "polygons": risk_geojson,
        "points": points,
        "targets": top_targets,
        "filtros_ativos": {
            "regiao": region,
            "faccao": faccao,
            "tipo_crime": tipo_crime
        }
    })

# --- DASHBOARD ESTRATÉGICO COM IA ---

@app.route('/dashboard-estrategico')
def dashboard_estrategico():
    """Página do dashboard estratégico descritivo"""
    return render_template('dashboard_estrategico.html')

@app.route('/relatorio-analise')
def relatorio_analise():
    """Página de relatório profissional da análise estratégica com PDF"""
    return render_template('relatorio_analise.html')

@app.route('/api/strategic_insights')
def get_strategic_insights():
    """Coleta dados agregados para análise estratégica
    
    IMPORTANTE: Crimes NÃO = Facção autora
    - Homicídios ocorrem em áreas de influência territorial
    - Criticidade por densidade de CVLI (homicídios) por AID/território
    - Facções são contexto geográfico, não autoria criminal
    """
    try:
        # Carregar crimes
        if not config.CONSOLIDATED_FILE.exists():
            return jsonify({"sucesso": False, "erro": "Dados não disponíveis"})
        
        df_crimes = pd.read_parquet(config.CONSOLIDATED_FILE)
        
        # Carregar predições por bairro
        pred_file = config.ARTIFACTS['CAPITAL']['prediction']
        if not pred_file.exists():
            return jsonify({"sucesso": False, "erro": "Predições não disponíveis"})
        
        df_pred = pd.read_csv(pred_file)
        
        # Análise por tipo de crime
        df_crimes_norm = df_crimes.copy()
        if 'tipo' not in df_crimes_norm.columns:
            df_crimes_norm['tipo'] = df_crimes_norm['natureza'].apply(classify_crime_type)
        
        # Calcular CVLI e CVP (em TODO o território, não só CAPITAL)
        cvli_count = len(df_crimes_norm[df_crimes_norm['tipo'].str.lower() == 'cvli'])
        cvp_count = len(df_crimes_norm[df_crimes_norm['tipo'].str.lower() == 'cvp'])
        
        crime_types = {
            'CVP': cvp_count,
            'CVLI': cvli_count
        }
        
        # ===== AJUSTE 2: CVLI/CVP por 3 MACRO REGIÕES =====
        cvli_capital = len(df_crimes_norm[(df_crimes_norm['regiao_sistema'] == 'CAPITAL') & (df_crimes_norm['tipo'].str.lower() == 'cvli')])
        cvli_rmf = len(df_crimes_norm[(df_crimes_norm['regiao_sistema'] == 'RMF') & (df_crimes_norm['tipo'].str.lower() == 'cvli')])
        cvli_interior = len(df_crimes_norm[(df_crimes_norm['regiao_sistema'] == 'INTERIOR') & (df_crimes_norm['tipo'].str.lower() == 'cvli')])
        
        cvp_capital = len(df_crimes_norm[(df_crimes_norm['regiao_sistema'] == 'CAPITAL') & (df_crimes_norm['tipo'].str.lower() == 'cvp')])
        cvp_rmf = len(df_crimes_norm[(df_crimes_norm['regiao_sistema'] == 'RMF') & (df_crimes_norm['tipo'].str.lower() == 'cvp')])
        cvp_interior = len(df_crimes_norm[(df_crimes_norm['regiao_sistema'] == 'INTERIOR') & (df_crimes_norm['tipo'].str.lower() == 'cvp')])
        
        crime_types_by_region = {
            'CVLI': {'FORTALEZA': cvli_capital, 'RMF': cvli_rmf, 'INTERIOR': cvli_interior},
            'CVP': {'FORTALEZA': cvp_capital, 'RMF': cvp_rmf, 'INTERIOR': cvp_interior}
        }
        
        # ===== MUDANÇA CRÍTICA =====
        # Em vez de agregar por FACÇÃO (que NÃO é autora),
        # agregamos por TERRITÓRIO (AID ORCRIM) - densidade de HOMICÍDIOS
        # Facções são apenas CONTEXTO GEOGRÁFICO
        
        # Homicídios por Território (AID ORCRIM)
        if 'aid_orcrim' in df_crimes_norm.columns:
            cvli_data = df_crimes_norm[df_crimes_norm['tipo'].str.lower() == 'cvli']
            
            # Separar por região antes de calcular top
            top_territorios = []
            
            # Top 10 territórios de CAPITAL (Fortaleza)
            capital_data = cvli_data[cvli_data['regiao_sistema'] == 'CAPITAL']
            if len(capital_data) > 0:
                homicidios_capital = capital_data.groupby('aid_orcrim').size().to_dict()
                
                # Alterar "SEM_AID" para "SEM ORCRIM" com descrição (apenas para CAPITAL)
                if 'SEM_AID' in homicidios_capital:
                    count_sem_aid = homicidios_capital.pop('SEM_AID')
                    homicidios_capital['SEM ORCRIM (Não localizado em polígonos ORCRIM)'] = count_sem_aid
                
                capital_top = sorted(homicidios_capital.items(), key=lambda x: x[1], reverse=True)[:10]
                
                for territorio, count in capital_top:
                    territorios_crimes = capital_data[capital_data['aid_orcrim'] == territorio]
                    faccao = 'DESCONHECIDA'
                    if len(territorios_crimes) > 0 and 'faccao_predominante' in territorios_crimes.columns:
                        faccao_counts = territorios_crimes['faccao_predominante'].value_counts()
                        if len(faccao_counts) > 0:
                            faccao = str(faccao_counts.index[0]).strip() if faccao_counts.index[0] else 'DESCONHECIDA'
                            if faccao.upper() == 'NAN' or faccao == '-' or faccao == '':
                                faccao = 'DESCONHECIDA'
                    
                    top_territorios.append({
                        'territorio': territorio,
                        'homicidios': count,
                        'faccao_dominante': faccao,
                        'regiao': 'CAPITAL',
                        'cidade': 'FORTALEZA'
                    })
            
            # Top 5 territórios de RMF + INTERIOR
            rmf_interior_data = cvli_data[cvli_data['regiao_sistema'].isin(['RMF', 'INTERIOR'])]
            if len(rmf_interior_data) > 0:
                homicidios_rmf = rmf_interior_data.groupby('aid_orcrim').size().to_dict()
                rmf_top = sorted(homicidios_rmf.items(), key=lambda x: x[1], reverse=True)[:10]
                
                for territorio, count in rmf_top:
                    territorios_crimes = rmf_interior_data[rmf_interior_data['aid_orcrim'] == territorio]
                    faccao = 'DESCONHECIDA'
                    regiao = 'DESCONHECIDA'
                    cidade = None
                    
                    if len(territorios_crimes) > 0:
                        # Facção
                        if 'faccao_predominante' in territorios_crimes.columns:
                            faccao_counts = territorios_crimes['faccao_predominante'].value_counts()
                            if len(faccao_counts) > 0:
                                faccao = str(faccao_counts.index[0]).strip() if faccao_counts.index[0] else 'DESCONHECIDA'
                                if faccao.upper() == 'NAN' or faccao == '-' or faccao == '':
                                    faccao = 'DESCONHECIDA'
                        
                        # Região
                        if 'regiao_sistema' in territorios_crimes.columns:
                            reg_counts = territorios_crimes['regiao_sistema'].value_counts()
                            if len(reg_counts) > 0:
                                regiao = str(reg_counts.index[0]).strip()
                        
                        # Cidade
                        if 'local_oficial' in territorios_crimes.columns:
                            cidade_counts = territorios_crimes['local_oficial'].value_counts()
                            if len(cidade_counts) > 0:
                                cidade = str(cidade_counts.index[0]).strip()
                    
                    top_territorios.append({
                        'territorio': territorio,
                        'homicidios': count,
                        'faccao_dominante': faccao,
                        'regiao': regiao,
                        'cidade': cidade
                    })
        else:
            top_territorios = []
        
        # Facções apenas como CONTEXTO GEOGRÁFICO (não agregação de crimes)
        facction_presence = {}
        
        # ===== AJUSTE 3: Bairros com facção dominante =====
        try:
            top_bairros_df = df_pred.nlargest(10, 'risco_previsto')[
                ['local_oficial', 'risco_previsto']
            ].drop_duplicates()
            top_bairros = []
            
            for _, row in top_bairros_df.iterrows():
                bairro = {
                    'local_oficial': str(row['local_oficial']),
                    'risco_previsto': float(row['risco_previsto']),
                    'faccoes_presentes': '-'
                }
                
                # Procurar todas as facções presentes neste bairro
                # Usa bairro_ciops pois local_oficial em df_crimes são cidades, não bairros
                bairro_nome = bairro['local_oficial']
                if 'bairro_ciops' in df_crimes_norm.columns:
                    bairro_crimes = df_crimes_norm[df_crimes_norm['bairro_ciops'] == bairro_nome]
                else:
                    bairro_crimes = df_crimes_norm[df_crimes_norm['local_oficial'] == bairro_nome]
                
                if len(bairro_crimes) > 0 and 'faccao_predominante' in bairro_crimes.columns:
                    faccoes_unicas = bairro_crimes['faccao_predominante'].dropna().unique()
                    faccoes_filtradas = [str(f).strip() for f in faccoes_unicas if f and str(f).strip() and str(f).strip().upper() != 'NAN' and str(f).strip().upper() != 'MISTO']
                    if faccoes_filtradas:
                        bairro['faccoes_presentes'] = '/'.join(sorted(set(faccoes_filtradas)))[:30]
                
                top_bairros.append(bairro)
        except Exception as bairro_err:
            top_bairros = []
        
        stats = {
            'total_crimes': len(df_crimes_norm),
            'crimes_capital': len(df_crimes_norm[df_crimes_norm['regiao_sistema'] == 'CAPITAL']),
            'crime_types': crime_types,
            'crime_types_by_region': crime_types_by_region,
            'top_territorios': top_territorios,
            'facctions': facction_presence,  # Contexto geográfico, não autoria
            'top_bairros': top_bairros
        }
        
        return jsonify({"sucesso": True, "data": stats})
        
    except Exception as e:
        import traceback
        erro_detalhes = traceback.format_exc()
        print(f"[ERRO] /api/strategic_insights: {str(e)}\n{erro_detalhes}")
        return jsonify({"sucesso": False, "erro": str(e), "detalhes": erro_detalhes})

@app.route('/api/strategic_insights_range')
def get_strategic_insights_range():
    """Coleta dados agregados com filtro de data e região
    
    IMPORTANTE: Crimes NÃO = Facção autora
    - Homicídios por TERRITÓRIO (AID), não por facção
    """
    try:
        from datetime import datetime, timedelta
        
        # Parâmetros de data
        data_inicio_str = request.args.get('data_inicio')
        data_fim_str = request.args.get('data_fim')
        regiao_filtro = request.args.get('regiao', '').upper()  # NOVO: Parâmetro de região
        
        if not data_inicio_str or not data_fim_str:
            # Fallback para último 30 dias
            hoje = datetime.now().date()
            data_fim = hoje
            data_inicio = hoje - timedelta(days=30)
        else:
            data_inicio = datetime.strptime(data_inicio_str, '%Y-%m-%d').date()
            data_fim = datetime.strptime(data_fim_str, '%Y-%m-%d').date()
        
        # Carregar crimes
        if not config.CONSOLIDATED_FILE.exists():
            return jsonify({"sucesso": False, "erro": "Dados não disponíveis"})
        
        df_crimes = pd.read_parquet(config.CONSOLIDATED_FILE)
        
        # Filtrar por período
        if 'data_hora' in df_crimes.columns:
            df_crimes['data_hora'] = pd.to_datetime(df_crimes['data_hora'])
            df_crimes = df_crimes[
                (df_crimes['data_hora'].dt.date >= data_inicio) & 
                (df_crimes['data_hora'].dt.date <= data_fim)
            ]
        
        # NOVO: Filtrar por região se especificada
        if regiao_filtro and regiao_filtro.upper() not in ['TODOS', '']:
            if regiao_filtro.upper() in ['CAPITAL', 'RMF', 'INTERIOR']:
                print(f"[DEBUG] Filtrando por região: {regiao_filtro}")
                df_crimes_before = len(df_crimes)
                # Fazer filtro case-insensitive
                df_crimes = df_crimes[df_crimes['regiao_sistema'].fillna('').str.upper() == regiao_filtro.upper()]
                df_crimes_after = len(df_crimes)
                print(f"[DEBUG] Crimes: {df_crimes_before} -> {df_crimes_after}")
        
        # Carregar predições por bairro
        pred_file = config.ARTIFACTS['CAPITAL']['prediction']
        if not pred_file.exists():
            return jsonify({"sucesso": False, "erro": "Predições não disponíveis"})
        
        df_pred = pd.read_csv(pred_file)
        
        # Análise por tipo de crime
        df_crimes_norm = df_crimes.copy()
        if 'tipo' not in df_crimes_norm.columns:
            df_crimes_norm['tipo'] = df_crimes_norm['natureza'].apply(classify_crime_type)
        
        # Calcular CVLI e CVP
        cvli_count = len(df_crimes_norm[df_crimes_norm['tipo'].str.lower() == 'cvli'])
        cvp_count = len(df_crimes_norm[df_crimes_norm['tipo'].str.lower() == 'cvp'])
        
        crime_types = {
            'CVP': cvp_count,
            'CVLI': cvli_count
        }
        
        # ===== AJUSTE 2: CVLI/CVP por 3 MACRO REGIÕES =====
        cvli_capital = len(df_crimes_norm[(df_crimes_norm['regiao_sistema'] == 'CAPITAL') & (df_crimes_norm['tipo'].str.lower() == 'cvli')])
        cvli_rmf = len(df_crimes_norm[(df_crimes_norm['regiao_sistema'] == 'RMF') & (df_crimes_norm['tipo'].str.lower() == 'cvli')])
        cvli_interior = len(df_crimes_norm[(df_crimes_norm['regiao_sistema'] == 'INTERIOR') & (df_crimes_norm['tipo'].str.lower() == 'cvli')])
        
        cvp_capital = len(df_crimes_norm[(df_crimes_norm['regiao_sistema'] == 'CAPITAL') & (df_crimes_norm['tipo'].str.lower() == 'cvp')])
        cvp_rmf = len(df_crimes_norm[(df_crimes_norm['regiao_sistema'] == 'RMF') & (df_crimes_norm['tipo'].str.lower() == 'cvp')])
        cvp_interior = len(df_crimes_norm[(df_crimes_norm['regiao_sistema'] == 'INTERIOR') & (df_crimes_norm['tipo'].str.lower() == 'cvp')])
        
        crime_types_by_region = {
            'CVLI': {'FORTALEZA': cvli_capital, 'RMF': cvli_rmf, 'INTERIOR': cvli_interior},
            'CVP': {'FORTALEZA': cvp_capital, 'RMF': cvp_rmf, 'INTERIOR': cvp_interior}
        }
        
        # ===== HOMICÍDIOS POR TERRITÓRIO =====
        if 'aid_orcrim' in df_crimes_norm.columns:
            cvli_data = df_crimes_norm[df_crimes_norm['tipo'].str.lower() == 'cvli']
            
            # Separar por região antes de calcular top
            top_territorios = []
            
            # Top 5 territórios de CAPITAL (Fortaleza)
            capital_data = cvli_data[cvli_data['regiao_sistema'] == 'CAPITAL']
            if len(capital_data) > 0:
                homicidios_capital = capital_data.groupby('aid_orcrim').size().to_dict()
                
                # Alterar "SEM_AID" para "SEM ORCRIM" com descrição (apenas para CAPITAL)
                if 'SEM_AID' in homicidios_capital:
                    count_sem_aid = homicidios_capital.pop('SEM_AID')
                    homicidios_capital['SEM ORCRIM (Não localizado em polígonos ORCRIM)'] = count_sem_aid
                
                capital_top = sorted(homicidios_capital.items(), key=lambda x: x[1], reverse=True)[:10]
                
                for territorio, count in capital_top:
                    territorios_crimes = capital_data[capital_data['aid_orcrim'] == territorio]
                    faccao = 'DESCONHECIDA'
                    if len(territorios_crimes) > 0 and 'faccao_predominante' in territorios_crimes.columns:
                        faccao_counts = territorios_crimes['faccao_predominante'].value_counts()
                        if len(faccao_counts) > 0:
                            faccao = str(faccao_counts.index[0]).strip() if faccao_counts.index[0] else 'DESCONHECIDA'
                            if faccao.upper() == 'NAN' or faccao == '-' or faccao == '':
                                faccao = 'DESCONHECIDA'
                    
                    top_territorios.append({
                        'territorio': territorio,
                        'homicidios': count,
                        'faccao_dominante': faccao,
                        'regiao': 'CAPITAL',
                        'cidade': 'FORTALEZA'
                    })
            
            # Top 5 territórios de RMF + INTERIOR
            rmf_interior_data = cvli_data[cvli_data['regiao_sistema'].isin(['RMF', 'INTERIOR'])]
            if len(rmf_interior_data) > 0:
                homicidios_rmf = rmf_interior_data.groupby('aid_orcrim').size().to_dict()
                rmf_top = sorted(homicidios_rmf.items(), key=lambda x: x[1], reverse=True)[:10]
                
                for territorio, count in rmf_top:
                    territorios_crimes = rmf_interior_data[rmf_interior_data['aid_orcrim'] == territorio]
                    faccao = 'DESCONHECIDA'
                    regiao = 'DESCONHECIDA'
                    cidade = None
                    
                    if len(territorios_crimes) > 0:
                        # Facção
                        if 'faccao_predominante' in territorios_crimes.columns:
                            faccao_counts = territorios_crimes['faccao_predominante'].value_counts()
                            if len(faccao_counts) > 0:
                                faccao = str(faccao_counts.index[0]).strip() if faccao_counts.index[0] else 'DESCONHECIDA'
                                if faccao.upper() == 'NAN' or faccao == '-' or faccao == '':
                                    faccao = 'DESCONHECIDA'
                        
                        # Região
                        if 'regiao_sistema' in territorios_crimes.columns:
                            reg_counts = territorios_crimes['regiao_sistema'].value_counts()
                            if len(reg_counts) > 0:
                                regiao = str(reg_counts.index[0]).strip()
                        
                        # Cidade
                        if 'local_oficial' in territorios_crimes.columns:
                            cidade_counts = territorios_crimes['local_oficial'].value_counts()
                            if len(cidade_counts) > 0:
                                cidade = str(cidade_counts.index[0]).strip()
                    
                    top_territorios.append({
                        'territorio': territorio,
                        'homicidios': count,
                        'faccao_dominante': faccao,
                        'regiao': regiao,
                        'cidade': cidade
                    })
        else:
            homicidios_por_territorio = {}
            top_territorios = []
        
        # Facções apenas como contexto
        facction_presence = {}
        
        # ===== AJUSTE 3: Bairros com facção dominante =====
        try:
            top_bairros_df = df_pred.nlargest(10, 'risco_previsto')[
                ['local_oficial', 'risco_previsto']
            ].drop_duplicates()
            top_bairros = []
            
            for _, row in top_bairros_df.iterrows():
                bairro = {
                    'local_oficial': str(row['local_oficial']),
                    'risco_previsto': float(row['risco_previsto']),
                    'faccoes_presentes': '-'  # Será preenchido do mapa abaixo
                }
                
                top_bairros.append(bairro)
        except Exception as bairro_err:
            top_bairros = []
        
        # Carregar mapa de facções por bairro (criado via spatial join com GeoJSON)
        try:
            import json
            bairro_faccoes_file = config.PROJECT_ROOT / 'data' / 'processed' / 'bairro_faccoes_map.json'
            if bairro_faccoes_file.exists():
                with open(bairro_faccoes_file, 'r', encoding='utf-8') as f:
                    bairro_faccoes = json.load(f)
                # Atualizar bairros com facções
                for bairro in top_bairros:
                    nome = bairro['local_oficial']
                    if nome in bairro_faccoes:
                        bairro['faccoes_presentes'] = bairro_faccoes[nome]
        except Exception as e:
            pass  # Continua com "-" se não conseguir carregar
        
        # Carregar mapa de facções por bairro (criado via spatial join com GeoJSON)
        try:
            import json
            bairro_faccoes_file = config.PROJECT_ROOT / 'data' / 'processed' / 'bairro_faccoes_map.json'
            if bairro_faccoes_file.exists():
                with open(bairro_faccoes_file, 'r', encoding='utf-8') as f:
                    bairro_faccoes = json.load(f)
                # Atualizar bairros com facções
                for bairro in top_bairros:
                    nome = bairro['local_oficial']
                    if nome in bairro_faccoes:
                        bairro['faccoes_presentes'] = bairro_faccoes[nome]
        except Exception as e:
            pass  # Continua com "-" se não conseguir carregar
        
        stats = {
            'total_crimes': len(df_crimes_norm),
            'crimes_capital': len(df_crimes_norm[df_crimes_norm['regiao_sistema'] == 'CAPITAL']),
            'crime_types': crime_types,
            'crime_types_by_region': crime_types_by_region,
            'top_territorios': top_territorios,
            'facctions': facction_presence,
            'top_bairros': top_bairros,
            'periodo': f'{data_inicio} a {data_fim}'
        }
        
        return jsonify({"sucesso": True, "data": stats})
        
    except Exception as e:
        import traceback
        erro_detalhes = traceback.format_exc()
        print(f"[ERRO] /api/strategic_insights_range: {str(e)}\n{erro_detalhes}")
        return jsonify({"sucesso": False, "erro": str(e), "detalhes": erro_detalhes})

@app.route('/api/ai_analysis', methods=['GET', 'POST'])
def get_ai_analysis():
    """Gera análise com Gemini para recomendações de atuação"""
    try:
        from gemini_client import GeminiRotator
        from datetime import datetime, timedelta
        
        # Obter parâmetros de data (GET ou POST)
        if request.method == 'POST':
            data_inicio_str = request.get_json().get('data_inicio') if request.get_json() else None
            data_fim_str = request.get_json().get('data_fim') if request.get_json() else None
        else:
            data_inicio_str = request.args.get('data_inicio')
            data_fim_str = request.args.get('data_fim')
        
        if not data_inicio_str or not data_fim_str:
            hoje = datetime.now().date()
            data_fim = hoje
            data_inicio = hoje - timedelta(days=365)
        else:
            data_inicio = datetime.strptime(data_inicio_str, '%Y-%m-%d').date()
            data_fim = datetime.strptime(data_fim_str, '%Y-%m-%d').date()
        
        # Obter insights
        if not config.CONSOLIDATED_FILE.exists():
            return jsonify({
                "sucesso": False, 
                "erro": "Dados não disponíveis",
                "timestamp": datetime.now().isoformat()
            })
        
        df_crimes = pd.read_parquet(config.CONSOLIDATED_FILE)
        
        # Filtrar por data se houver coluna de data
        if 'data_ocorrencia' in df_crimes.columns:
            df_crimes['data_ocorrencia'] = pd.to_datetime(df_crimes['data_ocorrencia'])
            df_crimes = df_crimes[(df_crimes['data_ocorrencia'].dt.date >= data_inicio) & 
                                  (df_crimes['data_ocorrencia'].dt.date <= data_fim)]
        
        pred_file = config.ARTIFACTS['CAPITAL']['prediction']
        if not pred_file.exists():
            return jsonify({
                "sucesso": False,
                "erro": "Predições não disponíveis",
                "timestamp": datetime.now().isoformat()
            })
        
        df_pred = pd.read_csv(pred_file)
        
        # Normalizar tipo
        if 'tipo' not in df_crimes.columns:
            df_crimes['tipo'] = df_crimes['natureza'].apply(classify_crime_type)
        
        crime_types = {
            'CVP': len(df_crimes[df_crimes['tipo'] == 'CVP']),
            'CVLI': len(df_crimes[df_crimes['tipo'] == 'CVLI'])
        }
        
        capital_data = df_crimes[df_crimes['regiao_sistema'] == 'CAPITAL']
        if 'faccao_predominante' in capital_data.columns:
            facction_crimes = capital_data.groupby('faccao_predominante').size().to_dict()
        elif 'faccao' in capital_data.columns:
            facction_crimes = capital_data.groupby('faccao').size().to_dict()
        else:
            facction_crimes = {}
        
        # Construir top_bairros com facção dominante
        try:
            top_bairros_df = df_pred.nlargest(10, 'risco_previsto')[
                ['local_oficial', 'risco_previsto']
            ].drop_duplicates()
            top_bairros = []
            
            for _, row in top_bairros_df.iterrows():
                bairro = {
                    'local_oficial': str(row['local_oficial']),
                    'risco_previsto': float(row['risco_previsto']),
                    'faccao_dominante': 'MISTO'
                }
                
                # Procurar facção mais frequente neste bairro
                bairro_crimes = df_crimes[df_crimes['local_oficial'] == bairro['local_oficial']]
                if len(bairro_crimes) > 0 and 'faccao_predominante' in bairro_crimes.columns:
                    faccao_mode = bairro_crimes['faccao_predominante'].mode()
                    if len(faccao_mode) > 0 and pd.notna(faccao_mode.iloc[0]):
                        bairro['faccao_dominante'] = str(faccao_mode.iloc[0])
                
                top_bairros.append(bairro)
        except:
            top_bairros = []
        
        # Construir prompt para Gemini
        prompt = f"""
Você é um analista estratégico de segurança pública com experiência em operações táticas.

CONTEXTO OPERACIONAL - FORTALEZA (CEARÁ):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 BASE DE DADOS:
- Total crimes em Fortaleza: {len(capital_data):,}
- Período analisado: {data_inicio} até {data_fim}
- Tipologia: CVP (roubos) {crime_types['CVP']:,}, CVLI (homicídios) {crime_types['CVLI']:,}

📍 PRESENÇA TERRITORIAL (por registro com identificação):
"""
        # REAPRESENTAÇÃO CRÍTICA: Contar bairros onde facção aparece, NÃO crimes por facção
        if facction_crimes:
            total_registros = sum(facction_crimes.values())
            
            # Calcular bairros únicos por facção para contexto territorial
            bairros_unicos = {}
            for faccao in capital_data['faccao_predominante'].dropna().unique():
                bairros_unicos[faccao] = capital_data[
                    capital_data['faccao_predominante'] == faccao
                ]['local_oficial'].nunique()
            
            total_bairros = capital_data['local_oficial'].nunique()
            
            # Apresentação simplificada: territórios, não crimes por facção
            for faccao, count in sorted(facction_crimes.items(), key=lambda x: x[1], reverse=True):
                pct_registros = (count / total_registros * 100) if total_registros > 0 else 0
                bairros = bairros_unicos.get(faccao, 0)
                pct_bairros = (bairros / total_bairros * 100) if total_bairros > 0 else 0
                prompt += f"   • Territórios com presença {faccao}: {count:,} crimes ({pct_registros:.1f}% do total) | Abrangência: {bairros} bairros ({pct_bairros:.1f}%)\n"
        
        prompt += f"""
🚨 HOTSPOTS CRÍTICOS (Próximos 15 dias):
"""
        for i, b in enumerate(top_bairros[:5], 1):
            prompt += f"   {i}. {b['local_oficial']}: {b['risco_previsto']:.2%}\n"
        
        prompt += f"""
SÍNTESE ESTRATÉGICA (máx. 120 palavras):

Dados: {len(capital_data):,} crimes em Fortaleza | Período: {data_inicio} a {data_fim}
Presença territorial (conforme identificação em registros)
Previsão: hotspots acima para próximos 15 dias

Apresente padrão observado e inferência para gestor tomar decisão operacional.
"""
        
        # Chamar Gemini
        rotator = GeminiRotator()
        response = rotator.generate_content(prompt)
        
        # Compilar dados para metadados do relatório
        total_crimes = len(capital_data)
        cvp_count = crime_types.get('CVP', 0)
        cvli_count = crime_types.get('CVLI', 0)
        
        return jsonify({
            "sucesso": True,
            "analise": response,
            "dados": {
                "total_crimes": total_crimes,
                "cvp": cvp_count,
                "cvli": cvli_count,
                "data_inicio": str(data_inicio),
                "data_fim": str(data_fim)
            },
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        from datetime import datetime
        return jsonify({
            "sucesso": False,
            "erro": str(e),
            "timestamp": datetime.now().isoformat()
        })

@app.route('/data/graph/<filename>')
def serve_geojson(filename):
    """Serve arquivos GeoJSON das facções para visualização no mapa."""
    try:
        geojson_path = Path(__file__).parent.parent / 'data' / 'graph' / filename
        if geojson_path.exists() and geojson_path.suffix == '.geojson':
            with open(geojson_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return jsonify(data)
        else:
            return jsonify({"erro": "Arquivo não encontrado"}), 404
    except Exception as e:
        return jsonify({"erro": str(e)}), 500

@app.route('/data/raw/<filename>')
def serve_raw_geojson(filename):
    """Serve arquivos GeoJSON de data/raw (limites do Ceará, etc)."""
    try:
        raw_path = Path(__file__).parent.parent / 'data' / 'raw' / filename
        if raw_path.exists() and raw_path.suffix == '.geojson':
            with open(raw_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return jsonify(data)
        else:
            return jsonify({"erro": "Arquivo não encontrado"}), 404
    except Exception as e:
        return jsonify({"erro": str(e)}), 500

@app.route('/api/recomendacoes_operacionais')
def get_recomendacoes_operacionais():
    """Recomendações táticas para gestor de policiamento com validação de predição.
    
    LÓGICA:
    - Crimes OBSERVADOS (período filtrado): Context real
    - Predição ST-GCN (próximos 15 dias): Base para recomendação
    - Ação: Combina ambas as informações
    """
    try:
        from datetime import datetime, timedelta
        
        # Parâmetros de data
        data_inicio_str = request.args.get('data_inicio')
        data_fim_str = request.args.get('data_fim')
        regiao_filtro = request.args.get('regiao', '').upper()  # NOVO: Parâmetro de região
        
        if not data_inicio_str or not data_fim_str:
            hoje = datetime.now().date()
            data_fim = hoje
            data_inicio = hoje - timedelta(days=30)  # Padrão: últimos 30 dias
        else:
            data_inicio = datetime.strptime(data_inicio_str, '%Y-%m-%d').date()
            data_fim = datetime.strptime(data_fim_str, '%Y-%m-%d').date()
        
        # Carregar crimes e predições
        if not config.CONSOLIDATED_FILE.exists():
            return jsonify({"sucesso": False, "erro": "Dados não disponíveis"})
        
        df_crimes = pd.read_parquet(config.CONSOLIDATED_FILE)
        
        # NOVO: Filtrar por região se especificada
        if regiao_filtro and regiao_filtro not in ['TODOS', '']:
            if regiao_filtro in ['CAPITAL', 'RMF', 'INTERIOR']:
                print(f"[DEBUG] Recomendações - Filtrando por região: {regiao_filtro}")
                df_crimes = df_crimes[df_crimes['regiao_sistema'].fillna('').str.upper() == regiao_filtro]
        
        pred_file = config.ARTIFACTS['CAPITAL']['prediction']
        if not pred_file.exists():
            return jsonify({"sucesso": False, "erro": "Predições não disponíveis"})
        
        df_pred = pd.read_csv(pred_file)
        
        # Classificar tipo de crime em TODOS os dados (para tendência histórica)
        if 'tipo' not in df_crimes.columns:
            df_crimes['tipo'] = df_crimes['natureza'].apply(classify_crime_type)
        else:
            df_crimes['tipo'] = df_crimes['tipo'].str.upper()
        
        # Preparar data
        if 'data_hora' in df_crimes.columns:
            df_crimes['data'] = pd.to_datetime(df_crimes['data_hora']).dt.date
        
        # PERÍODO ATUAL: para exibição (observado)
        df_crimes_periodo = df_crimes[(df_crimes['data'] >= data_inicio) & (df_crimes['data'] <= data_fim)]
        
        # Se período filtrado está vazio, expandir para últimos 90 dias (dados reais, sem mockup)
        periodo_original_inicio = data_inicio
        periodo_original_fim = data_fim
        if len(df_crimes_periodo) == 0:
            data_inicio = data_fim - timedelta(days=90)
            df_crimes_periodo = df_crimes[(df_crimes['data'] >= data_inicio) & (df_crimes['data'] <= data_fim)]
        
        # HISTÓRICO COMPLETO: para calcular tendência real (90 dias antes do período)
        dias_filtro = (data_fim - data_inicio).days
        data_historico_inicio = data_inicio - timedelta(days=dias_filtro)
        df_crimes_historico = df_crimes[(df_crimes['data'] >= data_historico_inicio) & (df_crimes['data'] <= data_fim)]
        
        # Agrupar por bairro (período atual)
        crimes_por_bairro_periodo = df_crimes_periodo.groupby('local_oficial').agg({
            'id_ocorrencia': 'count',
            'tipo': lambda x: (x == 'CVLI').sum()
        }).rename(columns={'id_ocorrencia': 'total_crimes', 'tipo': 'homicidios'})
        
        # Adicionar CVP por bairro (período)
        cvp_por_bairro_periodo = df_crimes_periodo[df_crimes_periodo['tipo'] == 'CVP'].groupby('local_oficial').size()
        
        # Agrupar por bairro (histórico completo para tendência)
        crimes_por_bairro_historico = df_crimes_historico.groupby('local_oficial').agg({
            'id_ocorrencia': 'count',
            'tipo': lambda x: (x == 'CVLI').sum()
        }).rename(columns={'id_ocorrencia': 'total_crimes_90d', 'tipo': 'homicidios_90d'})
        
        # Adicionar CVP por bairro (histórico 90d)
        cvp_por_bairro_historico = df_crimes_historico[df_crimes_historico['tipo'] == 'CVP'].groupby('local_oficial').size()
        
        # Juntar com predição de risco
        recomendacoes = []
        # Carregar mapa de facções por bairro
        faccoes_map_path = config.DATA_PROCESSED / 'bairro_faccoes_map.json'
        faccoes_map = {}
        if faccoes_map_path.exists():
            with open(faccoes_map_path, 'r', encoding='utf-8') as f:
                faccoes_map = json.load(f)
        
        for bairro in df_pred.nlargest(15, 'risco_previsto')['local_oficial'].unique():
            risco = df_pred[df_pred['local_oficial'] == bairro]['risco_previsto'].values[0] if bairro in df_pred['local_oficial'].values else 0
            
            # Obter facção dominante do mapa
            faccao_dominante = faccoes_map.get(bairro, 'DESCONHECIDA')
            if faccao_dominante == '-':
                faccao_dominante = 'DESCONHECIDA'
            
            # DEBUG
            print(f"[DEBUG] Bairro: {bairro} | Facção: {faccao_dominante}")
            
            # Dados do período atual (o que o gestor vê AGORA)
            crimes_reais = crimes_por_bairro_periodo.loc[bairro, 'total_crimes'] if bairro in crimes_por_bairro_periodo.index else 0
            homicidios_reais = crimes_por_bairro_periodo.loc[bairro, 'homicidios'] if bairro in crimes_por_bairro_periodo.index else 0
            
            # Dados históricos (para validar predição)
            crimes_90d = crimes_por_bairro_historico.loc[bairro, 'total_crimes_90d'] if bairro in crimes_por_bairro_historico.index else 0
            homicidios_90d = crimes_por_bairro_historico.loc[bairro, 'homicidios_90d'] if bairro in crimes_por_bairro_historico.index else 0
            
            # Tendência: comparar período atual com período anterior (mesmo tamanho, respeitando filtro)
            dias_filtro = (data_fim - data_inicio).days
            data_periodo_anterior_inicio = data_inicio - timedelta(days=dias_filtro)
            data_periodo_anterior_fim = data_inicio - timedelta(days=1)
            
            df_periodo_anterior = df_crimes[
                (df_crimes['data'] >= data_periodo_anterior_inicio) & 
                (df_crimes['data'] <= data_periodo_anterior_fim)
            ]
            crimes_periodo_anterior = len(df_periodo_anterior[df_periodo_anterior['local_oficial'] == bairro])
            
            # Calcular tendência percentual
            if crimes_periodo_anterior > 0:
                tendencia = ((crimes_reais - crimes_periodo_anterior) / crimes_periodo_anterior) * 100
            elif crimes_reais > 0:
                tendencia = 100.0  # Se anterior tinha 0 e agora tem crimes, é +100%
            else:
                tendencia = 0.0  # Ambos sem crimes
            
            # LÓGICA DE RECOMENDAÇÃO MELHORADA
            # Combina: risco previsto + atividade histórica + crimes observados
            if risco > 0.32:
                if homicidios_90d > 10:
                    # Alto risco + histórico de homicídios = INTENSIFICAR (ação imediata)
                    acao = "INTENSIFICAR"
                    motivo = "Histórico recorrente de homicídios + predição de agravamento. Reforçar presença nas ruas."
                    prioridade = "CRÍTICO"
                    equipes_delta = 3
                elif homicidios_90d > 0:
                    # Alto risco + alguns homicídios = AUMENTAR (preparar)
                    acao = "AUMENTAR"
                    motivo = "Padrão histórico de violência detectado. Predição aponta intensificação. Preparar mobilidade."
                    prioridade = "ALTO"
                    equipes_delta = 2
                else:
                    # Alto risco + sem homicídios = MONITORAR (atenção)
                    acao = "MONITORAR"
                    motivo = "Modelo detecta fatores de risco sem incidentes recentes. Manter vigilância estratégica."
                    prioridade = "ALTO"
                    equipes_delta = 1
            elif risco > 0.31:
                if homicidios_90d > 5:
                    acao = "AUMENTAR"
                    motivo = "Histórico de atividade criminal persistente. Manter reforço preventivo."
                    prioridade = "ALTO"
                    equipes_delta = 2
                else:
                    acao = "MANTER"
                    motivo = "Nível de vigilância compatível com risco identificado. Continuar monitoramento."
                    prioridade = "MÉDIO"
                    equipes_delta = 0
            elif risco < 0.20:
                acao = "REDUZIR"
                motivo = "Risco baixo consolidado. Realocação de equipes para áreas prioritárias."
                prioridade = "BAIXO"
                equipes_delta = -1
            else:
                acao = "MANTER"
                motivo = "Risco estável. Manter presença conforme planejado."
                prioridade = "MÉDIO"
                equipes_delta = 0
            
            # Determinação de horário baseado em dados reais
            horario_pico = "18h-06h"  # Padrão
            if homicidios_reais > 3 or homicidios_90d > 15:
                horario_pico = "20h-04h"  # Madrugada crítica
            
            # Confiança da predição: aumenta se história valida previsão
            confianca_base = 80
            if homicidios_90d > 0:
                confianca_predicao = min(confianca_base + 10, 95)  # +10% se tem histórico
            else:
                confianca_predicao = confianca_base
            
            # Dados de CVP e CVLI por bairro
            cvp_periodo = cvp_por_bairro_periodo.get(bairro, 0)
            cvli_periodo = homicidios_reais
            cvp_90d = cvp_por_bairro_historico.get(bairro, 0)
            cvli_90d = homicidios_90d
            
            recomendacoes.append({
                'bairro': bairro,
                'faccao_dominante': faccao_dominante,
                'acao': acao,
                'motivo': motivo,
                'prioridade': prioridade,
                'risco_previsto': float(risco),
                # Dados do período atual (observado)
                'cvp': int(cvp_periodo),
                'cvli': int(cvli_periodo),
                'homicidios': int(homicidios_reais),
                # Dados históricos (validação)
                'homicidios_90d': int(homicidios_90d),
                'cvp_90d': int(cvp_90d),
                'cvli_90d': int(cvli_90d),
                'crimes_90d': int(crimes_90d),
                'tendencia_percentual': float(tendencia),
                'equipes_recomendadas': int(equipes_delta),
                'horario_pico': horario_pico,
                'confianca_predicao': float(confianca_predicao)
            })
        
        # Ordenar por prioridade
        prioridade_ordem = {'CRÍTICO': 0, 'ALTO': 1, 'MÉDIO': 2, 'BAIXO': 3}
        recomendacoes.sort(key=lambda x: (prioridade_ordem[x['prioridade']], -x['risco_previsto']))
        
        # Calcular impacto total
        equipes_total = sum(r['equipes_recomendadas'] for r in recomendacoes if r['equipes_recomendadas'] > 0)
        impacto_esperado = min(equipes_total * 6, 40)  # Até 40% de redução
        
        return jsonify({
            "sucesso": True,
            "data": {
                "recomendacoes": recomendacoes[:10],  # Top 10
                "resumo": {
                    "equipes_necessarias": equipes_total,
                    "impacto_esperado_percentual": float(impacto_esperado),
                    "confianca_media": float(sum(r['confianca_predicao'] for r in recomendacoes) / len(recomendacoes))
                }
            }
        })
        
    except Exception as e:
        import traceback
        erro_detalhes = traceback.format_exc()
        print(f"[ERRO] /api/recomendacoes_operacionais: {str(e)}\n{erro_detalhes}")
        return jsonify({"sucesso": False, "erro": str(e), "detalhes": erro_detalhes})

if __name__ == '__main__':
    # Desabilita auto-reload para evitar loop infinito com .venv
    # Use --reload manualmente se necessário, ou configure em um .flaskenv file
    app.run(debug=False, port=5000, use_reloader=False)
