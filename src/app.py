from flask import Flask, render_template, jsonify, request, send_file
import pandas as pd
import geopandas as gpd
import json
import sys
import os
import re
from pathlib import Path
import numpy as np
from datetime import datetime, timedelta

# Import local config - support both relative and direct imports
try:
    from . import config
    from .data_adapter import init_adapter, get_adapter
except ImportError:
    import config
    from data_adapter import init_adapter, get_adapter

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


def load_prediction_csv(path):
    """Carrega e normaliza CSV de predição garantindo colunas esperadas.
    Retorna DataFrame com `local_oficial`, `risco_previsto` e `local_upper`."""
    df = pd.read_csv(path)
    # Garantir coluna de local_oficial
    if 'local_oficial' not in df.columns:
        if 'local' in df.columns:
            df['local_oficial'] = df['local']
        elif 'bairro' in df.columns:
            df['local_oficial'] = df['bairro']
        elif 'name' in df.columns:
            df['local_oficial'] = df['name']
        else:
            first_col = df.columns[0]
            df['local_oficial'] = df[first_col].astype(str)
    # Garantir coluna de risco
    if 'risco_previsto' not in df.columns:
        risco_cols = [c for c in df.columns if 'risco' in c.lower()]
        if risco_cols:
            df['risco_previsto'] = pd.to_numeric(df[risco_cols[0]], errors='coerce')
        elif len(df.columns) > 1:
            df['risco_previsto'] = pd.to_numeric(df.iloc[:,1], errors='coerce')
        else:
            df['risco_previsto'] = np.nan
    # Normalizar local_upper
    df['local_upper'] = df['local_oficial'].astype(str).str.upper().str.strip()
    return df


def load_risk_map(region_name, tipo_crime='CVLI'):
    # 1. Carregar GeoJSON da região
    geo_path = config.GEOJSON_PATHS.get(region_name)
    if not geo_path or not geo_path.exists():
        return None
    gdf = gpd.read_file(geo_path)

    # 2. Carregar Predição do modelo treinado para essa região
    pred_path = config.ARTIFACTS[region_name]['prediction']
    if not pred_path.exists():
        return json.loads(gdf.to_json())

    try:
        df_pred = load_prediction_csv(pred_path)
    except Exception:
        df_pred = pd.DataFrame()

    # Defensive: garantir DataFrame com colunas mínimas para evitar UnboundLocalError
    if not isinstance(df_pred, pd.DataFrame):
        df_pred = pd.DataFrame()
    if 'local_oficial' not in df_pred.columns:
        df_pred['local_oficial'] = ''
    if 'risco_previsto' not in df_pred.columns:
        df_pred['risco_previsto'] = 0

    # Calcular limiar crítico com base no percentil configurado e multiplicador
    try:
        scores = pd.to_numeric(df_pred['risco_previsto'].dropna(), errors='coerce')
        scores = scores[~scores.isna()]
        if len(scores) > 0:
            p = float(config.HyperParams.get('criticality_percentile', 0.90))
            mult = float(config.HyperParams.get('criticality_multiplier', 1.0))
            threshold_value = float(np.quantile(scores.values, p)) * mult
        else:
            threshold_value = None
    except Exception:
        threshold_value = None
    # expor limiar no df_pred para referência (mesmo valor para todas as linhas)
    df_pred['_critical_threshold'] = threshold_value
    
    # ★ IMPORTANTE: Criticidade é SEMPRE calculada baseada em CVLI
    # Mesmo que o filtro seja TODOS, a criticidade vem de CVLI
    # O filtro tipo_crime apenas filtra os PONTOS de crimes, não a criticidade
    
    # Se tipo_crime está filtrado para algo que NÃO é CVLI,
    # marca risco=0 mas MANTÉM a criticidade do GeoJSON
    # Se o filtro especificar CVP, marcar as áreas que têm CVP histórico
    if tipo_crime == 'CVP':
        df_crimes = load_occurrences()
        df_crimes = df_crimes[(df_crimes['regiao_sistema'] == region_name) & (df_crimes['tipo'] == 'CVP')]
        # Normalizar nome de coluna para compatibilidade com CSVs gerados
        if 'local_oficial' not in df_pred.columns:
            if 'local' in df_pred.columns:
                df_pred['local_oficial'] = df_pred['local']
            elif 'bairro' in df_pred.columns:
                df_pred['local_oficial'] = df_pred['bairro']
            elif 'name' in df_pred.columns:
                df_pred['local_oficial'] = df_pred['name']
            else:
                # fallback: use first column as local
                first_col = df_pred.columns[0]
                df_pred['local_oficial'] = df_pred[first_col].astype(str)
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
    # Marcar origem como predição (para frontend distinguir fonte)
    gdf_risk['source'] = 'prediction'

    # Marcar áreas críticas usando o limiar calculado (percentil * multiplicador)
    try:
        if df_pred.get('_critical_threshold') is not None and df_pred.get('_critical_threshold').notnull().any():
            thr = float(df_pred['_critical_threshold'].dropna().unique()[0])
        else:
            thr = None
    except Exception:
        thr = None

    if thr is not None:
        gdf_risk['is_critical'] = gdf_risk['risco'] >= thr
        # Forçar nível visual para crítico nas áreas acima do limiar
        gdf_risk.loc[gdf_risk['is_critical'], 'nivel_alerta'] = 'CRÍTICO'
        gdf_risk.loc[gdf_risk['is_critical'], 'cor_alerta'] = '#ff0000'
        gdf_risk['critical_threshold'] = thr
    else:
        gdf_risk['is_critical'] = False
        gdf_risk['critical_threshold'] = None

    # Calcular dominância histórica por tipo (CVLI vs CVP) para cada área
    try:
        df_crimes = load_occurrences()
        if not df_crimes.empty:
                df_region = df_crimes[df_crimes['regiao_sistema'] == region_name].copy()
                # Normalizar local para comparação
                df_region['local_upper'] = df_region['local_oficial'].astype(str).str.upper().str.strip()
                # Contar por tipo
                pivot = df_region.groupby(['local_upper', 'tipo']).size().unstack(fill_value=0)
                # Garantir colunas CVLI/CVP
                if 'CVLI' not in pivot.columns:
                    pivot['CVLI'] = 0
                if 'CVP' not in pivot.columns:
                    pivot['CVP'] = 0

                # Map counts back to gdf_risk
                def dominant_type(row):
                    key = row.get('name_upper', '')
                    try:
                        counts = pivot.loc[key]
                        cvli_c = int(counts.get('CVLI', 0))
                        cvp_c = int(counts.get('CVP', 0))
                    except Exception:
                        cvli_c = 0
                        cvp_c = 0
                    # If there is historical dominance, return that type
                    if (cvp_c > cvli_c) and (cvp_c > 0):
                        return 'CVP', cvli_c, cvp_c
                    elif (cvli_c > cvp_c) and (cvli_c > 0):
                        return 'CVLI', cvli_c, cvp_c
                    else:
                        # No clear historical evidence - mark as prediction-driven
                        return None, cvli_c, cvp_c

                doms = gdf_risk.apply(dominant_type, axis=1)
                gdf_risk['risk_by'] = [d[0] for d in doms]
                gdf_risk['historical_cvli_count'] = [d[1] for d in doms]
                gdf_risk['historical_cvp_count'] = [d[2] for d in doms]
                # Indicar tipo previsto pelo modelo (modelo atual prevê CVLI density)
                gdf_risk['predicted_target'] = 'CVLI'
    except Exception:
        # se falhar ao calcular dominância histórica, NÃO atribuir
        # um tipo histórico por padrão — marcar como prediction-driven
        gdf_risk['risk_by'] = None
        gdf_risk['historical_cvli_count'] = 0
        gdf_risk['historical_cvp_count'] = 0
        gdf_risk['predicted_target'] = 'CVLI'

    # Integrar eventos exógenos (se houverem) e expor contagem por área
    try:
        exog_file = config.DATA_RAW / 'exogenous_events.json'
        if exog_file.exists():
            with open(exog_file, 'r', encoding='utf-8') as f:
                exog_list = json.load(f)
            exog_counts = {}
            for ev in exog_list:
                local = str(ev.get('bairro') or ev.get('local') or '').upper().strip()
                if not local:
                    continue
                exog_counts[local] = exog_counts.get(local, 0) + 1

            gdf_risk['exogenous_events_count'] = gdf_risk['name_upper'].apply(lambda x: int(exog_counts.get(x, 0)))
            gdf_risk['has_exogenous'] = gdf_risk['exogenous_events_count'] > 0
        else:
            gdf_risk['exogenous_events_count'] = 0
            gdf_risk['has_exogenous'] = False
    except Exception:
        gdf_risk['exogenous_events_count'] = 0
        gdf_risk['has_exogenous'] = False
    
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
        # Normalizar nome de coluna local/local_oficial
        if 'local_oficial' not in df.columns and 'local' in df.columns:
            df['local_oficial'] = df['local']
        return df
    return pd.DataFrame()

# --- ROTAS ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/dashboard_data')
def dashboard_data():
    """
    Rota otimizada com novo modelo ST-GCN
    Retorna dados de risco baseados em predições do novo modelo
    """
    try:
        region = request.args.get('region', 'CAPITAL')
        faccao = request.args.get('faccao', 'TODAS')
        tipo_crime = request.args.get('tipo_crime', 'CVLI')
        
        # Usar novo DataAdapter para predições
        adapter = get_adapter()
        
        if adapter.df_pred is None:
            return jsonify({
                "polygons": None,
                "points": [],
                "targets": [],
                "filtros_ativos": {"regiao": region, "faccao": faccao, "tipo_crime": tipo_crime},
                "erro": "Predições não disponíveis"
            })
        
        # Carregar base de limites (continua igual)
        risk_geojson = load_risk_map(region, tipo_crime)
        
        # Enriquecer com predições do novo modelo
        if risk_geojson:
            for feature in risk_geojson.get('features', []):
                props = feature.setdefault('properties', {})
                bairro_name = props.get('name') or props.get('bairro') or ''
                
                # Buscar predição no novo modelo
                bairro_info = adapter.get_bairro_info(bairro_name)
                if bairro_info:
                    score_risco = (bairro_info['score_risco'] or 0) / 100  # Normalizar 0-1
                    props['risco_previsto'] = float(score_risco)
                    props['risco'] = float(score_risco)
                    props['cvli_predito'] = float(bairro_info['cvli_predito'])
                    
                    # Classificar nível de alerta
                    if score_risco >= 0.8:
                        props['nivel_alerta'] = 'CRÍTICO'
                        props['cor_alerta'] = '#ff0000'
                    elif score_risco >= 0.6:
                        props['nivel_alerta'] = 'ALTO'
                        props['cor_alerta'] = '#ff6b6b'
                    elif score_risco >= 0.4:
                        props['nivel_alerta'] = 'MÉDIO'
                        props['cor_alerta'] = '#ffcc00'
                    else:
                        props['nivel_alerta'] = 'BAIXO'
                        props['cor_alerta'] = '#00ff00'
        
        # Top alvos baseado em novo modelo
        top_bairros = adapter.get_top_bairros(10)
        top_targets = [
            {
                'local': b['bairro'],
                'nivel': 'CRÍTICO' if b['score_risco'] >= 80 else 'ALTO' if b['score_risco'] >= 60 else 'MÉDIO',
                'score': round(b['score_risco'] / 100, 3),
                'cvli': round(b['cvli_predito'], 4)
            }
            for b in top_bairros
        ]
        
        points = []  # Não usar pontos históricos com novo modelo
        
        return jsonify({
            "polygons": risk_geojson,
            "points": points,
            "targets": top_targets,
            "filtros_ativos": {
                "regiao": region,
                "faccao": faccao,
                "tipo_crime": tipo_crime,
                "fonte": "ST-GCN Model"
            }
        })
    
    except Exception as e:
        import traceback
        erro_detalhes = traceback.format_exc()
        print(f"[ERRO] /api/dashboard_data: {str(e)}\n{erro_detalhes}")
        return jsonify({
            "sucesso": False,
            "erro": str(e),
            "detalhes": erro_detalhes,
            "polygons": None,
            "points": [],
            "targets": []
        }), 200


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
        # Normalizar nomes de coluna para evitar KeyError (local/local_oficial/bairro/name)
        if 'local_oficial' not in df_crimes.columns:
            if 'local' in df_crimes.columns:
                df_crimes['local_oficial'] = df_crimes['local']
            elif 'bairro' in df_crimes.columns:
                df_crimes['local_oficial'] = df_crimes['bairro']
            elif 'name' in df_crimes.columns:
                df_crimes['local_oficial'] = df_crimes['name']
            else:
                df_crimes['local_oficial'] = 'DESCONHECIDO'
        # Normalizar nomes de coluna para evitar KeyError (local/local_oficial/bairro/name)
        if 'local_oficial' not in df_crimes.columns:
            if 'local' in df_crimes.columns:
                df_crimes['local_oficial'] = df_crimes['local']
            elif 'bairro' in df_crimes.columns:
                df_crimes['local_oficial'] = df_crimes['bairro']
            elif 'name' in df_crimes.columns:
                df_crimes['local_oficial'] = df_crimes['name']
            else:
                # Garantir coluna presente para evitar erros posteriores
                df_crimes['local_oficial'] = 'DESCONHECIDO'
        
        # Carregar predições por bairro
        pred_file = config.ARTIFACTS['CAPITAL']['prediction']
        if not pred_file.exists():
            return jsonify({"sucesso": False, "erro": "Predições não disponíveis"})
        
        df_pred = load_prediction_csv(pred_file)
        # Normalizar coluna de local na predição para 'local_oficial'
        if 'local_oficial' not in df_pred.columns:
            if 'local' in df_pred.columns:
                df_pred['local_oficial'] = df_pred['local']
            elif 'bairro' in df_pred.columns:
                df_pred['local_oficial'] = df_pred['bairro']
            elif 'name' in df_pred.columns:
                df_pred['local_oficial'] = df_pred['name']
            else:
                # usar primeira coluna como rótulo se necessário
                first_col = df_pred.columns[0]
                df_pred['local_oficial'] = df_pred[first_col].astype(str)
        # Garantir coluna de risco
        if 'risco_previsto' not in df_pred.columns:
            # tentar encontrar coluna que contenha 'risco' no nome
            risco_cols = [c for c in df_pred.columns if 'risco' in c.lower()]
            if risco_cols:
                df_pred['risco_previsto'] = pd.to_numeric(df_pred[risco_cols[0]], errors='coerce').fillna(0)
            else:
                # fallback: tentar usar a segunda coluna como valor de risco
                if len(df_pred.columns) > 1:
                    df_pred['risco_previsto'] = pd.to_numeric(df_pred.iloc[:,1], errors='coerce').fillna(0)
                else:
                    df_pred['risco_previsto'] = 0
        # Normalizar coluna de local na predição para 'local_oficial'
        if 'local_oficial' not in df_pred.columns:
            if 'local' in df_pred.columns:
                df_pred['local_oficial'] = df_pred['local']
            elif 'bairro' in df_pred.columns:
                df_pred['local_oficial'] = df_pred['bairro']
            elif 'name' in df_pred.columns:
                df_pred['local_oficial'] = df_pred['name']
            else:
                # tentar colunas alternativas (alguns CSVs usam 'index' ou nome na primeira coluna)
                first_col = df_pred.columns[0]
                df_pred['local_oficial'] = df_pred[first_col].astype(str)
        # Normalizar nome de coluna para compatibilidade com CSVs gerados
        if 'local_oficial' not in df_pred.columns and 'local' in df_pred.columns:
            df_pred['local_oficial'] = df_pred['local']
        
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
            # Fallback: usar janela configurada para CVLI (padrão mais amplo)
            hoje = datetime.now().date()
            data_fim = hoje
            dias_default = int(config.HyperParams.get('window_size_cvli', 180))
            data_inicio = hoje - timedelta(days=dias_default)
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
        
        df_pred = load_prediction_csv(pred_file)
        
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
        
        df_pred = load_prediction_csv(pred_file)
        
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


@app.route('/exogenous-event')
def exogenous_event_page():
    """Página com formulário para inserir eventos exógenos relevantes."""
    return render_template('exogenous_event.html')


@app.route('/api/exogenous_event', methods=['POST'])
def submit_exogenous_event():
    """Recebe POST JSON com evento exógeno e salva em data/raw/exogenous_events.json"""
    try:
        data = request.get_json(force=True)
        # Campos mínimos
        event_type = data.get('event_type')
        details = data.get('details')
        date = data.get('date')
        bairro = data.get('bairro') or data.get('local')

        if not event_type or not date or not details:
            return jsonify({"sucesso": False, "erro": "Campos obrigatórios: event_type, date, details"}), 400

        out = {
            'id': int(pd.Timestamp.now().timestamp()),
            'event_type': event_type,
            'date': date,
            'details': details,
            'bairro': bairro,
            'criminosos': data.get('criminosos', ''),
            'faccao': data.get('faccao', ''),
            'lat': data.get('lat'),
            'long': data.get('long')
        }

        exog_path = config.DATA_RAW / 'exogenous_events.json'
        # Ler existente
        events = []
        if exog_path.exists():
            try:
                with open(exog_path, 'r', encoding='utf-8') as f:
                    events = json.load(f)
            except Exception:
                events = []

        events.append(out)
        with open(exog_path, 'w', encoding='utf-8') as f:
            json.dump(events, f, ensure_ascii=False, indent=2)

        return jsonify({"sucesso": True, "evento": out})
    except Exception as e:
        import traceback
        return jsonify({"sucesso": False, "erro": str(e), "detalhes": traceback.format_exc()}), 500


@app.route('/api/simulate_teams', methods=['POST'])
def simulate_teams():
    """Simula adição de equipes e retorna impacto estimado."""
    try:
        payload = request.get_json(force=True)
        teams_added = int(payload.get('teams_added', 0))
        current_equipes = int(payload.get('current_equipes', 0))
        area_km2 = float(payload.get('area_km2_per_team', 0) or 0)

        new_total = current_equipes + teams_added
        impact_percent = min(new_total * 6, 40)  # mesma heurística do sistema

        return jsonify({
            'sucesso': True,
            'current_equipes': current_equipes,
            'teams_added': teams_added,
            'new_total_equipes': new_total,
            'area_km2_per_team': area_km2,
            'impact_percent': float(impact_percent)
        })
    except Exception as e:
        return jsonify({"sucesso": False, "erro": str(e)}), 500

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
            dias_default = int(config.HyperParams.get('window_size_cvli', 180))
            data_inicio = hoje - timedelta(days=dias_default)  # Padrão: janela configurada (CVLI)
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
        
        # Determinar região padrão para carregar predição
        regiao_pred = regiao_filtro if (regiao_filtro and regiao_filtro in ['CAPITAL', 'RMF', 'INTERIOR']) else 'CAPITAL'
        
        # Carregar predição da região apropriada
        pred_file = config.ARTIFACTS.get(regiao_pred, {}).get('prediction')
        if not pred_file:
            pred_file = config.ARTIFACTS['CAPITAL']['prediction']  # fallback
        
        if not pred_file.exists():
            return jsonify({"sucesso": False, "erro": "Predições não disponíveis"})
        
        df_pred = load_prediction_csv(pred_file)
        
        # Filtrar predições para cidades presentes em df_crimes da região selecionada
        if len(df_crimes) > 0 and 'local_oficial' in df_crimes.columns:
            cidades_region = df_crimes['local_oficial'].dropna().unique()
            df_pred = df_pred[df_pred['local_oficial'].isin(cidades_region)]
        
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
        
        try:
            top_locals = df_pred.nlargest(15, 'risco_previsto')['local_oficial'].unique()
        except Exception as e:
            # Retornar diagnóstico detalhado para facilitar depuração
            cols_pred = list(df_pred.columns)
            cols_crimes = list(df_crimes.columns)
            sample_pred = df_pred.head(5).to_dict(orient='records') if not df_pred.empty else []
            debug_msg = {
                'sucesso': False,
                'erro': str(e),
                'diagnostico': {
                    'df_pred_columns': cols_pred,
                    'df_crimes_columns': cols_crimes,
                    'df_pred_sample': sample_pred
                }
            }
            # Log diagnostic JSON to console for easier debugging (usuário pediu)
            try:
                print(json.dumps(debug_msg, ensure_ascii=False, indent=2))
            except Exception:
                # fallback simple print
                print(debug_msg)
            return jsonify(debug_msg)

        for bairro in top_locals:
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
            
            # LÓGICA DE RECOMENDAÇÃO REVISADA - PRIORIZA DADOS REAIS DE HOMICÍDIOS
            # Base: dados históricos (90 dias) + predição + atividade recente
            
            # Construir explicação da predição (para transparência)
            explicacao_modelo = []
            if homicidios_90d > 0:
                explicacao_modelo.append(f"📊 Histórico: {homicidios_90d} homicídios em 90 dias")
            if crimes_periodo_anterior > 0:
                explicacao_modelo.append(f"📈 Tendência: {tendencia:+.0f}% vs período anterior")
            if risco > 0:
                explicacao_modelo.append(f"🤖 Modelo prevê risco de {risco:.1%} para próximos dias")
            if len(explicacao_modelo) == 0:
                explicacao_modelo.append("✅ Sem atividade criminal detectada no período")
            
            # CRÍTICO: Muitos homicídios nos últimos 90 dias
            if homicidios_90d > 10:
                acao = "INTENSIFICAR"
                motivo = f"Histórico crítico: {homicidios_90d} homicídios em 90 dias. Reforço imediato necessário."
                prioridade = "CRÍTICO"
                equipes_delta = 3
            # ALTO: Atividade significativa de homicídios
            elif homicidios_90d > 5:
                acao = "AUMENTAR"
                motivo = f"Atividade persistente: {homicidios_90d} homicídios em 90 dias. Manter presença reforçada."
                prioridade = "ALTO"
                equipes_delta = 2
            # MÉDIO-ALTO: Alguns homicídios OU predição sugere risco
            elif homicidios_90d > 0 or risco > 0.25:
                acao = "MANTER"
                motivo = f"Padrão detectado: {homicidios_90d} homicídios (90d) + Risco previsto {risco:.1%}. Manter vigilância."
                prioridade = "MÉDIO"
                equipes_delta = 0
            # BAIXO: Risco baixo E sem atividade criminal recente (explicável pelo modelo)
            elif risco < 0.15 and homicidios_90d == 0 and crimes_periodo_anterior == 0:
                acao = "REDUZIR"
                motivo = "Risco baixo consolidado: nenhum incidente em 90 dias e modelo prevê continuidade. Realocação possível."
                prioridade = "BAIXO"
                equipes_delta = -1
                explicacao_modelo.append("✅ Modelo indica estabilidade mantida (baixo risco)")
            # REDUÇÃO COM CAUTELA: Se há histórico mas tendência MUITO positiva
            elif tendencia < -50 and homicidios_reais == 0 and risco < 0.20:
                acao = "REDUZIR"
                motivo = f"Redução significativa (-{abs(tendencia):.0f}% vs período anterior). Modelo corrobora: {risco:.1%} risco. Monitorar e realocação gradual."
                prioridade = "BAIXO"
                equipes_delta = -1
                explicacao_modelo.append(f"📉 Redução forte observada: -{ abs(tendencia):.0f}% vs histórico recente")
                explicacao_modelo.append("⚠️  Realocação viável, mas manter acompanhamento")
            # DEFAULT: Risco médio
            else:
                acao = "MANTER"
                motivo = f"Nível intermediário: Risco {risco:.1%}. Continuar monitoramento."
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
                'explicacao_modelo': ' | '.join(explicacao_modelo),  # Novo campo: justificativa da predição
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


# ============================================================================
# NOVAS ROTAS: PREDIÇÕES COM DINÂMICA DE FACÇÕES (180+30 DIAS)
# ============================================================================

@app.route('/api/cvli_forecast_extended')
def get_cvli_forecast_extended():
    """
    Predições de CVLI para 210 dias (180 + 30 janela)
    Integra dinâmica de facções (mudança territorial, conflito, volatilidade)
    
    Query params (opcionais):
    - top: número de bairros a retornar (default 20)
    - min_risco: filtro de risco mínimo (default 0)
    
    Retorna: Top bairros com maior risco de CVLI
    """
    try:
        # Parâmetros
        top_n = int(request.args.get('top', 20))
        min_risco = float(request.args.get('min_risco', 0))
        
        # Carregar predições
        pred_file = config.OUTPUT_DIR / 'predicoes_cvli.csv'
        if not pred_file.exists():
            return jsonify({
                "sucesso": False, 
                "erro": "Predições não disponíveis",
                "arquivo_esperado": str(pred_file)
            }), 404
        
        df_pred = pd.read_csv(pred_file)
        
        # Filtrar por risco mínimo
        df_pred = df_pred[df_pred['cvli_predito'] >= min_risco]
        
        # Top N por risco
        top_pred = df_pred.nlargest(top_n, 'cvli_predito')
        
        # Calcular percentil e classificação
        percentil_75 = df_pred['cvli_predito'].quantile(0.75)
        percentil_90 = df_pred['cvli_predito'].quantile(0.90)
        
        def classificar_risco(valor):
            if valor > percentil_90:
                return "CRÍTICO"
            elif valor > percentil_75:
                return "ALTO"
            elif valor > df_pred['cvli_predito'].median():
                return "MÉDIO"
            else:
                return "BAIXO"
        
        top_pred['classificacao'] = top_pred['cvli_predito'].apply(classificar_risco)
        
        # Estruturar resposta
        resultado = []
        for idx, row in top_pred.iterrows():
            resultado.append({
                'bairro': row['bairro'],
                'cvli_predito': float(row['cvli_predito']),
                'prob_mudanca': float(row.get('prob_mudanca', 0)),
                'volatilidade': float(row.get('volatilidade', 0)),
                'classificacao': row['classificacao'],
                'risco_territorialidade': 'SIM' if row.get('prob_mudanca', 0) > 0.3 else 'NÃO'
            })
        
        return jsonify({
            "sucesso": True,
            "data": {
                "horizonte_dias": 210,
                "periodo": f"23/01/2026 a 21/08/2026",
                "total_bairros": int(len(df_pred)),
                "bairros_criticos": int((df_pred['cvli_predito'] > percentil_90).sum()),
                "bairros_alto_risco": int(((df_pred['cvli_predito'] > percentil_75) & (df_pred['cvli_predito'] <= percentil_90)).sum()),
                "previsoes": resultado,
                "metricas": {
                    "cvli_medio": float(df_pred['cvli_predito'].mean()),
                    "cvli_max": float(df_pred['cvli_predito'].max()),
                    "cvli_min": float(df_pred['cvli_predito'].min()),
                    "bairros_com_mudanca_territorial": int((df_pred['prob_mudanca'] > 0.3).sum())
                }
            }
        })
    
    except Exception as e:
        import traceback
        print(f"[ERRO] /api/cvli_forecast_extended: {str(e)}\n{traceback.format_exc()}")
        return jsonify({"sucesso": False, "erro": str(e)}), 500


@app.route('/api/territorial_volatility/<bairro>')
def get_territorial_volatility(bairro):
    """
    Análise detalhada de volatilidade territorial por bairro
    
    Params:
    - bairro: nome do bairro (URL encoded)
    
    Retorna:
    - Risco de mudança territorial
    - Estabilidade do controle
    - Histórico de mudanças
    - Recomendações operacionais
    """
    try:
        # URL decode do bairro
        bairro = bairro.replace('%20', ' ').title()
        
        # Carregar predições
        pred_file = config.OUTPUT_DIR / 'predicoes_cvli.csv'
        if not pred_file.exists():
            return jsonify({"sucesso": False, "erro": "Predições não disponíveis"}), 404
        
        df_pred = pd.read_csv(pred_file)
        
        # Procurar bairro
        bairro_data = df_pred[df_pred['bairro'].str.lower() == bairro.lower()]
        
        if bairro_data.empty:
            # Sugerir bairros similares
            bairros_disponiveis = df_pred['bairro'].unique().tolist()
            return jsonify({
                "sucesso": False,
                "erro": f"Bairro '{bairro}' não encontrado",
                "bairros_disponibles_sample": bairros_disponiveis[:5]
            }), 404
        
        row = bairro_data.iloc[0]
        
        # Carregar análise de facções se disponível
        analise_faccoes_file = config.DATA_PROCESSED / 'analise_movimentacao_faccoes.json'
        faccoes_info = {}
        if analise_faccoes_file.exists():
            with open(analise_faccoes_file, 'r', encoding='utf-8') as f:
                analise_faccoes = json.load(f)
                faccoes_info = analise_faccoes.get('bairros', {}).get(bairro, {})
        
        # Calcular risco de volatilidade
        prob_mudanca = float(row.get('prob_mudanca', 0))
        volatilidade = float(row.get('volatilidade', 0))
        
        # Classificação de risco territorial
        if prob_mudanca > 0.5 or volatilidade > 0.7:
            nivel_volatilidade = "CRÍTICO"
            cor = "#ff0000"
        elif prob_mudanca > 0.3 or volatilidade > 0.4:
            nivel_volatilidade = "ALTO"
            cor = "#ff6600"
        elif prob_mudanca > 0.1 or volatilidade > 0.2:
            nivel_volatilidade = "MÉDIO"
            cor = "#ffcc00"
        else:
            nivel_volatilidade = "BAIXO"
            cor = "#00cc00"
        
        # Recomendações baseadas em volatilidade
        if nivel_volatilidade == "CRÍTICO":
            recomendacoes = [
                "⚠️ Alto risco de disputa territorial - Reforçar inteligência",
                "🚗 Aumentar patrulhamento móvel e sensores",
                "📞 Ativar protocolo de monitoramento de facções",
                "🎯 Coordenar com agências de inteligência"
            ]
        elif nivel_volatilidade == "ALTO":
            recomendacoes = [
                "🔍 Monitorar mudanças de controle territorial",
                "📡 Aumentar vigilância em horários críticos",
                "🤝 Manter diálogo com lideranças locais"
            ]
        elif nivel_volatilidade == "MÉDIO":
            recomendacoes = [
                "✓ Manter presença policial habitual",
                "📊 Acompanhar tendências de mudança"
            ]
        else:
            recomendacoes = [
                "✅ Territorialidade estável - Manutenção rotineira"
            ]
        
        return jsonify({
            "sucesso": True,
            "data": {
                "bairro": bairro,
                "cvli_predito": float(row['cvli_predito']),
                "volatilidade_territorial": {
                    "nivel": nivel_volatilidade,
                    "cor": cor,
                    "prob_mudanca": float(prob_mudanca),
                    "volatilidade_index": float(volatilidade)
                },
                "faccoes": faccoes_info,
                "recomendacoes": recomendacoes,
                "periodo_predicao": "210 dias (23/01/2026 a 21/08/2026)"
            }
        })
    
    except Exception as e:
        import traceback
        print(f"[ERRO] /api/territorial_volatility: {str(e)}\n{traceback.format_exc()}")
        return jsonify({"sucesso": False, "erro": str(e)}), 500


@app.route('/api/faction_timeline')
def get_faction_timeline():
    """
    Timeline de movimentação de facções baseada em snapshots GeoJSON
    
    Retorna:
    - Histórico de controle territorial por bairro
    - Mudanças detectadas ao longo do tempo
    - Padrões de conflito
    """
    try:
        # Carregar análise de facções
        analise_file = config.DATA_PROCESSED / 'analise_movimentacao_faccoes.json'
        historico_file = config.DATA_PROCESSED / 'historico_mudancas_territoriais.csv'
        
        if not analise_file.exists():
            return jsonify({
                "sucesso": False,
                "erro": "Análise de facções não disponível"
            }), 404
        
        # Carregar dados
        with open(analise_file, 'r', encoding='utf-8') as f:
            analise = json.load(f)
        
        # Tentar carregar histórico de mudanças
        timeline = []
        if historico_file.exists():
            df_historico = pd.read_csv(historico_file)
            timeline = df_historico.to_dict(orient='records')
        
        # Resumo de facções
        faccoes_resumo = analise.get('faccoes', {})
        bairros_resumo = analise.get('bairros', {})
        
        # Contar tipos de mudanças
        mudancas_detectadas = len([b for b in bairros_resumo.values() if b.get('mudancas', 0) > 0])
        
        return jsonify({
            "sucesso": True,
            "data": {
                "ultima_atualizacao": analise.get('data_analise', 'N/A'),
                "faccoes_identificadas": faccoes_resumo,
                "bairros_analisados": int(len(bairros_resumo)),
                "bairros_com_mudancas": int(mudancas_detectadas),
                "timeline": timeline[:50],  # Últimas 50 mudanças
                "resumo": {
                    "total_snapshots": int(analise.get('total_snapshots', 1)),
                    "periodo": f"23/01/2026"
                }
            }
        })
    
    except Exception as e:
        import traceback
        print(f"[ERRO] /api/faction_timeline: {str(e)}\n{traceback.format_exc()}")
        return jsonify({"sucesso": False, "erro": str(e)}), 500


# ============================================================================
# ROTAS DE SINCRONIZAÇÃO COM DASHBOARD (usando Data Adapter)
# ============================================================================

@app.route('/api/dashboard_sync')
def get_dashboard_sync():
    """
    Sincroniza dados do novo modelo (ST-GCN + facções) com o dashboard
    Substitui a busca por arquivo consolidado
    
    Retorna:
    - Top 15 bairros de risco
    - Métricas globais
    - Dados por região
    - Timeline de 30 dias
    """
    try:
        adapter = get_adapter()
        
        # Se não tem predições, tentar inicializar
        if adapter.df_pred is None:
            init_adapter()
            adapter = get_adapter()
        
        # Se ainda não tem, retornar erro
        if adapter.df_pred is None:
            return jsonify({
                "sucesso": False,
                "erro": "Predições não disponíveis",
                "acao": "Execute: python src/predict_with_factions.py"
            }), 503
        
        # Exportar tudo para dashboard
        dados = adapter.export_para_dashboard()
        
        return jsonify({
            "sucesso": True,
            "data": dados,
            "fonte": "ST-GCN com Dinâmica de Facções",
            "atualizado_em": datetime.now().isoformat()
        })
    
    except Exception as e:
        import traceback
        print(f"[ERRO] /api/dashboard_sync: {str(e)}\n{traceback.format_exc()}")
        return jsonify({"sucesso": False, "erro": str(e)}), 500


@app.route('/api/recomendacoes_simples')
def get_recomendacoes_simples():
    """
    Recomendações operacionais simples baseadas em top bairros críticos
    Usa dados do novo modelo ST-GCN
    """
    try:
        adapter = get_adapter()
        
        if adapter.df_pred is None:
            return jsonify({
                "sucesso": False,
                "erro": "Dados não disponíveis",
                "data": {"recomendacoes": []}
            })
        
        # Top 5 bairros críticos
        top_bairros = adapter.get_top_bairros(5)
        
        recomendacoes = []
        for i, bairro in enumerate(top_bairros, 1):
            score = bairro['score_risco']
            
            if score >= 80:
                nivel = "CRÍTICO"
                icon = "🔴"
                acao = "Operação imediata com deslocamento de equipes"
            elif score >= 60:
                nivel = "ALTO"
                icon = "🟠"
                acao = "Monitoramento intensificado e patrulhamento aumentado"
            else:
                nivel = "MÉDIO"
                icon = "🟡"
                acao = "Reforço operacional e inteligência preventiva"
            
            recomendacoes.append({
                "id": i,
                "tipo": f"Operação - {bairro['bairro']}",
                "prioridade": nivel,
                "icon": icon,
                "acao": acao,
                "bairro": bairro['bairro'],
                "score_risco": score,
                "cvli_predito": bairro['cvli_predito'],
                "regiao": bairro['regiao'],
                "motivo": f"Predição de CVLI: {bairro['cvli_predito']:.4f} eventos/dia"
            })
        
        return jsonify({
            "sucesso": True,
            "data": {"recomendacoes": recomendacoes}
        })
    
    except Exception as e:
        import traceback
        print(f"[ERRO] /api/recomendacoes_simples: {str(e)}\n{traceback.format_exc()}")
        return jsonify({
            "sucesso": False,
            "erro": str(e),
            "data": {"recomendacoes": []}
        }), 500


@app.route('/api/bairro_detalhes/<bairro>')
def get_bairro_detalhes(bairro):
    """
    Detalhes completos de um bairro com predição e análise
    
    Retorna:
    - Score de risco (0-100)
    - Predição CVLI
    - Análise de volatilidade
    - Recomendações
    """
    try:
        # URL decode
        bairro = bairro.replace('%20', ' ').replace('%27', "'")
        
        adapter = get_adapter()
        if adapter.df_pred is None:
            init_adapter()
            adapter = get_adapter()
        
        # Obter info do bairro
        info = adapter.get_bairro_info(bairro)
        
        if not info:
            return jsonify({
                "sucesso": False,
                "erro": f"Bairro '{bairro}' não encontrado"
            }), 404
        
        # Calcular recomendações
        score = info['score_risco']
        if score > 75:
            recomendacao = "🔴 CRÍTICO - Reforço policial imediato necessário"
            cor = "#ff0000"
        elif score > 50:
            recomendacao = "🟠 ALTO - Aumentar vigilância e patrulhamento"
            cor = "#ff6600"
        elif score > 25:
            recomendacao = "🟡 MÉDIO - Manter monitoramento rotineiro"
            cor = "#ffcc00"
        else:
            recomendacao = "🟢 BAIXO - Manutenção regular"
            cor = "#00cc00"
        
        return jsonify({
            "sucesso": True,
            "data": {
                **info,
                "recomendacao": recomendacao,
                "cor_risco": cor,
                "risco_territorial": "ALTO" if info['prob_mudanca'] > 0.3 else "NORMAL",
                "volatilidade_status": "CRÍTICA" if info['volatilidade'] > 0.7 else ("ALTA" if info['volatilidade'] > 0.4 else "NORMAL")
            }
        })
    
    except Exception as e:
        import traceback
        print(f"[ERRO] /api/bairro_detalhes: {str(e)}\n{traceback.format_exc()}")
        return jsonify({"sucesso": False, "erro": str(e)}), 500

if __name__ == '__main__':
    # Desabilita auto-reload para evitar loop infinito com .venv
    # Use --reload manualmente se necessário, ou configure em um .flaskenv file
    app.run(debug=False, port=5000, use_reloader=False)
