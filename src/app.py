<<<<<<< HEAD
import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
from streamlit_folium import st_folium
import os
import sys
import numpy as np
import re
import json

# Adiciona src ao path para importar módulos locais
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import config
from gemini_client import GeminiRotator
from pdf_generator import gerar_pdf_relatorio

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="SIGERAIO | Comando Tático",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="collapsed" # Inicia FECHADO para dar espaço ao mapa
)

# --- INICIALIZA SESSION STATE ANTES DE TUDO ---
if "relatorio_aberto" not in st.session_state:
    st.session_state.relatorio_aberto = False
    st.session_state.relatorio_conteudo = ""

# Adiciona botão de toggle do sidebar customizado no CSS com ícone moderno
st.markdown("""
<style>
    /* Botão de toggle do sidebar - MUITO MAIS VISÍVEL COM ÍCONE MODERNO */
    button[aria-label="Toggle sidebar"] {
        background-color: rgba(233, 69, 96, 0.2) !important;
        border: 2px solid #ff6b6b !important;
        border-radius: 8px !important;
        padding: 10px 14px !important;
        width: 50px !important;
        height: 50px !important;
        position: relative;
        overflow: visible !important;
        z-index: 999;
    }
    
    button[aria-label="Toggle sidebar"] * {
        display: none !important;
        visibility: hidden !important;
        width: 0 !important;
        height: 0 !important;
        overflow: hidden !important;
    }
    
    button[aria-label="Toggle sidebar"]::before {
        content: "☰" !important;
        position: absolute !important;
        top: 50% !important;
        left: 50% !important;
        transform: translate(-50%, -50%) !important;
        font-size: 24px !important;
        color: #ff6b6b !important;
        font-weight: bold !important;
        display: block !important;
        visibility: visible !important;
        width: auto !important;
        height: auto !important;
        overflow: visible !important;
    }
    
    button[aria-label="Toggle sidebar"]:hover {
        background-color: rgba(233, 69, 96, 0.4) !important;
        box-shadow: 0 0 15px rgba(233, 69, 96, 0.5) !important;
    }
</style>
""", unsafe_allow_html=True)

# --- ESTILIZAÇÃO MODERNA E HARMONIZADA ---
st.markdown("""
<style>
    /* Fundo Geral - Mais claro e harmônico */
    .stApp { 
        background: linear-gradient(135deg, #1a1f2e 0%, #16213e 100%);
        color: #e8e8e8;
    }
    
    /* Sidebar - Cor complementar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f3460 0%, #16213e 100%);
        border-right: 3px solid #e94560;
    }
    
    /* Texto do Sidebar - Melhor contraste */
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3, 
    section[data-testid="stSidebar"] label {
        color: #ffffff !important;
        font-weight: 500;
    }

    /* --- BOTÃO IA (Destaque Harmônico) --- */
    .stButton>button {
        width: 100%;
        font-weight: bold;
        border-radius: 8px;
        background: linear-gradient(90deg, #e94560, #ff6b6b) !important;
        border: none;
        color: white;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        box-shadow: 0 4px 12px rgba(233, 69, 96, 0.4);
    }
    /* Botão normal (sidebar) */
    .stSidebar .stButton>button {
        background-color: #353742;
        color: white;
        border: 1px solid #555;
    }
    
    /* --- CARTÕES DE MÉTRICAS --- */
    .metric-card { 
        background: linear-gradient(135deg, #16213e 0%, #0f3460 100%);
        padding: 12px; 
        border-radius: 8px; 
        border-left: 4px solid #ff6b6b;
        margin-bottom: 10px;
        box-shadow: 0 2px 8px rgba(233, 69, 96, 0.2);
        transition: all 0.2s ease;
    }
    .metric-card:hover {
        transform: translateX(4px);
        box-shadow: 0 4px 12px rgba(233, 69, 96, 0.3);
    }
    
    /* Texto harmônico */
    .stMarkdown, .stText {
        color: #e0e0e0;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #ffffff !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #16213e !important;
        color: #ffffff !important;
    }
    
    /* Tooltips e info */
    .stInfo, .stWarning, .stError {
        border-radius: 8px;
    }
    
    /* Modal Customizado */
    .modal-container {
        position: fixed;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        z-index: 9999;
        background: white;
        color: #1a1f2e;
        border-radius: 12px;
        padding: 30px;
        width: 90%;
        max-width: 850px;
        max-height: 85vh;
        overflow-y: auto;
        box-shadow: 0 10px 50px rgba(0,0,0,0.5);
        border: 2px solid #e94560;
    }
    
    .modal-overlay {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0,0,0,0.8);
        z-index: 9998;
    }
    
    /* Scrollbar no modal */
    .modal-container::-webkit-scrollbar {
        width: 8px;
    }
    .modal-container::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    .modal-container::-webkit-scrollbar-thumb {
        background: #e94560;
        border-radius: 10px;
    }
    .modal-container::-webkit-scrollbar-thumb:hover {
        background: #c92a48;
    }
</style>
""", unsafe_allow_html=True)

# --- CARREGAMENTO DE DADOS (CACHE) ---
@st.cache_data
def load_data(region_key):
    # 1. Carregar Mapa
    geo_path = config.GEOJSON_PATHS.get(region_key)
    if not geo_path or not geo_path.exists():
        return None, None, f"Mapa não encontrado: {geo_path}"
    
    try:
        gdf = gpd.read_file(geo_path)
    except Exception as e:
        return None, None, f"Erro GeoJSON: {e}"
    
    # 2. Carregar Previsão
    pred_path = config.ARTIFACTS[region_key]['prediction']
    
    if not pred_path.exists():
        gdf['CVLI'] = 0
        return gdf, 'CVLI', "Sem previsões. Rode o predict.py."
        
    try:
        df_pred = pd.read_csv(pred_path)
    except Exception as e:
        return gdf, None, f"Erro CSV: {e}"
    
    # 3. Cruzamento
    if 'name' in gdf.columns:
        gdf['name_upper'] = gdf['name'].astype(str).str.upper().str.strip()
    elif 'NM_MUNICIP' in gdf.columns:
        gdf['name_upper'] = gdf['NM_MUNICIP'].astype(str).str.upper().str.strip()
        gdf['name'] = gdf['NM_MUNICIP']
        
    df_pred['local'] = df_pred['bairro'].astype(str).str.upper().str.strip() 
    
    # Merge
    gdf_final = gdf.merge(df_pred, left_on='name_upper', right_on='local', how='left')
    
    # PRIORIZAR CVLI - SEMPRE usar CVLI como coluna alvo
    target_col = 'CVLI' 
    if target_col not in gdf_final.columns:
        # Se não houver CVLI, busca por outras colunas de crime, mas marca claramente
        available_cols = [c for c in gdf_final.columns if c.upper() not in ['NAME', 'NAME_UPPER', 'BAIRRO', 'LOCAL', 'GEOMETRY']]
        target_col = available_cols[-1] if available_cols else gdf_final.columns[-1]
        
    gdf_final[target_col] = gdf_final[target_col].fillna(0)
    
    return gdf_final, target_col, None

# --- SIDEBAR ESQUERDA (CONFIGURAÇÃO) ---
with st.sidebar:
    st.title("🦅 COMANDO")
    st.caption("Configuração Operacional")
    st.markdown("---")
    
    region_option = st.selectbox(
        "📍 TEATRO DE OPERAÇÕES",
        options=['CAPITAL', 'RMF', 'INTERIOR'],
        index=0
    )
    
    st.markdown("---")
    st.markdown("**📡 FILTROS DE SINAL**")
    
    risk_threshold = st.slider("Corte de Ruído", 0.0, 5.0, 0.5, 0.1)
    sensitivity = st.slider("Ganho de Contraste", 1.0, 4.0, 2.5, 0.1)
    
    st.markdown("---")
    if st.button("🔄 ATUALIZAR SISTEMA"):
        st.cache_data.clear()
        st.rerun()

# --- ÁREA PRINCIPAL ---
# Layout: Mapa (Grande) | Dados (Estreito)
col_map, col_data = st.columns([7, 2])

gdf, target_col, msg = load_data(region_option)

if msg and not gdf:
    st.error(msg)
    st.stop()

if gdf is not None:
    # Processamento Visual (Potência)
    max_val = gdf[target_col].max()
    if max_val == 0: max_val = 1
    
    gdf['display_risk'] = (gdf[target_col] / max_val) ** sensitivity * 10
    gdf['display_risk'] = gdf['display_risk'].apply(lambda x: x if (x * max_val) >= risk_threshold else 0)

    # --- MAPA ---
    with col_map:
        st.markdown(f"### 🗺️ Inteligência Geoespacial: {region_option}")
        
        if region_option == 'CAPITAL':
            center = [-3.78, -38.53]; zoom = 12
        elif region_option == 'RMF':
            center = [-3.90, -38.55]; zoom = 10
        else:
            center = [-5.2, -39.3]; zoom = 7

        m = folium.Map(location=center, zoom_start=zoom, tiles='CartoDB dark_matter', control_scale=True)
        
        if target_col in gdf.columns:
            folium.Choropleth(
                geo_data=gdf,
                data=gdf,
                columns=['name', 'display_risk'],
                key_on='feature.properties.name',
                fill_color='Reds',
                fill_opacity=0.8,
                line_opacity=0.1,
                line_weight=1,
                legend_name="Intensidade de Risco (CVLI - Crimes Violentos Letais)",
                nan_fill_opacity=0
            ).add_to(m)
            
            folium.GeoJson(
                gdf,
                style_function=lambda x: {'fillColor': '#00000000', 'color': '#00000000'},
                tooltip=folium.GeoJsonTooltip(
                    fields=['name', target_col],
                    aliases=['Zona:', 'Índice:'],
                    style="background-color: #111; color: #ff4b4b; font-weight: bold; border: 1px solid red;"
                )
            ).add_to(m)
        
        st_folium(m, width="100%", height=750)

    # --- COLUNA DE DADOS ---
    with col_data:
        # BOTÃO IA EM DESTAQUE (PRIMEIRA COISA)
        st.markdown("### 🧠 Inteligência Artificial")
        
        # Botão personalizado com CSS inline para destaque
        btn_ia = st.button("📄 GERAR ANÁLISE DE INTELIGÊNCIA", type="primary", use_container_width=True)
        
        if target_col in gdf.columns:
            # Garante que está ordenando por CVLI apenas (homicídios), filtrando zeros
            top_risks = gdf[gdf[target_col] > 0].sort_values(by=target_col, ascending=False).head(7)
            
            # Lógica do Botão IA
            if btn_ia:
                if top_risks.empty or top_risks[target_col].sum() == 0:
                    st.warning("Sem dados críticos para analisar.")
                else:
                    with st.spinner("Analisando padrões geográficos e eventos..."):
                        try:
                            client = GeminiRotator()
                            context = top_risks[['name', target_col]].to_string(index=False)
                            
                            # Prompt focado em análise de inteligência para GESTORES (não tática)
                            prompt = f"""
RELATÓRIO EXECUTIVO - ANÁLISE DE INTELIGÊNCIA
Região: {region_option}
Data: {pd.Timestamp.now().strftime('%d de %B de %Y')}

DADOS CRÍTICOS (CVLI - Crimes Violentos Letais Intencionais):
{context}

Você é Conselheiro de Inteligência apresentando para Gestores de Ações Policiais (NÃO para analistas técnicos).

ESTRUTURE O RELATÓRIO ASSIM:

## SITUAÇÃO CRÍTICA
Resuma em 2-3 linhas o QUADRO GERAL. Sem jargão técnico.

## PRIORIDADES OPERACIONAIS
- Aponte os 3 PRINCIPAIS PONTOS CRÍTICOS
- Por que cada um requer atenção
- Indicadores de intensidade (Alto/Crítico/Máximo)

## PADRÕES IDENTIFICADOS
- Que padrões espaciais existem?
- Qual a importância para o gestor?

## RECOMENDAÇÕES PARA AÇÃO
- 3-4 ações concretas de investigação
- Recursos necessários (equipes, dispositivos)
- Áreas que precisam de reforço

TONE: Executivo, direto, orientado a AÇÃO. Sem números técnicos excessivos.
                            """
                            
                            response = client.generate_content(prompt)
                            st.session_state.relatorio_conteudo = response
                            st.session_state.relatorio_aberto = True
                            st.rerun()
                        except Exception as e:
                            st.error(f"Erro na análise: {e}")
            
            # Sidebar com zonas prioritárias (sempre visível quando relatório não está aberto)
            if not st.session_state.relatorio_aberto:
                st.markdown("---")
                st.markdown("### 🎯 Zonas com Maior Incidência de CVLI")
                
                if not top_risks.empty and top_risks[target_col].sum() > 0:
                    for idx, (i, row) in enumerate(top_risks.iterrows(), 1):
                        score = row[target_col]
                        # Critério: se está no top 3, é crítico
                        icon = "🔴" if idx <= 3 else "🟠"
                        
                        st.markdown(
                            f"""
                            <div class="metric-card">
                                <div style="font-size: 12px; color: #aaa; margin-bottom: 4px;">#{idx} - {row['name']}</div>
                                <div style="display: flex; justify-content: space-between; align-items: center;">
                                    <span style="font-size: 18px;">{icon}</span>
                                    <span style="font-size: 18px; font-weight: bold; color: #ff6b6b;">{score:.2f}</span>
                                </div>
                                <div style="font-size: 10px; color: #888; margin-top: 4px;">Índice de Risco CVLI</div>
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )
                else:
                    st.info("✅ Nenhuma zona com risco elevado de homicídios no período.")
        else:
            st.info("Aguardando dados.")

else:
    st.error("Erro crítico.")

# --- MODAL POPUP (RENDERIZADO SOBRE TODA A PÁGINA) ---
# Modal Popup do Relatório - Renderizado sobre toda a página
if st.session_state.relatorio_aberto:
    # Prepara o conteúdo do relatório com formatação HTML
    conteudo_raw = st.session_state.relatorio_conteudo
    
    # Formatação com regex para preservar HTML
    conteudo_formatado = conteudo_raw
    
    # Quebras de linha
    conteudo_formatado = conteudo_formatado.replace("\n", "<br>")
    
    # Bold: ** texto **
    conteudo_formatado = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', conteudo_formatado)
    
    # Headers: ## Título
    conteudo_formatado = re.sub(r'^## (.*?)$', r'<h3>\1</h3>', conteudo_formatado, flags=re.MULTILINE)
    
    # Subheaders: ### Subtítulo
    conteudo_formatado = re.sub(r'^### (.*?)$', r'<h4>\1</h4>', conteudo_formatado, flags=re.MULTILINE)
    
    # Escape para JavaScript
    import json
    conteudo_json = json.dumps(conteudo_formatado)
    
    # CSS + HTML do modal injetado via st.markdown para garantir overlay
    st.markdown(f"""
    <style>
        .modal-backdrop {{
            position: fixed !important;
            top: 0 !important;
            left: 0 !important;
            width: 100% !important;
            height: 100% !important;
            background-color: rgba(0, 0, 0, 0.85) !important;
            z-index: 999999 !important;
        }}
        
        .modal-dialog {{
            position: fixed !important;
            top: 50% !important;
            left: 50% !important;
            transform: translate(-50%, -50%) !important;
            z-index: 9999999 !important;
            background: linear-gradient(135deg, #1a1f2e 0%, #16213e 100%) !important;
            border: 2px solid #ff6b6b !important;
            border-radius: 16px !important;
            width: 85vw !important;
            max-width: 900px !important;
            height: 80vh !important;
            display: flex !important;
            flex-direction: column !important;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.9) !important;
            font-family: Arial, sans-serif !important;
        }}
        
        .modal-header {{
            padding: 20px 25px !important;
            border-bottom: 2px solid #ff6b6b !important;
            display: flex !important;
            justify-content: space-between !important;
            align-items: center !important;
            background: rgba(233, 69, 96, 0.1) !important;
            border-radius: 14px 14px 0 0 !important;
        }}
        
        .modal-header h2 {{
            color: #fff !important;
            font-size: 22px !important;
            font-weight: bold !important;
            margin: 0 !important;
        }}
        
        .modal-close {{
            color: #ff6b6b !important;
            font-size: 28px !important;
            font-weight: bold !important;
            cursor: pointer !important;
            user-select: none !important;
        }}
        
        .modal-close:hover {{
            color: #ff4b4b !important;
        }}
        
        .modal-body {{
            flex: 1 !important;
            padding: 25px !important;
            overflow-y: auto !important;
            color: #e0e0e0 !important;
            font-size: 15px !important;
            line-height: 1.8 !important;
            background: rgba(26, 31, 46, 0.95) !important;
        }}
        
        .modal-body h3 {{
            color: #ff6b6b !important;
            margin-top: 20px !important;
            margin-bottom: 10px !important;
        }}
        
        .modal-body h4 {{
            color: #ffb366 !important;
            margin-top: 15px !important;
            margin-bottom: 8px !important;
        }}
        
        .modal-body b {{
            color: #ff6b6b !important;
        }}
        
        .modal-footer {{
            padding: 15px 25px !important;
            background: rgba(233, 69, 96, 0.1) !important;
            border-top: 2px solid #ff6b6b !important;
            border-radius: 0 0 14px 14px !important;
        }}
    </style>
    
    <div class="modal-backdrop"></div>
    <div class="modal-dialog">
        <div class="modal-header">
            <h2>📊 Análise de Inteligência Operacional</h2>
            <span class="modal-close" onclick="window.location.reload();">✕</span>
        </div>
        
        <div class="modal-body" id="modal-body-content"></div>
        
        <div class="modal-footer">
            <span style="color: #aaa; font-size: 12px;">Relatório gerado com sucesso - Use o botão abaixo para exportar PDF</span>
        </div>
    </div>
    
    <script>
        const conteudo = {conteudo_json};
        document.getElementById('modal-body-content').innerHTML = conteudo;
    </script>
    """, unsafe_allow_html=True)
    
    # Botões de ação fora do modal (no Streamlit) para gerar PDF
    st.markdown("---")
    col_spacer, col_pdf = st.columns([2, 1])
    
    with col_pdf:
        if st.button("📥 EXPORTAR PDF", use_container_width=True, key="btn_exportar_relatorio"):
            try:
                # Obtém os dados dos riscos
                top_risks = gdf[gdf[target_col] > 0].sort_values(by=target_col, ascending=False).head(7)
                dados_risco = top_risks[['name', target_col]].set_index('name')[target_col].to_dict()
                
                caminho_pdf = gerar_pdf_relatorio(
                    st.session_state.relatorio_conteudo,
                    f"Análise_{region_option}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}",
                    dados_risco=dados_risco
                )
                st.success(f"✅ PDF gerado: {caminho_pdf.name}", icon="✅")
                
                # Download
                with open(caminho_pdf, "rb") as f:
                    st.download_button(
                        label="⬇️ Baixar Arquivo",
                        data=f.read(),
                        file_name=caminho_pdf.name,
                        mime="application/pdf",
                        use_container_width=True,
                        key="download_relatorio"
                    )
            except Exception as e:
                st.error(f"❌ Erro ao gerar PDF: {str(e)}", icon="❌")
=======
from flask import Flask, render_template, jsonify, request
import pandas as pd
import geopandas as gpd
import json
import sys
import os
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import config

app = Flask(__name__)

def load_processed_data():
    if config.CONSOLIDATED_FILE.exists():
        return pd.read_parquet(config.CONSOLIDATED_FILE)
    return pd.DataFrame()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/data')
def get_data():
    df = load_processed_data()
    if df.empty: return jsonify([])
    
    # Filtros
    region = request.args.get('region')
    if region:
        df = df[df['regiao_sistema'] == region]
        
    # Converter para dict leve
    # Vamos enviar apenas o necessário para o mapa
    data = df[['lat', 'lng', 'natureza', 'bairro_ciops', 'faccao_predominante', 'data_hora']].to_dict(orient='records')
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
>>>>>>> 73db3feb (Initial commit: add project files, exclude venv)
