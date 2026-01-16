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