import pandas as pd
import json
import os
from pathlib import Path
import config

def normalize_columns(df):
    """
    Padroniza os nomes das colunas e tenta converter tipos.
    """
    # Mapa expandido de variações comuns em bases policiais
    column_map = {
        # Latitude
        'Latitude': 'lat', 'LATITUDE': 'lat', 'latitude': 'lat',
        'lat': 'lat', 'LAT': 'lat', 'nr_latitude': 'lat',
        'lat_geo': 'lat', 'geo_lat': 'lat',
        
        # Longitude
        'Longitude': 'long', 'LONGITUDE': 'long', 'longitude': 'long',
        'long': 'long', 'LONG': 'long', 'nr_longitude': 'long',
        'lng': 'long', 'LNG': 'long', 'long_geo': 'long', 'geo_long': 'long',
        
        # Cidade
        'Cidade': 'municipio', 'CIDADE': 'municipio', 'cidade': 'municipio',
        'Municipio': 'municipio', 'MUNICIPIO': 'municipio', 'municipio': 'municipio',
        'CidadeOcor': 'municipio', 'nm_municipio': 'municipio',
        
        # Data
        'Data': 'date', 'DATA': 'date', 'data': 'date',
        'DataOcor': 'date', 'data_hora': 'date', 'dt_fato': 'date',
        
        # Natureza
        'Natureza': 'natureza', 'NATUREZA': 'natureza', 'natureza': 'natureza',
        'Descricao': 'natureza', 'ds_natureza': 'natureza'
    }
    
    # Renomear
    df = df.rename(columns=column_map)
    
    # Validação de Coordenadas
    # Se 'lat' ou 'long' não existirem, tenta criá-las vazias para não dar KeyError no dropna
    if 'lat' not in df.columns:
        print("[!] AVISO: Coluna de LATITUDE não identificada automaticamente.")
    if 'long' not in df.columns:
        print("[!] AVISO: Coluna de LONGITUDE não identificada automaticamente.")
        
    # Garantir numérico se as colunas existirem
    if 'lat' in df.columns:
        df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
    if 'long' in df.columns:
        df['long'] = pd.to_numeric(df['long'], errors='coerce')
        
    return df

def load_raw_data():
    raw_path = config.DATA_RAW
    all_files = list(raw_path.glob("*.json"))
    
    if not all_files:
        print(f"[!] AVISO: Nenhum arquivo .json encontrado em {raw_path}")
        return pd.DataFrame()

    print(f"[-] Carregando {len(all_files)} arquivos de dados brutos...")
    
    combined_data = []
    
    for file_path in all_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # Tratamento para lista de registros
                if isinstance(data, list):
                    combined_data.extend(data)
                # Tratamento para dict com chave 'data' ou 'features'
                elif isinstance(data, dict):
                    if 'data' in data and isinstance(data['data'], list):
                        combined_data.extend(data['data'])
                    elif 'features' in data:
                        extracted = [f['properties'] for f in data['features']]
                        combined_data.extend(extracted)
                    else:
                        combined_data.append(data)
                        
            print(f"    [+] Lido: {file_path.name}")
            
        except Exception as e:
            print(f"    [X] Erro ao ler {file_path.name}: {e}")

    if not combined_data:
        return pd.DataFrame()
        
    df = pd.DataFrame(combined_data)
    
    # --- DEBUG TÁTICO ---
    # Isso vai imprimir as colunas reais para você me dizer quais são os nomes certos
    print("\n" + "="*40)
    print(" [DEBUG] COLUNAS ENCONTRADAS NO ARQUIVO:")
    print(list(df.columns))
    print("="*40 + "\n")
    # --------------------
    
    df = normalize_columns(df)
    
    # Limpeza Segura (Só roda dropna se as colunas existirem)
    if 'lat' in df.columns and 'long' in df.columns:
        initial_len = len(df)
        df = df.dropna(subset=['lat', 'long'])
        print(f"[V] Registros georreferenciados válidos: {len(df)} (de {initial_len})")
    else:
        print("[X] ERRO CRÍTICO: Não foi possível georreferenciar os dados.")
        print("    Motivo: As colunas de latitude/longitude não foram encontradas.")
        print("    Ação: Verifique o print de colunas acima e ajuste o 'column_map'.")
        # Retorna vazio para não quebrar o resto, mas avisa que falhou
        return pd.DataFrame()
    
    # Tratamento de Data
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.sort_values(by='date')
    
    return df

if __name__ == "__main__":
    df = load_raw_data()