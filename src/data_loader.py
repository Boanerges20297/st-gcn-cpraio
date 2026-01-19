import pandas as pd
import json
import os
from pathlib import Path
from . import config

# CVLI keywords para detecção automática
if not hasattr(config, 'CVLI_KEYWORDS'):
    config.CVLI_KEYWORDS = [
        'HOMICÍDIO',
        'LATROCÍNIO',
        'ROUBO',
        'ESTUPRO',
        'ASSALTO',
        'TENTATIVA'
    ]

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

# ============================================================================
# NOVAS FUNÇÕES PARA NORMALIZAÇÃO V2 (Sprint 1)
# ============================================================================

def detect_cvli(natureza_str, cvli_keywords=None):
    """
    Identifica se uma ocorrência é CVLI baseado em palavras-chave.
    
    Args:
        natureza_str: String da natureza da ocorrência
        cvli_keywords: Lista de palavras-chave CVLI (usa config se None)
    
    Returns:
        bool: True se é CVLI
        
    Exemplo:
        >>> detect_cvli("HOMICÍDIO - Art. 121/CPB")
        True
    """
    if cvli_keywords is None:
        cvli_keywords = config.CVLI_KEYWORDS
    
    if not isinstance(natureza_str, str):
        return False
    
    natureza_upper = natureza_str.upper()
    return any(kw in natureza_upper for kw in cvli_keywords)


def extract_lat_long(lat_long_str):
    """
    Extrai latitude e longitude do campo 'lat_long' do JSON operacional.
    
    Formato esperado: "-3.7668038,-38.584197"
    
    Args:
        lat_long_str: String com coords separadas por vírgula
    
    Returns:
        tuple: (lat, long) como floats, ou (None, None) se inválido
        
    Exemplo:
        >>> extract_lat_long("-3.7668038,-38.584197")
        (-3.7668038, -38.584197)
    """
    if not isinstance(lat_long_str, str):
        return None, None
    
    try:
        parts = lat_long_str.strip().split(',')
        lat = float(parts[0].strip())
        long = float(parts[1].strip())
        
        # Validar ranges
        if -90 <= lat <= 90 and -180 <= long <= 180:
            return lat, long
        else:
            return None, None
    except (ValueError, IndexError):
        return None, None


def parse_operational_json(json_path):
    """
    Converte JSON operacional para DataFrame estruturado com features CVLI.
    
    Args:
        json_path: Caminho para ocorrencia_policial_operacional.json
    
    Returns:
        pd.DataFrame: Dados normalizados com as seguintes colunas:
            - data: datetime
            - municipio: str
            - bairro: str
            - lat, long: float (coordenadas)
            - natureza: str
            - is_cvli: bool (NOVO!)
            - total_armas: int
            - total_drogas_g: float
            - has_large_seizure: bool (NOVO!)
            - has_weapons_drugs: bool (NOVO!)
            - dinheiro_apreendido: float
            - area_faccao: str
            - local_ocorrencia: str
    """
    print(f"\n>>> Parsing JSON Operacional: {json_path}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    # Filtrar apenas registros de dados (tipo='table' com data)
    records = []
    for item in raw_data:
        if isinstance(item, dict):
            # Estrutura: {"type": "table", "data": [...]}
            if item.get('type') == 'table' and 'data' in item:
                records.extend(item['data'])
            # Ou item direto é um registro
            elif 'Natureza' in item:
                records.append(item)
    
    print(f"    [+] {len(records)} registros extraídos")
    
    df = pd.DataFrame(records)
    
    # Mapeamento de colunas específico do JSON operacional
    column_map = {
        'Data': 'data',
        'CidadeOcor': 'municipio',
        'BairroOcor': 'bairro',
        'Natureza': 'natureza',
        'total_armas_cache': 'total_armas',
        'total_drogas_cache': 'total_drogas_g',
        'Dinheiro_Apreendido': 'dinheiro_apreendido',
        'area_faccao': 'area_faccao',
        'LocalOcor': 'local_ocorrencia',
        'lat_long': 'lat_long_str',  # Temporário
    }
    
    df = df.rename(columns=column_map)
    
    # Converter tipos
    print("    [·] Normalizando tipos de dados...")
    df['data'] = pd.to_datetime(df['data'], format='%Y-%m-%d', errors='coerce')
    df['total_armas'] = pd.to_numeric(df['total_armas'], errors='coerce').fillna(0).astype(int)
    df['total_drogas_g'] = pd.to_numeric(df['total_drogas_g'], errors='coerce').fillna(0)
    df['dinheiro_apreendido'] = pd.to_numeric(df['dinheiro_apreendido'], errors='coerce').fillna(0)
    
    # Extrair lat/long
    print("    [·] Extraindo coordenadas...")
    coords = df['lat_long_str'].apply(extract_lat_long)
    df['lat'] = coords.apply(lambda x: x[0])
    df['long'] = coords.apply(lambda x: x[1])
    
    # Detectar CVLI
    print("    [·] Detectando CVLI...")
    df['is_cvli'] = df['natureza'].apply(detect_cvli)
    
    # Detectar apreensões grandes (>= 1kg = 1000g)
    df['has_large_seizure'] = df['total_drogas_g'] >= 1000
    
    # Detectar armas + drogas
    df['has_weapons_drugs'] = (df['total_armas'] > 0) & (df['total_drogas_g'] > 0)
    
    # Descartar registros sem geolocalização válida
    initial_count = len(df)
    df = df.dropna(subset=['lat', 'long', 'data'])
    print(f"    [V] Registros com geoloc válida: {len(df)} / {initial_count}")
    
    # Ordenar por data
    df = df.sort_values('data').reset_index(drop=True)
    
    # Estatísticas
    print(f"\n    Resumo:")
    print(f"      - CVLI: {df['is_cvli'].sum()} ({100*df['is_cvli'].mean():.1f}%)")
    print(f"      - Apreensões >= 1kg: {df['has_large_seizure'].sum()} ({100*df['has_large_seizure'].mean():.1f}%)")
    print(f"      - Arma + Droga: {df['has_weapons_drugs'].sum()} ({100*df['has_weapons_drugs'].mean():.1f}%)")
    print(f"      - Data range: {df['data'].min().date()} → {df['data'].max().date()}")
    
    return df


def normalize_tropa_coordinates(coord_str):
    """
    Converte coordenadas em diferentes formatos para decimal (lat, long).
    
    Suporta:
    - Decimal: "-3.7668038,-38.584197"
    - DMS: "-5°15'53.4\"S,-37°56'37.8\"W"
    - Parcial: "-5°15' / -37°56'"
    
    Args:
        coord_str: String de coordenada em qualquer formato
    
    Returns:
        tuple: (lat, long) como floats, ou (None, None) se não parsear
    """
    import re
    
    if not isinstance(coord_str, str):
        return None, None
    
    coord_str = coord_str.strip()
    
    try:
        # Tentar formato decimal direto
        if ',' in coord_str and not '°' in coord_str:
            parts = coord_str.split(',')
            lat = float(parts[0].strip())
            long = float(parts[1].strip())
            if -90 <= lat <= 90 and -180 <= long <= 180:
                return lat, long
        
        # Tentar formato DMS: "-5°15'53.4"S,-37°56'37.8"W"
        dms_pattern = r"(-?\d+)°(\d+)'([\d.]+)[\"']?([NSEW])?,?(-?\d+)°(\d+)'([\d.]+)[\"']?([NSEW])?"
        match = re.search(dms_pattern, coord_str)
        
        if match:
            deg1, min1, sec1, dir1, deg2, min2, sec2, dir2 = match.groups()
            
            # Converter primeiro valor (lat)
            lat = int(deg1) + int(min1)/60 + float(sec1)/3600
            if dir1 and dir1 in ['S', 'W']:
                lat = -lat
            
            # Converter segundo valor (long)
            if deg2:
                long = int(deg2) + int(min2)/60 + float(sec2)/3600
                if dir2 and dir2 in ['S', 'W']:
                    long = -long
            else:
                return None, None
            
            if -90 <= lat <= 90 and -180 <= long <= 180:
                return lat, long
        
        return None, None
    
    except (ValueError, AttributeError):
        return None, None


def parse_tropa_narrative(narrative_str, date_str=None):
    """
    Extrai informações estruturadas de narrativas de operação de tropa.
    
    Exemplos de extração:
    - Presos: "02 presos", "3 indivíduos presos"
    - Drogas: "250g de cocaína", "1.5kg de maconha"
    - Armas: "1 revólver", "2 espingardas"
    - Coords: "-3.76,-38.58" ou "-5°15'53.4"S,-37°56'37.8"W"
    
    Args:
        narrative_str: Texto da operação
        date_str: Data da operação (YYYY-MM-DD)
    
    Returns:
        dict: Chaves extraídas:
            - total_presos: int
            - total_drogas_g: float
            - total_armas: int
            - lat, long: float
            - operacao_tipo: str
            - success_score: float (0-1 qualidade da extração)
    """
    import re
    
    result = {
        'total_presos': 0,
        'total_drogas_g': 0,
        'total_armas': 0,
        'lat': None,
        'long': None,
        'operacao_tipo': 'operacao_rotina',
        'success_score': 0.0,
        'date': date_str
    }
    
    if not isinstance(narrative_str, str) or len(narrative_str.strip()) == 0:
        return result
    
    narrative_lower = narrative_str.lower()
    score_hits = 0  # Track how many fields were successfully extracted
    
    # Extrair presos
    presos_patterns = [
        r'(\d+)\s+(?:preso|presos|indivíduo|indivíduos|homens?\s+presos)',
        r'(?:preso|presos).*?(\d+)',
        r'(\d+)\s+(?:capturado|capturados|detido|detidos)'
    ]
    for pattern in presos_patterns:
        match = re.search(pattern, narrative_lower)
        if match:
            result['total_presos'] = int(match.group(1))
            score_hits += 1
            break
    
    # Extrair drogas (em gramas)
    drogas_patterns = [
        r'(\d+(?:\.\d+)?)\s*(?:kg|quilos?|kilos?)\s+de\s+(?:drogas?|cocaína|crack|maconha)',
        r'(\d+(?:\.\d+)?)\s*(?:kg|quilos?|kilos?)',  # Genérico kg
        r'(\d+(?:\.\d+)?)\s*g(?:ramas?)?(?:\s+de\s+drogas?|de\s+cocaína)',
        r'(?:drogas?).*?(\d+(?:\.\d+)?)\s*(?:kg|g)',
    ]
    for pattern in drogas_patterns:
        match = re.search(pattern, narrative_lower)
        if match:
            amount = float(match.group(1))
            # Se unidade for kg, converter para gramas
            if 'kg' in match.group(0).lower():
                amount *= 1000
            result['total_drogas_g'] = amount
            score_hits += 1
            break
    
    # Extrair armas
    armas_patterns = [
        r'(\d+)\s+(?:revólver|revól|arma|espingarda|fuzil|pistola|metralhadora|carabina)',
        r'(?:arma|revólver|espingarda).*?(\d+)',
        r'(?:apreendida?|apreendidas?)\s+(?:uma?|)?(\d+)\s+(?:arma|revólver)',
    ]
    for pattern in armas_patterns:
        match = re.search(pattern, narrative_lower)
        if match:
            result['total_armas'] = int(match.group(1))
            score_hits += 1
            break
    
    # Extrair coordenadas
    coord_patterns = [
        r'(-?\d+\.\d+),?\s*(-?\d+\.\d+)',  # Decimal
        r'(-\d+)°(\d+)[\'°](\d+)[\"\']?([NSEW])?,?\s*(-?\d+)°(\d+)[\'°](\d+)[\"\']?([NSEW])?',  # DMS
    ]
    for pattern in coord_patterns:
        match = re.search(pattern, narrative_str)  # Use original case for coords
        if match:
            coord_str = match.group(0)
            lat, long = normalize_tropa_coordinates(coord_str)
            if lat is not None and long is not None:
                result['lat'] = lat
                result['long'] = long
                score_hits += 1
            break
    
    # Classificar tipo de operação
    if 'grande apreensão' in narrative_lower or 'megaoperação' in narrative_lower:
        result['operacao_tipo'] = 'megaoperacao'
    elif 'com_pouca_incidência' in narrative_lower:
        result['operacao_tipo'] = 'baixa_incidencia'
    elif 'patrulhamento' in narrative_lower:
        result['operacao_tipo'] = 'patrulhamento'
    
    # Score: quanto mais campos extraídos, melhor
    result['success_score'] = min(score_hits / 4.0, 1.0)  # 4 campos principais
    
    return result


def parse_tropa_dataset(json_path=None):
    """
    Processa dataset completo de tropas e retorna DataFrame normalizado.
    
    O arquivo de tropas é um export de PHPMyAdmin com estrutura:
    - id_ocorrencia: identificador
    - data_registro: data e hora
    - ocorrencia: texto narrativo com informações estruturadas
    - latitude, longitude: coordenadas (frequentemente None)
    - local_ocorrencia: descrição de local (frequentemente None)
    
    Args:
        json_path: Caminho para ocorrencias_tropa.json ou None para detectar
    
    Returns:
        pd.DataFrame: Dados normalizados das tropas com colunas:
            - data: datetime (extraída de ocorrencia ou data_registro)
            - municipio: str (extraído de ocorrencia)
            - total_presos: int (extraído de narrativa)
            - total_drogas_g: float (extraído de narrativa)
            - total_armas: int (extraído de narrativa)
            - lat, long: float (coordenadas)
            - natureza: str (tipo de ocorrência extraído de ocorrencia)
            - operacao_tipo: str
            - success_score: float (0-1)
    """
    
    # Detectar arquivo se não informado
    if json_path is None:
        candidates = [
            "data/raw/ocorrencias_tropa.json",
            "data/raw/tropas.json",
            "data/raw/tropa_operacoes.json"
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                json_path = candidate
                break
    
    if json_path is None or not os.path.exists(json_path):
        print(f"[AVISO] Arquivo de tropas não encontrado. Retornando DataFrame vazio.")
        return pd.DataFrame()
    
    print(f"\n>>> Parsing JSON Tropas (PHPMyAdmin export): {json_path}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    # Extrair registros (tipo = 'table')
    records = []
    for item in raw_data:
        if isinstance(item, dict) and item.get('type') == 'table':
            records = item.get('data', [])
            break
    
    print(f"    [+] {len(records)} registros extraídos")
    
    # Funções auxiliares para parsing de narrativa de tropa
    def extract_data_from_narrativa(narrativa_text):
        """Extrai data da narrativa estruturada"""
        import re
        # Procurar padrão: "01 - DATA:11/12/2025"
        match = re.search(r'01\s*-\s*DATA[:\s]+(\d{1,2})/(\d{1,2})/(\d{4})', narrativa_text or '')
        if match:
            day, month, year = match.groups()
            try:
                return pd.to_datetime(f"{year}-{month}-{day}", format='%Y-%m-%d')
            except:
                return None
        return None
    
    def extract_municipio(narrativa_text):
        """Extrai município da narrativa estruturada"""
        import re
        # Procurar "MUNICÍPIO:" ou "MUNICIPIO:"
        match = re.search(r'MUNIC[IÍ]PIO[:\s-]+([A-ZÁÉÍÓÚ\s]+?)(?:-CE|-SP|$|\n)', narrativa_text or '')
        if match:
            municipio = match.group(1).strip()
            # Remover sufixos comuns
            municipio = municipio.replace('-CE', '').replace('-SP', '').strip()
            return municipio
        return None
    
    def extract_drogas_from_narrativa(narrativa_text):
        """Extrai quantidade de drogas em gramas"""
        import re
        total_g = 0.0
        
        if not narrativa_text:
            return 0.0
        
        # Procurar padrão: "COCAÍNA (PESO EM GRAMA): 40 PAPELOTES (10G)"
        cocaine_match = re.search(r'COCAÍNA\s*\(PESO EM GRAMA[S]?\)[:\s]+([\d\.]+)\s*(?:PAPELOTES|G|GR|GRAMA)', narrativa_text, re.IGNORECASE)
        if cocaine_match:
            try:
                # Extrair apenas número
                amount_str = re.search(r'(\d+(?:\.\d+)?)', cocaine_match.group(1))
                if amount_str:
                    total_g += float(amount_str.group(1)) * 10  # Papelotes são ~10g cada
            except:
                pass
        
        # Procurar "MACONHA (PESO EM GRAMA): ..."
        marijuana_match = re.search(r'MACONHA\s*\(PESO EM GRAMA[S]?\)[:\s]+([\d\.\s,;PAPELOTES\(\)GRT]+)', narrativa_text, re.IGNORECASE)
        if marijuana_match:
            match_text = marijuana_match.group(1)
            # Procurar números com "G" ou "GRAMA"
            numbers = re.findall(r'(\d+(?:\.\d+)?)\s*(?:PAPELOTES|G|GRAMA|TABLETES)', match_text)
            total_g += sum(float(n) for n in numbers)
        
        # Procurar números em gramas genéricas
        generic_drugs = re.findall(r'(\d+(?:\.\d+)?)\s*[GG][RAMA]*\s+(?:de\s+)?(?:DROGAS?|COCAÍNA|CRACK|MACONHA)', narrativa_text, re.IGNORECASE)
        for amount in generic_drugs:
            total_g += float(amount)
        
        return total_g
    
    def extract_armas_from_narrativa(narrativa_text):
        """Extrai quantidade de armas"""
        import re
        # Procurar padrão: "UMA BALANÇA", "1 REVÓLVER", etc
        armas_patterns = [
            r'(\d+)\s+(?:REVÓLVER|REVOLVER|PISTOLA|FUZIL|ESCOPETA|METRALHADORA|CARABINA)',
            r'(?:UMA|UM)\s+(?:REVÓLVER|REVOLVER|PISTOLA|FUZIL)',
        ]
        
        total_armas = 0
        for pattern in armas_patterns:
            matches = re.findall(pattern, narrativa_text or '', re.IGNORECASE)
            for match in matches:
                if isinstance(match, str):
                    try:
                        total_armas += int(match)
                    except:
                        total_armas += 1
                else:
                    total_armas += 1
        
        return total_armas
    
    def extract_natureza(narrativa_text):
        """Extrai natureza da ocorrência"""
        import re
        # Procurar "10 - NATUREZA DA OCORRÊNCIA"
        match = re.search(r'10\s*-\s*NATUREZA[^:]*:[^0-9]*([0-9]+)(?:\s*-\s*(.+?))?(?=\n\d{2}\s*-|\Z)', narrativa_text or '', re.IGNORECASE)
        if match:
            return f"Artigo {match.group(1)}"
        
        # Fallback: procurar "NATUREZA"
        match = re.search(r'NATUREZA.*?:\s*([^\n]+)', narrativa_text or '', re.IGNORECASE)
        if match:
            return match.group(1).strip()
        
        return "Operação Tropa"
    
    # Processar cada registro
    processed_records = []
    for record in records:
        if not isinstance(record, dict):
            continue
        
        # Extrair campos básicos
        id_ocorrencia = record.get('id_ocorrencia')
        data_registro = record.get('data_registro')
        narrativa = record.get('ocorrencia', '')
        
        # Tentar extrair data da narrativa, fallback para data_registro
        data_dt = extract_data_from_narrativa(narrativa)
        if pd.isna(data_dt):
            try:
                data_dt = pd.to_datetime(data_registro, errors='coerce')
            except:
                data_dt = pd.NaT
        
        # Tentar extrair municipio
        municipio = extract_municipio(narrativa)
        if not municipio:
            municipio = "Desconhecido"
        
        # Extrair informações da narrativa
        total_presos = 0  # Não extraído facilmente da estrutura
        total_drogas_g = extract_drogas_from_narrativa(narrativa)
        total_armas = extract_armas_from_narrativa(narrativa)
        natureza = extract_natureza(narrativa)
        
        # Coordenadas
        lat = record.get('latitude')
        long = record.get('longitude')
        if lat and isinstance(lat, str):
            try:
                lat = float(lat)
            except:
                lat = None
        if long and isinstance(long, str):
            try:
                long = float(long)
            except:
                long = None
        
        # Score de sucesso
        score_hits = 0
        if total_drogas_g > 0:
            score_hits += 1
        if total_armas > 0:
            score_hits += 1
        if lat is not None and long is not None:
            score_hits += 1
        success_score = score_hits / 3.0
        
        processed_record = {
            'data': data_dt,
            'municipio': municipio,
            'natureza': natureza,
            'total_presos': total_presos,
            'total_drogas_g': total_drogas_g,
            'total_armas': total_armas,
            'lat': lat,
            'long': long,
            'operacao_tipo': 'operacao_tropa',
            'success_score': success_score
        }
        
        processed_records.append(processed_record)
    
    df_tropa = pd.DataFrame(processed_records)
    
    # Remover registros com datas inválidas
    initial_count = len(df_tropa)
    df_tropa = df_tropa.dropna(subset=['data'])
    print(f"    [V] Registros com data válida: {len(df_tropa)} / {initial_count}")
    
    print(f"    [V] Score médio de sucesso: {df_tropa['success_score'].mean():.2%}")
    
    # Estatísticas
    if len(df_tropa) > 0:
        print(f"\n    Resumo das Tropas:")
        print(f"      - Total presos: {df_tropa['total_presos'].sum():.0f}")
        print(f"      - Total drogas: {df_tropa['total_drogas_g'].sum():.0f}g")
        print(f"      - Total armas: {df_tropa['total_armas'].sum():.0f}")
        print(f"      - Operações com coords: {df_tropa[['lat', 'long']].notna().all(axis=1).sum()}")
        print(f"      - Operações com drogas: {(df_tropa['total_drogas_g'] > 0).sum()}")
        print(f"      - Operações com armas: {(df_tropa['total_armas'] > 0).sum()}")
    
    return df_tropa


if __name__ == "__main__":
    df = load_raw_data()