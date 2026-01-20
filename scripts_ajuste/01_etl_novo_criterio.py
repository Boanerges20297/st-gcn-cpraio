"""
NOVO ETL COM CRITÉRIO CVLI-CENTRIC
====================================
1. Treino: 2022-2024 (CVLI apenas)
2. Validação: 2025 (CVLI apenas)
3. CVP é apenas contexto histórico, não afeta criticidade
4. Análise de prisões RAIO em 2025
"""

import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
from datetime import datetime
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import config

def load_dados_status_ocorrencias():
    """
    Carrega dados_status_ocorrencias_gerais.json (já enriquecido com bairros)
    """
    # Tentar carregar versão enriquecida
    enriched_path = config.DATA_PROCESSED / "dados_status_enriquecidos_com_bairros.parquet"
    
    if enriched_path.exists():
        print("[-] Carregando versão enriquecida (com bairros)...")
        df = pd.read_parquet(enriched_path)
        print(f"[+] Carregados {len(df)} registros enriquecidos")
    else:
        print("[!] Versão enriquecida não encontrada")
        print("    Execute primeiro: python scripts_ajuste/00_spatial_join_enriquecimento.py")
        
        json_path = config.DATA_RAW / "dados_status_ocorrencias_gerais.json"
        
        print("[-] Carregando dados_status_ocorrencias_gerais.json...")
        
        try:
            with open(json_path, 'r', encoding='utf-8-sig', errors='replace') as f:
                content = json.load(f)
        except Exception as e:
            print(f"[X] Erro ao ler JSON: {e}")
            return pd.DataFrame()
        
        # O arquivo tem estrutura: [header, database, table, {data: [registros]}]
        raw_data = []
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and 'data' in item:
                    raw_data = item['data']
                    break
        
        if not raw_data:
            print("[!] Nenhum registro encontrado")
            return pd.DataFrame()
        
        df = pd.DataFrame(raw_data)
        print(f"[+] Carregados {len(df)} registros brutos")
    
    # Renomear colunas para padronização
    cols_map = {
        'latitude': 'lat', 'longitude': 'lng',
        'data': 'data', 'hora': 'hora',
        'tipo_evento': 'natureza',
        'id': 'id_ocorrencia',
        'bairro': 'bairro',  # Já vem do spatial join se enriquecido
        'tipo_crime': 'tipo_crime',  # Renomeado pelo spatial join
        'area_faccao': 'faccao',
        'cidade': 'municipio',
        'ais': 'ais',
        'agencia': 'agencia'
    }
    
    # Aplicar rename apenas para colunas que existem
    cols_existentes = [k for k in cols_map.keys() if k in df.columns]
    df = df.rename(columns={k: cols_map[k] for k in cols_existentes})
    
    # Converter data
    df['data'] = pd.to_datetime(df['data'], errors='coerce')
    
    # Limpar GPS
    df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
    df['lng'] = pd.to_numeric(df['lng'], errors='coerce')
    df = df.dropna(subset=['lat', 'lng', 'data'])
    
    # Validar tipo de crime
    df['tipo_crime'] = df['tipo_crime'].str.lower().fillna('desconhecido')
    
    # Filtrar coordenadas válidas (Ceará)
    df = df[(df['lat'] < -2) & (df['lat'] > -8) & (df['lng'] < -37) & (df['lng'] > -42)]
    
    print(f"[V] Após validação GPS e período: {len(df)} registros")
    
    return df

def split_train_validation(df):
    """
    Divide dados em:
    - TREINO: 2022-2024
    - VALIDAÇÃO: 2025
    """
    df_train = df[(df['data'].dt.year >= 2022) & (df['data'].dt.year <= 2024)].copy()
    df_val = df[df['data'].dt.year == 2025].copy()
    
    print(f"\n[SPLIT TEMPORAL]")
    print(f"  Treino (2022-2024): {len(df_train)} registros")
    print(f"  Validação (2025):   {len(df_val)} registros")
    
    return df_train, df_val

def analyze_cvli_vs_cvp(df):
    """
    Analisa distribuição CVLI vs CVP
    """
    print(f"\n[ANÁLISE TIPO DE CRIME]")
    
    tipo_dist = df['tipo_crime'].value_counts()
    print(f"  Distribuição:")
    for tipo, count in tipo_dist.items():
        pct = 100 * count / len(df)
        print(f"    {tipo}: {count} ({pct:.1f}%)")
    
    cvli_count = len(df[df['tipo_crime'] == 'cvli'])
    cvp_count = len(df[df['tipo_crime'] == 'cvp'])
    
    print(f"\n  CVLI (será usado para criticidade): {cvli_count}")
    print(f"  CVP (apenas contexto histórico): {cvp_count}")
    
    return cvli_count, cvp_count

def create_criticality_index(df_train, df_val=None):
    """
    Cria índice de criticidade BASEADO APENAS EM CVLI
    
    Métrica: Frequência de CVLI por AIS (Área de Integração de Segurança)
    """
    print(f"\n[CRITICIDADE INDEXING - CVLI ONLY]")
    
    # Filtrar APENAS CVLI para criticidade
    df_cvli = df_train[df_train['tipo_crime'] == 'cvli'].copy()
    
    # Agrupar por AIS (que é o campo disponível no JSON)
    criticidad_ais = df_cvli.groupby('ais').size().reset_index(name='count_cvli')
    criticidad_ais = criticidad_ais.sort_values('count_cvli', ascending=False)
    
    # Remover entradas sem AIS
    criticidad_ais = criticidad_ais[criticidad_ais['ais'].notna()]
    
    print(f"  Total CVLI para análise: {len(df_cvli)}")
    print(f"  AIS com CVLI: {len(criticidad_ais)}")
    
    # Normalizar para [0, 1]
    if len(criticidad_ais) > 0:
        max_count = criticidad_ais['count_cvli'].max()
        criticidad_ais['criticidad_score'] = criticidad_ais['count_cvli'] / max_count
    
    print(f"\n  Top 10 AIS por Criticidade CVLI:")
    print(criticidad_ais.head(10).to_string(index=False))
    
    return criticidad_ais

def analyze_faccao_territory(df_train, criticidad_df):
    """
    Relaciona crimes CVLI com territórios de facções
    """
    print(f"\n[ANÁLISE TERRITORIO - FACÇÕES E CVLI]")
    
    df_cvli = df_train[df_train['tipo_crime'] == 'cvli'].copy()
    
    if len(df_cvli) == 0:
        print("  [!] Nenhum crime CVLI para análise de facções")
        return pd.DataFrame()
    
    # Contar facções
    print(f"  Total CVLI: {len(df_cvli)}")
    print(f"  Valores únicos de faccao: {df_cvli['faccao'].nunique()}")
    
    # Agrupar por facção
    faccao_stats = df_cvli.groupby('faccao').agg({
        'id_ocorrencia': 'count'
    }).rename(columns={'id_ocorrencia': 'count_cvli'}).sort_values('count_cvli', ascending=False)
    
    print(f"\n  Crimes CVLI por Facção:")
    print(faccao_stats.to_string())
    
    return faccao_stats

def analyze_prisoes_raio_2025(df_val):
    """
    Analisa prisões RAIO em 2025 (arquivo ocorrencia_policial_operacional.json)
    """
    print(f"\n[ANÁLISE PRISÕES RAIO - 2025]")
    
    raio_path = config.DATA_RAW / "ocorrencia_policial_operacional.json"
    
    if not raio_path.exists():
        print(f"[!] Arquivo RAIO não encontrado: {raio_path}")
        return pd.DataFrame()
    
    try:
        with open(raio_path, 'r', encoding='utf-8-sig', errors='replace') as f:
            raio_data = json.load(f)
    except Exception as e:
        print(f"[X] Erro ao ler RAIO: {e}")
        return pd.DataFrame()
    
    if not isinstance(raio_data, list):
        print("[!] Formato inesperado do arquivo RAIO")
        return pd.DataFrame()
    
    df_raio = pd.DataFrame(raio_data)
    
    # Filtrar operações relevantes (tráfico, mandado de prisão, etc)
    relevant_keywords = ['TRÁFICO', 'PRISÃO', 'MANDADO', 'APREENSÃO']
    
    if 'Natureza' in df_raio.columns:
        df_raio['Natureza_upper'] = df_raio['Natureza'].astype(str).str.upper()
        df_raio = df_raio[df_raio['Natureza_upper'].str.contains('|'.join(relevant_keywords), na=False)]
    
    # Converter data
    if 'Data' in df_raio.columns:
        df_raio['Data'] = pd.to_datetime(df_raio['Data'], errors='coerce')
        df_raio = df_raio.dropna(subset=['Data'])
    
    print(f"  Total operações RAIO em 2025: {len(df_raio)}")
    
    # Tipos de operação
    if 'Natureza_upper' in df_raio.columns:
        print(f"\n  Tipos de Operações:")
        tipos = df_raio['Natureza_upper'].str.extract(r'(TRÁFICO|MANDADO|APREENSÃO|PRISÃO)', expand=False).value_counts()
        print(tipos.to_string())
    
    return df_raio

def save_datasets(df_train, df_val, criticidad_df, faccao_stats, df_raio):
    """
    Salva datasets processados para próximas etapas
    """
    output_dir = config.DATA_PROCESSED
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n[SALVANDO DATASETS]")
    
    # Treino CVLI
    df_train_cvli = df_train[df_train['tipo_crime'] == 'cvli'].copy()
    train_file = output_dir / "dataset_treino_cvli_2022_2024.parquet"
    df_train_cvli.to_parquet(train_file)
    print(f"  [+] Treino CVLI: {train_file}")
    
    # Validação CVLI
    df_val_cvli = df_val[df_val['tipo_crime'] == 'cvli'].copy()
    val_file = output_dir / "dataset_validacao_cvli_2025.parquet"
    df_val_cvli.to_parquet(val_file)
    print(f"  [+] Validação CVLI: {val_file}")
    
    # Criticidade
    crit_file = output_dir / "criticidad_index_cvli_only.csv"
    criticidad_df.to_csv(crit_file, index=False)
    print(f"  [+] Índice Criticidade: {crit_file}")
    
    # Facções
    faccao_file = output_dir / "faccao_territorio_stats.csv"
    faccao_stats.to_csv(faccao_file)
    print(f"  [+] Stats Facções: {faccao_file}")
    
    # RAIO
    raio_file = output_dir / "prisoes_raio_2025.parquet"
    df_raio.to_parquet(raio_file)
    print(f"  [+] Prisões RAIO 2025: {raio_file}")
    
    return {
        'train': train_file,
        'val': val_file,
        'criticidad': crit_file,
        'faccao': faccao_file,
        'raio': raio_file
    }

def main():
    print("=" * 60)
    print(" ETL - NOVO CRITÉRIO CVLI-CENTRIC")
    print("=" * 60)
    
    # 1. Carregar dados
    df = load_dados_status_ocorrencias()
    if df.empty:
        print("[X] Falha ao carregar dados")
        return
    
    # 2. Split temporal
    df_train, df_val = split_train_validation(df)
    
    # 3. Análise CVLI vs CVP
    analyze_cvli_vs_cvp(df)
    
    # 4. Criar índice de criticidade (CVLI ONLY)
    criticidad_df = create_criticality_index(df_train, df_val)
    
    # 5. Análise facções-territorios-CVLI
    faccao_stats = analyze_faccao_territory(df_train, criticidad_df)
    
    # 6. Análise prisões RAIO
    df_raio = analyze_prisoes_raio_2025(df_val)
    
    # 7. Salvar datasets
    datasets = save_datasets(df_train, df_val, criticidad_df, faccao_stats, df_raio)
    
    print("\n" + "=" * 60)
    print(" ETL CONCLUÍDO COM SUCESSO")
    print("=" * 60)
    
    return datasets

if __name__ == "__main__":
    main()
