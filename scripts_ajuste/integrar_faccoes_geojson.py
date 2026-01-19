#!/usr/bin/env python3
"""
Script para integrar dados de facções com geolocalização (GeoJSON) do SIGERAIO.
Baixa os arquivos de facções fragmentados e integra aos dados consolidados.

Uso: python scripts_ajuste/integrar_faccoes_geojson.py
"""

import os
import json
import requests
import pandas as pd
import geopandas as gpd
from pathlib import Path
from datetime import datetime

# Configurações
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "graph"
CONSOLIDATED_FILE = BASE_DIR / "data" / "processed" / "base_consolidada.parquet"

# URLs dos dados de facções do SIGERAIO (GitHub)
GITHUB_FACCOES_BASE = "https://raw.githubusercontent.com/JeffFelipe/sigeraio/4b5db756bf7587c5f8315d7b5d57d92e22d827d7/static/data/geojson/faccoes"

FACCOES_LISTA = [
    "COMANDO VERMELHO",
    "PRIMEIRO COMANDO DA CAPITAL",
    "TERCEIRO COMANDO PURO",
    "MASSA",
    "OKAIDA",
    "GUARDIOES DO ESTADO",
]

def normalizar_nome_arquivo(nome):
    """Normaliza nome da facção para nome de arquivo"""
    return nome.upper().replace(" ", "_").replace("Á", "A").replace("Ã", "A")

def baixar_geojson_faccoes():
    """Baixa arquivos GeoJSON de facções do GitHub"""
    print("[*] Iniciando download de GeoJSON de facções...")
    
    faccoes_geojson = {}
    
    for faccao in FACCOES_LISTA:
        nome_arquivo = normalizar_nome_arquivo(faccao)
        url = f"{GITHUB_FACCOES_BASE}/{nome_arquivo}.geojson"
        
        try:
            print(f"   📥 Baixando {faccao}...")
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                geojson_data = response.json()
                faccoes_geojson[faccao] = geojson_data
                
                # Salvar localmente
                local_path = DATA_DIR / f"faccao_{nome_arquivo}.geojson"
                with open(local_path, 'w', encoding='utf-8') as f:
                    json.dump(geojson_data, f, ensure_ascii=False, indent=2)
                
                # Contar features
                num_features = len(geojson_data.get('features', []))
                print(f"   ✓ {faccao}: {num_features} polígonos salvos")
            else:
                print(f"   ❌ Erro ao baixar {faccao}: HTTP {response.status_code}")
        
        except requests.exceptions.RequestException as e:
            print(f"   ❌ Erro ao conectar para {faccao}: {e}")
        except json.JSONDecodeError as e:
            print(f"   ❌ Erro ao parsear JSON de {faccao}: {e}")
        except Exception as e:
            print(f"   ❌ Erro desconhecido para {faccao}: {e}")
    
    return faccoes_geojson

def enriquecer_dados_consolidados(faccoes_geojson):
    """Enriquece dados consolidados com informações de localização de facções"""
    print("\n[*] Enriquecendo dados consolidados com localização de facções...")
    
    if not CONSOLIDATED_FILE.exists():
        print(f"❌ Arquivo consolidado não encontrado: {CONSOLIDATED_FILE}")
        return False
    
    try:
        df = pd.read_parquet(CONSOLIDATED_FILE)
        print(f"   ✓ Carregado: {len(df)} registros")
        
        # Adicionar coluna de facção com localização se não existir
        if 'faccao_localizada' not in df.columns:
            df['faccao_localizada'] = 'DESCONHECIDA'
        
        # Para cada facção, marcar crimes dentro de seus polígonos
        for faccao_nome, geojson in faccoes_geojson.items():
            if 'features' not in geojson or len(geojson['features']) == 0:
                print(f"   ⚠️ Nenhuma feature em {faccao_nome}")
                continue
            
            print(f"   🔍 Processando {faccao_nome}...")
            
            # Converter GeoJSON para GeoDataFrame
            gdf_faccao = gpd.GeoDataFrame.from_features(geojson['features'])
            
            # Converter crimes com coordenadas para GeoDataFrame
            if 'latitude' in df.columns and 'longitude' in df.columns:
                # Criar pontos de crimes
                df_geo = df[(df['latitude'].notna()) & (df['longitude'].notna())].copy()
                geometry = gpd.points_from_xy(df_geo['longitude'], df_geo['latitude'])
                gdf_crimes = gpd.GeoDataFrame(df_geo, geometry=geometry, crs='EPSG:4326')
                
                # Intersecção espacial (quais crimes estão dentro da facção)
                crimes_na_faccao = gpd.sjoin(gdf_crimes, gdf_faccao, how='inner', predicate='within')
                
                # Marcar esses crimes
                for idx in crimes_na_faccao.index:
                    if idx in df.index:
                        df.loc[idx, 'faccao_localizada'] = faccao_nome
                
                print(f"      → {len(crimes_na_faccao)} crimes dentro de {faccao_nome}")
            else:
                print(f"      ⚠️ Coordenadas não encontradas para {faccao_nome}")
        
        # Salvar arquivo enriquecido
        df.to_parquet(CONSOLIDATED_FILE)
        print(f"\n✓ Dados consolidados atualizados com informações de facções")
        
        # Estatísticas
        print("\n📊 ESTATÍSTICAS DE LOCALIZAÇÃO DE FACÇÕES:")
        faccoes_encontradas = df['faccao_localizada'].value_counts()
        for faccao, count in faccoes_encontradas.items():
            print(f"   • {faccao}: {count} crimes")
        
        return True
    
    except Exception as e:
        print(f"❌ Erro ao enriquecer dados: {e}")
        return False

def criar_mapa_territorial_facções():
    """Cria arquivo consolidado com todos os territórios de facções"""
    print("\n[*] Criando mapa territorial consolidado...")
    
    try:
        # Combinar todos os GeoJSON de facções
        todas_features = []
        
        for arquivo in DATA_DIR.glob("faccao_*.geojson"):
            with open(arquivo, 'r', encoding='utf-8') as f:
                geojson = json.load(f)
                features = geojson.get('features', [])
                
                # Adicionar informação de facção
                for feature in features:
                    nome_faccao = arquivo.stem.replace('faccao_', '').replace('_', ' ').upper()
                    feature['properties']['faccao'] = nome_faccao
                    todas_features.append(feature)
        
        if todas_features:
            consolidado = {
                'type': 'FeatureCollection',
                'features': todas_features
            }
            
            # Salvar
            output_path = DATA_DIR / "territorio_faccoes_consolidado.geojson"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(consolidado, f, ensure_ascii=False, indent=2)
            
            print(f"✓ Mapa territorial consolidado criado: {output_path}")
            print(f"  → Total de {len(todas_features)} polígonos de facções")
        else:
            print("⚠️ Nenhuma feature encontrada para consolidar")
    
    except Exception as e:
        print(f"❌ Erro ao criar mapa consolidado: {e}")

def gerar_relatorio():
    """Gera relatório dos arquivos criados"""
    print("\n" + "="*60)
    print("📋 RELATÓRIO FINAL - INTEGRAÇÃO DE FACÇÕES")
    print("="*60)
    
    print("\n✓ Arquivos GeoJSON de Facções criados em /data/graph/:")
    for arquivo in sorted(DATA_DIR.glob("faccao_*.geojson")):
        size_kb = arquivo.stat().st_size / 1024
        print(f"   • {arquivo.name} ({size_kb:.1f} KB)")
    
    print("\n✓ Mapa Consolidado:")
    mapa_consol = DATA_DIR / "territorio_faccoes_consolidado.geojson"
    if mapa_consol.exists():
        size_mb = mapa_consol.stat().st_size / (1024 * 1024)
        print(f"   • {mapa_consol.name} ({size_mb:.2f} MB)")
    
    print("\n✓ Dados Consolidados Enriquecidos:")
    if CONSOLIDATED_FILE.exists():
        df = pd.read_parquet(CONSOLIDATED_FILE)
        print(f"   • {CONSOLIDATED_FILE.name}")
        print(f"     - Total de registros: {len(df):,}")
        if 'faccao_localizada' in df.columns:
            print(f"     - Crimes com facção localizada: {(df['faccao_localizada'] != 'DESCONHECIDA').sum():,}")
    
    print("\n" + "="*60)
    print("🎯 PRÓXIMOS PASSOS:")
    print("="*60)
    print("1. Dashboard vai usar /data/graph/territorio_faccoes_consolidado.geojson")
    print("2. Cada facção tem sua localização exata mapeada")
    print("3. Crimes linkados a facções geograficamente corretos")
    print("4. Gestão pode analisar atuação por localização, não ranking")
    print("="*60 + "\n")

def main():
    print("\n" + "="*60)
    print("🏴 INTEGRAÇÃO DE FACÇÕES COM GEOLOCALIZAÇÃO")
    print("="*60 + "\n")
    
    # Garantir diretório
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Etapa 1: Baixar GeoJSON
    faccoes_geojson = baixar_geojson_faccoes()
    
    if not faccoes_geojson:
        print("\n❌ Nenhum arquivo de facção foi baixado")
        return False
    
    # Etapa 2: Enriquecer dados
    sucesso = enriquecer_dados_consolidados(faccoes_geojson)
    if not sucesso:
        print("\n⚠️ Continuando mesmo com erro no enriquecimento...")
    
    # Etapa 3: Criar mapa territorial consolidado
    criar_mapa_territorial_facções()
    
    # Etapa 4: Relatório final
    gerar_relatorio()
    
    return True

if __name__ == '__main__':
    sucesso = main()
    exit(0 if sucesso else 1)
