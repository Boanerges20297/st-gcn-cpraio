#!/usr/bin/env python3
"""
Script para consolidar territórios de facções (CAPITAL, INTERIOR, RMF) 
em um único GeoJSON por facção com padrão de nomenclatura correto.

Uso: python scripts_ajuste/consolidar_territorios_faccoes.py
"""

import json
import pandas as pd
import geopandas as gpd
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent.parent
GRAPH_DIR = BASE_DIR / "data" / "graph"

# Mapeamento de facções: nome_original → sigla
FACCOES_MAP = {
    'COMANDO VERMELHO': 'cv',
    'PRIMEIRO COMANDO DA CAPITAL': 'pcc',
    'TERCEIRO COMANDO PURO': 'tcp',
    'MASSA': 'massa',
    'OKAIDA': 'okaida',
    'GUARDIOES DO ESTADO': 'gde',
    'COMUNIDADES EM DISPUTA': 'disputa',
}

# Padrões de arquivos por região
REGIOES = ['CAPITAL', 'INTERIOR', 'RMF']

print("=" * 70)
print("🗺️  CONSOLIDAÇÃO DE TERRITÓRIOS DE FACÇÕES")
print("=" * 70)
print()

for faccao_nome, faccao_sigla in FACCOES_MAP.items():
    print(f"[*] Processando {faccao_nome} ({faccao_sigla})...")
    
    # Procurar arquivo principal (nome completo)
    arquivo_principal = GRAPH_DIR / f"{faccao_nome}.geojson"
    
    if arquivo_principal.exists():
        print(f"   ✓ Arquivo principal encontrado: {arquivo_principal.name}")
        
        # Carregar e copiar diretamente
        try:
            gdf = gpd.read_file(arquivo_principal)
            output_file = GRAPH_DIR / f"faccao_{faccao_sigla}.geojson"
            gdf.to_file(output_file, driver='GeoJSON')
            print(f"   ✅ Consolidado: {len(gdf)} polígonos → {output_file.name}")
        except Exception as e:
            print(f"   ❌ Erro ao processar: {e}")
    else:
        # Se não existe arquivo principal, tentar mesclar por regiões
        print(f"   ℹ️  Arquivo principal não encontrado, procurando por regiões...")
        
        gdfs_regioes = []
        
        for regiao in REGIOES:
            arquivo_regiao = GRAPH_DIR / f"territorio_{faccao_sigla}_{regiao.lower()}.geojson"
            
            if arquivo_regiao.exists():
                try:
                    gdf = gpd.read_file(arquivo_regiao)
                    gdf['regiao'] = regiao
                    gdfs_regioes.append(gdf)
                    print(f"      ✓ {regiao}: {len(gdf)} polígonos")
                except Exception as e:
                    print(f"      ❌ {regiao}: Erro - {e}")
        
        # Mesclar todas as regiões
        if gdfs_regioes:
            try:
                gdf_consolidado = gpd.GeoDataFrame(
                    pd.concat(gdfs_regioes, ignore_index=True),
                    crs=gdfs_regioes[0].crs
                )
                
                output_file = GRAPH_DIR / f"faccao_{faccao_sigla}.geojson"
                gdf_consolidado.to_file(output_file, driver='GeoJSON')
                
                print(f"   ✅ Consolidado ({len(gdfs_regioes)} regiões): {len(gdf_consolidado)} polígonos → {output_file.name}")
            except Exception as e:
                print(f"   ❌ Erro ao consolidar regiões: {e}")
        else:
            print(f"   ⚠️  Nenhum arquivo de região encontrado para {faccao_nome}")
    
    print()

# ============================================================================
# RELATÓRIO FINAL
# ============================================================================
print()
print("=" * 70)
print("✅ CONSOLIDAÇÃO CONCLUÍDA")
print("=" * 70)
print()
print("Arquivos gerados para dashboard:")

for faccao_nome, faccao_sigla in FACCOES_MAP.items():
    output_file = GRAPH_DIR / f"faccao_{faccao_sigla}.geojson"
    if output_file.exists():
        size_mb = output_file.stat().st_size / 1024 / 1024
        print(f"  ✅ faccao_{faccao_sigla}.geojson ({size_mb:.2f} MB)")
    else:
        print(f"  ❌ faccao_{faccao_sigla}.geojson (não encontrado)")

print()
print("🚀 Próximos passos:")
print("  1. Reiniciar dashboard: python src/app.py")
print("  2. Acessar: http://localhost:5000/dashboard-estrategico")
print("  3. Mapa mostrará territórios de todas as facções")
print()
print("=" * 70)
