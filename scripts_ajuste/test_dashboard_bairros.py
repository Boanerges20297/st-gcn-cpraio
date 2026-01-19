#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Teste da API do Dashboard com Predições por Bairro
Verifica se load_risk_map retorna 138 bairros de Fortaleza
"""

import sys
import json
import pandas as pd
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from app import load_risk_map

def test_dashboard_bairros():
    print("\n" + "="*80)
    print("TESTE: API Dashboard com Predições por Bairro")
    print("="*80)
    
    # 1. Testar sem filtro de tipo_crime
    print(f"\n1. Carregando mapa de risco para CAPITAL (sem filtro de crime)")
    try:
        geojson = load_risk_map('CAPITAL')
        features = geojson.get('features', [])
        print(f"   ✓ Total de features: {len(features)}")
        
        # Verificar que são bairros (não locais)
        names = [f['properties'].get('name', 'UNKNOWN') for f in features]
        print(f"   Primeiros 5 bairros: {names[:5]}")
        
        # Verificar que tem risco
        risks = [f['properties'].get('risco') for f in features[:5]]
        print(f"   Primeiros 5 riscos: {risks}")
        
        if len(features) >= 130:
            print(f"   ✓ SUCESSO: Tem {len(features)} bairros (esperado ~138)")
        else:
            print(f"   ❌ ERRO: Só tem {len(features)} features")
            return False
            
    except Exception as e:
        print(f"   ❌ ERRO ao carregar: {e}")
        return False
    
    # 2. Testar com filtro CVP
    print(f"\n2. Carregando mapa com filtro CVP")
    try:
        geojson_cvp = load_risk_map('CAPITAL', tipo_crime='CVP')
        features_cvp = geojson_cvp.get('features', [])
        print(f"   ✓ Total de features com CVP: {len(features_cvp)}")
        
        # Contar quantos têm risco > 0 (áreas com CVP)
        areas_com_cvp = len([f for f in features_cvp if f['properties'].get('risco', 0) > 0])
        print(f"   Áreas com CVP (risco > 0): {areas_com_cvp}")
        
    except Exception as e:
        print(f"   ❌ ERRO ao carregar com filtro CVP: {e}")
        return False
    
    # 3. Testar com filtro CVLI
    print(f"\n3. Carregando mapa com filtro CVLI")
    try:
        geojson_cvli = load_risk_map('CAPITAL', tipo_crime='CVLI')
        features_cvli = geojson_cvli.get('features', [])
        print(f"   ✓ Total de features com CVLI: {len(features_cvli)}")
        
        # Contar quantos têm risco > 0 (áreas com CVLI)
        areas_com_cvli = len([f for f in features_cvli if f['properties'].get('risco', 0) > 0])
        print(f"   Áreas com CVLI (risco > 0): {areas_com_cvli}")
        
    except Exception as e:
        print(f"   ❌ ERRO ao carregar com filtro CVLI: {e}")
        return False
    
    # 4. Validar distribuição de risco
    print(f"\n4. Validar distribuição de risco (sem filtro)")
    risks = [f['properties'].get('risco') for f in features]
    risks = [r for r in risks if r is not None]
    print(f"   Min risco: {min(risks):.4f}")
    print(f"   Max risco: {max(risks):.4f}")
    print(f"   Média: {sum(risks)/len(risks):.4f}")
    
    print(f"\n" + "="*80)
    print(f"✓ TESTE COMPLETO: Dashboard usando predições por bairro")
    print(f"✓ {len(features)} bairros carregados para operações táticas")
    print("="*80 + "\n")
    
    return True

if __name__ == "__main__":
    success = test_dashboard_bairros()
    sys.exit(0 if success else 1)
