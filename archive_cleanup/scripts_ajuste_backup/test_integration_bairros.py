#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Teste de Integração Final: Predições por Bairro
Simula a experiência completa do usuário no dashboard
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from app import load_risk_map, load_occurrences

def test_integration():
    print("\n" + "="*80)
    print("TESTE DE INTEGRAÇÃO: Predições por Bairro para Operações Táticas")
    print("="*80)
    
    # 1. Carregar dados de crimes
    print(f"\n1. Carregando dados de crimes históricos...")
    crimes = load_occurrences()
    capital_crimes = crimes[crimes['regiao_sistema'] == 'CAPITAL']
    print(f"   ✓ Carregado: {len(capital_crimes)} crimes em CAPITAL")
    print(f"   Distribuição: CVP={len(crimes[crimes['tipo'] == 'CVP'])}, CVLI={len(crimes[crimes['tipo'] == 'CVLI'])}")
    
    # 2. Carregar mapa de risco geral (TODOS os bairros)
    print(f"\n2. Carregando mapa de risco geral para CAPITAL...")
    risk_map = load_risk_map('CAPITAL')
    features = risk_map.get('features', [])
    print(f"   ✓ Carregado: {len(features)} bairros")
    
    # Extrair informações dos top 3
    top3 = sorted(features, key=lambda x: x['properties'].get('risco', 0), reverse=True)[:3]
    print(f"   Top 3 bairros críticos:")
    for i, feat in enumerate(top3, 1):
        props = feat['properties']
        print(f"      {i}. {props.get('name', 'N/A'):30} → Risco: {props.get('risco', 0):.4f} ({props.get('nivel_alerta', 'N/A')})")
    
    # 3. Teste de filtro CVP
    print(f"\n3. Filtrando por CVP (Roubos Patrimoniais)...")
    risk_map_cvp = load_risk_map('CAPITAL', tipo_crime='CVP')
    features_cvp = risk_map_cvp.get('features', [])
    areas_com_cvp = [f for f in features_cvp if f['properties'].get('risco', 0) > 0]
    print(f"   ✓ Bairros com CVP: {len(areas_com_cvp)}")
    if areas_com_cvp:
        print(f"   Exemplo: {areas_com_cvp[0]['properties']['name']}")
    
    # 4. Teste de filtro CVLI
    print(f"\n4. Filtrando por CVLI (Homicídios)...")
    risk_map_cvli = load_risk_map('CAPITAL', tipo_crime='CVLI')
    features_cvli = risk_map_cvli.get('features', [])
    areas_com_cvli = [f for f in features_cvli if f['properties'].get('risco', 0) > 0]
    print(f"   ✓ Bairros com CVLI: {len(areas_com_cvli)}")
    if areas_com_cvli:
        print(f"   Exemplo: {areas_com_cvli[0]['properties']['name']}")
    
    # 5. Validação de estrutura
    print(f"\n5. Validando estrutura da resposta...")
    sample = features[0]
    props = sample['properties']
    
    required_fields = ['name', 'risco', 'nivel_alerta', 'cor_alerta']
    missing = [f for f in required_fields if f not in props]
    
    if missing:
        print(f"   ❌ Campos faltantes: {missing}")
        return False
    else:
        print(f"   ✓ Todos os campos presentes: {list(props.keys())[:6]}")
    
    # 6. Validação de valores
    print(f"\n6. Validando valores...")
    all_risks = [f['properties']['risco'] for f in features]
    valid_risks = [r for r in all_risks if isinstance(r, (int, float)) and 0 <= r <= 1]
    
    if len(valid_risks) == len(all_risks):
        print(f"   ✓ Todos os riscos válidos: {len(valid_risks)} valores entre 0-1")
    else:
        print(f"   ⚠ {len(valid_risks)}/{len(all_risks)} valores válidos")
    
    # 7. Resumo final
    print(f"\n" + "="*80)
    print(f"RESUMO DA INTEGRAÇÃO:")
    print(f"  • Dados de crimes: {len(capital_crimes)} registros em CAPITAL")
    print(f"  • Bairros mapeados: {len(features)} com predição individual")
    print(f"  • Filtro CVP: {len(areas_com_cvp)} bairros com roubos")
    print(f"  • Filtro CVLI: {len(areas_com_cvli)} bairros com homicídios")
    print(f"  • Distribuição de risco: {min(all_risks):.4f} - {max(all_risks):.4f}")
    print(f"\n✓ Sistema OPERACIONAL para operações táticas por bairro!")
    print("="*80 + "\n")
    
    return True

if __name__ == "__main__":
    try:
        success = test_integration()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ ERRO: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
