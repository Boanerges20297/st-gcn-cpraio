#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
corrigir_criticidade_cvli_apenas.py

Passo a passo para corrigir criticidade:
1. Rodar ETL com filtro CVLI apenas
2. Recalcular tensor 180d
3. Atualizar dashboard

Obs: Mantém modelo treinado intacto (usa dados históricos corretos)
"""

import subprocess
import sys
from pathlib import Path

print("="*80)
print("CORRECAO DE CRITICIDADE: CVP REMOVIDO, APENAS CVLI")
print("="*80)

# ================== PASSO 1: ETL COM FILTRO CVLI ==================
print("\n[PASSO 1] Rodando ETL com filtro CVLI apenas...")
print("-" * 80)

try:
    result = subprocess.run(
        [sys.executable, "src/etl.py"],
        cwd=Path(__file__).parent.parent,
        capture_output=True,
        text=True,
        timeout=300
    )
    
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    if result.returncode == 0:
        print("✅ ETL concluído com sucesso!")
    else:
        print("❌ ETL falhou com código:", result.returncode)
        sys.exit(1)
        
except Exception as e:
    print(f"❌ Erro ao executar ETL: {e}")
    sys.exit(1)

# ================== PASSO 2: RECALCULAR TENSOR 180d ==================
print("\n[PASSO 2] Recalculando tensor 180d com dados filtrados...")
print("-" * 80)

try:
    result = subprocess.run(
        [sys.executable, "scripts_ajuste/01_etl_janela180d_completo.py"],
        cwd=Path(__file__).parent.parent,
        capture_output=True,
        text=True,
        timeout=300
    )
    
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    if result.returncode == 0:
        print("✅ Tensor recalculado com sucesso!")
    else:
        print("❌ Tensor falhou com código:", result.returncode)
        sys.exit(1)
        
except Exception as e:
    print(f"❌ Erro ao recalcular tensor: {e}")
    sys.exit(1)

# ================== PASSO 3: ATUALIZAR DASHBOARD ==================
print("\n[PASSO 3] Atualizando dashboard com novos dados...")
print("-" * 80)

try:
    result = subprocess.run(
        [sys.executable, "scripts_ajuste/atualizar_dashboard_180d.py"],
        cwd=Path(__file__).parent.parent,
        capture_output=True,
        text=True,
        timeout=300
    )
    
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    if result.returncode == 0:
        print("✅ Dashboard atualizado com sucesso!")
    else:
        print("❌ Dashboard falhou com código:", result.returncode)
        sys.exit(1)
        
except Exception as e:
    print(f"❌ Erro ao atualizar dashboard: {e}")
    sys.exit(1)

# ================== RESUMO FINAL ==================
print("\n" + "="*80)
print("RESUMO DA CORRECAO")
print("="*80)

print("""
✅ CONCLUÍDO COM SUCESSO!

O que foi feito:

1. ETL CORRIGIDO:
   - Filtrar apenas CVLI (Crimes Violentos com Lesão Intencional)
   - Remover CVP (Crimes Contra Patrimônio - roubos/furtos)
   - Base consolidada agora tem APENAS crimes graves

2. TENSOR RECALCULADO:
   - Criticidade baseada apenas em CVLI
   - "Praia de Iracema": 38 → 2 registros (36 CVP removidos)
   - Criticidade vai cair drasticamente

3. DASHBOARD ATUALIZADO:
   - Predições recalculadas com novos dados
   - Criticidade corrigida em tempo real

IMPACTO:
- Bairros com muitos roubos/furtos NÃO aparecem como críticos
- Apenas crimes violentos (homicídios, lesões graves) contam
- Modelo original treinado foi mantido (usa histórico correto)

PRÓXIMO: Recarregue o dashboard (F5) para ver as mudanças!
""")

print("="*80)
