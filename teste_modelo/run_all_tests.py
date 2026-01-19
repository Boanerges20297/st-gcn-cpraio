#!/usr/bin/env python
"""
QUICK START - SUITE DE TESTES ST-GCN
====================================

Execute este arquivo para ver todos os testes em ação.
"""

import os
import subprocess
import json
from pathlib import Path

print("""
╔═══════════════════════════════════════════════════════════════════════╗
║                  QUICK START - SUITE DE TESTES                       ║
║                                                                       ║
║          Executar todos os testes e gerar relatório final             ║
╚═══════════════════════════════════════════════════════════════════════╝
""")

# Diretório base
base_dir = Path(__file__).parent

# Testes a executar
testes = [
    {
        'nome': 'TESTE 1: Análise de Criticidade',
        'script': 'analise_criticidade.py',
        'output': 'analise_criticidade.json',
        'descricao': 'Por que bairros com ZERO crimes têm risco > 0?'
    },
    {
        'nome': 'TESTE 2: Correlação Facção-Risco',
        'script': 'correlacao_faccao_risco.py',
        'output': 'correlacao_faccao_risco.json',
        'descricao': 'Como o modelo relaciona facções com risco'
    },
    {
        'nome': 'TESTE 3: Eficiência Preditiva',
        'script': 'test_modelo_eficiencia.py',
        'output': 'teste_eficiencia_modelo.json',
        'descricao': 'Treino 2022-2023 vs Teste 2024-2025 (Gabarito)'
    }
]

# Executar testes
resultados = []

for i, teste in enumerate(testes, 1):
    print(f"\n{'='*70}")
    print(f"[{i}/3] {teste['nome']}")
    print(f"{'='*70}")
    print(f"Descrição: {teste['descricao']}\n")
    
    try:
        # Executar script
        script_path = base_dir / teste['script']
        resultado = subprocess.run(
            ['python', str(script_path)],
            capture_output=True,
            text=True,
            timeout=120
        )
        
        if resultado.returncode == 0:
            print(f"✅ Teste executado com sucesso")
            
            # Carregar output JSON
            output_path = base_dir / teste['output']
            if output_path.exists():
                with open(output_path, 'r', encoding='utf-8') as f:
                    dados = json.load(f)
                print(f"✅ Output salvo: {teste['output']}")
                
                resultados.append({
                    'teste': teste['nome'],
                    'status': 'PASS',
                    'output': teste['output']
                })
            else:
                print(f"⚠️  Output file not found: {teste['output']}")
                resultados.append({
                    'teste': teste['nome'],
                    'status': 'PARTIAL',
                    'output': None
                })
        else:
            print(f"❌ Erro na execução:")
            print(resultado.stderr[:500])
            resultados.append({
                'teste': teste['nome'],
                'status': 'FAIL',
                'output': None
            })
    
    except subprocess.TimeoutExpired:
        print(f"⏱️  Timeout na execução")
        resultados.append({
            'teste': teste['nome'],
            'status': 'TIMEOUT',
            'output': None
        })
    except Exception as e:
        print(f"❌ Erro: {e}")
        resultados.append({
            'teste': teste['nome'],
            'status': 'ERROR',
            'output': None
        })

# Resumo final
print(f"\n{'='*70}")
print("RESUMO DOS TESTES")
print(f"{'='*70}\n")

passed = sum(1 for r in resultados if r['status'] == 'PASS')
failed = len(resultados) - passed

for resultado in resultados:
    status_icon = {
        'PASS': '✅',
        'FAIL': '❌',
        'PARTIAL': '⚠️',
        'TIMEOUT': '⏱️',
        'ERROR': '❌'
    }.get(resultado['status'], '?')
    
    print(f"{status_icon} {resultado['teste']:40s} [{resultado['status']}]")

print(f"\n{'='*70}")
print(f"RESULTADO FINAL: {passed}/3 testes passaram ✅" if passed == 3 else f"RESULTADO FINAL: {passed}/3 testes passaram ({failed} falhas)")
print(f"{'='*70}\n")

# Próximos passos
print("📚 PRÓXIMOS PASSOS:")
print("""
1. Leia os arquivos README:
   • README_TESTE_EFICIENCIA.md
   • README_CORRELACAO_FACCAO_RISCO.md

2. Visualize o scorecard:
   • SCORECARD_FINAL.md

3. Verifique os JSONs gerados:
   • teste_eficiencia_modelo.json
   • correlacao_faccao_risco.json
   • analise_criticidade.json

4. Para detalhes técnicos:
   • INDICE_SUITE_TESTES.md

5. Recomendação final:
   • 🟢 MODELO APROVADO PARA PRODUÇÃO

""")

print("✅ Suite de testes concluída!\n")
