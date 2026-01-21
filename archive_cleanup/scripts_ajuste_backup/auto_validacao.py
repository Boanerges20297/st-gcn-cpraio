#!/usr/bin/env python
"""
AUTO-VALIDAÇÃO - Aguarda fim do treino e executa validação
"""

import subprocess
import sys
import time
from pathlib import Path
import json
from datetime import datetime

def wait_for_training(check_interval=60, max_wait=3600):
    """Aguarda conclusão do treinamento"""
    model_file = Path("outputs/models/model_cvli_novo_criterio.pth")
    stats_file = Path("outputs/models/stats_cvli_novo_criterio.pt")
    
    print("=" * 70)
    print(" AUTO-VALIDAÇÃO - AGUARDANDO FIM DO TREINAMENTO")
    print("=" * 70)
    print(f"\nArquivos esperados:")
    print(f"  - {model_file}")
    print(f"  - {stats_file}")
    print(f"\nAguardando ({max_wait}s de timeout)...")
    
    elapsed = 0
    while elapsed < max_wait:
        if model_file.exists() and stats_file.exists():
            print(f"\n✅ Treinamento concluído em {elapsed}s!")
            return True
        
        print(f"\r⏳ {elapsed}s... {model_file.exists()} | {stats_file.exists()}", end="", flush=True)
        time.sleep(check_interval)
        elapsed += check_interval
    
    print(f"\n❌ Timeout após {max_wait}s")
    return False

def run_validation():
    """Executa script de validação"""
    print("\n" + "=" * 70)
    print(" INICIANDO VALIDAÇÃO")
    print("=" * 70)
    
    validation_script = Path("scripts_ajuste/04_validacao_prisoes_raio.py")
    
    if not validation_script.exists():
        print(f"[X] Script não encontrado: {validation_script}")
        return False
    
    print(f"\nExecutando: {validation_script}\n")
    
    try:
        result = subprocess.run(
            [sys.executable, str(validation_script)],
            cwd=str(Path.cwd()),
            capture_output=False
        )
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"[X] Erro ao executar validação: {e}")
        return False

def generate_summary():
    """Gera resumo de conclusão"""
    summary = {
        'timestamp': datetime.now().isoformat(),
        'status': 'PIPELINE COMPLETO',
        'etapas': [
            {'nome': 'Spatial Join', 'status': 'Concluído'},
            {'nome': 'ETL CVLI-Centric', 'status': 'Concluído'},
            {'nome': 'Graph Builder', 'status': 'Concluído'},
            {'nome': 'Treinamento ST-GCN', 'status': 'Concluído'},
            {'nome': 'Validação com RAIO', 'status': 'Concluído'}
        ],
        'documentacao': [
            'RESUMO_NOVO_PIPELINE_CVLI.md',
            'IMPLEMENTACAO_NOVO_CRITERIO_CVLI_COMPLETA.md'
        ]
    }
    
    output_file = Path('outputs/pipeline_summary.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n[+] Resumo salvo: {output_file}")
    
    return summary

def main():
    import os
    os.chdir(Path(__file__).parent.parent)
    
    # Aguardar treinamento
    if not wait_for_training():
        print("\n[!] Treinamento não foi concluído no tempo esperado")
        print("    Verifique: python scripts_ajuste/03_trainer_novo_criterio.py")
        return 1
    
    # Executar validação
    if not run_validation():
        print("\n[!] Validação apresentou erros")
        return 1
    
    # Gerar resumo
    generate_summary()
    
    print("\n" + "=" * 70)
    print(" ✅ PIPELINE COMPLETO COM SUCESSO!")
    print("=" * 70)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
