"""
ORQUESTRAÇÃO - NOVO PIPELINE CVLI-CENTRIC
=========================================
Executa os 4 estágios em sequência:
1. ETL (09_etl_novo_criterio.py)
2. Graph Builder (02_graph_builder_novo.py)
3. Trainer (03_trainer_novo_criterio.py)
4. Validação (04_validacao_prisoes_raio.py)
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime

def run_stage(stage_num, script_name, description):
    """Executa um estágio do pipeline"""
    print("\n" + "=" * 70)
    print(f" ESTÁGIO {stage_num}: {description}")
    print("=" * 70)
    
    script_path = Path(__file__).parent / script_name
    
    if not script_path.exists():
        print(f"[X] Script não encontrado: {script_path}")
        return False
    
    print(f"[*] Executando: {script_path}")
    print(f"[*] Timestamp: {datetime.now().isoformat()}\n")
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(Path(__file__).parent.parent),
            capture_output=False,
            text=True
        )
        
        if result.returncode != 0:
            print(f"[X] Erro ao executar {script_name}")
            return False
        
        print(f"\n[V] Estágio {stage_num} concluído com sucesso")
        return True
        
    except Exception as e:
        print(f"[X] Exceção ao executar {script_name}: {e}")
        return False

def main():
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " PIPELINE DE RETRAINAMENTO - NOVO CRITÉRIO CVLI-CENTRIC ".center(68) + "║")
    print("╚" + "=" * 68 + "╝")
    
    print(f"\nInício: {datetime.now().isoformat()}")
    
    stages = [
        (0, "00_spatial_join_enriquecimento.py", "Spatial Join - Mapear lat/lng aos bairros"),
        (1, "01_etl_novo_criterio.py", "ETL - Carregamento e split CVLI/CVP"),
        (2, "02_graph_builder_novo.py", "Graph Builder - Construção de grafos"),
        (3, "03_trainer_novo_criterio.py", "Trainer - Treinamento ST-GCN"),
        (4, "04_validacao_prisoes_raio.py", "Validação - Análise RAIO e impacto"),
    ]
    
    results = {}
    
    for stage_num, script, desc in stages:
        success = run_stage(stage_num, script, desc)
        results[stage_num] = (desc, success)
        
        if not success:
            print(f"\n[!] Pipeline interrompido no estágio {stage_num}")
            print(f"[!] Erro: Falha ao executar {script}")
            break
    
    # Resumo final
    print("\n" + "=" * 70)
    print(" RESUMO DO PIPELINE")
    print("=" * 70)
    
    for stage_num, (desc, success) in results.items():
        status = "[✓]" if success else "[✗]"
        print(f"{status} Estágio {stage_num}: {desc}")
    
    all_success = all(success for _, success in results.values())
    
    print("\n" + "=" * 70)
    if all_success:
        print(" ✓ PIPELINE CONCLUÍDO COM SUCESSO")
    else:
        print(" ✗ PIPELINE COM ERROS")
    print("=" * 70)
    
    print(f"\nFim: {datetime.now().isoformat()}\n")
    
    return 0 if all_success else 1

if __name__ == "__main__":
    sys.exit(main())
