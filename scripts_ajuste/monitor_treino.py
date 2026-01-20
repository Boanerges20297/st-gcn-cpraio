#!/usr/bin/env python
"""
MONITOR - Acompanhar progresso do treinamento
"""

import time
from pathlib import Path
import os

MODEL_FILE = Path("outputs/models/model_cvli_novo_criterio.pth")
STATS_FILE = Path("outputs/models/stats_cvli_novo_criterio.pt")

def get_file_size_mb(file_path):
    """Retorna tamanho em MB"""
    if file_path.exists():
        return os.path.getsize(file_path) / (1024 * 1024)
    return 0

def monitor():
    print("\n" + "="*60)
    print(" MONITOR DE TREINAMENTO")
    print("="*60)
    
    print(f"\nModel File: {MODEL_FILE}")
    print(f"  Existe: {MODEL_FILE.exists()}")
    print(f"  Tamanho: {get_file_size_mb(MODEL_FILE):.2f} MB")
    
    print(f"\nStats File: {STATS_FILE}")
    print(f"  Existe: {STATS_FILE.exists()}")
    print(f"  Tamanho: {get_file_size_mb(STATS_FILE):.2f} MB")
    
    if MODEL_FILE.exists() and STATS_FILE.exists():
        print("\n✅ TREINAMENTO CONCLUÍDO COM SUCESSO!")
        print("\nPróximo passo:")
        print("  python scripts_ajuste/04_validacao_prisoes_raio.py")
    else:
        print("\n⏳ Treinamento ainda em progresso...")
        print("   Próximo check em ~60 segundos")

if __name__ == "__main__":
    os.chdir(Path(__file__).parent.parent)
    monitor()
