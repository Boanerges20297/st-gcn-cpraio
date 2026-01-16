"""
VERIFICADOR DE DEPENDÊNCIAS E CONFIGURAÇÃO
==========================================
Script que verifica e instala dependências necessárias para diagnosticar chaves API.
"""

import subprocess
import sys
import importlib
from pathlib import Path

# Dependências obrigatórias
DEPENDENCIES = {
    'google.generativeai': 'google-generativeai',
    'requests': 'requests',
    'dotenv': 'python-dotenv',
}

BASE_DIR = Path(__file__).resolve().parent.parent.parent


def verificar_dependencia(module_name: str, package_name: str) -> bool:
    """Verifica se uma dependência está instalada."""
    try:
        importlib.import_module(module_name)
        print(f"✓ {package_name} instalado")
        return True
    except ImportError:
        print(f"✗ {package_name} NÃO INSTALADO")
        return False


def instalar_dependencia(package_name: str) -> bool:
    """Instala uma dependência via pip."""
    print(f"\n[*] Instalando {package_name}...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name, "-q"])
        print(f"✓ {package_name} instalado com sucesso!")
        return True
    except subprocess.CalledProcessError:
        print(f"✗ Erro ao instalar {package_name}")
        return False


def main():
    print("=" * 70)
    print("VERIFICADOR DE DEPENDÊNCIAS")
    print("=" * 70)

    # Verifica todas as dependências
    print("\n[*] Verificando dependências necessárias...\n")

    faltantes = []
    for module, package in DEPENDENCIES.items():
        if not verificar_dependencia(module, package):
            faltantes.append(package)

    # Se houver faltantes, instala
    if faltantes:
        print(f"\n[!] {len(faltantes)} dependência(s) faltando. Instalando...\n")

        for package in faltantes:
            instalar_dependencia(package)

        print("\n[+] Todas as dependências foram instaladas!")
    else:
        print("\n[+] Todas as dependências já estão instaladas!")

    print("\n" + "=" * 70)
    print("PRÓXIMO PASSO: Execute o diagnóstico")
    print("=" * 70)
    print("\npython src/scripts/verify_api_keys.py\n")


if __name__ == "__main__":
    main()
