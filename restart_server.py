#!/usr/bin/env python3
"""Reiniciar servidor Flask"""
import subprocess
import sys
import time
import os
import signal

print("🔄 Reiniciando servidor Flask...")
print()

# Matar processos Python existentes (se houver)
try:
    import psutil
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if 'python' in proc.info['name'].lower():
                cmdline = ' '.join(proc.info['cmdline'] or [])
                if 'app.py' in cmdline or 'src/app.py' in cmdline:
                    print(f"Matando processo: {proc.info['pid']}")
                    proc.kill()
                    time.sleep(0.5)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
except:
    print("⚠️ psutil não disponível, pulando kill de processos")

print("✅ Processos antigos encerrados")
print()

# Aguardar um pouco
time.sleep(1)

# Iniciar novo servidor
print("🚀 Iniciando novo servidor Flask...")
print()

try:
    # Usar a venv
    venv_python = r".\.venv\Scripts\python.exe"
    
    # Comando para iniciar o servidor
    cmd = [venv_python, r"src\app.py"]
    
    # Executar em nova janela
    os.system(f'start cmd /k "{" ".join(cmd)}"')
    
    print("✅ Servidor iniciado em nova janela")
    print("   URL: http://localhost:5000/dashboard-estrategico")
    
except Exception as e:
    print(f"❌ Erro ao iniciar servidor: {e}")
    sys.exit(1)
