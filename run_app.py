#!/usr/bin/env python
"""
Script para iniciar a aplicação Flask sem loop de restart.
"""
import os
import sys
from pathlib import Path

# Adiciona o diretório raiz ao path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

# Variáveis de ambiente
os.environ['FLASK_APP'] = 'src/app.py'
os.environ['FLASK_ENV'] = 'development'
os.environ['FLASK_DEBUG'] = '0'  # Desabilita debug mode para evitar restart loop

if __name__ == '__main__':
    from src.app import app
    
    print("\n" + "="*60)
    print("🚀 INICIANDO DASHBOARD ESTRATÉGICO - CPRAIO")
    print("="*60)
    print("\n✓ Debug Mode: DESABILITADO (evita loop de restart)")
    print("✓ Servidor: http://127.0.0.1:5000")
    print("✓ Para reiniciar: Ctrl+C e execute novamente")
    print("\n" + "="*60 + "\n")
    
    # Executa sem reloader para evitar o loop infinito
    app.run(
        host='127.0.0.1',
        port=5000,
        debug=False,
        use_reloader=False,
        use_debugger=False,
        threaded=True
    )
