#!/usr/bin/env python
# -*- coding: utf-8 -*-
import urllib.request
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

try:
    response = urllib.request.urlopen('http://localhost:5000/api/recomendacoes_operacionais', timeout=5)
    print("API chamada com sucesso - veja o terminal do servidor para DEBUG")
except Exception as e:
    print(f"Erro: {e}")
