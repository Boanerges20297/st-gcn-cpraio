#!/usr/bin/env python3
"""
SCRIPT DE VALIDAÇÃO DA IMPLANTAÇÃO
Verifica se tudo está funcionando corretamente
"""

import os
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent

def check_files():
    """Verifica se arquivos críticos existem"""
    print("\n" + "="*70)
    print("✓ VERIFICAÇÃO DE ARQUIVOS")
    print("="*70)
    
    files_to_check = [
        "data/processed/tensor_cvli_prisoes_faccoes.npy",
        "data/processed/metadata_producao_v2.json",
        "outputs/model_stgcn_faccoes.pth",
        "src/model_faction_adapter.py",
        "src/predict_with_factions.py",
        "IMPLANTACAO_COMPLETA_FACCOES.md",
        "DEPLOYMENT_GUIDE.md",
    ]
    
    missing = []
    for file in files_to_check:
        path = project_root / file
        if path.exists():
            size = path.stat().st_size / 1024 / 1024 if path.is_file() else 0
            print(f"  ✅ {file:<45} {size:.1f} MB" if size else f"  ✅ {file}")
        else:
            print(f"  ❌ {file:<45} FALTANDO")
            missing.append(file)
    
    return len(missing) == 0

def check_python_packages():
    """Verifica se dependências estão instaladas"""
    print("\n" + "="*70)
    print("✓ VERIFICAÇÃO DE DEPENDÊNCIAS")
    print("="*70)
    
    packages = ["torch", "numpy", "pandas", "geopandas"]
    
    all_ok = True
    for package in packages:
        try:
            __import__(package)
            print(f"  ✅ {package:<20} OK")
        except ImportError:
            print(f"  ❌ {package:<20} NÃO INSTALADO")
            all_ok = False
    
    return all_ok

def check_model():
    """Verifica se modelo carrega corretamente"""
    print("\n" + "="*70)
    print("✓ VERIFICAÇÃO DE MODELO")
    print("="*70)
    
    try:
        import torch
        sys.path.insert(0, str(project_root / 'src'))
        from model_faction_adapter import STGCN_DynamicFactions
        
        model = STGCN_DynamicFactions(input_features=7, hidden_dim=32, num_nodes=121)
        print(f"  ✅ Modelo criado: {sum(p.numel() for p in model.parameters()):,} parâmetros")
        
        # Teste forward pass
        X = torch.randn(4, 14, 121, 7)
        output, aux = model(X, return_aux=True)
        print(f"  ✅ Forward pass: output {output.shape}, aux {aux.shape}")
        
        return True
    except Exception as e:
        print(f"  ⚠️  Modelo: {e}")
        return False

def check_data():
    """Verifica se dados estão carregáveis"""
    print("\n" + "="*70)
    print("✓ VERIFICAÇÃO DE DADOS")
    print("="*70)
    
    try:
        import numpy as np
        import json
        
        # Carregar tensor
        tensor_path = project_root / "data/processed/tensor_cvli_prisoes_faccoes.npy"
        X = np.load(tensor_path)
        print(f"  ✅ Tensor carregado: {X.shape} ({X.nbytes/1024/1024:.1f} MB)")
        
        # Carregar metadata
        meta_path = project_root / "data/processed/metadata_producao_v2.json"
        with open(meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        print(f"  ✅ Metadata: {meta['dias']} dias, {meta['bairros']} bairros")
        
        return True
    except Exception as e:
        print(f"  ❌ Erro ao carregar dados: {e}")
        return False

def run_quick_test():
    """Executa teste rápido de predição"""
    print("\n" + "="*70)
    print("✓ TESTE RÁPIDO DE PREDIÇÃO")
    print("="*70)
    
    try:
        print("  → Carregando preditor...")
        sys.path.insert(0, str(project_root / 'src'))
        from predict_with_factions import CVLIPredictor
        
        predictor = CVLIPredictor(
            model_path=project_root / 'outputs/model_stgcn_faccoes.pth',
            tensor_path=project_root / 'data/processed/tensor_cvli_prisoes_faccoes.npy',
            metadata_path=project_root / 'data/processed/metadata_producao_v2.json'
        )
        
        print("  → Gerando predições...")
        predictions = predictor.predict_next_window()
        
        print(f"  ✅ Predições geradas para {len(predictions)} bairros")
        print(f"  ✅ CVLI médio: {predictions['cvli_predito'].mean():.2f}")
        print(f"  ✅ Top 3 bairros:")
        for i, row in predictions.head(3).iterrows():
            print(f"      {i+1}. {row['bairro']}: {row['cvli_predito']:.2f}")
        
        return True
    except Exception as e:
        print(f"  ⚠️  Teste rápido: {e}")
        return False

def main():
    os.chdir(project_root)
    
    print("\n")
    print("=" * 70)
    print("VALIDACAO DA IMPLANTACAO ST-GCN")
    print("=" * 70)
    
    results = {
        "Arquivos": check_files(),
        "Dependências": check_python_packages(),
        "Modelo": check_model(),
        "Dados": check_data(),
    }
    
    # Teste rápido (opcional)
    print("\n" + "="*70)
    print("TESTE RÁPIDO DE PREDIÇÃO (OPCIONAL)")
    print("="*70)
    try_predict = input("Deseja executar teste rápido de predição? (s/n): ").lower() == 's'
    if try_predict:
        results["Predição"] = run_quick_test()
    
    # Sumário
    print("\n" + "="*70)
    print("SUMÁRIO")
    print("="*70)
    
    for check, result in results.items():
        status = "✅ OK" if result else "❌ FALHA"
        print(f"  {check:<20} {status}")
    
    all_pass = all(results.values())
    
    print("\n" + "="*70)
    if all_pass:
        print("✅ VALIDAÇÃO COMPLETA - SISTEMA PRONTO PARA PRODUÇÃO")
        print("="*70)
        print("\n📋 Próximos passos:")
        print("  1. Executar: python src/predict_with_factions.py")
        print("  2. Consultar outputs/RELATORIO_PREDICOES.md")
        print("  3. Revisar DEPLOYMENT_GUIDE.md para setup completo")
    else:
        print("❌ VALIDAÇÃO FALHOU - REVISAR ERROS ACIMA")
        print("="*70)
    
    return 0 if all_pass else 1

if __name__ == "__main__":
    sys.exit(main())
