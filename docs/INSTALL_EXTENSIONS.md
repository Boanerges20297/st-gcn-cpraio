# INSTRUÇÕES PARA INSTALAR TORCH-SCATTER E TORCH-SPARSE
# ========================================================

# Problema: torch-scatter e torch-sparse são extensões C++ que precisam ser compiladas.
# Isso requer Microsoft C++ Build Tools com Windows SDK instalado.

# OPÇÃO 1: Usar Conda (RECOMENDADO - mais fácil)
# ================================================
# 1. Instale Miniconda: https://docs.conda.io/projects/miniconda/en/latest/
# 2. Abra Anaconda Prompt e execute:
#
#    conda create -n stgcn python=3.10 -y
#    conda activate stgcn
#    conda install pytorch cpuonly -c pytorch -y
#    conda install pytorch-scatter pytorch-sparse -c pyg -y
#    pip install -r requirements-core.txt
#
# 3. Use este ambiente Conda para rodar o projeto


# OPÇÃO 2: Instalar Microsoft C++ Build Tools (requer ~10GB)
# ===========================================================
# 1. Baixe "Visual Studio Build Tools 2022":
#    https://visualstudio.microsoft.com/visual-cpp-build-tools/
#
# 2. Execute o instalador e selecione:
#    ✓ "Desktop development with C++"
#    ✓ "Windows 11 SDK" (ou versão mais recente)
#
# 3. Após instalar, execute no PowerShell (.venv ativado):
#    python -m pip install torch-scatter torch-sparse --no-build-isolation
#
# 4. Isso compilará as extensões localmente


# OPÇÃO 3: Procurar wheels pré-compilados em repositórios
# =========================================================
# Às vezes, repositories como HuggingFace ou CSDN têm wheels pré-compilados.
# Exemplo (se disponível):
#    pip install "path/to/torch_scatter-2.1.2-cp310-cp310-win_amd64.whl"
#    pip install "path/to/torch_sparse-0.6.18-cp310-cp310-win_amd64.whl"


# INFORMAÇÕES DE DEBUG
# ====================
# Se nenhuma opção funcionar, o projeto pode ser usado sem estas dependências
# para muitos cenários, mas torch-geometric + torch-scatter/sparse são necessários
# para usar o modelo ST-GCN completo.
