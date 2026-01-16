<#
Detecta a versão do PyTorch e instala wheels PyG (torch-scatter, torch-sparse, torch-cluster, torch-spline-conv)
Uso: execute com o venv ativado ou o script ativará o python do venv se você passar o caminho.
#>

# Caminho para o python do venv (ajuste se necessário)
$venvPython = "./venv/Scripts/python.exe"
if (-Not (Test-Path $venvPython)) {
    Write-Error "Não encontrou $venvPython. Ative seu venv antes de rodar ou ajuste a variável.`nEx: .\venv\Scripts\Activate.ps1"
    exit 1
}

# Detectar versão do torch e cuda
try {
    $out = & $venvPython -c "import torch;print(torch.__version__.split('+')[0] + '|' + (torch.version.cuda or 'cpu'))"
} catch {
    Write-Error "Falha ao importar torch. Certifique-se de que o venv tem torch instalado. Erro: $_"
    exit 1
}

$out = $out.Trim()
if (-not $out.Contains('|')) {
    Write-Error "Saída inesperada do python: $out"
    exit 1
}

$parts = $out.Split('|')
$torchVer = $parts[0]
$cudaRaw = $parts[1]

if ([string]::IsNullOrWhiteSpace($cudaRaw) -or $cudaRaw -eq 'None') { $cudaRaw = 'cpu' }

if ($cudaRaw -eq 'cpu') {
    $cudaTag = 'cpu'
} else {
    # transformar '11.8' => 'cu118'
    $cudaTag = 'cu' + ($cudaRaw -replace '\.','')
}

$url = "https://data.pyg.org/whl/torch_$torchVer+$cudaTag.html"

Write-Host "Detectado: PyTorch=$torchVer  CUDA=$cudaRaw"
Write-Host "Usando index: $url"

$pkgs = "torch-scatter torch-sparse torch-cluster torch-spline-conv"

Write-Host "Comando a ser executado: pip install --no-cache-dir $pkgs -f $url"

$confirm = Read-Host "Deseja prosseguir com a instalação? (s/N)"
if ($confirm.ToLower() -ne 's' -and $confirm.ToLower() -ne 'y') {
    Write-Host "Aborting. Nenhuma alteração feita."
    exit 0
}

# Executar instalação (uso do call operator para capturar o exit code)
$cmd = "$venvPython -m pip install --no-cache-dir $pkgs -f $url"
Write-Host "Executando: $cmd"

$installArgs = @('-m','pip','install','--no-cache-dir') + $pkgs.Split() + @('-f',$url)
& $venvPython @installArgs
if ($LASTEXITCODE -ne 0) {
    Write-Error "pip retornou código exit $LASTEXITCODE. Veja a saída acima para detalhes."
    exit $LASTEXITCODE
}

Write-Host "Instalação concluída com sucesso. Testando importações..."
try {
    & $venvPython -c "import torch; import importlib; import sys; importlib.import_module('torch_scatter'); importlib.import_module('torch_sparse'); print('IMPORTS_OK')"
    Write-Host "IMPORTS_OK"
} catch {
    Write-Error "Falha ao importar os módulos instalados: $_"
    exit 1
}

Write-Host "Tudo pronto."