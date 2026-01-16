@echo off
REM Script para instalar torch-scatter e torch-sparse no .venv
REM Requer: Windows SDK e Visual Studio Build Tools instalados

echo.
echo ===================================================
echo Instalando torch-scatter e torch-sparse...
echo ===================================================
echo.

REM Ativar .venv
call .venv\Scripts\activate.bat

REM Tentar instalar com compilacao
echo Instalando torch-scatter e torch-sparse sem isolamento de build...
pip install torch-scatter torch-sparse --no-build-isolation

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ✓ Instalacao bem-sucedida!
    echo.
) else (
    echo.
    echo ✗ Instalacao falhou. Verifique se Windows SDK esta instalado.
    echo   Veja INSTALL_EXTENSIONS.md para detalhes.
    echo.
    exit /b 1
)
