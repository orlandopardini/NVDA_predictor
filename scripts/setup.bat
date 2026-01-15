@echo off
REM ============================================
REM Script de Instalação Inicial
REM Stock LSTM Flask - Setup com Python 3.12
REM ============================================

echo.
echo ====================================
echo  STOCK LSTM - INSTALACAO INICIAL
echo ====================================
echo.

REM Verifica se Python 3.12 está instalado
py -3.12 --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERRO] Python 3.12 nao encontrado!
    echo Instalando Python 3.12...
    winget install --id Python.Python.3.12 --accept-source-agreements --accept-package-agreements
    if %errorlevel% neq 0 (
        echo [ERRO] Falha ao instalar Python 3.12!
        echo Por favor, instale manualmente de: https://www.python.org/
        pause
        exit /b 1
    )
)

echo [1/4] Verificando Python 3.12...
py -3.12 --version
echo.

REM Remove ambiente virtual antigo se existir
if exist "venv" (
    echo [2/4] Removendo ambiente virtual antigo...
    rmdir /s /q venv
)

REM Cria ambiente virtual com Python 3.12
echo [2/4] Criando ambiente virtual com Python 3.12...
py -3.12 -m venv venv
if %errorlevel% neq 0 (
    echo [ERRO] Falha ao criar ambiente virtual!
    pause
    exit /b 1
)
echo OK Ambiente virtual criado!
echo.

REM Ativa ambiente e instala dependências
echo [3/4] Instalando dependencias...
call venv\Scripts\activate.bat
python -m pip install --upgrade pip --quiet
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [ERRO] Falha ao instalar dependencias!
    pause
    exit /b 1
)
echo OK Dependencias instaladas!
echo.
REM Inicializa banco de dados
echo [4/4] Inicializando banco de dados...
python -c "from app import create_app, db; app = create_app(); app.app_context().push(); db.create_all(); print('OK!')"
if %errorlevel% neq 0 (
    echo [AVISO] Banco de dados pode ja existir.
) else (
    echo OK Banco de dados criado!
)
echo.

echo ====================================
echo  INSTALACAO CONCLUIDA COM SUCESSO!
echo ====================================
echo.
echo Para iniciar o servidor, execute: start.bat
echo.
pause

echo   1. Execute: start.bat
echo   2. Acesse: http://localhost:5000
echo   3. Swagger: http://localhost:5000/apidocs
echo.
echo Para popular o banco com dados:
echo   - Acesse: http://localhost:5000/simulate
echo.
pause
