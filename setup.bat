@echo off
REM ============================================
REM Script de Instalação Inicial
REM Stock LSTM Flask - Setup Completo
REM ============================================

echo.
echo ====================================
echo  STOCK LSTM - INSTALACAO INICIAL
echo ====================================
echo.

REM Verifica se Python está instalado
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERRO] Python nao encontrado!
    echo Por favor, instale Python 3.12+ de: https://www.python.org/
    pause
    exit /b 1
)

echo [1/5] Verificando Python...
python --version
echo.

REM Cria ambiente virtual se não existir
if not exist ".venv" (
    echo [2/5] Criando ambiente virtual...
    python -m venv .venv
    echo OK Ambiente virtual criado!
) else (
    echo [2/5] Ambiente virtual ja existe.
)
echo.

REM Verifica se dependências já estão instaladas
echo [3/5] Verificando dependencias...
call .venv\Scripts\activate.bat
.venv\Scripts\python.exe -c "import flask, tensorflow, pandas" 2>nul
if %errorlevel% neq 0 (
    echo Instalando dependencias...
    python -m pip install --upgrade pip --quiet
    pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo [ERRO] Falha ao instalar dependencias!
        pause
        exit /b 1
    )
    echo OK Dependencias instaladas!
) else (
    echo OK Dependencias principais ja instaladas.
)

REM SEMPRE atualiza yfinance para versão mais recente
echo Verificando atualizacao do yfinance...
pip install --upgrade yfinance --quiet
if %errorlevel% equ 0 (
    echo OK yfinance atualizado para versao mais recente!
) else (
    echo [AVISO] Nao foi possivel atualizar yfinance (continuando...)
)
echo.

REM Cria pastas necessárias
echo [4/5] Criando estrutura de pastas...
if not exist "instance" mkdir instance
if not exist "models" mkdir models
if not exist "logs" mkdir logs
echo OK Pastas criadas!
echo.

REM Cria arquivo .env se não existir
if not exist ".env" (
    echo [5/5] Criando arquivo .env...
    (
        echo # Flask Configuration
        echo SECRET_KEY=dev-secret-key-change-in-production
        echo FLASK_ENV=development
        echo.
        echo # API Key for admin tasks
        echo API_KEY=dev-api-key
        echo DISABLE_API_KEY=1
        echo.
        echo # Database
        echo DATABASE_URL=sqlite:///instance/app.db
        echo.
        echo # Models Directory
        echo MODELS_DIR=models
    ) > .env
    echo OK Arquivo .env criado!
) else (
    echo [5/5] Arquivo .env ja existe.
)
echo.

echo ====================================
echo  INSTALACAO CONCLUIDA COM SUCESSO!
echo ====================================
echo.
echo Proximos passos:
echo   1. Execute: start.bat
echo   2. Acesse: http://localhost:5000
echo   3. Swagger: http://localhost:5000/apidocs
echo.
echo Para popular o banco com dados:
echo   - Acesse: http://localhost:5000/simulate
echo.
pause
