@echo off
REM ============================================
REM Script de Inicialização do Flask
REM Stock LSTM Flask com Python 3.12
REM ============================================

echo.
echo ====================================
echo  STOCK LSTM - INICIANDO SERVIDOR
echo ====================================
echo.

REM Verifica se o ambiente virtual existe
if not exist "venv\Scripts\python.exe" (
    echo [ERRO] Ambiente virtual nao encontrado!
    echo.
    echo Execute primeiro: setup.bat
    echo.
    pause
    exit /b 1
)

REM Ativa o ambiente virtual
call venv\Scripts\activate.bat

REM Inicia o servidor Flask
echo Iniciando servidor Flask...
echo Acesse: http://127.0.0.1:5000
echo.
echo Pressione CTRL+C para parar o servidor
echo.

python -m flask run --host=0.0.0.0 --port=5000

pause

)

REM Verifica se as dependências estão instaladas
.\.venv\Scripts\python.exe -c "import flask" 2>nul
if %errorlevel% neq 0 (
    echo [AVISO] Dependencias nao instaladas!
    echo.
    echo Execute: setup.bat
    echo.
    pause
    exit /b 1
)

echo [OK] Ambiente configurado corretamente
echo.
echo Atualizando yfinance para versao mais recente...
.\.venv\Scripts\python.exe -m pip install --upgrade yfinance --quiet
if %errorlevel% equ 0 (
    echo [OK] yfinance atualizado!
) else (
    echo [AVISO] Nao foi possivel atualizar yfinance (continuando...)
)
echo.
echo Iniciando Flask...
echo.
echo Servidor disponivel em:
echo   - http://127.0.0.1:5000
echo   - http://127.0.0.1:5000/apidocs (Swagger)
echo.
echo Pressione Ctrl+C para parar o servidor
echo.
echo ====================================
echo.

REM Inicia a aplicação
.\.venv\Scripts\python.exe wsgi.py
