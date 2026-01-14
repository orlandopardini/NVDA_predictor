# Script PowerShell para iniciar Flask em background
Write-Host "Iniciando Flask em background usando venv..." -ForegroundColor Green

# Mata processos Python existentes
Get-Process -Name python -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Seconds 2

# Inicia Flask usando Python do venv
$process = Start-Process -FilePath ".\.venv\Scripts\python.exe" -ArgumentList "wsgi.py" -WindowStyle Hidden -PassThru

Start-Sleep -Seconds 5

# Testa se Flask iniciou
try {
    $response = Invoke-WebRequest -Uri "http://localhost:5000/" -UseBasicParsing
    if ($response.StatusCode -eq 200) {
        Write-Host "OK Flask iniciado com sucesso! PID: $($process.Id)" -ForegroundColor Green
        Write-Host "  - Aplicacao: http://localhost:5000" -ForegroundColor Cyan
        Write-Host "  - Grafana: http://localhost:3000" -ForegroundColor Cyan
        Write-Host "  - Prometheus: http://localhost:9090" -ForegroundColor Cyan
    }
} catch {
    Write-Host "ERRO ao iniciar Flask" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Yellow
}
