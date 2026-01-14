# 🚀 Quick Start Guide

## Instalação Rápida (Windows)

### Pré-requisitos
- **Python 3.12+** ([Download](https://www.python.org/downloads/))
- **Git** (opcional, para clone)

### Passo 1: Clone o Repositório
```bash
git clone <repository-url>
cd stock-lstm-flask
```

### Passo 2: Execute o Setup
```bash
setup.bat
```

Este script irá:
- ✅ Criar ambiente virtual (`.venv`)
- ✅ Instalar todas as dependências
- ✅ Criar estrutura de pastas (`instance/`, `models/`, `logs/`)
- ✅ Gerar arquivo `.env` com configurações padrão

### Passo 3: Inicie o Servidor
```bash
start.bat
```

### Passo 4: Acesse a Aplicação
- 🌐 **Dashboard:** http://127.0.0.1:5000/
- 📚 **API Docs (Swagger):** http://127.0.0.1:5000/apidocs
- 🔧 **Simulador (Popular Dados):** http://127.0.0.1:5000/simulate

---

## Instalação Manual (Alternativa)

Se preferir instalar manualmente:

```bash
# 1. Criar ambiente virtual
python -m venv .venv

# 2. Ativar ambiente (Windows)
.venv\Scripts\activate

# 3. Atualizar pip
python -m pip install --upgrade pip

# 4. Instalar dependências
pip install -r requirements.txt

# 5. Criar pastas
mkdir instance models logs

# 6. Criar arquivo .env (copie de .env.example)
copy .env.example .env

# 7. Editar .env e configurar SECRET_KEY e API_KEY

# 8. Executar aplicação
python wsgi.py
```

---

## Primeiros Passos Após Instalação

### 1. Popular o Banco de Dados
Acesse: http://127.0.0.1:5000/simulate

Clique em **"Simular dados NVDA"** para:
- Baixar dados históricos do Yahoo Finance
- Criar registros no banco SQLite
- Preparar o sistema para uso

### 2. Treinar Primeiro Modelo (Opcional)
Execute via API:
```bash
curl -X POST "http://localhost:5000/api/tasks/daily_update" ^
  -H "X-API-KEY: dev-api-key" ^
  -H "Content-Type: application/json" ^
  -d "{\"ticker\":\"AAPL\"}"
```

### 3. Explorar a API
Acesse o Swagger: http://127.0.0.1:5000/apidocs

Endpoints disponíveis:
- `GET /api/tickers` - Lista tickers
- `GET /api/predict?ticker=AAPL` - Previsão
- `GET /api/backtest?ticker=AAPL` - Backtest
- `GET /api/models/best?ticker=AAPL` - Melhor modelo

---

## Troubleshooting

### Erro: "Python não encontrado"
**Solução:** Instale Python 3.12+ de https://www.python.org/

### Erro: "Ambiente virtual não encontrado"
**Solução:** Execute `setup.bat` primeiro

### Erro: "ModuleNotFoundError: No module named 'flask'"
**Solução:** 
```bash
.venv\Scripts\activate
pip install -r requirements.txt
```

### Erro: "Port 5000 already in use"
**Solução:** Mate o processo:
```bash
# PowerShell
Get-Process -Name python | Stop-Process -Force

# CMD
taskkill /F /IM python.exe
```

### Banco de dados vazio
**Solução:** Acesse http://127.0.0.1:5000/simulate e popule os dados

---

## Configuração Avançada

### Variáveis de Ambiente (.env)

```env
# Flask
SECRET_KEY=seu-secret-key-seguro-aqui
FLASK_ENV=development

# API Key (protege endpoints administrativos)
API_KEY=sua-api-key-segura-aqui
DISABLE_API_KEY=0  # Set 1 para dev, 0 para produção

# Database
DATABASE_URL=sqlite:///instance/app.db

# Models
MODELS_DIR=models

# TensorFlow (opcional)
TF_ENABLE_ONEDNN_OPTS=0  # Para reprodutibilidade
```

### Modo Produção

Para executar em produção use Gunicorn:
```bash
gunicorn -b 0.0.0.0:5000 wsgi:app --workers=1 --threads=4 --timeout=120
```

---

## Estrutura de Pastas Criadas

```
stock-lstm-flask/
├── .venv/          # Ambiente virtual (ignorado pelo git)
├── instance/       # Banco SQLite (ignorado pelo git)
├── models/         # Modelos ML treinados (ignorado pelo git)
├── logs/           # Logs da aplicação (ignorado pelo git)
└── .env            # Configurações locais (ignorado pelo git)
```

---

## Scripts Disponíveis

| Script | Descrição |
|--------|-----------|
| `setup.bat` | Instalação inicial completa |
| `start.bat` | Inicia o servidor Flask |
| `start.ps1` | Inicia Flask em background (PowerShell) |

---

## Próximos Passos

1. ✅ Leia o [README.md](README.md) completo
2. ✅ Explore a [documentação da API](http://127.0.0.1:5000/apidocs)
3. ✅ Configure monitoramento (ver `docs/GRAFANA_SETUP.md`)
4. ✅ Treine modelos customizados (ver `docs/INTEGRACAO_TREINO_AVANCADO.md`)

---

**⭐ Projeto pronto para uso! Qualquer dúvida, consulte a documentação em `/docs`**
