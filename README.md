# 📈 Stock LSTM Prediction Platform
### Sistema End-to-End de Previsão de Séries Financeiras com Machine Learning

![Python](https://img.shields.io/badge/Python-3.12-blue) ![Flask](https://img.shields.io/badge/Flask-3.0-green) ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18-orange) ![Quality](https://img.shields.io/badge/Code%20Quality-90.1%2F100-brightgreen) ![License](https://img.shields.io/badge/License-MIT-yellow)

> **Plataforma completa de previsão de preços de ações** usando LSTM (Long Short-Term Memory) com API REST, frontend interativo, monitoramento Prometheus e pipeline automatizado de ML.

---

##  Início Rápido (Windows)

###  Configuração em 2 Passos

**Pré-requisitos:** Python 3.12+ ([Download](https://www.python.org/downloads/))

**1️⃣ Configurar o Ambiente (Executar UMA VEZ)**
```bash
setup.bat
```
Este script irá:
- ✅ Criar o ambiente virtual Python
- ✅ Instalar todas as dependências (Flask, TensorFlow, etc)
- ✅ Criar o banco de dados SQLite
- ✅ Preparar o projeto para execução

**2️⃣ Iniciar o Servidor (Sempre que quiser usar)**
```bash
start.bat
```
Este script irá:
- ✅ Ativar o ambiente virtual
- ✅ Iniciar o servidor Flask na porta 5000
- ✅ Abrir automaticamente no navegador

🌐 **Acesse:** http://127.0.0.1:5000

### 🔄 Scripts Disponíveis

| Script | Quando Usar | O Que Faz |
|--------|-------------|-----------|
| `setup.bat` | **Primeira vez** ou após atualizar dependências | Instala/atualiza ambiente completo |
| `start.bat` | **Sempre que quiser iniciar** o servidor | Inicia aplicação Flask |
| `start.ps1` | Alternativa PowerShell para `start.bat` | Mesma função do start.bat |

### 📝 Observações Importantes

- **Primeira vez:** Execute `setup.bat` antes de usar `start.bat`
- **Problemas com setup:** Execute como Administrador ou verifique se Python está no PATH
- **Porta em uso:** Se a porta 5000 estiver ocupada, edite a porta em `start.bat`

---

##  O Que Este Sistema Faz

### 📊 **Previsão de Preços**
- **Previsão de próximo dia útil** para ações (AAPL, NVDA, MSFT, GOOGL, AMZN, TSLA)
- **Modelos LSTM** treinados com janelas temporais (60-90 dias)
- **Seleção automática** do melhor modelo baseado em métricas de validação

### 🔄 **Pipeline Automatizado de ML**
- **Ingestão automática** de dados via Yahoo Finance
- **Treinamento incremental** com detecção de drift
- **Avaliação contínua** e registro de performance
- **Retreinamento automático** via cron job diário

### 📉 **Análise e Visualização**
- **Backtesting** com gráficos Real vs. Previsto
- **Análise de erro** com MAE rolling e dispersão
- **Dashboard interativo** com Plotly (tema dark + neon)
- **Métricas consolidadas** por modelo e ticker

### 🛠️ **API REST Completa**
- **9 endpoints** documentados com Swagger
- **Autenticação** via API Key para tasks administrativas
- **Respostas otimizadas** com cache e índices de banco
- **Rate limiting** e validação de entrada

### 📡 **Monitoramento Operacional**
- **Métricas Prometheus** (latência, requests, recursos)
- **Health checks** automáticos
- **Logs estruturados** de treino e previsão
- **Alertas** de drift e degradação de performance

---

## 🏗️ Arquitetura Resumida

```
┌─────────────────────────────────────────────────────────────┐
│                     FLASK APPLICATION                        │
├──────────────┬──────────────────┬────────────────────────────┤
│   Frontend   │    REST API      │    ML Pipeline             │
│   (Plotly)   │   (9 endpoints)  │   (TensorFlow/Keras)       │
├──────────────┼──────────────────┼────────────────────────────┤
│ • Dashboard  │ • /api/series    │ • Data Ingestion (yfinance)│
│ • Backtest   │ • /api/predict   │ • Feature Engineering      │
│ • Simulate   │ • /api/backtest  │ • Model Training (LSTM)    │
│              │ • /api/models/*  │ • Evaluation & Selection   │
│              │ • /api/tasks/*   │ • Model Registry           │
└──────────────┴──────────────────┴────────────────────────────┘
                         │
            ┌────────────┴────────────┐
            │    SQLite Database      │
            ├─────────────────────────┤
            │ • PrecoDiario (OHLCV)   │
            │ • ModelRegistry         │
            │ • ResultadoMetricas     │
            │ • RetrainHistory        │
            └─────────────────────────┘
```

**Stack Tecnológico:**
- **Backend:** Flask 3.0 + SQLAlchemy + Gunicorn
- **ML:** TensorFlow 2.18 + Keras + scikit-learn
- **Frontend:** HTML5 + Plotly.js + Bootstrap
- **Data:** yfinance + pandas + numpy
- **Monitoring:** Prometheus + psutil
- **Deploy:** Render.com (Docker-ready)

---

## 📊 Engenharia de Dados & Métricas

###  Métricas de Avaliação de Modelos

| Métrica | Descrição | Ideal | Uso no Sistema |
|---------|-----------|-------|----------------|
| **RMSE** | Raiz do Erro Quadrático Médio | <10% do preço | Penaliza outliers fortemente |
| **MAE** | Erro Absoluto Médio | <5% do preço | Métrica principal de seleção |
| **MAPE** | Erro Percentual Absoluto Médio | <10% | Comparação entre tickers |
| **R²** | Coeficiente de Determinação | >0.80 | Qualidade do ajuste |
| **ACC** | Acurácia Direcional (↑/↓) | >55% | Decisão de trade |

### 📈 Performance Atual do Sistema

```
┌─────────────────────────────────────────────────────────────┐
│  MÉTRICAS DE QUALIDADE DE CÓDIGO (Score: 90.1/100 - Grade A) │
├─────────────────────────────────────────────────────────────┤
│  Documentação:          42.0% (3,210 linhas)         🟢     │
│  Complexidade (CC):     2.95 (média)                 ✅     │
│  Manutenibilidade (MI): 68.7/100 (B+)                🟢     │
│  Acoplamento:           100/100 (Excelente)          ✅     │
│  Cobertura Type Hints:  56.6%                        🟢     │
├─────────────────────────────────────────────────────────────┤
│  Total de Linhas:       9,315                               │
│  Arquivos Python:       39                                  │
│  Funções:               232                                 │
│  Classes:               42                                  │
└─────────────────────────────────────────────────────────────┘
```

### 🔬 Pipeline de Dados

**1. Ingestão e Limpeza**
```python
# Fonte: Yahoo Finance API (yfinance)
# Frequência: Diária (após fechamento do mercado)
# Período histórico: 2-5 anos por ticker
# Tratamento: Remoção de NaN, outliers, ajuste de dividendos
```

**2. Feature Engineering**
```python
# Features temporais:
- Lags (1, 5, 10, 20 dias)
- Returns (simples e log)
- Médias móveis (7, 21, 50 dias)
- Volatilidade (rolling std 20 dias)
- Volume normalizado

# Normalização: MinMaxScaler por ticker
# Window size: 60-90 dias (sequências LSTM)
```

**3. Split Strategy**
```python
# Train:      70% (dados mais antigos)
# Validation: 15% (período intermediário)
# Test:       15% (dados mais recentes)
# Método: Temporal split (sem shuffle para preservar ordem)
```

**4. Model Performance (Exemplo AAPL)**
```
┌──────────────────────────────────────────────────┐
│  Modelo: LSTM(64x64) + Dense                     │
│  Versão: AAPL_2_20251008_180812                  │
├──────────────────────────────────────────────────┤
│  RMSE:     9.05  (<10% threshold ✅)             │
│  MAE:      7.10  (<8% threshold ✅)              │
│  R²:       0.861 (>0.80 threshold ✅)            │
│  ACC:      46.6% (direção do movimento)          │
│  Latência: <200ms (inference)                    │
└──────────────────────────────────────────────────┘
```

### 🗄️ Esquema de Dados

**Tabela: `PrecoDiario`**
- Armazena OHLCV (Open, High, Low, Close, Volume)
- Índice único: `(ticker, date)`
- ~2,000-5,000 registros/ticker
- Atualização: Diária via cron

**Tabela: `ModelRegistry`**
- Catálogo de modelos treinados
- Flag `is_winner` identifica melhor modelo
- Metadados: hyperparameters, versão, timestamp
- ~5-20 modelos/ticker (versionamento)

**Tabela: `ResultadoMetricas`**
- Histórico de avaliações (RMSE, MAE, MAPE, R², ACC)
- Usado para análise de drift e retreinamento
- ~100-500 registros/ticker

**Tabela: `RetrainHistory`**
- Log de execuções de treino (sucesso/falha)
- Duração, exceções, dataset size
- Debugging e auditoria

### 📡 Monitoramento de Dados

```python
# Métricas de Drift (monitor.py)
rolling_mae = MAE(real[-20:], pred[-20:])  # Janela de 20 dias
baseline_mae = 7.10  # MAE histórico do modelo

if rolling_mae > baseline_mae * 1.25:
    trigger_retrain(ticker, reason="drift_detected")
```

---

## 📖 Instalação Manual (Linux/Mac)

Para instalação manual ou em outros sistemas operacionais, consulte: [INSTALL.md](INSTALL.md)

### 🌐 Acesse a Aplicação

- � **Dashboard:** http://127.0.0.1:5000/
- 📚 **API Docs (Swagger):** http://127.0.0.1:5000/apidocs
- 🔧 **Simulador (Popular Dados):** http://127.0.0.1:5000/simulate
- 📊 **Métricas Prometheus:** http://127.0.0.1:5000/metrics

### 🎯 Primeiro Uso

1. **Acesse:** http://127.0.0.1:5000/simulate
2. **Clique:** "Simular dados NVDA" para popular o banco
3. **Explore:** Dashboard e API endpoints

---

## 🔌 API Endpoints

### 📊 Dados e Previsões

| Endpoint | Método | Descrição | Autenticação |
|----------|--------|-----------|--------------|
| `/api/tickers` | GET | Lista tickers suportados | Não |
| `/api/series` | GET | Série histórica de preços | Não |
| `/api/predict` | GET | Previsão do próximo dia | Não |
| `/api/backtest` | GET | Dados de backtest (real vs previsto) | Não |

### 🏆 Modelos

| Endpoint | Método | Descrição | Autenticação |
|----------|--------|-----------|--------------|
| `/api/models/best` | GET | Melhor modelo para ticker | Não |
| `/api/models/summary` | GET | Todos os modelos do ticker | Não |

### 🔧 Tarefas Administrativas

| Endpoint | Método | Descrição | Autenticação |
|----------|--------|-----------|--------------|
| `/api/tasks/daily_update` | POST | Atualiza dados e treina modelos | **X-API-KEY** |
| `/api/tasks/retrain` | POST | Força retreinamento de modelo | **X-API-KEY** |
| `/api/tasks/status` | GET | Status das tarefas em execução | **X-API-KEY** |

### 📖 Exemplo de Uso

```bash
# Obter previsão para AAPL
curl "http://localhost:5000/api/predict?ticker=AAPL"

# Resposta:
{
  "date_next": "2025-11-15",
  "pred": 234.56,
  "ticker": "AAPL",
  "model_version": "AAPL_2_20251008_180812"
}

# Executar atualização diária (com autenticação)
curl -X POST "http://localhost:5000/api/tasks/daily_update" \
  -H "X-API-KEY: your-secret-key" \
  -H "Content-Type: application/json" \
  -d '{"ticker": "AAPL"}'
```

---

## 🎨 Frontend Features

### Dashboard Principal (`/`)
- **Gráfico de Previsão (30 dias):** Real vs Previsto com banda de confiança
- **Análise de Erro:** MAE rolling (20 dias) + dispersão
- **Série Histórica (365 dias):** Visualização de longo prazo
- **Seletor de Ticker:** Dropdown com 6 ações principais
- **Métricas do Modelo:** RMSE, MAE, R², ACC em tempo real

### Simulador (`/simulate`)
- **Preenchimento de dados** para desenvolvimento/testes
- **Geração de séries sintéticas** para novos tickers
- **Validação de pipeline** end-to-end

### Design
- ✨ **Tema Dark** com bordas neon (cyan/purple)
- 📱 **Responsivo** (Bootstrap 5)
- ⚡ **Plotly interativo** com zoom, pan, hover
- 🎯 **UX otimizada** para análise técnica

---

## 🚀 Deploy (Render.com)

### Configuração `render.yaml`

```yaml
services:
  - type: web
    name: stock-lstm-flask
    env: python
    buildCommand: "pip install -r requirements.txt"
    startCommand: "gunicorn -b 0.0.0.0:$PORT wsgi:app --workers=1 --threads=4 --timeout=120"
    envVars:
      - key: SECRET_KEY
        generateValue: true
      - key: API_KEY
        value: change-me-in-production
      - key: MODELS_DIR
        value: models
    disk:
      name: models-disk
      mountPath: /opt/render/project/src/models
      sizeGB: 1

cronJobs:
  - name: daily-update
    schedule: "0 9 * * *"  # 09:00 UTC (06:00 BRT)
    command: >
      curl -X POST "$RENDER_EXTERNAL_URL/api/tasks/daily_update"
      -H "X-API-KEY: $API_KEY"
      -H "Content-Type: application/json"
      -d '{"ticker":"AAPL"}'
```

### Checklist de Deploy

- ✅ **Persistir volumes:** `models/` e `instance/` (SQLite)
- ✅ **Configurar `$PORT`:** Gunicorn deve bindar em `0.0.0.0:$PORT`
- ✅ **Setar `API_KEY`:** Proteger endpoints administrativos
- ✅ **Workers:** Usar **1 worker** + threads para TensorFlow
- ✅ **Cron timezone:** Render usa **UTC** (ajustar horários)

---

## 🔐 Variáveis de Ambiente

| Variável | Padrão | Descrição | Obrigatório |
|----------|--------|-----------|-------------|
| `SECRET_KEY` | — | Chave secreta Flask (sessions, CSRF) | ✅ Produção |
| `API_KEY` | — | Chave para autenticação de tasks | ✅ Produção |
| `MODELS_DIR` | `models` | Diretório de armazenamento de modelos | Não |
| `DISABLE_API_KEY` | `0` | Desabilita auth em dev (set `1`) | Não |
| `FLASK_ENV` | `production` | Ambiente (`development` para debug) | Não |
| `TF_ENABLE_ONEDNN_OPTS` | — | Set `0` para reprodutibilidade TF | Não |

---

## 📚 Estrutura do Projeto

```
stock-lstm-flask/
├── app/
│   ├── __init__.py              # App factory + configuração
│   ├── models.py                # ORM SQLAlchemy (4 tabelas)
│   ├── monitoring.py            # Prometheus metrics + middleware
│   ├── monitoring_simple.py     # Métricas CPU/RAM (psutil)
│   ├── ml/
│   │   ├── constants.py         # Hyperparameters e configuração
│   │   ├── data.py              # Ingestão e feature engineering
│   │   ├── model_zoo.py         # Arquiteturas LSTM
│   │   ├── trainer_advanced.py  # Training loop e callbacks
│   │   ├── eval.py              # Métricas e backtesting
│   │   ├── pipeline.py          # Orquestração end-to-end
│   │   └── monitor.py           # Drift detection
│   ├── routes/
│   │   ├── api.py               # 9 endpoints REST
│   │   └── web.py               # Frontend HTML + assets
│   ├── static/
│   │   ├── index.html           # Dashboard principal
│   │   ├── simulate.html        # Simulador
│   │   ├── style.css            # Tema dark + neon
│   │   └── index.js             # Lógica de gráficos
│   └── utils/
│       └── timing.py            # Stopwatch para performance
├── models/                       # Modelos .keras + scalers
├── instance/                     # SQLite database
├── wsgi.py                       # Entry point Gunicorn
├── requirements.txt              # Dependências Python
├── render.yaml                   # Deploy config
└── README.md
```

---

## 🧪 Desenvolvimento

### Executar Testes

```bash
# Testes unitários (em desenvolvimento)
pytest tests/

# Teste manual de endpoint
curl "http://localhost:5000/api/predict?ticker=AAPL"

# Verificar qualidade de código
radon cc app/ -a -s         # Complexidade ciclomática
radon mi app/ -s            # Índice de manutenibilidade
```

### Treinar Modelo Manualmente

```python
from app.ml.trainer_advanced import train_all_models_fast_mode
from app import create_app

app = create_app()
with app.app_context():
    train_all_models_fast_mode(ticker='AAPL', window=60)
```

### Análise de Qualidade

```bash
# Executar análise completa
python detailed_code_rules_analysis_v2.py

# Métricas esperadas:
# - Score: 90.1/100 (A - Excelente)
# - Documentação: 42%
# - Complexidade: 2.95 (baixa)
# - MI: 68.7/100 (B+)
```

---

## 🐛 Troubleshooting

### Problema: Banco "zera" após deploy
**Solução:** Persistir a pasta `instance/` como volume no Render

### Problema: Previsões inconsistentes
**Solução:** 
- Fixar `TF_ENABLE_ONEDNN_OPTS=0`
- Verificar scaler por ticker
- Conferir window size (60-90 dias)

### Problema: Erro 401 em tasks
**Solução:** Adicionar header `X-API-KEY` com valor correto

### Problema: Alta latência em requests
**Solução:**
- Usar 1 worker + 4-8 threads (Gunicorn)
- Implementar cache de previsões (Redis)
- Otimizar queries com índices

### Problema: Gráficos não carregam
**Solução:**
- Verificar console do navegador (F12)
- Conferir endpoints `/api/series` e `/api/backtest`
- Validar formato JSON retornado

---

## 📈 Roadmap

### Em Desenvolvimento
- [ ] **Testes automatizados** (pytest + coverage >80%)
- [ ] **CI/CD Pipeline** (GitHub Actions)
- [ ] **Cache Redis** para previsões
- [ ] **Autoscaling** baseado em carga

### Futuro
- [ ] **Novos modelos:** GRU, Transformer, Prophet
- [ ] **Mais features:** Sentimento (Twitter), notícias, indicadores técnicos
- [ ] **Multi-asset:** Forex, crypto, commodities
- [ ] **Trading bot:** Integração com corretoras (Alpaca, IB)
- [ ] **A/B Testing** de modelos em produção

---

## 📄 Licença

MIT License - Veja [LICENSE](LICENSE) para detalhes.

---

## 👨‍💻 Autor

Desenvolvido com ❤️ usando Python, TensorFlow e Flask.

**Status do Projeto:** ✅ Production Ready (Score: 90.1/100)

---

## 📞 Suporte

- 📚 **Documentação API:** `/apidocs` (Swagger)
- 📊 **Métricas:** `/metrics` (Prometheus)
- 🐛 **Issues:** GitHub Issues
- 💬 **Discussões:** GitHub Discussions

---

**⭐ Se este projeto foi útil, considere dar uma estrela!**
