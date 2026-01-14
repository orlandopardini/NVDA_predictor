# 📋 Mapeamento Completo do api.py

## 📊 Estatísticas Gerais
- **Total de linhas**: 1472
- **Total de rotas**: 24 endpoints
- **Funções auxiliares**: 7 helpers
- **Imports duplicados**: 15+ imports redundantes
- **Complexidade**: God Object (viola Single Responsibility Principle)

---

## 🎯 Plano de Divisão em 5 Arquivos

### 1️⃣ **utils/data_helpers.py** (Helpers de dados)
**Funções a extrair:**
- `_normalize_ohlcv(df, ticker)` - Linha 31 (47 linhas)
- `_fetch_yahoo_block(ticker, d0, d1)` - Linha 77 (13 linhas)
- `_fetch_stooq_block(ticker, d0, d1)` - Linha 90 (17 linhas)
- `_fetch_resilient_yearly(ticker, start)` - Linha 107 (56 linhas)
- `update_winner_flag(ticker)` - Linha 163 (51 linhas)

**Total estimado**: ~180 linhas

---

### 2️⃣ **utils/auth_helpers.py** (Helpers de autenticação)
**Funções a extrair:**
- `_auth_ok(req)` - Linha 214 (6 linhas)
- `require_basic_auth(f)` - Linha 220 (14 linhas)

**Total estimado**: ~20 linhas

---

### 3️⃣ **api_data.py** (Rotas de dados)
**Rotas:**
- `POST /update_data` - Linha 246 (52 linhas) - Atualiza dados do ticker
- `GET /series` - Linha 580 (16 linhas) - Série OHLCV
- `GET /tickers` - Linha 670 (17 linhas) - Lista tickers disponíveis
- `POST /load_ticker_data` - Linha 689 (140 linhas) - Carrega dados em lotes

**Total estimado**: ~300 linhas
**Imports necessários**: PrecoDiario, db, yfinance, pandas, data_helpers

---

### 4️⃣ **api_train.py** (Rotas de treinamento)
**Rotas:**
- `POST /train` - Linha 298 (31 linhas) - Treino básico
- `POST /train-custom` - Linha 829 (264 linhas) - Treino customizado
- `POST /train-advanced` - Linha 1405 (85 linhas) - Treino com 30 modelos
- `POST /models/update-winner` - Linha 1572 (15 linhas) - Atualiza winner flag

**Total estimado**: ~450 linhas
**Imports necessários**: trainer, trainer_advanced, model_zoo, model_zoo_advanced, ModelRegistry

---

### 5️⃣ **api_predict.py** (Rotas de predição)
**Rotas:**
- `GET /predict` - Linha 409 (55 linhas) - Predição 1 passo
- `GET /simulate` - Linha 464 (76 linhas) - Predição multi-passo
- `POST /predict-loaded-model` - Linha 1226 (80 linhas) - Predição com modelo carregado

**Total estimado**: ~250 linhas
**Imports necessários**: trainer, keras, joblib, yfinance, pandas

---

### 6️⃣ **api_models.py** (Rotas de modelos)
**Rotas:**
- `GET /models/best` - Linha 329 (20 linhas) - Melhor modelo
- `GET /models/summary` - Linha 349 (60 linhas) - Resumo de modelos
- `GET /models-info` - Linha 1538 (28 linhas) - Info dos 30 modelos
- `GET /download-model` - Linha 1093 (43 linhas) - Download de modelo
- `POST /load-model` - Linha 1136 (90 linhas) - Upload de modelo
- `GET /advanced-model-predictions` - Linha 1323 (82 linhas) - Predições avançadas

**Total estimado**: ~400 linhas
**Imports necessários**: ModelRegistry, keras, joblib, model_zoo_advanced

---

### 7️⃣ **api_monitoring.py** (Rotas de monitoring)
**Rotas:**
- `GET /health` - Linha 234 (4 linhas) - Health check
- `GET /metrics` - Linha 540 (13 linhas) - Métricas
- `GET /retrain/history` - Linha 553 (13 linhas) - Histórico de retreino
- `GET /metrics/history` - Linha 596 (24 linhas) - Histórico de métricas
- `GET /backtest` - Linha 620 (50 linhas) - Backtest
- `POST /tasks/daily_update` - Linha 566 (14 linhas) - Update diário (cron)
- `GET /train-progress` - Linha 1297 (26 linhas) - Progresso do treino

**Total estimado**: ~200 linhas
**Imports necessários**: ResultadoMetricas, RetrainHistory, monitoring, eval

---

## 🔧 Ordem de Execução

### Fase 1: Preparação (15 min)
1. Criar `utils/data_helpers.py` com 5 funções auxiliares
2. Criar `utils/auth_helpers.py` com 2 funções de autenticação

### Fase 2: Divisão de Rotas (2-3h)
3. Criar `api_data.py` - 4 rotas
4. Criar `api_train.py` - 4 rotas
5. Criar `api_predict.py` - 3 rotas
6. Criar `api_models.py` - 6 rotas
7. Criar `api_monitoring.py` - 7 rotas

### Fase 3: Integração (30 min)
8. Atualizar `__init__.py` para registrar 5 novos blueprints
9. Manter `api.py` original como backup (renomear para `api_backup.py`)

### Fase 4: Testes (30 min)
10. Criar `test_api_routes.py` para validar todas as rotas
11. Executar testes e corrigir imports

### Fase 5: Análise (15 min)
12. Executar análise de qualidade final
13. Comparar métricas antes/depois

---

## ⚠️ Riscos e Mitigações

| Risco | Probabilidade | Mitigação |
|-------|---------------|-----------|
| Imports quebrados | Alta | Testar imports imediatamente após cada arquivo |
| Dependências circulares | Média | Helpers em utils/, não em routes/ |
| Blueprints não registrados | Baixa | Seguir padrão de web.py |
| Rotas não encontradas | Baixa | Manter prefixo `/api` em todos |

---

## ✅ Checklist de Validação

- [ ] Todos os 24 endpoints acessíveis
- [ ] Flask registra 5 blueprints (api_data, api_train, api_predict, api_models, api_monitoring)
- [ ] Imports funcionam sem circular dependency
- [ ] Testes passam 100%
- [ ] Score de qualidade melhora (92.2 → ?)
- [ ] Documentação atualizada

---

## 📈 Resultado Esperado

**Antes:**
- api.py: 1472 linhas (God Object)
- 24 rotas em 1 arquivo
- Complexidade: 0.0/100
- Manutenibilidade: Difícil

**Depois:**
- api_data.py: ~300 linhas (4 rotas)
- api_train.py: ~450 linhas (4 rotas)
- api_predict.py: ~250 linhas (3 rotas)
- api_models.py: ~400 linhas (6 rotas)
- api_monitoring.py: ~200 linhas (7 rotas)
- utils/data_helpers.py: ~180 linhas
- utils/auth_helpers.py: ~20 linhas
- **Total: ~1800 linhas** (divididas em 7 arquivos organizados)
- Complexidade: >70/100
- Manutenibilidade: Excelente

---

## 🎯 Próximos Passos

1. **Criar utils/data_helpers.py** ✅
2. **Criar utils/auth_helpers.py** ✅
3. **Criar api_data.py** (4 rotas)
4. **Criar api_train.py** (4 rotas)
5. **Criar api_predict.py** (3 rotas)
6. **Criar api_models.py** (6 rotas)
7. **Criar api_monitoring.py** (7 rotas)
8. **Atualizar __init__.py**
9. **Testar tudo**
10. **Análise final**
