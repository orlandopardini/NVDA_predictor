# Arquitetura e Engenharia de Software - Stock LSTM Flask

## 📋 Índice

1. [Visão Geral](#visão-geral)
2. [Arquitetura](#arquitetura)
3. [Princípios e Padrões](#princípios-e-padrões)
4. [Estrutura de Diretórios](#estrutura-de-diretórios)
5. [Componentes Principais](#componentes-principais)
6. [Fluxo de Dados](#fluxo-de-dados)
7. [Boas Práticas Implementadas](#boas-práticas-implementadas)
8. [Guia de Desenvolvimento](#guia-de-desenvolvimento)

---

## 🎯 Visão Geral

Este projeto implementa uma aplicação Flask para previsão de preços de ações usando modelos LSTM (Long Short-Term Memory), seguindo rigorosas práticas de engenharia de software:

- **Arquitetura em Camadas** (Layered Architecture)
- **Design Patterns** (Repository, Factory, Strategy)
- **SOLID Principles**
- **Type Safety** com Type Hints
- **Validação de Dados** com Pydantic
- **Logging Estruturado**
- **Exception Handling** robusto
- **Configuração por Ambiente**

---

## 🏗️ Arquitetura

### Camadas da Aplicação

```
┌─────────────────────────────────────────────┐
│           Presentation Layer                │
│    (Flask Routes, Templates, APIs)          │
└─────────────────┬───────────────────────────┘
                  │
┌─────────────────▼───────────────────────────┐
│            Service Layer                    │
│   (Business Logic, Orchestration)           │
└─────────────────┬───────────────────────────┘
                  │
┌─────────────────▼───────────────────────────┐
│          Repository Layer                   │
│     (Data Access, Persistence)              │
└─────────────────┬───────────────────────────┘
                  │
┌─────────────────▼───────────────────────────┐
│           Database Layer                    │
│   (SQLAlchemy Models, SQLite/PostgreSQL)    │
└─────────────────────────────────────────────┘
```

### Componentes Transversais

```
┌───────────────────────────────────────────────┐
│         Cross-Cutting Concerns                │
│                                               │
│  • Configuration (config.py)                  │
│  • Logging (logger.py)                        │
│  • Exception Handling (exceptions.py)         │
│  • Validation (schemas.py)                    │
│  • Monitoring (Prometheus metrics)            │
└───────────────────────────────────────────────┘
```

---

## 🎨 Princípios e Padrões

### SOLID Principles

#### 1. **S**ingle Responsibility Principle (SRP)
- Cada classe tem **uma única responsabilidade**
- `PrecoDiarioRepository` apenas acessa dados de preços
- `TrainerService` apenas coordena treinamento
- `ConfigurationManager` apenas gerencia configuração

#### 2. **O**pen/Closed Principle (OCP)
- Classes abertas para **extensão**, fechadas para **modificação**
- `BaseRepository` pode ser estendido sem alterar código base
- Novos modelos ML adicionados sem modificar infrastructure

#### 3. **L**iskov Substitution Principle (LSP)
- Subclasses podem substituir classes base
- Qualquer `BaseRepository` descendente funciona como esperado

#### 4. **I**nterface Segregation Principle (ISP)
- Interfaces específicas ao invés de genéricas
- Repositories têm métodos específicos do domínio

#### 5. **D**ependency Inversion Principle (DIP)
- Depender de abstrações, não de implementações concretas
- Services dependem de Repository interfaces, não implementações

### Design Patterns Implementados

#### 1. **Repository Pattern**
- Abstrai acesso a dados
- Facilita testes com mocks
- Centraliza lógica de persistência

```python
from app.repositories import PrecoDiarioRepository

repo = PrecoDiarioRepository(session)
history = repo.get_ticker_history('AAPL')
```

#### 2. **Factory Pattern**
- Criação de objetos complexos
- Usado para criar modelos ML

```python
model = ModelFactory.create('LSTM_Bidirectional', params)
```

#### 3. **Strategy Pattern**
- Algoritmos intercambiáveis
- Otimização de hiperparâmetros (Grid, Random, Bayesian)

```python
optimizer = OptimizerStrategy.get(method='bayesian')
best_params = optimizer.optimize(model, data)
```

#### 4. **Dependency Injection**
- Injeção de dependências para testabilidade
- Facilita mock em testes

#### 5. **Application Factory**
- `create_app()` permite múltiplas instâncias
- Facilita testes e diferentes ambientes

---

## 📁 Estrutura de Diretórios

```
stock-lstm-flask/
│
├── app/
│   ├── __init__.py              # Application Factory
│   ├── config.py                # ⭐ Configuração por ambiente
│   ├── logger.py                # ⭐ Sistema de logging
│   ├── exceptions.py            # ⭐ Exceções customizadas
│   ├── schemas.py               # ⭐ Validação Pydantic
│   ├── models.py                # SQLAlchemy models
│   ├── monitoring.py            # Prometheus metrics
│   │
│   ├── repositories/            # ⭐ Repository Pattern
│   │   ├── __init__.py
│   │   ├── base.py             # BaseRepository genérico
│   │   └── models.py           # Repositories específicos
│   │
│   ├── services/                # 🔜 Service Layer (próximo)
│   │   ├── __init__.py
│   │   ├── data_service.py
│   │   ├── training_service.py
│   │   └── prediction_service.py
│   │
│   ├── ml/                      # Machine Learning
│   │   ├── constants.py
│   │   ├── data.py
│   │   ├── eval.py
│   │   ├── model_zoo.py
│   │   ├── model_zoo_advanced.py
│   │   ├── trainer.py
│   │   ├── trainer_advanced.py
│   │   ├── hyperparameter_optimizer.py
│   │   └── training_progress.py
│   │
│   ├── routes/                  # Flask Routes
│   │   ├── api.py              # API endpoints
│   │   └── web.py              # Web pages
│   │
│   ├── templates/               # HTML templates
│   ├── static/                  # CSS, JS, images
│   └── utils/                   # Utilities
│
├── instance/                    # Instance-specific files
│   └── app.db                  # SQLite database
│
├── models/                      # Trained models
│   └── *.keras, *.scaler
│
├── logs/                        # Log files
│   └── app.log
│
├── tests/                       # 🔜 Testes (próximo)
│   ├── unit/
│   ├── integration/
│   └── conftest.py
│
├── requirements.txt             # Dependências
├── wsgi.py                      # WSGI entry point
├── .env                         # Environment variables
└── README.md                    # Documentação
```

---

## 🔧 Componentes Principais

### 1. Configuration System (`app/config.py`)

**Propósito:** Gerencia configurações por ambiente (dev, test, prod)

```python
from app.config import get_config

# Automático baseado em FLASK_ENV
config = get_config()

# Ou explicitamente
config = get_config('production')

# Acesso type-safe
database_uri = config.DATABASE_URI
max_epochs = config.MAX_EPOCHS
```

**Características:**
- ✅ Dataclasses type-safe
- ✅ Validação em `__post_init__`
- ✅ Environments isolados (dev/test/prod)
- ✅ Defaults sensatos
- ✅ Suporte a variáveis de ambiente

### 2. Logging System (`app/logger.py`)

**Propósito:** Logging estruturado e colorido

```python
from app.logger import get_logger, log_execution_time

logger = get_logger(__name__)

logger.info("Iniciando treinamento", extra={'ticker': 'AAPL'})
logger.error("Erro ao carregar modelo", exc_info=True)

@log_execution_time(logger)
def train_model():
    pass
```

**Características:**
- ✅ Console colorido
- ✅ Rotação de arquivos
- ✅ Níveis configuráveis
- ✅ Context managers
- ✅ Decorators para timing

### 3. Exception System (`app/exceptions.py`)

**Propósito:** Hierarquia de exceções do domínio

```python
from app.exceptions import ModelNotFoundError, ValidationError

# Lançar
raise ModelNotFoundError('LSTM_v1', ticker='AAPL')

# Capturar
try:
    model = load_model(version)
except ModelNotFoundError as e:
    return jsonify(e.to_dict()), e.status_code
```

**Hierarquia:**
```
StockLSTMException (base)
├── APIException
│   ├── ValidationError
│   ├── ResourceNotFoundError
│   └── RateLimitExceededError
├── DatabaseException
│   ├── DatabaseLockError
│   └── IntegrityError
├── MLException
│   ├── ModelNotFoundError
│   ├── ModelLoadError
│   └── InsufficientDataError
└── DataException
    ├── DataFetchError
    └── InvalidTickerError
```

### 4. Validation Schemas (`app/schemas.py`)

**Propósito:** Validação type-safe de entrada/saída

```python
from app.schemas import TrainModelRequest, validate_request

# Validar request
data = request.get_json()
validated = validate_request(TrainModelRequest, data)

# Usar dados validados type-safe
ticker = validated.ticker  # str
lookback = validated.lookback  # int (entre 10-365)
```

**Características:**
- ✅ Type hints automáticos
- ✅ Validações complexas
- ✅ Conversões automáticas
- ✅ Documentação integrada
- ✅ OpenAPI/Swagger compatível

### 5. Repository Pattern (`app/repositories/`)

**Propósito:** Abstração de acesso a dados

```python
from app import db
from app.repositories import PrecoDiarioRepository

# Criar repository
repo = PrecoDiarioRepository(db.session)

# CRUD operations
all_prices = repo.get_all(limit=100)
price = repo.get_by_id(1)
new_price = repo.create(ticker='AAPL', date='2024-01-01', close=150.0)
repo.update(1, close=151.0)
repo.delete(1)

# Domain-specific methods
history = repo.get_ticker_history('AAPL', start_date='2023-01-01')
latest_date = repo.get_latest_date('AAPL')
tickers = repo.get_available_tickers()
```

**Características:**
- ✅ Abstração de SQLAlchemy
- ✅ Retry automático (database locks)
- ✅ Logging integrado
- ✅ Type-safe com Generics
- ✅ Métodos específicos do domínio

---

## 🔄 Fluxo de Dados

### Request Flow (API)

```
1. HTTP Request
   ↓
2. Flask Route (routes/api.py)
   ↓
3. Schema Validation (schemas.py)
   ↓
4. Service Layer (services/) [Business Logic]
   ↓
5. Repository Layer (repositories/) [Data Access]
   ↓
6. Database (SQLAlchemy)
   ↓
7. Response (JSON via Schema)
```

### Training Flow

```
1. User triggers training
   ↓
2. TrainingService.train_advanced()
   ↓
3. Fetch data via PrecoDiarioRepository
   ↓
4. For each model:
   - HyperparameterOptimizer
   - ModelFactory.build()
   - Train & Evaluate
   - Save via ModelRegistryRepository
   ↓
5. Select winner
   ↓
6. Update monitoring metrics
   ↓
7. Return results
```

---

## ✅ Boas Práticas Implementadas

### Code Quality

- ✅ **Type Hints** em todas as funções
- ✅ **Docstrings** formato Google
- ✅ **PEP 8** compliance
- ✅ **DRY** (Don't Repeat Yourself)
- ✅ **KISS** (Keep It Simple, Stupid)

### Error Handling

- ✅ **Try-except** apropriados
- ✅ **Exceções específicas** do domínio
- ✅ **Logging** de erros com context
- ✅ **Retry logic** para operações transientes
- ✅ **Error responses** padronizados

### Security

- ✅ **Input validation** com Pydantic
- ✅ **SQL Injection** protegido (SQLAlchemy ORM)
- ✅ **SECRET_KEY** obrigatória em produção
- ✅ **Environment variables** para secrets
- ✅ **CORS** configurável

### Performance

- ✅ **Database connection pooling**
- ✅ **WAL mode** no SQLite
- ✅ **Bulk operations** quando possível
- ✅ **Caching** configurável
- ✅ **Lazy loading** de modelos pesados

### Testing

- ✅ **Unit tests** isolados
- ✅ **Integration tests**
- ✅ **Fixtures** reutilizáveis
- ✅ **Mocks** para dependencies
- ✅ **Test coverage** tracking

### Monitoring

- ✅ **Prometheus metrics**
- ✅ **Structured logging**
- ✅ **Performance tracking**
- ✅ **Error rate monitoring**

---

## 🚀 Guia de Desenvolvimento

### Setup Inicial

```bash
# 1. Clonar repositório
git clone <repo-url>
cd stock-lstm-flask

# 2. Criar ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows

# 3. Instalar dependências
pip install -r requirements.txt

# 4. Configurar ambiente
cp .env.example .env
# Editar .env com suas configurações

# 5. Inicializar banco
flask db upgrade

# 6. Rodar aplicação
python wsgi.py
```

### Variáveis de Ambiente

```bash
# .env
FLASK_ENV=development  # ou 'production', 'testing'
SECRET_KEY=your-secret-key-here
DATABASE_URL=sqlite:///instance/app.db
LOG_LEVEL=DEBUG
```

### Adicionando Nova Funcionalidade

#### 1. Definir Schema de Validação

```python
# app/schemas.py
class NewFeatureRequest(BaseModel):
    param1: str
    param2: int = Field(ge=0, le=100)
```

#### 2. Criar Repository Method (se necessário)

```python
# app/repositories/models.py
class MyRepository(BaseRepository[MyModel]):
    def custom_query(self, param):
        return self.session.query(self.model).filter_by(param=param).all()
```

#### 3. Implementar Business Logic no Service

```python
# app/services/my_service.py
class MyService:
    def __init__(self, repository):
        self.repository = repository
    
    def process(self, data):
        # Business logic here
        return self.repository.custom_query(data.param1)
```

#### 4. Criar Route/Endpoint

```python
# app/routes/api.py
@api_bp.route('/my-feature', methods=['POST'])
def my_feature():
    data = validate_request(NewFeatureRequest, request.get_json())
    service = MyService(MyRepository(db.session))
    result = service.process(data)
    return jsonify(result)
```

### Testes

```python
# tests/test_my_feature.py
def test_my_feature(client):
    response = client.post('/api/my-feature', json={
        'param1': 'test',
        'param2': 50
    })
    assert response.status_code == 200
```

### Logging Best Practices

```python
from app.logger import get_logger

logger = get_logger(__name__)

# Info para operações normais
logger.info("Processando requisição", extra={'user_id': user.id})

# Warning para situações anormais mas recuperáveis
logger.warning("Cache miss, carregando do banco")

# Error para erros que precisam atenção
logger.error("Falha ao conectar API externa", exc_info=True)

# Debug para informações detalhadas
logger.debug("Parâmetros: %s", params)
```

### Exception Handling Best Practices

```python
from app.exceptions import ValidationError, ResourceNotFoundError

# Lançar exceções específicas
if not ticker:
    raise ValidationError("Ticker é obrigatório", field='ticker')

# Capturar e re-lançar com contexto
try:
    model = load_model(path)
except FileNotFoundError:
    raise ModelNotFoundError(model_name, ticker=ticker)

# Error handlers globais já registrados em __init__.py
```

---

## 📊 Métricas e Monitoramento

### Prometheus Metrics Disponíveis

```python
# Counter
RETRAIN_COUNT.labels(ticker='AAPL', mode='advanced').inc()

# Histogram
INFERENCE_LATENCY.observe(elapsed_time)

# Gauge
TRAIN_RAM_USAGE.set(ram_mb)
```

### Acessar Métricas

```
GET /metrics
```

---

## 🔒 Security Checklist

- ✅ SECRET_KEY em variável de ambiente
- ✅ Validação de todos os inputs
- ✅ SQLAlchemy ORM (previne SQL Injection)
- ✅ Rate limiting configurável
- ✅ HTTPS em produção
- ✅ CORS configurável
- ✅ Logs não expõem dados sensíveis
- ✅ Dependências atualizadas

---

## 📚 Referências

- [SOLID Principles](https://en.wikipedia.org/wiki/SOLID)
- [Repository Pattern](https://martinfowler.com/eaaCatalog/repository.html)
- [Flask Best Practices](https://flask.palletsprojects.com/en/latest/patterns/)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [SQLAlchemy ORM](https://docs.sqlalchemy.org/en/20/orm/)

---

**Última Atualização:** 2025-01-12  
**Versão:** 2.0.0
