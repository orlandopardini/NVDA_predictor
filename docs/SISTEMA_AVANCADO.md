# 🎯 SISTEMA AVANÇADO DE TREINO - 30 MODELOS + OTIMIZAÇÃO

## Visão Geral

Expandimos o sistema de 10 para **30 arquiteturas LSTM/GRU** com **otimização inteligente de hiperparâmetros**.

---

## 🏗️ Arquivos Criados

### 1. `app/ml/model_zoo_advanced.py`
**30 arquiteturas de alto desempenho** organizadas em 6 categorias:

#### Categoria 1: LSTM Base & Variants (1-5)
- **Modelo 1**: LSTM Classic (64/32)
- **Modelo 2**: LSTM + LayerNormalization
- **Modelo 3**: LSTM + BatchNormalization
- **Modelo 4**: LSTM Narrow-Deep (32³)
- **Modelo 5**: LSTM Wide-Shallow (256)

#### Categoria 2: GRU Base & Variants (6-10)
- **Modelo 6**: GRU Classic (64/32)
- **Modelo 7**: GRU Deep (128/64/32)
- **Modelo 8**: GRU Wide (192/96)
- **Modelo 9**: GRU Residual Dense
- **Modelo 10**: GRU Hybrid (80/80/40)

#### Categoria 3: Bidirectional (11-15)
- **Modelo 11**: BiLSTM Classic (64/32)
- **Modelo 12**: BiGRU Classic (64/32)
- **Modelo 13**: BiLSTM Deep (96/64/32)
- **Modelo 14**: BiGRU Deep (96/64/32)
- **Modelo 15**: BiLSTM+BiGRU Mix

#### Categoria 4: Stacked Deep Networks (16-20)
- **Modelo 16**: Stacked LSTM (128→32)
- **Modelo 17**: Stacked GRU (128→32)
- **Modelo 18**: Pyramid LSTM (256→16)
- **Modelo 19**: Inverted Pyramid (32→128)
- **Modelo 20**: Diamond LSTM (64/128/128/64)

#### Categoria 5: Residual & Skip Connections (21-25)
- **Modelo 21**: LSTM Residual v1
- **Modelo 22**: LSTM Residual v2
- **Modelo 23**: Skip Connection Dense
- **Modelo 24**: Highway LSTM
- **Modelo 25**: DenseNet-style LSTM

#### Categoria 6: Attention & Hybrid (26-30)
- **Modelo 26**: Self-Attention LSTM
- **Modelo 27**: Multi-Head Attention
- **Modelo 28**: CNN+LSTM Hybrid
- **Modelo 29**: LSTM TimeDistributed
- **Modelo 30**: Ensemble Multi-Path

---

### 2. `app/ml/hyperparameter_optimizer.py`
**Sistema de otimização com 3 estratégias:**

#### A. Grid Search
- Testa **TODAS** as combinações possíveis
- **Prós**: Garante encontrar melhor combinação
- **Contras**: MUITO lento (milhares de testes)
- **Quando usar**: Poucos modelos (1-5) e tempo disponível

#### B. Random Search
- Testa **N amostras aleatórias**
- **Prós**: Rápido e eficiente
- **Contras**: Pode perder ótimo global
- **Quando usar**: Padrão para 5-15 modelos

#### C. Bayesian Optimization
- **Aprende** com resultados anteriores
- Explora vizinhança dos melhores (70% exploitation)
- Adiciona exploração aleatória (30% exploration)
- **Prós**: Mais inteligente, converge rápido
- **Contras**: Precisa de >5 amostras iniciais
- **Quando usar**: >15 modelos ou tempo limitado

**Espaço de Busca:**
```python
'learning_rate': [0.0001, 0.0005, 0.001, 0.005, 0.01]
'batch_size': [16, 32, 64, 128]
'dropout_rate': [0.1, 0.15, 0.2, 0.25, 0.3, 0.4]
'epochs': [10, 20, 30, 50]
'activation': ['relu', 'tanh', 'elu', 'selu', 'swish', 'gelu', 'leaky_relu']
```

**Funções de Ativação (15 disponíveis):**
- Básicas: relu, tanh, sigmoid, linear
- Avançadas: leaky_relu, elu, selu
- Exponenciais: exponential, softplus, softsign
- Modernas: swish/SiLU, mish, gelu
- Hard variants: hard_sigmoid, hard_swish

**Early Stopping Inteligente:**
- Para se validação não melhora por N epochs (patience=5)
- Detecta divergência (loss aumentando)
- Evita overfitting

---

### 3. `app/ml/trainer_advanced.py`
**Dois modos de treino:**

#### MODO RÁPIDO (`train_all_models_fast_mode`)
- Testa 30 modelos com **1 epoch cada**
- Parâmetros fixos (batch_size=32, learning_rate=0.001)
- **Tempo estimado**: ~5 minutos
- **Objetivo**: Identificar arquiteturas promissoras rapidamente
- ✅ **Responde à pergunta**: "O treino rápido testa todas as funções?"
  - **NÃO**, usa parâmetros fixos (ativação padrão = relu)

#### MODO OTIMIZADO (`train_all_models_with_optimization`)
- Para CADA modelo, faz busca de hiperparâmetros
- Testa learning_rate, batch_size, dropout, activation, epochs
- **Tempo estimado**: 30-60 minutos (depende de n_trials e estratégia)
- **Objetivo**: Encontrar melhor configuração possível
- Salva campeão no disco (.keras + .scaler + .json)
- Registra no banco de dados (ModelRegistry)
- ✅ **Responde à pergunta**: "O treino forçado faz depuração e altera parâmetros?"
  - **SIM**, testa múltiplas combinações e busca melhor cenário

**Callbacks Avançados:**
- `EarlyStopping`: para se val_loss não melhora
- `ReduceLROnPlateau`: reduz learning rate em 50% se estagnado

---

### 4. `app/routes/api.py` (Novas Rotas)

#### POST `/api/train-advanced`
**Body JSON:**
```json
{
  "mode": "fast",  // ou "optimized"
  "model_ids": [1, 2, 3],  // null = todos 30
  "optimization_strategy": "random",  // grid/random/bayesian
  "n_trials": 20,  // tentativas por modelo
  "lookback": 60,
  "horizon": 1
}
```

**Resposta (modo fast):**
```json
{
  "status": "success",
  "mode": "fast",
  "winner": {
    "model_id": 11,
    "model_name": "BiLSTM Classic (64/32)",
    "rmse": 0.0234
  },
  "total_models": 30,
  "results": [...]
}
```

**Resposta (modo optimized):**
```json
{
  "status": "success",
  "mode": "optimized",
  "winner": {
    "model_id": 26,
    "model_name": "Self-Attention LSTM",
    "best_rmse": 0.0187,
    "best_params": {
      "learning_rate": 0.001,
      "batch_size": 64,
      "dropout_rate": 0.2,
      "epochs": 30,
      "activation": "swish"
    },
    "n_trials": 20,
    "elapsed_time": 342.5
  },
  "total_models_tested": 30,
  "total_time": 1823.4,
  "avg_time_per_model": 60.8,
  "optimization_strategy": "bayesian",
  "all_results": [...]
}
```

#### GET `/api/models-info`
Retorna lista dos 30 modelos organizados por categoria.

---

### 5. `app/templates/advanced_training.html`
**Interface gráfica moderna com:**
- Seleção de modo (Rápido vs Otimizado)
- Configuração de estratégia (Grid/Random/Bayesian)
- Número de tentativas por modelo
- Lookback e Horizon
- Barra de progresso animada
- Card do campeão com métricas
- Tabela com ranking de todos os modelos
- Medals (🥇🥈🥉) para top-3

---

## 🚀 Como Usar

### 1. Acessar a Interface
```
http://localhost:5000/advanced-training
```

### 2. Modo Rápido (Exploração)
1. Selecione "Modo Rápido"
2. Ajuste lookback/horizon se necessário
3. Clique "Iniciar Treino Avançado"
4. Aguarde ~5 minutos
5. Veja qual arquitetura teve melhor RMSE

### 3. Modo Otimizado (Produção)
1. Selecione "Modo Otimizado"
2. Escolha estratégia:
   - **Random Search** (recomendado): rápido e eficiente
   - **Bayesian**: mais inteligente, converge melhor
   - **Grid Search**: completo mas lento
3. Configure tentativas (20 = bom equilíbrio)
4. Clique "Iniciar Treino Avançado"
5. Aguarde 30-60 minutos
6. Modelo campeão salvo automaticamente em `models/`

### 4. Via API (Python)
```python
import requests

# Modo rápido
response = requests.post('http://localhost:5000/api/train-advanced', json={
    "mode": "fast",
    "lookback": 60,
    "horizon": 1
})
print(response.json())

# Modo otimizado
response = requests.post('http://localhost:5000/api/train-advanced', json={
    "mode": "optimized",
    "optimization_strategy": "bayesian",
    "n_trials": 30,
    "model_ids": [1, 11, 21, 26, 30],  # testar apenas 5 modelos específicos
    "lookback": 60,
    "horizon": 1
})
print(response.json())
```

---

## 📊 Comparação de Estratégias

| Estratégia | Velocidade | Qualidade | Uso de Memória | Recomendado Para |
|-----------|-----------|-----------|----------------|------------------|
| Grid Search | ⭐ (lento) | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ (alto) | Poucos modelos, tempo ilimitado |
| Random Search | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ (baixo) | **Uso geral (padrão)** |
| Bayesian | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ (médio) | Muitos modelos, produção |

---

## 🎯 Respostas às Perguntas

### "O treino rápido testa todas as funções?"
**NÃO**. O modo rápido (`fast`) testa apenas as **30 arquiteturas** com parâmetros fixos:
- 1 epoch
- batch_size=32
- learning_rate=0.001 (Adam)
- dropout_rate=0.2
- activation=relu (padrão de cada layer)

**Objetivo**: Identificar qual **arquitetura** (LSTM, GRU, BiLSTM, Attention, etc.) é mais promissora.

---

### "O treino forçado (optimized) faz depuração e altera parâmetros?"
**SIM**! O modo otimizado faz **busca inteligente**:

1. **Para CADA modelo** (1-30):
   - Testa N combinações de hiperparâmetros (n_trials=20 padrão)
   - Varia: learning_rate, batch_size, dropout, epochs, activation
   - Total de combinações possíveis: 5 × 4 × 6 × 4 × 7 = **3.360 combinações**

2. **Estratégia de busca** (Random/Bayesian):
   - Não testa todas 3.360 (seria semanas)
   - Random: testa 20 aleatórias por modelo
   - Bayesian: aprende com resultados e foca em regiões promissoras

3. **Para cada tentativa**:
   - Treina modelo com parâmetros específicos
   - Avalia RMSE em validação
   - Early stopping se não melhorar

4. **Resultado**:
   - Melhor configuração por modelo
   - Campeão global entre todos os modelos
   - **Salva automaticamente** o vencedor

**Exemplo de resultado:**
```
Modelo 26 (Self-Attention LSTM):
  - Testou 20 combinações
  - Melhor: learning_rate=0.001, batch_size=64, dropout=0.2, activation=swish, epochs=30
  - RMSE: 0.0187
  
CAMPEÃO GLOBAL: Modelo 26
```

---

## 🎨 Funcionalidades de Ativação

Cada modelo pode usar **15 funções de ativação diferentes**:

### Básicas
- `relu`: Rectified Linear Unit (padrão)
- `tanh`: Tangente Hiperbólica (padrão LSTM)
- `sigmoid`: Sigmoid (0 a 1)
- `linear`: Sem ativação

### Avançadas (Leaky família)
- `leaky_relu`: Permite pequeno gradiente negativo
- `elu`: Exponential Linear Unit
- `selu`: Scaled ELU (auto-normalizante)

### Exponenciais
- `exponential`: Crescimento exponencial
- `softplus`: Suave versão de ReLU
- `softsign`: Versão suave de tanh

### Modernas (State-of-the-art)
- `swish`: x * sigmoid(x) - usado em EfficientNet
- `mish`: x * tanh(softplus(x)) - melhor que ReLU
- `gelu`: Gaussian Error Linear Unit - usado em BERT/GPT

### Hard Variants
- `hard_sigmoid`: Versão rápida de sigmoid
- `hard_swish`: Versão rápida de swish

**Otimização automática testa essas ativações** e escolhe a melhor para cada modelo!

---

## 🔧 Próximos Passos

1. **Testar sistema**:
   ```bash
   # Acessar interface
   http://localhost:5000/advanced-training
   
   # Modo rápido primeiro (5 min)
   # Ver qual categoria de modelos funciona melhor
   
   # Depois modo otimizado (30-60 min)
   # Deixar rodando overnight para melhor resultado
   ```

2. **Monitorar progresso**:
   - Logs em tempo real no terminal
   - Barra de progresso na interface
   - Métricas Prometheus em `/metrics`

3. **Analisar resultados**:
   - Ranking completo de todos os modelos
   - Hiperparâmetros do campeão
   - Tempo de treino por modelo

4. **Usar campeão**:
   - Automaticamente salvo em `models/NVDA_{model_id}_{timestamp}.keras`
   - Já pode usar em `/custom-model` para predições
   - Registrado no banco como `is_winner=True`

---

## 📈 Performance Esperada

### Modo Rápido
- **Tempo**: ~5 minutos (30 modelos × 1 epoch)
- **Uso de RAM**: ~2-4 GB
- **CPU**: 50-70%
- **Resultado**: Top-3 arquiteturas promissoras

### Modo Otimizado (n_trials=20)
- **Tempo**: ~30-60 minutos (30 modelos × 20 trials = 600 treinos)
- **Uso de RAM**: ~4-8 GB (com early stopping)
- **CPU**: 70-90%
- **Resultado**: Modelo otimizado pronto para produção

### Modo Otimizado (n_trials=50)
- **Tempo**: ~2-3 horas (30 modelos × 50 trials = 1500 treinos)
- **Resultado**: Melhor modelo possível

---

## ✅ Sistema Completo

✅ **30 arquiteturas** (LSTM, GRU, BiLSTM, BiGRU, Residual, Attention, Hybrid)
✅ **15 funções de ativação** (relu, tanh, swish, gelu, mish, etc.)
✅ **3 estratégias de otimização** (Grid, Random, Bayesian)
✅ **5 hiperparâmetros otimizados** (lr, batch, dropout, epochs, activation)
✅ **Early stopping** inteligente
✅ **Interface gráfica** moderna
✅ **API REST** completa
✅ **Salvamento automático** do campeão
✅ **Registro no banco** de dados

**Sistema pronto para encontrar o melhor modelo LSTM para NVIDIA!**
