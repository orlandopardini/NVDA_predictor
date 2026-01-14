# 📊 Visualizações Avançadas - Modelo Customizado LSTM

## ✨ Novas Funcionalidades Implementadas

Agora quando você treinar um modelo customizado, receberá **análises completas e detalhadas**:

### 1. 📋 Métricas Expandidas

**ANTES:** Apenas 4 cards simples  
**AGORA:** 6 cards informativos:

- ✅ **MAE** (Mean Absolute Error)
- ✅ **RMSE** (Root Mean Squared Error)  
- ✅ **MAPE** (Mean Absolute Percentage Error)
- ✅ **Epochs** (Épocas treinadas)
- ✅ **Duração** (Tempo total de treino em segundos)
- ✅ **RAM** (Memória utilizada em MB)

### 2. 🏗️ Arquitetura da Rede Neural

Visualização interativa da estrutura do modelo:

- **Cards coloridos** para cada layer:
  - 🔵 LSTM/GRU (azul)
  - 🟢 Dense (verde)
  - 🟠 Dropout (laranja)
  - ⚫ Outros (cinza)

- **Informações por layer:**
  - Tipo (LSTM, Dense, Dropout, etc.)
  - Número de units
  - Shape de saída
  - Total de parâmetros
  - Função de ativação

- **Resumo expandível:**
  - Total de parâmetros do modelo
  - Parâmetros treináveis
  - Texto completo do `model.summary()`

**Exemplo visual:**
```
┌─────────────────────────────────────────┐
│ 🏗️ Arquitetura da Rede Neural          │
├─────────────────────────────────────────┤
│ Total de Parâmetros: 4,385             │
│ Parâmetros Treináveis: 4,385           │
├─────────────────────────────────────────┤
│ ▌LSTM (32 units)                        │
│   Shape: (None, 32) | Params: 4,352    │
│   Activation: tanh                      │
├─────────────────────────────────────────┤
│ ▌Dense (1 units)                        │
│   Shape: (None, 1) | Params: 33        │
│   Activation: linear                    │
└─────────────────────────────────────────┘
```

### 3. 📈 Histórico de Treino (Loss & MAE)

**Gráfico interativo Plotly** com 2 séries:

- 🔴 **Training Loss** (linha + marcadores)
- 🟢 **Validation Loss** (linha + marcadores)

**Recursos:**
- Hover mostra valores exatos
- Zoom com mouse
- Pan (arrastar)
- Resetar zoom
- Download como PNG

### 4. 📉 Ajuste Temporal (Real vs Predito)

Visualização da série temporal completa:

- 🔵 **Linha azul**: Valores reais
- 🔴 **Linha pontilhada vermelha**: Valores preditos

**Recursos especiais:**
- 🟢 **Linha verde vertical**: Separação Treino | Validação
- Hover unified (mostra ambas as séries)
- Identificação visual de overfitting/underfitting

### 5. 🎯 Gráfico de Dispersão (Real vs Predito)

**Scatter plot** para avaliar qualidade das predições:

- Cada ponto representa uma predição
- Cores em gradiente (Viridis) mostram valores
- **Linha diagonal vermelha**: Linha ideal (y=x)

**Interpretação:**
- Pontos na diagonal = predições perfeitas
- Pontos acima da linha = modelo superestima
- Pontos abaixo da linha = modelo subestima

### 6. 📊 Distribuição de Resíduos

**Scatter plot temporal dos erros**:

- 🟢 Pontos verdes: Erros positivos (modelo subestimou)
- 🔴 Pontos vermelhos: Erros negativos (modelo superestimou)
- Linha cinza tracejada no zero

**Análise ideal:**
- Resíduos distribuídos aleatoriamente ao redor de zero
- Sem padrões visíveis (indica heterocedasticidade)
- Variância constante ao longo do tempo

### 7. 📊 Histograma de Erros

**Distribuição estatística dos resíduos**:

- 30 bins (barras)
- Cor azul uniforme

**Interpretação:**
- Distribuição normal centrada em zero = modelo bem ajustado
- Assimetria = viés sistemático
- Caudas longas = outliers

## 🚀 Como Usar

### 1. Acessar a Página

```
http://localhost:5000/custom-model
```

### 2. Configurar o Modelo

1. **Adicionar Layers:**
   - Clique em "Adicionar LSTM", "Adicionar Dense", etc.
   - Configure units, dropout, activation para cada layer

2. **Configurar Hiperparâmetros:**
   - Ticker (símbolo da ação)
   - Epochs (número de épocas)
   - Batch Size
   - Lookback (janela temporal)
   - Validation Split

3. **Treinar:**
   - Clique em "Treinar Modelo"
   - Aguarde (pode levar alguns minutos)

### 3. Analisar Resultados

Após o treino, role a página para baixo e explore:

1. **Métricas** → Avalie a performance geral
2. **Arquitetura** → Entenda a estrutura do modelo
3. **Histórico de Treino** → Verifique convergência
4. **Ajuste Temporal** → Veja como o modelo prevê a série
5. **Dispersão** → Analise qualidade das predições
6. **Resíduos** → Identifique vieses ou padrões
7. **Histograma** → Confirme distribuição normal dos erros

## 📝 Exemplo de Configuração

### Modelo Básico (Rápido - 1 minuto)

```json
{
  "ticker": "NVDA",
  "epochs": 5,
  "batch_size": 32,
  "config": {
    "layers": [
      {"type": "LSTM", "units": 32},
      {"type": "Dense", "units": 1}
    ]
  }
}
```

**Resultado esperado:**
- Treino: ~10-15 segundos
- Predições: ~3000 pontos
- MAE: ~5-10 (depende do ativo)

### Modelo Avançado (Preciso - 5 minutos)

```json
{
  "ticker": "AAPL",
  "epochs": 50,
  "batch_size": 16,
  "config": {
    "layers": [
      {"type": "LSTM", "units": 128, "return_sequences": true, "dropout": 0.2},
      {"type": "LSTM", "units": 64, "dropout": 0.2},
      {"type": "Dense", "units": 32, "activation": "relu"},
      {"type": "Dropout", "rate": 0.3},
      {"type": "Dense", "units": 1}
    ]
  }
}
```

**Resultado esperado:**
- Treino: ~3-5 minutos
- Predições: ~3000 pontos
- MAE: ~3-7 (melhor performance)

## 🎨 Paleta de Cores

Toda a interface usa a paleta **AZUL** (#1e90ff):

- Títulos: `#1e90ff`
- Botões: `#1e90ff` (hover: `#1c7ed6`)
- Bordas: `#1e90ff`
- Cards de métricas: `#1e90ff`

## 🔧 Recursos Técnicos

### Backend (Flask)

**Endpoint:** `POST /api/train-custom`

**Retorno JSON:**
```json
{
  "status": "success",
  "model_name": "NVDA_CUSTOM_20251111_171234",
  "metrics": {
    "mae": 8.13,
    "rmse": 9.88,
    "mape": 7.42
  },
  "history": {
    "loss": [0.012, 0.008, ...],
    "val_loss": [0.015, 0.011, ...],
    "mae": [0.051, 0.042, ...],
    "val_mae": [0.055, 0.045, ...]
  },
  "epochs_trained": 5,
  "resources": {
    "duration_sec": 12.45,
    "ram_used_mb": 125.3,
    "cpu_percent_avg": 45.2
  },
  "predictions": {
    "y_true": [245.2, 247.8, ...],  // 3000 valores
    "y_pred": [244.9, 248.1, ...],  // 3000 valores
    "residuals": [0.3, -0.3, ...],  // 3000 valores
    "split_index": 2400  // Índice de divisão treino/validação
  },
  "architecture": {
    "summary": [
      {
        "name": "lstm",
        "type": "LSTM",
        "output_shape": "(None, 32)",
        "params": 4352,
        "units": 32,
        "activation": "tanh"
      },
      {
        "name": "dense",
        "type": "Dense",
        "output_shape": "(None, 1)",
        "params": 33,
        "units": 1,
        "activation": "linear"
      }
    ],
    "text": "Model: \"sequential\"\n_______...",
    "total_params": 4385,
    "trainable_params": 4385
  }
}
```

### Frontend (JavaScript + Plotly)

**Funções de Plotagem:**

1. `plotArchitecture(architecture)` - Visualiza estrutura
2. `plotTrainingHistory(history)` - Loss ao longo das épocas
3. `plotTimeSeries(predictions)` - Série temporal real vs predito
4. `plotScatter(predictions)` - Dispersão y_real vs y_pred
5. `plotResiduals(predictions)` - Resíduos temporais
6. `plotErrorHistogram(predictions)` - Distribuição de erros

**Biblioteca de Gráficos:**
- Plotly.js 2.14.0 (carregado via CDN)
- Gráficos responsivos
- Interativos (zoom, pan, hover)

## 🐛 Troubleshooting

### Problema: Flask crasha durante treino

**Solução:** Use o virtual environment correto:
```powershell
.\start.ps1  # Inicia Flask com venv
```

### Problema: Gráficos não aparecem

**Verificar:**
1. Console do navegador (F12) para erros JavaScript
2. Se `response.status_code == 200`
3. Se todos os campos estão no JSON de resposta

### Problema: Treino muito lento

**Otimizações:**
- Reduzir `epochs` (testar com 5-10 primeiro)
- Aumentar `batch_size` (32 ou 64)
- Reduzir `units` nas layers LSTM
- Usar menos layers

### Problema: Métricas ruins (MAE alto)

**Ajustes:**
- Aumentar `epochs` (50-100)
- Adicionar mais layers LSTM
- Aumentar `units` (64, 128)
- Ajustar `lookback` (testar 30, 60, 90)
- Adicionar Dropout (0.2-0.3)

## 📚 Referências

- **TensorFlow/Keras:** https://www.tensorflow.org/api_docs/python/tf/keras
- **Plotly.js:** https://plotly.com/javascript/
- **LSTM Networks:** https://colah.github.io/posts/2015-08-Understanding-LSTMs/

## ✅ Status Atual

**Implementado e Testado:**

- ✅ 6 cards de métricas (incluindo duração e RAM)
- ✅ Visualização interativa da arquitetura
- ✅ Gráfico de histórico de treino (loss + val_loss)
- ✅ Série temporal (real vs predito) com linha de divisão treino/validação
- ✅ Scatter plot (dispersão) com linha ideal
- ✅ Gráfico de resíduos temporais (verde/vermelho)
- ✅ Histograma de distribuição de erros
- ✅ Endpoint retorna todos os dados necessários
- ✅ Frontend renderiza todos os gráficos
- ✅ Paleta de cores azul (#1e90ff)
- ✅ Título "Modelo Personalizado LSTM" (sem GRU)

**Testado com sucesso:**
- Ticker: NVDA
- Epochs: 1
- Resultado: 200 OK, 2951 predições, 2 layers, 3.15s

---

**Última Atualização:** 11/11/2025  
**Versão:** 2.0 (Visualizações Avançadas)  
**Autor:** AI Assistant
