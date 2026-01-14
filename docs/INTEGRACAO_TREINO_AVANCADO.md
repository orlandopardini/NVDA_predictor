# 🔗 INTEGRAÇÃO: Treino Avançado ↔ Tela Principal

## ✅ SIM! O modelo treinado no Treino Avançado aparece na tela principal

### Como funciona:

#### 1️⃣ **Quando você treina no Treino Avançado (Modo Otimizado)**:

O sistema automaticamente:
- 🎯 Encontra o **melhor modelo** entre os 30
- 🧪 Testa **múltiplas combinações** de hiperparâmetros
- 💾 **Salva** o campeão em `models/NVDA_{model_id}_{timestamp}.keras`
- 📊 **Registra no banco** em DUAS tabelas:
  - `ModelRegistry`: informações técnicas do modelo
  - `ResultadoMetricas`: métricas que aparecem na tela principal

#### 2️⃣ **Na Tela Principal (`/`)**:

Quando você acessa a home, o sistema:
```python
# app/routes/web.py (linha 12)
latest = ResultadoMetricas.query.filter_by(ticker='NVDA')\
    .order_by(ResultadoMetricas.trained_at.desc()).first()
```

**Traduzindo**: Busca o registro MAIS RECENTE de `ResultadoMetricas` para NVDA.

Como o Treino Avançado **também salva em ResultadoMetricas**, o modelo aparece automaticamente!

---

## 📋 O que aparece na tela principal:

Quando você treina um modelo avançado, estas informações são salvas:

```json
{
  "ticker": "NVDA",
  "model_version": "20251111_235959",  // timestamp do treino
  "horizon": 1,
  "mae": 2.34,       // ⬅️ Estas métricas aparecem na tela
  "rmse": 3.12,      // ⬅️
  "mape": 1.89,      // ⬅️
  "trained_at": "2025-11-11 23:59:59"
}
```

---

## 🔄 Fluxo Completo:

### Treino Avançado (modo optimized):
```
1. Usuário clica "Iniciar Treino Avançado"
   ↓
2. Sistema testa 30 modelos × N tentativas (ex: 20)
   = 600 treinos!
   ↓
3. Encontra campeão: Modelo 26 (Self-Attention LSTM)
   - learning_rate: 0.001
   - batch_size: 64
   - activation: swish
   - RMSE: 0.0187 (melhor!)
   ↓
4. Re-treina campeão com melhores parâmetros
   ↓
5. SALVA:
   ✅ models/NVDA_26_20251111_235959.keras
   ✅ models/NVDA_26_20251111_235959.scaler
   ✅ models/NVDA_26_20251111_235959.json (metadata)
   ↓
6. REGISTRA NO BANCO:
   ✅ ModelRegistry (is_winner=True)
   ✅ ResultadoMetricas (trained_at=agora) ⬅️ APARECE NA TELA!
```

### Tela Principal:
```
1. Usuário acessa http://localhost:5000/
   ↓
2. Sistema busca: 
   SELECT * FROM resultado_metricas 
   WHERE ticker='NVDA' 
   ORDER BY trained_at DESC 
   LIMIT 1
   ↓
3. Retorna: Modelo treinado há 2 minutos
   - MAE: 2.34
   - RMSE: 3.12
   - MAPE: 1.89%
   ↓
4. EXIBE no card "Modelo Vencedor Atual"
```

---

## 🆚 Comparação: Treino Normal vs Avançado

| Aspecto | Treino Normal (Botão "Treinar") | Treino Avançado |
|---------|--------------------------------|-----------------|
| **Modelos testados** | 10 fixos | **30 arquiteturas** |
| **Otimização** | ❌ Não (1 epoch fixo) | ✅ **Busca de hiperparâmetros** |
| **Tempo** | ~2 minutos | 5-60 minutos |
| **Ativações** | Padrão (relu/tanh) | **15 funções** testadas |
| **Salva em ResultadoMetricas** | ✅ Sim | ✅ **Sim** (agora!) |
| **Aparece na tela principal** | ✅ Sim | ✅ **Sim** |
| **is_winner no ModelRegistry** | ✅ Sim | ✅ **Sim** |

---

## ✅ Resposta Final:

### **SIM!** 

Quando você roda o Treino Avançado e ele encontra o melhor modelo:

1. ✅ O modelo é **salvo no disco** (`models/`)
2. ✅ É **registrado no banco** (`ModelRegistry` + `ResultadoMetricas`)
3. ✅ Fica marcado como **vencedor** (`is_winner=True`)
4. ✅ **APARECE NA TELA PRINCIPAL** assim que você acessar `/`

A tela principal **sempre mostra o último modelo treinado** (ordem por `trained_at DESC`).

Se você treinar:
- 10h00: Treino normal → aparece na tela
- 11h00: Treino avançado → **substitui** o anterior na tela
- 12h00: Treino normal → substitui o avançado

**Sempre o mais recente aparece!**

---

## 🎯 Exemplo Prático:

```bash
# Antes do treino avançado
Tela principal mostra: Modelo 3 (LSTM Stacked) - RMSE: 0.0234

# Você roda treino avançado (30 min depois)
Treino avançado encontra: Modelo 26 (Self-Attention) - RMSE: 0.0187

# Após o treino avançado
Tela principal mostra: Modelo 26 (Self-Attention) - RMSE: 0.0187 ⬅️ NOVO!
```

**O modelo campeão do Treino Avançado se torna o novo vencedor global do sistema!** 🏆
