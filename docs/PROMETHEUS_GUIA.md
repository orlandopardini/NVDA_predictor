# Prometheus - Guia Rápido

## ✅ O que já está configurado:

### **1. Coleta Automática de Métricas**
- ✅ URL: http://localhost:9090
- ✅ Target: `flask-app` apontando para `host.docker.internal:5000/metrics`
- ✅ Intervalo: 15 segundos
- ✅ Status: Monitorando sua aplicação Flask

### **2. Como Verificar se Está Funcionando**

#### Opção A: Interface Web
1. Acesse: http://localhost:9090
2. Vá em **Status** → **Targets**
3. Deve mostrar `flask-app` com status **UP** (verde)

#### Opção B: PowerShell
```powershell
curl http://localhost:9090/api/v1/targets
```

---

## 📊 Métricas Disponíveis no Prometheus

Acesse http://localhost:9090/graph e teste estas queries:

### **1. Taxa de Requisições HTTP**
```promql
rate(http_requests_total[5m])
```

### **2. Latência Média das Requisições**
```promql
rate(http_request_duration_seconds_sum[5m]) / rate(http_request_duration_seconds_count[5m])
```

### **3. Requisições em Andamento**
```promql
http_requests_in_progress
```

### **4. Tempo de Inferência do Modelo**
```promql
rate(inference_seconds_sum[5m]) / rate(inference_seconds_count[5m])
```

### **5. Total de Retreinagens**
```promql
retrain_total
```

### **6. Duração dos Treinos**
```promql
rate(retrain_duration_seconds_sum[5m]) / rate(retrain_duration_seconds_count[5m])
```

---

## 🔍 Páginas Úteis do Prometheus

| Página | URL | Descrição |
|--------|-----|-----------|
| **Gráficos** | http://localhost:9090/graph | Execute queries PromQL |
| **Targets** | http://localhost:9090/targets | Status dos endpoints monitorados |
| **Service Discovery** | http://localhost:9090/service-discovery | Descoberta de serviços |
| **Config** | http://localhost:9090/config | Configuração atual |
| **Flags** | http://localhost:9090/flags | Flags de inicialização |
| **Status** | http://localhost:9090/status | Informações do sistema |

---

## 🎯 O Que o Prometheus Faz

1. **Coleta** métricas do endpoint `/metrics` da sua aplicação Flask a cada 15s
2. **Armazena** em banco de dados de séries temporais (TSDB)
3. **Disponibiliza** via API e interface web para o Grafana consumir

---

## 🔄 Fluxo Completo

```
Flask App (:5000/metrics)
    ↓
    [Exposição de métricas Prometheus]
    ↓
Prometheus (:9090)
    ↓
    [Coleta a cada 15s e armazena]
    ↓
Grafana (:3000)
    ↓
    [Consulta Prometheus e mostra dashboards]
```

---

## ⚙️ Configuração (prometheus.yml)

```yaml
global:
  scrape_interval: 15s      # Coleta a cada 15 segundos

scrape_configs:
  - job_name: 'flask-app'
    static_configs:
      - targets: ['host.docker.internal:5000']
        labels:
          app: 'stock-lstm-flask'
    metrics_path: '/metrics'
```

---

## 🚨 Troubleshooting

### **Target está DOWN (vermelho)**
1. Verifique se sua aplicação Flask está rodando: http://localhost:5000
2. Teste o endpoint de métricas: http://localhost:5000/metrics
3. Se estiver rodando, aguarde ~15 segundos

### **Sem dados nos gráficos**
- Use a aplicação Flask (faça requisições, treine modelos)
- As métricas aparecem conforme você usa o sistema

### **Recarregar configuração sem reiniciar**
```powershell
curl -X POST http://localhost:9090/-/reload
```

---

## 📝 Resumo

✅ **Prometheus**: JÁ está configurado e rodando  
✅ **Target flask-app**: JÁ está monitorando sua aplicação  
✅ **Grafana**: JÁ tem o datasource Prometheus configurado  
✅ **Dashboard**: JÁ está criado e funcional  

**Tudo portável e automático!** 🎉
