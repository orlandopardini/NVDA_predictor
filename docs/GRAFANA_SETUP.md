# Guia de Configuração: Grafana + Prometheus

## 1. Pré-requisitos
- Docker Desktop instalado e rodando
- Aplicação Flask rodando na porta 5000

## 2. Iniciar Prometheus e Grafana

Abra o PowerShell na pasta do projeto e execute:

```powershell
docker-compose up -d
```

Isso iniciará:
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000

## 3. Verificar Prometheus

1. Acesse http://localhost:9090
2. Vá em Status > Targets
3. Verifique se `flask-app` está com status "UP"
4. Se estiver "DOWN", certifique-se que sua aplicação Flask está rodando

## 4. Configurar Grafana

### 4.1 Login Inicial
1. Acesse http://localhost:3000
2. Login: `admin`
3. Senha: `admin`
4. (Pode pular a troca de senha)

### 4.2 Adicionar Data Source
1. Menu lateral > ⚙️ Configuration > Data Sources
2. Clique "Add data source"
3. Selecione "Prometheus"
4. Configure:
   - **Name**: Prometheus
   - **URL**: http://prometheus:9090
5. Clique "Save & Test" (deve aparecer "Data source is working")

### 4.3 Criar Dashboard

1. Menu lateral > ➕ Create > Dashboard
2. Clique "Add new panel"

**Painel 1: Total de Requisições HTTP**
- Metric: `http_requests`
- Legend: `{{method}} {{endpoint}}`

**Painel 2: Latência das Requisições**
- Metric: `rate(http_request_duration_seconds_sum[5m]) / rate(http_request_duration_seconds_count[5m])`
- Legend: `{{endpoint}}`

**Painel 3: Requisições em Progresso**
- Metric: `http_requests_in_progress`
- Legend: `{{endpoint}}`

**Painel 4: Tempo de Inferência do Modelo**
- Metric: `inference_seconds`
- Legend: `{{ticker}} v{{version}}`

**Painel 5: Total de Retreinagens**
- Metric: `retrain_total`
- Legend: `{{ticker}} - {{mode}}`

**Painel 6: Duração dos Treinos**
- Metric: `retrain_duration_seconds`
- Legend: `{{ticker}} - {{mode}}`

3. Salve o dashboard com nome "Stock LSTM Metrics"

## 5. Atualizar Links no Sistema

Os links já foram atualizados:
- **Grafana**: http://localhost:3000
- **Prometheus**: http://localhost:9090

## 6. Comandos Úteis

```powershell
# Iniciar serviços
docker-compose up -d

# Ver logs
docker-compose logs -f

# Parar serviços
docker-compose down

# Parar e remover volumes (reset completo)
docker-compose down -v
```

## 7. Troubleshooting

**Prometheus não consegue coletar métricas da aplicação:**
- Verifique se a aplicação Flask está rodando em http://localhost:5000
- Teste manualmente: abra http://localhost:5000/metrics no navegador
- No Windows, `host.docker.internal` resolve para o host

**Grafana não conecta no Prometheus:**
- Use `http://prometheus:9090` como URL (nome do serviço Docker)
- NÃO use `localhost:9090` dentro do container

**Porta já está em uso:**
- Altere as portas no `docker-compose.yml`:
  - Prometheus: `"9091:9090"` (acesse via 9091)
  - Grafana: `"3001:3000"` (acesse via 3001)

## 8. Dashboard Pronto (Opcional)

Para importar um dashboard pronto:
1. Copie o JSON do arquivo `grafana-dashboard.json` (se criado)
2. Grafana > ➕ Create > Import
3. Cole o JSON e clique "Load"
