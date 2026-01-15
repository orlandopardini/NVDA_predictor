# Scripts de Instalação e Inicialização

Esta pasta contém scripts para facilitar a configuração e uso do projeto.

## Scripts Disponíveis

### `setup.bat`
Configura o ambiente pela primeira vez:
- Instala Python 3.12 (se necessário)
- Cria ambiente virtual
- Instala dependências
- Inicializa banco de dados

**Uso:** Execute uma vez na primeira instalação
```bash
setup.bat
```

### `start.bat`
Inicia o servidor Flask

**Uso:** Execute sempre que quiser usar a aplicação
```bash
start.bat
```

Acesse: http://127.0.0.1:5000
