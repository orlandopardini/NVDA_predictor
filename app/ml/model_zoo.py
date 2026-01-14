"""
Model Zoo - Catálogo de 10 arquiteturas LSTM/GRU pré-definidas.

Este módulo implementa o Factory Pattern para criação de modelos,
oferecendo 10 arquiteturas diferentes para comparação de performance.

Arquiteturas disponíveis:
1. LSTM Base (64/32) - Arquitetura clássica
2. LSTM Skip (64/64) - Com skip connections
3. BiLSTM (64/32) - Bidirecional
4. Stacked LSTM (96/64/32) - 3 camadas empilhadas
5. LSTM Deep (128/64/32/16) - 4 camadas profundas
6. GRU Base (64/32) - Alternativa ao LSTM
7. BiGRU (64/32) - GRU bidirecional
8. LSTM Wide (256/128) - Camadas largas
9. LSTM Attention-like - TimeDistributed
10. LSTM Hybrid (96/96/48) - BatchNorm + LayerNorm

Princípios aplicados:
- Factory Pattern para criação de objetos
- Type hints para segurança de tipos
- Funções pequenas e especializadas (SRP)
- Documentação completa
"""

from typing import Tuple, Dict, Any, Callable, Union
from tensorflow import keras
from tensorflow.keras import layers, Model, Sequential
from tensorflow.keras.optimizers import Optimizer, Adam, RMSprop, SGD, Adamax, Nadam


# ==================== ARQUITETURAS INDIVIDUAIS ====================

def _build_lstm_base(input_shape: Tuple[int, int]) -> Sequential:
    """
    Modelo 1: LSTM Base (64/32).
    
    Arquitetura clássica de duas camadas LSTM com dropout.
    Boa baseline para comparação.
    
    Args:
        input_shape: Tupla (timesteps, features)
        
    Returns:
        Modelo Keras compilado
    """
    model = Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(32),
        layers.Dense(16, activation='relu'),
        layers.Dense(1)
    ])
    return model


def _build_lstm_skip(input_shape: Tuple[int, int]) -> Model:
    """
    Modelo 2: LSTM com skip connection.
    
    Implementa conexão residual entre camadas para
    facilitar gradiente e melhorar treinamento.
    
    Args:
        input_shape: Tupla (timesteps, features)
        
    Returns:
        Modelo Keras funcional compilado
    """
    x_in = layers.Input(shape=input_shape)
    x = layers.LSTM(64, return_sequences=True)(x_in)
    x = layers.LSTM(64)(x)
    h = layers.Dense(32, activation='relu')(x)
    out = layers.Dense(1)(layers.Concatenate()([x, h]))
    
    model = Model(inputs=x_in, outputs=out)
    return model


def _build_bilstm(input_shape: Tuple[int, int]) -> Sequential:
    """
    Modelo 3: Bidirectional LSTM (64/32).
    
    Processa sequência em ambas as direções,
    capturando padrões forward e backward.
    
    Args:
        input_shape: Tupla (timesteps, features)
        
    Returns:
        Modelo Keras compilado
    """
    model = Sequential([
        layers.Input(shape=input_shape),
        layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
        layers.Dropout(0.3),
        layers.Bidirectional(layers.LSTM(32)),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)
    ])
    return model


def _build_stacked_lstm(input_shape: Tuple[int, int]) -> Sequential:
    """
    Modelo 4: Stacked LSTM (96/64/32).
    
    Três camadas LSTM empilhadas com LayerNormalization
    para estabilizar treinamento profundo.
    
    Args:
        input_shape: Tupla (timesteps, features)
        
    Returns:
        Modelo Keras compilado
    """
    model = Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(96, return_sequences=True),
        layers.LayerNormalization(),
        layers.Dropout(0.3),
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(32),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)
    ])
    return model


def _build_lstm_deep(input_shape: Tuple[int, int]) -> Sequential:
    """
    Modelo 5: LSTM Deep (128/64/32/16).
    
    Arquitetura profunda com 4 camadas LSTM.
    Maior capacidade de representação.
    
    Args:
        input_shape: Tupla (timesteps, features)
        
    Returns:
        Modelo Keras compilado
    """
    model = Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(128, return_sequences=True),
        layers.Dropout(0.25),
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(32, return_sequences=True),
        layers.LSTM(16),
        layers.Dense(1)
    ])
    return model


def _build_gru_base(input_shape: Tuple[int, int]) -> Sequential:
    """
    Modelo 6: GRU Base (64/32).
    
    Alternativa ao LSTM com menos parâmetros.
    Geralmente mais rápido para treinar.
    
    Args:
        input_shape: Tupla (timesteps, features)
        
    Returns:
        Modelo Keras compilado
    """
    model = Sequential([
        layers.Input(shape=input_shape),
        layers.GRU(64, return_sequences=True),
        layers.Dropout(0.2),
        layers.GRU(32),
        layers.Dense(16, activation='relu'),
        layers.Dense(1)
    ])
    return model


def _build_bigru(input_shape: Tuple[int, int]) -> Sequential:
    """
    Modelo 7: Bidirectional GRU.
    
    GRU bidirecional para capturar padrões
    em ambas as direções temporais.
    
    Args:
        input_shape: Tupla (timesteps, features)
        
    Returns:
        Modelo Keras compilado
    """
    model = Sequential([
        layers.Input(shape=input_shape),
        layers.Bidirectional(layers.GRU(64, return_sequences=True)),
        layers.Dropout(0.3),
        layers.Bidirectional(layers.GRU(32)),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)
    ])
    return model


def _build_lstm_wide(input_shape: Tuple[int, int]) -> Sequential:
    """
    Modelo 8: LSTM Wide (256/128).
    
    Camadas mais largas para maior capacidade.
    Requer mais dados e tempo de treinamento.
    
    Args:
        input_shape: Tupla (timesteps, features)
        
    Returns:
        Modelo Keras compilado
    """
    model = Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(256, return_sequences=True),
        layers.Dropout(0.3),
        layers.LSTM(128),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(1)
    ])
    return model


def _build_lstm_attention_like(input_shape: Tuple[int, int]) -> Model:
    """
    Modelo 9: LSTM com Attention-like.
    
    Usa TimeDistributed para criar mecanismo
    similar a attention sobre sequências.
    
    Args:
        input_shape: Tupla (timesteps, features)
        
    Returns:
        Modelo Keras funcional compilado
    """
    x_in = layers.Input(shape=input_shape)
    x = layers.LSTM(64, return_sequences=True)(x_in)
    x = layers.TimeDistributed(layers.Dense(64))(x)
    x = layers.LSTM(64)(x)
    x = layers.Dense(32, activation='relu')(x)
    out = layers.Dense(1)(x)
    
    model = Model(inputs=x_in, outputs=out)
    return model


def _build_lstm_hybrid(input_shape: Tuple[int, int]) -> Sequential:
    """
    Modelo 10: LSTM Hybrid (96/96/48).
    
    Combina BatchNormalization e múltiplas camadas.
    Híbrido de técnicas de normalização.
    
    Args:
        input_shape: Tupla (timesteps, features)
        
    Returns:
        Modelo Keras compilado
    """
    model = Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(96, return_sequences=True),
        layers.BatchNormalization(),
        layers.Dropout(0.25),
        layers.LSTM(96, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(48),
        layers.Dense(24, activation='relu'),
        layers.Dense(1)
    ])
    return model


# ==================== FACTORY PATTERN ====================

# Mapeamento de ID para função construtora
_MODEL_BUILDERS: Dict[int, Callable[[Tuple[int, int]], Union[Sequential, Model]]] = {
    1: _build_lstm_base,
    2: _build_lstm_skip,
    3: _build_bilstm,
    4: _build_stacked_lstm,
    5: _build_lstm_deep,
    6: _build_gru_base,
    7: _build_bigru,
    8: _build_lstm_wide,
    9: _build_lstm_attention_like,
    10: _build_lstm_hybrid,
}

# Nomes descritivos dos modelos
MODEL_NAMES: Dict[int, str] = {
    1: "LSTM Base (64/32)",
    2: "LSTM Skip (64/64)",
    3: "BiLSTM (64/32)",
    4: "Stacked LSTM (96/64/32)",
    5: "LSTM Deep (128/64/32/16)",
    6: "GRU Base (64/32)",
    7: "BiGRU (64/32)",
    8: "LSTM Wide (256/128)",
    9: "LSTM Attention-like",
    10: "LSTM Hybrid (96/96/48)",
}


def build_model(
    model_id: int,
    input_shape: Tuple[int, int]
) -> Union[Sequential, Model]:
    """
    Factory Method: Cria modelo baseado no ID.
    
    Implementa o Factory Pattern para centralizar criação
    de modelos. Todos os modelos são compilados com:
        - Optimizer: Adam
        - Loss: MSE (Mean Squared Error)
        - Metrics: MAE (Mean Absolute Error)
    
    Args:
        model_id: Identificador do modelo (1-10)
        input_shape: Tupla (timesteps, features), ex: (60, 1)
        
    Returns:
        Modelo Keras compilado e pronto para treinar
        
    Raises:
        ValueError: Se model_id não estiver entre 1-10
        
    Example:
        >>> model = build_model(model_id=1, input_shape=(60, 1))
        >>> model.fit(X_train, y_train, epochs=50)
    """
    if model_id not in _MODEL_BUILDERS:
        raise ValueError(
            f"model_id deve ser entre 1-10. Recebido: {model_id}. "
            f"IDs disponíveis: {list(_MODEL_BUILDERS.keys())}"
        )
    
    # Usar factory para construir modelo
    builder_function = _MODEL_BUILDERS[model_id]
    model = builder_function(input_shape)
    
    # Compilar modelo com configuração padrão
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )
    
    return model


# ==================== MODELO CUSTOMIZADO ====================

def _create_recurrent_layer(
    layer_cfg: Dict[str, Any]
) -> Union[layers.Layer, layers.Bidirectional]:
    """
    Cria camada recorrente (LSTM ou GRU) baseada em configuração.
    
    Args:
        layer_cfg: Dicionário com configurações da camada
        
    Returns:
        Camada Keras (pode ser envolta em Bidirectional)
    """
    layer_type = layer_cfg.get("type")
    units = layer_cfg.get("units", 64)
    return_seq = layer_cfg.get("return_sequences", False)
    dropout_rate = layer_cfg.get("dropout", 0.0)
    bidirectional = layer_cfg.get("bidirectional", False)
    
    # Criar camada base
    if layer_type == "LSTM":
        base_layer = layers.LSTM(
            units,
            return_sequences=return_seq,
            dropout=dropout_rate
        )
    elif layer_type == "GRU":
        base_layer = layers.GRU(
            units,
            return_sequences=return_seq,
            dropout=dropout_rate
        )
    else:
        raise ValueError(f"Tipo de camada recorrente inválido: {layer_type}")
    
    # Aplicar Bidirectional se solicitado
    if bidirectional:
        return layers.Bidirectional(base_layer)
    
    return base_layer


def _create_layer_from_config(layer_cfg: Dict[str, Any]) -> layers.Layer:
    """
    Factory para criar camada Keras baseada em configuração.
    
    Args:
        layer_cfg: Dicionário com tipo e parâmetros da camada
        
    Returns:
        Camada Keras configurada
        
    Raises:
        ValueError: Se tipo de camada não for suportado
    """
    layer_type = layer_cfg.get("type")
    
    # Camadas recorrentes
    if layer_type in ("LSTM", "GRU"):
        return _create_recurrent_layer(layer_cfg)
    
    # Camadas densas
    elif layer_type == "Dense":
        units = layer_cfg.get("units", 32)
        activation = layer_cfg.get("activation", None)
        return layers.Dense(units, activation=activation)
    
    # Regularização
    elif layer_type == "Dropout":
        rate = layer_cfg.get("rate", 0.2)
        return layers.Dropout(rate)
    
    # Normalização
    elif layer_type == "LayerNorm":
        return layers.LayerNormalization()
    
    elif layer_type == "BatchNorm":
        return layers.BatchNormalization()
    
    else:
        raise ValueError(f"Tipo de camada não suportado: {layer_type}")


def _create_optimizer(opt_config: Dict[str, Any]) -> Optimizer:
    """
    Cria otimizador Keras baseado em configuração.
    
    Args:
        opt_config: Dicionário com nome e parâmetros do otimizador
        
    Returns:
        Otimizador Keras configurado
    """
    opt_name = opt_config.get("name", "Adam")
    lr = opt_config.get("learning_rate", 0.001)
    
    if opt_name == "Adam":
        return Adam(learning_rate=lr)
    
    elif opt_name == "RMSprop":
        return RMSprop(learning_rate=lr)
    
    elif opt_name == "SGD":
        momentum = opt_config.get("momentum", 0.9)
        return SGD(learning_rate=lr, momentum=momentum)
    
    elif opt_name == "Adamax":
        return Adamax(learning_rate=lr)
    
    elif opt_name == "Nadam":
        return Nadam(learning_rate=lr)
    
    else:
        # Fallback para Adam se nome inválido
        return "adam"


def build_custom_model(
    config: Dict[str, Any],
    input_shape: Tuple[int, int]
) -> Sequential:
    """
    Constrói modelo LSTM totalmente personalizado baseado em configuração.
    
    Permite criação dinâmica de arquiteturas complexas via JSON/dict.
    Suporta LSTM, GRU, Dense, Dropout, normalização e diversos otimizadores.
    
    Args:
        config: Dicionário de configuração com estrutura:
            {
                "layers": [
                    {"type": "LSTM", "units": 64, "return_sequences": True, 
                     "bidirectional": False, "dropout": 0.2},
                    {"type": "LayerNorm"},
                    {"type": "Dense", "units": 16, "activation": "relu"},
                    {"type": "Dropout", "rate": 0.3},
                    {"type": "Dense", "units": 1}
                ],
                "optimizer": {"name": "Adam", "learning_rate": 0.001},
                "loss": "mse"
            }
        input_shape: Tupla (timesteps, features)
        
    Returns:
        Modelo Keras compilado
        
    Raises:
        ValueError: Se configuração for inválida
        
    Example:
        >>> config = {
        ...     "layers": [
        ...         {"type": "LSTM", "units": 64, "return_sequences": True},
        ...         {"type": "LSTM", "units": 32},
        ...         {"type": "Dense", "units": 1}
        ...     ],
        ...     "optimizer": {"name": "Adam", "learning_rate": 0.001},
        ...     "loss": "mse"
        ... }
        >>> model = build_custom_model(config, input_shape=(60, 1))
    """
    # Validação básica
    if "layers" not in config:
        raise ValueError("Configuração deve conter chave 'layers'")
    
    # Construir lista de camadas
    model_layers = [layers.Input(shape=input_shape)]
    
    for layer_cfg in config["layers"]:
        layer = _create_layer_from_config(layer_cfg)
        model_layers.append(layer)
    
    # Criar modelo sequencial
    model = Sequential(model_layers)
    
    # Configurar otimizador
    opt_config = config.get("optimizer", {"name": "Adam", "learning_rate": 0.001})
    optimizer = _create_optimizer(opt_config)
    
    # Compilar modelo
    loss = config.get("loss", "mse")
    model.compile(optimizer=optimizer, loss=loss, metrics=['mae'])
    
    return model
