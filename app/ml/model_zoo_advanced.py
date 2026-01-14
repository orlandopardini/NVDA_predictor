# app/ml/model_zoo_advanced.py
"""
🎯 MODEL ZOO AVANÇADO - 30 Arquiteturas de Alto Desempenho
Inclui: LSTM, GRU, BiLSTM, BiGRU, Stacked, Residual, Attention, Hybrid
"""
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf

# ===== FUNÇÕES DE ATIVAÇÃO DISPONÍVEIS =====
ACTIVATION_FUNCTIONS = {
    # Básicas
    'relu': 'relu',
    'tanh': 'tanh',
    'sigmoid': 'sigmoid',
    'linear': 'linear',
    
    # Avançadas (Leaky/ELU família)
    'leaky_relu': layers.LeakyReLU(alpha=0.1),
    'elu': 'elu',
    'selu': 'selu',
    
    # Exponenciais
    'exponential': 'exponential',
    'softplus': 'softplus',
    'softsign': 'softsign',
    
    # Modernas
    'swish': 'swish',  # também conhecido como SiLU
    'mish': lambda x: x * tf.nn.tanh(tf.nn.softplus(x)),
    'gelu': 'gelu',
    
    # Hard variants
    'hard_sigmoid': 'hard_sigmoid',
    'hard_swish': lambda x: x * tf.nn.relu6(x + 3) / 6,
}

def get_activation(name='relu'):
    """Retorna função de ativação por nome"""
    if name in ACTIVATION_FUNCTIONS:
        act = ACTIVATION_FUNCTIONS[name]
        return act if isinstance(act, str) else layers.Activation(act)
    return 'relu'  # fallback


# ============================================================================
# BUILDERS INDIVIDUAIS: LSTM BASE & VARIANTS (1-5)
# ============================================================================

def _build_lstm_classic(input_shape, act, dropout_rate: float) -> keras.Model:
    """
    Model 1: LSTM Clássico (64/32).
    
    Arquitetura LSTM clássica com duas camadas recorrentes (64→32 unidades).
    Usa dropout para regularização. Ideal como baseline robusto para séries temporais.
    """
    return keras.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(dropout_rate),
        layers.LSTM(32),
        layers.Dense(16, activation=act),
        layers.Dense(1)
    ], name="LSTM_Classic")


def _build_lstm_layer_norm(input_shape, act, dropout_rate: float) -> keras.Model:
    """
    Model 2: LSTM com Layer Normalization.
    
    LSTM com Layer Normalization após cada camada recorrente. Estabiliza o treinamento
    e acelera convergência, especialmente útil para séries com mudanças de escala.
    """
    return keras.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(80, return_sequences=True),
        layers.LayerNormalization(),
        layers.Dropout(dropout_rate),
        layers.LSTM(40),
        layers.LayerNormalization(),
        layers.Dense(20, activation=act),
        layers.Dense(1)
    ], name="LSTM_LayerNorm")


def _build_lstm_batch_norm(input_shape, act, dropout_rate: float) -> keras.Model:
    """
    Model 3: LSTM com Batch Normalization.
    
    LSTM com Batch Normalization, que normaliza ativações em mini-batches. Dropout aumentado (1.2x)
    para compensar o efeito regularizador do BatchNorm. Ótimo para dados com alta variância.
    """
    return keras.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(96, return_sequences=True),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate * 1.2),
        layers.LSTM(48),
        layers.BatchNormalization(),
        layers.Dense(24, activation=act),
        layers.Dense(1)
    ], name="LSTM_BatchNorm")


def _build_lstm_narrow_deep(input_shape, act, dropout_rate: float) -> keras.Model:
    """
    Model 4: LSTM Narrow-Deep (32³).
    
    Arquitetura narrow-deep: 3 camadas LSTM de 32 unidades cada. Processa informação
    em múltiplos níveis hierárquicos. Bom para capturar padrões complexos com poucos parâmetros.
    """
    return keras.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(32, return_sequences=True),
        layers.Dropout(dropout_rate),
        layers.LSTM(32, return_sequences=True),
        layers.Dropout(dropout_rate),
        layers.LSTM(32),
        layers.Dense(16, activation=act),
        layers.Dense(1)
    ], name="LSTM_NarrowDeep")


def _build_lstm_wide_shallow(input_shape, act, dropout_rate: float) -> keras.Model:
    """
    Model 5: LSTM Wide-Shallow (256).
    
    LSTM wide-shallow com uma única camada de 256 unidades. Alta capacidade representacional
    em camada única. Dropout elevado (1.5x) evita overfitting. Rápido em treinamento.
    """
    return keras.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(256),
        layers.Dropout(dropout_rate * 1.5),
        layers.Dense(128, activation=act),
        layers.Dropout(dropout_rate),
        layers.Dense(1)
    ], name="LSTM_WideShallow")


# ============================================================================
# BUILDERS INDIVIDUAIS: GRU BASE & VARIANTS (6-10)
# ============================================================================

def _build_gru_classic(input_shape, act, dropout_rate: float) -> keras.Model:
    """
    Model 6: GRU Clássico (64/32).
    
    GRU clássico (64→32). Mais eficiente que LSTM (menos parâmetros, 2 gates vs 3).
    Excelente para séries com memória de curto/médio prazo. Treina mais rápido que LSTM equivalente.
    """
    return keras.Sequential([
        layers.Input(shape=input_shape),
        layers.GRU(64, return_sequences=True),
        layers.Dropout(dropout_rate),
        layers.GRU(32),
        layers.Dense(16, activation=act),
        layers.Dense(1)
    ], name="GRU_Classic")


def _build_gru_deep(input_shape, act, dropout_rate: float) -> keras.Model:
    """
    Model 7: GRU Deep (128/64/32).
    
    GRU profundo com 3 camadas (128→64→32) e Layer Normalization. Captura hierarquias
    temporais complexas. Dropout progressivo para regularização gradual.
    """
    return keras.Sequential([
        layers.Input(shape=input_shape),
        layers.GRU(128, return_sequences=True),
        layers.LayerNormalization(),
        layers.Dropout(dropout_rate * 1.2),
        layers.GRU(64, return_sequences=True),
        layers.Dropout(dropout_rate),
        layers.GRU(32),
        layers.Dense(16, activation=act),
        layers.Dense(1)
    ], name="GRU_Deep")


def _build_gru_wide(input_shape, act, dropout_rate: float) -> keras.Model:
    """
    Model 8: GRU Wide (192/96).
    
    GRU largo com 2 camadas (192→96). Alta capacidade de memória. Ideal para séries
    com muitas features ou padrões intrincados. Requer mais dados para treinar bem.
    """
    return keras.Sequential([
        layers.Input(shape=input_shape),
        layers.GRU(192, return_sequences=True),
        layers.Dropout(dropout_rate * 1.3),
        layers.GRU(96),
        layers.Dense(48, activation=act),
        layers.Dense(1)
    ], name="GRU_Wide")


def _build_gru_residual_dense(input_shape, act, dropout_rate: float) -> keras.Model:
    """
    Model 9: GRU com Residual Dense.
    
    GRU com conexões residuais densas. Skip connections permitem gradiente fluir diretamente.
    Reduz vanishing gradient. Camada Dense final integra múltiplas resoluções temporais.
    """
    x_in = layers.Input(shape=input_shape)
    x = layers.GRU(64, return_sequences=True)(x_in)
    x = layers.GRU(64)(x)
    h1 = layers.Dense(32, activation=act)(x)
    h2 = layers.Dense(32, activation=act)(layers.Concatenate()([x, h1]))
    out = layers.Dense(1)(h2)
    return keras.Model(x_in, out, name="GRU_ResidualDense")


def _build_gru_hybrid(input_shape, act, dropout_rate: float) -> keras.Model:
    """
    Model 10: GRU Hybrid (80/80/40).
    
    GRU híbrido (80→80→40) combinando camadas paralelas e sequenciais. Processa informação
    em diferentes escalas simultaneamente. Boa generalização em diversos tipos de séries.
    """
    return keras.Sequential([
        layers.Input(shape=input_shape),
        layers.GRU(80, return_sequences=True),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate),
        layers.GRU(80, return_sequences=True),
        layers.Dropout(dropout_rate),
        layers.GRU(40),
        layers.Dense(20, activation=act),
        layers.Dense(1)
    ], name="GRU_Hybrid")


# ============================================================================
# BUILDERS INDIVIDUAIS: BIDIRECTIONAL (11-15)
# ============================================================================

def _build_bilstm_classic(input_shape, act, dropout_rate: float) -> keras.Model:
    """
    Model 11: BiLSTM Classic (64/32).
    
    BiLSTM (Bidirectional LSTM) clássico. Processa sequência em ambas direções (passado→futuro
    e futuro→passado). Captura dependências bidirecionais. Dobra parâmetros vs LSTM unidirecional.
    """
    return keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
        layers.Dropout(dropout_rate * 1.5),
        layers.Bidirectional(layers.LSTM(32)),
        layers.Dense(32, activation=act),
        layers.Dense(1)
    ], name="BiLSTM_Classic")


def _build_bigru_classic(input_shape, act, dropout_rate: float) -> keras.Model:
    """
    Model 12: BiGRU Classic (64/32).
    
    BiGRU clássico. Versão bidirecional do GRU. Mais eficiente que BiLSTM, mantendo poder
    expressivo. Ideal quando contexto futuro é informativamente relevante para previsão.
    """
    return keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Bidirectional(layers.GRU(64, return_sequences=True)),
        layers.Dropout(dropout_rate * 1.5),
        layers.Bidirectional(layers.GRU(32)),
        layers.Dense(32, activation=act),
        layers.Dense(1)
    ], name="BiGRU_Classic")


def _build_bilstm_deep(input_shape, act, dropout_rate: float) -> keras.Model:
    """
    Model 13: BiLSTM Deep (96/64/32).
    
    BiLSTM profundo (3 camadas: 96→64→32). Múltiplos níveis de abstração bidirecional.
    Layer Normalization estabiliza camadas profundas. Excelente para padrões temporais complexos não-lineares.
    """
    return keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Bidirectional(layers.LSTM(96, return_sequences=True)),
        layers.LayerNormalization(),
        layers.Dropout(dropout_rate * 1.3),
        layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
        layers.Dropout(dropout_rate),
        layers.Bidirectional(layers.LSTM(32)),
        layers.Dense(48, activation=act),
        layers.Dense(1)
    ], name="BiLSTM_Deep")


def _build_bigru_deep(input_shape, act, dropout_rate: float) -> keras.Model:
    """
    Model 14: BiGRU Deep (96/64/32).
    
    BiGRU profundo (3 camadas: 96→64→32). Versão GRU do BiLSTM Deep. Treinamento mais rápido
    com eficácia similar. Bom balanço entre performance e custo computacional.
    """
    return keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Bidirectional(layers.GRU(96, return_sequences=True)),
        layers.LayerNormalization(),
        layers.Dropout(dropout_rate * 1.3),
        layers.Bidirectional(layers.GRU(64, return_sequences=True)),
        layers.Dropout(dropout_rate),
        layers.Bidirectional(layers.GRU(32)),
        layers.Dense(48, activation=act),
        layers.Dense(1)
    ], name="BiGRU_Deep")


def _build_bilstm_bigru_mix(input_shape, act, dropout_rate: float) -> keras.Model:
    """
    Model 15: BiLSTM+BiGRU Mix.
    
    Arquitetura mista: BiLSTM seguido de BiGRU. Combina força de ambos: LSTM captura
    dependências longas, GRU refina com eficiência. Dropout entre transições reduz co-adaptação.
    """
    return keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
        layers.Dropout(dropout_rate),
        layers.Bidirectional(layers.GRU(64)),
        layers.Dense(64, activation=act),
        layers.Dropout(dropout_rate),
        layers.Dense(1)
    ], name="BiLSTM_BiGRU_Mix")


# ============================================================================
# BUILDERS INDIVIDUAIS: STACKED DEEP NETWORKS (16-20)
# ============================================================================

def _build_stacked_lstm_deep(input_shape, act, dropout_rate: float) -> keras.Model:
    """
    Model 16: Stacked LSTM (128→96→64→32).
    
    LSTM empilhado com decaimento progressivo (128→96→64→32). Cada camada aprende representações
    de maior abstração. Pyramid stacking: entrada larga, saída focada. Excelente para dados complexos.
    """
    return keras.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(128, return_sequences=True),
        layers.LayerNormalization(),
        layers.Dropout(dropout_rate),
        layers.LSTM(96, return_sequences=True),
        layers.Dropout(dropout_rate),
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(dropout_rate),
        layers.LSTM(32),
        layers.Dense(16, activation=act),
        layers.Dense(1)
    ], name="Stacked_LSTM_Deep")


def _build_stacked_gru_deep(input_shape, act, dropout_rate: float) -> keras.Model:
    """
    Model 17: Stacked GRU (128→96→64→32).
    
    GRU empilhado com mesma estratégia de decaimento (128→96→64→32). Versão GRU do Stacked LSTM.
    Menos parâmetros, treinamento rápido. Boa escolha para produção com recursos limitados.
    """
    return keras.Sequential([
        layers.Input(shape=input_shape),
        layers.GRU(128, return_sequences=True),
        layers.LayerNormalization(),
        layers.Dropout(dropout_rate),
        layers.GRU(96, return_sequences=True),
        layers.Dropout(dropout_rate),
        layers.GRU(64, return_sequences=True),
        layers.Dropout(dropout_rate),
        layers.GRU(32),
        layers.Dense(16, activation=act),
        layers.Dense(1)
    ], name="Stacked_GRU_Deep")


def _build_pyramid_lstm(input_shape, act, dropout_rate: float) -> keras.Model:
    """
    Model 18: Pyramid LSTM (256→128→64→32→16).
    
    Pirâmide LSTM extrema (256→128→64→32→16). Processa informação em 5 níveis hierárquicos.
    Captura desde padrões locais até tendências globais. Requer muitos dados para evitar overfitting.
    """
    return keras.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(256, return_sequences=True),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate * 1.2),
        layers.LSTM(128, return_sequences=True),
        layers.Dropout(dropout_rate),
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(dropout_rate),
        layers.LSTM(32, return_sequences=True),
        layers.LSTM(16),
        layers.Dense(1)
    ], name="Pyramid_LSTM")


def _build_inverted_pyramid_lstm(input_shape, act, dropout_rate: float) -> keras.Model:
    """
    Model 19: Inverted Pyramid (32→64→128).
    
    Pirâmide invertida (32→64→128). Começa focado e expande representação. Útil quando entrada
    é compacta mas padrões subjacentes são complexos. Design contra-intuitivo mas eficaz.
    """
    return keras.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(32, return_sequences=True),
        layers.Dropout(dropout_rate),
        layers.LSTM(64, return_sequences=True),
        layers.LayerNormalization(),
        layers.Dropout(dropout_rate),
        layers.LSTM(128),
        layers.Dense(64, activation=act),
        layers.Dropout(dropout_rate),
        layers.Dense(1)
    ], name="InvertedPyramid_LSTM")


def _build_diamond_lstm(input_shape, act, dropout_rate: float) -> keras.Model:
    """
    Model 20: Diamond LSTM (64→128→128→64).
    
    Arquitetura diamante (64→128→128→64). Expande no meio para captura máxima, depois comprime.
    Balanceia foco local e contexto global. Dropout variável preserva informação crítica.
    """
    return keras.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(dropout_rate),
        layers.LSTM(128, return_sequences=True),
        layers.LayerNormalization(),
        layers.Dropout(dropout_rate),
        layers.LSTM(128, return_sequences=True),
        layers.Dropout(dropout_rate),
        layers.LSTM(64),
        layers.Dense(32, activation=act),
        layers.Dense(1)
    ], name="Diamond_LSTM")


# ============================================================================
# BUILDERS INDIVIDUAIS: RESIDUAL & SKIP CONNECTIONS (21-25)
# ============================================================================

def _build_lstm_residual_v1(input_shape, act, dropout_rate: float) -> keras.Model:
    """
    Model 21: LSTM Residual v1.
    
    LSTM com conexões residuais v1. Skip connections adicionam entrada diretamente à saída
    de camadas intermediárias. Facilita treinamento profundo. Reduz degradação de gradiente.
    """
    x_in = layers.Input(shape=input_shape)
    x = layers.LSTM(96, return_sequences=True)(x_in)
    x_res = layers.LSTM(96, return_sequences=True)(x)
    x = layers.Add()([x, x_res])
    x = layers.LSTM(48)(x)
    x = layers.Dense(24, activation=act)(x)
    out = layers.Dense(1)(x)
    return keras.Model(x_in, out, name="LSTM_Residual_v1")


def _build_lstm_residual_v2(input_shape, act, dropout_rate: float) -> keras.Model:
    """
    Model 22: LSTM Residual v2 (Multiple Shortcuts).
    
    LSTM residual v2 com múltiplas shortcuts. Implementa esquema ResNet para redes recorrentes.
    Dense final integra todas as resoluções. Treina redes muito profundas estavelmente.
    """
    x_in = layers.Input(shape=input_shape)
    x1 = layers.LSTM(64, return_sequences=True)(x_in)
    x2 = layers.LSTM(64, return_sequences=True)(x1)
    x = layers.Add()([x1, x2])
    x3 = layers.LSTM(64, return_sequences=True)(x)
    x = layers.Add()([x, x3])
    x = layers.LSTM(32)(x)
    x = layers.Dense(16, activation=act)(x)
    out = layers.Dense(1)(x)
    return keras.Model(x_in, out, name="LSTM_Residual_v2")


def _build_skip_dense(input_shape, act, dropout_rate: float) -> keras.Model:
    """
    Model 23: Skip Connection Dense.
    
    Dense Skip Connections: cada camada conecta a todas anteriores (DenseNet-style).
    Máxima reutilização de features. Concatenação preserva todas resoluções temporais.
    """
    x_in = layers.Input(shape=input_shape)
    x = layers.LSTM(80, return_sequences=True)(x_in)
    x = layers.LSTM(80)(x)
    h1 = layers.Dense(40, activation=act)(x)
    h2 = layers.Dense(40, activation=act)(h1)
    h3 = layers.Dense(40, activation=act)(layers.Concatenate()([h1, h2]))
    out = layers.Dense(1)(h3)
    return keras.Model(x_in, out, name="Skip_Dense")


def _build_highway_lstm(input_shape, act, dropout_rate: float) -> keras.Model:
    """
    Model 24: Highway LSTM.
    
    Highway LSTM: gates aprendidos controlam fluxo de informação através de shortcuts.
    Inspirado em Highway Networks. Modelo decide dinamicamente quando usar skip connections vs transformações.
    """
    x_in = layers.Input(shape=input_shape)
    x = layers.LSTM(96, return_sequences=True)(x_in)
    gate = layers.Dense(96, activation='sigmoid')(x)
    transform = layers.Dense(96, activation=act)(x)
    x = layers.Add()([
        layers.Multiply()([gate, transform]),
        layers.Multiply()([layers.Lambda(lambda g: 1-g)(gate), x])
    ])
    x = layers.LSTM(48)(x)
    out = layers.Dense(1)(x)
    return keras.Model(x_in, out, name="Highway_LSTM")


def _build_densenet_lstm(input_shape, act, dropout_rate: float) -> keras.Model:
    """
    Model 25: DenseNet-style LSTM.
    
    DenseNet-style LSTM: concatena outputs de todas camadas anteriores. Growth rate controlado.
    Feature reuse extremo. Excelente performance mas computacionalmente caro. Para datasets grandes.
    """
    x_in = layers.Input(shape=input_shape)
    x1 = layers.LSTM(48, return_sequences=True)(x_in)
    x2 = layers.LSTM(48, return_sequences=True)(x1)
    x_concat1 = layers.Concatenate()([x1, x2])
    x3 = layers.LSTM(48, return_sequences=True)(x_concat1)
    x_concat2 = layers.Concatenate()([x1, x2, x3])
    x = layers.LSTM(48)(x_concat2)
    x = layers.Dense(24, activation=act)(x)
    out = layers.Dense(1)(x)
    return keras.Model(x_in, out, name="DenseNet_LSTM")


# ============================================================================
# BUILDERS INDIVIDUAIS: ATTENTION & HYBRID MECHANISMS (26-30)
# ============================================================================

def _build_attention_lstm(input_shape, act, dropout_rate: float) -> keras.Model:
    """
    Model 26: Self-Attention LSTM.
    
    Self-Attention sobre LSTM. Attention layer aprende quais timesteps são mais relevantes.
    Pesos de atenção dinâmicos. Captura dependências não-locais. Interpretabilidade via attention weights.
    """
    x_in = layers.Input(shape=input_shape)
    x = layers.LSTM(64, return_sequences=True)(x_in)
    attention_weights = layers.Dense(1, activation='softmax')(x)
    x = layers.Multiply()([x, attention_weights])
    x = layers.LSTM(64)(x)
    x = layers.Dense(32, activation=act)(x)
    out = layers.Dense(1)(x)
    return keras.Model(x_in, out, name="Attention_LSTM")


def _build_multihead_attention(input_shape, act, dropout_rate: float) -> keras.Model:
    """
    Model 27: Multi-Head Attention.
    
    Multi-Head Attention (estilo Transformer) após LSTM. 3 cabeças de atenção capturam diferentes aspectos
    temporais simultaneamente. Concatena e projeta resultados. State-of-the-art para séries complexas.
    """
    x_in = layers.Input(shape=input_shape)
    x = layers.LSTM(96, return_sequences=True)(x_in)
    att1 = layers.Dense(32, activation='softmax')(x)
    att2 = layers.Dense(32, activation='softmax')(x)
    att3 = layers.Dense(32, activation='softmax')(x)
    x_att = layers.Concatenate()([
        layers.Multiply()([x, att1]),
        layers.Multiply()([x, att2]),
        layers.Multiply()([x, att3])
    ])
    x = layers.LSTM(96)(x_att)
    x = layers.Dense(48, activation=act)(x)
    out = layers.Dense(1)(x)
    return keras.Model(x_in, out, name="MultiHead_Attention")


def _build_cnn_lstm_hybrid(input_shape, act, dropout_rate: float) -> keras.Model:
    """
    Model 28: CNN+LSTM Hybrid.
    
    CNN+LSTM híbrido. Conv1D extrai features locais, LSTM captura dependências temporais.
    Combina força de ambas arquiteturas. MaxPooling reduz dimensionalidade. Excelente para sinais com padrões locais repetitivos.
    """
    x_in = layers.Input(shape=input_shape)
    x = layers.Conv1D(64, kernel_size=3, padding='same', activation=act)(x_in)
    x = layers.Conv1D(64, kernel_size=3, padding='same', activation=act)(x)
    x = layers.LSTM(64, return_sequences=True)(x)
    x = layers.LSTM(32)(x)
    x = layers.Dense(16, activation=act)(x)
    out = layers.Dense(1)(x)
    return keras.Model(x_in, out, name="CNN_LSTM_Hybrid")


def _build_lstm_timedistributed(input_shape, act, dropout_rate: float) -> keras.Model:
    """
    Model 29: LSTM TimeDistributed.
    
    LSTM com TimeDistributed Dense layers. Aplica mesma camada Dense a cada timestep independentemente.
    Útil para prediction windows. Compartilha pesos temporalmente. Reduz parâmetros vs Dense separadas.
    """
    x_in = layers.Input(shape=input_shape)
    x = layers.LSTM(96, return_sequences=True)(x_in)
    x = layers.TimeDistributed(layers.Dense(96, activation=act))(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.LSTM(96, return_sequences=True)(x)
    x = layers.TimeDistributed(layers.Dense(48, activation=act))(x)
    x = layers.LSTM(48)(x)
    out = layers.Dense(1)(x)
    return keras.Model(x_in, out, name="LSTM_TimeDistributed")


def _build_ensemble_multipath(input_shape, act, dropout_rate: float) -> keras.Model:
    """
    Model 30: Ensemble Multi-Path.
    
    Ensemble Multi-Path: múltiplas streams paralelas (LSTM, GRU, BiLSTM) processam entrada simultaneamente.
    Concatena outputs antes da predição. Wisdom of crowds. Robusto mas pesado. Melhor generalização.
    """
    x_in = layers.Input(shape=input_shape)
    # Path 1: LSTM
    path1 = layers.LSTM(64, return_sequences=True)(x_in)
    path1 = layers.LSTM(32)(path1)
    # Path 2: GRU
    path2 = layers.GRU(64, return_sequences=True)(x_in)
    path2 = layers.GRU(32)(path2)
    # Path 3: BiLSTM
    path3 = layers.Bidirectional(layers.LSTM(32))(x_in)
    # Combine
    x = layers.Concatenate()([path1, path2, path3])
    x = layers.Dense(96, activation=act)(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(48, activation=act)(x)
    out = layers.Dense(1)(x)
    return keras.Model(x_in, out, name="Ensemble_MultiPath")


def build_advanced_model(model_id: int, input_shape, activation='relu', dropout_rate=0.2):
    """
    30 arquiteturas avançadas com suporte a múltiplas funções de ativação.
    
    Categorias:
    - 1-5:   LSTM Base & Variants
    - 6-10:  GRU Base & Variants  
    - 11-15: Bidirectional (LSTM + GRU)
    - 16-20: Stacked Deep Networks
    - 21-25: Residual & Skip Connections
    - 26-30: Attention & Hybrid Mechanisms
    """
    m = None
    act = get_activation(activation)

    # ========== LSTM BASE & VARIANTS (1-5) ==========
    if model_id == 1:
        m = _build_lstm_classic(input_shape, act, dropout_rate)
    elif model_id == 2:
        m = _build_lstm_layer_norm(input_shape, act, dropout_rate)
    elif model_id == 3:
        m = _build_lstm_batch_norm(input_shape, act, dropout_rate)
    elif model_id == 4:
        m = _build_lstm_narrow_deep(input_shape, act, dropout_rate)
    elif model_id == 5:
        m = _build_lstm_wide_shallow(input_shape, act, dropout_rate)

    # ========== GRU BASE & VARIANTS (6-10) ==========
    elif model_id == 6:
        m = _build_gru_classic(input_shape, act, dropout_rate)
    elif model_id == 7:
        m = _build_gru_deep(input_shape, act, dropout_rate)
    elif model_id == 8:
        m = _build_gru_wide(input_shape, act, dropout_rate)
    elif model_id == 9:
        m = _build_gru_residual_dense(input_shape, act, dropout_rate)
    elif model_id == 10:
        m = _build_gru_hybrid(input_shape, act, dropout_rate)

    # ========== BIDIRECTIONAL (11-15) ==========
    elif model_id == 11:
        m = _build_bilstm_classic(input_shape, act, dropout_rate)
    elif model_id == 12:
        m = _build_bigru_classic(input_shape, act, dropout_rate)
    elif model_id == 13:
        m = _build_bilstm_deep(input_shape, act, dropout_rate)
    elif model_id == 14:
        m = _build_bigru_deep(input_shape, act, dropout_rate)
    elif model_id == 15:
        m = _build_bilstm_bigru_mix(input_shape, act, dropout_rate)

    # ========== STACKED DEEP NETWORKS (16-20) ==========
    elif model_id == 16:
        m = _build_stacked_lstm_deep(input_shape, act, dropout_rate)
    elif model_id == 17:
        m = _build_stacked_gru_deep(input_shape, act, dropout_rate)
    elif model_id == 18:
        m = _build_pyramid_lstm(input_shape, act, dropout_rate)
    elif model_id == 19:
        m = _build_inverted_pyramid_lstm(input_shape, act, dropout_rate)
    elif model_id == 20:
        m = _build_diamond_lstm(input_shape, act, dropout_rate)

    # ========== RESIDUAL & SKIP CONNECTIONS (21-25) ==========
    elif model_id == 21:
        m = _build_lstm_residual_v1(input_shape, act, dropout_rate)
    elif model_id == 22:
        m = _build_lstm_residual_v2(input_shape, act, dropout_rate)
    elif model_id == 23:
        m = _build_skip_dense(input_shape, act, dropout_rate)
    elif model_id == 24:
        m = _build_highway_lstm(input_shape, act, dropout_rate)
    elif model_id == 25:
        m = _build_densenet_lstm(input_shape, act, dropout_rate)

    # ========== ATTENTION & HYBRID (26-30) ==========
    elif model_id == 26:
        m = _build_attention_lstm(input_shape, act, dropout_rate)
    elif model_id == 27:
        m = _build_multihead_attention(input_shape, act, dropout_rate)
    elif model_id == 28:
        m = _build_cnn_lstm_hybrid(input_shape, act, dropout_rate)
    elif model_id == 29:
        m = _build_lstm_timedistributed(input_shape, act, dropout_rate)
    elif model_id == 30:
        m = _build_ensemble_multipath(input_shape, act, dropout_rate)

    else:
        raise ValueError(f"model_id deve ser 1..30, recebido: {model_id}")

    # Compilar com Adam
    m.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return m


# Nomes dos modelos
ADVANCED_MODEL_NAMES = {
    1: "LSTM Classic (64/32)",
    2: "LSTM + LayerNorm",
    3: "LSTM + BatchNorm",
    4: "LSTM Narrow-Deep (32³)",
    5: "LSTM Wide-Shallow (256)",
    
    6: "GRU Classic (64/32)",
    7: "GRU Deep (128/64/32)",
    8: "GRU Wide (192/96)",
    9: "GRU Residual Dense",
    10: "GRU Hybrid (80/80/40)",
    
    11: "BiLSTM Classic (64/32)",
    12: "BiGRU Classic (64/32)",
    13: "BiLSTM Deep (96/64/32)",
    14: "BiGRU Deep (96/64/32)",
    15: "BiLSTM+BiGRU Mix",
    
    16: "Stacked LSTM (128→32)",
    17: "Stacked GRU (128→32)",
    18: "Pyramid LSTM (256→16)",
    19: "Inverted Pyramid (32→128)",
    20: "Diamond LSTM (64/128/128/64)",
    
    21: "LSTM Residual v1",
    22: "LSTM Residual v2",
    23: "Skip Connection Dense",
    24: "Highway LSTM",
    25: "DenseNet-style LSTM",
    
    26: "Self-Attention LSTM",
    27: "Multi-Head Attention",
    28: "CNN+LSTM Hybrid",
    29: "LSTM TimeDistributed",
    30: "Ensemble Multi-Path",
}


# Descrições técnicas detalhadas de cada modelo
ADVANCED_MODEL_DESCRIPTIONS = {
    1: "Arquitetura LSTM clássica com duas camadas recorrentes (64→32 unidades). Usa dropout para regularização. Ideal como baseline robusto para séries temporais.",
    2: "LSTM com Layer Normalization após cada camada recorrente. Estabiliza o treinamento e acelera convergência, especialmente útil para séries com mudanças de escala.",
    3: "LSTM com Batch Normalization, que normaliza ativações em mini-batches. Dropout aumentado (1.2x) para compensar o efeito regularizador do BatchNorm. Ótimo para dados com alta variância.",
    4: "Arquitetura narrow-deep: 3 camadas LSTM de 32 unidades cada. Processa informação em múltiplos níveis hierárquicos. Bom para capturar padrões complexos com poucos parâmetros.",
    5: "LSTM wide-shallow com uma única camada de 256 unidades. Alta capacidade representacional em camada única. Dropout elevado (1.5x) evita overfitting. Rápido em treinamento.",
    
    6: "GRU clássico (64→32). Mais eficiente que LSTM (menos parâmetros, 2 gates vs 3). Excelente para séries com memória de curto/médio prazo. Treina mais rápido que LSTM equivalente.",
    7: "GRU profundo com 3 camadas (128→64→32) e Layer Normalization. Captura hierarquias temporais complexas. Dropout progressivo para regularização gradual.",
    8: "GRU largo com 2 camadas (192→96). Alta capacidade de memória. Ideal para séries com muitas features ou padrões intrincados. Requer mais dados para treinar bem.",
    9: "GRU com conexões residuais densas. Skip connections permitem gradiente fluir diretamente. Reduz vanishing gradient. Camada Dense final integra múltiplas resoluções temporais.",
    10: "GRU híbrido (80→80→40) combinando camadas paralelas e sequenciais. Processa informação em diferentes escalas simultaneamente. Boa generalização em diversos tipos de séries.",
    
    11: "BiLSTM (Bidirectional LSTM) clássico. Processa sequência em ambas direções (passado→futuro e futuro→passado). Captura dependências bidirecionais. Dobrando parâmetros vs LSTM unidirecional.",
    12: "BiGRU clássico. Versão bidirecional do GRU. Mais eficiente que BiLSTM, mantendo poder expressivo. Ideal quando contexto futuro é informativamente relevante para previsão.",
    13: "BiLSTM profundo (3 camadas: 96→64→32). Múltiplos níveis de abstração bidirecional. Layer Normalization estabiliza camadas profundas. Excelente para padrões temporais complexos não-lineares.",
    14: "BiGRU profundo (3 camadas: 96→64→32). Versão GRU do BiLSTM Deep. Treinamento mais rápido com eficácia similar. Bom balanço entre performance e custo computacional.",
    15: "Arquitetura mista: BiLSTM seguido de BiGRU. Combina força de ambos: LSTM captura dependências longas, GRU refina com eficiência. Dropout entre transições reduz co-adaptação.",
    
    16: "LSTM empilhado com decaimento progressivo (128→64→32→16). Cada camada aprende representações de maior abstração. Pyramid stacking: entrada larga, saída focada. Excelente para dados complexos.",
    17: "GRU empilhado com mesma estratégia de decaimento (128→64→32→16). Versão GRU do Stacked LSTM. Menos parâmetros, treinamento rápido. Boa escolha para produção com recursos limitados.",
    18: "Pirâmide LSTM extrema (256→128→64→32→16). Processa informação em 5 níveis hierárquicos. Captura desde padrões locais até tendências globais. Requer muitos dados para evitar overfitting.",
    19: "Pirâmide invertida (32→64→128→256). Começa focado e expande representação. Útil quando entrada é compacta mas padrões subjacentes são complexos. Design contra-intuitivo mas eficaz.",
    20: "Arquitetura diamante (64→128→128→64). Expande no meio para captura máxima, depois comprime. Balanceia foco local e contexto global. Dropout variável preserva informação crítica.",
    
    21: "LSTM com conexões residuais v1. Skip connections adicionam entrada diretamente à saída de camadas intermediárias. Facilita treinamento profundo. Reduz degradação de gradiente.",
    22: "LSTM residual v2 com múltiplas shortcuts. Implementa esquema ResNet para redes recorrentes. Dense final integra todas as resoluções. Treina redes muito profundas estávelmente.",
    23: "Dense Skip Connections: cada camada conecta a todas anteriores (DenseNet-style). Máxima reutilização de features. Concatenação preserva todas resoluções temporais. Alto uso de memória.",
    24: "Highway LSTM: gates aprendidos controlam fluxo de informação através de shortcuts. Inspirado em Highway Networks. Modelo decide dinamicamente quando usar skip connections vs transformações.",
    25: "DenseNet-style LSTM: concatena outputs de todas camadas anteriores. Growth rate controlado. Feature reuse extremo. Excelente performance mas computacionalmente caro. Para datasets grandes.",
    
    26: "Self-Attention sobre LSTM. Attention layer aprende quais timesteps são mais relevantes. Pesos de atenção dinâmicos. Captura dependências não-locais. Interpretabilidade via attention weights.",
    27: "Multi-Head Attention (estilo Transformer) após LSTM. 4 cabeças de atenção capturam diferentes aspectos temporais simultaneamente. Concatena e projeta resultados. State-of-the-art para séries complexas.",
    28: "CNN+LSTM híbrido. Conv1D extrai features locais, LSTM captura dependências temporais. Combina força de ambas arquiteturas. MaxPooling reduz dimensionalidade. Excelente para sinais com padrões locais repetitivos.",
    29: "LSTM com TimeDistributed Dense layers. Aplica mesma camada Dense a cada timestep independentemente. Útil para prediction windows. Compartilha pesos temporalmente. Reduz parâmetros vs Dense separadas.",
    30: "Ensemble Multi-Path: múltiplas streams paralelas (LSTM, GRU, CNN) processam entrada simultaneamente. Concatena outputs antes da predição. Wisdom of crowds. Robusto mas pesado. Melhor generalização.",
}
