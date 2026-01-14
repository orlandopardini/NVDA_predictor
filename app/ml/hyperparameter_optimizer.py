# app/ml/hyperparameter_optimizer.py
"""
🔬 HYPERPARAMETER OPTIMIZATION SYSTEM
Implementa Grid Search, Random Search e Bayesian Optimization
"""
import itertools
import random
import time
import numpy as np
from typing import Dict, List, Tuple, Any
import logging

logger = logging.getLogger(__name__)


# ========== ESPAÇO DE BUSCA ==========
HYPERPARAMETER_SPACE = {
    'learning_rate': [0.0001, 0.0005, 0.001, 0.005, 0.01],
    'batch_size': [16, 32, 64, 128],
    'dropout_rate': [0.1, 0.15, 0.2, 0.25, 0.3, 0.4],
    'epochs': [10, 20, 30, 50],
    'activation': ['relu', 'tanh', 'elu', 'selu', 'swish', 'gelu', 'leaky_relu'],
}


class HyperparameterOptimizer:
    """
    Otimizador de hiperparâmetros com múltiplas estratégias:
    - Grid Search: testa TODAS combinações (pode ser lento)
    - Random Search: testa N combinações aleatórias (mais rápido)
    - Bayesian: usa resultados anteriores para guiar busca (mais inteligente)
    """
    
    def __init__(self, search_space: Dict = None):
        self.search_space = search_space or HYPERPARAMETER_SPACE
        self.history = []  # guarda (params, score)
        
    def grid_search_combinations(self, max_combinations: int = 100) -> List[Dict]:
        """
        Gera todas as combinações possíveis (Grid Search completo).
        ATENÇÃO: Pode gerar MILHARES de combinações!
        """
        keys = list(self.search_space.keys())
        values = [self.search_space[k] for k in keys]
        
        all_combos = list(itertools.product(*values))
        logger.info(f"Grid Search: {len(all_combos)} combinações possíveis")
        
        # Limita para evitar explosão combinatória
        if len(all_combos) > max_combinations:
            logger.warning(f"Limitando a {max_combinations} combinações")
            all_combos = random.sample(all_combos, max_combinations)
        
        combinations = []
        for combo in all_combos:
            combinations.append({keys[i]: combo[i] for i in range(len(keys))})
        
        return combinations
    
    def random_search_combinations(self, n_samples: int = 50) -> List[Dict]:
        """
        Gera N combinações aleatórias (Random Search).
        Geralmente mais eficiente que Grid Search completo.
        """
        logger.info(f"Random Search: gerando {n_samples} combinações")
        combinations = []
        
        for _ in range(n_samples):
            combo = {
                key: random.choice(values) 
                for key, values in self.search_space.items()
            }
            combinations.append(combo)
        
        return combinations
    
    def bayesian_next_candidate(self, n_candidates: int = 10) -> List[Dict]:
        """
        Usa Bayesian Optimization simplificada:
        1. Explora regiões próximas aos melhores resultados anteriores
        2. Adiciona exploração aleatória (exploration vs exploitation)
        """
        if len(self.history) < 5:
            # Começa com random search
            logger.info("Bayesian: poucos dados, usando random search inicial")
            return self.random_search_combinations(n_candidates)
        
        # Ordena histórico por score (menor é melhor para RMSE)
        sorted_history = sorted(self.history, key=lambda x: x[1])
        best_params = [h[0] for h in sorted_history[:3]]  # top-3
        
        logger.info(f"Bayesian: explorando vizinhança dos top-3 modelos")
        candidates = []
        
        # 70% exploitation (vizinhança dos melhores)
        n_exploit = int(n_candidates * 0.7)
        for _ in range(n_exploit):
            base = random.choice(best_params)
            mutated = self._mutate_params(base)
            candidates.append(mutated)
        
        # 30% exploration (aleatório)
        n_explore = n_candidates - n_exploit
        candidates.extend(self.random_search_combinations(n_explore))
        
        return candidates
    
    def _mutate_params(self, params: Dict) -> Dict:
        """
        Muta parâmetros ligeiramente (vizinhança local).
        Estratégia: muda 1-2 parâmetros para valores próximos.
        """
        mutated = params.copy()
        keys_to_mutate = random.sample(list(mutated.keys()), k=random.randint(1, 2))
        
        for key in keys_to_mutate:
            options = self.search_space[key]
            current_idx = options.index(mutated[key]) if mutated[key] in options else 0
            
            # Move para valor adjacente
            if len(options) > 1:
                if current_idx == 0:
                    new_idx = 1
                elif current_idx == len(options) - 1:
                    new_idx = len(options) - 2
                else:
                    new_idx = current_idx + random.choice([-1, 1])
                
                mutated[key] = options[new_idx]
        
        return mutated
    
    def record_result(self, params: Dict, score: float):
        """Registra resultado de uma tentativa (para Bayesian)"""
        self.history.append((params, score))
        logger.debug(f"Registrado: {params} -> score={score:.6f}")
    
    def get_best_params(self) -> Tuple[Dict, float]:
        """Retorna os melhores parâmetros encontrados"""
        if not self.history:
            raise ValueError("Nenhum resultado registrado ainda")
        
        best = min(self.history, key=lambda x: x[1])
        return best[0], best[1]


class EarlyStoppingMonitor:
    """
    Early Stopping inteligente para evitar overfitting.
    Para o treino se:
    - Loss de validação não melhora por N epochs (patience)
    - Loss começa a aumentar (divergência)
    """
    
    def __init__(self, patience: int = 5, min_delta: float = 0.0001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.should_stop = False
    
    def update(self, val_loss: float) -> bool:
        """
        Atualiza monitor com perda de validação.
        Retorna True se deve parar o treino.
        """
        if val_loss < (self.best_loss - self.min_delta):
            # Melhorou
            self.best_loss = val_loss
            self.counter = 0
            return False
        else:
            # Não melhorou
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                logger.info(f"Early stopping ativado! (patience={self.patience})")
                return True
        
        return False


def optimize_hyperparameters(
    model_id: int,
    ticker: str,
    train_fn,  # função que treina e retorna score
    strategy: str = 'random',
    n_trials: int = 30,
    verbose: bool = True,
    progress_callback = None  # callback(trial_num) para reportar progresso
) -> Dict[str, Any]:
    """
    Orquestra a otimização de hiperparâmetros.
    
    Args:
        model_id: ID do modelo (1-30)
        ticker: Ticker da ação (sempre NVDA)
        train_fn: Função que recebe (model_id, ticker, **params) e retorna score
        strategy: 'grid', 'random' ou 'bayesian'
        n_trials: Número de tentativas
        verbose: Exibir progresso
        progress_callback: Função callback(trial_num) chamada a cada trial
    
    Returns:
        {
            'best_params': {...},
            'best_score': float,
            'all_results': [...],
            'elapsed_time': float
        }
    """
    logger.info(f"Iniciando otimização para model_id={model_id}, strategy={strategy}")
    start_time = time.time()
    
    optimizer = HyperparameterOptimizer()
    
    # Gera candidatos conforme estratégia
    if strategy == 'grid':
        candidates = optimizer.grid_search_combinations(max_combinations=n_trials)
    elif strategy == 'random':
        candidates = optimizer.random_search_combinations(n_samples=n_trials)
    elif strategy == 'bayesian':
        candidates = []
        # Bayesian é iterativo: gera candidatos conforme aprende
        for i in range(n_trials):
            if i % 10 == 0:
                batch = optimizer.bayesian_next_candidate(n_candidates=10)
                candidates.extend(batch)
    else:
        raise ValueError(f"Estratégia inválida: {strategy}")
    
    results = []
    for i, params in enumerate(candidates[:n_trials], 1):
        if verbose:
            logger.info(f"Trial {i}/{n_trials}: {params}")
        
        # Callback de progresso
        if progress_callback:
            try:
                progress_callback(i)
            except Exception as e:
                logger.warning(f"Progress callback falhou: {e}")
        
        try:
            score = train_fn(model_id, ticker, **params)
            optimizer.record_result(params, score)
            results.append({'params': params, 'score': score})
            
            if verbose:
                logger.info(f"  -> Score: {score:.6f}")
        
        except Exception as e:
            logger.error(f"Trial {i} falhou: {e}")
            continue
    
    elapsed_time = time.time() - start_time
    
    if not results:
        raise RuntimeError("Nenhum trial foi bem-sucedido!")
    
    # Encontra melhor resultado
    best_result = min(results, key=lambda r: r['score'])
    
    summary = {
        'best_params': best_result['params'],
        'best_score': best_result['score'],
        'all_results': results,
        'elapsed_time': elapsed_time,
        'n_trials': len(results)
    }
    
    logger.info(f"Otimização concluída em {elapsed_time:.1f}s")
    logger.info(f"Melhor score: {best_result['score']:.6f}")
    logger.info(f"Melhores params: {best_result['params']}")
    
    return summary


def get_optimizer_recommendation(n_models: int, max_time_minutes: int = 60) -> str:
    """
    Recomenda estratégia de otimização baseado em:
    - Número de modelos
    - Tempo disponível
    """
    total_combinations = np.prod([len(v) for v in HYPERPARAMETER_SPACE.values()])
    
    if n_models <= 5:
        # Poucos modelos: pode fazer grid search
        return 'grid'
    elif n_models <= 15:
        # Moderado: random search é bom
        return 'random'
    else:
        # Muitos modelos: bayesian é mais eficiente
        return 'bayesian'
