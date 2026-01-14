# app/ml/training_progress.py
"""
Sistema de acompanhamento de progresso de treino em tempo real
"""
from typing import Optional, Dict, Any
from threading import Lock

class TrainingProgress:
    """
    Armazena o estado atual do treino avançado.
    Thread-safe para acesso de múltiplas threads.
    """
    
    def __init__(self):
        self._lock = Lock()
        self._progress_data: Dict[str, Any] = {
            'is_training': False,
            'mode': None,  # 'fast' ou 'optimized'
            'current_model': 0,
            'total_models': 0,
            'current_trial': 0,
            'total_trials': 0,
            'percent': 0.0,
            'message': '',
            'model_name': '',
            'error': None
        }
    
    def start_training(self, mode: str, total_models: int, total_trials: int = 1):
        """Inicia um novo treino"""
        with self._lock:
            self._progress_data = {
                'is_training': True,
                'mode': mode,
                'current_model': 0,
                'total_models': total_models,
                'current_trial': 0,
                'total_trials': total_trials,
                'percent': 0.0,
                'message': f'Iniciando treino {mode}...',
                'model_name': '',
                'error': None
            }
    
    def update_progress(
        self, 
        current_model: int, 
        model_name: str = '',
        current_trial: int = 0,
        message: str = ''
    ):
        """Atualiza o progresso do treino"""
        with self._lock:
            self._progress_data['current_model'] = current_model
            self._progress_data['current_trial'] = current_trial
            self._progress_data['model_name'] = model_name
            
            # Calcula percentual real
            if self._progress_data['mode'] == 'fast':
                # Modo rápido: cada modelo é 1 unidade
                percent = (current_model / self._progress_data['total_models']) * 100
            else:
                # Modo otimizado: considera trials dentro de cada modelo
                total_work = self._progress_data['total_models'] * self._progress_data['total_trials']
                work_done = ((current_model - 1) * self._progress_data['total_trials']) + current_trial
                percent = (work_done / total_work) * 100
            
            self._progress_data['percent'] = min(percent, 99.9)  # nunca 100% até finalizar
            
            if message:
                self._progress_data['message'] = message
            else:
                if self._progress_data['mode'] == 'fast':
                    self._progress_data['message'] = f'Treinando modelo {current_model}/{self._progress_data["total_models"]}: {model_name}'
                else:
                    self._progress_data['message'] = f'Modelo {current_model}/{self._progress_data["total_models"]}, Trial {current_trial}/{self._progress_data["total_trials"]}: {model_name}'
    
    def finish_training(self, success: bool = True, error: str = None):
        """Finaliza o treino"""
        with self._lock:
            self._progress_data['is_training'] = False
            self._progress_data['percent'] = 100.0
            self._progress_data['error'] = error
            
            if success:
                self._progress_data['message'] = '✅ Treino concluído com sucesso!'
            else:
                self._progress_data['message'] = f'❌ Treino falhou: {error}'
    
    def get_progress(self) -> Dict[str, Any]:
        """Retorna o estado atual do progresso"""
        with self._lock:
            return self._progress_data.copy()
    
    def reset(self):
        """Reseta o progresso"""
        with self._lock:
            self._progress_data = {
                'is_training': False,
                'mode': None,
                'current_model': 0,
                'total_models': 0,
                'current_trial': 0,
                'total_trials': 0,
                'percent': 0.0,
                'message': '',
                'model_name': '',
                'error': None
            }


# Instância global única
_global_progress = TrainingProgress()

def get_training_progress() -> TrainingProgress:
    """Retorna a instância global de progresso"""
    return _global_progress
