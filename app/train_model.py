"""
Script para treinar modelos via linha de comando
Uso: python -m app.train_model NVDA 50
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import create_app
from app.ml.trainer import train_all_models_fast

def main():
    ticker = sys.argv[1] if len(sys.argv) > 1 else 'NVDA'
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    
    app = create_app()
    with app.app_context():
        print(f"\n🚀 Treinando modelo para {ticker} com {epochs} epochs...\n")
        
        result = train_all_models_fast(
            ticker=ticker,
            lookback=60,
            horizon=1,
            epochs=epochs,
            batch_size=32,
            reuse_if_exists=False
        )
        
        winner = result.get('winner', {})
        metrics = winner.get('metrics', {})
        
        print(f"\n✅ Treinamento concluído!")
        print(f"🏆 Modelo vencedor: {winner.get('model_name')}")
        print(f"📊 Métricas:")
        print(f"   MAE:  {metrics.get('mae', 0):.4f}")
        print(f"   RMSE: {metrics.get('rmse', 0):.4f}")
        print(f"   R²:   {metrics.get('r2', 0):.4f}")
        print(f"   Accuracy: {metrics.get('accuracy', 0):.2%}")
        print(f"⏱️  Tempo: {result.get('walltime_sec', 0):.1f}s\n")

if __name__ == '__main__':
    main()
