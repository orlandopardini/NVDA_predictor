"""
Data Helpers - Funções auxiliares para manipulação de dados de mercado.

Este módulo contém funções utilitárias para:
- Normalização de dados OHLCV do Yahoo Finance/Stooq
- Download de dados históricos com retry e fallback
- Atualização do flag de modelo vencedor (winner)

Autor: Sistema de Trading LSTM
Data: 2025-01-14
"""

import time
import logging
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


def _normalize_ohlcv(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Normaliza DataFrame OHLCV para formato padrão.
    
    Args:
        df: DataFrame com dados de mercado (pode ter MultiIndex)
        ticker: Símbolo do ticker (ex: 'NVDA', 'AAPL')
    
    Returns:
        DataFrame normalizado com colunas: Open, High, Low, Close, Adj_Close, Volume
        
    Notes:
        - Remove MultiIndex se presente
        - Padroniza nomes de colunas
        - Remove linhas com Close ausente
        - Converte índice para datetime
    """
    if df is None or df.empty:
        return pd.DataFrame()

    # Trata MultiIndex vindo do Yahoo Finance
    if isinstance(df.columns, pd.MultiIndex):
        upper = ticker.upper()
        selected = None
        
        # Procura nível com o ticker
        for level in range(df.columns.nlevels - 1, -1, -1):
            vals = [str(v).upper() for v in df.columns.get_level_values(level)]
            if upper in vals:
                selected = level
                break
        
        if selected is not None:
            try:
                df = df.xs(upper, axis=1, level=selected, drop_level=True)
            except Exception:
                df = df.swaplevel(selected, 0, axis=1)
                df = df[upper]
        else:
            # Fallback: concatena níveis
            df.columns = ['_'.join([str(x) for x in c if x is not None]) for c in df.columns]

    # Padroniza nomes de colunas
    rename = {}
    for c in df.columns:
        lc = str(c).lower().strip()
        if lc in ("open", "high", "low", "close", "volume", "adj close", "adj_close", "adjclose"):
            if "adj" in lc:
                rename[c] = "Adj_Close"
            elif lc == "volume":
                rename[c] = "Volume"
            else:
                rename[c] = lc.capitalize()
    
    df = df.rename(columns=rename)

    # Garante coluna Adj_Close
    if "Adj_Close" not in df.columns and "Close" in df.columns:
        df["Adj_Close"] = df["Close"]

    # Seleciona apenas colunas válidas
    keep = [c for c in ["Open", "High", "Low", "Close", "Adj_Close", "Volume"] if c in df.columns]
    df = df[keep].copy()
    
    # Remove linhas sem Close
    if "Close" in df.columns:
        df = df.dropna(subset=["Close"])
    
    df.index = pd.to_datetime(df.index)
    return df


def _fetch_yahoo_block(ticker: str, d0, d1) -> pd.DataFrame:
    """
    Baixa dados do Yahoo Finance para um período específico.
    
    Args:
        ticker: Símbolo do ticker (ex: 'NVDA')
        d0: Data inicial (date ou string)
        d1: Data final (date ou string)
    
    Returns:
        DataFrame com dados OHLCV ou vazio se falhar
        
    Notes:
        - Tenta dois métodos: yf.download() e yf.Ticker().history()
        - Usa auto_adjust=False para manter volume original
    """
    end_exc = pd.Timestamp(d1) + pd.Timedelta(days=1)  # end exclusivo
    
    # Método 1: yf.download
    df = yf.download(
        ticker, 
        start=str(d0), 
        end=str(end_exc.date()),
        auto_adjust=False, 
        progress=False, 
        threads=False
    )
    if df is not None and not df.empty:
        return df
    
    # Método 2: yf.Ticker().history
    df2 = yf.Ticker(ticker).history(
        start=str(d0), 
        end=str(end_exc.date()),
        interval="1d", 
        auto_adjust=False
    )
    return df2


def _fetch_stooq_block(ticker: str, d0, d1) -> pd.DataFrame:
    """
    Baixa dados do Stooq como fallback do Yahoo Finance.
    
    Args:
        ticker: Símbolo do ticker (ex: 'NVDA')
        d0: Data inicial (date ou string)
        d1: Data final (date ou string)
    
    Returns:
        DataFrame com dados OHLCV ou vazio se falhar
        
    Notes:
        - Tenta primeiro com sufixo .US (ex: NVDA.US)
        - Se falhar, tenta sem sufixo
        - Requer pandas_datareader instalado
    """
    try:
        from pandas_datareader import data as pdr
    except ImportError:
        logger.warning("pandas_datareader não instalado, Stooq indisponível")
        return pd.DataFrame()

    for code in (f"{ticker}.US", ticker):
        try:
            df = pdr.DataReader(code, "stooq", start=d0, end=d1)
            if df is not None and not df.empty:
                df = df.sort_index()
                return df
        except Exception as e:
            logger.debug(f"Stooq falhou para {code}: {e}")
            continue
    
    return pd.DataFrame()


def _fetch_resilient_yearly(ticker: str, start: str) -> pd.DataFrame:
    """
    Baixa dados históricos com retry, backoff e fallback para Stooq.
    
    Args:
        ticker: Símbolo do ticker (ex: 'NVDA')
        start: Data inicial em formato 'YYYY-MM-DD'
    
    Returns:
        DataFrame consolidado com todos os dados disponíveis
        
    Strategy:
        1. Divide período em blocos anuais
        2. Para cada bloco:
           - Tenta Yahoo Finance com 6 retries + exponential backoff
           - Se falhar, tenta Stooq
        3. Consolida todos os blocos e remove duplicatas
        
    Notes:
        - Backoff: 1.5 * (2 ** tentativa) segundos
        - Pausa de 1s entre blocos para evitar rate limit
        - Remove duplicatas mantendo registro mais recente
    """
    start_date = pd.to_datetime(start).date()
    end_date = pd.Timestamp.today().date()
    frames = []
    
    y = start_date.year
    while y <= end_date.year:
        d0 = pd.Timestamp(f"{y}-01-01").date()
        if y == start_date.year and d0 < start_date:
            d0 = start_date
        
        d1 = pd.Timestamp(f"{y}-12-31").date()
        if d1 > end_date:
            d1 = end_date

        # Yahoo com retries e exponential backoff
        got = False
        last_exc = None
        for i in range(6):
            try:
                df = _fetch_yahoo_block(ticker, d0, d1)
                if df is not None and not df.empty:
                    frames.append(_normalize_ohlcv(df, ticker))
                    got = True
                    break
                last_exc = RuntimeError("Yahoo retornou vazio")
            except Exception as e:
                last_exc = e
            
            # Exponential backoff
            time.sleep(1.5 * (2 ** i))

        # Fallback para Stooq se Yahoo falhar
        if not got:
            logger.warning(f"Yahoo falhou para {ticker} {y}, tentando Stooq...")
            df = _fetch_stooq_block(ticker, d0, d1)
            if df is not None and not df.empty:
                frames.append(_normalize_ohlcv(df, ticker))
            else:
                logger.error(f"Ambos Yahoo e Stooq falharam para {ticker} {y}")

        # Pausa entre blocos
        time.sleep(1.0)
        y += 1

    if not frames:
        return pd.DataFrame()
    
    # Consolida e remove duplicatas
    out = pd.concat(frames, axis=0)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out


def update_winner_flag(ticker: str, db_session) -> None:
    """
    Recalcula e atualiza o flag is_winner baseado em score combinado.
    
    Args:
        ticker: Símbolo do ticker (ex: 'NVDA')
        db_session: Sessão do SQLAlchemy (db.session)
    
    Algorithm:
        1. Busca todos os modelos do ticker com RMSE válido
        2. Calcula score combinado: 60% RMSE + 40% Pearson
        3. Normaliza valores entre 0 e 1
        4. Define modelo com menor score como winner
        
    Score Formula:
        score = (0.6 * rmse_norm) + (0.4 * (1 - pearson_norm))
        
        Onde:
        - rmse_norm = (rmse - min_rmse) / (max_rmse - min_rmse)
        - pearson_norm = (pearson - min_pearson) / (max_pearson - min_pearson)
        
    Notes:
        - Menor score = melhor modelo
        - RMSE tem peso 60% (erro mais importante)
        - Pearson tem peso 40% (correlação também conta)
        - Remove is_winner de todos antes de definir novo
    """
    from ..models import ModelRegistry
    
    try:
        rows = (ModelRegistry.query
                .filter_by(ticker=ticker)
                .filter(ModelRegistry.rmse.isnot(None))
                .all())
        
        if not rows:
            logger.info(f"Nenhum modelo encontrado para {ticker}")
            return
        
        # Coleta valores para normalização
        rmse_values = [r.rmse for r in rows]
        pearson_values = [r.pearson_corr if r.pearson_corr is not None else 0.0 for r in rows]
        
        min_rmse = min(rmse_values)
        max_rmse = max(rmse_values)
        min_pearson = min(pearson_values)
        max_pearson = max(pearson_values)
        
        best_model = None
        best_score = float('inf')
        
        # Calcula score para cada modelo
        for r in rows:
            # Normaliza RMSE (0 = melhor, 1 = pior)
            rmse_norm = (r.rmse - min_rmse) / (max_rmse - min_rmse + 1e-10)
            
            # Normaliza Pearson (1 = melhor, 0 = pior)
            pearson_val = r.pearson_corr if r.pearson_corr is not None else 0.0
            pearson_norm = (pearson_val - min_pearson) / (max_pearson - min_pearson + 1e-10)
            
            # Score combinado: 60% RMSE + 40% Pearson invertido
            score = (0.6 * rmse_norm) + (0.4 * (1 - pearson_norm))
            
            if score < best_score:
                best_score = score
                best_model = r
        
        if best_model:
            # Remove winner de todos
            ModelRegistry.query.filter_by(ticker=ticker).update({"is_winner": False})
            
            # Define novo winner
            best_model.is_winner = True
            db_session.commit()
            
            logger.info(
                f"✅ Winner atualizado para {ticker}: "
                f"modelo {best_model.model_id} (score: {best_score:.4f})"
            )
    
    except Exception as e:
        logger.error(f"❌ Erro ao atualizar winner flag para {ticker}: {e}")
        db_session.rollback()
        raise
