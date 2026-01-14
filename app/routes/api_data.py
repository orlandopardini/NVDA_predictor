"""
API Data Routes - Endpoints para gerenciamento de dados de mercado.

Este módulo contém rotas para:
- Atualização de dados históricos (Yahoo Finance/Stooq)
- Carregamento de dados em lotes
- Consulta de séries temporais OHLCV
- Listagem de tickers disponíveis no banco

Rotas:
    POST /api/update_data - Atualiza dados do ticker
    POST /api/load_ticker_data - Carrega dados em lotes
    GET /api/series - Retorna série OHLCV
    GET /api/tickers - Lista todos os tickers

Autor: Sistema de Trading LSTM
Data: 2025-01-14
"""

import time
import pandas as pd
from flask import Blueprint, request, jsonify
from flasgger import swag_from

from .. import db
from ..models import PrecoDiario
from ..utils.data_helpers import (
    _normalize_ohlcv,
    _fetch_yahoo_block,
    _fetch_stooq_block,
    _fetch_resilient_yearly
)
from ..utils.auth_helpers import _auth_ok

# Blueprint
api_data_bp = Blueprint('api_data', __name__)


@api_data_bp.post('/update_data')
@swag_from({
    'tags': ['Dados'],
    'parameters': [
        {'name': 'ticker', 'in': 'query', 'schema': {'type': 'string'}, 'required': True},
        {'name': 'start',  'in': 'query', 'schema': {'type': 'string'}, 'required': False},
    ],
    'responses': {200: {'description': 'Dados atualizados com sucesso'}}
})
def update_data():
    """
    Atualiza dados históricos do ticker no banco de dados.
    
    Query Params:
        ticker (str): Símbolo do ticker (ex: NVDA, AAPL)
        start (str): Data inicial no formato YYYY-MM-DD (default: 2010-01-01)
    
    Returns:
        JSON com estatísticas:
        - ticker: Símbolo atualizado
        - rows_added: Número de registros novos inseridos
        - range: [primeira_data, última_data]
        - note: Mensagem de erro se não conseguir dados
    
    Security:
        Requer autenticação via X-API-KEY header
    
    Example:
        POST /api/update_data?ticker=NVDA&start=2020-01-01
        Headers: X-API-KEY: your-key
        
        Response:
        {
            "ticker": "NVDA",
            "rows_added": 150,
            "range": ["2020-01-01", "2025-01-14"]
        }
    """
    if not _auth_ok(request):
        return jsonify({'error': 'unauthorized'}), 401

    ticker = request.args.get('ticker', 'NVDA').upper()
    start = request.args.get('start', '2010-01-01')

    df = _fetch_resilient_yearly(ticker, start)
    if df is None or df.empty:
        return jsonify({
            'ticker': ticker, 
            'rows_added': 0, 
            'note': 'No data (Yahoo/Stooq indisponíveis ou rede bloqueada)'
        })

    added = 0
    for idx, row in df.iterrows():
        d = idx.date()
        rec = PrecoDiario.query.filter_by(ticker=ticker, date=d).first()
        
        if rec:
            # Atualiza registro existente
            rec.open = float(row["Open"]) if "Open" in df.columns else rec.open
            rec.high = float(row["High"]) if "High" in df.columns else rec.high
            rec.low = float(row["Low"]) if "Low" in df.columns else rec.low
            rec.close = float(row["Close"]) if "Close" in df.columns else rec.close
            rec.adj_close = float(row["Adj_Close"]) if "Adj_Close" in df.columns else rec.adj_close
            rec.volume = int(row["Volume"]) if "Volume" in df.columns else rec.volume
        else:
            # Insere novo registro
            db.session.add(PrecoDiario(
                ticker=ticker, 
                date=d,
                open=float(row["Open"]) if "Open" in df.columns else None,
                high=float(row["High"]) if "High" in df.columns else None,
                low=float(row["Low"]) if "Low" in df.columns else None,
                close=float(row["Close"]) if "Close" in df.columns else None,
                adj_close=float(row["Adj_Close"]) if "Adj_Close" in df.columns else (
                    float(row["Close"]) if "Close" in df.columns else None
                ),
                volume=int(row["Volume"]) if "Volume" in df.columns else 0
            ))
            added += 1
    
    db.session.commit()

    first = df.index.min().strftime("%Y-%m-%d")
    last = df.index.max().strftime("%Y-%m-%d")
    
    return jsonify({
        'ticker': ticker, 
        'rows_added': added, 
        'range': [first, last]
    })


@api_data_bp.post('/load_ticker_data')
@swag_from({
    'tags': ['Dados'],
    'parameters': [
        {'name': 'body', 'in': 'body', 'schema': {
            'type': 'object',
            'properties': {
                'ticker': {'type': 'string', 'default': 'NVDA'},
                'start': {'type': 'string', 'default': '2015-01-01'},
                'batch_n': {'type': 'integer', 'default': 100}
            }
        }}
    ],
    'responses': {200: {'description': 'Dados carregados em lotes'}}
})
def load_ticker_data():
    """
    Carrega dados históricos do ticker em lotes (baseado no notebook).
    
    JSON Body:
        ticker (str): Símbolo do ticker (default: NVDA)
        start (str): Data inicial YYYY-MM-DD (default: 2015-01-01)
        batch_n (int): Número de dias úteis por lote (default: 100)
    
    Returns:
        JSON com estatísticas finais:
        - ticker: Símbolo processado
        - inserted: Total de novos registros
        - total: Total de registros no banco após operação
        - start: Primeira data disponível
        - end: Última data disponível
        - note: Mensagem informativa se já atualizado
    
    Strategy:
        1. Determina ponto de partida (última data no banco ou start)
        2. Divide período em lotes de ~batch_n dias úteis
        3. Para cada lote:
           - Baixa dados com retry e fallback
           - Insere ou atualiza no banco
        4. Retorna estatísticas consolidadas
    
    Security:
        Requer autenticação via X-API-KEY header
    
    Example:
        POST /api/load_ticker_data
        Headers: X-API-KEY: your-key
        Body: {
            "ticker": "AAPL",
            "start": "2020-01-01",
            "batch_n": 100
        }
        
        Response:
        {
            "ticker": "AAPL",
            "inserted": 1250,
            "total": 1250,
            "start": "2020-01-01",
            "end": "2025-01-14"
        }
    """
    if not _auth_ok(request):
        return jsonify({'error': 'unauthorized'}), 401

    payload = request.get_json(silent=True) or {}
    ticker = payload.get('ticker', 'NVDA').upper()
    start_str = payload.get('start', '2015-01-01')
    batch_n = int(payload.get('batch_n', 100))
    
    sleep_between = 1.0
    max_retries = 6

    def get_db_span(tick):
        """Retorna (total, min_date, max_date) para o ticker."""
        result = db.session.query(
            db.func.count(PrecoDiario.id),
            db.func.min(PrecoDiario.date),
            db.func.max(PrecoDiario.date)
        ).filter_by(ticker=tick).first()
        
        total, mind, maxd = result if result else (0, None, None)
        return total, mind, maxd

    def upsert_df(df, tick):
        """Insere ou atualiza dados do DataFrame no banco."""
        df = _normalize_ohlcv(df, tick)
        if df.empty:
            return 0

        inserted = 0
        for idx, r in df.iterrows():
            date_obj = pd.Timestamp(idx).date()
            
            existing = PrecoDiario.query.filter_by(
                ticker=tick,
                date=date_obj
            ).first()

            if existing:
                # Atualiza existente
                existing.open = float(r["Open"]) if "Open" in df.columns and pd.notna(r["Open"]) else existing.open
                existing.high = float(r["High"]) if "High" in df.columns and pd.notna(r["High"]) else existing.high
                existing.low = float(r["Low"]) if "Low" in df.columns and pd.notna(r["Low"]) else existing.low
                existing.close = float(r["Close"]) if "Close" in df.columns and pd.notna(r["Close"]) else existing.close
                existing.adj_close = float(r["Adj_Close"]) if "Adj_Close" in df.columns and pd.notna(r["Adj_Close"]) else existing.adj_close
                existing.volume = int(r["Volume"]) if "Volume" in df.columns and pd.notna(r["Volume"]) else existing.volume
            else:
                # Cria novo
                new_record = PrecoDiario(
                    ticker=tick,
                    date=date_obj,
                    open=float(r["Open"]) if "Open" in df.columns and pd.notna(r["Open"]) else None,
                    high=float(r["High"]) if "High" in df.columns and pd.notna(r["High"]) else None,
                    low=float(r["Low"]) if "Low" in df.columns and pd.notna(r["Low"]) else None,
                    close=float(r["Close"]) if "Close" in df.columns and pd.notna(r["Close"]) else None,
                    adj_close=float(r["Adj_Close"]) if "Adj_Close" in df.columns and pd.notna(r["Adj_Close"])
                             else (float(r["Close"]) if "Close" in df.columns and pd.notna(r["Close"]) else None),
                    volume=int(r["Volume"]) if "Volume" in df.columns and pd.notna(r["Volume"]) else 0
                )
                db.session.add(new_record)
                inserted += 1

        db.session.commit()
        return inserted

    def next_window(start_date_str, batch):
        """Retorna (d0, d1) cobrindo ~batch dias úteis."""
        d0 = pd.to_datetime(start_date_str).date()
        bdays = pd.bdate_range(d0, periods=batch)
        d1 = bdays[-1].date() if len(bdays) else d0
        today = pd.Timestamp.today().date()
        if d1 > today:
            d1 = today
        return d0, d1

    def fetch_block_resilient(tick, d0, d1, retries=max_retries):
        """Baixa dados com retries e fallback para Stooq."""
        last_exc = None
        for i in range(retries):
            try:
                df = _fetch_yahoo_block(tick, d0, d1)
                if df is not None and not df.empty:
                    return df
                last_exc = RuntimeError("Yahoo vazio")
            except Exception as e:
                last_exc = e
            
            s = 1.5 * (2 ** i)
            time.sleep(s)

        # Fallback para Stooq
        df = _fetch_stooq_block(tick, d0, d1)
        if df is not None and not df.empty:
            return df
        
        raise last_exc if last_exc else RuntimeError("Falha ao baixar dados")

    # Define ponto de partida
    total, mind, maxd = get_db_span(ticker)
    if maxd:
        start = (pd.to_datetime(maxd).date() + pd.Timedelta(days=1)).isoformat()
        if pd.to_datetime(start) < pd.to_datetime(start_str):
            start = start_str
    else:
        start = start_str

    today = pd.Timestamp.today().date()
    if pd.to_datetime(start).date() > today:
        return jsonify({
            "ticker": ticker,
            "inserted": 0,
            "total": total,
            "start": mind.isoformat() if mind else None,
            "end": maxd.isoformat() if maxd else None,
            "note": "Dados já estão atualizados"
        })

    # Carrega dados em lotes
    inserted_total = 0
    cur = start
    
    while True:
        d0, d1 = next_window(cur, batch_n)
        if d0 > today:
            break

        try:
            df = fetch_block_resilient(ticker, d0, d1)
        except Exception as e:
            return jsonify({"error": f"Erro ao baixar dados: {str(e)}"}), 500

        n = upsert_df(df, ticker)
        inserted_total += n

        # Próximo lote
        cur = (pd.bdate_range(d1, periods=2)[-1].date()).isoformat()
        if pd.to_datetime(cur).date() > today:
            break
        
        time.sleep(sleep_between)

    # Resumo final
    tot, mi, mx = get_db_span(ticker)
    return jsonify({
        "ticker": ticker,
        "inserted": inserted_total,
        "total": tot,
        "start": mi.isoformat() if mi else None,
        "end": mx.isoformat() if mx else None
    })


@api_data_bp.get('/series')
def series():
    """
    Retorna série temporal OHLCV para o ticker.
    
    Query Params:
        ticker (str): Símbolo do ticker (default: NVDA)
        limit (int): Número máximo de dias (default: 800)
    
    Returns:
        JSON com:
        - ticker: Símbolo consultado
        - data: Lista de dicts com [date, open, high, low, close, volume]
    
    Notes:
        - Dados retornados em ordem cronológica crescente
        - Busca os últimos N dias disponíveis no banco
    
    Example:
        GET /api/series?ticker=AAPL&limit=100
        
        Response:
        {
            "ticker": "AAPL",
            "data": [
                {
                    "date": "2024-09-01",
                    "open": 150.5,
                    "high": 152.3,
                    "low": 149.8,
                    "close": 151.2,
                    "volume": 50000000
                },
                ...
            ]
        }
    """
    ticker = request.args.get('ticker', 'NVDA').upper()
    limit = int(request.args.get('limit', 800))
    
    rows = (PrecoDiario.query
            .filter_by(ticker=ticker)
            .order_by(PrecoDiario.date.desc())
            .limit(limit)
            .all())
    
    data = [{
        "date": r.date.isoformat(),
        "open": r.open,
        "high": r.high,
        "low": r.low,
        "close": r.close,
        "volume": r.volume
    } for r in reversed(rows)]
    
    return jsonify({"ticker": ticker, "data": data})


@api_data_bp.get('/tickers')
def list_tickers():
    """
    Lista todos os tickers disponíveis no banco de dados.
    
    Returns:
        JSON com:
        - tickers: Lista de dicts com [ticker, start, end, rows]
    
    Notes:
        - Retorna estatísticas de cada ticker:
          * ticker: Símbolo
          * start: Primeira data disponível
          * end: Última data disponível
          * rows: Total de registros
        - Ordenado alfabeticamente por ticker
    
    Example:
        GET /api/tickers
        
        Response:
        {
            "tickers": [
                {
                    "ticker": "AAPL",
                    "start": "2015-01-01",
                    "end": "2025-01-14",
                    "rows": 2500
                },
                {
                    "ticker": "NVDA",
                    "start": "2010-01-01",
                    "end": "2025-01-14",
                    "rows": 3750
                }
            ]
        }
    """
    rows = (db.session.query(
                PrecoDiario.ticker,
                db.func.min(PrecoDiario.date),
                db.func.max(PrecoDiario.date),
                db.func.count(PrecoDiario.id)
            )
            .group_by(PrecoDiario.ticker)
            .order_by(PrecoDiario.ticker.asc())
            .all())

    out = []
    for t, dmin, dmax, n in rows:
        out.append({
            "ticker": t,
            "start": dmin.isoformat() if dmin else None,
            "end": dmax.isoformat() if dmax else None,
            "rows": int(n)
        })
    
    return jsonify({"tickers": out})
