"""
Auth Helpers - Funções auxiliares para autenticação e autorização.

Este módulo contém funções utilitárias para:
- Validação de API Key
- Autenticação básica HTTP
- Proteção de endpoints

Autor: Sistema de Trading LSTM
Data: 2025-01-14
"""

import os
from functools import wraps
from flask import request, Response


def _auth_ok(req) -> bool:
    """
    Valida autenticação via API Key.
    
    Args:
        req: Objeto flask.request
    
    Returns:
        True se autenticação válida, False caso contrário
        
    Rules:
        1. Em DEV (127.0.0.1 ou ::1), sempre retorna True
        2. Se DISABLE_API_KEY=1, sempre retorna True
        3. Caso contrário, valida header X-API-KEY contra API_KEY
        
    Notes:
        - Variável de ambiente API_KEY deve estar definida em produção
        - DISABLE_API_KEY=1 desabilita autenticação (apenas para DEV!)
    """
    disable_api_key = os.getenv('DISABLE_API_KEY') == '1'
    
    # Em DEV ou se desabilitado, não exige autenticação
    if disable_api_key or req.remote_addr in ('127.0.0.1', '::1'):
        return True
    
    # Valida API Key
    api_key = os.getenv('API_KEY', 'change-me')
    return req.headers.get('X-API-KEY') == api_key


def require_basic_auth(f):
    """
    Decorator para exigir autenticação HTTP Basic.
    
    Args:
        f: Função de rota Flask a ser protegida
    
    Returns:
        Função decorada com validação de autenticação
        
    Usage:
        @api_bp.post('/admin/endpoint')
        @require_basic_auth
        def admin_endpoint():
            return {'data': 'sensitive'}
    
    Environment Variables:
        - ADMIN_USER: Username para autenticação (default: 'admin')
        - ADMIN_PASS: Password para autenticação (se vazio, não exige senha em DEV)
        
    Notes:
        - Se ADMIN_PASS não estiver definido, permite acesso sem senha (DEV)
        - Retorna 401 Unauthorized com prompt de autenticação se falhar
    """
    @wraps(f)
    def wrapper(*args, **kwargs):
        user = os.getenv('ADMIN_USER', 'admin')
        pw = os.getenv('ADMIN_PASS')
        
        # Se senha não configurada, permite acesso (DEV)
        if not pw:
            return f(*args, **kwargs)
        
        # Valida credenciais
        auth = request.authorization
        if not auth or auth.username != user or auth.password != pw:
            return Response(
                'Autenticação necessária',
                401,
                {'WWW-Authenticate': 'Basic realm="Admin Area"'}
            )
        
        return f(*args, **kwargs)
    
    return wrapper
