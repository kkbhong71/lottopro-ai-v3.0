# ===== app_config.py =====
import os
from pathlib import Path

class Config:
    """기본 설정"""
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'lottopro-ai-v3-secret-key'
    
    # GitHub 설정
    GITHUB_REPO = os.environ.get('GITHUB_REPO', 'your-username/lottopro-algorithms')
    GITHUB_TOKEN = os.environ.get('GITHUB_TOKEN', '')
    
    # 데이터 경로
    DATA_PATH = Path('data')
    STATIC_PATH = Path('static')
    
    # 캐시 설정
    CACHE_TYPE = 'simple'
    CACHE_DEFAULT_TIMEOUT = 300
    
    # 세션 설정
    SESSION_TYPE = 'filesystem'
    SESSION_PERMANENT = False
    SESSION_USE_SIGNER = True
    
    # Flask 설정
    JSON_AS_ASCII = False
    JSONIFY_PRETTYPRINT_REGULAR = True
    
    # 보안 설정
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    
    # CORS 설정
    CORS_ORIGINS = ['https://lottopro-ai-v3-0.onrender.com', 'http://localhost:5000']

class DevelopmentConfig(Config):
    """개발 환경 설정"""
    DEBUG = True
    SESSION_COOKIE_SECURE = False
    CORS_ORIGINS = ['*']
    
class ProductionConfig(Config):
    """프로덕션 환경 설정"""
    DEBUG = False
    
    # 에러 모니터링
    SENTRY_DSN = os.environ.get('SENTRY_DSN')
    
    # 성능 설정
    SEND_FILE_MAX_AGE_DEFAULT = 31536000  # 1년
    
    # 보안 강화
    FORCE_HTTPS = True
    
class TestingConfig(Config):
    """테스트 환경 설정"""
    TESTING = True
    WTF_CSRF_ENABLED = False

# 환경별 설정 선택
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}
