def health_check():
    """서비스 상태 확인"""
    try:
        # 기본 상태 정보
        import psutil
        import time
        from datetime import datetime
        
        status = {
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'version': '3.0.0',
            'uptime': time.time(),
            'memory': {
                'used': psutil.Process().memory_info().rss / 1024 / 1024,  # MB
                'percent': psutil.virtual_memory().percent
            },
            'disk': {
                'percent': psutil.disk_usage('/').percent
            },
            'services': {
                'flask': True,
                'github_api': True,  # GitHub 연결 상태는 별도 체크
                'data_access': True
            }
        }
        
        return status
        
    except Exception as e:
        return {
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }
