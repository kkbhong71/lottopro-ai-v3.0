# ===== Procfile (Render.com용) =====
web: gunicorn app:app --bind 0.0.0.0:$PORT --workers 4 --worker-class eventlet --worker-connections 1000 --timeout 60 --keepalive 5 --max-requests 1000 --max-requests-jitter 50 --preload
