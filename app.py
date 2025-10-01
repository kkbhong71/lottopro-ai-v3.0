from flask import Flask, render_template, request, jsonify, session
from datetime import datetime, timedelta
import json
import os
import pandas as pd
import numpy as np
import requests
from pathlib import Path
import tempfile
import logging
import uuid
import hashlib
import time
from functools import wraps
import importlib.util
import sys
from collections import Counter

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'lottopro-ai-v3-secret-key-2025')
app.config['JSON_AS_ASCII'] = False

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# GitHub 설정 (환경변수 우선, 기본값으로 사용자 저장소)
GITHUB_REPO = os.environ.get('GITHUB_REPO', 'kkbhong71/lottopro-ai-v3.0')
GITHUB_TOKEN = os.environ.get('GITHUB_TOKEN', '')
GITHUB_API_BASE = f'https://api.github.com/repos/{GITHUB_REPO}'

# 알고리즘 실행 제한
ALGORITHM_CACHE = {}
LAST_EXECUTION = {}
EXECUTION_LIMIT = 60  # 60초마다 실행 가능

def rate_limit(limit_seconds=60):
    """API 호출 제한 데코레이터"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            key = f"{request.remote_addr}_{func.__name__}"
            now = time.time()
            
            if key in LAST_EXECUTION:
                if now - LAST_EXECUTION[key] < limit_seconds:
                    return jsonify({
                        'status': 'error', 
                        'message': f'{limit_seconds}초 후에 다시 시도해주세요.'
                    }), 429
            
            LAST_EXECUTION[key] = now
            return func(*args, **kwargs)
        return wrapper
    return decorator

class LottoProAI:
    def __init__(self):
        self.data_path = Path('data')
        self.data_path.mkdir(exist_ok=True)
        
        # 사용자 데이터 저장소
        self.user_data_path = self.data_path / 'user_predictions.json'
        self.algorithm_info_path = Path('algorithms/algorithm_info.json')
        
        # 캐시 디렉토리
        self.cache_path = self.data_path / 'cache'
        self.cache_path.mkdir(exist_ok=True)
        
        # 초기화
        self.load_algorithm_info()
        self.load_lotto_data()
        
    def load_algorithm_info(self):
        """알고리즘 정보 로드"""
        try:
            with open(self.algorithm_info_path, 'r', encoding='utf-8') as f:
                self.algorithm_info = json.load(f)
            
            if 'algorithms' in self.algorithm_info:
                logger.info(f"Loaded {len(self.algorithm_info.get('algorithms', {}))} algorithms")
            else:
                logger.warning("Converting old algorithm info format to new format")
                algorithms_dict = {}
                for key, value in self.algorithm_info.items():
                    if isinstance(value, dict) and 'name' in value:
                        algorithms_dict[key] = value
                
                self.algorithm_info = {
                    "version": "3.0",
                    "algorithms": algorithms_dict,
                    "categories": {},
                    "difficulty_levels": {}
                }
                logger.info(f"Converted {len(algorithms_dict)} algorithms")
                
        except FileNotFoundError:
            logger.warning("Algorithm info file not found, using default")
            self.algorithm_info = {
                "version": "3.0",
                "algorithms": {},
                "categories": {},
                "difficulty_levels": {}
            }
            
    def load_lotto_data(self):
        """로또 당첨번호 데이터 로드"""
        try:
            # 최신 회차 파일로 변경 (1191)
            csv_path = self.data_path / 'new_1191.csv'
            if not csv_path.exists():
                csv_path = Path('new_1191.csv')
            
            self.lotto_df = pd.read_csv(csv_path)
            
            # 컬럼명 표준화
            expected_columns = ['round', 'draw date', 'num1', 'num2', 'num3', 'num4', 'num5', 'num6', 'bonus num']
            if list(self.lotto_df.columns) == expected_columns:
                logger.info(f"Loaded {len(self.lotto_df)} lottery records (최신 회차: 1191)")
            else:
                logger.warning(f"Column names may not match expected format: {list(self.lotto_df.columns)}")
                
        except FileNotFoundError:
            logger.error("Lottery data file not found (new_1191.csv)")
            self.lotto_df = pd.DataFrame()
        except Exception as e:
            logger.error(f"Error loading lottery data: {str(e)}")
            self.lotto_df = pd.DataFrame()
    
    def get_algorithm_cache_key(self, algorithm_id):
        """알고리즘 캐시 키 생성"""
        data_hash = hashlib.md5(str(len(self.lotto_df)).encode()).hexdigest()[:8]
        return f"{algorithm_id}_{data_hash}"
    
    def execute_github_algorithm(self, algorithm_id, user_numbers=None):
        """GitHub에서 알고리즘 코드를 안전하게 실행"""
        try:
            # 알고리즘 정보 확인
            if algorithm_id not in self.algorithm_info.get('algorithms', {}):
                raise Exception(f"Algorithm '{algorithm_id}' not found")
            
            algorithm_info = self.algorithm_info['algorithms'][algorithm_id]
            
            # 캐시 확인 (user_numbers가 없을 때만)
            if not user_numbers:
                cache_key = self.get_algorithm_cache_key(algorithm_id)
                cache_file = self.cache_path / f"{cache_key}.json"
                
                # 캐시된 결과가 있고 1시간 이내라면 사용
                if cache_file.exists():
                    cache_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
                    if datetime.now() - cache_time < timedelta(hours=1):
                        with open(cache_file, 'r', encoding='utf-8') as f:
                            cached_result = json.load(f)
                            cached_result['cached'] = True
                            logger.info(f"Using cached result for {algorithm_id}")
                            return cached_result
            
            # GitHub에서 코드 다운로드
            algorithm_path = algorithm_info.get('github_path', f'algorithms/{algorithm_id}.py')
            github_url = f'https://raw.githubusercontent.com/{GITHUB_REPO}/main/{algorithm_path}'
            
            headers = {}
            if GITHUB_TOKEN:
                headers['Authorization'] = f'token {GITHUB_TOKEN}'
            
            logger.info(f"Downloading algorithm from: {github_url}")
            response = requests.get(github_url, headers=headers, timeout=30)
            
            if response.status_code != 200:
                raise Exception(f"Failed to download algorithm: HTTP {response.status_code}")
            
            code_content = response.text
            
            # 보안 검사 (기본적인 위험 코드 패턴 체크)
            dangerous_patterns = [
                'import os', 'subprocess', 'system(',
                'rm ', 'del ', 'remove(', 'shutil'
            ]
            
            for pattern in dangerous_patterns:
                if pattern in code_content:
                    logger.warning(f"Potentially dangerous pattern found: {pattern}")
            
            # 제한된 import 함수 정의 (수정됨!)
            def safe_import(name, *args, **kwargs):
                """보안을 위해 제한된 모듈만 import 허용"""
                # ✅ 수정: Set에 모든 항목을 올바르게 포함
                allowed_modules = {
                    'random', 'math', 'datetime', 'collections', 
                    'itertools', 'functools', 're', 'statistics',
                    'operator', 'bisect', 'heapq', 'array',
                    'pandas', 'numpy', 'pd', 'np',
                    'warnings'  # 이제 올바르게 Set의 일부
                }
                if name in allowed_modules:
                    return __import__(name, *args, **kwargs)
                raise ImportError(f"Module '{name}' is not allowed for security reasons")
            
            # 안전한 실행 환경 구성
            safe_globals = {
                '__builtins__': {
                    # 기본 함수들
                    'len': len, 'range': range, 'enumerate': enumerate,
                    'zip': zip, 'map': map, 'filter': filter,
                    'sum': sum, 'max': max, 'min': min, 'abs': abs,
                    'round': round, 'int': int, 'float': float,
                    'str': str, 'list': list, 'dict': dict, 'set': set,
                    'tuple': tuple, 'bool': bool, 'type': type,
                    'sorted': sorted, 'reversed': reversed,
                    'any': any, 'all': all,
                    'isinstance': isinstance,
                    '__import__': safe_import,
                    'print': print,
                    'globals': lambda: safe_globals  # globals() 지원
                },
                'pd': pd,
                'np': np,
                'Counter': Counter,
                'lotto_data': self.lotto_df.copy(),
                'data_path': str(self.data_path),
                'datetime': datetime,
                'random': np.random,
                'user_numbers': user_numbers or []
            }
            
            # 코드 실행 (에러 핸들링 강화)
            logger.info(f"Executing algorithm: {algorithm_id}")
            try:
                exec(code_content, safe_globals)
            except SyntaxError as e:
                raise Exception(f"Syntax error in algorithm code: {str(e)}")
            except Exception as e:
                raise Exception(f"Runtime error: {str(e)}")
            
            # 예측 함수 호출 (여러 함수명 시도)
            result = None
            for func_name in ['predict_numbers', 'predict', 'generate_numbers', 'main']:
                if func_name in safe_globals:
                    logger.info(f"Calling function: {func_name}")
                    if user_numbers:
                        result = safe_globals[func_name](user_numbers)
                    else:
                        result = safe_globals[func_name]()
                    break
            
            if result is None:
                raise Exception("No prediction function found (tried: predict_numbers, predict, generate_numbers, main)")
            
            # 결과 검증
            if not isinstance(result, (list, tuple)) or len(result) != 6:
                raise Exception(f"Algorithm must return exactly 6 numbers, got {len(result) if isinstance(result, (list, tuple)) else 'non-list'}")
            
            # 번호 범위 검증 (1-45)
            if not all(isinstance(n, (int, np.integer)) and 1 <= n <= 45 for n in result):
                raise Exception("All numbers must be integers between 1 and 45")
            
            # 중복 검사
            if len(set(result)) != 6:
                raise Exception("All numbers must be unique")
            
            # 결과 정리
            prediction_result = {
                'status': 'success',
                'numbers': sorted(list(map(int, result))),
                'algorithm': algorithm_id,
                'algorithm_name': algorithm_info.get('name', algorithm_id),
                'accuracy_rate': algorithm_info.get('accuracy_rate', algorithm_info.get('accuracy', 0)),
                'timestamp': datetime.now().isoformat(),
                'cached': False
            }
            
            # 결과 캐싱 (user_numbers가 없을 때만)
            if not user_numbers:
                cache_file = self.cache_path / f"{self.get_algorithm_cache_key(algorithm_id)}.json"
                with open(cache_file, 'w', encoding='utf-8') as f:
                    json.dump(prediction_result, f, ensure_ascii=False, indent=2)
                logger.info(f"Cached result for {algorithm_id}")
            
            return prediction_result
                
        except requests.RequestException as e:
            logger.error(f"Network error downloading algorithm: {str(e)}")
            return {
                'status': 'error',
                'message': f'네트워크 오류: {str(e)}',
                'algorithm': algorithm_id
            }
        except Exception as e:
            logger.error(f"Algorithm execution failed: {str(e)}", exc_info=True)
            return {
                'status': 'error',
                'message': str(e),
                'algorithm': algorithm_id
            }
    
    def save_user_prediction(self, user_id, prediction_data):
        """사용자 예측 저장"""
        try:
            # 기존 데이터 로드
            if self.user_data_path.exists():
                with open(self.user_data_path, 'r', encoding='utf-8') as f:
                    user_data = json.load(f)
            else:
                user_data = {}
            
            # 사용자별 데이터 구조 초기화
            if user_id not in user_data:
                user_data[user_id] = {
                    'created_at': datetime.now().isoformat(),
                    'predictions': [],
                    'stats': {
                        'total_predictions': 0,
                        'total_matches': 0,
                        'best_match': 0,
                        'algorithm_usage': {}
                    }
                }
            
            # 예측 데이터 추가
            prediction_entry = {
                'id': str(uuid.uuid4()),
                'numbers': prediction_data['numbers'],
                'algorithm': prediction_data['algorithm'],
                'algorithm_name': prediction_data.get('algorithm_name', ''),
                'timestamp': prediction_data.get('timestamp', datetime.now().isoformat()),
                'round_predicted': prediction_data.get('round_predicted', 1191),  # 업데이트된 회차
                'is_checked': False,
                'match_result': None,
                'cached': prediction_data.get('cached', False)
            }
            
            user_data[user_id]['predictions'].append(prediction_entry)
            user_data[user_id]['stats']['total_predictions'] += 1
            
            # 알고리즘 사용 통계
            algo_stats = user_data[user_id]['stats']['algorithm_usage']
            algo_id = prediction_data['algorithm']
            algo_stats[algo_id] = algo_stats.get(algo_id, 0) + 1
            
            # 파일 저장
            with open(self.user_data_path, 'w', encoding='utf-8') as f:
                json.dump(user_data, f, ensure_ascii=False, indent=2)
            
            return {'status': 'success', 'prediction_id': prediction_entry['id']}
            
        except Exception as e:
            logger.error(f"Failed to save prediction: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    def compare_with_winning_numbers(self, user_id, prediction_id, winning_numbers):
        """당첨번호와 비교"""
        try:
            with open(self.user_data_path, 'r', encoding='utf-8') as f:
                user_data = json.load(f)
            
            if user_id not in user_data:
                return {'status': 'error', 'message': 'User not found'}
            
            # 해당 예측 찾기
            prediction = None
            for pred in user_data[user_id]['predictions']:
                if pred['id'] == prediction_id:
                    prediction = pred
                    break
            
            if not prediction:
                return {'status': 'error', 'message': 'Prediction not found'}
            
            # 매치 수 계산
            predicted_numbers = set(prediction['numbers'])
            winning_set = set(winning_numbers[:6])  # 보너스 번호 제외
            matches = len(predicted_numbers.intersection(winning_set))
            
            # 등수 계산
            prize_info = {
                6: {'rank': '1등', 'description': '6개 일치'},
                5: {'rank': '2등', 'description': '5개 일치'},
                4: {'rank': '3등', 'description': '4개 일치'},
                3: {'rank': '4등', 'description': '3개 일치'},
                2: {'rank': '5등', 'description': '2개 일치'},
                1: {'rank': '6등', 'description': '1개 일치'},
                0: {'rank': '낙첨', 'description': '일치 없음'}
            }
            
            # 결과 저장
            prediction['is_checked'] = True
            prediction['match_result'] = {
                'matches': matches,
                'winning_numbers': winning_numbers,
                'matched_numbers': sorted(list(predicted_numbers.intersection(winning_set))),
                'prize_info': prize_info.get(matches, prize_info[0]),
                'check_date': datetime.now().isoformat()
            }
            
            # 통계 업데이트
            user_data[user_id]['stats']['total_matches'] += matches
            if matches > user_data[user_id]['stats']['best_match']:
                user_data[user_id]['stats']['best_match'] = matches
            
            # 저장
            with open(self.user_data_path, 'w', encoding='utf-8') as f:
                json.dump(user_data, f, ensure_ascii=False, indent=2)
            
            return {
                'status': 'success',
                'matches': matches,
                'result': prediction['match_result']
            }
            
        except Exception as e:
            logger.error(f"Comparison failed: {str(e)}")
            return {'status': 'error', 'message': str(e)}

# 전역 인스턴스 생성
lotto_ai = LottoProAI()

# ===== 라우트 정의 =====

@app.route('/')
def index():
    """메인 페이지"""
    algorithm_count = len(lotto_ai.algorithm_info.get('algorithms', {}))
    data_count = len(lotto_ai.lotto_df) if not lotto_ai.lotto_df.empty else 0
    
    return render_template('index.html', 
                         algorithm_count=algorithm_count,
                         data_count=data_count,
                         latest_round=1191,  # 업데이트된 회차
                         version="3.0")

@app.route('/algorithms')
def algorithms():
    """알고리즘 선택 페이지"""
    algorithms_data = lotto_ai.algorithm_info.get('algorithms', {})
    categories = lotto_ai.algorithm_info.get('categories', {})
    difficulty_levels = lotto_ai.algorithm_info.get('difficulty_levels', {})
    
    return render_template('algorithm.html', 
                         algorithms=algorithms_data,
                         categories=categories,
                         difficulty_levels=difficulty_levels)

@app.route('/api/execute/<algorithm_id>')
@rate_limit(60)
def execute_algorithm(algorithm_id):
    """알고리즘 실행 API (GET)"""
    if algorithm_id not in lotto_ai.algorithm_info.get('algorithms', {}):
        return jsonify({'status': 'error', 'message': 'Algorithm not found'}), 404
    
    result = lotto_ai.execute_github_algorithm(algorithm_id)
    return jsonify(result)

@app.route('/api/predict', methods=['POST'])
@rate_limit(60)
def predict_numbers():
    """알고리즘 예측 API (POST)"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'status': 'error',
                'message': 'No data provided'
            }), 400
        
        algorithm_id = data.get('algorithm')
        user_numbers = data.get('user_numbers', [])
        
        if not algorithm_id:
            return jsonify({
                'status': 'error',
                'message': 'Algorithm ID is required'
            }), 400
        
        if algorithm_id not in lotto_ai.algorithm_info.get('algorithms', {}):
            return jsonify({
                'status': 'error',
                'message': f'Algorithm "{algorithm_id}" not found'
            }), 404
        
        # 사용자 번호 검증 (선택사항)
        if user_numbers:
            if not isinstance(user_numbers, list):
                return jsonify({
                    'status': 'error',
                    'message': 'user_numbers must be a list'
                }), 400
            
            if not all(isinstance(n, int) and 1 <= n <= 45 for n in user_numbers):
                return jsonify({
                    'status': 'error',
                    'message': 'All user numbers must be integers between 1 and 45'
                }), 400
            
            if len(set(user_numbers)) != len(user_numbers):
                return jsonify({
                    'status': 'error',
                    'message': 'User numbers must be unique'
                }), 400
        
        # 알고리즘 실행
        result = lotto_ai.execute_github_algorithm(algorithm_id, user_numbers)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Predict API error: {str(e)}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': f'Internal server error: {str(e)}'
        }), 500

@app.route('/api/save-prediction', methods=['POST'])
def save_prediction():
    """예측 저장 API"""
    data = request.get_json()
    
    # 세션에서 사용자 ID 가져오기 또는 생성
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())
    
    user_id = session['user_id']
    result = lotto_ai.save_user_prediction(user_id, data)
    
    return jsonify(result)

@app.route('/api/compare-numbers', methods=['POST'])
def compare_numbers():
    """당첨번호 비교 API"""
    data = request.get_json()
    
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'status': 'error', 'message': 'User session not found'}), 400
    
    # 당첨번호 유효성 검사
    winning_numbers = data.get('winning_numbers', [])
    if not isinstance(winning_numbers, list) or len(winning_numbers) < 6:
        return jsonify({'status': 'error', 'message': 'Invalid winning numbers'}), 400
    
    result = lotto_ai.compare_with_winning_numbers(
        user_id, 
        data['prediction_id'], 
        winning_numbers
    )
    
    return jsonify(result)

@app.route('/api/user-predictions')
def get_user_predictions():
    """사용자 예측 목록 조회"""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({
            'predictions': [], 
            'stats': {
                'total_predictions': 0, 
                'total_matches': 0, 
                'best_match': 0,
                'algorithm_usage': {}
            }
        })
    
    try:
        if lotto_ai.user_data_path.exists():
            with open(lotto_ai.user_data_path, 'r', encoding='utf-8') as f:
                user_data = json.load(f)
                user_info = user_data.get(user_id, {
                    'predictions': [], 
                    'stats': {
                        'total_predictions': 0, 
                        'total_matches': 0, 
                        'best_match': 0,
                        'algorithm_usage': {}
                    }
                })
                return jsonify(user_info)
    except Exception as e:
        logger.error(f"Failed to load user predictions: {str(e)}")
    
    return jsonify({
        'predictions': [], 
        'stats': {
            'total_predictions': 0, 
            'total_matches': 0, 
            'best_match': 0,
            'algorithm_usage': {}
        }
    })

@app.route('/saved-numbers')
def saved_numbers():
    """저장된 번호 관리 페이지"""
    return render_template('saved_numbers.html')

@app.route('/compare')
def compare():
    """당첨번호 비교 페이지"""
    return render_template('compare.html')

@app.route('/statistics')
def statistics():
    """통계 분석 페이지"""
    return render_template('statistics.html')

@app.route('/api/lottery-data')
def get_lottery_data():
    """로또 데이터 API"""
    try:
        if lotto_ai.lotto_df.empty:
            return jsonify({
                'status': 'error',
                'message': 'No lottery data available'
            })
            
        return jsonify({
            'status': 'success',
            'data': lotto_ai.lotto_df.to_dict('records'),
            'total_records': len(lotto_ai.lotto_df),
            'latest_round': 1191  # 업데이트된 회차
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/algorithm-info')
def get_all_algorithm_info():
    """전체 알고리즘 정보 조회"""
    try:
        # 중복 제거된 알고리즘 정보 반환
        algorithms = lotto_ai.algorithm_info.get('algorithms', {})
        
        # 중복 제거
        unique_algorithms = {}
        for key, value in algorithms.items():
            if key not in unique_algorithms:
                unique_algorithms[key] = value
        
        return jsonify({
            'status': 'success',
            'info': unique_algorithms,
            'count': len(unique_algorithms),
            'latest_round': 1191,  # 업데이트된 회차
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Failed to get algorithm info: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/algorithm-info/<algorithm_id>')
def get_algorithm_info(algorithm_id):
    """특정 알고리즘 정보 조회"""
    algorithms = lotto_ai.algorithm_info.get('algorithms', {})
    if algorithm_id not in algorithms:
        return jsonify({'status': 'error', 'message': 'Algorithm not found'}), 404
    
    return jsonify({
        'status': 'success',
        'algorithm': algorithms[algorithm_id]
    })

@app.route('/api/health')
def health_check():
    """서비스 상태 확인"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'algorithms_loaded': len(lotto_ai.algorithm_info.get('algorithms', {})),
        'data_records': len(lotto_ai.lotto_df) if not lotto_ai.lotto_df.empty else 0,
        'latest_round': 1191,  # 업데이트된 회차
        'version': '3.0'
    })

# ===== 에러 핸들러 =====

@app.errorhandler(404)
def not_found(error):
    if request.path.startswith('/api/'):
        return jsonify({
            'status': 'error',
            'message': 'API endpoint not found'
        }), 404
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {str(error)}", exc_info=True)
    if request.path.startswith('/api/'):
        return jsonify({
            'status': 'error',
            'message': 'Internal server error'
        }), 500
    return render_template('500.html'), 500

# ===== CORS 설정 =====

@app.after_request
def after_request(response):
    """모든 응답에 CORS 헤더 추가"""
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,PUT,DELETE,OPTIONS')
    return response

# ===== 서버 실행 =====

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    logger.info(f"Starting LottoPro-AI v3.0 server on port {port}")
    logger.info(f"Debug mode: {debug_mode}")
    logger.info(f"Algorithms loaded: {len(lotto_ai.algorithm_info.get('algorithms', {}))}")
    logger.info(f"Lottery data records: {len(lotto_ai.lotto_df) if not lotto_ai.lotto_df.empty else 0}")
    logger.info(f"Latest round: 1191")
    
    app.run(host='0.0.0.0', port=port, debug=debug_mode)
