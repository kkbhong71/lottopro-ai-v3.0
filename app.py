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
import builtins
import re

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'lottopro-ai-v3-secret-key-2025')
app.config['JSON_AS_ASCII'] = False

# 로깅 설정 - 더 상세한 정보 포함
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# GitHub 설정
GITHUB_REPO = os.environ.get('GITHUB_REPO', 'kkbhong71/lottopro-ai-v3.0')
GITHUB_TOKEN = os.environ.get('GITHUB_TOKEN', '')
GITHUB_API_BASE = f'https://api.github.com/repos/{GITHUB_REPO}'

# 알고리즘 실행 제한
ALGORITHM_CACHE = {}
LAST_EXECUTION = {}
EXECUTION_LIMIT = 60

# 🆕 데이터 흐름 검증을 위한 전역 변수
DATA_FLOW_STATS = {
    'csv_load_time': None,
    'csv_load_success': False,
    'total_records': 0,
    'last_algorithm_execution': None,
    'data_validation_results': {}
}

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

@app.before_request
def ensure_session():
    """모든 요청 전에 세션 ID 확인 및 생성"""
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())
        logger.info(f"New user session created: {session['user_id']}")

class LottoProAI:
    def __init__(self):
        self.data_path = Path('data')
        self.data_path.mkdir(exist_ok=True)
        
        self.user_data_path = self.data_path / 'user_predictions.json'
        self.algorithm_info_path = Path('algorithms/algorithm_info.json')
        
        self.cache_path = self.data_path / 'cache'
        self.cache_path.mkdir(exist_ok=True)
        
        # 🆕 데이터 검증 결과 저장
        self.data_validation = {
            'csv_found': False,
            'csv_path': None,
            'load_timestamp': None,
            'records_loaded': 0,
            'columns_verified': False,
            'data_quality': {}
        }
        
        self.load_algorithm_info()
        self.load_lotto_data()
        
    def load_algorithm_info(self):
        """알고리즘 정보 로드"""
        try:
            with open(self.algorithm_info_path, 'r', encoding='utf-8') as f:
                self.algorithm_info = json.load(f)
            
            if 'algorithms' in self.algorithm_info:
                logger.info(f"✅ Loaded {len(self.algorithm_info.get('algorithms', {}))} algorithms")
            else:
                logger.warning("⚠️ Converting old algorithm info format to new format")
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
                logger.info(f"✅ Converted {len(algorithms_dict)} algorithms")
                
        except FileNotFoundError:
            logger.warning("⚠️ Algorithm info file not found, using default")
            self.algorithm_info = {
                "version": "3.0",
                "algorithms": {},
                "categories": {},
                "difficulty_levels": {}
            }
            
    def load_lotto_data(self):
        """로또 당첨번호 데이터 로드 - 🆕 검증 기능 강화"""
        try:
            # 여러 가능한 경로 시도
            possible_paths = [
                self.data_path / 'new_1194.csv',
                Path('data/new_1194.csv'),
                Path('new_1194.csv'),
                Path('/opt/render/project/src/data/new_1194.csv'),
                Path('/opt/render/project/src/new_1194.csv')
            ]
            
            csv_path = None
            logger.info("🔍 CSV 파일 검색 시작...")
            
            for path in possible_paths:
                logger.info(f"  시도: {path}")
                if path.exists():
                    csv_path = path
                    logger.info(f"✅ CSV 파일 발견: {path}")
                    self.data_validation['csv_found'] = True
                    self.data_validation['csv_path'] = str(path)
                    break
            
            if csv_path is None:
                logger.error(f"❌ CSV 파일을 찾을 수 없음. 시도한 경로:")
                for path in possible_paths:
                    logger.error(f"  - {path} (exists: {path.exists()})")
                raise FileNotFoundError("new_1194.csv 파일을 찾을 수 없습니다")
            
            # CSV 파일 로드
            load_start_time = time.time()
            self.lotto_df = pd.read_csv(csv_path)
            load_duration = time.time() - load_start_time
            
            # 🆕 검증 1: 기본 정보
            logger.info("=" * 70)
            logger.info("📊 CSV 데이터 로드 검증")
            logger.info("=" * 70)
            logger.info(f"✅ 로드 시간: {load_duration:.3f}초")
            logger.info(f"✅ 총 회차: {len(self.lotto_df)}")
            logger.info(f"✅ 원본 컬럼: {list(self.lotto_df.columns)}")
            
            self.data_validation['load_timestamp'] = datetime.now().isoformat()
            self.data_validation['records_loaded'] = len(self.lotto_df)
            self.data_validation['load_duration'] = load_duration
            
            # 🆕 검증 2: 샘플 데이터
            if not self.lotto_df.empty:
                first_row = self.lotto_df.iloc[0].to_dict()
                last_row = self.lotto_df.iloc[-1].to_dict()
                logger.info(f"🎲 첫 회차: {first_row}")
                logger.info(f"🎲 최신 회차: {last_row}")
                
                self.data_validation['first_record'] = first_row
                self.data_validation['latest_record'] = last_row
            
            # 🆕 검증 3: 데이터 품질 검사
            expected_columns = ['round', 'draw date', 'num1', 'num2', 'num3', 'num4', 'num5', 'num6', 'bonus num']
            actual_columns = list(self.lotto_df.columns)
            
            logger.info(f"📋 컬럼 검증:")
            if actual_columns == expected_columns:
                logger.info(f"  ✅ 컬럼 형식 정상")
                self.data_validation['columns_verified'] = True
            else:
                logger.warning(f"  ⚠️ 컬럼명 불일치:")
                logger.warning(f"    예상: {expected_columns}")
                logger.warning(f"    실제: {actual_columns}")
                self.data_validation['columns_verified'] = False
            
            # 🆕 검증 4: 번호 컬럼 품질
            number_cols = ['num1', 'num2', 'num3', 'num4', 'num5', 'num6']
            quality_report = {}
            
            logger.info(f"🔢 번호 데이터 품질:")
            for col in number_cols:
                if col in self.lotto_df.columns:
                    null_count = self.lotto_df[col].isnull().sum()
                    invalid_count = ((self.lotto_df[col] < 1) | (self.lotto_df[col] > 45)).sum()
                    valid_count = len(self.lotto_df) - null_count - invalid_count
                    
                    quality_report[col] = {
                        'total': len(self.lotto_df),
                        'valid': int(valid_count),
                        'null': int(null_count),
                        'out_of_range': int(invalid_count),
                        'quality_percentage': round(valid_count / len(self.lotto_df) * 100, 2)
                    }
                    
                    logger.info(f"  - {col}: 유효={valid_count}, NULL={null_count}, 범위외={invalid_count} "
                              f"({quality_report[col]['quality_percentage']}% ✅)")
                else:
                    logger.warning(f"  - {col}: ❌ 컬럼 없음!")
                    quality_report[col] = {'error': 'column_not_found'}
            
            self.data_validation['data_quality'] = quality_report
            
            # 🆕 검증 5: 통계 요약
            if all(col in self.lotto_df.columns for col in number_cols):
                logger.info(f"📈 데이터 통계:")
                for col in number_cols:
                    stats = self.lotto_df[col].describe()
                    logger.info(f"  - {col}: min={stats['min']}, max={stats['max']}, "
                              f"mean={stats['mean']:.2f}, std={stats['std']:.2f}")
            
            logger.info("=" * 70)
            logger.info(f"✅ 로또 데이터 로드 완료 - {len(self.lotto_df)}회차")
            logger.info("=" * 70)
            
            # 🆕 전역 통계 업데이트
            DATA_FLOW_STATS['csv_load_time'] = datetime.now().isoformat()
            DATA_FLOW_STATS['csv_load_success'] = True
            DATA_FLOW_STATS['total_records'] = len(self.lotto_df)
            DATA_FLOW_STATS['data_validation_results'] = quality_report
                
        except FileNotFoundError as e:
            logger.error(f"❌ CSV 파일 없음: {str(e)}")
            self.lotto_df = pd.DataFrame()
            self.data_validation['error'] = str(e)
            DATA_FLOW_STATS['csv_load_success'] = False
        except pd.errors.EmptyDataError:
            logger.error("❌ CSV 파일이 비어있음")
            self.lotto_df = pd.DataFrame()
            self.data_validation['error'] = 'empty_csv'
            DATA_FLOW_STATS['csv_load_success'] = False
        except Exception as e:
            logger.error(f"❌ 데이터 로드 실패: {str(e)}", exc_info=True)
            self.lotto_df = pd.DataFrame()
            self.data_validation['error'] = str(e)
            DATA_FLOW_STATS['csv_load_success'] = False
    
    def check_dangerous_code(self, code_content):
        """개선된 보안 검사 - 정규표현식 사용"""
        dangerous_patterns = [
            (r'\brm\s+-[rf]', 'Shell command: rm -rf (파일 삭제)'),
            (r'\bos\.system\s*\(', 'os.system() 호출'),
            (r'\bos\.remove\s*\(', 'os.remove() 호출'),
            (r'\bos\.rmdir\s*\(', 'os.rmdir() 호출'),
            (r'\bos\.unlink\s*\(', 'os.unlink() 호출'),
            (r'\bsubprocess\.', 'subprocess 모듈 사용'),
            (r'\bshutil\.rmtree\s*\(', 'shutil.rmtree() 호출'),
            (r'\bexec\s*\(', 'exec() 호출'),
            (r'\beval\s*\(', 'eval() 호출'),
            (r'\b__import__\s*\(', '동적 import'),
            (r'\bopen\s*\([^)]*[\'"]w[\'"]', '파일 쓰기 모드'),
        ]
        
        found_issues = []
        for pattern, description in dangerous_patterns:
            if re.search(pattern, code_content, re.IGNORECASE):
                found_issues.append(description)
                logger.warning(f"⚠️ Potentially dangerous pattern: {description}")
        
        return found_issues
    
    def get_algorithm_cache_key(self, algorithm_id):
        """알고리즘 캐시 키 생성"""
        data_hash = hashlib.md5(str(len(self.lotto_df)).encode()).hexdigest()[:8]
        return f"{algorithm_id}_{data_hash}"
    
    def execute_github_algorithm(self, algorithm_id, user_numbers=None):
        """GitHub에서 알고리즘 코드를 안전하게 실행 - 🆕 검증 기능 강화"""
        execution_log = {
            'algorithm_id': algorithm_id,
            'start_time': datetime.now().isoformat(),
            'user_numbers': user_numbers,
            'data_transfer_verified': False,
            'execution_success': False
        }
        
        try:
            if algorithm_id not in self.algorithm_info.get('algorithms', {}):
                raise Exception(f"Algorithm '{algorithm_id}' not found")
            
            # 🆕 검증 1: 데이터 로드 상태 확인
            if self.lotto_df.empty:
                logger.error("❌ 로또 데이터가 비어있음 - 알고리즘 실행 불가")
                execution_log['error'] = 'empty_dataframe'
                raise Exception("로또 데이터를 로드할 수 없습니다")
            
            algorithm_info = self.algorithm_info['algorithms'][algorithm_id]
            
            # 캐시 확인
            if not user_numbers:
                cache_key = self.get_algorithm_cache_key(algorithm_id)
                cache_file = self.cache_path / f"{cache_key}.json"
                
                if cache_file.exists():
                    cache_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
                    if datetime.now() - cache_time < timedelta(hours=1):
                        with open(cache_file, 'r', encoding='utf-8') as f:
                            cached_result = json.load(f)
                            cached_result['cached'] = True
                            logger.info(f"✅ Cached result for {algorithm_id}")
                            return cached_result
            
            # GitHub에서 알고리즘 다운로드
            algorithm_path = algorithm_info.get('github_path', f'algorithms/{algorithm_id}.py')
            github_url = f'https://raw.githubusercontent.com/{GITHUB_REPO}/main/{algorithm_path}'
            
            headers = {}
            if GITHUB_TOKEN:
                headers['Authorization'] = f'token {GITHUB_TOKEN}'
            
            logger.info(f"📥 Downloading algorithm from: {github_url}")
            response = requests.get(github_url, headers=headers, timeout=30)
            
            if response.status_code != 200:
                raise Exception(f"Failed to download algorithm: HTTP {response.status_code}")
            
            code_content = response.text
            
            # 보안 검사
            dangerous_issues = self.check_dangerous_code(code_content)
            if dangerous_issues:
                logger.warning(f"⚠️ Security check found {len(dangerous_issues)} potential issues in {algorithm_id}")
                for issue in dangerous_issues:
                    logger.warning(f"  - {issue}")
            
            # 🆕 검증 2: 데이터 전달 전 로깅
            logger.info("=" * 70)
            logger.info(f"🚀 알고리즘 실행 준비: {algorithm_id}")
            logger.info("=" * 70)
            logger.info(f"📊 전달할 데이터:")
            logger.info(f"  - 회차 수: {len(self.lotto_df)}")
            logger.info(f"  - 컬럼: {list(self.lotto_df.columns)}")
            
            if not self.lotto_df.empty:
                latest_numbers = self.lotto_df.iloc[-1][['num1', 'num2', 'num3', 'num4', 'num5', 'num6']].tolist()
                logger.info(f"  - 최신 회차 샘플: {latest_numbers}")
                execution_log['latest_sample'] = latest_numbers
            
            execution_log['data_rows'] = len(self.lotto_df)
            execution_log['data_columns'] = list(self.lotto_df.columns)
            
            # 🆕 검증 3: 디버깅 코드 주입
            debug_code = """
# ===== 데이터 수신 검증 코드 =====
import sys
_verification_passed = False

try:
    if 'lotto_data' in globals():
        print(f"✅ [VERIFY] lotto_data 수신 성공: {len(lotto_data)}회차")
        print(f"✅ [VERIFY] 컬럼: {list(lotto_data.columns)}")
        
        if not lotto_data.empty:
            sample_row = lotto_data.iloc[-1]
            sample_numbers = [sample_row['num1'], sample_row['num2'], sample_row['num3'], 
                            sample_row['num4'], sample_row['num5'], sample_row['num6']]
            print(f"✅ [VERIFY] 최신 회차 샘플: {sample_numbers}")
            _verification_passed = True
        else:
            print("❌ [VERIFY] lotto_data가 비어있음!")
    else:
        print("❌ [VERIFY] lotto_data가 globals()에 없음!")
except Exception as e:
    print(f"❌ [VERIFY] 검증 중 오류: {str(e)}")

if not _verification_passed:
    print("⚠️ [VERIFY] 데이터 검증 실패 - fallback 모드로 전환 가능")

# ===== 검증 코드 끝 =====

"""
            
            # Safe globals 구성
            original_import = builtins.__import__
            
            def safe_import(name, *args, **kwargs):
                """보안을 위해 제한된 모듈만 import 허용"""
                allowed_modules = {
                    'random', 'math', 'datetime', 'collections', 
                    'itertools', 'functools', 're', 'statistics',
                    'operator', 'bisect', 'heapq', 'array',
                    'pandas', 'numpy', 'pd', 'np',
                    'warnings'
                }
                if name in allowed_modules:
                    return original_import(name, *args, **kwargs)
                raise ImportError(f"Module '{name}' is not allowed for security reasons")
            
            safe_globals = {
                '__builtins__': {
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
                    '__build_class__': builtins.__build_class__,
                    '__name__': '__main__',
                    'print': print,
                    'globals': lambda: safe_globals,
                    'Exception': Exception,
                    'ValueError': ValueError,
                    'TypeError': TypeError,
                    'KeyError': KeyError,
                    'IndexError': IndexError,
                    'AttributeError': AttributeError,
                    'ImportError': ImportError,
                    'RuntimeError': RuntimeError,
                    'ZeroDivisionError': ZeroDivisionError,
                    'StopIteration': StopIteration,
                    'NameError': NameError,
                },
                'pd': pd,
                'np': np,
                'Counter': Counter,
                'lotto_data': self.lotto_df.copy(),  # ← CSV 데이터를 알고리즘에 전달
                'data_path': str(self.data_path),
                'datetime': datetime,
                'random': np.random,
                'user_numbers': user_numbers or [],
                '__name__': '__main__',
            }
            
            execution_log['data_transfer_verified'] = True
            
            # 🆕 디버깅 코드 + 알고리즘 코드 실행
            full_code = debug_code + code_content
            
            logger.info(f"⚙️ Executing algorithm: {algorithm_id} (데이터: {len(self.lotto_df)}회차)")
            
            try:
                exec(full_code, safe_globals)
                execution_log['code_execution_success'] = True
            except SyntaxError as e:
                execution_log['syntax_error'] = str(e)
                raise Exception(f"Syntax error in algorithm code: {str(e)}")
            except Exception as e:
                execution_log['runtime_error'] = str(e)
                raise Exception(f"Runtime error: {str(e)}")
            
            # 결과 함수 호출
            result = None
            for func_name in ['predict_numbers', 'predict', 'generate_numbers', 'main']:
                if func_name in safe_globals:
                    logger.info(f"✅ Calling function: {func_name}")
                    execution_log['function_called'] = func_name
                    
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
            
            if not all(isinstance(n, (int, np.integer)) and 1 <= n <= 45 for n in result):
                raise Exception("All numbers must be integers between 1 and 45")
            
            if len(set(result)) != 6:
                raise Exception("All numbers must be unique")
            
            execution_log['execution_success'] = True
            execution_log['result'] = list(map(int, result))
            
            prediction_result = {
                'status': 'success',
                'numbers': sorted(list(map(int, result))),
                'algorithm': algorithm_id,
                'algorithm_name': algorithm_info.get('name', algorithm_id),
                'accuracy_rate': algorithm_info.get('accuracy_rate', algorithm_info.get('accuracy', 0)),
                'timestamp': datetime.now().isoformat(),
                'cached': False,
                'execution_log': execution_log  # 🆕 실행 로그 포함
            }
            
            # 캐시 저장
            if not user_numbers:
                cache_file = self.cache_path / f"{self.get_algorithm_cache_key(algorithm_id)}.json"
                with open(cache_file, 'w', encoding='utf-8') as f:
                    json.dump(prediction_result, f, ensure_ascii=False, indent=2)
                logger.info(f"💾 Cached result for {algorithm_id}")
            
            # 🆕 전역 통계 업데이트
            DATA_FLOW_STATS['last_algorithm_execution'] = {
                'algorithm_id': algorithm_id,
                'timestamp': datetime.now().isoformat(),
                'success': True
            }
            
            logger.info("=" * 70)
            logger.info(f"✅ 알고리즘 실행 완료: {prediction_result['numbers']}")
            logger.info("=" * 70)
            
            return prediction_result
                
        except requests.RequestException as e:
            logger.error(f"❌ Network error downloading algorithm: {str(e)}")
            execution_log['error'] = f'network_error: {str(e)}'
            DATA_FLOW_STATS['last_algorithm_execution'] = {
                'algorithm_id': algorithm_id,
                'timestamp': datetime.now().isoformat(),
                'success': False,
                'error': str(e)
            }
            return {
                'status': 'error',
                'message': f'네트워크 오류: {str(e)}',
                'algorithm': algorithm_id,
                'execution_log': execution_log
            }
        except Exception as e:
            logger.error(f"❌ Algorithm execution failed: {str(e)}", exc_info=True)
            execution_log['error'] = str(e)
            DATA_FLOW_STATS['last_algorithm_execution'] = {
                'algorithm_id': algorithm_id,
                'timestamp': datetime.now().isoformat(),
                'success': False,
                'error': str(e)
            }
            return {
                'status': 'error',
                'message': str(e),
                'algorithm': algorithm_id,
                'execution_log': execution_log
            }
    
    def save_user_prediction(self, user_id, prediction_data):
        """사용자 예측 저장"""
        try:
            if self.user_data_path.exists():
                with open(self.user_data_path, 'r', encoding='utf-8') as f:
                    user_data = json.load(f)
            else:
                user_data = {}
            
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
            
            prediction_entry = {
                'id': str(uuid.uuid4()),
                'numbers': prediction_data['numbers'],
                'algorithm': prediction_data['algorithm'],
                'algorithm_name': prediction_data.get('algorithm_name', ''),
                'timestamp': prediction_data.get('timestamp', datetime.now().isoformat()),
                'round_predicted': prediction_data.get('round_predicted', 1194),
                'is_checked': False,
                'match_result': None,
                'cached': prediction_data.get('cached', False)
            }
            
            user_data[user_id]['predictions'].append(prediction_entry)
            user_data[user_id]['stats']['total_predictions'] += 1
            
            algo_stats = user_data[user_id]['stats']['algorithm_usage']
            algo_id = prediction_data['algorithm']
            algo_stats[algo_id] = algo_stats.get(algo_id, 0) + 1
            
            with open(self.user_data_path, 'w', encoding='utf-8') as f:
                json.dump(user_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"💾 Prediction saved for user {user_id}: {prediction_entry['id']}")
            
            return {'status': 'success', 'prediction_id': prediction_entry['id']}
            
        except Exception as e:
            logger.error(f"❌ Failed to save prediction: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    def delete_user_prediction(self, user_id, prediction_id):
        """사용자 예측 삭제"""
        try:
            if not self.user_data_path.exists():
                return {'status': 'error', 'message': 'No predictions found'}
            
            with open(self.user_data_path, 'r', encoding='utf-8') as f:
                user_data = json.load(f)
            
            if user_id not in user_data:
                return {'status': 'error', 'message': 'User not found'}
            
            predictions = user_data[user_id]['predictions']
            original_count = len(predictions)
            
            user_data[user_id]['predictions'] = [p for p in predictions if p['id'] != prediction_id]
            
            if len(user_data[user_id]['predictions']) == original_count:
                return {'status': 'error', 'message': 'Prediction not found'}
            
            user_data[user_id]['stats']['total_predictions'] = len(user_data[user_id]['predictions'])
            
            with open(self.user_data_path, 'w', encoding='utf-8') as f:
                json.dump(user_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"🗑️ Prediction {prediction_id} deleted for user {user_id}")
            
            return {'status': 'success', 'message': 'Prediction deleted'}
            
        except Exception as e:
            logger.error(f"❌ Failed to delete prediction: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    def compare_with_winning_numbers(self, user_id, prediction_id, winning_numbers):
        """당첨번호와 비교"""
        try:
            with open(self.user_data_path, 'r', encoding='utf-8') as f:
                user_data = json.load(f)
            
            if user_id not in user_data:
                return {'status': 'error', 'message': 'User not found'}
            
            prediction = None
            for pred in user_data[user_id]['predictions']:
                if pred['id'] == prediction_id:
                    prediction = pred
                    break
            
            if not prediction:
                return {'status': 'error', 'message': 'Prediction not found'}
            
            predicted_numbers = set(prediction['numbers'])
            winning_set = set(winning_numbers[:6])
            matches = len(predicted_numbers.intersection(winning_set))
            
            prize_info = {
                6: {'rank': '1등', 'description': '6개 일치'},
                5: {'rank': '2등', 'description': '5개 일치'},
                4: {'rank': '3등', 'description': '4개 일치'},
                3: {'rank': '4등', 'description': '3개 일치'},
                2: {'rank': '5등', 'description': '2개 일치'},
                1: {'rank': '6등', 'description': '1개 일치'},
                0: {'rank': '낙첨', 'description': '일치 없음'}
            }
            
            prediction['is_checked'] = True
            prediction['match_result'] = {
                'matches': matches,
                'winning_numbers': winning_numbers,
                'matched_numbers': sorted(list(predicted_numbers.intersection(winning_set))),
                'prize_info': prize_info.get(matches, prize_info[0]),
                'check_date': datetime.now().isoformat()
            }
            
            user_data[user_id]['stats']['total_matches'] += matches
            if matches > user_data[user_id]['stats']['best_match']:
                user_data[user_id]['stats']['best_match'] = matches
            
            with open(self.user_data_path, 'w', encoding='utf-8') as f:
                json.dump(user_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"🎯 Comparison completed for prediction {prediction_id}: {matches} matches")
            
            return {
                'status': 'success',
                'matches': matches,
                'result': prediction['match_result']
            }
            
        except Exception as e:
            logger.error(f"❌ Comparison failed: {str(e)}")
            return {'status': 'error', 'message': str(e)}

lotto_ai = LottoProAI()

@app.route('/')
def index():
    """메인 페이지"""
    algorithm_count = len(lotto_ai.algorithm_info.get('algorithms', {}))
    data_count = len(lotto_ai.lotto_df) if not lotto_ai.lotto_df.empty else 0
    
    return render_template('index.html', 
                         algorithm_count=algorithm_count,
                         data_count=data_count,
                         latest_round=1194,
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
        
        result = lotto_ai.execute_github_algorithm(algorithm_id, user_numbers)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"❌ Predict API error: {str(e)}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': f'Internal server error: {str(e)}'
        }), 500

@app.route('/api/save-prediction', methods=['POST'])
def save_prediction():
    """예측 저장 API"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'status': 'error', 'message': 'No data provided'}), 400
        
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({'status': 'error', 'message': 'Session not found'}), 401
        
        required_fields = ['numbers', 'algorithm']
        for field in required_fields:
            if field not in data:
                return jsonify({'status': 'error', 'message': f'Missing field: {field}'}), 400
        
        result = lotto_ai.save_user_prediction(user_id, data)
        
        if result['status'] == 'success':
            return jsonify(result), 200
        else:
            return jsonify(result), 500
            
    except Exception as e:
        logger.error(f"❌ Save prediction error: {str(e)}", exc_info=True)
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/delete-prediction', methods=['POST'])
def delete_prediction():
    """예측 삭제 API"""
    try:
        data = request.get_json()
        
        if not data or 'prediction_id' not in data:
            return jsonify({'status': 'error', 'message': 'prediction_id is required'}), 400
        
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({'status': 'error', 'message': 'Session not found'}), 401
        
        result = lotto_ai.delete_user_prediction(user_id, data['prediction_id'])
        
        if result['status'] == 'success':
            return jsonify(result), 200
        else:
            return jsonify(result), 404
            
    except Exception as e:
        logger.error(f"❌ Delete prediction error: {str(e)}", exc_info=True)
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/compare-numbers', methods=['POST'])
def compare_numbers():
    """당첨번호 비교 API"""
    try:
        data = request.get_json()
        
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({'status': 'error', 'message': 'User session not found'}), 401
        
        if not data or 'winning_numbers' not in data or 'prediction_id' not in data:
            return jsonify({'status': 'error', 'message': 'Invalid request data'}), 400
        
        winning_numbers = data.get('winning_numbers', [])
        if not isinstance(winning_numbers, list) or len(winning_numbers) < 6:
            return jsonify({'status': 'error', 'message': 'Invalid winning numbers'}), 400
        
        result = lotto_ai.compare_with_winning_numbers(
            user_id, 
            data['prediction_id'], 
            winning_numbers
        )
        
        if result['status'] == 'success':
            return jsonify(result), 200
        else:
            return jsonify(result), 404
            
    except Exception as e:
        logger.error(f"❌ Compare numbers error: {str(e)}", exc_info=True)
        return jsonify({'status': 'error', 'message': str(e)}), 500

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
        logger.error(f"❌ Failed to load user predictions: {str(e)}")
    
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
            'latest_round': 1194
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/algorithm-info')
def get_all_algorithm_info():
    """전체 알고리즘 정보 조회"""
    try:
        algorithms = lotto_ai.algorithm_info.get('algorithms', {})
        
        unique_algorithms = {}
        for key, value in algorithms.items():
            if key not in unique_algorithms:
                unique_algorithms[key] = value
        
        return jsonify({
            'status': 'success',
            'info': unique_algorithms,
            'count': len(unique_algorithms),
            'latest_round': 1194,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"❌ Failed to get algorithm info: {str(e)}")
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

# 🆕 ===== 검증용 API 엔드포인트 =====

@app.route('/api/debug/data-flow')
def debug_data_flow():
    """🆕 데이터 흐름 디버깅 API - 전체 파이프라인 검증"""
    try:
        result = {
            'timestamp': datetime.now().isoformat(),
            'version': '3.0',
            'steps': []
        }
        
        # Step 1: CSV 로드 상태
        step1 = {
            'step': 1,
            'name': 'CSV 파일 로드',
            'status': 'success' if not lotto_ai.lotto_df.empty else 'failed',
            'data_count': len(lotto_ai.lotto_df),
            'columns': list(lotto_ai.lotto_df.columns) if not lotto_ai.lotto_df.empty else [],
            'validation': lotto_ai.data_validation
        }
        
        if not lotto_ai.lotto_df.empty:
            step1['first_record'] = lotto_ai.lotto_df.iloc[0].to_dict()
            step1['latest_record'] = lotto_ai.lotto_df.iloc[-1].to_dict()
        
        result['steps'].append(step1)
        
        # Step 2: 데이터 품질
        if not lotto_ai.lotto_df.empty:
            number_cols = ['num1', 'num2', 'num3', 'num4', 'num5', 'num6']
            quality_check = {}
            
            for col in number_cols:
                if col in lotto_ai.lotto_df.columns:
                    quality_check[col] = {
                        'null_count': int(lotto_ai.lotto_df[col].isnull().sum()),
                        'valid_range': int(((lotto_ai.lotto_df[col] >= 1) & (lotto_ai.lotto_df[col] <= 45)).sum()),
                        'total': len(lotto_ai.lotto_df),
                        'quality_percentage': round(
                            ((lotto_ai.lotto_df[col] >= 1) & (lotto_ai.lotto_df[col] <= 45)).sum() / len(lotto_ai.lotto_df) * 100, 
                            2
                        )
                    }
            
            step2 = {
                'step': 2,
                'name': '데이터 품질 확인',
                'status': 'success',
                'quality_check': quality_check
            }
            result['steps'].append(step2)
        
        # Step 3: 알고리즘 로드 상태
        step3 = {
            'step': 3,
            'name': '알고리즘 로드 상태',
            'status': 'success',
            'algorithm_count': len(lotto_ai.algorithm_info.get('algorithms', {})),
            'algorithms': list(lotto_ai.algorithm_info.get('algorithms', {}).keys())
        }
        result['steps'].append(step3)
        
        # Step 4: 전역 통계
        step4 = {
            'step': 4,
            'name': '전역 데이터 흐름 통계',
            'status': 'success',
            'global_stats': DATA_FLOW_STATS
        }
        result['steps'].append(step4)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"❌ Debug API error: {str(e)}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/debug/verify-csv')
def verify_csv():
    """🆕 CSV 파일 직접 검증 API"""
    try:
        verification = {
            'timestamp': datetime.now().isoformat(),
            'csv_path_tested': [],
            'csv_found': False,
            'csv_path': None,
            'file_info': {}
        }
        
        possible_paths = [
            Path('data/new_1194.csv'),
            Path('new_1194.csv'),
            Path('/opt/render/project/src/data/new_1194.csv'),
            Path('/opt/render/project/src/new_1194.csv')
        ]
        
        for path in possible_paths:
            path_info = {
                'path': str(path),
                'exists': path.exists(),
                'is_file': path.is_file() if path.exists() else False,
                'size': path.stat().st_size if path.exists() else 0
            }
            
            verification['csv_path_tested'].append(path_info)
            
            if path.exists():
                verification['csv_found'] = True
                verification['csv_path'] = str(path)
                
                # 파일 상세 정보
                stat = path.stat()
                verification['file_info'] = {
                    'size_bytes': stat.st_size,
                    'size_kb': round(stat.st_size / 1024, 2),
                    'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    'readable': os.access(path, os.R_OK)
                }
                
                # 간단한 로드 테스트
                try:
                    test_df = pd.read_csv(path, nrows=5)
                    verification['load_test'] = {
                        'success': True,
                        'sample_rows': len(test_df),
                        'columns': list(test_df.columns),
                        'first_row': test_df.iloc[0].to_dict() if not test_df.empty else {}
                    }
                except Exception as load_error:
                    verification['load_test'] = {
                        'success': False,
                        'error': str(load_error)
                    }
                
                break
        
        return jsonify(verification)
        
    except Exception as e:
        logger.error(f"❌ CSV Verify error: {str(e)}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/health')
def health_check():
    """🔄 서비스 상태 확인 - 검증 정보 강화"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'algorithms_loaded': len(lotto_ai.algorithm_info.get('algorithms', {})),
        'data_records': len(lotto_ai.lotto_df) if not lotto_ai.lotto_df.empty else 0,
        'csv_load_success': not lotto_ai.lotto_df.empty,
        'csv_validation': lotto_ai.data_validation,
        'latest_round': 1194,
        'version': '3.0',
        'session_active': 'user_id' in session,
        'data_flow_stats': DATA_FLOW_STATS
    })

# 🆕 ===== 검증용 API 엔드포인트 끝 =====

@app.errorhandler(404)
def not_found(error):
    if request.path.startswith('/api/'):
        return jsonify({
            'status': 'error',
            'message': 'API endpoint not found'
        }), 404
    try:
        return render_template('404.html'), 404
    except:
        return jsonify({'status': 'error', 'message': 'Page not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"❌ Internal server error: {str(error)}", exc_info=True)
    if request.path.startswith('/api/'):
        return jsonify({
            'status': 'error',
            'message': 'Internal server error'
        }), 500
    try:
        return render_template('500.html'), 500
    except:
        return jsonify({'status': 'error', 'message': 'Internal server error'}), 500

@app.after_request
def after_request(response):
    """모든 응답에 CORS 헤더 추가"""
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,PUT,DELETE,OPTIONS')
    return response

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    logger.info("=" * 70)
    logger.info(f"🚀 Starting LottoPro-AI v3.0 server on port {port}")
    logger.info("=" * 70)
    logger.info(f"📝 Debug mode: {debug_mode}")
    logger.info(f"🤖 Algorithms loaded: {len(lotto_ai.algorithm_info.get('algorithms', {}))}")
    logger.info(f"📊 Lottery data records: {len(lotto_ai.lotto_df) if not lotto_ai.lotto_df.empty else 0}")
    logger.info(f"🎯 Latest round: 1194")
    logger.info(f"✅ CSV load status: {'SUCCESS' if not lotto_ai.lotto_df.empty else 'FAILED'}")
    
    if not lotto_ai.lotto_df.empty:
        logger.info(f"📂 CSV path: {lotto_ai.data_validation.get('csv_path', 'Unknown')}")
        logger.info(f"🔢 Data quality: {len([k for k, v in lotto_ai.data_validation.get('data_quality', {}).items() if v.get('quality_percentage', 0) > 99])} / 6 columns perfect")
    
    logger.info("=" * 70)
    logger.info("🔍 Debug APIs available:")
    logger.info("  - GET /api/debug/data-flow  : 전체 데이터 흐름 검증")
    logger.info("  - GET /api/debug/verify-csv : CSV 파일 직접 검증")
    logger.info("  - GET /api/health           : 서비스 상태 확인 (검증 정보 포함)")
    logger.info("=" * 70)
    
    app.run(host='0.0.0.0', port=port, debug=debug_mode)
