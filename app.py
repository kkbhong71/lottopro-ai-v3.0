from flask import Flask, render_template, request, jsonify, session
from datetime import datetime, timedelta
import json
import os
import pandas as pd
import numpy as np
import requests
from pathlib import Path
import subprocess
import tempfile
import logging
from werkzeug.security import generate_password_hash, check_password_hash
import uuid

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'lottopro-ai-v3-secret-key')
app.config['JSON_AS_ASCII'] = False

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# GitHub 설정
GITHUB_REPO = os.environ.get('GITHUB_REPO', 'your-username/lottopro-algorithms')
GITHUB_TOKEN = os.environ.get('GITHUB_TOKEN', '')
GITHUB_API_BASE = f'https://api.github.com/repos/{GITHUB_REPO}'

class LottoProAI:
    def __init__(self):
        self.data_path = Path('data')
        self.data_path.mkdir(exist_ok=True)
        self.user_data_path = self.data_path / 'user_predictions.json'
        self.algorithm_info_path = Path('algorithms/algorithm_info.json')
        self.load_algorithm_info()
        self.load_lotto_data()
        
    def load_algorithm_info(self):
        """알고리즘 정보 로드"""
        try:
            with open(self.algorithm_info_path, 'r', encoding='utf-8') as f:
                self.algorithm_info = json.load(f)
        except FileNotFoundError:
            logger.warning("Algorithm info file not found, using default")
            self.algorithm_info = {"algorithms": {}}
            
    def load_lotto_data(self):
        """로또 당첨번호 데이터 로드"""
        try:
            self.lotto_df = pd.read_csv(self.data_path / 'new_1190.csv')
            logger.info(f"Loaded {len(self.lotto_df)} lottery records")
        except FileNotFoundError:
            logger.error("Lottery data file not found")
            self.lotto_df = pd.DataFrame()
    
    def execute_github_algorithm(self, algorithm_id):
        """GitHub에서 알고리즘 코드를 실행"""
        try:
            algorithm_path = self.algorithm_info['algorithms'][algorithm_id]['github_path']
            
            # GitHub에서 코드 다운로드
            response = requests.get(
                f'https://raw.githubusercontent.com/{GITHUB_REPO}/main/{algorithm_path}',
                headers={'Authorization': f'token {GITHUB_TOKEN}' if GITHUB_TOKEN else None}
            )
            
            if response.status_code != 200:
                raise Exception(f"Failed to download algorithm: {response.status_code}")
            
            code_content = response.text
            
            # 임시 파일에 코드 저장 및 실행
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code_content)
                temp_file = f.name
            
            try:
                # 안전한 코드 실행 환경 구성
                exec_globals = {
                    'pd': pd,
                    'np': np,
                    'lotto_data': self.lotto_df,
                    'data_path': str(self.data_path)
                }
                
                exec(code_content, exec_globals)
                
                # 예측 결과 반환 (표준 함수명: predict_numbers)
                if 'predict_numbers' in exec_globals:
                    result = exec_globals['predict_numbers']()
                    return {
                        'status': 'success',
                        'numbers': result,
                        'algorithm': algorithm_id,
                        'timestamp': datetime.now().isoformat()
                    }
                else:
                    raise Exception("predict_numbers function not found in algorithm")
                    
            finally:
                os.unlink(temp_file)
                
        except Exception as e:
            logger.error(f"Algorithm execution failed: {str(e)}")
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
                    'predictions': [],
                    'stats': {
                        'total_predictions': 0,
                        'total_matches': 0,
                        'best_match': 0
                    }
                }
            
            # 예측 데이터 추가
            prediction_entry = {
                'id': str(uuid.uuid4()),
                'numbers': prediction_data['numbers'],
                'algorithm': prediction_data['algorithm'],
                'timestamp': prediction_data.get('timestamp', datetime.now().isoformat()),
                'round_predicted': prediction_data.get('round_predicted'),
                'is_checked': False,
                'match_result': None
            }
            
            user_data[user_id]['predictions'].append(prediction_entry)
            user_data[user_id]['stats']['total_predictions'] += 1
            
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
            winning_set = set(winning_numbers)
            matches = len(predicted_numbers.intersection(winning_set))
            
            # 결과 저장
            prediction['is_checked'] = True
            prediction['match_result'] = {
                'matches': matches,
                'winning_numbers': winning_numbers,
                'matched_numbers': list(predicted_numbers.intersection(winning_set))
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

@app.route('/')
def index():
    """메인 페이지"""
    return render_template('index.html', 
                         algorithm_count=len(lotto_ai.algorithm_info.get('algorithms', {})))

@app.route('/algorithms')
def algorithms():
    """알고리즘 선택 페이지"""
    algorithms_data = lotto_ai.algorithm_info.get('algorithms', {})
    return render_template('algorithm.html', algorithms=algorithms_data)

@app.route('/api/execute/<algorithm_id>')
def execute_algorithm(algorithm_id):
    """알고리즘 실행 API"""
    if algorithm_id not in lotto_ai.algorithm_info.get('algorithms', {}):
        return jsonify({'status': 'error', 'message': 'Algorithm not found'}), 404
    
    result = lotto_ai.execute_github_algorithm(algorithm_id)
    return jsonify(result)

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
    
    result = lotto_ai.compare_with_winning_numbers(
        user_id, 
        data['prediction_id'], 
        data['winning_numbers']
    )
    
    return jsonify(result)

@app.route('/api/user-predictions')
def get_user_predictions():
    """사용자 예측 목록 조회"""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'predictions': [], 'stats': {}})
    
    try:
        if lotto_ai.user_data_path.exists():
            with open(lotto_ai.user_data_path, 'r', encoding='utf-8') as f:
                user_data = json.load(f)
                user_info = user_data.get(user_id, {
                    'predictions': [], 
                    'stats': {'total_predictions': 0, 'total_matches': 0, 'best_match': 0}
                })
                return jsonify(user_info)
    except Exception as e:
        logger.error(f"Failed to load user predictions: {str(e)}")
    
    return jsonify({'predictions': [], 'stats': {}})

@app.route('/saved-numbers')
def saved_numbers():
    """저장된 번호 관리 페이지"""
    return render_template('saved_numbers.html')

@app.route('/compare')
def compare():
    """당첨번호 비교 페이지"""
    return render_template('compare.html')

@app.route('/api/lottery-data')
def get_lottery_data():
    """로또 데이터 API"""
    try:
        return jsonify({
            'status': 'success',
            'data': lotto_ai.lotto_df.to_dict('records'),
            'total_records': len(lotto_ai.lotto_df)
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

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

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
