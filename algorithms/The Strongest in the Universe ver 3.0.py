"""
The Strongest in the Universe ver 3.0 - Simplified
웹앱 연동용 표준화된 버전 (핵심 로직만 유지)
"""

import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import random
import json
from pathlib import Path
from datetime import datetime
import math

class StrongestUniverseV3Predictor:
    def __init__(self, data_file_path='data/new_1190.csv'):
        """
        Strongest Universe v3.0 예측기 초기화
        Args:
            data_file_path: 로또 데이터 CSV 파일 경로
        """
        self.data_file_path = data_file_path
        self.df = None
        self.advanced_features = {}
        self.cosmic_weights = {}
        
    def load_data(self):
        """로또 데이터 로드"""
        try:
            # 다양한 경로에서 데이터 파일 찾기
            possible_paths = [
                self.data_file_path,
                'new_1190.csv',
                '../data/new_1190.csv',
                'data/new_1190.csv'
            ]
            
            for path in possible_paths:
                try:
                    self.df = pd.read_csv(path, encoding='utf-8-sig')
                    print(f"✅ 데이터 로드 성공: {path}")
                    return True
                except:
                    continue
                    
            print("❌ 데이터 파일을 찾을 수 없습니다.")
            return False
            
        except Exception as e:
            print(f"❌ 데이터 로드 실패: {e}")
            return False
    
    def standardize_columns(self):
        """컬럼명 표준화"""
        if self.df is None:
            return False
            
        # 컬럼명 표준화 (다양한 형태 지원)
        column_mapping = {}
        for col in self.df.columns:
            col_lower = col.lower().strip()
            if 'num1' in col_lower or '1번' in col_lower or 'draw_date' in col_lower:
                if 'date' in col_lower:
                    column_mapping[col] = 'draw_date'
                else:
                    column_mapping[col] = 'num1'
            elif 'num2' in col_lower or '2번' in col_lower:
                column_mapping[col] = 'num2'
            elif 'num3' in col_lower or '3번' in col_lower:
                column_mapping[col] = 'num3'
            elif 'num4' in col_lower or '4번' in col_lower:
                column_mapping[col] = 'num4'
            elif 'num5' in col_lower or '5번' in col_lower:
                column_mapping[col] = 'num5'
            elif 'num6' in col_lower or '6번' in col_lower:
                column_mapping[col] = 'num6'
            elif 'bonus' in col_lower or '보너스' in col_lower:
                column_mapping[col] = 'bonus_num'
            elif 'round' in col_lower or '회차' in col_lower:
                column_mapping[col] = 'round'
        
        # 컬럼명 변경
        self.df = self.df.rename(columns=column_mapping)
        
        # 필수 컬럼 확인
        required_cols = ['num1', 'num2', 'num3', 'num4', 'num5', 'num6']
        if not all(col in self.df.columns for col in required_cols):
            print("❌ 필수 컬럼이 없습니다:", required_cols)
            return False
            
        return True
    
    def extract_advanced_features(self):
        """고급 특성 추출 (간소화 버전)"""
        if self.df is None:
            return False
            
        print("🔬 고급 특성 추출 중...")
        
        self.advanced_features = {}
        
        # 1. 피보나치 수열 분석
        fibonacci_numbers = [1, 1, 2, 3, 5, 8, 13, 21, 34]
        fib_appearances = {}
        
        for num in fibonacci_numbers:
            if num <= 45:
                count = 0
                for col in ['num1', 'num2', 'num3', 'num4', 'num5', 'num6']:
                    if col in self.df.columns:
                        count += (self.df[col] == num).sum()
                fib_appearances[num] = count
        
        self.advanced_features['fibonacci'] = fib_appearances
        
        # 2. 소수 분석
        def is_prime(n):
            if n <= 1: return False
            if n <= 3: return True
            if n % 2 == 0 or n % 3 == 0: return False
            i = 5
            while i * i <= n:
                if n % i == 0 or n % (i + 2) == 0: return False
                i += 6
            return True
        
        primes = [num for num in range(2, 46) if is_prime(num)]
        prime_appearances = {}
        
        for prime in primes:
            count = 0
            for col in ['num1', 'num2', 'num3', 'num4', 'num5', 'num6']:
                if col in self.df.columns:
                    count += (self.df[col] == prime).sum()
            prime_appearances[prime] = count
            
        self.advanced_features['primes'] = prime_appearances
        
        # 3. 황금비 기반 수열 분석
        golden_ratio = 1.618
        golden_numbers = []
        for i in range(1, 28):
            golden_num = int(i * golden_ratio)
            if golden_num <= 45 and golden_num not in golden_numbers:
                golden_numbers.append(golden_num)
        
        self.advanced_features['golden_numbers'] = golden_numbers
        
        # 4. 주기성 분석
        periodicity = self.analyze_periodicity()
        self.advanced_features['periodicity'] = periodicity
        
        # 5. 연관 패턴 분석
        association_patterns = self.analyze_association_patterns()
        self.advanced_features['associations'] = association_patterns
        
        print(f"✅ 고급 특성 추출 완료: {len(self.advanced_features)}개 특성군")
        return True
    
    def analyze_periodicity(self):
        """주기성 분석 (간소화)"""
        periodicity_scores = {}
        
        for num in range(1, 46):
            appearances = []
            
            # 각 번호가 나타나는 회차 찾기
            for idx, row in self.df.iterrows():
                number_cols = ['num1', 'num2', 'num3', 'num4', 'num5', 'num6']
                row_numbers = [row[col] for col in number_cols if col in row]
                if num in row_numbers:
                    appearances.append(idx)
            
            if len(appearances) >= 2:
                # 출현 간격 계산
                intervals = [appearances[i+1] - appearances[i] for i in range(len(appearances)-1)]
                
                if intervals:
                    periodicity_scores[num] = {
                        'avg_interval': np.mean(intervals),
                        'last_appearance': appearances[-1],
                        'predicted_next': appearances[-1] + np.mean(intervals)
                    }
        
        return periodicity_scores
    
    def analyze_association_patterns(self):
        """연관 패턴 분석 (간소화)"""
        co_occurrence = defaultdict(int)
        number_counts = defaultdict(int)
        
        # 동시 출현 빈도 계산
        for _, row in self.df.iterrows():
            numbers = [row[f'num{i}'] for i in range(1, 7) if f'num{i}' in row]
            
            # 각 번호의 총 출현 횟수
            for num in numbers:
                number_counts[num] += 1
            
            # 번호 쌍의 동시 출현
            for i in range(len(numbers)):
                for j in range(i+1, len(numbers)):
                    pair = tuple(sorted([numbers[i], numbers[j]]))
                    co_occurrence[pair] += 1
        
        # 연관성 점수 계산
        association_scores = {}
        total_draws = len(self.df)
        
        for i in range(1, 46):
            association_scores[i] = {}
            for j in range(1, 46):
                if i == j:
                    continue
                    
                pair = tuple(sorted([i, j]))
                co_freq = co_occurrence.get(pair, 0)
                
                if co_freq > 0:
                    # 단순 연관성 점수
                    p_i = number_counts.get(i, 0) / total_draws
                    p_j = number_counts.get(j, 0) / total_draws
                    p_ij = co_freq / total_draws
                    
                    if p_i > 0 and p_j > 0:
                        # PMI (Pointwise Mutual Information) 기반 점수
                        pmi = math.log(p_ij / (p_i * p_j)) if p_i * p_j > 0 else 0
                        association_scores[i][j] = max(0, pmi)  # 음수는 0으로
        
        return association_scores
    
    def calculate_cosmic_weights(self):
        """우주적 가중치 계산 (간소화)"""
        if not self.advanced_features:
            return False
            
        print("🌌 우주적 가중치 계산 중...")
        
        self.cosmic_weights = {}
        
        for num in range(1, 46):
            weight = 1.0
            
            # 1. 기본 출현 빈도
            total_appearances = 0
            for col in ['num1', 'num2', 'num3', 'num4', 'num5', 'num6']:
                if col in self.df.columns:
                    total_appearances += (self.df[col] == num).sum()
            
            freq_weight = total_appearances / len(self.df) if len(self.df) > 0 else 0
            weight += freq_weight * 2.0
            
            # 2. 피보나치 보너스
            if num in self.advanced_features['fibonacci']:
                fibonacci_bonus = self.advanced_features['fibonacci'][num] / len(self.df)
                weight += fibonacci_bonus * 1.3
            
            # 3. 소수 보너스
            if num in self.advanced_features['primes']:
                prime_bonus = self.advanced_features['primes'][num] / len(self.df)
                weight += prime_bonus * 1.2
            
            # 4. 황금비 보너스
            if num in self.advanced_features['golden_numbers']:
                weight *= 1.15
            
            # 5. 주기성 점수
            if num in self.advanced_features['periodicity']:
                period_info = self.advanced_features['periodicity'][num]
                current_round = len(self.df)
                predicted_round = period_info['predicted_next']
                
                # 예측 출현 회차와 현재 회차의 거리
                distance = abs(current_round - predicted_round)
                if distance <= 3:  # 3회차 이내
                    proximity_bonus = (4 - distance) * 0.1
                    weight += proximity_bonus
            
            # 6. 최근 출현 패턴 (간소화)
            recent_appearances = 0
            if len(self.df) >= 10:
                recent_df = self.df.tail(10)
                for col in ['num1', 'num2', 'num3', 'num4', 'num5', 'num6']:
                    if col in recent_df.columns:
                        recent_appearances += (recent_df[col] == num).sum()
            
            if recent_appearances == 0:  # 최근 미출현 보너스
                weight *= 1.2
            elif recent_appearances >= 2:  # 최근 과도출현 페널티
                weight *= 0.85
            
            # 7. 숫자학적 특성 (간소화)
            digit_sum = sum(int(d) for d in str(num))
            if digit_sum in [7, 11, 13]:  # 행운의 숫자
                weight *= 1.05
            
            self.cosmic_weights[num] = max(weight, 0.1)  # 최소값 보장
        
        # 가중치 정규화
        total_weight = sum(self.cosmic_weights.values())
        for num in self.cosmic_weights:
            self.cosmic_weights[num] /= total_weight
        
        print("✅ 우주적 가중치 계산 완료")
        return True
    
    def quantum_selection(self, count=1, user_numbers=None):
        """
        양자역학적 선택 알고리즘 (간소화)
        Args:
            count: 생성할 번호 세트 수
            user_numbers: 사용자 선호 번호
        Returns:
            예측된 번호 세트들
        """
        if not self.cosmic_weights:
            return []
        
        print("⚛️ 양자역학적 번호 선택 중...")
        
        predicted_sets = []
        
        for _ in range(count):
            selected_numbers = []
            
            # 사용자 선호 번호 먼저 추가
            if user_numbers:
                valid_user_numbers = [n for n in user_numbers if 1 <= n <= 45]
                selected_numbers.extend(valid_user_numbers[:2])  # 최대 2개까지
            
            # 나머지 번호 선택
            while len(selected_numbers) < 6:
                # 가중치 기반 확률적 선택
                available_numbers = [n for n in range(1, 46) if n not in selected_numbers]
                weights = [self.cosmic_weights.get(n, 0.001) for n in available_numbers]
                
                # 연관성 보정
                if len(selected_numbers) > 0:
                    associations = self.advanced_features.get('associations', {})
                    for i, num in enumerate(available_numbers):
                        association_bonus = 0
                        for selected in selected_numbers:
                            if selected in associations and num in associations[selected]:
                                association_bonus += associations[selected][num]
                        weights[i] += association_bonus * 0.1
                
                # 정규화
                total_weight = sum(weights)
                if total_weight > 0:
                    weights = [w / total_weight for w in weights]
                else:
                    weights = [1 / len(available_numbers)] * len(available_numbers)
                
                # 선택
                selected_num = np.random.choice(available_numbers, p=weights)
                selected_numbers.append(selected_num)
            
            predicted_sets.append(sorted(selected_numbers))
        
        return predicted_sets
    
    def get_algorithm_info(self):
        """알고리즘 정보 반환"""
        return {
            'name': 'The Strongest in the Universe v3.0',
            'description': '우주 최강 AI 예측 시스템 - 고급 수학적 패턴과 양자역학적 선택',
            'version': '3.0.0',
            'features': [
                '피보나치 수열 패턴 분석',
                '소수 분포 최적화',
                '황금비 기반 수열 활용',
                '주기성 예측 모델링',
                '연관성 패턴 매트릭스',
                '우주적 가중치 시스템',
                '양자역학적 선택 알고리즘',
                '다차원 특성 융합'
            ],
            'accuracy_focus': '고급 수학적 모델과 우주적 패턴 인식',
            'recommendation': '최첨단 AI 기술을 원하는 고급 사용자',
            'complexity': 'high',
            'execution_time': 'medium'
        }

def run_strongest_universe_v3(data_file_path='data/new_1190.csv', user_numbers=None):
    """
    Strongest Universe v3.0 실행 함수 (웹앱 연동용)
    Args:
        data_file_path: 데이터 파일 경로
        user_numbers: 사용자 선호 번호 (선택사항)
    Returns:
        결과 딕셔너리
    """
    predictor = StrongestUniverseV3Predictor(data_file_path)
    
    # 단계별 실행
    if not predictor.load_data():
        return {
            'success': False,
            'error': '데이터 로드 실패',
            'numbers': [],
            'algorithm_info': predictor.get_algorithm_info()
        }
    
    if not predictor.standardize_columns():
        return {
            'success': False,
            'error': '데이터 컬럼 표준화 실패',
            'numbers': [],
            'algorithm_info': predictor.get_algorithm_info()
        }
    
    if not predictor.extract_advanced_features():
        return {
            'success': False,
            'error': '고급 특성 추출 실패',
            'numbers': [],
            'algorithm_info': predictor.get_algorithm_info()
        }
    
    if not predictor.calculate_cosmic_weights():
        return {
            'success': False,
            'error': '우주적 가중치 계산 실패',
            'numbers': [],
            'algorithm_info': predictor.get_algorithm_info()
        }
    
    # 양자역학적 번호 선택
    predicted_sets = predictor.quantum_selection(count=3, user_numbers=user_numbers)
    
    if not predicted_sets:
        return {
            'success': False,
            'error': '양자역학적 선택 실패',
            'numbers': [],
            'algorithm_info': predictor.get_algorithm_info()
        }
    
    # 성공 결과 반환
    return {
        'success': True,
        'numbers': predicted_sets[0],  # 첫 번째 세트 반환
        'alternative_sets': predicted_sets[1:],  # 나머지 대안들
        'algorithm_info': predictor.get_algorithm_info(),
        'advanced_features': len(predictor.advanced_features),
        'data_rounds': len(predictor.df),
        'cosmic_energy': sum(predictor.cosmic_weights.values()),
        'timestamp': datetime.now().isoformat()
    }

# 직접 실행 시 테스트
if __name__ == "__main__":
    result = run_strongest_universe_v3()
    
    if result['success']:
        print("🌟 Strongest Universe v3.0 예측 결과:")
        print(f"우주 선택 번호: {', '.join(map(str, result['numbers']))}")
        print(f"데이터 회차: {result['data_rounds']}")
        print(f"고급 특성군: {result['advanced_features']}개")
        print(f"우주적 에너지: {result['cosmic_energy']:.4f}")
    else:
        print(f"❌ 오류: {result['error']}")