"""
Super ver 1.0 - Feature Engineering Focused
웹앱 연동용 표준화된 버전
"""

import pandas as pd
import numpy as np
from collections import Counter
import random
import json
from pathlib import Path
from datetime import datetime

class SuperV1Predictor:
    def __init__(self, data_file_path='data/new_1190.csv'):
        """
        Super v1.0 예측기 초기화
        Args:
            data_file_path: 로또 데이터 CSV 파일 경로
        """
        self.data_file_path = data_file_path
        self.df = None
        self.features = {}
        self.weights = {}
        
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
    
    def extract_features(self):
        """피처 추출 및 분석"""
        if self.df is None:
            return False
            
        # 컬럼명 표준화 (다양한 형태 지원)
        column_mapping = {}
        for col in self.df.columns:
            col_lower = col.lower().strip()
            if 'num1' in col_lower or '1번' in col_lower:
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
                column_mapping[col] = 'bonus'
        
        # 컬럼명 변경
        self.df = self.df.rename(columns=column_mapping)
        
        # 필수 컬럼 확인
        required_cols = ['num1', 'num2', 'num3', 'num4', 'num5', 'num6']
        if not all(col in self.df.columns for col in required_cols):
            print("❌ 필수 컬럼이 없습니다:", required_cols)
            return False
        
        # 당첨 번호와 보너스 번호 추출
        winning_numbers = self.df[required_cols].values
        bonus_numbers = self.df['bonus'].values if 'bonus' in self.df.columns else np.zeros(len(self.df))
        
        # 모든 당첨 번호를 하나의 리스트로 합치기
        all_numbers = [num for sublist in winning_numbers for num in sublist]
        
        # 피처 계산
        self.features = {}
        
        # 1. 번호 빈도 계산
        self.features['number_frequency'] = Counter(all_numbers)
        
        # 2. 연속 번호 쌍 빈도 계산
        pair_frequency = Counter()
        for nums in winning_numbers:
            nums = sorted(nums)
            for i in range(len(nums) - 1):
                pair = (nums[i], nums[i + 1])
                pair_frequency[pair] += 1
        self.features['pair_frequency'] = pair_frequency
        
        # 3. 홀짝 비율 계산
        odd_even_ratios = [sum(1 for num in nums if num % 2 == 1) / 6 for nums in winning_numbers]
        self.features['avg_odd_ratio'] = np.mean(odd_even_ratios)
        
        # 4. 고저 비율 계산
        low_high_ratios = [sum(1 for num in nums if num <= 22) / 6 for nums in winning_numbers]
        self.features['avg_low_ratio'] = np.mean(low_high_ratios)
        
        # 5. 최근 20회차 등장 번호
        recent_numbers = set()
        for nums in winning_numbers[-20:]:
            recent_numbers.update(nums)
        self.features['recent_numbers'] = recent_numbers
        
        # 6. 보너스 번호 빈도 계산
        self.features['bonus_frequency'] = Counter(bonus_numbers)
        
        # 7. 번호별 등장 간격 계산
        last_appearance = {}
        for i, nums in enumerate(winning_numbers):
            for num in nums:
                last_appearance[num] = i
        current_round = len(winning_numbers)
        self.features['appearance_gap'] = {
            num: current_round - last_appearance.get(num, current_round) 
            for num in range(1, 46)
        }
        
        # 8. 번호 그룹화
        group_frequency = Counter()
        for nums in winning_numbers:
            for num in nums:
                if 1 <= num <= 15:
                    group_frequency['1-15'] += 1
                elif 16 <= num <= 30:
                    group_frequency['16-30'] += 1
                elif 31 <= num <= 45:
                    group_frequency['31-45'] += 1
        self.features['group_frequency'] = group_frequency
        
        # 9. 번호 합계 분석
        sums = [sum(nums) for nums in winning_numbers]
        sum_histogram = Counter(sums)
        sum_freq = sorted(sum_histogram.items(), key=lambda x: x[1], reverse=True)
        top_sum_range = sum_freq[:int(0.5 * len(sum_freq))]
        self.features['sum_range'] = (
            min([s for s, _ in top_sum_range]),
            max([s for s, _ in top_sum_range])
        )
        
        print(f"✅ 피처 추출 완료: {len(self.features)}개 피처")
        return True
    
    def calculate_weights(self):
        """번호별 가중치 계산"""
        if not self.features:
            return False
            
        self.weights = {}
        
        for num in range(1, 46):
            weight = 0
            
            # 기본 빈도 가중치
            weight += self.features['number_frequency'].get(num, 0) * 1.5
            
            # 연속 번호 쌍 가중치
            for pair in self.features['pair_frequency']:
                if num in pair:
                    weight += self.features['pair_frequency'][pair] * 0.6
            
            # 최근 출현 보너스
            if num in self.features['recent_numbers']:
                weight += 10
            
            # 보너스 번호 가중치
            weight += self.features['bonus_frequency'].get(num, 0) * 1.0
            
            # 등장 간격 가중치
            weight += self.features['appearance_gap'][num] * 0.3
            
            # 그룹별 가중치
            if 1 <= num <= 15:
                weight += self.features['group_frequency']['1-15'] * 0.15
            elif 16 <= num <= 30:
                weight += self.features['group_frequency']['16-30'] * 0.15
            elif 31 <= num <= 45:
                weight += self.features['group_frequency']['31-45'] * 0.15
            
            # 홀짝 패턴 가중치
            is_odd = num % 2 == 1
            avg_odd = self.features['avg_odd_ratio']
            if (is_odd and avg_odd > 0.5) or (not is_odd and avg_odd <= 0.5):
                weight += 7
            
            # 고저 패턴 가중치
            is_low = num <= 22
            avg_low = self.features['avg_low_ratio']
            if (is_low and avg_low > 0.5) or (not is_low and avg_low <= 0.5):
                weight += 7
            
            # 최소 가중치 보장
            self.weights[num] = max(weight, 1)
        
        print("✅ 가중치 계산 완료")
        return True
    
    def predict_numbers(self, count=1, user_numbers=None):
        """
        번호 예측
        Args:
            count: 생성할 번호 세트 수
            user_numbers: 사용자가 선호하는 번호들 (선택사항)
        Returns:
            예측된 번호 세트들의 리스트
        """
        if not self.weights:
            return []
        
        # 가중치를 확률로 변환
        total_weight = sum(self.weights.values())
        prob = [self.weights[i] / total_weight for i in range(1, 46)]
        numbers = list(range(1, 46))
        
        predicted_sets = []
        max_attempts = 1000
        attempts = 0
        
        while len(predicted_sets) < count and attempts < max_attempts:
            selected = []
            
            # 사용자 선호 번호 먼저 추가
            if user_numbers:
                valid_user_numbers = [n for n in user_numbers if 1 <= n <= 45]
                selected.extend(valid_user_numbers[:3])  # 최대 3개까지
            
            # 나머지 번호 선택
            while len(selected) < 6:
                candidate = random.choices(numbers, weights=prob, k=1)[0]
                if candidate not in selected:
                    selected.append(candidate)
            
            selected.sort()
            selected_sum = sum(selected)
            
            # 합계 범위 확인
            min_sum, max_sum = self.features['sum_range']
            if min_sum <= selected_sum <= max_sum:
                predicted_sets.append(selected)
            
            attempts += 1
        
        return predicted_sets
    
    def get_algorithm_info(self):
        """알고리즘 정보 반환"""
        return {
            'name': 'Super v1.0',
            'description': 'Feature Engineering Focused - 고급 특성 공학 기반 예측',
            'version': '1.0.0',
            'features': [
                '번호 출현 빈도 분석',
                '연속 번호 패턴 분석', 
                '홀짝 비율 최적화',
                '고저 구간 밸런싱',
                '최근 출현 패턴 반영',
                '보너스 번호 상관관계',
                '번호 그룹별 가중치',
                '합계 범위 최적화'
            ],
            'accuracy_focus': '패턴 기반 통계 분석',
            'recommendation': '안정적이고 균형잡힌 예측을 원하는 사용자'
        }

def run_super_v1(data_file_path='data/new_1190.csv', user_numbers=None):
    """
    Super v1.0 실행 함수 (웹앱 연동용)
    Args:
        data_file_path: 데이터 파일 경로
        user_numbers: 사용자 선호 번호 (선택사항)
    Returns:
        결과 딕셔너리
    """
    predictor = SuperV1Predictor(data_file_path)
    
    # 단계별 실행
    if not predictor.load_data():
        return {
            'success': False,
            'error': '데이터 로드 실패',
            'numbers': [],
            'algorithm_info': predictor.get_algorithm_info()
        }
    
    if not predictor.extract_features():
        return {
            'success': False,
            'error': '피처 추출 실패',
            'numbers': [],
            'algorithm_info': predictor.get_algorithm_info()
        }
    
    if not predictor.calculate_weights():
        return {
            'success': False,
            'error': '가중치 계산 실패',
            'numbers': [],
            'algorithm_info': predictor.get_algorithm_info()
        }
    
    # 번호 예측
    predicted_sets = predictor.predict_numbers(count=5, user_numbers=user_numbers)
    
    if not predicted_sets:
        return {
            'success': False,
            'error': '조건을 만족하는 번호 생성 실패',
            'numbers': [],
            'algorithm_info': predictor.get_algorithm_info()
        }
    
    # 성공 결과 반환
    return {
        'success': True,
        'numbers': predicted_sets[0],  # 첫 번째 세트 반환
        'alternative_sets': predicted_sets[1:],  # 나머지 대안들
        'algorithm_info': predictor.get_algorithm_info(),
        'features_count': len(predictor.features),
        'data_rounds': len(predictor.df),
        'timestamp': datetime.now().isoformat()
    }

# 직접 실행 시 테스트
if __name__ == "__main__":
    result = run_super_v1()
    
    if result['success']:
        print("🎯 Super v1.0 예측 결과:")
        print(f"추천 번호: {', '.join(map(str, result['numbers']))}")
        print(f"데이터 회차: {result['data_rounds']}")
        print(f"사용된 피처: {result['features_count']}개")
    else:
        print(f"❌ 오류: {result['error']}")