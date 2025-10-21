"""
Super ver 1.0 - Web App Standardized Version
Feature Engineering Focused - 웹앱 표준화 버전

웹앱 표준 템플릿 적용:
- predict_numbers() 진입점 함수
- 글로벌 변수 사용 (lotto_data, pd, np)
- 웹앱 안전 실행 환경 준수
- 고급 특성 공학 기반 예측
- JSON 직렬화 안전성 보장
"""

import pandas as pd
import numpy as np
from collections import Counter
import random

# 안전한 warnings 처리
try:
    import warnings
    warnings.filterwarnings('ignore')
except ImportError:
    # warnings 모듈을 사용할 수 없는 환경
    pass

def convert_to_python_int(value):
    """numpy 타입을 Python int로 안전하게 변환"""
    try:
        if isinstance(value, (np.integer, np.floating)):
            return int(value)
        elif isinstance(value, (int, float)):
            return int(value)
        else:
            return int(float(value))
    except (ValueError, TypeError, OverflowError):
        return random.randint(1, 45)

def predict_numbers():
    """
    웹앱 표준 예측 함수 - Super v1.0 시스템
    
    글로벌 변수 사용:
    - lotto_data: pandas DataFrame (로또 당첨번호 데이터)
    - pd: pandas 라이브러리  
    - np: numpy 라이브러리
    - data_path: 데이터 폴더 경로 (문자열)
    
    Returns:
        list: 정확히 6개의 로또 번호 [1-45 범위의 Python int]
    """
    try:
        # 1. 데이터 검증
        if 'lotto_data' not in globals() or lotto_data.empty:
            print("⚠️ [FALLBACK] lotto_data 없음 - 안전 모드")
            return generate_safe_fallback()
        
        df = lotto_data.copy()
        print(f"✅ [VERIFY] 데이터 로드 성공: {len(df)}회차")
        
        # 2. 데이터 전처리
        df = preprocess_data(df)
        
        # 3. Super v1.0 알고리즘 실행
        result = run_super_v1_algorithm(df)
        
        # 4. 결과 검증 및 반환
        final_result = validate_result(result)
        print(f"🔧 [SUPER] 최종 결과: {final_result}")
        
        return final_result
        
    except Exception as e:
        print(f"❌ [ERROR] Super v1.0: {str(e)[:100]}")
        return generate_safe_fallback()

def preprocess_data(df):
    """데이터 전처리 - Super v1.0용"""
    try:
        # 컬럼명 정규화
        df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]
        
        # 표준 컬럼 매핑
        if len(df.columns) >= 9:
            standard_cols = ['round', 'draw_date', 'num1', 'num2', 'num3', 'num4', 'num5', 'num6', 'bonus_num']
            mapping = dict(zip(df.columns[:9], standard_cols))
            df = df.rename(columns=mapping)
        
        # 숫자 컬럼 변환 및 타입 안전성 보장
        number_cols = ['num1', 'num2', 'num3', 'num4', 'num5', 'num6']
        for col in number_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                # ✅ numpy 타입을 Python int로 변환
                df[col] = df[col].apply(lambda x: convert_to_python_int(x) if pd.notna(x) else random.randint(1, 45))
        
        # 유효성 필터링
        df = df.dropna(subset=number_cols)
        for col in number_cols:
            if col in df.columns:
                df = df[(df[col] >= 1) & (df[col] <= 45)]
        
        return df.sort_values('round' if 'round' in df.columns else df.columns[0]).reset_index(drop=True)
        
    except Exception as e:
        print(f"⚠️ [PREPROCESS] 오류: {str(e)[:50]}")
        return df

def run_super_v1_algorithm(df):
    """Super v1.0 핵심 알고리즘"""
    try:
        if len(df) < 5:
            print("⚠️ [DATA] 데이터 부족 - 스마트 랜덤 모드")
            return generate_smart_random()
        
        number_cols = ['num1', 'num2', 'num3', 'num4', 'num5', 'num6']
        
        # 피처 추출
        features = extract_features(df, number_cols)
        
        # 가중치 계산
        weights = calculate_weights(df, features)
        
        # 번호 예측
        final_prediction = predict_with_features(weights, features)
        
        # ✅ 모든 요소를 Python int로 확실히 변환
        safe_prediction = [convert_to_python_int(num) for num in final_prediction]
        
        return safe_prediction
        
    except Exception as e:
        print(f"❌ [ALGORITHM] Super v1.0 오류: {str(e)[:50]}")
        return generate_smart_random()

def extract_features(df, number_cols):
    """피처 추출 및 분석"""
    try:
        features = {}
        
        # 당첨 번호와 보너스 번호 추출
        winning_numbers = df[number_cols].values
        bonus_numbers = df['bonus_num'].values if 'bonus_num' in df.columns else np.zeros(len(df))
        
        # 모든 당첨 번호를 하나의 리스트로 합치기 - 타입 안전성 보장
        all_numbers = []
        for sublist in winning_numbers:
            for num in sublist:
                all_numbers.append(convert_to_python_int(num))
        
        # 1. 번호 빈도 계산 - Python int 보장
        features['number_frequency'] = Counter(all_numbers)
        
        # 2. 연속 번호 쌍 빈도 계산
        pair_frequency = Counter()
        for nums in winning_numbers:
            nums_converted = [convert_to_python_int(num) for num in nums]
            nums_sorted = sorted(nums_converted)
            for i in range(len(nums_sorted) - 1):
                pair = (nums_sorted[i], nums_sorted[i + 1])
                pair_frequency[pair] += 1
        features['pair_frequency'] = pair_frequency
        
        # 3. 홀짝 비율 계산
        odd_even_ratios = []
        for nums in winning_numbers:
            nums_converted = [convert_to_python_int(num) for num in nums]
            odd_count = sum(1 for num in nums_converted if num % 2 == 1)
            odd_even_ratios.append(float(odd_count / 6))  # ✅ Python float 변환
        features['avg_odd_ratio'] = float(np.mean(odd_even_ratios))  # ✅ Python float 변환
        
        # 4. 고저 비율 계산
        low_high_ratios = []
        for nums in winning_numbers:
            nums_converted = [convert_to_python_int(num) for num in nums]
            low_count = sum(1 for num in nums_converted if num <= 22)
            low_high_ratios.append(float(low_count / 6))  # ✅ Python float 변환
        features['avg_low_ratio'] = float(np.mean(low_high_ratios))  # ✅ Python float 변환
        
        # 5. 최근 20회차 등장 번호
        recent_numbers = set()
        recent_data = winning_numbers[-20:] if len(winning_numbers) >= 20 else winning_numbers
        for nums in recent_data:
            nums_converted = [convert_to_python_int(num) for num in nums]
            recent_numbers.update(nums_converted)
        features['recent_numbers'] = recent_numbers
        
        # 6. 보너스 번호 빈도 계산
        bonus_converted = [convert_to_python_int(num) for num in bonus_numbers]
        features['bonus_frequency'] = Counter(bonus_converted)
        
        # 7. 번호별 등장 간격 계산
        last_appearance = {}
        for i, nums in enumerate(winning_numbers):
            nums_converted = [convert_to_python_int(num) for num in nums]
            for num in nums_converted:
                last_appearance[num] = i
        current_round = len(winning_numbers)
        features['appearance_gap'] = {}
        for num in range(1, 46):
            gap = current_round - last_appearance.get(num, current_round)
            features['appearance_gap'][num] = int(gap)  # ✅ Python int 변환
        
        # 8. 번호 그룹화
        group_frequency = Counter()
        for nums in winning_numbers:
            nums_converted = [convert_to_python_int(num) for num in nums]
            for num in nums_converted:
                if 1 <= num <= 15:
                    group_frequency['1-15'] += 1
                elif 16 <= num <= 30:
                    group_frequency['16-30'] += 1
                elif 31 <= num <= 45:
                    group_frequency['31-45'] += 1
        features['group_frequency'] = group_frequency
        
        # 9. 번호 합계 분석
        sums = []
        for nums in winning_numbers:
            nums_converted = [convert_to_python_int(num) for num in nums]
            sums.append(sum(nums_converted))
        
        sum_histogram = Counter(sums)
        sum_freq = sorted(sum_histogram.items(), key=lambda x: x[1], reverse=True)
        top_sum_range = sum_freq[:int(0.5 * len(sum_freq))] if sum_freq else [(120, 1), (180, 1)]
        
        if top_sum_range:
            min_sum = min([s for s, _ in top_sum_range])
            max_sum = max([s for s, _ in top_sum_range])
            features['sum_range'] = (int(min_sum), int(max_sum))  # ✅ Python int 변환
        else:
            features['sum_range'] = (120, 180)
        
        return features
        
    except Exception as e:
        print(f"⚠️ [FEATURES] 특성 추출 오류: {str(e)[:50]}")
        # 기본 피처 반환
        return {
            'number_frequency': Counter(range(1, 46)),
            'pair_frequency': Counter(),
            'avg_odd_ratio': 0.5,
            'avg_low_ratio': 0.5,
            'recent_numbers': set(range(1, 21)),
            'bonus_frequency': Counter(),
            'appearance_gap': {i: 5 for i in range(1, 46)},
            'group_frequency': Counter({'1-15': 100, '16-30': 100, '31-45': 100}),
            'sum_range': (120, 180)
        }

def calculate_weights(df, features):
    """번호별 가중치 계산"""
    try:
        weights = {}
        
        for num in range(1, 46):
            weight = 0.0
            
            # 기본 빈도 가중치
            freq_count = features['number_frequency'].get(num, 0)
            weight += float(freq_count * 1.5)  # ✅ Python float 변환
            
            # 연속 번호 쌍 가중치
            for pair in features['pair_frequency']:
                if num in pair:
                    pair_count = features['pair_frequency'][pair]
                    weight += float(pair_count * 0.6)  # ✅ Python float 변환
            
            # 최근 출현 보너스
            if num in features['recent_numbers']:
                weight += 10.0
            
            # 보너스 번호 가중치
            bonus_count = features['bonus_frequency'].get(num, 0)
            weight += float(bonus_count * 1.0)  # ✅ Python float 변환
            
            # 등장 간격 가중치
            gap = features['appearance_gap'][num]
            weight += float(gap * 0.3)  # ✅ Python float 변환
            
            # 그룹별 가중치
            if 1 <= num <= 15:
                group_count = features['group_frequency']['1-15']
                weight += float(group_count * 0.15)  # ✅ Python float 변환
            elif 16 <= num <= 30:
                group_count = features['group_frequency']['16-30']
                weight += float(group_count * 0.15)  # ✅ Python float 변환
            elif 31 <= num <= 45:
                group_count = features['group_frequency']['31-45']
                weight += float(group_count * 0.15)  # ✅ Python float 변환
            
            # 홀짝 패턴 가중치
            is_odd = num % 2 == 1
            avg_odd = features['avg_odd_ratio']
            if (is_odd and avg_odd > 0.5) or (not is_odd and avg_odd <= 0.5):
                weight += 7.0
            
            # 고저 패턴 가중치
            is_low = num <= 22
            avg_low = features['avg_low_ratio']
            if (is_low and avg_low > 0.5) or (not is_low and avg_low <= 0.5):
                weight += 7.0
            
            # 최소 가중치 보장
            weights[num] = max(float(weight), 1.0)  # ✅ Python float 변환
        
        return weights
        
    except Exception as e:
        print(f"⚠️ [WEIGHTS] 가중치 계산 오류: {str(e)[:50]}")
        return {i: 10.0 for i in range(1, 46)}

def predict_with_features(weights, features):
    """피처 기반 번호 예측"""
    try:
        # 가중치를 확률로 변환
        total_weight = sum(weights.values())
        prob = [weights[i] / total_weight for i in range(1, 46)]
        numbers = list(range(1, 46))
        
        max_attempts = 100
        attempts = 0
        
        while attempts < max_attempts:
            selected = []
            
            # 번호 선택
            while len(selected) < 6:
                candidate = random.choices(numbers, weights=prob, k=1)[0]
                candidate = convert_to_python_int(candidate)  # ✅ Python int 변환
                if candidate not in selected:
                    selected.append(candidate)
            
            selected.sort()
            selected_sum = sum(selected)
            
            # 합계 범위 확인
            min_sum, max_sum = features['sum_range']
            if min_sum <= selected_sum <= max_sum:
                # ✅ 최종 결과를 Python int로 확실히 변환
                result = [convert_to_python_int(num) for num in selected]
                return result
            
            attempts += 1
        
        # 범위 조건을 만족하는 조합을 찾지 못한 경우 기본 로직
        selected = []
        sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        candidates = [num for num, weight in sorted_weights[:20]]
        
        # 상위 가중치에서 6개 선택
        while len(selected) < 6:
            candidate = random.choice(candidates)
            candidate = convert_to_python_int(candidate)  # ✅ Python int 변환
            if candidate not in selected:
                selected.append(candidate)
        
        # ✅ 최종 결과를 Python int로 확실히 변환하고 정렬
        result = sorted([convert_to_python_int(num) for num in selected])
        return result
        
    except Exception as e:
        print(f"⚠️ [PREDICTION] 예측 오류: {str(e)[:50]}")
        return generate_smart_random()

def generate_smart_random():
    """지능형 랜덤 생성"""
    try:
        # 통계적으로 합리적한 범위에서 선택
        candidates = []
        
        # 각 구간에서 고르게 선택
        zones = [range(1, 10), range(10, 19), range(19, 28), range(28, 37), range(37, 46)]
        for zone in zones:
            if random.random() > 0.3:  # 70% 확률로 각 구간에서 선택
                candidates.append(random.choice(zone))
        
        # 부족하면 전체 범위에서 추가
        while len(candidates) < 6:
            num = random.randint(1, 45)
            if num not in candidates:
                candidates.append(num)
        
        # ✅ Python int로 확실히 변환하여 정렬
        result = sorted([convert_to_python_int(num) for num in candidates[:6]])
        return result
        
    except Exception as e:
        print(f"⚠️ [SMART_RANDOM] 오류: {str(e)[:30]}")
        return generate_safe_fallback()

def generate_safe_fallback():
    """최후 안전장치"""
    try:
        result = sorted(random.sample(range(1, 46), 6))
        # ✅ Python int로 확실히 변환
        return [convert_to_python_int(num) for num in result]
    except Exception as e:
        print(f"⚠️ [FALLBACK] 최후 안전장치 오류: {str(e)[:30]}")
        return [7, 14, 21, 28, 35, 42]

def validate_result(result):
    """결과 유효성 검증 - 강화된 타입 안전성"""
    try:
        if not isinstance(result, (list, tuple)):
            print("⚠️ [VALIDATE] 리스트가 아님 - 안전 모드")
            return generate_safe_fallback()
        
        if len(result) != 6:
            print(f"⚠️ [VALIDATE] 길이 오류: {len(result)} != 6")
            return generate_safe_fallback()
        
        # ✅ 정수 변환 및 범위 확인 - 강화된 버전
        valid_numbers = []
        for num in result:
            try:
                if isinstance(num, (int, float, np.number)):
                    int_num = convert_to_python_int(num)
                    if 1 <= int_num <= 45:
                        valid_numbers.append(int_num)
                    else:
                        print(f"⚠️ [VALIDATE] 범위 외: {int_num}")
                        valid_numbers.append(random.randint(1, 45))
                else:
                    print(f"⚠️ [VALIDATE] 잘못된 타입: {type(num)}")
                    valid_numbers.append(random.randint(1, 45))
            except Exception as conv_error:
                print(f"⚠️ [VALIDATE] 변환 오류: {conv_error}")
                valid_numbers.append(random.randint(1, 45))
        
        if len(valid_numbers) != 6:
            print(f"⚠️ [VALIDATE] 유효 번호 부족: {len(valid_numbers)}")
            return generate_safe_fallback()
        
        # 중복 제거 및 채우기
        unique_numbers = []
        for num in valid_numbers:
            if num not in unique_numbers:
                unique_numbers.append(num)
        
        # 중복 제거 후 부족하면 채우기
        while len(unique_numbers) < 6:
            new_num = random.randint(1, 45)
            if new_num not in unique_numbers:
                unique_numbers.append(new_num)
        
        # 6개로 제한하고 정렬
        final_result = sorted(unique_numbers[:6])
        
        # ✅ 최종 검증: 모두 Python int인지 확인
        verified_result = [convert_to_python_int(num) for num in final_result]
        
        # 타입 확인 로그
        print(f"🔍 [TYPE_CHECK] 결과 타입: {[type(x).__name__ for x in verified_result]}")
        
        return verified_result
        
    except Exception as e:
        print(f"❌ [VALIDATE] 검증 실패: {str(e)[:50]}")
        return generate_safe_fallback()

# 테스트 코드 (개발용)
if __name__ == "__main__":
    # 테스트용 더미 데이터
    import pandas as pd
    import numpy as np
    
    test_data = []
    for i in range(50):
        numbers = sorted(random.sample(range(1, 46), 6))
        test_data.append({
            'round': i + 1,
            'num1': numbers[0], 'num2': numbers[1], 'num3': numbers[2],
            'num4': numbers[3], 'num5': numbers[4], 'num6': numbers[5],
            'bonus_num': random.randint(1, 45)
        })
    
    # 글로벌 변수 설정
    lotto_data = pd.DataFrame(test_data)
    
    # 테스트 실행
    result = predict_numbers()
    print(f"🔧 Super v1.0 Result: {result}")
    print(f"✅ Valid: {isinstance(result, list) and len(result) == 6 and all(isinstance(n, int) and 1 <= n <= 45 for n in result)}")
    print(f"🔍 Type Check: {[type(x).__name__ for x in result]}")
