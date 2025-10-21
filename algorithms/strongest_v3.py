"""
The Strongest in the Universe ver 3.0 - Web App Standardized Version
우주 최강 AI 예측 시스템 3.0 - 웹앱 표준화 버전

웹앱 표준 템플릿 적용:
- predict_numbers() 진입점 함수
- 글로벌 변수 사용 (lotto_data, pd, np)
- 웹앱 안전 실행 환경 준수
- 우주적 패턴 분석 및 양자역학적 선택
- JSON 직렬화 안전성 보장
"""

import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import random
import math

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
    웹앱 표준 예측 함수 - Strongest Universe v3.0 시스템
    
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
        
        # 3. Strongest Universe v3.0 알고리즘 실행
        result = run_strongest_universe_v3_algorithm(df)
        
        # 4. 결과 검증 및 반환
        final_result = validate_result(result)
        print(f"🌟 [STRONGEST] 최종 결과: {final_result}")
        
        return final_result
        
    except Exception as e:
        print(f"❌ [ERROR] Strongest Universe v3.0: {str(e)[:100]}")
        return generate_safe_fallback()

def preprocess_data(df):
    """데이터 전처리 - Strongest Universe v3.0용"""
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

def run_strongest_universe_v3_algorithm(df):
    """Strongest Universe v3.0 핵심 알고리즘"""
    try:
        if len(df) < 5:
            print("⚠️ [DATA] 데이터 부족 - 스마트 랜덤 모드")
            return generate_smart_random()
        
        number_cols = ['num1', 'num2', 'num3', 'num4', 'num5', 'num6']
        
        # 고급 특성 추출
        advanced_features = extract_advanced_features(df, number_cols)
        
        # 우주적 가중치 계산
        cosmic_weights = calculate_cosmic_weights(df, advanced_features)
        
        # 양자역학적 선택
        final_prediction = quantum_selection(cosmic_weights, advanced_features)
        
        # ✅ 모든 요소를 Python int로 확실히 변환
        safe_prediction = [convert_to_python_int(num) for num in final_prediction]
        
        return safe_prediction
        
    except Exception as e:
        print(f"❌ [ALGORITHM] Strongest v3.0 오류: {str(e)[:50]}")
        return generate_smart_random()

def extract_advanced_features(df, number_cols):
    """고급 특성 추출"""
    try:
        features = {}
        
        # 1. 피보나치 수열 분석
        fibonacci_numbers = [1, 1, 2, 3, 5, 8, 13, 21, 34]
        fib_appearances = {}
        
        for num in fibonacci_numbers:
            if num <= 45:
                count = 0
                for col in number_cols:
                    if col in df.columns:
                        count += (df[col] == num).sum()
                fib_appearances[num] = int(count)  # ✅ Python int 변환
        
        features['fibonacci'] = fib_appearances
        
        # 2. 소수 분석
        primes = [num for num in range(2, 46) if is_prime(num)]
        prime_appearances = {}
        
        for prime in primes:
            count = 0
            for col in number_cols:
                if col in df.columns:
                    count += (df[col] == prime).sum()
            prime_appearances[prime] = int(count)  # ✅ Python int 변환
            
        features['primes'] = prime_appearances
        
        # 3. 황금비 기반 수열 분석
        golden_ratio = 1.618
        golden_numbers = []
        for i in range(1, 28):
            golden_num = int(i * golden_ratio)
            if golden_num <= 45 and golden_num not in golden_numbers:
                golden_numbers.append(golden_num)
        
        features['golden_numbers'] = golden_numbers
        
        # 4. 주기성 분석
        periodicity = analyze_periodicity(df, number_cols)
        features['periodicity'] = periodicity
        
        # 5. 연관 패턴 분석
        association_patterns = analyze_association_patterns(df, number_cols)
        features['associations'] = association_patterns
        
        return features
        
    except Exception as e:
        print(f"⚠️ [FEATURES] 특성 추출 오류: {str(e)[:50]}")
        return {'fibonacci': {}, 'primes': {}, 'golden_numbers': [], 'periodicity': {}, 'associations': {}}

def is_prime(n):
    """소수 판별"""
    n = int(n)  # ✅ 안전한 타입 변환
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def analyze_periodicity(df, number_cols):
    """주기성 분석"""
    try:
        periodicity_scores = {}
        
        for num in range(1, 46):
            appearances = []
            
            # 각 번호가 나타나는 회차 찾기
            for idx, row in df.iterrows():
                row_numbers = [convert_to_python_int(row[col]) for col in number_cols if col in row]
                if num in row_numbers:
                    appearances.append(int(idx))  # ✅ Python int 변환
            
            if len(appearances) >= 2:
                # 출현 간격 계산
                intervals = [appearances[i+1] - appearances[i] for i in range(len(appearances)-1)]
                
                if intervals:
                    periodicity_scores[num] = {
                        'avg_interval': float(np.mean(intervals)),  # ✅ Python float 변환
                        'last_appearance': int(appearances[-1]),   # ✅ Python int 변환
                        'predicted_next': float(appearances[-1] + np.mean(intervals))  # ✅ Python float 변환
                    }
        
        return periodicity_scores
        
    except Exception as e:
        print(f"⚠️ [PERIODICITY] 주기성 분석 오류: {str(e)[:30]}")
        return {}

def analyze_association_patterns(df, number_cols):
    """연관 패턴 분석"""
    try:
        co_occurrence = defaultdict(int)
        number_counts = defaultdict(int)
        
        # 동시 출현 빈도 계산
        for _, row in df.iterrows():
            numbers = [convert_to_python_int(row[col]) for col in number_cols if col in row]
            
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
        total_draws = len(df)
        
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
                        association_scores[i][j] = max(0.0, float(pmi))  # ✅ Python float 변환
        
        return association_scores
        
    except Exception as e:
        print(f"⚠️ [ASSOCIATION] 연관 분석 오류: {str(e)[:30]}")
        return {}

def calculate_cosmic_weights(df, advanced_features):
    """우주적 가중치 계산"""
    try:
        number_cols = ['num1', 'num2', 'num3', 'num4', 'num5', 'num6']
        cosmic_weights = {}
        
        for num in range(1, 46):
            weight = 1.0
            
            # 1. 기본 출현 빈도
            total_appearances = 0
            for col in number_cols:
                if col in df.columns:
                    total_appearances += int((df[col] == num).sum())  # ✅ Python int 변환
            
            freq_weight = total_appearances / len(df) if len(df) > 0 else 0
            weight += freq_weight * 2.0
            
            # 2. 피보나치 보너스
            if num in advanced_features['fibonacci']:
                fibonacci_bonus = advanced_features['fibonacci'][num] / len(df)
                weight += fibonacci_bonus * 1.3
            
            # 3. 소수 보너스
            if num in advanced_features['primes']:
                prime_bonus = advanced_features['primes'][num] / len(df)
                weight += prime_bonus * 1.2
            
            # 4. 황금비 보너스
            if num in advanced_features['golden_numbers']:
                weight *= 1.15
            
            # 5. 주기성 점수
            if num in advanced_features['periodicity']:
                period_info = advanced_features['periodicity'][num]
                current_round = len(df)
                predicted_round = period_info['predicted_next']
                
                # 예측 출현 회차와 현재 회차의 거리
                distance = abs(current_round - predicted_round)
                if distance <= 3:  # 3회차 이내
                    proximity_bonus = (4 - distance) * 0.1
                    weight += proximity_bonus
            
            # 6. 최근 출현 패턴
            recent_appearances = 0
            if len(df) >= 10:
                recent_df = df.tail(10)
                for col in number_cols:
                    if col in recent_df.columns:
                        recent_appearances += int((recent_df[col] == num).sum())  # ✅ Python int 변환
            
            if recent_appearances == 0:  # 최근 미출현 보너스
                weight *= 1.2
            elif recent_appearances >= 2:  # 최근 과도출현 페널티
                weight *= 0.85
            
            # 7. 숫자학적 특성
            digit_sum = sum(int(d) for d in str(num))
            if digit_sum in [7, 11, 13]:  # 행운의 숫자
                weight *= 1.05
            
            cosmic_weights[num] = max(float(weight), 0.1)  # ✅ Python float 변환, 최소값 보장
        
        # 가중치 정규화
        total_weight = sum(cosmic_weights.values())
        for num in cosmic_weights:
            cosmic_weights[num] = float(cosmic_weights[num] / total_weight)  # ✅ Python float 변환
        
        return cosmic_weights
        
    except Exception as e:
        print(f"⚠️ [WEIGHTS] 가중치 계산 오류: {str(e)[:50]}")
        return {i: 1.0/45 for i in range(1, 46)}

def quantum_selection(cosmic_weights, advanced_features):
    """양자역학적 선택 알고리즘"""
    try:
        selected_numbers = []
        
        # 여러 시도를 통해 최적 조합 찾기
        best_combination = None
        best_score = -1
        
        for attempt in range(50):
            selected = []
            
            # 가중치 기반 선택
            while len(selected) < 6:
                available_numbers = [n for n in range(1, 46) if n not in selected]
                weights = [cosmic_weights.get(n, 0.001) for n in available_numbers]
                
                # 연관성 보정
                if len(selected) > 0:
                    associations = advanced_features.get('associations', {})
                    for i, num in enumerate(available_numbers):
                        association_bonus = 0.0
                        for selected_num in selected:
                            if selected_num in associations and num in associations[selected_num]:
                                association_bonus += associations[selected_num][num]
                        weights[i] += association_bonus * 0.1
                
                # 정규화
                total_weight = sum(weights)
                if total_weight > 0:
                    weights = [w / total_weight for w in weights]
                else:
                    weights = [1 / len(available_numbers)] * len(available_numbers)
                
                # 선택
                selected_num = np.random.choice(available_numbers, p=weights)
                selected.append(convert_to_python_int(selected_num))  # ✅ Python int 변환
            
            # 조합 평가
            score = evaluate_quantum_combination(selected, advanced_features)
            
            if score > best_score:
                best_score = score
                best_combination = selected
        
        if best_combination:
            # ✅ 최종 결과를 Python int로 확실히 변환
            result = [convert_to_python_int(num) for num in sorted(best_combination)]
            return result
        else:
            return generate_smart_random()
        
    except Exception as e:
        print(f"⚠️ [QUANTUM] 양자 선택 오류: {str(e)[:50]}")
        return generate_smart_random()

def evaluate_quantum_combination(selected, advanced_features):
    """양자 조합 평가"""
    try:
        score = 0
        
        # 기본 조화성
        total_sum = sum(selected)
        odd_count = sum(1 for n in selected if n % 2 == 1)
        
        if 120 <= total_sum <= 180:
            score += 100
        
        if 2 <= odd_count <= 4:
            score += 100
        
        # 피보나치 보너스
        fibonacci_count = sum(1 for n in selected if n in advanced_features.get('fibonacci', {}))
        score += fibonacci_count * 20
        
        # 소수 보너스
        prime_count = sum(1 for n in selected if n in advanced_features.get('primes', {}))
        score += prime_count * 15
        
        # 황금비 보너스
        golden_count = sum(1 for n in selected if n in advanced_features.get('golden_numbers', []))
        score += golden_count * 25
        
        # 연관성 점수
        associations = advanced_features.get('associations', {})
        association_score = 0.0
        for i in range(len(selected)):
            for j in range(i+1, len(selected)):
                num1, num2 = selected[i], selected[j]
                if num1 in associations and num2 in associations[num1]:
                    association_score += associations[num1][num2]
        
        score += association_score * 50
        
        return float(score)  # ✅ Python float 변환
        
    except Exception as e:
        print(f"⚠️ [EVAL] 평가 오류: {str(e)[:30]}")
        return 0.0

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
    print(f"🌟 Strongest Universe v3.0 Result: {result}")
    print(f"✅ Valid: {isinstance(result, list) and len(result) == 6 and all(isinstance(n, int) and 1 <= n <= 45 for n in result)}")
    print(f"🔍 Type Check: {[type(x).__name__ for x in result]}")
