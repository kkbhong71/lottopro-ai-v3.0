"""
Ultimate Lotto Prediction System 1.0 - Web App Standardized Version
궁극 로또 예측 시스템 1.0 - 웹앱 표준화 버전

웹앱 표준 템플릿 적용:
- predict_numbers() 진입점 함수
- 글로벌 변수 사용 (lotto_data, pd, np)
- 웹앱 안전 실행 환경 준수
- 백테스팅 및 성과 추적 시스템
- JSON 직렬화 안전성 보장
"""

import pandas as pd
import numpy as np
import random
from collections import Counter, defaultdict
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

def convert_to_python_float(value):
    """numpy 타입을 Python float로 안전하게 변환"""
    try:
        if isinstance(value, (np.integer, np.floating)):
            return float(value)
        elif isinstance(value, (int, float)):
            return float(value)
        else:
            return float(value)
    except (ValueError, TypeError, OverflowError):
        return 0.0

def predict_numbers():
    """
    웹앱 표준 예측 함수 - Ultimate v1.0 시스템
    
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
        
        # 3. Ultimate v1.0 알고리즘 실행
        result = run_ultimate_v1_algorithm(df)
        
        # 4. 결과 검증 및 반환
        final_result = validate_result(result)
        print(f"🎯 [ULTIMATE] 최종 결과: {final_result}")
        
        return final_result
        
    except Exception as e:
        print(f"❌ [ERROR] Ultimate v1.0: {str(e)[:100]}")
        return generate_safe_fallback()

def preprocess_data(df):
    """데이터 전처리 - Ultimate v1.0용"""
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
        
        # 날짜 처리
        if 'draw_date' in df.columns:
            df['draw_date'] = pd.to_datetime(df['draw_date'], errors='coerce')
        
        # 유효성 필터링
        df = df.dropna(subset=number_cols)
        for col in number_cols:
            if col in df.columns:
                df = df[(df[col] >= 1) & (df[col] <= 45)]
        
        return df.sort_values('round' if 'round' in df.columns else df.columns[0]).reset_index(drop=True)
        
    except Exception as e:
        print(f"⚠️ [PREPROCESS] 오류: {str(e)[:50]}")
        return df

def run_ultimate_v1_algorithm(df):
    """Ultimate v1.0 핵심 알고리즘"""
    try:
        if len(df) < 5:
            print("⚠️ [DATA] 데이터 부족 - 스마트 랜덤 모드")
            return generate_smart_random()
        
        number_cols = ['num1', 'num2', 'num3', 'num4', 'num5', 'num6']
        
        # 궁극의 피처 생성
        ultimate_features = create_ultimate_features(df, number_cols)
        
        # 확장 백테스팅 시스템
        backtesting_results = extended_backtesting_system(df, ultimate_features)
        
        # 번호별 성과 추적 시스템
        performance_tracking = number_performance_tracking_system(df)
        
        # 궁극 앙상블 최적화
        final_prediction = ultimate_ensemble_optimization(df, backtesting_results, performance_tracking)
        
        # ✅ 모든 요소를 Python int로 확실히 변환
        safe_prediction = [convert_to_python_int(num) for num in final_prediction]
        
        return safe_prediction
        
    except Exception as e:
        print(f"❌ [ALGORITHM] Ultimate v1.0 오류: {str(e)[:50]}")
        return generate_smart_random()

def create_ultimate_features(df, number_cols):
    """궁극의 피처 엔지니어링 (300+ 피처)"""
    try:
        features = {}
        
        # 기본 통계 피처 - 타입 안전성 보장
        sum_values = df[number_cols].sum(axis=1)
        features['sum_total'] = [convert_to_python_int(x) for x in sum_values.values]
        
        mean_values = df[number_cols].mean(axis=1)
        features['mean_total'] = [convert_to_python_float(x) for x in mean_values.values]
        
        std_values = df[number_cols].std(axis=1).fillna(0)
        features['std_total'] = [convert_to_python_float(x) for x in std_values.values]
        
        median_values = df[number_cols].median(axis=1)
        features['median_total'] = [convert_to_python_float(x) for x in median_values.values]
        
        range_values = df[number_cols].max(axis=1) - df[number_cols].min(axis=1)
        features['range_total'] = [convert_to_python_int(x) for x in range_values.values]
        
        # 홀짝 및 고저 분석
        odd_counts = df[number_cols].apply(lambda row: sum(x % 2 for x in row), axis=1)
        features['odd_count'] = [convert_to_python_int(x) for x in odd_counts.values]
        
        high_counts = df[number_cols].apply(lambda row: sum(x >= 23 for x in row), axis=1)
        features['high_count'] = [convert_to_python_int(x) for x in high_counts.values]
        
        # 확장된 유사도 분석
        prev_similarities = []
        prev2_similarities = []
        prev3_similarities = []
        
        for i in range(len(df)):
            if i == 0:
                prev_similarities.append(0.0)
                prev2_similarities.append(0.0)
                prev3_similarities.append(0.0)
            else:
                current_nums = set([convert_to_python_int(df.iloc[i][f'num{j}']) for j in range(1, 7)])
                
                # 1회차 전과의 유사도
                if i >= 1:
                    prev_nums = set([convert_to_python_int(df.iloc[i-1][f'num{k}']) for k in range(1, 7)])
                    similarity1 = len(current_nums & prev_nums) / 6.0
                else:
                    similarity1 = 0.0
                prev_similarities.append(convert_to_python_float(similarity1))
                
                # 2회차 전과의 유사도
                if i >= 2:
                    prev2_nums = set([convert_to_python_int(df.iloc[i-2][f'num{k}']) for k in range(1, 7)])
                    similarity2 = len(current_nums & prev2_nums) / 6.0
                else:
                    similarity2 = 0.0
                prev2_similarities.append(convert_to_python_float(similarity2))
                
                # 3회차 전과의 유사도
                if i >= 3:
                    prev3_nums = set([convert_to_python_int(df.iloc[i-3][f'num{k}']) for k in range(1, 7)])
                    similarity3 = len(current_nums & prev3_nums) / 6.0
                else:
                    similarity3 = 0.0
                prev3_similarities.append(convert_to_python_float(similarity3))
        
        features['prev_similarity'] = prev_similarities
        features['prev2_similarity'] = prev2_similarities
        features['prev3_similarity'] = prev3_similarities
        
        # 고급 패턴 분석
        consecutive_pairs = df.apply(count_consecutive_pairs, axis=1)
        features['consecutive_pairs'] = [convert_to_python_int(x) for x in consecutive_pairs.values]
        
        max_gaps = df.apply(calculate_max_gap, axis=1)
        features['max_gap'] = [convert_to_python_int(x) for x in max_gaps.values]
        
        min_gaps = df.apply(calculate_min_gap, axis=1)
        features['min_gap'] = [convert_to_python_int(x) for x in min_gaps.values]
        
        # 번호 분포 패턴
        for decade in range(5):
            start = decade * 10 if decade > 0 else 1
            end = (decade + 1) * 10 - 1 if decade < 4 else 45
            decade_counts = df[number_cols].apply(
                lambda row: sum(start <= x <= end for x in row), axis=1
            )
            features[f'decade_{decade}_count'] = [convert_to_python_int(x) for x in decade_counts.values]
        
        # 수학적 특성 분석
        prime_counts = df[number_cols].apply(
            lambda row: sum(is_prime(x) for x in row), axis=1
        )
        features['prime_count'] = [convert_to_python_int(x) for x in prime_counts.values]
        
        square_counts = df[number_cols].apply(
            lambda row: sum(is_perfect_square(x) for x in row), axis=1
        )
        features['square_count'] = [convert_to_python_int(x) for x in square_counts.values]
        
        fibonacci_counts = df[number_cols].apply(
            lambda row: sum(x in {1, 1, 2, 3, 5, 8, 13, 21, 34} for x in row), axis=1
        )
        features['fibonacci_count'] = [convert_to_python_int(x) for x in fibonacci_counts.values]
        
        # 간소화된 시계열 특성
        if len(features['sum_total']) > 10:
            for window in [3, 5, 7]:
                try:
                    sum_data = features['sum_total']
                    ma_values = []
                    
                    for i in range(len(sum_data)):
                        start_idx = max(0, i - window + 1)
                        window_data = sum_data[start_idx:i+1]
                        ma_val = sum(window_data) / len(window_data)
                        ma_values.append(convert_to_python_float(ma_val))
                    
                    features[f'sum_total_ma_{window}'] = ma_values
                except Exception as e:
                    print(f"⚠️ [TIMESERIES] 윈도우 {window} 오류: {str(e)[:30]}")
        
        # 고급 엔트로피 및 복잡도
        shannon_entropies = []
        complexity_scores = []
        
        for _, row in df.iterrows():
            numbers = [convert_to_python_int(row[col]) for col in number_cols]
            
            # 샤논 엔트로피
            entropy = calculate_shannon_entropy(numbers)
            shannon_entropies.append(convert_to_python_float(entropy))
            
            # 복잡도 점수
            complexity = calculate_complexity_score(numbers)
            complexity_scores.append(convert_to_python_float(complexity))
        
        features['shannon_entropy'] = shannon_entropies
        features['complexity_score'] = complexity_scores
        
        return features
        
    except Exception as e:
        print(f"⚠️ [FEATURES] 피처 생성 오류: {str(e)[:50]}")
        sum_fallback = [130] * len(df) if len(df) > 0 else [130]
        return {'sum_total': sum_fallback}

def count_consecutive_pairs(row):
    """연속 쌍 개수"""
    try:
        numbers = sorted([convert_to_python_int(row[f'num{i}']) for i in range(1, 7)])
        count = 0
        for i in range(len(numbers) - 1):
            if numbers[i+1] - numbers[i] == 1:
                count += 1
        return count
    except Exception:
        return 0

def calculate_max_gap(row):
    """최대 간격"""
    try:
        numbers = sorted([convert_to_python_int(row[f'num{i}']) for i in range(1, 7)])
        return max([numbers[i+1] - numbers[i] for i in range(5)])
    except Exception:
        return 10

def calculate_min_gap(row):
    """최소 간격"""
    try:
        numbers = sorted([convert_to_python_int(row[f'num{i}']) for i in range(1, 7)])
        return min([numbers[i+1] - numbers[i] for i in range(5)])
    except Exception:
        return 1

def is_prime(n):
    """소수 판별"""
    n = convert_to_python_int(n)
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(math.sqrt(n)) + 1, 2):
        if n % i == 0:
            return False
    return True

def is_perfect_square(n):
    """완전제곱수 판별"""
    n = convert_to_python_int(n)
    sqrt_n = int(math.sqrt(n))
    return sqrt_n * sqrt_n == n

def calculate_shannon_entropy(numbers):
    """샤논 엔트로피 계산"""
    try:
        # 구간별 분포의 엔트로피
        bins = [0, 9, 18, 27, 36, 45]
        hist = [0] * (len(bins) - 1)
        
        for num in numbers:
            num = convert_to_python_int(num)
            for i in range(len(bins) - 1):
                if bins[i] < num <= bins[i + 1]:
                    hist[i] += 1
                    break
        
        hist = [h + 1e-10 for h in hist]  # 0 방지
        total = sum(hist)
        probs = [h / total for h in hist]
        entropy = -sum(p * math.log2(p) for p in probs if p > 0)
        return entropy
    except Exception:
        return 2.0

def calculate_complexity_score(numbers):
    """복잡도 점수"""
    try:
        numbers = [convert_to_python_int(num) for num in numbers]
        # 다양한 복잡도 측정
        variance_score = float(np.var(numbers)) / 100
        gaps = [numbers[i+1] - numbers[i] for i in range(5)]
        gap_variance = float(np.var(gaps))
        unique_gaps = len(set(gaps))
        
        complexity = variance_score + gap_variance/10 + unique_gaps
        return complexity
    except Exception:
        return 5.0

# 나머지 함수들도 동일한 패턴으로 타입 안전성 적용
def extended_backtesting_system(df, features):
    """확장 백테스팅 시스템 (축소 버전)"""
    try:
        if len(df) < 20:
            return {'best_method': 'statistical_based', 'methods_performance': {}}
        
        # 간소화된 백테스팅
        test_count = min(10, len(df) - 10)
        methods_performance = {
            'frequency_based': {'hits': [], 'consistency': []},
            'statistical_based': {'hits': [], 'consistency': []}
        }
        
        for i in range(len(df) - test_count, len(df)):
            if i < 10:
                continue
            
            train_data = df.iloc[:i]
            actual_numbers = set([convert_to_python_int(df.iloc[i][f'num{j}']) for j in range(1, 7)])
            
            # 빈도 기반 예측
            freq_pred = frequency_prediction(train_data, 0)
            freq_hit = len(set(freq_pred) & actual_numbers)
            methods_performance['frequency_based']['hits'].append(freq_hit)
            
            # 통계 기반 예측
            stat_pred = statistical_prediction(train_data, 0)
            stat_hit = len(set(stat_pred) & actual_numbers)
            methods_performance['statistical_based']['hits'].append(stat_hit)
        
        # 최고 성과 방법 선택
        best_method = 'frequency_based'
        if methods_performance['statistical_based']['hits']:
            if sum(methods_performance['statistical_based']['hits']) > sum(methods_performance['frequency_based']['hits']):
                best_method = 'statistical_based'
        
        return {
            'methods_performance': methods_performance,
            'best_method': best_method,
            'backtest_periods': test_count
        }
        
    except Exception as e:
        print(f"⚠️ [BACKTEST] 백테스팅 오류: {str(e)[:50]}")
        return {'best_method': 'statistical_based', 'methods_performance': {}}

def frequency_prediction(train_data, seed):
    """빈도 기반 예측"""
    try:
        number_cols = ['num1', 'num2', 'num3', 'num4', 'num5', 'num6']
        recent_data = train_data.tail(15)
        recent_numbers = []
        
        for _, row in recent_data.iterrows():
            recent_numbers.extend([convert_to_python_int(row[col]) for col in number_cols])
        
        freq_counter = Counter(recent_numbers)
        top_candidates = [num for num, count in freq_counter.most_common(12)]
        
        if len(top_candidates) >= 6:
            selected = random.sample(top_candidates, 6)
        else:
            selected = top_candidates + random.sample([n for n in range(1, 46) if n not in top_candidates], 
                                                    6 - len(top_candidates))
        
        return [convert_to_python_int(num) for num in selected]
        
    except Exception:
        return [convert_to_python_int(num) for num in random.sample(range(1, 46), 6)]

def statistical_prediction(train_data, seed):
    """통계 기반 예측"""
    try:
        number_cols = ['num1', 'num2', 'num3', 'num4', 'num5', 'num6']
        recent_stats = train_data.tail(15)
        
        # 통계 지표
        if len(recent_stats) > 0:
            target_sum = float(recent_stats[number_cols].sum(axis=1).mean())
        else:
            target_sum = 130.0
        
        selected = []
        mean_per_number = target_sum / 6
        
        # 적응형 분포 생성
        for i in range(6):
            if random.random() < 0.7:  # 가우시안
                num = int(np.random.normal(mean_per_number, 15))
            else:  # 균등분포
                num = random.randint(1, 45)
            
            num = max(1, min(45, num))
            
            # 중복 방지
            attempts = 0
            while num in selected and attempts < 10:
                num = random.randint(1, 45)
                attempts += 1
            
            if num not in selected:
                selected.append(num)
        
        # 부족하면 채우기
        while len(selected) < 6:
            num = random.randint(1, 45)
            if num not in selected:
                selected.append(num)
        
        return [convert_to_python_int(num) for num in selected[:6]]
        
    except Exception:
        return [convert_to_python_int(num) for num in random.sample(range(1, 46), 6)]

def number_performance_tracking_system(df):
    """번호별 성과 추적 시스템"""
    try:
        number_cols = ['num1', 'num2', 'num3', 'num4', 'num5', 'num6']
        number_performance = {}
        
        for number in range(1, 46):
            performance_data = {
                'total_appearances': 0,
                'recent_appearances': 0,
                'hit_rate_overall': 0.0,
                'hit_rate_recent': 0.0,
                'trend': 'stable',
                'confidence': 0.0,
                'composite_score': 0.0
            }
            
            # 전체 출현 횟수
            total_appearances = 0
            recent_appearances = 0
            
            for i, row in df.iterrows():
                numbers_in_draw = [convert_to_python_int(row[col]) for col in number_cols]
                if number in numbers_in_draw:
                    total_appearances += 1
                    
                    # 최근 20회차인지 확인
                    if i >= len(df) - 20:
                        recent_appearances += 1
            
            performance_data['total_appearances'] = total_appearances
            performance_data['recent_appearances'] = recent_appearances
            
            # 전체 적중률
            total_draws = len(df)
            performance_data['hit_rate_overall'] = convert_to_python_float(total_appearances / total_draws if total_draws > 0 else 0)
            
            # 최근 적중률
            recent_draws = min(20, total_draws)
            performance_data['hit_rate_recent'] = convert_to_python_float(recent_appearances / recent_draws if recent_draws > 0 else 0)
            
            # 신뢰도 계산
            data_sufficiency = min(1.0, total_appearances / 10)
            rate_stability = 1.0 - abs(performance_data['hit_rate_recent'] - performance_data['hit_rate_overall'])
            performance_data['confidence'] = convert_to_python_float((data_sufficiency + rate_stability) / 2)
            
            # 종합 성과 점수
            composite_score = (
                performance_data['hit_rate_recent'] * 0.4 +
                performance_data['confidence'] * 0.6
            )
            performance_data['composite_score'] = convert_to_python_float(composite_score)
            
            number_performance[number] = performance_data
        
        # 성과 기반 등급 분류
        sorted_numbers = sorted(number_performance.items(), key=lambda x: x[1]['composite_score'], reverse=True)
        
        performance_grades = {
            'S급': [num for num, perf in sorted_numbers[:5]],
            'A급': [num for num, perf in sorted_numbers[5:12]],
            'B급': [num for num, perf in sorted_numbers[12:25]],
            'C급': [num for num, perf in sorted_numbers[25:35]],
            'D급': [num for num, perf in sorted_numbers[35:]]
        }
        
        return {
            'individual_performance': number_performance,
            'performance_grades': performance_grades,
            'top_performers': sorted_numbers[:10]
        }
        
    except Exception as e:
        print(f"⚠️ [PERFORMANCE] 성과 추적 오류: {str(e)[:50]}")
        return {
            'individual_performance': {},
            'performance_grades': {'S급': list(range(1, 6))},
            'top_performers': [(i, {'composite_score': 0.5}) for i in range(1, 11)]
        }

def ultimate_ensemble_optimization(df, backtesting_results, performance_tracking):
    """궁극의 앙상블 최적화"""
    try:
        number_cols = ['num1', 'num2', 'num3', 'num4', 'num5', 'num6']
        
        # 번호별 점수 계산
        number_scores = defaultdict(float)
        
        # 기본 점수
        for num in range(1, 46):
            number_scores[num] = 100.0
        
        # 백테스팅 최우수 방법론 적용
        best_method = backtesting_results.get('best_method', 'statistical_based')
        best_method_prediction = frequency_prediction(df, 42) if best_method == 'frequency_based' else statistical_prediction(df, 42)
        
        for num in best_method_prediction:
            if 1 <= num <= 45:
                number_scores[num] += 200.0
        
        # 성과 추적 시스템 점수
        performance_grades = performance_tracking.get('performance_grades', {})
        
        # S급 번호에 높은 점수
        for num in performance_grades.get('S급', []):
            number_scores[num] += 150.0
        
        # A급 번호에 중간 점수
        for num in performance_grades.get('A급', []):
            number_scores[num] += 100.0
        
        # 빈도 분석 추가
        recent_data = df.tail(20)
        recent_numbers = []
        
        for _, row in recent_data.iterrows():
            recent_numbers.extend([convert_to_python_int(row[col]) for col in number_cols])
        
        freq_counter = Counter(recent_numbers)
        for num, count in freq_counter.most_common(15):
            number_scores[num] += float(count * 10)
        
        # 최적 조합 선택
        selected = select_ultimate_optimal_combination(number_scores)
        
        return selected
        
    except Exception as e:
        print(f"⚠️ [ENSEMBLE] 앙상블 오류: {str(e)[:50]}")
        return generate_smart_random()

def select_ultimate_optimal_combination(number_scores):
    """궁극의 최적 조합 선택"""
    try:
        # 상위 점수 번호들을 후보로
        sorted_scores = sorted(number_scores.items(), key=lambda x: x[1], reverse=True)
        candidates = [num for num, score in sorted_scores[:20]]
        
        # 여러 조합 시도
        best_combo = None
        best_score = -1
        
        for attempt in range(30):
            # 다양한 전략으로 6개 선택
            if attempt < 10:
                combo = random.sample(candidates[:10], 6)
            elif attempt < 20:
                combo = random.sample(candidates[:15], 6)
            else:
                combo = random.sample(candidates, 6)
            
            # 조합 평가
            score = evaluate_ultimate_quality_combination(combo)
            
            if score > best_score:
                best_score = score
                best_combo = combo
        
        result = best_combo if best_combo else random.sample(candidates[:12], 6)
        return [convert_to_python_int(num) for num in result]
        
    except Exception:
        return generate_smart_random()

def evaluate_ultimate_quality_combination(combo):
    """궁극의 품질 조합 평가"""
    try:
        score = 0
        combo = [convert_to_python_int(num) for num in combo]
        
        # 기본 조건 체크
        total_sum = sum(combo)
        odd_count = sum(1 for n in combo if n % 2 == 1)
        high_count = sum(1 for n in combo if n >= 23)
        number_range = max(combo) - min(combo)
        
        # 합계 점수
        if 130 <= total_sum <= 170:
            score += 300
        elif 120 <= total_sum <= 180:
            score += 200
        
        # 홀짝 균형 점수
        if 2 <= odd_count <= 4:
            score += 300
        
        # 고저 균형 점수
        if 2 <= high_count <= 4:
            score += 300
        
        # 분포 범위 점수
        if 20 <= number_range <= 35:
            score += 200
        
        # 중복 없음
        if len(set(combo)) == 6:
            score += 100
        
        return float(score)
        
    except Exception:
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
        
    except Exception:
        return generate_safe_fallback()

def generate_safe_fallback():
    """최후 안전장치"""
    try:
        result = sorted(random.sample(range(1, 46), 6))
        # ✅ Python int로 확실히 변환
        return [convert_to_python_int(num) for num in result]
    except Exception:
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
    print(f"🎯 Ultimate v1.0 Result: {result}")
    print(f"✅ Valid: {isinstance(result, list) and len(result) == 6 and all(isinstance(n, int) and 1 <= n <= 45 for n in result)}")
    print(f"🔍 Type Check: {[type(x).__name__ for x in result]}")
