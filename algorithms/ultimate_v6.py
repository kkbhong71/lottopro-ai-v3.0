"""
Ultimate Prediction 6.0 - Sum Range Analysis
합계 범위 분석 알고리즘 - 웹앱 표준화 버전

특징:
- 당첨번호 합계의 최적 범위 분석
- 통계적 분포 기반 예측
- 확률론적 범위 설정
- 웹앱 표준 인터페이스 준수
"""

import pandas as pd
import numpy as np
import random
from collections import Counter, defaultdict

# 안전한 warnings 처리
try:
    import warnings
    warnings.filterwarnings('ignore')
except ImportError:
    pass

def predict_numbers():
    """
    웹앱 표준 예측 함수 - Ultimate v6.0 합계 범위 분석
    
    글로벌 변수 사용:
    - lotto_data: pandas DataFrame (로또 당첨번호 데이터)
    - pd: pandas 라이브러리
    - np: numpy 라이브러리
    
    Returns:
        list: 정확히 6개의 로또 번호 [1-45 범위의 정수]
    """
    try:
        # ⭐ 1단계: globals() 체크 (필수!)
        if 'lotto_data' not in globals():
            print("⚠️ lotto_data not found in globals()")
            return generate_safe_fallback()
        
        # ⭐ 2단계: DataFrame empty 체크 (필수!)
        if lotto_data.empty:
            print("⚠️ lotto_data is empty")
            return generate_safe_fallback()
        
        # ⭐ 3단계: 데이터 복사
        df = lotto_data.copy()
        
        if len(df) < 5:
            return generate_safe_fallback()
        
        # 4단계: 데이터 전처리
        df = preprocess_data(df)
        
        if len(df) < 5:
            return generate_safe_fallback()
        
        # 5단계: 합계 범위 분석 실행
        sum_analysis = analyze_sum_ranges(df)
        
        # 6단계: 최적 범위 내에서 번호 생성
        selected_numbers = generate_numbers_in_range(df, sum_analysis)
        
        # 7단계: 결과 검증 및 반환
        return validate_result(selected_numbers)
        
    except NameError as e:
        print(f"❌ NameError: lotto_data를 찾을 수 없음 - {e}")
        return generate_safe_fallback()
    except AttributeError as e:
        print(f"❌ AttributeError: {e}")
        return generate_safe_fallback()
    except Exception as e:
        print(f"❌ Ultimate v6.0 실행 오류: {e}")
        return generate_safe_fallback()

def preprocess_data(df):
    """데이터 전처리 - 컬럼명 정규화 및 유효성 검증"""
    try:
        # 컬럼명 정규화
        df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]
        
        # 표준 컬럼 매핑
        if len(df.columns) >= 9:
            standard_cols = ['round', 'draw_date', 'num1', 'num2', 'num3', 'num4', 'num5', 'num6', 'bonus_num']
            mapping = dict(zip(df.columns[:9], standard_cols))
            df = df.rename(columns=mapping)
        
        # ⭐ 숫자 컬럼 변환 (num1~num6 사용)
        number_cols = ['num1', 'num2', 'num3', 'num4', 'num5', 'num6']
        for col in number_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 유효성 필터링
        df = df.dropna(subset=number_cols)
        for col in number_cols:
            if col in df.columns:
                df = df[(df[col] >= 1) & (df[col] <= 45)]
        
        # 회차순 정렬
        if 'round' in df.columns:
            df = df.sort_values('round').reset_index(drop=True)
        
        return df
        
    except Exception as e:
        print(f"전처리 오류: {e}")
        return df

def analyze_sum_ranges(df):
    """합계 범위 분석 - Ultimate v6.0 핵심 알고리즘"""
    try:
        number_cols = ['num1', 'num2', 'num3', 'num4', 'num5', 'num6']
        
        # 1. 모든 회차의 합계 계산
        sum_totals = []
        for _, row in df.iterrows():
            row_sum = sum([row[col] for col in number_cols if col in row and pd.notna(row[col])])
            if row_sum > 0:
                sum_totals.append(row_sum)
        
        if len(sum_totals) == 0:
            return get_default_sum_analysis()
        
        # 2. 통계적 분석
        mean_sum = np.mean(sum_totals)
        std_sum = np.std(sum_totals)
        median_sum = np.median(sum_totals)
        
        # 3. 분포 분석 (백분위수)
        percentiles = {
            'p10': np.percentile(sum_totals, 10),
            'p20': np.percentile(sum_totals, 20),
            'p30': np.percentile(sum_totals, 30),
            'p40': np.percentile(sum_totals, 40),
            'p50': np.percentile(sum_totals, 50),
            'p60': np.percentile(sum_totals, 60),
            'p70': np.percentile(sum_totals, 70),
            'p80': np.percentile(sum_totals, 80),
            'p90': np.percentile(sum_totals, 90)
        }
        
        # 4. 최적 범위 계산 (상위 60% 구간)
        optimal_min = int(percentiles['p20'])
        optimal_max = int(percentiles['p80'])
        
        # 5. 최근 트렌드 분석 (최근 20회차)
        recent_sums = sum_totals[-20:] if len(sum_totals) >= 20 else sum_totals
        recent_mean = np.mean(recent_sums)
        
        # 6. 빈도 분석 (가장 자주 나오는 합계 구간)
        sum_ranges = defaultdict(int)
        for s in sum_totals:
            range_key = (s // 10) * 10  # 10 단위로 그룹화
            sum_ranges[range_key] += 1
        
        most_frequent_range = max(sum_ranges.items(), key=lambda x: x[1])[0] if sum_ranges else 135
        
        return {
            'mean_sum': mean_sum,
            'std_sum': std_sum,
            'median_sum': median_sum,
            'percentiles': percentiles,
            'optimal_range': (optimal_min, optimal_max),
            'recent_trend': recent_mean,
            'most_frequent_range': most_frequent_range,
            'sum_distribution': dict(sum_ranges),
            'total_samples': len(sum_totals)
        }
        
    except Exception as e:
        print(f"합계 분석 오류: {e}")
        return get_default_sum_analysis()

def get_default_sum_analysis():
    """기본 합계 분석 데이터"""
    return {
        'mean_sum': 135,
        'std_sum': 25,
        'median_sum': 135,
        'percentiles': {
            'p20': 110, 'p40': 125, 'p50': 135, 'p60': 145, 'p80': 160
        },
        'optimal_range': (110, 160),
        'recent_trend': 135,
        'most_frequent_range': 130,
        'sum_distribution': {},
        'total_samples': 0
    }

def generate_numbers_in_range(df, sum_analysis):
    """최적 합계 범위 내에서 번호 조합 생성"""
    try:
        optimal_range = sum_analysis['optimal_range']
        target_min = optimal_range[0]
        target_max = optimal_range[1]
        
        # 최근 데이터 기반 후보 번호 추출
        number_cols = ['num1', 'num2', 'num3', 'num4', 'num5', 'num6']
        candidate_numbers = []
        
        # 최근 50회차 데이터에서 빈도 높은 번호 추출
        recent_data = df.tail(50) if len(df) >= 50 else df
        for _, row in recent_data.iterrows():
            for col in number_cols:
                if col in row and pd.notna(row[col]):
                    candidate_numbers.append(int(row[col]))
        
        # 빈도 분석
        number_frequency = Counter(candidate_numbers)
        hot_numbers = [num for num, _ in number_frequency.most_common(30)]
        
        # 후보 풀 확장 (전체 번호)
        if len(hot_numbers) < 20:
            hot_numbers.extend([i for i in range(1, 46) if i not in hot_numbers])
        
        # 최적 조합 탐색 (최대 200회 시도)
        best_combination = None
        best_score = 0
        
        for attempt in range(200):
            # 후보에서 6개 선택
            selected = select_balanced_numbers(
                hot_numbers, 
                target_min, 
                target_max,
                sum_analysis
            )
            
            if len(selected) == 6:
                # 조합 평가
                score = evaluate_combination(selected, sum_analysis)
                
                if score > best_score:
                    best_score = score
                    best_combination = selected
        
        if best_combination and len(best_combination) == 6:
            return sorted(best_combination)
        else:
            # 목표 범위 내 랜덤 생성
            return generate_random_in_range(target_min, target_max)
        
    except Exception as e:
        print(f"번호 생성 오류: {e}")
        return generate_safe_fallback()

def select_balanced_numbers(candidates, target_min, target_max, sum_analysis):
    """균형잡힌 번호 선택"""
    try:
        # 후보 중에서 랜덤 선택
        if len(candidates) < 6:
            candidates = list(range(1, 46))
        
        selected = random.sample(candidates, min(6, len(candidates)))
        
        # 부족하면 보충
        while len(selected) < 6:
            num = random.randint(1, 45)
            if num not in selected:
                selected.append(num)
        
        selected = selected[:6]
        current_sum = sum(selected)
        
        # 합계 범위 조정 (최대 10번 시도)
        for adjustment in range(10):
            if target_min <= current_sum <= target_max:
                break
            
            if current_sum < target_min:
                # 합계가 작으면 큰 번호로 교체
                min_idx = selected.index(min(selected))
                new_num = random.randint(max(selected) + 1, 45)
                if new_num <= 45 and new_num not in selected:
                    selected[min_idx] = new_num
            else:
                # 합계가 크면 작은 번호로 교체
                max_idx = selected.index(max(selected))
                new_num = random.randint(1, min(selected) - 1)
                if new_num >= 1 and new_num not in selected:
                    selected[max_idx] = new_num
            
            current_sum = sum(selected)
        
        return selected
        
    except Exception as e:
        print(f"균형 선택 오류: {e}")
        return list(range(1, 7))

def evaluate_combination(selected, sum_analysis):
    """조합 평가 점수 계산"""
    try:
        score = 0
        current_sum = sum(selected)
        
        # 1. 최적 범위 내 점수 (50점)
        optimal_range = sum_analysis['optimal_range']
        if optimal_range[0] <= current_sum <= optimal_range[1]:
            score += 50
        else:
            # 범위 밖이면 거리에 따라 감점
            if current_sum < optimal_range[0]:
                distance = optimal_range[0] - current_sum
            else:
                distance = current_sum - optimal_range[1]
            score += max(0, 50 - distance * 2)
        
        # 2. 평균과의 근접도 (30점)
        mean_sum = sum_analysis['mean_sum']
        distance_from_mean = abs(current_sum - mean_sum)
        score += max(0, 30 - distance_from_mean)
        
        # 3. 최근 트렌드와의 유사도 (20점)
        recent_trend = sum_analysis['recent_trend']
        trend_distance = abs(current_sum - recent_trend)
        score += max(0, 20 - trend_distance)
        
        # 4. 홀짝 균형 (보너스 10점)
        odd_count = sum(1 for num in selected if num % 2 == 1)
        if 2 <= odd_count <= 4:
            score += 10
        
        # 5. 고저 균형 (보너스 10점)
        high_count = sum(1 for num in selected if num >= 23)
        if 2 <= high_count <= 4:
            score += 10
        
        return score
        
    except Exception as e:
        print(f"평가 오류: {e}")
        return 0

def generate_random_in_range(target_min, target_max):
    """목표 범위 내 랜덤 번호 생성"""
    try:
        attempts = 0
        max_attempts = 100
        
        while attempts < max_attempts:
            # 랜덤으로 6개 선택
            selected = sorted(random.sample(range(1, 46), 6))
            current_sum = sum(selected)
            
            # 범위 체크
            if target_min <= current_sum <= target_max:
                return selected
            
            attempts += 1
        
        # 실패 시 중간값 기준 생성
        target_sum = (target_min + target_max) // 2
        return generate_by_target_sum(target_sum)
        
    except Exception as e:
        print(f"범위 내 생성 오류: {e}")
        return generate_safe_fallback()

def generate_by_target_sum(target_sum):
    """목표 합계에 맞춰 번호 생성"""
    try:
        # 평균값 기준
        avg = target_sum / 6
        
        selected = []
        remaining_sum = target_sum
        
        for i in range(5):
            # 남은 합계를 고려하여 번호 선택
            min_val = max(1, remaining_sum - (5 - i) * 45)
            max_val = min(45, remaining_sum - (5 - i))
            
            if min_val <= max_val:
                num = random.randint(int(min_val), int(max_val))
                if num not in selected:
                    selected.append(num)
                    remaining_sum -= num
        
        # 마지막 번호
        if 1 <= remaining_sum <= 45 and remaining_sum not in selected:
            selected.append(remaining_sum)
        
        # 6개가 안되면 보충
        while len(selected) < 6:
            num = random.randint(1, 45)
            if num not in selected:
                selected.append(num)
        
        return sorted(selected[:6])
        
    except Exception as e:
        print(f"목표 합계 생성 오류: {e}")
        return generate_safe_fallback()

def generate_safe_fallback():
    """안전장치: 기본 번호 생성"""
    try:
        # 통계적으로 안정적인 범위에서 생성
        # 평균 합계 135 근처 (120~150)
        selected = []
        
        # 각 구간에서 고르게 선택
        zones = [
            range(1, 10),    # 저구간
            range(10, 19),   # 중저구간
            range(19, 28),   # 중구간
            range(28, 37),   # 중고구간
            range(37, 46)    # 고구간
        ]
        
        for zone in zones[:5]:
            num = random.choice(zone)
            if num not in selected:
                selected.append(num)
        
        # 6번째 번호 추가
        while len(selected) < 6:
            num = random.randint(1, 45)
            if num not in selected:
                selected.append(num)
        
        return sorted(selected[:6])
        
    except Exception:
        # 최후의 수단
        return [7, 14, 21, 28, 35, 42]

def validate_result(result):
    """결과 유효성 검증"""
    try:
        if not isinstance(result, (list, tuple)):
            return generate_safe_fallback()
        
        if len(result) != 6:
            return generate_safe_fallback()
        
        # 정수 변환 및 범위 확인
        valid_numbers = []
        for num in result:
            if isinstance(num, (int, float, np.integer)):
                int_num = int(num)
                if 1 <= int_num <= 45:
                    valid_numbers.append(int_num)
        
        if len(valid_numbers) != 6:
            return generate_safe_fallback()
        
        # 중복 제거
        if len(set(valid_numbers)) != 6:
            return generate_safe_fallback()
        
        return sorted(valid_numbers)
        
    except Exception:
        return generate_safe_fallback()

# 테스트 코드 (개발용)
if __name__ == "__main__":
    print("=" * 60)
    print("Ultimate Prediction 6.0 - Sum Range Analysis")
    print("합계 범위 분석 알고리즘 테스트")
    print("=" * 60)
    
    # 테스트용 더미 데이터 생성
    test_data = []
    for i in range(100):
        numbers = sorted(random.sample(range(1, 46), 6))
        test_data.append({
            'round': i + 1,
            'draw_date': f'2024.{(i%12)+1:02d}.{(i%28)+1:02d}',
            'num1': numbers[0],
            'num2': numbers[1],
            'num3': numbers[2],
            'num4': numbers[3],
            'num5': numbers[4],
            'num6': numbers[5],
            'bonus_num': random.randint(1, 45)
        })
    
    # 글로벌 변수 설정
    lotto_data = pd.DataFrame(test_data)
    
    print(f"\n테스트 데이터: {len(lotto_data)}회차")
    print(f"컬럼: {list(lotto_data.columns)}")
    
    # 알고리즘 실행
    print("\n알고리즘 실행 중...")
    result = predict_numbers()
    
    print(f"\n✅ 예측 결과: {result}")
    print(f"   타입: {type(result)}")
    print(f"   개수: {len(result)}")
    print(f"   합계: {sum(result)}")
    print(f"   홀수: {sum(1 for n in result if n % 2 == 1)}개")
    print(f"   고수: {sum(1 for n in result if n >= 23)}개")
    
    # 유효성 검증
    is_valid = (
        isinstance(result, list) and
        len(result) == 6 and
        all(isinstance(n, int) and 1 <= n <= 45 for n in result) and
        len(set(result)) == 6
    )
    
    print(f"\n{'✅' if is_valid else '❌'} 유효성 검사: {is_valid}")
    print("=" * 60)
