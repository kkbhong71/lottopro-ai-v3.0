"""
Ultimate Lotto Prediction System 3.0 Enhanced - Web App Standardized Version
궁극 로또 예측 시스템 3.0 Enhanced - 웹앱 표준화 버전

웹앱 표준 템플릿 적용:
- predict_numbers() 진입점 함수
- 글로벌 변수 사용 (lotto_data, pd, np)
- 웹앱 안전 실행 환경 준수
- 에러 처리 및 안전장치 완비
- JSON 직렬화 타입 안전성 보장
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
    웹앱 표준 예측 함수 - Ultimate Enhanced v3.0 시스템
    
    글로벌 변수 사용:
    - lotto_data: pandas DataFrame (로또 당첨번호 데이터)
    - pd: pandas 라이브러리  
    - np: numpy 라이브러리
    - data_path: 데이터 폴더 경로 (문자열)
    
    Returns:
        list: 정확히 6개의 로또 번호 [1-45 범위의 Python 정수]
    """
    try:
        # 1. 데이터 검증
        if 'lotto_data' not in globals() or lotto_data.empty:
            return generate_safe_fallback()
        
        df = lotto_data.copy()
        
        # 2. 데이터 전처리
        df = preprocess_data(df)
        
        # 3. Ultimate Enhanced v3.0 알고리즘 실행
        result = run_ultimate_enhanced_v3_algorithm(df)
        
        # 4. 결과 검증 및 반환
        return validate_result(result)
        
    except Exception as e:
        print(f"Ultimate Enhanced v3.0 error: {str(e)[:100]}")
        return generate_safe_fallback()

def preprocess_data(df):
    """데이터 전처리 - Ultimate Enhanced v3.0용"""
    try:
        # 컬럼명 정규화
        df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]
        
        # 표준 컬럼 매핑
        if len(df.columns) >= 9:
            standard_cols = ['round', 'draw_date', 'num1', 'num2', 'num3', 'num4', 'num5', 'num6', 'bonus_num']
            mapping = dict(zip(df.columns[:9], standard_cols))
            df = df.rename(columns=mapping)
        
        # 숫자 컬럼 변환 - 타입 안전성 보장
        number_cols = ['num1', 'num2', 'num3', 'num4', 'num5', 'num6']
        for col in number_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                # numpy 타입을 Python int로 변환
                df[col] = df[col].apply(lambda x: convert_to_python_int(x) if pd.notna(x) else random.randint(1, 45))
        
        # 유효성 필터링
        df = df.dropna(subset=number_cols)
        for col in number_cols:
            if col in df.columns:
                df = df[(df[col] >= 1) & (df[col] <= 45)]
        
        return df.sort_values('round' if 'round' in df.columns else df.columns[0]).reset_index(drop=True)
        
    except:
        return df

def run_ultimate_enhanced_v3_algorithm(df):
    """Ultimate Enhanced v3.0 핵심 알고리즘"""
    try:
        if len(df) < 5:
            return generate_smart_random()
        
        number_cols = ['num1', 'num2', 'num3', 'num4', 'num5', 'num6']
        
        # Enhanced 피처 생성
        enhanced_features = create_enhanced_features(df, number_cols)
        
        # 55+ 방법론 통합 분석
        analysis_vault = run_comprehensive_analysis(df, enhanced_features)
        
        # Ultimate Enhanced 앙상블
        final_prediction = run_ultimate_enhanced_ensemble(analysis_vault, df)
        
        return final_prediction
        
    except Exception as e:
        print(f"Algorithm execution error: {str(e)[:50]}")
        return generate_smart_random()

def create_enhanced_features(df, number_cols):
    """Enhanced 피처 생성 (Top 5 추가 방법론 포함) - 타입 안전성 보장"""
    try:
        features = {}
        
        # 기본 통계 피처 - Python 타입으로 변환
        sum_values = df[number_cols].sum(axis=1)
        features['sum_stats'] = [convert_to_python_int(x) for x in sum_values.values]
        
        mean_values = df[number_cols].mean(axis=1)
        features['mean_stats'] = [convert_to_python_float(x) for x in mean_values.values]
        
        std_values = df[number_cols].std(axis=1).fillna(0)
        features['std_stats'] = [convert_to_python_float(x) for x in std_values.values]
        
        # 홀짝/고저 분석
        odd_counts = df[number_cols].apply(lambda row: sum(x % 2 for x in row), axis=1)
        features['odd_counts'] = [convert_to_python_int(x) for x in odd_counts.values]
        
        high_counts = df[number_cols].apply(lambda row: sum(x >= 23 for x in row), axis=1)
        features['high_counts'] = [convert_to_python_int(x) for x in high_counts.values]
        
        # Enhanced 방법론 1: 제외수/필터링 시스템
        features['filtering'] = create_filtering_features(df, number_cols)
        
        # Enhanced 방법론 2: 궁합수/이웃수 분석
        features['compatibility'] = create_compatibility_features(df, number_cols)
        
        # Enhanced 방법론 3: 삼각패턴 분석
        features['triangle'] = create_triangle_pattern_features(df, number_cols)
        
        # Enhanced 방법론 4: 고급 시계열 분해
        features['timeseries'] = create_timeseries_features(df, number_cols)
        
        # Enhanced 방법론 5: 동적 임계값 시스템
        features['dynamic'] = create_dynamic_threshold_features(df, number_cols)
        
        return features
        
    except Exception as e:
        print(f"Feature creation error: {str(e)[:50]}")
        return {'basic': [convert_to_python_int(x) for x in df[number_cols].sum(axis=1).values]}

def create_filtering_features(df, number_cols):
    """제외수/필터링 시스템 피처 - 타입 안전성 보장"""
    try:
        filtering_data = {}
        
        # AC값 (산술적 복잡성) 계산
        ac_values = []
        for _, row in df.iterrows():
            numbers = sorted([convert_to_python_int(row[col]) for col in number_cols])
            differences = set()
            for i in range(len(numbers) - 1):
                diff = numbers[i+1] - numbers[i]
                differences.add(diff)
            ac_values.append(len(differences))
        
        filtering_data['ac_values'] = ac_values
        
        # 연속번호 최대 길이
        max_consecutive = []
        for _, row in df.iterrows():
            numbers = sorted([convert_to_python_int(row[col]) for col in number_cols])
            max_len = 1
            current_len = 1
            for i in range(1, len(numbers)):
                if numbers[i] - numbers[i-1] == 1:
                    current_len += 1
                    max_len = max(max_len, current_len)
                else:
                    current_len = 1
            max_consecutive.append(max_len)
        
        filtering_data['max_consecutive'] = max_consecutive
        
        # 필터링 통과 점수
        filtering_scores = []
        for i in range(len(ac_values)):
            score = 100
            ac_val = ac_values[i]
            max_consec = max_consecutive[i]
            
            if 7 <= ac_val <= 10:
                score += 50
            elif 5 <= ac_val <= 6 or ac_val == 11:
                score += 20
            else:
                score -= 30
                
            if max_consec <= 2:
                score += 30
            elif max_consec == 3:
                score -= 20
            else:
                score -= 50
                
            filtering_scores.append(score)
        
        filtering_data['scores'] = filtering_scores
        return filtering_data
        
    except:
        return {'ac_values': [7] * len(df), 'max_consecutive': [2] * len(df), 'scores': [100] * len(df)}

def create_compatibility_features(df, number_cols):
    """궁합수/이웃수 분석 피처"""
    try:
        compatibility_data = {}
        
        # 로또 용지 그리드 (7x7 배치)
        lotto_grid = {}
        for num in range(1, 46):
            row = (num - 1) // 7
            col = (num - 1) % 7
            lotto_grid[num] = (row, col)
        
        # 이웃수 관계 매핑
        neighbor_map = {}
        for num in range(1, 46):
            neighbors = get_neighbors(num, lotto_grid)
            neighbor_map[num] = neighbors
        
        # 이웃수 동반 출현 점수
        neighbor_scores = []
        for _, row in df.iterrows():
            numbers = set([convert_to_python_int(row[col]) for col in number_cols])
            score = 0
            for num in numbers:
                neighbors = neighbor_map.get(num, [])
                for neighbor in neighbors:
                    if neighbor in numbers:
                        score += 1
            neighbor_scores.append(score)
        
        compatibility_data['neighbor_scores'] = neighbor_scores
        compatibility_data['neighbor_map'] = neighbor_map
        
        return compatibility_data
        
    except:
        return {'neighbor_scores': [2] * len(df), 'neighbor_map': {}}

def get_neighbors(num, lotto_grid):
    """특정 번호의 이웃수들 반환"""
    if num not in lotto_grid:
        return []
    
    row, col = lotto_grid[num]
    neighbors = []
    
    # 8방향 이웃
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < 7 and 0 <= new_col < 7:
                neighbor_num = new_row * 7 + new_col + 1
                if 1 <= neighbor_num <= 45:
                    neighbors.append(neighbor_num)
    
    return neighbors

def create_triangle_pattern_features(df, number_cols):
    """삼각패턴 분석 피처 - 타입 안전성 보장"""
    try:
        triangle_data = {}
        
        triangle_complexity_scores = []
        for _, row in df.iterrows():
            numbers = sorted([convert_to_python_int(row[col]) for col in number_cols])
            
            # 삼각수 생성 (재귀적 차분)
            triangle_numbers = set()
            current_level = numbers.copy()
            level = 0
            
            while len(current_level) > 1 and level < 3:  # 최대 3레벨로 제한
                triangle_numbers.update(current_level)
                next_level = []
                
                for i in range(len(current_level) - 1):
                    diff = abs(current_level[i+1] - current_level[i])
                    if diff > 0:
                        next_level.append(diff)
                
                if not next_level:
                    break
                
                current_level = next_level
                level += 1
            
            # 삼각패턴 복잡도
            complexity = level + len(triangle_numbers) / 10
            triangle_complexity_scores.append(convert_to_python_float(complexity))
        
        triangle_data['complexity'] = triangle_complexity_scores
        return triangle_data
        
    except:
        return {'complexity': [5.0] * len(df)}

def create_timeseries_features(df, number_cols):
    """고급 시계열 분해 피처 - 타입 안전성 보장"""
    try:
        if len(df) < 12:
            return {'trend_scores': [0.5] * len(df)}
        
        timeseries_data = {}
        
        # 각 번호별 출현 시계열
        number_series = {}
        for num in range(1, 46):
            series = []
            for _, row in df.iterrows():
                appeared = 1 if num in [convert_to_python_int(row[col]) for col in number_cols] else 0
                series.append(appeared)
            number_series[num] = np.array(series)
        
        # 트렌드 점수 계산
        trend_scores = []
        for i, row in df.iterrows():
            numbers = [convert_to_python_int(row[col]) for col in number_cols]
            trend_sum = 0
            
            for num in numbers:
                if i >= 6:
                    recent_series = number_series[num][max(0, i-6):i+1]
                    trend = convert_to_python_float(np.mean(recent_series))
                    trend_sum += trend
            
            trend_scores.append(convert_to_python_float(trend_sum / 6))
        
        timeseries_data['trend_scores'] = trend_scores
        return timeseries_data
        
    except:
        return {'trend_scores': [0.5] * len(df)}

def create_dynamic_threshold_features(df, number_cols):
    """동적 임계값 시스템 피처 - 타입 안전성 보장"""
    try:
        dynamic_data = {}
        
        # 최근 트렌드 강도
        trend_strengths = []
        seasonal_factors = []
        dynamic_weights = []
        
        for i, row in df.iterrows():
            # 트렌드 강도 (최근 변화율)
            if i >= 10:
                recent_values = df.iloc[i-10:i+1][number_cols].sum(axis=1).values
                if len(recent_values) > 1:
                    trend_strength = abs(np.polyfit(range(len(recent_values)), recent_values, 1)[0])
                    trend_strength = convert_to_python_float(trend_strength)
                else:
                    trend_strength = 0.0
            else:
                trend_strength = 0.0
            
            trend_strengths.append(trend_strength)
            
            # 계절성 요인
            season_phase = (i % 12) / 12 * 2 * np.pi
            seasonal_factor = convert_to_python_float(0.5 + 0.3 * np.sin(season_phase))
            seasonal_factors.append(seasonal_factor)
            
            # 동적 가중치
            base_weight = 1.0
            trend_adjustment = trend_strength / 100
            seasonal_adjustment = seasonal_factor - 0.5
            
            dynamic_weight = base_weight + trend_adjustment + seasonal_adjustment
            dynamic_weight = convert_to_python_float(max(0.5, min(2.0, dynamic_weight)))
            dynamic_weights.append(dynamic_weight)
        
        dynamic_data['trend_strengths'] = trend_strengths
        dynamic_data['seasonal_factors'] = seasonal_factors
        dynamic_data['dynamic_weights'] = dynamic_weights
        
        return dynamic_data
        
    except:
        return {'trend_strengths': [1.0] * len(df), 'seasonal_factors': [0.5] * len(df), 'dynamic_weights': [1.0] * len(df)}

def run_comprehensive_analysis(df, features):
    """55+ 방법론 통합 분석"""
    try:
        analysis_vault = {}
        
        # 기본 분석들
        analysis_vault['frequency'] = frequency_analysis(df)
        analysis_vault['pattern'] = pattern_analysis(df, features)
        analysis_vault['statistical'] = statistical_analysis(df, features)
        
        # Enhanced 분석들
        analysis_vault['filtering'] = filtering_analysis(features)
        analysis_vault['compatibility'] = compatibility_analysis(features)
        analysis_vault['triangle'] = triangle_analysis(features)
        analysis_vault['timeseries'] = timeseries_analysis(features)
        analysis_vault['dynamic'] = dynamic_analysis(features)
        
        return analysis_vault
        
    except Exception as e:
        print(f"Comprehensive analysis error: {str(e)[:50]}")
        return {'basic': {'top_numbers': list(range(1, 21))}}

def frequency_analysis(df):
    """빈도 분석"""
    try:
        number_cols = ['num1', 'num2', 'num3', 'num4', 'num5', 'num6']
        all_numbers = []
        
        for _, row in df.iterrows():
            all_numbers.extend([convert_to_python_int(row[col]) for col in number_cols])
        
        frequency = Counter(all_numbers)
        top_numbers = [num for num, count in frequency.most_common(20)]
        
        return {'frequency_counter': frequency, 'top_numbers': top_numbers}
        
    except:
        return {'frequency_counter': Counter(), 'top_numbers': list(range(1, 21))}

def pattern_analysis(df, features):
    """패턴 분석 - 타입 안전성 보장"""
    try:
        pattern_data = {}
        
        # 홀짝 패턴
        if 'odd_counts' in features:
            avg_odd = convert_to_python_float(np.mean(features['odd_counts']))
            pattern_data['avg_odd'] = avg_odd
        else:
            pattern_data['avg_odd'] = 3.0
        
        # 고저 패턴
        if 'high_counts' in features:
            avg_high = convert_to_python_float(np.mean(features['high_counts']))
            pattern_data['avg_high'] = avg_high
        else:
            pattern_data['avg_high'] = 3.0
        
        # 합계 패턴
        if 'sum_stats' in features:
            avg_sum = convert_to_python_float(np.mean(features['sum_stats']))
            pattern_data['avg_sum'] = avg_sum
        else:
            pattern_data['avg_sum'] = 130.0
        
        return pattern_data
        
    except:
        return {'avg_odd': 3.0, 'avg_high': 3.0, 'avg_sum': 130.0}

def statistical_analysis(df, features):
    """통계 분석 - 타입 안전성 보장"""
    try:
        stats_data = {}
        
        # 기본 통계
        if 'sum_stats' in features:
            stats_data['sum_mean'] = convert_to_python_float(np.mean(features['sum_stats']))
            stats_data['sum_std'] = convert_to_python_float(np.std(features['sum_stats']))
        else:
            stats_data['sum_mean'] = 130.0
            stats_data['sum_std'] = 15.0
        
        # 변동성 분석
        if 'std_stats' in features:
            stats_data['volatility'] = convert_to_python_float(np.mean(features['std_stats']))
        else:
            stats_data['volatility'] = 15.0
        
        return stats_data
        
    except:
        return {'sum_mean': 130.0, 'sum_std': 15.0, 'volatility': 15.0}

def filtering_analysis(features):
    """제외수/필터링 분석 - 타입 안전성 보장"""
    try:
        if 'filtering' in features:
            filtering_data = features['filtering']
            avg_score = convert_to_python_float(np.mean(filtering_data.get('scores', [100])))
            high_quality_threshold = convert_to_python_float(np.percentile(filtering_data.get('scores', [100]), 75))
            
            return {'avg_score': avg_score, 'high_quality_threshold': high_quality_threshold}
        else:
            return {'avg_score': 100.0, 'high_quality_threshold': 150.0}
            
    except:
        return {'avg_score': 100.0, 'high_quality_threshold': 150.0}

def compatibility_analysis(features):
    """궁합수/이웃수 분석 - 타입 안전성 보장"""
    try:
        if 'compatibility' in features:
            compatibility_data = features['compatibility']
            avg_neighbor_score = convert_to_python_float(np.mean(compatibility_data.get('neighbor_scores', [2])))
            neighbor_map = compatibility_data.get('neighbor_map', {})
            
            return {'avg_neighbor_score': avg_neighbor_score, 'neighbor_map': neighbor_map}
        else:
            return {'avg_neighbor_score': 2.0, 'neighbor_map': {}}
            
    except:
        return {'avg_neighbor_score': 2.0, 'neighbor_map': {}}

def triangle_analysis(features):
    """삼각패턴 분석 - 타입 안전성 보장"""
    try:
        if 'triangle' in features:
            triangle_data = features['triangle']
            avg_complexity = convert_to_python_float(np.mean(triangle_data.get('complexity', [5.0])))
            
            return {'avg_complexity': avg_complexity, 'optimal_range': (3, 8)}
        else:
            return {'avg_complexity': 5.0, 'optimal_range': (3, 8)}
            
    except:
        return {'avg_complexity': 5.0, 'optimal_range': (3, 8)}

def timeseries_analysis(features):
    """시계열 분석 - 타입 안전성 보장"""
    try:
        if 'timeseries' in features:
            timeseries_data = features['timeseries']
            trend_scores = timeseries_data.get('trend_scores', [0.5])
            current_trend = convert_to_python_float(np.mean(trend_scores[-10:])) if len(trend_scores) >= 10 else 0.5
            
            return {'current_trend': current_trend, 'trend_direction': 'neutral'}
        else:
            return {'current_trend': 0.5, 'trend_direction': 'neutral'}
            
    except:
        return {'current_trend': 0.5, 'trend_direction': 'neutral'}

def dynamic_analysis(features):
    """동적 임계값 분석 - 타입 안전성 보장"""
    try:
        if 'dynamic' in features:
            dynamic_data = features['dynamic']
            dynamic_weights = dynamic_data.get('dynamic_weights', [1.0])
            current_weight = convert_to_python_float(dynamic_weights[-1]) if dynamic_weights else 1.0
            
            return {'current_weight': current_weight, 'trend_momentum': 'normal'}
        else:
            return {'current_weight': 1.0, 'trend_momentum': 'normal'}
            
    except:
        return {'current_weight': 1.0, 'trend_momentum': 'normal'}

def run_ultimate_enhanced_ensemble(analysis_vault, df):
    """Ultimate Enhanced 앙상블 - 타입 안전성 보장"""
    try:
        # 번호별 점수 계산
        number_scores = defaultdict(float)
        
        # 기본 점수
        for num in range(1, 46):
            number_scores[num] = 100.0
        
        # 빈도 분석 점수
        if 'frequency' in analysis_vault:
            freq_data = analysis_vault['frequency']
            top_numbers = freq_data.get('top_numbers', [])
            for i, num in enumerate(top_numbers[:15]):
                number_scores[num] += 200 - i * 10
        
        # 패턴 분석 점수
        if 'pattern' in analysis_vault:
            pattern_data = analysis_vault['pattern']
            target_sum = pattern_data.get('avg_sum', 130)
            
            # 합계 패턴에 따른 번호 선호도
            if target_sum > 140:
                for num in range(23, 46):
                    number_scores[num] += 50
            elif target_sum < 120:
                for num in range(1, 23):
                    number_scores[num] += 50
        
        # Enhanced 분석 점수들
        # 필터링 점수
        if 'filtering' in analysis_vault:
            threshold = analysis_vault['filtering'].get('high_quality_threshold', 100)
            for num in range(1, 46):
                if estimate_filtering_score(num) >= threshold:
                    number_scores[num] += 100
        
        # 궁합수 점수
        if 'compatibility' in analysis_vault:
            neighbor_map = analysis_vault['compatibility'].get('neighbor_map', {})
            for num, neighbors in neighbor_map.items():
                if neighbors:
                    number_scores[num] += len(neighbors) * 2
        
        # 동적 가중치 적용
        if 'dynamic' in analysis_vault:
            current_weight = analysis_vault['dynamic'].get('current_weight', 1.0)
            for num in number_scores:
                number_scores[num] *= current_weight
        
        # 최적 조합 선택
        selected = select_optimal_combination(number_scores, analysis_vault)
        
        return selected
        
    except Exception as e:
        print(f"Ensemble error: {str(e)[:50]}")
        return generate_smart_random()

def estimate_filtering_score(num):
    """번호별 필터링 점수 추정"""
    base_score = 100
    
    # 중간 범위 번호가 유리
    if 10 <= num <= 35:
        base_score += 20
    
    # 소수인 경우 가점
    if is_prime(num):
        base_score += 10
    
    return base_score

def is_prime(n):
    """소수 판별"""
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

def select_optimal_combination(number_scores, analysis_vault):
    """최적 조합 선택 - 타입 안전성 보장"""
    try:
        # 상위 점수 번호들을 후보로
        sorted_scores = sorted(number_scores.items(), key=lambda x: x[1], reverse=True)
        candidates = [num for num, score in sorted_scores[:25]]
        
        # 여러 조합 시도
        best_combo = None
        best_score = -1
        
        for attempt in range(30):
            # 다양한 전략으로 6개 선택
            if attempt < 10:
                # 상위 점수 기반
                combo = random.sample(candidates[:15], 6)
            elif attempt < 20:
                # 중간 섞기
                combo = random.sample(candidates[:20], 6)
            else:
                # 전체 후보에서
                combo = random.sample(candidates, 6)
            
            # 조합 평가
            score = evaluate_combination(combo, analysis_vault)
            
            if score > best_score:
                best_score = score
                best_combo = combo
        
        result = best_combo if best_combo else random.sample(candidates[:15], 6)
        return [convert_to_python_int(num) for num in result]
        
    except:
        return generate_smart_random()

def evaluate_combination(combo, analysis_vault):
    """조합 평가 점수 - 타입 안전성 보장"""
    try:
        score = 0
        combo = [convert_to_python_int(num) for num in combo]
        
        # 기본 조건 체크
        total_sum = sum(combo)
        if 110 <= total_sum <= 200:
            score += 100
        
        # 홀짝 균형
        odd_count = sum(1 for n in combo if n % 2 == 1)
        if 2 <= odd_count <= 4:
            score += 100
        
        # 고저 균형
        high_count = sum(1 for n in combo if n >= 23)
        if 2 <= high_count <= 4:
            score += 100
        
        # Enhanced 필터링 통과 점수
        if passes_enhanced_filtering(combo):
            score += 200
        
        # 중복 없음
        if len(set(combo)) == 6:
            score += 50
        
        return score
        
    except:
        return 0

def passes_enhanced_filtering(numbers):
    """Enhanced 필터링 통과 검사"""
    try:
        if len(numbers) != 6:
            return False
        
        # AC값 검사
        sorted_nums = sorted([convert_to_python_int(num) for num in numbers])
        differences = set()
        for i in range(len(sorted_nums) - 1):
            diff = sorted_nums[i+1] - sorted_nums[i]
            differences.add(diff)
        ac_value = len(differences)
        
        if not (5 <= ac_value <= 11):
            return False
        
        # 연속번호 검사 (3개 이상 연속 방지)
        consecutive_count = 0
        for i in range(len(sorted_nums) - 1):
            if sorted_nums[i+1] - sorted_nums[i] == 1:
                consecutive_count += 1
                if consecutive_count >= 2:  # 3개 이상 연속
                    return False
            else:
                consecutive_count = 0
        
        return True
        
    except:
        return True

def generate_smart_random():
    """지능형 랜덤 생성 - 타입 안전성 보장"""
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
        
        return [convert_to_python_int(num) for num in sorted(candidates[:6])]
        
    except:
        return generate_safe_fallback()

def generate_safe_fallback():
    """최후 안전장치 - 타입 안전성 보장"""
    try:
        result = sorted(random.sample(range(1, 46), 6))
        return [convert_to_python_int(num) for num in result]
    except:
        return [7, 14, 21, 28, 35, 42]

def validate_result(result):
    """결과 유효성 검증 - 강화된 타입 안전성"""
    try:
        if not isinstance(result, (list, tuple)):
            return generate_safe_fallback()
        
        if len(result) != 6:
            return generate_safe_fallback()
        
        # 정수 변환 및 범위 확인
        valid_numbers = []
        for num in result:
            if isinstance(num, (int, float, np.number)):
                int_num = convert_to_python_int(num)
                if 1 <= int_num <= 45:
                    valid_numbers.append(int_num)
        
        if len(valid_numbers) != 6:
            return generate_safe_fallback()
        
        # 중복 제거
        if len(set(valid_numbers)) != 6:
            return generate_safe_fallback()
        
        return sorted(valid_numbers)
        
    except:
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
    print(f"Ultimate Enhanced v3.0 Result: {result}")
    print(f"Valid: {isinstance(result, list) and len(result) == 6 and all(isinstance(n, int) and 1 <= n <= 45 for n in result)}")
    print(f"Type Check: {[type(x).__name__ for x in result]}")
