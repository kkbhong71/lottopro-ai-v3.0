"""성능 예산 체크 스크립트"""

import json
import sys
from pathlib import Path

def check_performance_budget():
    """성능 예산 확인"""
    
    print("📊 성능 예산 체크 시작...")
    
    # 성능 기준값 설정
    performance_budget = {
        'first_contentful_paint': 2.0,  # 2초
        'largest_contentful_paint': 4.0,  # 4초
        'cumulative_layout_shift': 0.1,   # 0.1
        'total_blocking_time': 300,       # 300ms
        'speed_index': 4.0                # 4초
    }
    
    # Lighthouse 결과 파일 확인
    lighthouse_file = Path('.lighthouseci/lhr-*.json')
    
    if not any(Path('.').glob('.lighthouseci/lhr-*.json')):
        print("⚠️  Lighthouse 결과 파일이 없습니다. 성능 체크를 건너뜁니다.")
        return True
    
    try:
        # 최신 Lighthouse 결과 로드
        latest_result = max(Path('.').glob('.lighthouseci/lhr-*.json'), key=lambda p: p.stat().st_mtime)
        
        with open(latest_result, 'r') as f:
            lighthouse_data = json.load(f)
        
        # 성능 지표 추출
        audits = lighthouse_data.get('audits', {})
        
        metrics = {
            'first_contentful_paint': audits.get('first-contentful-paint', {}).get('numericValue', 0) / 1000,
            'largest_contentful_paint': audits.get('largest-contentful-paint', {}).get('numericValue', 0) / 1000,
            'cumulative_layout_shift': audits.get('cumulative-layout-shift', {}).get('numericValue', 0),
            'total_blocking_time': audits.get('total-blocking-time', {}).get('numericValue', 0),
            'speed_index': audits.get('speed-index', {}).get('numericValue', 0) / 1000
        }
        
        # 성능 예산 체크
        violations = []
        
        for metric, value in metrics.items():
            budget = performance_budget.get(metric, 0)
            if value > budget:
                violations.append({
                    'metric': metric,
                    'value': value,
                    'budget': budget,
                    'over_by': value - budget
                })
        
        # 결과 출력
        if violations:
            print("❌ 성능 예산 초과:")
            for v in violations:
                print(f"  - {v['metric']}: {v['value']:.2f} (예산: {v['budget']:.2f}, 초과: {v['over_by']:.2f})")
            return False
        else:
            print("✅ 모든 성능 지표가 예산 내에 있습니다.")
            for metric, value in metrics.items():
                budget = performance_budget.get(metric, 0)
                print(f"  - {metric}: {value:.2f} / {budget:.2f}")
            return True
            
    except Exception as e:
        print(f"❌ 성능 체크 중 오류: {e}")
        return False

if __name__ == '__main__':
    success = check_performance_budget()
    exit(0 if success else 1)
