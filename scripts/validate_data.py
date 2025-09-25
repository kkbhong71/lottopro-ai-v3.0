"""데이터 검증 스크립트"""

import pandas as pd
from pathlib import Path
import json

def validate_lottery_data():
    """로또 데이터 유효성 검증"""
    
    print("🔍 데이터 검증 시작...")
    
    data_file = Path('data/new_1190.csv')
    if not data_file.exists():
        print("❌ 데이터 파일이 존재하지 않습니다.")
        return False
    
    try:
        df = pd.read_csv(data_file, encoding='utf-8')
        errors = []
        
        # 필수 컬럼 확인
        required_columns = ['round', 'date', 'num1', 'num2', 'num3', 'num4', 'num5', 'num6', 'bonus']
        for col in required_columns:
            if col not in df.columns:
                errors.append(f"필수 컬럼 누락: {col}")
        
        if errors:
            for error in errors:
                print(f"❌ {error}")
            return False
        
        # 데이터 유효성 검증
        for idx, row in df.iterrows():
            # 번호 범위 확인 (1-45)
            numbers = [row['num1'], row['num2'], row['num3'], row['num4'], row['num5'], row['num6'], row['bonus']]
            for i, num in enumerate(numbers):
                if not (1 <= num <= 45):
                    errors.append(f"회차 {row['round']}: 잘못된 번호 범위 ({num})")
            
            # 중복번호 확인 (보너스 제외)
            main_numbers = [row['num1'], row['num2'], row['num3'], row['num4'], row['num5'], row['num6']]
            if len(set(main_numbers)) != 6:
                errors.append(f"회차 {row['round']}: 중복 번호 존재")
        
        if errors:
            for error in errors[:10]:  # 최대 10개만 출력
                print(f"❌ {error}")
            if len(errors) > 10:
                print(f"... 그외 {len(errors)-10}개 오류")
            return False
        
        print(f"✅ 데이터 검증 완료 - 총 {len(df)}회차")
        return True
        
    except Exception as e:
        print(f"❌ 데이터 검증 중 오류: {e}")
        return False

if __name__ == '__main__':
    success = validate_lottery_data()
    exit(0 if success else 1)
