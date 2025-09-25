"""로또 당첨번호 자동 크롤링 스크립트"""

import requests
import pandas as pd
from bs4 import BeautifulSoup
import json
from datetime import datetime
from pathlib import Path
import time

def fetch_latest_lottery_data():
    """최신 로또 당첨번호 크롤링"""
    
    print("🎰 로또 당첨번호 크롤링 시작...")
    
    # 동행복권 API (공식 데이터)
    base_url = "https://www.dhlottery.co.kr/common.do?method=getLottoNumber"
    
    # 기존 데이터 로드
    data_file = Path('data/new_1190.csv')
    if data_file.exists():
        existing_df = pd.read_csv(data_file, encoding='utf-8')
        last_round = existing_df['round'].max()
        print(f"📊 기존 데이터 최신 회차: {last_round}")
    else:
        existing_df = pd.DataFrame()
        last_round = 1000  # 시작 회차
    
    new_data = []
    current_round = last_round + 1
    consecutive_failures = 0
    
    while consecutive_failures < 3:  # 3회 연속 실패 시 중단
        try:
            # API 요청
            response = requests.get(f"{base_url}&drwNo={current_round}")
            data = response.json()
            
            if data.get('returnValue') == 'success':
                # 데이터 파싱
                lottery_data = {
                    'round': current_round,
                    'date': data['drwNoDate'],
                    'num1': data['drwtNo1'],
                    'num2': data['drwtNo2'], 
                    'num3': data['drwtNo3'],
                    'num4': data['drwtNo4'],
                    'num5': data['drwtNo5'],
                    'num6': data['drwtNo6'],
                    'bonus': data['bnusNo'],
                    'first_prize_amount': data.get('firstPrzwnerCo', 0),
                    'first_prize_winners': data.get('firstAccumamnt', 0)
                }
                
                new_data.append(lottery_data)
                print(f"✅ {current_round}회차 데이터 수집 완료")
                consecutive_failures = 0
                current_round += 1
                
            else:
                consecutive_failures += 1
                print(f"❌ {current_round}회차 데이터 없음 ({consecutive_failures}/3)")
                current_round += 1
                
            time.sleep(0.5)  # 요청 간 딜레이
            
        except Exception as e:
            consecutive_failures += 1
            print(f"❌ {current_round}회차 처리 중 오류: {e}")
            current_round += 1
    
    if new_data:
        # 새 데이터를 기존 데이터와 병합
        new_df = pd.DataFrame(new_data)
        
        if not existing_df.empty:
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            combined_df = new_df
            
        # 중복 제거 및 정렬
        combined_df = combined_df.drop_duplicates(subset=['round'])
        combined_df = combined_df.sort_values('round')
        
        # 파일 저장
        data_dir = Path('data')
        data_dir.mkdir(exist_ok=True)
        
        combined_df.to_csv(data_file, index=False, encoding='utf-8')
        print(f"💾 {len(new_data)}개 신규 회차 데이터 저장 완료")
        
        # 백업 파일도 생성
        backup_file = data_dir / f'backup_{datetime.now().strftime("%Y%m%d")}.csv'
        combined_df.to_csv(backup_file, index=False, encoding='utf-8')
        
        return True
    else:
        print("ℹ️  신규 데이터가 없습니다.")
        return False

if __name__ == '__main__':
    success = fetch_latest_lottery_data()
    exit(0 if success else 1)
