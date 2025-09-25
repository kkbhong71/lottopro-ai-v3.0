"""알고리즘 메타데이터 자동 업데이트 스크립트"""

import json
import os
import requests
from pathlib import Path

def update_algorithm_metadata():
    """GitHub 알고리즘 저장소에서 메타데이터 업데이트"""
    
    # GitHub API 설정
    repo = os.getenv('ALGORITHM_REPO', 'username/lottopro-algorithms')
    token = os.getenv('ALGORITHM_REPO_TOKEN')
    
    if not token:
        print("❌ ALGORITHM_REPO_TOKEN이 설정되지 않았습니다.")
        return False
    
    headers = {
        'Authorization': f'token {token}',
        'Accept': 'application/vnd.github.v3+json'
    }
    
    # 알고리즘 정보 수집
    algorithms_info = []
    algorithm_dirs = [
        'super_v1', 'strongest_universe_v1',
        'ultimate_v1', 'ultimate_v2', 'ultimate_v3', 
        'ultimate_v4', 'ultimate_v5', 'ultimate_v6'
    ]
    
    for algo_dir in algorithm_dirs:
        try:
            # info.json 파일 가져오기
            url = f'https://api.github.com/repos/{repo}/contents/{algo_dir}/info.json'
            response = requests.get(url, headers=headers)
            
            if response.status_code == 200:
                import base64
                content = base64.b64decode(response.json()['content']).decode('utf-8')
                info = json.loads(content)
                algorithms_info.append(info)
                print(f"✅ {algo_dir} 메타데이터 업데이트 완료")
            else:
                print(f"⚠️  {algo_dir} 정보를 찾을 수 없습니다.")
                
        except Exception as e:
            print(f"❌ {algo_dir} 처리 중 오류: {e}")
    
    # 메타데이터 파일 저장
    algorithms_dir = Path('algorithms')
    algorithms_dir.mkdir(exist_ok=True)
    
    metadata_file = algorithms_dir / 'algorithm_info.json'
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump({
            'last_updated': datetime.utcnow().isoformat(),
            'algorithms': algorithms_info,
            'total_count': len(algorithms_info)
        }, f, ensure_ascii=False, indent=2)
    
    print(f"📊 총 {len(algorithms_info)}개 알고리즘 메타데이터 업데이트 완료")
    return True

if __name__ == '__main__':
    from datetime import datetime
    success = update_algorithm_metadata()
    exit(0 if success else 1)
