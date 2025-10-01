// github-api.js 수정 - 중복 제거 및 이벤트 바인딩 개선

class GitHubAPIManager {
    constructor() {
        this.baseUrl = '/api';
        this.algorithms = {};
        this.initialized = false;
        
        console.log('GitHubAPIManager 생성됨');
    }

    async initialize() {
        try {
            console.log('알고리즘 정보 로드 시작...');
            await this.loadAlgorithmInfo();
            this.initialized = true;
            console.log('GitHubAPIManager 초기화 완료');
        } catch (error) {
            console.error('초기화 실패:', error);
        }
    }

    async loadAlgorithmInfo() {
        try {
            const url = `${this.baseUrl}/algorithm-info`;
            console.log('API 호출:', url);
            
            const response = await fetch(url);
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const data = await response.json();
            console.log('API 응답 받음:', data);
            
            if (data.status === 'success' && data.info) {
                this.algorithms = data.info;
                console.log(`알고리즘 ${Object.keys(data.info).length}개 로드 완료`);
                console.log('로드된 알고리즘:', Object.keys(data.info));
            }
        } catch (error) {
            console.error('알고리즘 정보 로드 실패:', error);
            throw error;
        }
    }

    // 중복 제거 및 정렬 로직 개선
    renderAlgorithmCards(containerId = 'algorithm-grid', sortBy = 'name', filterAccuracy = 'all') {
        const container = document.getElementById(containerId);
        if (!container) {
            console.error(`컨테이너를 찾을 수 없음: ${containerId}`);
            return;
        }

        console.log('카드 렌더링 시작:', { sortBy, filterAccuracy });

        // 중복 제거를 위해 Set 사용
        const algorithmKeys = [...new Set(Object.keys(this.algorithms))];
        
        let filteredAlgorithms = algorithmKeys.map(key => ({
            key: key,
            ...this.algorithms[key]
        }));

        // 필터 적용
        if (filterAccuracy !== 'all') {
            filteredAlgorithms = filteredAlgorithms.filter(algo => {
                const accuracy = parseFloat(algo.accuracy) || 0;
                if (filterAccuracy === 'high') return accuracy >= 85;
                if (filterAccuracy === 'medium') return accuracy >= 70 && accuracy < 85;
                if (filterAccuracy === 'low') return accuracy < 70;
                return true;
            });
        }

        // 정렬 적용
        filteredAlgorithms.sort((a, b) => {
            if (sortBy === 'accuracy') {
                return (parseFloat(b.accuracy) || 0) - (parseFloat(a.accuracy) || 0);
            } else if (sortBy === 'name') {
                return (a.name || '').localeCompare(b.name || '');
            }
            return 0;
        });

        // 컨테이너 비우기
        container.innerHTML = '';

        // 카드 렌더링
        filteredAlgorithms.forEach(algo => {
            const card = this.createAlgorithmCard(algo);
            container.appendChild(card);
        });

        // 렌더링 후 이벤트 리스너 재바인딩
        this.bindCardEvents();

        console.log(`${filteredAlgorithms.length}개 카드 렌더링 완료`);
    }

    // 알고리즘 카드 생성
    createAlgorithmCard(algo) {
        const card = document.createElement('div');
        card.className = 'algorithm-card';
        card.setAttribute('data-algorithm', algo.key);
        
        const accuracy = parseFloat(algo.accuracy) || 0;
        const accuracyClass = accuracy >= 85 ? 'high' : accuracy >= 70 ? 'medium' : 'low';

        card.innerHTML = `
            <div class="algorithm-header">
                <h3>${algo.name || '알고리즘'}</h3>
                <span class="accuracy-badge ${accuracyClass}">
                    ${accuracy.toFixed(1)}%
                </span>
            </div>
            <p class="algorithm-description">${algo.description || '설명 없음'}</p>
            <div class="algorithm-stats">
                <div class="stat">
                    <span class="stat-label">예측 정확도</span>
                    <span class="stat-value">${accuracy.toFixed(1)}%</span>
                </div>
                <div class="stat">
                    <span class="stat-label">실시간</span>
                    <span class="stat-value">✓</span>
                </div>
            </div>
            <button class="run-algorithm-btn" data-algorithm="${algo.key}">
                <span class="btn-icon">🚀</span>
                <span class="btn-text">AI 예측 실행</span>
            </button>
        `;

        return card;
    }

    // 카드 이벤트 바인딩 함수
    bindCardEvents() {
        const buttons = document.querySelectorAll('.run-algorithm-btn');
        console.log(`${buttons.length}개 버튼에 이벤트 바인딩 시작`);

        buttons.forEach(button => {
            // 기존 이벤트 리스너 제거 (중복 방지)
            const newButton = button.cloneNode(true);
            button.parentNode.replaceChild(newButton, button);

            // 새로운 이벤트 리스너 추가
            newButton.addEventListener('click', async (e) => {
                e.preventDefault();
                e.stopPropagation();
                
                const algorithmKey = newButton.getAttribute('data-algorithm');
                console.log('알고리즘 실행 버튼 클릭:', algorithmKey);
                
                await this.runAlgorithm(algorithmKey);
            });
        });

        console.log('이벤트 바인딩 완료');
    }

    // 알고리즘 실행 함수 (에러 처리 강화)
    async runAlgorithm(algorithmKey) {
        try {
            const button = document.querySelector(`button[data-algorithm="${algorithmKey}"]`);
            if (button) {
                button.disabled = true;
                button.innerHTML = `
                    <span class="btn-icon">⏳</span>
                    <span class="btn-text">실행 중...</span>
                `;
            }

            console.log('알고리즘 실행 시작:', algorithmKey);

            const response = await fetch(`${this.baseUrl}/predict`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    algorithm: algorithmKey,
                    user_numbers: []
                })
            });

            // 에러 응답 처리 개선
            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.message || `HTTP ${response.status} 에러`);
            }

            const result = await response.json();
            console.log('예측 결과:', result);

            // 에러 상태 체크
            if (result.status === 'error') {
                throw new Error(result.message || '알고리즘 실행 실패');
            }

            // 빈 결과 체크
            if (!result.numbers || result.numbers.length === 0) {
                throw new Error('예측 번호가 생성되지 않았습니다. 잠시 후 다시 시도해주세요.');
            }

            // 결과 표시
            this.displayPredictionResult(result);

            // 버튼 복원
            if (button) {
                button.disabled = false;
                button.innerHTML = `
                    <span class="btn-icon">🚀</span>
                    <span class="btn-text">AI 예측 실행</span>
                `;
            }

        } catch (error) {
            console.error('알고리즘 실행 실패:', error);
            
            // 사용자 친화적 에러 메시지
            let errorMessage = error.message;
            if (errorMessage.includes('warnings')) {
                errorMessage = '서버 설정 문제가 발생했습니다. 잠시 후 다시 시도해주세요.';
            }
            
            alert(`알고리즘 실행 실패\n\n${errorMessage}`);
            
            // 버튼 복원
            const button = document.querySelector(`button[data-algorithm="${algorithmKey}"]`);
            if (button) {
                button.disabled = false;
                button.innerHTML = `
                    <span class="btn-icon">🚀</span>
                    <span class="btn-text">AI 예측 실행</span>
                `;
            }
        }
    }

    // 예측 결과 표시 함수 (개선)
    displayPredictionResult(result) {
        let resultContainer = document.getElementById('prediction-result');
        
        if (!resultContainer) {
            resultContainer = document.createElement('div');
            resultContainer.id = 'prediction-result';
            resultContainer.className = 'prediction-result-modal';
            document.body.appendChild(resultContainer);
        }

        const numbers = result.numbers || [];
        const numbersHtml = numbers.map(num => 
            `<span class="lotto-number">${num}</span>`
        ).join('');
        
        // 알고리즘 이름 표시 개선
        const algorithmInfo = this.algorithms[result.algorithm] || {};
        const algorithmName = result.algorithm_name || algorithmInfo.name || result.algorithm;
        const accuracy = result.accuracy_rate || algorithmInfo.accuracy || 'N/A';

        resultContainer.innerHTML = `
            <div class="modal-content">
                <div class="modal-header">
                    <h2>🎯 AI 예측 결과</h2>
                    <button class="close-modal" onclick="this.closest('.prediction-result-modal').remove()">✕</button>
                </div>
                <div class="modal-body">
                    <div class="predicted-numbers">
                        ${numbersHtml}
                    </div>
                    <div class="result-info">
                        <p><strong>알고리즘:</strong> ${algorithmName}</p>
                        <p><strong>예측 정확도:</strong> ${accuracy}%</p>
                        <p><strong>예측 시간:</strong> ${new Date().toLocaleString('ko-KR')}</p>
                        ${result.cached ? '<p class="cache-info">⚡ 캐시된 결과 (1시간 이내)</p>' : ''}
                    </div>
                </div>
                <div class="modal-footer">
                    <button class="save-numbers-btn" onclick="window.saveNumbers && window.saveNumbers(${JSON.stringify(numbers)}, '${result.algorithm}')">
                        💾 번호 저장하기
                    </button>
                    <button class="close-btn" onclick="this.closest('.prediction-result-modal').remove()">
                        닫기
                    </button>
                </div>
            </div>
        `;

        resultContainer.style.display = 'flex';
    }
}

// 전역 인스턴스 생성
let githubAPIManager = null;

// DOM 로드 완료 후 초기화
document.addEventListener('DOMContentLoaded', async () => {
    console.log('DOMContentLoaded - GitHubAPIManager 초기화 시작');
    
    githubAPIManager = new GitHubAPIManager();
    await githubAPIManager.initialize();
    
    // 알고리즘 페이지인 경우 카드 렌더링
    if (document.getElementById('algorithm-grid')) {
        githubAPIManager.renderAlgorithmCards();
    }
    
    // 전역 객체로 등록
    window.githubAPIManager = githubAPIManager;
    console.log('GitHubAPIManager 전역 등록 완료');
    console.log('상태:', {
        initialized: githubAPIManager.initialized,
        algorithmCount: Object.keys(githubAPIManager.algorithms).length
    });
});
